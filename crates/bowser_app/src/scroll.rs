//! Nested scroll containers: discovery, dispatch, and scroll chaining.
//!
//! Phase C of the scroll model. The viewport scroller ([`Tab::scroll()`])
//! used to be the only scroller; now every element box with a scrollable
//! `overflow` axis whose content overflows its scrollport is a scroll
//! container with its own offset, shared with the page's JS web APIs
//! through [`ElementScrollStates`](bowser_js::ElementScrollStates).
//!
//! The pieces:
//!
//! * [`ScrollTree::build`] discovers the containers of one layout and
//!   links them into a tree (the viewport scroller is always index 0).
//! * [`ScrollTree::scroll_chain`] scrolls from a starting container,
//!   chaining the unconsumed remainder per axis to outer scrollers up
//!   to the viewport (scroll chaining). The window worker picks the
//!   start container through the DOM hit test first (so a fixed overlay
//!   intercepts the wheel) with [`ScrollTree::innermost_at`] as the
//!   geometric fallback.
//! * [`ScrollTree::reveal_node`] scrolls a node into view through its
//!   container chain (fragment navigation, `scrollIntoView()`).
//! * `scrollend` bookkeeping: [`ScrollTree::mark_scrolled`] stamps every
//!   moved container with the current event-loop tick and
//!   [`ScrollTree::end_tick`] fires `scrollend` for containers that saw
//!   a full tick with no movement (see the window worker's
//!   `pump_scripts`).
//! * [`ScrollTree::bake_paint_tree`] / [`ScrollTree::sync_paint_offsets`]
//!   keep a paint-side copy of the layout tree whose scrolled subtrees
//!   are shifted by their containers' offsets; the display list is built
//!   from that copy, so nested scrolling moves content without a
//!   relayout. The existing overflow clip in
//!   `bowser_paint::display_list` keeps the shifted content inside the
//!   scrollport. With no nested scrollers the paint tree borrows the
//!   layout tree instead of cloning it.
//! * Nested-scroll fast path: [`ScrollTree::wrap_scroll_transforms`]
//!   wraps each eligible container's scrolling content in a
//!   display-list `PushTransform`/`PopTransform` pair after a full
//!   build, and [`ScrollTree::try_fast_fold`] moves scrolled content by
//!   patching the transform matrix in place — a scroll tick then costs
//!   only the transform patch plus the epilogue re-push (scrollbars +
//!   glow), like the viewport's GPU-uniform fast path. Structural
//!   changes (sticky re-bakes past the exact fold, DOM mutations) fall
//!   back to `sync_paint_offsets` + a full rebuild, which re-wraps. A
//!   sticky box whose scroll-driven shift is exactly foldable
//!   ([`ScrollTree::sticky_shifts_at`]) stays on the fast path instead:
//!   the tick folds its shift into the fast-path patch rather than
//!   re-baking the constraint.
//! * Scroll snap (CSS Scroll Snap 1): after a scroll completes on a
//!   container whose `scroll-snap-type` is not `none`,
//!   [`ScrollTree::eval_snap`] picks the snap target per axis (closest
//!   snap-area edge per `scroll-snap-align`, `mandatory` always snapping,
//!   `proximity` only within the snapport) and launches a follow-up
//!   smooth scroll toward it — a normal [`SmoothScroll`], so the
//!   interruption rule and the `scrollend` quiet-tick machinery apply
//!   unchanged. `scroll-snap-type`/`align` come from the real
//!   `bowser_css` parsing, and `scroll-snap-stop` is read per snap
//!   area off the target's style; the snap areas come from the real
//!   `bowser_layout::snap_areas_for_container` query, axis-mapped and
//!   nested-container-pruned in [`snap_areas_for_engine`]. Every
//!   evaluation records the winning area and resting position per
//!   axis ([`SnapRest`]), and every scroll chain and reveal
//!   re-evaluates each container it moved against that container's
//!   own areas, so nested scrollers settle independently. On
//!   relayout, [`ScrollTree::resnap_after_relayout`] follows the
//!   recorded targets to their new positions (re-snap after layout
//!   changes), as a no-op when nothing relevant moved.
//!
//! Documented v1 limitations:
//!
//! * Negative scrollable overflow: the per-container minimum offset
//!   ([`ScrollContainer::origin`], from layout's `scroll_origin`) is
//!   plumbed through discovery, read by
//!   [`ScrollTree::document_max_scroll`], and published onto the shared
//!   [`ScrollState`] (`set_min_offset`, via
//!   [`ElementScrollStates::publish_with_origins`]). Wheel/keyboard
//!   ([`ScrollTree::scroll_chain`]) and script-driven scrolls clamp
//!   through that shared state's degenerate-safe
//!   [`ScrollState::set_offset`] — a degenerate `origin > max` publish
//!   collapses the range instead of panicking — so negative offsets are
//!   reachable from both paths alike.
//! * `overscroll-behavior` chaining is implemented: `contain`/`none`
//!   stop per-axis chaining past the container (CSS Overscroll Behavior
//!   §3). The overscroll affordance is phase 1: [`ScrollTree::scroll_chain`]
//!   records edge presses ([`EdgeHit`]) and stretches a transient,
//!   paint-time rubber-band shift ([`ScrollTree::overscroll_shift`],
//!   sprung back to zero by [`ScrollTree::advance_overscroll`]) that is
//!   decoupled from the layout offset — no relayout, no `scroll` events.
//!   `none` suppresses the shift while `contain` keeps it, so the two
//!   values stay distinct in [`OverscrollBehavior`]. The shift folds
//!   into the paint tree (`bake_paint_tree`/`sync_paint_offsets`) for
//!   nested containers while the viewport's own shift rides the
//!   per-frame GPU uniform; the window worker advances the spring every
//!   event-loop turn and keeps the loop turning while
//!   [`ScrollTree::has_overscroll_in_flight`] holds.
//!
//! Out-of-flow descendants used to shift with their container in the
//! paint tree even when their containing block lay outside it; since
//! wave 25 the bake/sync walks skip `position: fixed` subtrees and
//! `position: absolute` boxes whose containing block is outside the
//! container, mirroring the `cb_inside` rule layout uses for scrollable
//! overflow.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::time::{Duration, Instant};

use bowser_css::{SnapAlign, SnapStop, SnapType, props};
use bowser_dom::{Document, NodeId};
use bowser_js::{ElementScrollStates, ScrollBehavior, ScrollState};
use bowser_layout::{
    BoxType, LayoutBox, LayoutTree, Position, Rect, Size, establishes_fixed_cb, position_of,
    snap_areas_for_container,
};
use bowser_paint::display_list::{
    Color, DisplayItem, DisplayList, Edge4, edge_glow_band, push_scrollbar,
};

use crate::tab::RenderPipeline;

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Computed `overscroll-behavior` on one axis (CSS Overscroll Behavior
/// §2).
///
/// `Contain` and `None` agree on the chaining decision — neither lets a
/// wheel/keyboard scroll chain past the container on that axis — but stay
/// distinct for the overscroll affordance (glow/bounce,
/// pull-to-refresh), which `none` suppresses and `contain` keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OverscrollBehavior {
    /// Initial value: chaining propagates to the outer scroller.
    #[default]
    Auto,
    /// No chaining beyond this container's scroll boundary on the axis.
    Contain,
    /// No chaining past this container on the axis, ever.
    None,
}

/// Map a computed `overscroll-behavior-x`/`-y` value to its behavior:
/// `CssValue::Keyword("auto" | "contain" | "none")`, initial `auto`.
/// A missing or unparseable value — including the property itself until
/// `bowser_css` parses it — is `auto`.
fn overscroll_value(value: Option<&bowser_css::CssValue>) -> OverscrollBehavior {
    match value {
        Some(bowser_css::CssValue::Keyword(k)) if k.eq_ignore_ascii_case("contain") => {
            OverscrollBehavior::Contain
        }
        Some(bowser_css::CssValue::Keyword(k)) if k.eq_ignore_ascii_case("none") => {
            OverscrollBehavior::None
        }
        _ => OverscrollBehavior::Auto,
    }
}

/// Both axes' computed `overscroll-behavior` from a container's own
/// computed style (`None` for a style-less box: `auto`/`auto`).
fn overscroll_axes(
    style: Option<&bowser_css::ComputedStyle>,
) -> (OverscrollBehavior, OverscrollBehavior) {
    (
        overscroll_value(style.and_then(|s| s.get_by_index(props::OVERSCROLL_BEHAVIOR_X))),
        overscroll_value(style.and_then(|s| s.get_by_index(props::OVERSCROLL_BEHAVIOR_Y))),
    )
}

/// Computed `scrollbar-gutter` (CSS Overflow 4): whether a scroll
/// container reserves space for the scrollbar in its layout even when
/// no scrollbar is shown, so content never shifts when scrollbars
/// appear or disappear.
///
/// WIRED: `bowser_css` registers `scrollbar-gutter` as a computed
/// property, `bowser_layout` narrows scroll containers' child width by
/// the reservation (see `bowser_layout::scrollbar_gutter_reservation`),
/// and [`ScrollTree::gutter_rects`] computes the reserved strips. Paint
/// never paints a strip (gutters are empty space by design).
///
/// The viewport's gutter propagates from root/body (see [`ScrollTree::build`]).
/// `both-edges` under RTL mirrors correctly via the `rtl` flag.
/// `scrollbar-gutter` is a registered, validated computed property in
/// `bowser_css` (see `bowser_css::props::SCROLLBAR_GUTTER`), so through
/// a real stylesheet the computed lookup reads the authored value.
///
/// `Stable` below describes geometry that the pipeline consumes: see
/// the consumers noted on the variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScrollbarGutter {
    /// Initial value: no reserved space. The scrollbar overlays the
    /// scrollport's inline-end edge only while shown.
    #[default]
    Auto,
    /// The intended reservation geometry: [`SCROLLBAR_THICKNESS`] on
    /// the scrollport's inline-end edge (the vertical scrollbar's side;
    /// the left edge under `rtl`) whenever the box is a scroll
    /// container, whether or not a scrollbar is currently shown.
    /// `both_edges` is the `both-edges` keyword: the inline-start edge
    /// reserves the same strip.
    ///
    /// Consumers read this variant: `bowser_layout` narrows the scroll
    /// container's child width by the reservation (see
    /// `bowser_layout::scrollbar_gutter_reservation`), and
    /// [`ScrollTree::gutter_rects`] computes the reserved strips. Paint
    /// leaves the strips empty (gutters are empty space by design).
    Stable { both_edges: bool },
}

/// Map a computed `scrollbar-gutter` value to its gutter mode. The
/// property is registered and validated in `bowser_css`, so the value
/// is `auto`, `stable`, or the `stable both-edges` keyword list (either
/// order); anything else — a missing value included — reserves nothing.
/// Keywords are ASCII-lowercased at parse time, so the match is exact.
fn gutter_mode_from_value(value: Option<&bowser_css::CssValue>) -> ScrollbarGutter {
    use bowser_css::CssValue;
    let is = |v: &CssValue, word: &str| matches!(v, CssValue::Keyword(k) if k == word);
    match value {
        Some(CssValue::Keyword(k)) if k == "stable" => {
            ScrollbarGutter::Stable { both_edges: false }
        }
        Some(CssValue::List(items))
            if items.len() == 2
                && items.iter().any(|i| is(i, "stable"))
                && items.iter().any(|i| is(i, "both-edges")) =>
        {
            ScrollbarGutter::Stable { both_edges: true }
        }
        _ => ScrollbarGutter::Auto,
    }
}

/// A scroller's computed `scrollbar-gutter` from its own computed
/// style (`None` for a style-less box: `auto`). The property is not
/// inherited, so the box's own style is the right source.
fn scrollbar_gutter(style: Option<&bowser_css::ComputedStyle>) -> ScrollbarGutter {
    gutter_mode_from_value(style.and_then(|s| s.get_by_index(props::SCROLLBAR_GUTTER)))
}

/// Per-direction pan permission from a container's computed
/// `touch-action` (CSS Pointer Events 3): a touch drag or fling may
/// move the scroll offset only in the allowed directions. The
/// `touch-action` keywords describe the content's pan direction:
/// `pan-left` lets content pan left, i.e. the offset grows on x
/// (`dx > 0` in the touch pipeline's inverted-delta space).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TouchAction {
    /// `dx < 0` allowed (finger moves right, offset shrinks on x).
    pub x_neg: bool,
    /// `dx > 0` allowed (finger moves left, offset grows on x).
    pub x_pos: bool,
    /// `dy < 0` allowed (finger moves down, offset shrinks on y).
    pub y_neg: bool,
    /// `dy > 0` allowed (finger moves up, offset grows on y).
    pub y_pos: bool,
}

impl TouchAction {
    /// Every direction pans: the `auto` initial.
    pub fn all() -> Self {
        Self {
            x_neg: true,
            x_pos: true,
            y_neg: true,
            y_pos: true,
        }
    }

    /// Zero the disallowed components of a `(dx, dy)` scroll delta.
    pub fn gate(self, dx: f32, dy: f32) -> (f32, f32) {
        let gate = |v: f32, neg: bool, pos: bool| {
            if (v > 0.0 && !pos) || (v < 0.0 && !neg) {
                0.0
            } else {
                v
            }
        };
        (
            gate(dx, self.x_neg, self.x_pos),
            gate(dy, self.y_neg, self.y_pos),
        )
    }

    /// Intersection of two masks (Pointer Events 3 §5.2): a direction
    /// pans only when both masks allow it.
    pub fn intersect(self, other: TouchAction) -> TouchAction {
        TouchAction {
            x_neg: self.x_neg && other.x_neg,
            x_pos: self.x_pos && other.x_pos,
            y_neg: self.y_neg && other.y_neg,
            y_pos: self.y_pos && other.y_pos,
        }
    }
}

/// Map a computed `touch-action` value to its pan permissions. The
/// property is registered and validated in `bowser_css`, so the value
/// is a keyword or a two-keyword list; anything else — a missing
/// value, `auto`, `manipulation`, or garbage — pans every direction
/// (the `auto` initial). `none` pans nothing; `pan-x`/`pan-y` allow an
/// axis both ways; the directional keywords allow only their direction
/// (`pan-left` = content pans left = `dx > 0`); two-keyword values
/// union their permissions (e.g. `pan-x pan-up`).
pub(crate) fn touch_action_from_value(value: Option<&bowser_css::CssValue>) -> TouchAction {
    use bowser_css::CssValue;
    let mut keywords: Vec<&str> = Vec::new();
    match value {
        Some(CssValue::Keyword(k)) => keywords.push(k.as_str()),
        Some(CssValue::List(items)) => {
            for item in items {
                if let CssValue::Keyword(k) = item {
                    keywords.push(k.as_str());
                }
            }
        }
        _ => return TouchAction::all(),
    }
    if keywords.iter().any(|k| k.eq_ignore_ascii_case("none")) {
        return TouchAction::default();
    }
    let mut action = TouchAction::default();
    for k in keywords {
        // Keywords are ASCII-lowercased at parse time; the
        // case-insensitive match is belt-and-braces for hand-built
        // values.
        if k.eq_ignore_ascii_case("pan-x") {
            action.x_neg = true;
            action.x_pos = true;
        } else if k.eq_ignore_ascii_case("pan-left") {
            action.x_pos = true;
        } else if k.eq_ignore_ascii_case("pan-right") {
            action.x_neg = true;
        } else if k.eq_ignore_ascii_case("pan-y") {
            action.y_neg = true;
            action.y_pos = true;
        } else if k.eq_ignore_ascii_case("pan-up") {
            action.y_pos = true;
        } else if k.eq_ignore_ascii_case("pan-down") {
            action.y_neg = true;
        } else {
            // `auto`, `manipulation`, or anything unrecognized: the
            // permissive initial.
            return TouchAction::all();
        }
    }
    action
}

/// Pointer Events 3 §5.2 pre-gate: the `touch-action` of the touched
/// element and its ancestors, up to and including the first scrolling
/// element, are intersected to decide whether the gesture may pan. The
/// intersection runs over the flat tree — shadow boundaries are
/// crossed via the shadow host, so a `touch-action` fence on a host
/// (or above it) still applies to touches inside its shadow tree. The
/// scroller part of that intersection stays live — the touch pipeline
/// re-gates every delta through the anchored container's current mask
/// ([`TouchScroller::touch_action`](crate::scroll_touch::TouchScroller::touch_action)),
/// and the touch chain re-gates each outer scroller as the residual
/// ascends — so this returns only the non-scroller part: the touched
/// element's mask intersected with every non-scrolling ancestor's
/// mask, up to but excluding the anchored container's element
/// (`anchor`; `None` = the viewport scroller, so the chain runs to the
/// root).
///
/// A missing hit (no DOM node under the finger, e.g. the
/// [`ScrollTree::innermost_at`] fallback) returns the permissive
/// initial; a chain node with no element box (or no computed value)
/// contributes the permissive initial too.
///
/// Known §5.2 deviation, documented rather than papered over: a
/// `display: contents` wrapper generates no layout box, and its
/// computed `touch-action` is not recoverable at this layer —
/// computed styles live only on layout boxes, the DOM retains none,
/// and the cascade `Styler` is transient at document load (re-running
/// it per touch start would need the host's retained stylesheets, UA
/// sheet, and viewport, which the scroll layer does not hold). So a
/// `touch-action: none` on such a wrapper does not fence the gesture;
/// the pre-gate stays permissive for it. Fixing this properly means
/// retaining per-node computed styles (or the `Styler`) where the
/// scroll layer can reach them.
///
/// The pre-gate therefore only ever restricts when an author actually
/// fenced the touch target off with a boxed element — e.g. a
/// `touch-action: none` div inside a `pan-y` scroller, which the
/// container-only masks consulted before could not see.
pub(crate) fn touch_action_pregate(
    doc: &Document,
    root: &LayoutBox,
    hit: Option<NodeId>,
    anchor: Option<NodeId>,
) -> TouchAction {
    let Some(hit) = hit else {
        return TouchAction::all();
    };
    // The spec's chain, innermost-first: the touched element and its
    // flat-tree ancestors (shadow boundaries crossed via the host) up
    // to but excluding the anchored container's element.
    let mut chain: Vec<NodeId> = Vec::new();
    let mut cursor = Some(hit);
    while let Some(n) = cursor {
        if Some(n) == anchor {
            break;
        }
        chain.push(n);
        // Pointer Events 3 §5.2 intersects over the flat tree, host
        // included: a shadow-root fragment is unlinked, so the
        // light-DOM parent walk stops there — continue from the
        // fragment's host, or the host's and outer ancestors'
        // `touch-action` would never intersect a touch inside the
        // shadow tree. (Host-into-own-shadow cycles are rejected by
        // `Document`, so the walk still terminates.)
        cursor = doc.parent(n).or_else(|| doc.shadow_host_for(n));
    }
    if chain.is_empty() {
        return TouchAction::all();
    }
    // One layout-tree pass collecting the chain's masks. Element boxes
    // carry their element's own computed `touch-action`; the first box
    // found per node wins (a node split across boxes keeps one mask).
    // The pass stops early once every chain node has its mask.
    let wanted: HashSet<NodeId> = chain.iter().copied().collect();
    let mut masks: HashMap<NodeId, TouchAction> = HashMap::with_capacity(chain.len());
    let mut stack = vec![root];
    while let Some(box_) = stack.pop() {
        if let Some(node) = box_.node
            && wanted.contains(&node)
            && !masks.contains_key(&node)
        {
            masks.insert(
                node,
                touch_action_from_value(
                    box_.style
                        .as_deref()
                        .and_then(|s| s.get_by_index(props::TOUCH_ACTION)),
                ),
            );
            if masks.len() == wanted.len() {
                break;
            }
        }
        stack.extend(box_.children.iter());
    }
    let mut acc = TouchAction::all();
    for n in chain {
        // Box-less chain nodes (`display: contents` wrappers — see the
        // documented §5.2 deviation above) contribute the permissive
        // initial: their computed `touch-action` is not recoverable
        // from the layout tree.
        if let Some(&mask) = masks.get(&n) {
            acc = acc.intersect(mask);
        }
    }
    acc
}

/// Map a computed `scroll-behavior` value to its behavior:
/// `CssValue::Keyword("smooth")` animates; anything else — including
/// the property's `auto` initial — is instant.
fn scroll_behavior_value(value: Option<&bowser_css::CssValue>) -> ScrollBehavior {
    match value {
        Some(bowser_css::CssValue::Keyword(k)) if k.eq_ignore_ascii_case("smooth") => {
            ScrollBehavior::Smooth
        }
        _ => ScrollBehavior::Auto,
    }
}

/// A scroller's CSS `scroll-behavior` default from its own computed
/// style (`None` for a style-less box: `auto`). `scroll-behavior` is
/// not inherited, so the box's own style is the right source.
fn css_scroll_behavior(style: Option<&bowser_css::ComputedStyle>) -> ScrollBehavior {
    scroll_behavior_value(style.and_then(|s| s.get("scroll-behavior")))
}

/// The viewport scroller's CSS `scroll-behavior` default: the root
/// element's computed value, falling back to the body's when the root's
/// is `auto` — mirroring the `overflow`/`overscroll-behavior` viewport
/// propagation ([`viewport_user_axes`], [`ScrollTree::build`]).
fn viewport_scroll_behavior(doc: &Document, root: &LayoutBox) -> ScrollBehavior {
    if css_scroll_behavior(root.style.as_deref()) == ScrollBehavior::Smooth {
        return ScrollBehavior::Smooth;
    }
    css_scroll_behavior(body_style(doc, root))
}

/// Whether the viewport scrolls as RTL: the root element's computed
/// `direction`, falling back to the body's when the root's is not `rtl`.
/// The fallback is sound in both directions: nothing inherits into the
/// root element, so its computed `ltr` means it specified nothing `rtl`;
/// and the body's computed `direction` inherits from the root, so a
/// computed `rtl` on the body while the root computes `ltr` means the
/// body specified `rtl` itself. (The one conflation computed style
/// cannot resolve — root *explicitly* `ltr` vs root unspecified — both
/// read as "the root didn't ask for rtl", so the body's explicit `rtl`
/// wins there too; the specified value is not observable from the
/// layout tree.) This mirrors the `scroll-behavior` and `writing-mode`
/// body fallbacks and keeps the scroll-tree flag in agreement with the
/// `scrollLeft` sign convention the bindings apply to both viewport
/// aliases (see `element_is_rtl`).
fn viewport_is_rtl(doc: &Document, root: &LayoutBox) -> bool {
    if root
        .style
        .as_deref()
        .and_then(|s| s.get_str("direction"))
        .is_some_and(|dir| dir.eq_ignore_ascii_case("rtl"))
    {
        return true;
    }
    body_style(doc, root)
        .and_then(|s| s.get_str("direction"))
        .is_some_and(|dir| dir.eq_ignore_ascii_case("rtl"))
}

/// A scroll container found in the layout tree: element `node` scrolls on
/// at least one axis, with `max` the per-axis maximum offset (scrollable
/// overflow extent minus scrollport, floored at zero).
struct Discovered {
    node: NodeId,
    scrollport: Rect,
    max: (f32, f32),
    /// Per-axis minimum offset: layout's `scroll_origin` — the content's
    /// nearest edge relative to the scrollport origin, negative when
    /// content overflows above/left of it, `(0, 0)` otherwise.
    origin: (f32, f32),
    /// Computed `overscroll-behavior` per axis, read from the
    /// container's own box style in [`recover_parents`].
    overscroll: (OverscrollBehavior, OverscrollBehavior),
    /// The container's computed CSS `scroll-behavior` default, read
    /// from its own box style in [`recover_parents`].
    css_behavior: ScrollBehavior,
    /// Whether the container's computed `direction` is `rtl`, read
    /// from its own box style in [`recover_parents`]. The vertical
    /// scrollbar sits on the scrollport's inline-end edge, which is
    /// the left edge under `rtl` (see [`ScrollTree::scrollbar_geom`]).
    rtl: bool,
    /// Whether the container's block axis is the physical y axis
    /// (`writing-mode: horizontal-tb`, the default): snap-event target
    /// pairs are logical (block, inline), so the physical `snap_rest`
    /// axes map through this flag. Read from its own box style in
    /// [`recover_parents`].
    block_is_y: bool,
    /// The container's computed `scrollbar-gutter` (CSS Overflow 4),
    /// read from its own box style in [`recover_parents`]. The
    /// property is not inherited, so the box's style is the right
    /// source.
    gutter: ScrollbarGutter,
    /// The container's computed `touch-action` pan permissions (CSS
    /// Pointer Events 3), read from its own box style in
    /// [`recover_parents`]. The property is not inherited, so the
    /// box's style is the right source.
    touch_action: TouchAction,
    /// Whether the container's scrollport is viewport-anchored: the box
    /// itself or an ancestor box is `position: fixed`, so layout reports
    /// its scrollport in viewport coordinates rather than page
    /// coordinates (see [`recover_parents`]; finding #1).
    viewport_anchored: bool,
    /// Whether the container's own box skips its scroller ancestors'
    /// paint shifts: `position: absolute` with no positioned box above
    /// it (its containing block is outside the scrolled content), so
    /// the bake/sync walks leave the box at its layout position
    /// ([`child_shift`]'s absolute-outside reset). `position: fixed`
    /// boxes are covered by `viewport_anchored` instead. Scrollbar
    /// painting follows the painted box, so it must skip the ancestor
    /// subtraction for these containers too.
    unshifted: bool,
    /// The container's scroll-snap configuration (`scroll-snap-type`,
    /// `scroll-padding`), read from its own box style in
    /// [`recover_parents`]. The snap areas themselves are collected in
    /// [`ScrollTree::build`], which owns the layout root.
    snap: SnapConfig,
    /// The nearest ancestor box that is itself a discovered container,
    /// by DOM node; `None` parents to the viewport scroller. Recovered
    /// by box-tree ancestry (see [`recover_parents`]), never by depth
    /// comparison: [`bowser_layout::ScrollContainerInfo::depth`] is a
    /// *layout-tree* depth, and sibling subtrees can sit at different
    /// layout depths, so depth comparison mis-parents (F4).
    parent: Option<NodeId>,
    /// Wheel/keyboard-scrollable axes (`auto`/`scroll`; `hidden` axes are
    /// containers but never consume user scrolls).
    user_axes: (bool, bool),
    /// Scroll-container axes (`auto`/`scroll`/`hidden` per axis): the
    /// [`bowser_layout::overflow_axes`] container classification. The
    /// `scrollbar-gutter` reservation applies on an axis whose overflow
    /// is not `visible` (CSS Overflow 4), which `user_axes` alone cannot
    /// distinguish from `hidden`.
    axes: (bool, bool),
    /// Accumulated `transform: translate()` of the container's own box
    /// and its box-tree ancestors, in px, baked by [`recover_parents`].
    /// Layout reports scrollports pre-transform; query-time code
    /// ([`ScrollTree::innermost_at`], [`ScrollTree::scrollbar_frame_rect`])
    /// applies this so wheel/drag/scrollbar input hits the visual
    /// location. The accumulation resets at a `position: fixed`
    /// boundary with no transformed ancestor above — a fresh
    /// viewport frame, mirroring `innermost_at`'s anchored descent
    /// reset. Only translations are baked (see [`box_translate`]).
    transform_offset: (f32, f32),
}

/// Whether a `writing-mode` value lays the block axis out vertically
/// (any `vertical-*` / `sideways-*` mode): the block axis is then the
/// physical x axis. Everything else (including a missing value) is
/// `horizontal-tb`, where block is y.
fn is_vertical_wm(w: &str) -> bool {
    w.eq_ignore_ascii_case("vertical-rl")
        || w.eq_ignore_ascii_case("vertical-lr")
        || w.eq_ignore_ascii_case("sideways-rl")
        || w.eq_ignore_ascii_case("sideways-lr")
}

/// Whether the block axis of a box with computed style `style` is the
/// physical y axis (see [`is_vertical_wm`]).
fn writing_block_is_y(style: Option<&bowser_css::ComputedStyle>) -> bool {
    match style.and_then(|s| s.get_str("writing-mode")) {
        Some(w) => !is_vertical_wm(w),
        None => true,
    }
}

/// Whether `node` scrolls the viewport rather than forming a nested
/// container: the document element or `<body>`. Mirrors
/// `bowser_js`'s viewport-scroller rule so the element container set and
/// the JS `scrollTop` mapping agree: `overflow` on `html`/`body`
/// propagates to the viewport instead of creating a nested scroller.
fn is_viewport_scroller(doc: &Document, node: NodeId) -> bool {
    if doc.document_element() == Some(node) {
        return true;
    }
    matches!(
        doc.kind(node),
        bowser_dom::NodeKind::Element(d) if d.tag_name == "body"
    )
}

/// The body's computed style, for viewport `overflow` propagation (see
/// [`viewport_user_axes`]): the style of the layout box built for the
/// `<body>` element, if the document has one and it produced a box.
///
/// Iterative: the tree can be arbitrarily deep (finding #4).
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

/// Wheel/keyboard-scrollable axes of the viewport scroller, from
/// propagated `overflow` (CSS Overflow 3 §2.4): the root element's used
/// values win, unless both are `visible` — then the body's propagate
/// instead. `visible` computes to `auto` on the viewport itself, so a
/// both-`visible` source still scrolls on both axes; only `hidden`
/// disables a user axis. The viewport stays a scroll container either
/// way: programmatic scrolls (`scrollTo`, fragment navigation,
/// `reveal_node`) bypass the mask, exactly like nested `hidden`
/// containers.
///
/// Before this, the viewport was unconditionally `(true, true)`, so
/// `body { overflow: hidden }` — the standard scroll-lock — still
/// scrolled on wheel and keys while a nested `hidden` container
/// correctly did not.
fn viewport_user_axes(doc: &Document, root: &LayoutBox) -> (bool, bool) {
    viewport_propagated_axes(doc, root, |(_, user)| user)
}

/// Scroll-container axes (`auto`/`scroll`/`hidden` per axis) of the
/// viewport scroller, from propagated `overflow` (CSS Overflow 3 §2.4):
/// same propagation as [`viewport_user_axes`] but selects the used
/// scroll-container axes (`.0`) instead of the wheel/keyboard axes.
/// The root element's values win unless both are `visible`, then the
/// body's propagate; both `visible` behaves as `auto` on both axes.
fn viewport_container_axes(doc: &Document, root: &LayoutBox) -> (bool, bool) {
    viewport_propagated_axes(doc, root, |(container, _)| container)
}

/// Viewport-scroller axes from propagated `overflow`: the root
/// element's used values win unless both are `visible`, then the body's
/// propagate; a both-`visible` source behaves as `auto` on both axes.
/// `select` picks which [`bowser_layout::overflow_axes`] pair to
/// return — the used scroll-container axes (`.0`) or the
/// wheel/keyboard-scrollable axes (`.1`).
fn viewport_propagated_axes(
    doc: &Document,
    root: &LayoutBox,
    select: impl Fn(((bool, bool), (bool, bool))) -> (bool, bool),
) -> (bool, bool) {
    /// True when the style's used overflow values are both `visible`
    /// (after the visible→auto pairing): it propagates nothing.
    fn both_visible(style: &bowser_css::ComputedStyle) -> bool {
        let ((cx, cy), _) = bowser_layout::overflow_axes(style);
        !cx && !cy
    }
    let propagated = match root.style.as_deref() {
        // The root element's values win unless both are `visible`.
        Some(s) if !both_visible(s) => Some(s),
        // Root is `visible`/`visible` (or styless): the body's values
        // propagate; without a usable body everything stays scrollable.
        _ => body_style(doc, root).filter(|s| !both_visible(s)),
    };
    match propagated {
        Some(s) => select(bowser_layout::overflow_axes(s)),
        // Both `visible`: behaves as `auto` on the viewport.
        None => (true, true),
    }
}

/// Find the scroll containers of `root` in tree order (parents before
/// children, siblings in layout order).
///
/// Container discovery (candidate walk, deepest-first extent confirmation,
/// scrollable-overflow maxima) now lives in layout itself
/// ([`bowser_layout::scroll_containers_with_offset`]); this only applies
/// the app's viewport-scroller rule on top: `overflow` on the document
/// element or `<body>` propagates to the viewport scroller instead of
/// forming a nested container. Each entry's scroller ancestor is then
/// recovered by box-tree ancestry ([`recover_parents`]).
///
/// `viewport` feeds `scroll-padding` resolution (the `vw`/`vh` bases).
///
/// Viewport-anchored (`position: fixed`) containers keep their
/// viewport-coordinate scrollports: the offset is passed as zero, so
/// layout does not translate them, and they are marked
/// [`Discovered::viewport_anchored`] instead. Translating once at build
/// went stale on every later document scroll — `innermost_at` mis-hit-
/// tested by (live − build-time) and reveals into fixed containers
/// mis-targeted (finding #1). Query-time code translates by the live
/// viewport offset instead (see [`ScrollTree::innermost_at`],
/// [`ScrollTree::reveal_node_with_options`]).
fn discover(doc: &Document, root: &LayoutBox, viewport: bowser_layout::Size) -> Vec<Discovered> {
    // `rem` in `scroll-padding` resolves against the root element's
    // computed font size; a style-less root falls back to the CSS
    // initial value (`medium` = 16px).
    let root_font_size = root
        .style
        .as_ref()
        .map(|s| px_font_size(s, 16.0))
        .unwrap_or(16.0);
    let mut out: Vec<Discovered> = bowser_layout::scroll_containers_with_offset(root, (0.0, 0.0))
        .into_iter()
        .filter_map(|c| {
            let node = c.node?;
            if is_viewport_scroller(doc, node) {
                return None;
            }
            Some(Discovered {
                node,
                scrollport: c.scrollport,
                max: c.max,
                origin: c.scroll_origin,
                // `recover_parents` visits this container's own box and
                // reads the real computed values there.
                overscroll: (OverscrollBehavior::Auto, OverscrollBehavior::Auto),
                // The container's own computed value is read by
                // `recover_parents`; until then touch pans freely (the
                // `auto` initial).
                touch_action: TouchAction::all(),
                css_behavior: ScrollBehavior::Auto,
                rtl: false,
                block_is_y: true,
                viewport_anchored: false,
                unshifted: false,
                snap: SnapConfig::none(),
                parent: None,
                user_axes: c.user_axes,
                // The container-axis classification gates the
                // `scrollbar-gutter` reservation per axis (see
                // `Discovered::axes`); the gutter mode itself is read
                // from the box's own style by `recover_parents` below.
                axes: c.axes,
                gutter: ScrollbarGutter::Auto,
                // Baked by `recover_parents` below.
                transform_offset: (0.0, 0.0),
            })
        })
        .collect();
    recover_parents(root, &mut out, root_font_size, viewport);
    out
}

/// One frame of [`recover_parents`]' preorder walk: the box, whether
/// this is an exit marker, and the inherited flags (fixed box above,
/// positioned box above, accumulated translate, fixed-CB trigger above).
type WalkFrame<'a> = (&'a LayoutBox, bool, bool, bool, (f32, f32), bool);

/// Recover each discovered container's scroller ancestor by box-tree
/// ancestry: one preorder walk of `root` tracking the nearest ancestor
/// box that is itself a discovered container.
///
/// [`bowser_layout::scroll_containers_with_offset`] emits containers in
/// box preorder and [`discover`] only filters that order, so this walk meets container
/// boxes in exactly the entries' order — the k-th container box visited
/// is `discovered[k]`, and its scroller ancestor is whatever container
/// is innermost on the ancestor stack at that moment (`None` for the
/// viewport). No depth arithmetic: a container in a sibling subtree can
/// have greater layout depth than an earlier container without being
/// inside it, which is what the old depth-stack got wrong.
///
/// Box-tree ancestry, not DOM ancestry: out-of-flow boxes are hoisted in
/// the box tree, so e.g. a `position: fixed` container's nearest DOM
/// scroller ancestor is not its real one — the viewport is.
///
/// The same walk marks viewport-anchored containers and resolves their
/// scroll-snap configuration: it tracks whether a `position: fixed`
/// box sits above the current box, mirroring layout's own
/// `viewport_anchored` computation (`anchored || fixed`, inherited down
/// — see `bowser_layout::scroll_containers_with_offset`). `root_font_size`
/// and `viewport` are the `scroll-padding` resolution bases.
fn recover_parents(
    root: &LayoutBox,
    discovered: &mut [Discovered],
    root_font_size: f32,
    viewport: bowser_layout::Size,
) {
    let is_container: HashSet<NodeId> = discovered.iter().map(|d| d.node).collect();
    // Nearest ancestor container nodes, innermost last.
    let mut ancestors: Vec<NodeId> = Vec::new();
    // Next `discovered` entry: entries are in box preorder, matching
    // this walk's visit order.
    let mut next = 0usize;
    // Iterative preorder with explicit exit markers (no recursion).
    // The third tuple element tracks a `position: fixed` box above the
    // current one, for `viewport_anchored`; the fourth tracks any
    // non-`static` positioned box above it, for `unshifted` (mirroring
    // `child_shift`'s `cb_above`: an absolutely positioned box with no
    // positioned ancestor is laid out against the initial containing
    // block, outside any scrolled content). The fifth is the accumulated
    // `transform: translate()` above the current box, in px, in the
    // current frame (see `Discovered::transform_offset`); the sixth
    // tracks a fixed containing-block trigger above it in that frame
    // (any of [`establishes_fixed_cb`]'s triggers), so a
    // `position: fixed` box with no such trigger above starts a
    // fresh viewport frame instead of inheriting page-frame translates.
    let tctx = TranslateCtx {
        root_font_size,
        viewport_w: viewport.w,
        viewport_h: viewport.h,
    };
    let mut stack: Vec<WalkFrame> = vec![(root, false, false, false, (0.0, 0.0), false)];
    while let Some((box_, exit, fixed_above, positioned_above, tacc_above, fixed_cb_above)) =
        stack.pop()
    {
        if exit {
            if box_.node.is_some_and(|n| is_container.contains(&n)) {
                ancestors.pop();
            }
            continue;
        }
        // Layout's rule, mirrored: a fixed box anchors its whole subtree
        // to the viewport.
        let is_fixed = box_
            .style
            .as_ref()
            .is_some_and(|s| position_of(s) == Position::Fixed);
        let fixed_here = fixed_above || is_fixed;
        let positioned_here = positioned_above
            || box_
                .style
                .as_ref()
                .is_some_and(|s| position_of(s) != Position::Static);
        let own_translate = box_translate(box_.style.as_deref(), box_.rect.w, box_.rect.h, tctx);
        // A fixed box with no fixed-CB trigger above lives in the
        // viewport frame: page-frame translates above it do not move
        // it, so the accumulation restarts with its own translate. A
        // fixed box under such a trigger rides that ancestor's frame
        // (CSS Transforms 1 §3 and its filter/perspective/will-change/
        // contain siblings) and keeps accumulating.
        let fresh_frame = is_fixed && !fixed_cb_above;
        let tacc_here = if fresh_frame {
            own_translate
        } else {
            (
                tacc_above.0 + own_translate.0,
                tacc_above.1 + own_translate.1,
            )
        };
        let fixed_cb_here = if fresh_frame {
            box_.style.as_deref().is_some_and(establishes_fixed_cb)
        } else {
            fixed_cb_above || box_.style.as_deref().is_some_and(establishes_fixed_cb)
        };
        if let Some(n) = box_.node
            && is_container.contains(&n)
        {
            if let Some(d) = discovered.get_mut(next) {
                debug_assert_eq!(
                    d.node, n,
                    "scroll_containers_with_offset is preorder: entry matches visited box"
                );
                d.parent = ancestors.last().copied();
                d.viewport_anchored = fixed_here;
                d.transform_offset = tacc_here;
                // An absolutely positioned container with no positioned
                // ancestor above it is laid out against the initial
                // containing block: its box skips the ancestor paint
                // shifts (`child_shift`), so scrollbar painting must
                // not subtract them either. A fixed box is already
                // `viewport_anchored`.
                d.unshifted = !fixed_here
                    && box_
                        .style
                        .as_ref()
                        .is_some_and(|s| position_of(s) == Position::Absolute)
                    && !positioned_above;
                // The container's own computed style: `overscroll-behavior`,
                // `scroll-behavior`, `scroll-snap-type`, `scroll-snap-stop`
                // and `scroll-padding` are not inherited, so the box's
                // style is the right source.
                let style = box_.style.as_deref();
                d.overscroll = overscroll_axes(style);
                d.css_behavior = css_scroll_behavior(style);
                // The vertical scrollbar sits on the scrollport's
                // inline-end edge: the left edge under `direction: rtl`.
                // `direction` is inherited, so the container's own
                // computed value is the used one; missing (or anything
                // but `rtl`) is the `ltr` initial value.
                d.rtl = style
                    .and_then(|s| s.get_str("direction"))
                    .is_some_and(|dir| dir.eq_ignore_ascii_case("rtl"));
                // Block axis mapping for snap events: vertical /
                // sideways writing modes put the block axis on x.
                d.block_is_y = writing_block_is_y(style);
                // `scrollbar-gutter` is not inherited: the container's
                // own computed value is the used one.
                d.gutter = scrollbar_gutter(style);
                // `touch-action` is not inherited either: the container's
                // own computed value gates its touch drags and flings.
                d.touch_action = touch_action_from_value(
                    style.and_then(|s| s.get_by_index(props::TOUCH_ACTION)),
                );
                d.snap = SnapConfig {
                    snap_type: style
                        .map(|s| s.scroll_snap_type())
                        .unwrap_or(SnapType::None),
                    padding: scroll_padding(
                        style,
                        root_font_size,
                        viewport.w,
                        viewport.h,
                        d.scrollport.w,
                        d.scrollport.h,
                    ),
                };
            }
            next += 1;
            ancestors.push(n);
        }
        stack.push((
            box_,
            true,
            fixed_above,
            positioned_above,
            tacc_above,
            fixed_cb_above,
        ));
        for child in box_.children.iter().rev() {
            stack.push((
                child,
                false,
                fixed_here,
                positioned_here,
                tacc_here,
                fixed_cb_here,
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// Scroll tree
// ---------------------------------------------------------------------------

/// One out-of-flow descendant that may need its own scroll-transform
/// wrap on the nested-scroll fast path: a `position: absolute` box with
/// non-`auto` z-index, or a `position: fixed` box, whose baked shift
/// includes a scroller's offset. Recorded per shifting scroller by
/// [`ScrollTree::bake_paint_tree`]; [`ScrollTree::wrap_scroll_transforms`]
/// resolves the boxes against the paint tree the display list was built
/// from (rects recorded at bake time go stale when
/// [`ScrollTree::sync_paint_offsets`] folds offsets into the paint
/// tree), arms the out-of-range ones with that scroller's translation,
/// and [`ScrollTree::try_fast_fold`] patches them alongside the
/// scroller's own wrap.
#[derive(Debug)]
pub(crate) struct OofWrap {
    /// DOM node of the out-of-flow box (for tests/debugging, and to
    /// resolve its current baked rect at wrap time).
    node: Option<NodeId>,
    /// DOM node of the containing block, for absolutely positioned
    /// boxes (`None` for `position: fixed`, which has none — its drain
    /// never leaves a wrapper pair).
    cb_node: Option<NodeId>,
}

/// One scroll container: the viewport (`node == None`) or an element box
/// with a scrollable overflow axis.
#[derive(Debug)]
pub struct ScrollContainer {
    /// Element node; `None` for the viewport scroller.
    pub node: Option<NodeId>,
    /// Scrollport (border box) in page coords: the clip region, fixed
    /// while the content scrolls under it.
    pub scrollport: Rect,
    /// Per-axis minimum scroll offset: the content's nearest edge
    /// relative to the scrollport origin, negative when content
    /// overflows above/left of it (`(0, 0)` otherwise). Wheel and reveal
    /// clamps use `[origin, max]`; the shared [`ScrollState`] carries
    /// the same minimum (published by
    /// [`ElementScrollStates::publish_with_origins`]), so script-driven
    /// `scrollTop` writes clamp into `[origin, max]` too.
    pub origin: (f32, f32),
    /// Computed `overscroll-behavior` per axis: `contain`/`none` stop
    /// wheel/keyboard chaining past this container on that axis.
    pub overscroll: (OverscrollBehavior, OverscrollBehavior),
    /// Computed `touch-action` pan permissions: the touch pipeline
    /// gates drag/fling deltas through this before chaining, so a
    /// `touch-action: pan-x` scroller never pans vertically from touch.
    pub touch_action: TouchAction,
    /// Shared offset state. The viewport's aliases [`Tab::scroll()`];
    /// element containers alias the JS-visible per-element states, so
    /// wheel/keyboard and `Element.scrollTop` mutate the same cells.
    pub state: Rc<ScrollState>,
    /// `scroll`-event report baseline / dedup marker only: both
    /// [`ScrollTree::try_fast_fold`] and [`ScrollTree::sync_paint_offsets`]
    /// advance it to the live offset, so a later fold sees no drift.
    /// It is deliberately NOT the shift base for the paint tree —
    /// [`ScrollContainer::paint_base`] is — so the resnap reset in
    /// [`rebuild_after_layout`] can re-arm reporting without
    /// mis-shifting rects.
    painted: Cell<(f32, f32)>,
    /// Transient overscroll (rubber-band) shift baked into the tab's
    /// paint tree; [`ScrollTree::sync_paint_offsets`] diffs against
    /// [`ScrollContainer::overscroll_delta`] to fold the spring's
    /// per-turn movement into the paint tree without a relayout. The
    /// viewport's own shift is never baked — it rides the per-frame GPU
    /// uniform — so index 0 keeps `(0.0, 0.0)` here.
    ///
    /// Like [`ScrollContainer::painted`], the fast path advances this
    /// without touching rects when the container is wrapped (see
    /// [`ScrollContainer::wrap_base_os`]).
    painted_os: Cell<(f32, f32)>,
    /// The offset the paint-tree rects actually hold — the shift base
    /// for [`ScrollTree::sync_paint_offsets`]'s unwrapped fallback.
    /// Seeded at bake with the baked offset (like
    /// [`ScrollContainer::painted`]), but unlike `painted` it is never
    /// reset for reporting: after a resnap reset the rects still hold
    /// `-post`, so the next fold diffs against `post` and reports the
    /// move without mis-shifting rects by (pre − post).
    paint_base: Cell<(f32, f32)>,
    /// Overscroll twin of [`ScrollContainer::paint_base`]: the
    /// transient overscroll (rubber-band) shift the paint-tree rects
    /// hold, folded by [`ScrollTree::sync_paint_offsets`] alongside the
    /// scroll shift.
    paint_base_os: Cell<(f32, f32)>,
    /// Whether this container's scrolled content may ride a display-list
    /// `PushTransform` instead of a paint-tree re-bake: computed by
    /// [`ScrollTree::bake_paint_tree`], which starts every nested
    /// container eligible and disqualifies a container only when a
    /// `position: absolute`/`fixed` strict descendant both shifts with
    /// its scroll and cannot be covered by a transform — neither by the
    /// container's own range wrap (in-range drains) nor by one of the
    /// extra [`ScrollContainer::oof_wraps`] pairs (out-of-range
    /// drains). Conservative: disqualified containers keep the
    /// [`ScrollTree::sync_paint_offsets`] + full-rebuild path.
    pub(crate) transform_fast: Cell<bool>,
    /// Extra scroll-transform wraps for out-of-flow descendants that
    /// shift with this container's scroll but may drain outside its
    /// overflow-clip range: one [`OofWrap`] per qualifying
    /// `position: absolute` (non-`auto` z-index) / `position: fixed`
    /// strict descendant, recorded by [`ScrollTree::bake_paint_tree`]
    /// (cleared on every bake). [`ScrollTree::wrap_scroll_transforms`]
    /// splices a `PushTransform`/`PopTransform` pair around each
    /// out-of-range box's items, armed with this container's scroll
    /// translation — the box is positioned relative to the scroller, so
    /// it takes the same translation as the scrolled content — and
    /// [`ScrollTree::try_fast_fold`] patches those pairs alongside the
    /// container's own wrap. Viewport-positioned boxes
    /// (`position: fixed` with no fixed-CB trigger inside) shift with
    /// no scroller and are never recorded here.
    pub(crate) oof_wraps: RefCell<Vec<OofWrap>>,
    /// Display-list indices of this container's armed extra
    /// [`OofWrap`] `PushTransform`s, set by
    /// [`ScrollTree::wrap_scroll_transforms`] (cleared on every wrap);
    /// [`ScrollTree::try_fast_fold`] patches them with this container's
    /// scroll translation alongside the container's own wrap.
    pub(crate) oof_wrap_items: RefCell<Vec<usize>>,
    /// Dedicated [`StickyShift`] transform pairs, armed by
    /// [`ScrollTree::wrap_scroll_transforms`] (cleared on every wrap):
    /// `(sticky box node, display-list index of its `PushTransform`)`.
    /// [`ScrollTree::try_fast_fold`] patches each pair with the box's
    /// exact sticky shift from [`ScrollTree::sticky_shifts_at`], on top
    /// of the container's scroll transform (the pair nests inside the
    /// container's own wrap). A sticky box whose items cannot be
    /// isolated to a clean display-list range gets no pair; the tick
    /// then falls back to re-bake rather than mis-shifting content.
    pub(crate) sticky_wrap_items: RefCell<Vec<(NodeId, usize)>>,
    /// Display-list index of this container's scroll `PushTransform`
    /// item, recorded by [`ScrollTree::wrap_scroll_transforms`]; `None`
    /// when the container is not wrapped (ineligible, ambiguous clip,
    /// or the wrap was invalidated by a
    /// [`ScrollTree::sync_paint_offsets`] fallback).
    pub(crate) wrap_item: Cell<Option<usize>>,
    /// Scroll offset the paint-tree rects — and hence the wrapped
    /// display-list items — are baked at. [`ScrollTree::bake_paint_tree`]
    /// and [`ScrollTree::sync_paint_offsets`] keep it in sync with the
    /// rects; the transform fast path leaves the rects alone.
    wrap_base: Cell<(f32, f32)>,
    /// Transient overscroll shift baked into the paint-tree rects, kept
    /// alongside [`ScrollContainer::wrap_base`].
    wrap_base_os: Cell<(f32, f32)>,
    /// The container's border-box rect as baked into the paint tree
    /// (layout rect plus ancestor scroll shifts): the rect the
    /// display-list builder copies into the container's overflow
    /// `PushClip`/`PushRoundedClip`, used to locate the content range
    /// [`ScrollTree::wrap_scroll_transforms`] wraps. Recorded by
    /// [`ScrollTree::bake_paint_tree`] on first entry; `None` before
    /// the first bake.
    clip_rect: Cell<Option<Rect>>,
    /// Wheel/keyboard-scrollable axes: false on `overflow: hidden` axes.
    /// Hidden containers are still scroll containers (JS `scrollTop`
    /// writes the shared state directly and works), but a user scroll
    /// delta chains past them untouched.
    pub user_scroll: (bool, bool),
    /// Scroll-container axes (`auto`/`scroll`/`hidden` per axis): the
    /// [`bowser_layout::overflow_axes`] container classification, used
    /// to gate the `scrollbar-gutter` reservation per axis (CSS Overflow
    /// 4 reserves the gutter whenever overflow is not `visible`, which
    /// includes `hidden` axes that never consume user scrolls).
    axes: (bool, bool),
    /// Computed `scrollbar-gutter` (CSS Overflow 4): when `Stable`, the
    /// scroll tree reserves the scrollbar width in the container's
    /// layout (see [`ScrollTree::gutter_rects`]) even when no scrollbar
    /// is shown, so content never shifts when scrollbars appear or
    /// disappear.
    pub gutter: ScrollbarGutter,
    /// Whether `scrollport` is viewport-anchored: the container's box or
    /// an ancestor box is `position: fixed`, so the scrollport is stored
    /// in viewport coordinates, untranslated (finding #1). Query-time
    /// code (`innermost_at`, `reveal_node_with_options`) translates by
    /// the live viewport offset instead of trusting a build-time
    /// translation that goes stale on the next document scroll.
    pub viewport_anchored: bool,
    /// Whether the container's own box skips its scroller ancestors'
    /// paint shifts (`position: absolute` against the initial
    /// containing block; see [`Discovered::unshifted`]). Scrollbar
    /// painting follows the painted box and skips the ancestor
    /// subtraction for these containers.
    unshifted: bool,
    /// Accumulated `transform: translate()` of the container's own box
    /// and its box-tree ancestors, in px, baked at discovery (see
    /// [`Discovered::transform_offset`]). Layout reports scrollports
    /// pre-transform; [`ScrollTree::innermost_at`] and
    /// [`ScrollTree::scrollbar_frame_rect`] apply this so wheel, drag,
    /// and scrollbar input hits the visual location rather than the
    /// pre-transform one.
    transform_offset: (f32, f32),
    /// Scroll-snap configuration (`scroll-snap-type`, `scroll-padding`),
    /// resolved at build from the container's own style. The type is
    /// `None` (and snapping inert) when the container opts out.
    snap: SnapConfig,
    /// Snap areas of the container's in-flow descendants (scroll-margin
    /// expanded, in container content coordinates), collected at build
    /// by [`snap_areas_for_engine`], each with its target's
    /// `scroll-snap-stop`. Empty when nothing in the container opts
    /// into `scroll-snap-align`.
    snap_areas: Vec<SnapAreaInfo>,
    /// The container's `scroll-initial-target` position (CSS Scroll
    /// Snap 2 §3.1.2): the aligned offset of the descendant marked
    /// `scroll-initial-target: nearest` nearest the container's
    /// initial offset, in the container's clamped offset space,
    /// resolved at build by [`initial_target_for`]. `None` when the
    /// container has no initial target. Seated by
    /// [`ScrollTree::carry_scroll_state`]: once per distinct value
    /// per shared state (a changed value re-seats under the guard
    /// documented there).
    initial_target: Option<(f32, f32)>,
    /// The container's offset at the last snap evaluation: the pre-scroll
    /// offset for the next `scroll-snap-stop: always` interval. Updated
    /// by [`ScrollTree::eval_snap`]; carried across relayouts by
    /// [`ScrollTree::carry_scroll_state`].
    snap_base: Cell<(f32, f32)>,
    /// The container's snapped state per axis (`x`, `y`): the snap
    /// area and position the last [`ScrollTree::eval_snap`] selected,
    /// or `None` on an axis where it selected nothing (proximity
    /// miss, no candidates, axis uncovered by the snap type,
    /// `scroll-snap-type: none`). The container counts as snapped on
    /// an axis when the rest position equals the live offset within
    /// [`SNAP_EPS`]. Consumed by
    /// [`ScrollTree::resnap_after_relayout`]; carried across
    /// relayouts by [`ScrollTree::carry_scroll_state`].
    snap_rest: Cell<(Option<SnapRest>, Option<SnapRest>)>,
    /// Whether the container's block axis is the physical y axis
    /// (`writing-mode: horizontal-tb`): snap-event target pairs are
    /// logical (block, inline) while [`ScrollContainer::snap_rest`] is
    /// physical (x, y), so the event pair maps through this flag.
    /// Resolved at build like [`ScrollContainer::rtl`] (the viewport's
    /// from the root element's style, with body fallback).
    pub block_is_y: bool,
    /// The target pair the last `scrollsnapchanging` reported for this
    /// container (CSS Scroll Snap 2 §5.1.3): `None` until the first
    /// report, which is distinct from `Some` of an all-null pair (a
    /// reported unsnap). The event layer queues a fresh
    /// `scrollsnapchanging` whenever the current pair differs from
    /// this. Carried across relayouts by
    /// [`ScrollTree::carry_scroll_state`]; seeded at first install by
    /// [`ScrollTree::initial_snap_update`].
    snap_changing_sent: Cell<Option<SnapTargets>>,
    /// The target pair the last `scrollsnapchange` reported for this
    /// container (CSS Scroll Snap 2 §5.1.2): like
    /// [`ScrollContainer::snap_changing_sent`] but compared at scroll
    /// completion ([`ScrollTree::end_tick`]).
    snap_change_sent: Cell<Option<SnapTargets>>,
    /// Whether this container's snap-event slots have been seeded
    /// (by the initial layout-snap update, or at build for a
    /// non-snapping container): an unseeded container has never had
    /// its resting pair baselined, so the first
    /// [`ScrollTree::carry_scroll_state`] runs the initial update for
    /// it instead of diffing against an empty baseline.
    snap_events_seeded: Cell<bool>,
    /// The scroller's computed CSS `scroll-behavior` (`smooth` only when
    /// the value is the `smooth` keyword): the default behavior for
    /// scroll commands that don't name one — `scrollTo()` without
    /// `behavior`, `scrollIntoView()` with `behavior: "auto"`.
    /// Resolved at build from the container's own style (the root
    /// element's, with body fallback, for the viewport); the embedder
    /// also publishes the viewport's onto the shared [`ScrollState`]
    /// (see [`rebuild_after_layout`]) so the script natives can resolve
    /// it synchronously.
    pub css_behavior: ScrollBehavior,
    /// Whether the container's computed `direction` is `rtl`: the
    /// vertical scrollbar sits on the scrollport's inline-end edge
    /// (the left edge under `rtl`). Read from the container's own box
    /// style in [`recover_parents`]; the viewport's comes from the
    /// root element (see [`ScrollTree::build`]).
    pub rtl: bool,
    /// In-flight smooth-scroll animation, if any. Set by
    /// [`ScrollTree::start_smooth_scroll`], advanced by
    /// [`ScrollTree::advance_smooth`]; a new scroll command on this
    /// container replaces it (interruption rule).
    smooth: Cell<Option<SmoothScroll>>,
    /// `scrollend` bookkeeping: the [`ScrollTree`]'s tick counter value
    /// at this container's most recent scroll movement, or 0 when the
    /// container is not mid-scroll. A container counts as scrolling from
    /// its first movement until a full event-loop tick passes with no
    /// movement — then the window worker fires `scrollend` and resets
    /// this to 0. Set through [`ScrollTree::mark_scrolled`], drained by
    /// [`ScrollTree::end_tick`].
    scroll_tick: Cell<u64>,
    /// Transient overscroll (rubber-band) shift in CSS px, per axis: a
    /// PAINT-TIME translation only, decoupled from [`ScrollState`].
    /// Stretched by [`ScrollTree::scroll_chain`] edge presses
    /// (`residual * OVERSCROLL_RESISTANCE`, capped at
    /// [`OVERSCROLL_MAX_SHIFT`]), sprung back to zero by
    /// [`ScrollTree::advance_overscroll`]. Read by the paint worker via
    /// [`ScrollTree::overscroll_shift`]; never moves the layout offset,
    /// never fires `scroll`/`scrollend`.
    overscroll_delta: Cell<(f32, f32)>,
    /// Spring velocity (px/s) of the transient shift. Zeroed on the
    /// pushed axis by each new edge press, so a fresh push never
    /// inherits release momentum from the settling spring.
    overscroll_vel: Cell<(f32, f32)>,
}

/// One smooth-scroll animation in flight on a container: the offset
/// interpolates from `from` to `to` over [`SMOOTH_SCROLL_DURATION`]
/// with [`ease_in_out_cubic`], advanced by
/// [`ScrollTree::advance_smooth`].
#[derive(Debug, Clone, Copy)]
struct SmoothScroll {
    /// Offset the animation started from.
    from: (f32, f32),
    /// Target offset, clamped into the container's `[origin, max]`.
    to: (f32, f32),
    /// Offset written by the last tick: a tick that finds the live
    /// offset elsewhere cancels the animation — some other scroll
    /// command (wheel, key, instant `scrollTo`, `scrollTop` write, ...)
    /// moved the container in between, and it wins (interruption rule).
    last: (f32, f32),
    /// Wall-clock start of the animation.
    start: Instant,
}

/// Wall-clock length of one smooth scroll. The exact duration is
/// implementation-defined (CSSOM View leaves smooth scrolling's timing
/// to the UA); 300ms matches the common browser feel.
const SMOOTH_SCROLL_DURATION: Duration = Duration::from_millis(300);

/// Cubic ease-in-out for smooth scrolls: slow at both ends, fast in the
/// middle. The exact curve is implementation-defined (CSSOM View); this
/// is the standard ease-in-out cubic.
fn ease_in_out_cubic(t: f32) -> f32 {
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
    }
}

// ---------------------------------------------------------------------------
// Overscroll affordance, phase 1: transient rubber-band state
// ---------------------------------------------------------------------------

/// While a gesture pushes past a scroll edge, each px of residual
/// delta stretches the container's transient overscroll shift by this
/// fraction.
///
/// A constant 0.3, not a friction curve: predictable, and the hard
/// [`OVERSCROLL_MAX_SHIFT`] cap bounds a sustained push (e.g. holding
/// an arrow key at the edge) the same way a diminishing curve would.
/// Tunable by the follow-up workers; the paint-time fold only reads
/// [`ScrollTree::overscroll_shift`].
const OVERSCROLL_RESISTANCE: f32 = 0.3;
/// Hard cap on the transient overscroll shift per axis (CSS px):
/// a sustained boundary push saturates here instead of growing
/// without bound.
const OVERSCROLL_MAX_SHIFT: f32 = 96.0;
/// Spring-back angular frequency (rad/s): ~4.5 Hz, so the band settles
/// in roughly 250 ms. Critically damped (`c = 2ω`), so it returns to
/// zero with no overshoot past the edge.
const OVERSCROLL_OMEGA: f32 = 28.0;
/// Largest spring integration step (s): clamps a tab-switch or
/// debugger pause so one huge `dt` cannot fling the band.
const OVERSCROLL_MAX_DT: f32 = 0.05;
/// Rest detection: shift below this (px) with velocity below
/// [`OVERSCROLL_VEL_EPS`] snaps to exactly zero, so the spring
/// terminates instead of decaying asymptotically forever.
const OVERSCROLL_POS_EPS: f32 = 0.02;
/// Rest detection: velocity below this (px/s); see
/// [`OVERSCROLL_POS_EPS`].
const OVERSCROLL_VEL_EPS: f32 = 1.0;
/// Exact closed-form step of the critically-damped overscroll
/// oscillator `x'' = -ω²x - 2ωx'` over `dt` seconds (`ω` =
/// [`OVERSCROLL_OMEGA`]):
///   a = v0 + ω·x0
///   x(t) = (x0 + a·t)·e^(-ωt),  v(t) = (v0 - ω·a·t)·e^(-ωt).
/// Unconditionally stable for any `dt`, unlike the semi-implicit Euler
/// this replaces, which diverged (sign-flipping oscillation, NaN, a
/// spring that never settled) once `dt` passed ~30 ms — reachable via a
/// slow paint turn or SIGSTOP between event-loop turns.
fn overscroll_spring_step(x0: f32, v0: f32, dt: f32) -> (f32, f32) {
    let a = v0 + OVERSCROLL_OMEGA * x0;
    let decay = (-OVERSCROLL_OMEGA * dt).exp();
    let x = (x0 + a * dt) * decay;
    let v = (v0 - OVERSCROLL_OMEGA * a * dt) * decay;
    (x, v)
}

/// Non-finite deltas (`Home`/`End` keys drive
/// [`ScrollTree::scroll_chain`] with ±∞) finite-clamp to this residual
/// before feeding the rubber band: a full-strength push, never NaN/∞.
const EDGE_HIT_MAX_RESIDUAL: f32 = 4096.0;

/// Per-axis alignment for [`ScrollTree::reveal_node_with_options`]:
/// the `scrollIntoView()` `block`/`inline` options (WebIDL
/// `ScrollLogicalPosition`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RevealAlign {
    /// Leave the axis's scroll offset untouched. This is the historical
    /// [`ScrollTree::reveal_node`] behavior on the horizontal axis.
    #[default]
    Keep,
    /// Bring the revealed rect's leading edge (expanded by
    /// `scroll-margin`) to the scrollport's leading edge:
    /// `scrollIntoView({ block: "start" })` / `{ inline: "start" }`.
    Start,
    /// Bring the revealed rect's trailing edge to the scrollport's
    /// trailing edge: `{ block: "end" }` / `{ inline: "end" }`.
    End,
    /// Center the revealed rect in the scrollport: `{ block: "center" }`
    /// / `{ inline: "center" }`.
    Center,
    /// The minimal scroll that makes the revealed rect fully visible:
    /// nothing when it already is; otherwise the leading edges align
    /// when the rect starts above/left of the visible range, the
    /// trailing edges when it ends below/right of it (a rect larger
    /// than the scrollport aligns its leading edge, like browsers).
    /// `{ block: "nearest" }` / `{ inline: "nearest" }` (the `inline`
    /// default).
    Nearest,
}

impl From<bowser_js::ScrollIntoViewAlign> for RevealAlign {
    fn from(a: bowser_js::ScrollIntoViewAlign) -> Self {
        match a {
            bowser_js::ScrollIntoViewAlign::Start => RevealAlign::Start,
            bowser_js::ScrollIntoViewAlign::Center => RevealAlign::Center,
            bowser_js::ScrollIntoViewAlign::End => RevealAlign::End,
            bowser_js::ScrollIntoViewAlign::Nearest => RevealAlign::Nearest,
        }
    }
}

/// Which axis an overscroll edge press pushed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverscrollAxis {
    /// The horizontal axis (the `scrollLeft` direction).
    Horizontal,
    /// The vertical axis (the `scrollTop` direction).
    Vertical,
}

/// Which bound of the scroll range an overscroll press pushed against:
/// bound-relative, so a negative [`ScrollContainer::origin`] (negative
/// scrollable overflow, RTL) reports [`OverscrollEdge::Origin`], never
/// `Max`, no matter the sign of the coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverscrollEdge {
    /// The minimum-offset edge (`origin`).
    Origin,
    /// The maximum-offset edge.
    Max,
}

/// One container-edge press recorded by [`ScrollTree::scroll_chain`]:
/// the container's offset did not move on `axis` while residual delta
/// remained on it — the gesture pressed past the scroll boundary.
///
/// The residual stretched the container's transient overscroll shift
/// (see [`ScrollTree::overscroll_shift`]) unless the pressed axis's
/// `overscroll-behavior` is `none`, which suppresses the affordance
/// while `contain` keeps it.
#[derive(Debug, Clone, Copy)]
pub struct EdgeHit {
    /// Index of the pressed container in this [`ScrollTree`].
    pub index: usize,
    /// Container node (`None` = viewport), matching
    /// [`ScrollOutcome::deltas`].
    pub node: Option<NodeId>,
    /// The axis the residual delta pressed on.
    pub axis: OverscrollAxis,
    /// Which bound the press pushed against.
    pub edge: OverscrollEdge,
    /// The unconsumed delta (CSS px, signed) that pressed against the
    /// edge: finite-clamped (see `EDGE_HIT_MAX_RESIDUAL`).
    pub residual: f32,
}

/// The outcome of dispatching a scroll delta: which containers moved.
#[derive(Debug, Default)]
pub struct ScrollOutcome {
    /// `(container node, consumed delta)` innermost-first; `None` is the
    /// viewport scroller. Only containers whose offset actually changed.
    pub deltas: Vec<(Option<NodeId>, (f32, f32))>,
    /// Overscroll edge presses, innermost-first: containers that clamped
    /// on an axis while residual delta remained on it (see [`EdgeHit`]).
    /// Populated even when no offset moved; feeds the transient
    /// overscroll shift unless the pressed axis is
    /// `overscroll-behavior: none`, which suppresses the affordance.
    pub edge_hits: Vec<EdgeHit>,
    /// The delta that exited the chain unconsumed, in CSS px, signed
    /// per axis: `(0, 0)` when some container consumed everything. An
    /// `overscroll-behavior: contain`/`none` stop kills the remainder
    /// in-chain — the axes are zeroed before publication, so a stop
    /// publishes `(0, 0)` on the stopped axes, not the killed amount.
    /// Otherwise this is the signed leftover no container took: every
    /// scroller clamped on that axis, or the touch path gated the
    /// delta to zero. Published for future consumers (cross-document
    /// chaining, haptic edge feedback); nothing reads it yet.
    pub terminal_residual: (f32, f32),
}

impl ScrollOutcome {
    /// Whether any container's offset moved.
    pub fn moved(&self) -> bool {
        !self.deltas.is_empty()
    }

    /// Whether any container was pressed against a scroll edge.
    pub fn has_edge_hits(&self) -> bool {
        !self.edge_hits.is_empty()
    }

    /// Scrolled container nodes innermost-first (`None` = viewport), for
    /// `scroll` event dispatch.
    pub fn nodes(&self) -> Vec<Option<NodeId>> {
        self.deltas.iter().map(|(node, _)| *node).collect()
    }
}

/// Everything [`ScrollTree::push_container`] needs for one container,
/// bundled so the arg list stays under the too-many-arguments lint.
struct ContainerInit {
    node: Option<NodeId>,
    scrollport: Rect,
    state: Rc<ScrollState>,
    parent: usize,
    user_scroll: (bool, bool),
    /// Scroll-container axes (`auto`/`scroll`/`hidden` per axis).
    axes: (bool, bool),
    /// Computed `scrollbar-gutter`.
    gutter: ScrollbarGutter,
    origin: (f32, f32),
    overscroll: (OverscrollBehavior, OverscrollBehavior),
    /// Computed `touch-action` pan permissions.
    touch_action: TouchAction,
    css_behavior: ScrollBehavior,
    rtl: bool,
    block_is_y: bool,
    viewport_anchored: bool,
    unshifted: bool,
    transform_offset: (f32, f32),
    snap: SnapConfig,
    snap_areas: Vec<SnapAreaInfo>,
    initial_target: Option<(f32, f32)>,
}

/// The scroll containers of one layout: index 0 is always the viewport
/// scroller; nested containers follow in tree order (parents before
/// children, siblings in layout order).
#[derive(Debug)]
pub struct ScrollTree {
    containers: Vec<ScrollContainer>,
    /// `parents[i]`: index of `i`'s scroller ancestor (0 for the viewport).
    parents: Vec<usize>,
    /// `children[i]`: `i`'s nested containers, in tree order.
    children: Vec<Vec<usize>>,
    /// Element node -> container index (viewport excluded).
    index: HashMap<NodeId, usize>,
    /// One [`StickyBand`] per `position: sticky` box that takes
    /// scroll-driven constraints, rebuilt by [`ScrollTree::build`]
    /// after each layout; empty before the first layout. Sticky
    /// constraints are baked at layout time, so a scroll that could
    /// move a stuck box must dirty layout to re-bake them — but only
    /// then: the window worker checks the moved scroller's bands
    /// (usually a handful of range tests) on every scroll instead of
    /// dirtying layout on every tick of a sticky page. See
    /// [`ScrollTree::sticky_constrained_at`].
    ///
    /// Each band also carries the box's bake-time position, insets, and
    /// containing-block relation, which is enough to compute the box's
    /// scroll-driven shift exactly in the common regimes (stuck plateau
    /// with the scroller's own content as containing block, provably
    /// free intervals). [`ScrollTree::sticky_shifts_at`] folds those
    /// shifts into the nested-scroll fast path so the transform wrap
    /// survives scroll ticks that used to re-bake.
    sticky_bands: Vec<StickyBand>,
    /// Event-loop tick counter for `scrollend` bookkeeping: every
    /// container that moves is stamped with the current tick (see
    /// [`ScrollTree::mark_scrolled`]), and [`ScrollTree::end_tick`]
    /// advances the counter once per event-loop turn, firing
    /// `scrollend` for containers stamped on an earlier tick. Carried
    /// across relayouts by [`ScrollTree::carry_scroll_state`] so a
    /// mid-scroll rebuild does not silently drop the pending
    /// `scrollend`. Starts at 1: tick 0 is the "not scrolling" sentinel
    /// (see [`ScrollContainer::scroll_tick`]), so the first real tick
    /// must stamp a nonzero value.
    tick: Cell<u64>,
    /// Wall-clock instant of the last
    /// [`ScrollTree::advance_overscroll`] integration: every
    /// container's overscroll spring advances on the same event-loop
    /// `dt`, so one tree-wide stamp suffices. `None` before the first
    /// integration (the first call establishes the baseline and
    /// integrates nothing).
    overscroll_tick: Cell<Option<Instant>>,
    /// Set when the viewport scroller moved without a paint rebuild
    /// (see [`crate::window::sync_nested_paint`]): the viewport's own
    /// bars and `position: fixed` containers' bars bake the live
    /// viewport offset into page coordinates, so without a repaint
    /// those baked rects would go stale — visually sliding while
    /// [`ScrollTree::scrollbar_at`] hit-tests the live offset. The
    /// next refresh re-pushes just the bar items (see
    /// [`ScrollTree::refresh_epilogue`]) instead of rebuilding
    /// the whole display list, preserving the GPU-uniform fast path.
    /// Cleared by [`ScrollTree::refresh_epilogue`] and by the
    /// full-repaint path, which re-pushes the bars itself.
    scrollbars_dirty: bool,
    /// Display-list item count just before the overscroll glow items
    /// begin: the epilogue (glow, then scrollbars) re-pushes from here
    /// (see [`ScrollTree::refresh_epilogue`]). [`usize::MAX`] until
    /// the first paint.
    glow_split: usize,
    /// The baked glow items are stale: a container's transient
    /// overscroll shift moved without a paint rebuild (the viewport's
    /// own shift rides the per-frame GPU uniform, so it needs no
    /// pipeline dirt). The next `refresh` re-pushes the epilogue
    /// instead of rebuilding the whole display list. Cleared by
    /// [`ScrollTree::refresh_epilogue`] and by the full-repaint path,
    /// which re-pushes the glow itself.
    glow_dirty: bool,
    /// Pending `scrollsnapchanging` / `scrollsnapchange` deliveries
    /// (CSS Scroll Snap 2 §5), queued by the snap-event layer as pairs
    /// change and drained by the window worker at the ordering points
    /// in `window.rs` (before `scroll`, at the top of a turn, and
    /// before `scrollend`). Value payloads only — the tree has no JS
    /// access; dispatch resolves the node ids to wrappers. FIFO.
    snap_events: RefCell<Vec<SnapEventDelivery>>,
    /// Whether this tree came from [`ScrollTree::build`] (a real
    /// layout) rather than [`ScrollTree::for_viewport`] (the
    /// pre-first-layout placeholder): `carry_scroll_state` only
    /// trusts the placeholder's viewport as an event-baseline match
    /// when this is set — the placeholder was never seeded against a
    /// real snap configuration.
    from_build: bool,
}

impl ScrollTree {
    /// A tree with only the viewport scroller: the pre-first-layout
    /// state, where wheels are no-ops. `user_scroll` is the viewport's
    /// wheel/keyboard-scrollable axes — `(true, true)` before the first
    /// layout (the maxima are `(0, 0)` there anyway, so nothing can
    /// move); after layout [`ScrollTree::build`] derives it from
    /// propagated `overflow` (see [`viewport_user_axes`]).
    pub fn for_viewport(
        state: &Rc<ScrollState>,
        viewport: bowser_layout::Size,
        user_scroll: (bool, bool),
    ) -> Self {
        let mut tree = ScrollTree {
            containers: Vec::new(),
            parents: Vec::new(),
            children: Vec::new(),
            index: HashMap::new(),
            sticky_bands: Vec::new(),
            // Tick 0 is the "not scrolling" sentinel; start at 1 so the
            // first real movement stamps a nonzero tick.
            tick: Cell::new(1),
            overscroll_tick: Cell::new(None),
            scrollbars_dirty: false,
            glow_split: usize::MAX,
            glow_dirty: false,
            snap_events: RefCell::new(Vec::new()),
            from_build: false,
        };
        tree.push_container(ContainerInit {
            node: None,
            scrollport: Rect {
                x: 0.0,
                y: 0.0,
                w: viewport.w,
                h: viewport.h,
            },
            state: state.clone(),
            parent: 0,
            user_scroll,
            axes: (true, true),
            gutter: ScrollbarGutter::Auto,
            // The document origin is published by `rebuild_after_layout`
            // once the layout is known; pre-layout nothing scrolls.
            origin: (0.0, 0.0),
            overscroll: (OverscrollBehavior::Auto, OverscrollBehavior::Auto),
            // `touch-action` is not inherited: the viewport's value is
            // published by `ScrollTree::build` from the root style (with
            // the body's as fallback); pre-layout touch pans freely (the
            // `auto` initial).
            touch_action: TouchAction::all(),
            // The CSS default is published by `rebuild_after_layout`
            // once the root style is known; pre-layout everything is
            // instant.
            css_behavior: ScrollBehavior::Auto,
            rtl: false,
            block_is_y: true,
            viewport_anchored: false,
            // The viewport scroller is never out-of-flow.
            unshifted: false,
            // The viewport scroller is never transformed.
            transform_offset: (0.0, 0.0),
            // Snap configuration is resolved by `ScrollTree::build` once
            // the root style is known; pre-layout snapping is inert.
            snap: SnapConfig::none(),
            snap_areas: Vec::new(),
            initial_target: None,
        });
        tree
    }

    fn push_container(&mut self, init: ContainerInit) -> usize {
        let idx = self.containers.len();
        let painted = init.state.offset();
        self.containers.push(ScrollContainer {
            node: init.node,
            scrollport: init.scrollport,
            origin: init.origin,
            overscroll: init.overscroll,
            touch_action: init.touch_action,
            css_behavior: init.css_behavior,
            rtl: init.rtl,
            block_is_y: init.block_is_y,
            // A non-snapping container's event baseline is the all-null
            // pair from the start: it can never report a target, so its
            // slots are seeded here and it never queues an event. A
            // snapping container baselines at its first install (see
            // [`ScrollTree::initial_snap_update`]).
            snap_changing_sent: Cell::new(if init.snap.snap_type == SnapType::None {
                Some(SnapTargets::default())
            } else {
                None
            }),
            snap_change_sent: Cell::new(if init.snap.snap_type == SnapType::None {
                Some(SnapTargets::default())
            } else {
                None
            }),
            snap_events_seeded: Cell::new(init.snap.snap_type == SnapType::None),
            state: init.state,
            painted: Cell::new(painted),
            painted_os: Cell::new((0.0, 0.0)),
            paint_base: Cell::new(painted),
            paint_base_os: Cell::new((0.0, 0.0)),
            transform_fast: Cell::new(false),
            oof_wraps: RefCell::new(Vec::new()),
            oof_wrap_items: RefCell::new(Vec::new()),
            sticky_wrap_items: RefCell::new(Vec::new()),
            wrap_item: Cell::new(None),
            wrap_base: Cell::new(painted),
            wrap_base_os: Cell::new((0.0, 0.0)),
            clip_rect: Cell::new(None),
            user_scroll: init.user_scroll,
            axes: init.axes,
            gutter: init.gutter,
            viewport_anchored: init.viewport_anchored,
            unshifted: init.unshifted,
            transform_offset: init.transform_offset,
            snap: init.snap,
            snap_areas: init.snap_areas,
            initial_target: init.initial_target,
            snap_base: Cell::new(painted),
            snap_rest: Cell::new((None, None)),
            smooth: Cell::new(None),
            scroll_tick: Cell::new(0),
            overscroll_delta: Cell::new((0.0, 0.0)),
            overscroll_vel: Cell::new((0.0, 0.0)),
        });
        self.parents.push(init.parent);
        self.children.push(Vec::new());
        if init.parent != idx {
            self.children[init.parent].push(idx);
        }
        if let Some(n) = init.node {
            // One scrollable box per NodeId: `discover` emits one entry
            // per box, and bake/sync's `entered[ci]` first-match logic
            // shifts exactly one box per container node. Last-wins here
            // would silently drop a container's shift.
            //
            // NOTE: the insert must happen outside `debug_assert!` —
            // the assert's expression is compiled out in release
            // builds, and housing the insert inside it left
            // `self.index` empty in release (141 scroll/window test
            // failures, wave 42).
            let prev = self.index.insert(n, idx);
            debug_assert!(prev.is_none());
        }
        idx
    }

    /// Discover the containers of `root` and link them into a tree.
    ///
    /// Publishes each container's maximum offset into `elements`
    /// (reusing the previous layout's shared states so offsets survive
    /// relayout, re-clamped into the new range, and pruning containers
    /// that went away). The viewport container aliases `doc_state`
    /// ([`Tab::scroll()`]).
    pub fn build(
        doc: &Document,
        root: &LayoutBox,
        viewport: bowser_layout::Size,
        doc_state: &Rc<ScrollState>,
        elements: &Rc<ElementScrollStates>,
    ) -> Self {
        // `rem` in `scroll-padding`/`scroll-margin` resolves against the
        // root element's computed font size; a style-less root falls back
        // to the CSS initial value (`medium` = 16px).
        let root_font_size = root
            .style
            .as_ref()
            .map(|s| px_font_size(s, 16.0))
            .unwrap_or(16.0);
        let discovered = discover(doc, root, viewport);
        // Same container set and order as discovery: each entry also
        // carries layout's `scroll_origin` so the shared states' minima
        // match the app-side `[origin, max]` clamps. All-zero origins
        // are identical to the old `publish` call (which forwarded
        // `(0, 0)` minima itself).
        elements.publish_with_origins(
            &discovered
                .iter()
                .map(|d| (d.node, d.origin, d.max))
                .collect::<Vec<_>>(),
        );
        let mut tree = ScrollTree::for_viewport(doc_state, viewport, viewport_user_axes(doc, root));
        // The root element's `overscroll-behavior` applies to the
        // viewport, falling back per-axis to the body's computed value
        // when the root's is `auto` (CSS Overscroll Behavior §2: the
        // used value propagates from `body` when the root is `auto`).
        // Behaviorally neutral for chaining (the chain breaks at the
        // viewport regardless), but the field stays truthful for the
        // overscroll affordance.
        let (mut obx, mut oby) = overscroll_axes(root.style.as_deref());
        if obx == OverscrollBehavior::Auto || oby == OverscrollBehavior::Auto {
            let (bbx, bby) = overscroll_axes(body_style(doc, root));
            if obx == OverscrollBehavior::Auto {
                obx = bbx;
            }
            if oby == OverscrollBehavior::Auto {
                oby = bby;
            }
        }
        tree.containers[0].overscroll = (obx, oby);
        // The viewport's CSS `scroll-behavior` default: the root
        // element's value, falling back to the body's when the root's is
        // `auto` (see `viewport_scroll_behavior`).
        tree.containers[0].css_behavior = viewport_scroll_behavior(doc, root);
        // The viewport's inline-end edge mirrors under `direction: rtl`:
        // the viewport's vertical scrollbar flips to the left edge,
        // matching nested containers. The root element's computed value
        // wins; when the root does not specify one (computed `ltr` from
        // the initial value), the body's propagates — the common
        // authoring pattern of `direction: rtl` on `<body>` with nothing
        // on `<html>` — mirroring the `scroll-behavior`/`writing-mode`
        // body fallbacks (`viewport_scroll_behavior`/`body_style`).
        tree.containers[0].rtl = viewport_is_rtl(doc, root);
        // The viewport's block-axis mapping: the root element's
        // `writing-mode`, falling back to the body's when the root
        // does not specify one (mirroring `viewport_snap_type`).
        tree.containers[0].block_is_y = match root
            .style
            .as_deref()
            .and_then(|s| s.get_str("writing-mode"))
        {
            Some(w) => !is_vertical_wm(w),
            None => writing_block_is_y(body_style(doc, root)),
        };
        // `scrollbar-gutter` propagates to the viewport like `overflow`
        // does: the root element's value wins, falling back to the
        // body's when the root's is `auto` (mirrors
        // `viewport_scroll_behavior`). PROVISIONAL: CSS Overflow 4 was
        // not verified to mandate this propagation for
        // `scrollbar-gutter`; the rule is an analogy to `overflow`
        // propagation.
        let root_gutter = scrollbar_gutter(root.style.as_deref());
        tree.containers[0].gutter = if root_gutter == ScrollbarGutter::Auto {
            scrollbar_gutter(body_style(doc, root))
        } else {
            root_gutter
        };
        // `touch-action` propagates to the viewport the same way: the
        // root element's value wins, falling back to the body's when the
        // root's is `auto` (mirrors `viewport_scroll_behavior`). Only an
        // explicit non-`auto` keyword on the root skips the fallback, so
        // `manipulation` (behaviorally `auto` for panning) still picks
        // up the body's value.
        let root_touch = root
            .style
            .as_deref()
            .and_then(|s| s.get_by_index(props::TOUCH_ACTION));
        let root_touch_is_auto = match root_touch {
            None => true,
            Some(bowser_css::CssValue::Keyword(k)) => k.eq_ignore_ascii_case("auto"),
            Some(_) => false,
        };
        tree.containers[0].touch_action = if root_touch_is_auto {
            touch_action_from_value(
                body_style(doc, root).and_then(|s| s.get_by_index(props::TOUCH_ACTION)),
            )
        } else {
            touch_action_from_value(root_touch)
        };
        tree.containers[0].axes = viewport_container_axes(doc, root);
        // The viewport's scroll-snap configuration:
        // `scroll-snap-type` (root element, body's as fallback —
        // mirroring `viewport_scroll_behavior`), `scroll-snap-stop`
        // and `scroll-padding` on the document element.
        tree.containers[0].snap = SnapConfig {
            snap_type: viewport_snap_type(doc, root),
            padding: scroll_padding(
                root.style.as_deref(),
                root_font_size,
                viewport.w,
                viewport.h,
                viewport.w,
                viewport.h,
            ),
        };
        // The placeholder viewport was seeded as a non-snapping
        // container by `for_viewport`; a real snap type baselines at
        // first install instead (see `initial_snap_update`).
        if tree.containers[0].snap.snap_type != SnapType::None {
            tree.containers[0].snap_changing_sent.set(None);
            tree.containers[0].snap_change_sent.set(None);
            tree.containers[0].snap_events_seeded.set(false);
        }
        // `discovered` is in tree order (parents precede children) and
        // each entry carries its scroller ancestor from
        // `recover_parents`, so the parent index resolves directly —
        // no depth arithmetic.
        let container_nodes: HashSet<NodeId> = discovered.iter().map(|d| d.node).collect();
        // The viewport scroller's snap areas: the document element's
        // in-flow descendants (nested containers prune into their own
        // lists, exactly as for a nested scroller). Container 0 is not
        // part of `discovered` (the viewport scroller is filtered out
        // there), so without this its list stays empty and viewport
        // scrolls — `window.scrollTo()`, wheel, keyboard — never snap
        // at all (wave 50).
        if let Some(root_node) = root.node.or_else(|| doc.document_element()) {
            tree.containers[0].snap_areas =
                snap_areas_for_engine(root, root_node, &container_nodes, doc);
        }
        for d in &discovered {
            let parent = d.parent.and_then(|n| tree.index_of(n)).unwrap_or(0);
            let state = elements
                .state_for(d.node)
                .expect("published immediately above");
            // Snap areas of the container's in-flow descendants, in
            // container content coordinates (scroll-margin expanded by
            // the layout query); pruned at nested container boxes so
            // each subtree contributes to exactly one container.
            let snap_areas = snap_areas_for_engine(root, d.node, &container_nodes, doc);
            let initial_target = initial_target_for(
                root,
                d.node,
                &container_nodes,
                d.scrollport,
                d.snap.padding,
                d.origin,
                d.max,
            );
            tree.push_container(ContainerInit {
                node: Some(d.node),
                scrollport: d.scrollport,
                state,
                parent,
                user_scroll: d.user_axes,
                axes: d.axes,
                gutter: d.gutter,
                origin: d.origin,
                overscroll: d.overscroll,
                touch_action: d.touch_action,
                css_behavior: d.css_behavior,
                rtl: d.rtl,
                block_is_y: d.block_is_y,
                viewport_anchored: d.viewport_anchored,
                unshifted: d.unshifted,
                transform_offset: d.transform_offset,
                snap: d.snap,
                snap_areas,
                initial_target,
            });
        }
        // The viewport scroller's initial target, resolved once the
        // nested index is complete so `document_max_scroll` prunes
        // nested interiors exactly like the caller's max publish.
        if let Some(root_node) = root.node.or_else(|| doc.document_element()) {
            let (origin, max) = tree.document_max_scroll(root, viewport);
            tree.containers[0].initial_target = initial_target_for(
                root,
                root_node,
                &container_nodes,
                Rect {
                    x: 0.0,
                    y: 0.0,
                    w: viewport.w,
                    h: viewport.h,
                },
                tree.containers[0].snap.padding,
                origin,
                max,
            );
        }
        // One walk of the fresh layout tree per layout: the window
        // worker's per-scroll sticky check scans these bands instead of
        // walking the tree.
        tree.sticky_bands = tree.collect_sticky_bands(root, viewport);
        tree.from_build = true;
        tree
    }

    /// Number of containers (viewport included).
    pub fn len(&self) -> usize {
        self.containers.len()
    }

    /// Whether the tree holds no containers at all (never true for
    /// built trees, which always contain the viewport scroller).
    pub fn is_empty(&self) -> bool {
        self.containers.is_empty()
    }

    /// Whether the tree holds only the viewport scroller.
    pub fn is_viewport_only(&self) -> bool {
        self.containers.len() <= 1
    }

    /// Whether the layout tree this tree was built from contains a
    /// `position: sticky` box that takes scroll-driven constraints.
    /// Sticky boxes under a `fixed` ancestor take none (mirroring
    /// `bowser_layout`'s sticky pass), so they are not counted. The
    /// window worker dirties layout on scroll only then (see the field
    /// docs and [`crate::window::sync_nested_paint`]).
    pub fn has_sticky(&self) -> bool {
        !self.sticky_bands.is_empty()
    }

    /// Whether any sticky box constrained by scroller `scroller`
    /// (0 = viewport) may be stuck or transitioning at offset
    /// `(x, y)`: the hysteresis gate behind [`ScrollTree::has_sticky`].
    ///
    /// For nested scrollers this is fold-aware, not a pure band test:
    /// it reports `false` while every in-band box's scroll-driven shift
    /// is exactly foldable (see [`ScrollTree::sticky_shifts_at`]), so
    /// those ticks keep the transform fast path instead of dirtying
    /// layout. It reports `true` when a box may be in a constraint
    /// transition — the band still over-approximates, and every
    /// approximation errs toward re-baking. The viewport scroller keeps
    /// the pure band test: its sticky boxes ride the GPU uniform and
    /// have no per-box fold mechanism.
    pub fn sticky_constrained_at(&self, scroller: usize, x: f32, y: f32) -> bool {
        if scroller == 0 {
            return self.sticky_bands.iter().any(|b| {
                b.scroller == scroller && x >= b.lo_x && x <= b.hi_x && y >= b.lo_y && y <= b.hi_y
            });
        }
        self.sticky_bands
            .iter()
            .any(|b| b.scroller == scroller && self.sticky_band_fold(b, x, y).is_none())
    }

    /// Exact per-box sticky shifts for a nested scroll to `(x, y)`.
    ///
    /// For every `position: sticky` box constrained by `scroller` whose
    /// band covers `(x, y)`, returns the extra `(dx, dy)` to apply to
    /// that box's items **on top of** the scroller's scroll transform —
    /// e.g. via a dedicated `PushTransform`/`PopTransform` pair whose
    /// matrix the tick path patches alongside
    /// [`ScrollTree::try_fast_fold`]'s scroll-transform patch. Boxes
    /// whose exact shift is zero are omitted. Shifts are relative to
    /// the band's bake offset; the scroller's own transform stays the
    /// caller's responsibility.
    ///
    /// Returns `None` when any in-band box's shift is not exactly
    /// foldable (a constraint transition may be in flight): the caller
    /// must re-bake ([`ScrollTree::sync_paint_offsets`] + full rebuild),
    /// which also re-tracks the bands. The viewport scroller (0) always
    /// returns `None`: its sticky boxes ride the per-frame GPU uniform
    /// and have no per-box fold mechanism, so they keep the re-bake
    /// path.
    pub(crate) fn sticky_shifts_at(
        &self,
        scroller: usize,
        x: f32,
        y: f32,
    ) -> Option<Vec<StickyShift>> {
        if scroller == 0 {
            return None;
        }
        let mut shifts = Vec::new();
        for b in &self.sticky_bands {
            if b.scroller != scroller {
                continue;
            }
            let (dx, dy) = self.sticky_band_fold(b, x, y)?;
            if dx != 0.0 || dy != 0.0 {
                shifts.push(StickyShift {
                    node: b.node,
                    dx,
                    dy,
                });
            }
        }
        Some(shifts)
    }

    /// The container at `index` (0 = viewport).
    pub fn container(&self, index: usize) -> &ScrollContainer {
        &self.containers[index]
    }

    /// Index of `index`'s scroller ancestor (0 for the viewport itself).
    pub fn parent(&self, index: usize) -> usize {
        self.parents[index]
    }

    /// Container index for an element node, if it is a scroll container.
    pub fn index_of(&self, node: NodeId) -> Option<usize> {
        self.index.get(&node).copied()
    }

    /// The live scroll offset of every nested scroll container, keyed
    /// by the container's DOM node — the map
    /// [`layout_with_scroll_and_offsets`](bowser_layout::layout_with_scroll_and_offsets)
    /// takes for its per-scroller sticky constraints.
    ///
    /// No key translation is needed: [`ScrollContainer::node`] *is*
    /// the scroller element's DOM [`NodeId`], the same id
    /// [`LayoutBox::node`](bowser_layout::LayoutBox::node) reports for
    /// the scroller's box (arena box indices are not stable across
    /// layouts, which is why layout keys on the DOM node). The
    /// viewport scroller is excluded — it rides layout's `scroll`
    /// argument instead. Containers at offset zero are omitted: an
    /// absent entry constrains at zero, so the smaller map is
    /// behaviorally identical.
    ///
    /// The offsets come from the containers' shared [`ScrollState`]s,
    /// which survive relayout ([`ElementScrollStates`] publishes the
    /// same `Rc`s across builds, re-clamped), so this is valid while
    /// `refresh` runs layout — before `rebuild_after_layout` installs
    /// the new tree.
    pub fn nested_offsets(&self) -> HashMap<NodeId, (f32, f32)> {
        let mut map = HashMap::new();
        // Index 0 is the viewport scroller; nested containers follow.
        for c in self.containers.iter().skip(1) {
            if let Some(node) = c.node {
                let (ox, oy) = c.state.offset();
                if ox != 0.0 || oy != 0.0 {
                    map.insert(node, (ox, oy));
                }
            }
        }
        map
    }

    /// Innermost scroll container containing viewport point `(x, y)`.
    ///
    /// Descends the tree accumulating ancestor scroll offsets: a child
    /// container's scrollport (in page coords) contains the cursor when
    /// the page point plus the ancestors' offsets falls inside it, and
    /// the child's own offset joins the accumulator on the way down.
    /// Layout's scrollports are pre-transform, so the point is
    /// un-translated by each child's baked [`ScrollContainer::transform_offset`]
    /// before the test — input hits the visual location. Children are
    /// visited last-to-first: later siblings paint on top, so they win.
    pub fn innermost_at(&self, x: f32, y: f32) -> usize {
        if self.containers.is_empty() {
            return 0;
        }
        let (vox, voy) = self.containers[0].state.offset();
        // The descent point starts in page coordinates (viewport point +
        // the viewport's offset) and accumulates each descended
        // container's own offset, matching layout's offset-0 frame for
        // nested scrollports. Viewport-anchored (`position: fixed`)
        // containers keep viewport-coordinate scrollports that never
        // move with page-frame scrolls: crossing from a page-frame
        // container into one resets the descent base to the raw viewport
        // point (plus the anchored container's own offset), instead of
        // accumulating a stale build-time translation (finding #1).
        // Anchored-ness propagates down the box tree, so an anchored
        // container's whole scroll subtree is anchored and the base
        // never crosses back.
        let (mut px, mut py) = (x + vox, y + voy);
        let mut idx = 0;
        loop {
            let idx_anchored = self.containers[idx].viewport_anchored;
            let mut next = None;
            for &child in self.children[idx].iter().rev() {
                let child_anchored = self.containers[child].viewport_anchored;
                // Crossing page frame -> viewport frame: test the raw
                // viewport point; otherwise the descent point is
                // already in the child's frame.
                let (qx, qy) = if child_anchored && !idx_anchored {
                    (x, y)
                } else {
                    (px, py)
                };
                let child_c = &self.containers[child];
                // Layout's scrollports are pre-transform: un-translate
                // the point into the container's local frame before
                // testing (see `Discovered::transform_offset`).
                let (tx, ty) = child_c.transform_offset;
                let (qx, qy) = (qx - tx, qy - ty);
                let r = &child_c.scrollport;
                if qx >= r.x && qx < r.x + r.w && qy >= r.y && qy < r.y + r.h {
                    next = Some(child);
                    break;
                }
            }
            match next {
                Some(c) => {
                    let (ox, oy) = self.containers[c].state.offset();
                    if self.containers[c].viewport_anchored && !idx_anchored {
                        px = x + ox;
                        py = y + oy;
                    } else {
                        px += ox;
                        py += oy;
                    }
                    idx = c;
                }
                None => return idx,
            }
        }
    }

    /// Scroll the container at `start` by `(dx, dy)`, chaining the
    /// unconsumed remainder per axis to outer scrollers up to the
    /// viewport. Each container clamps into `[origin, max]`; a container
    /// that cannot move on an axis passes that axis's remainder up. An
    /// `overflow: hidden` axis never consumes a user delta at all: the
    /// full delta chains past it.
    ///
    /// `overscroll-behavior: contain`/`none` on an axis stops the chain
    /// there instead: the remainder on that axis dies at the container
    /// rather than propagating outward (CSS Overscroll Behavior). The
    /// two values agree on the chaining decision; the distinction is
    /// kept in [`OverscrollBehavior`] for the overscroll affordance.
    ///
    /// A user scroll is a relative scroll operation: every container
    /// the chain moved is snap re-evaluated afterwards
    /// ([`ScrollTree::eval_snap`]), launching a follow-up smooth scroll
    /// when its snap target differs from the resting offset.
    ///
    /// Production wheel handling reaches this through `apply_wheel` in
    /// `window.rs`, which picks `start` via the DOM hit test first (a
    /// hit-tested overlay must intercept the gesture) and falls back to
    /// the pure-geometry [`ScrollTree::innermost_at`].
    pub fn scroll_chain(&self, start: usize, dx: f32, dy: f32) -> ScrollOutcome {
        self.scroll_chain_impl(start, dx, dy, false)
    }

    /// Touch-driven variant of [`ScrollTree::scroll_chain`] (Pointer
    /// Events 3 §5.2): the pan permissions are intersected up to each
    /// element that implements the gesture, so when the unconsumed
    /// remainder chains outward to an outer scroller, that scroller
    /// implements that portion and its own `touch-action` mask gates
    /// the remainder before it may consume it. Concretely: an inner
    /// `touch-action: auto` scroller at its scroll edge passes a
    /// vertical remainder up, but an outer `touch-action: pan-x`
    /// scroller drops it instead of panning vertically.
    ///
    /// Wheel and keyboard scrolls must keep using the ungated
    /// [`ScrollTree::scroll_chain`]: `touch-action` does not apply to
    /// them, and gating their chains would break e.g. a vertical wheel
    /// delta chaining into a `touch-action: pan-x` outer scroller.
    pub fn scroll_chain_touch(&self, start: usize, dx: f32, dy: f32) -> ScrollOutcome {
        self.scroll_chain_impl(start, dx, dy, true)
    }

    /// Shared chain loop. `touch_gating` re-gates the residual through
    /// each container's `touch-action` mask as it ascends (see
    /// [`ScrollTree::scroll_chain_touch`]); the wheel/keyboard path
    /// passes `false` and stays ungated.
    fn scroll_chain_impl(
        &self,
        start: usize,
        dx: f32,
        dy: f32,
        touch_gating: bool,
    ) -> ScrollOutcome {
        let mut outcome = ScrollOutcome::default();
        if self.containers.is_empty() {
            // No scroller exists to take anything: the whole delta is
            // residual, same as a fully-clamped chain.
            outcome.terminal_residual = (dx, dy);
            return outcome;
        }
        // Containers this chain step moved, with their pre-step offset:
        // scroll snap re-evaluates each of them after the loop.
        let mut moved: Vec<(usize, (f32, f32))> = Vec::new();
        let (mut rx, mut ry) = (dx, dy);
        let mut idx = start.min(self.containers.len() - 1);
        loop {
            let c = &self.containers[idx];
            // Touch path only (`touch_gating`): re-gate the residual
            // through this container's `touch-action` mask before it
            // may consume anything — every scroller the remainder
            // ascends through implements its own portion (Pointer
            // Events 3 §5.2). Zeroes stay zeroed, so gating the anchor
            // again after the touch pipeline's entry gate is idempotent.
            if touch_gating {
                let (gx, gy) = c.touch_action.gate(rx, ry);
                rx = gx;
                ry = gy;
            }
            // Only `auto`/`scroll` axes consume wheel/keyboard deltas: an
            // `overflow: hidden` axis lets the delta pass through to the
            // next scroller untouched. JS-driven scrolls bypass this mask
            // — they write the shared state directly.
            let (sx, sy) = (
                if c.user_scroll.0 { rx } else { 0.0 },
                if c.user_scroll.1 { ry } else { 0.0 },
            );
            let before = c.state.offset();
            // Clamp into `[origin, max]` per axis: `origin` is negative
            // when content overflows above/left of the scrollport. The
            // clamp is [`ScrollState::set_offset`]'s own degenerate-safe
            // one — the minimum is published alongside the maxima (see
            // [`ScrollTree::build`], [`rebuild_after_layout`]), and a
            // degenerate publish (`origin` above `max`) collapses the
            // range instead of panicking `f32::clamp` the way a raw
            // `.clamp(origin, max)` would.
            c.state.set_offset(before.0 + sx, before.1 + sy);
            let after = c.state.offset();
            let (cx, cy) = (after.0 - before.0, after.1 - before.1);
            if cx != 0.0 || cy != 0.0 {
                outcome.deltas.push((c.node, (cx, cy)));
                moved.push((idx, before));
            }
            // Overscroll affordance, phase 1: a container that clamped
            // on an axis while residual delta remained was pressed
            // against its scroll edge. The edge press is recorded and
            // the container's transient overscroll shift stretched — a
            // paint-time translation only; the offset never leaves
            // `[origin, max]`, so no `scroll` event fires. Recorded
            // before the `contain`/`none` remainder kill below: the
            // press happened regardless of whether the remainder dies
            // here, and every input path (wheel, keyboard, touch drag,
            // fling) funnels through this loop.
            self.record_edge_hit(&mut outcome, idx, OverscrollAxis::Horizontal, sx, cx);
            self.record_edge_hit(&mut outcome, idx, OverscrollAxis::Vertical, sy, cy);
            rx -= cx;
            ry -= cy;
            // `overscroll-behavior: contain`/`none` on an axis: the
            // remainder on that axis dies here — it must not propagate
            // past this container.
            if c.overscroll.0 != OverscrollBehavior::Auto {
                rx = 0.0;
            }
            if c.overscroll.1 != OverscrollBehavior::Auto {
                ry = 0.0;
            }
            // No float-dust kill here (finding #8): sub-1/4096px
            // residues are real scroll distance (high-precision
            // trackpads), and the loop provably terminates — parent
            // indices strictly decrease, so it runs at most depth
            // iterations regardless of residue size.
            if idx == 0 || (rx == 0.0 && ry == 0.0) {
                break;
            }
            idx = self.parents[idx];
        }
        // Terminal residual: what exited the chain unconsumed — the
        // chain ended at the viewport (idx 0) with delta left over. An
        // overscroll-behavior stop zeroed its axes above, so it
        // publishes (0, 0) there, not the killed amount. Published on
        // the outcome for future consumers; the edge-hit affordance
        // above already recorded the presses that produced it.
        outcome.terminal_residual = (rx, ry);
        // Scroll snap (CSS Scroll Snap 1): a user scroll is a relative
        // scroll operation, so each container it moved re-evaluates its
        // snap target from its pre-chain offset and launches a
        // follow-up smooth scroll when one applies.
        for (idx, pre) in moved {
            let natural = self.containers[idx].state.offset();
            self.eval_snap(idx, pre, natural, true);
        }
        outcome
    }

    /// Record one axis's overscroll edge press in `outcome` and stretch
    /// the container's transient overscroll shift. `offered` is the
    /// delta the container was offered on the axis, `consumed` what
    /// `set_offset` actually moved: a clamp (`consumed == 0.0`) with
    /// residual left (`offered != 0.0`) is a press against the edge the
    /// delta pointed at (`offered > 0.0` → [`OverscrollEdge::Max`],
    /// `offered < 0.0` → [`OverscrollEdge::Origin`]) — bound-relative,
    /// so a negative origin under RTL reports `Origin`, never `Max`.
    ///
    /// `overscroll-behavior: none` on the axis suppresses the
    /// affordance: the hit is still recorded in the outcome (the press
    /// happened), but the transient shift stays zero. `contain` keeps
    /// the affordance while stopping the chain.
    ///
    /// Only full clamps record: a container that consumed part of the
    /// delta is not at a boundary press — its remainder chains outward
    /// and the terminal clamper records instead.
    fn record_edge_hit(
        &self,
        outcome: &mut ScrollOutcome,
        idx: usize,
        axis: OverscrollAxis,
        offered: f32,
        consumed: f32,
    ) {
        if consumed != 0.0 || offered == 0.0 || offered.is_nan() {
            return;
        }
        let residual = if offered.is_finite() {
            offered
        } else {
            EDGE_HIT_MAX_RESIDUAL * offered.signum()
        };
        let c = &self.containers[idx];
        outcome.edge_hits.push(EdgeHit {
            index: idx,
            node: c.node,
            axis,
            edge: if offered > 0.0 {
                OverscrollEdge::Max
            } else {
                OverscrollEdge::Origin
            },
            residual,
        });
        let behavior = match axis {
            OverscrollAxis::Horizontal => c.overscroll.0,
            OverscrollAxis::Vertical => c.overscroll.1,
        };
        if behavior == OverscrollBehavior::None {
            return;
        }
        let (mut dx, mut dy) = c.overscroll_delta.get();
        let (mut vx, mut vy) = c.overscroll_vel.get();
        let push =
            (residual * OVERSCROLL_RESISTANCE).clamp(-OVERSCROLL_MAX_SHIFT, OVERSCROLL_MAX_SHIFT);
        match axis {
            OverscrollAxis::Horizontal => {
                dx = (dx + push).clamp(-OVERSCROLL_MAX_SHIFT, OVERSCROLL_MAX_SHIFT);
                vx = 0.0;
            }
            OverscrollAxis::Vertical => {
                dy = (dy + push).clamp(-OVERSCROLL_MAX_SHIFT, OVERSCROLL_MAX_SHIFT);
                vy = 0.0;
            }
        }
        c.overscroll_delta.set((dx, dy));
        c.overscroll_vel.set((vx, vy));
    }

    /// Mark the containers in `nodes` (`None` = viewport) as having
    /// moved on the current event-loop tick, for `scrollend`
    /// bookkeeping. The window worker calls this from its single
    /// `scroll`-event funnel, so every movement path — wheel,
    /// keyboard, fragment navigation, `scrollIntoView()`, script-driven
    /// scrolls — is covered. Nodes that are not scroll containers in
    /// this tree (stale ids) are skipped, never misattributed to the
    /// viewport.
    ///
    /// Script-driven instant scrolls bypass `scroll_chain` and
    /// `advance_smooth`, so as a backstop any marked container with
    /// unevaluated movement (live offset != `snap_base`) and no smooth
    /// scroll in flight is snap re-evaluated here as a non-relative
    /// scroll ([`ScrollTree::eval_snap`]).
    pub fn mark_scrolled(&self, nodes: &[Option<NodeId>]) {
        self.mark_scrolled_impl(nodes, false);
    }

    /// [`mark_scrolled`](Self::mark_scrolled) without the scroll-snap
    /// backstop, for direct manipulation (scrollbar thumb drags): the
    /// backstop would snap-evaluate every motion, launching a smooth
    /// scroll toward the snap point that fights the cursor
    /// mid-gesture. `scrollend` tick bookkeeping is unaffected. The
    /// gesture settles with one non-relative evaluation on release
    /// ([`ScrollTree::snap_after_direct_manipulation`]).
    pub fn mark_scrolled_direct(&self, nodes: &[Option<NodeId>]) {
        self.mark_scrolled_impl(nodes, true);
    }

    /// Settle scroll-snap after direct manipulation ends — a scrollbar
    /// thumb-drag release, or a mid-gesture cancel: one non-relative
    /// [`ScrollTree::eval_snap`] for `node`'s container, mirroring
    /// [`mark_scrolled`](Self::mark_scrolled)'s backstop but exactly
    /// once. No-op when the container is gone or a smooth scroll is
    /// already in flight.
    pub fn snap_after_direct_manipulation(&self, node: Option<NodeId>) {
        let Some(idx) = self.scrollbar_container_idx(node) else {
            return;
        };
        let Some(c) = self.containers.get(idx) else {
            return;
        };
        if c.smooth.get().is_some() {
            return;
        }
        let live = c.state.offset();
        let base = c.snap_base.get();
        if live != base {
            self.eval_snap(idx, base, live, false);
        }
    }

    fn mark_scrolled_impl(&self, nodes: &[Option<NodeId>], direct: bool) {
        let tick = self.tick.get();
        for node in nodes {
            let idx = match node {
                None => 0,
                Some(n) => match self.index_of(*n) {
                    Some(i) => i,
                    None => continue,
                },
            };
            let Some(c) = self.containers.get(idx) else {
                continue;
            };
            c.scroll_tick.set(tick);
            // Scroll snap backstop: script-driven instant scrolls
            // (`scrollTop` writes, instant `scrollTo`, instant reveals)
            // move the shared state without passing through
            // `scroll_chain` or `advance_smooth`, so they never see a
            // snap evaluation otherwise. `live != snap_base` means
            // unevaluated movement since the last snap check; a
            // container with a smooth scroll in flight is skipped —
            // its completion (or interruption) evaluates instead, so
            // this never double-launches against `scroll_chain`'s own
            // hook. These are non-relative scroll operations, so
            // `scroll-snap-stop: always` degrades to `normal` here.
            // Skipped for direct manipulation (thumb drags): snapping
            // mid-gesture fights the cursor; the gesture settles once
            // on release instead.
            if direct {
                // ...but `scrollsnapchanging` must still track the
                // drag: peek at the selection the dragged offset would
                // settle on and report pair changes, moving nothing.
                self.peek_changing(idx);
            } else if c.smooth.get().is_none() {
                let live = c.state.offset();
                if live != c.snap_base.get() {
                    self.eval_snap(idx, c.snap_base.get(), live, false);
                }
            }
        }
    }

    /// End the current event-loop tick: advance the tick counter and
    /// return the containers whose scroll gesture ended — stamped
    /// scrolling on an earlier tick but with no movement this tick —
    /// innermost-first (`None` = viewport), clearing their scrolling
    /// state. The window worker calls this once per event-loop turn,
    /// after firing that turn's `scroll` events, so a container that
    /// moved this tick is never ended in the same tick: `scrollend`
    /// fires on the first quiet tick after the last movement.
    pub fn end_tick(&self) -> Vec<Option<NodeId>> {
        let tick = self.tick.get();
        let mut ended = Vec::new();
        for (idx, c) in self.containers.iter().enumerate() {
            let stamped = c.scroll_tick.get();
            if stamped != 0 && stamped != tick {
                ended.push(c.node);
                c.scroll_tick.set(0);
                // `scrollsnapchange` (CSS Scroll Snap 2 §5.1.2): the
                // scroll completed, so the settled pair is the
                // container's current pair; queue the event if it
                // differs from the pair the last `scrollsnapchange`
                // reported. The window worker drains the queue before
                // firing `scrollend` for these containers.
                self.note_snap_event(idx, SnapEventKind::Change);
            }
        }
        // Containers are stored parents-before-children; reverse for
        // the innermost-first order the `scroll` dispatches use.
        ended.reverse();
        self.tick.set(tick + 1);
        ended
    }

    /// Drain the queued snap-event deliveries (FIFO). The window
    /// worker calls this at its three ordering points (before the
    /// `scroll` dispatches of a movement, at the top of a turn for
    /// layout-snap deliveries, and before `scrollend`); taking the
    /// whole queue before dispatching any of it keeps a listener that
    /// scrolls from looping the drain — its deliveries wait for the
    /// next drain point.
    pub fn take_snap_events(&self) -> Vec<SnapEventDelivery> {
        std::mem::take(&mut *self.snap_events.borrow_mut())
    }

    /// Queue a snap event of `kind` for container `idx` when its
    /// current target pair differs from the pair that kind last
    /// reported, recording the pair as reported. The single dedup
    /// point behind `scrollsnapchanging` (snap evaluations, direct
    /// manipulation peeks) and `scrollsnapchange` (scroll completion,
    /// layout snaps).
    fn note_snap_event(&self, idx: usize, kind: SnapEventKind) {
        let Some(c) = self.containers.get(idx) else {
            return;
        };
        self.note_snap_event_pair(c, kind, c.snap_targets());
    }

    /// [`ScrollTree::note_snap_event`] with the pair supplied instead
    /// of read from `snap_rest`: the direct-manipulation peek reports
    /// the pair a selection *would* settle on without recording it as
    /// the container's rest state.
    fn note_snap_event_pair(&self, c: &ScrollContainer, kind: SnapEventKind, pair: SnapTargets) {
        let slot = match kind {
            SnapEventKind::Changing => &c.snap_changing_sent,
            SnapEventKind::Change => &c.snap_change_sent,
        };
        if slot.get() != Some(pair) {
            slot.set(Some(pair));
            self.snap_events.borrow_mut().push(SnapEventDelivery {
                container: c.node,
                kind,
                targets: pair,
            });
        }
    }

    /// The layout-snap update (CSS Scroll Snap 2 §5.1.4): after a
    /// re-snap the eventual pair is the settled pair, so update the
    /// `scrollsnapchanging` targets first, then the
    /// `scrollsnapchange` targets.
    fn note_layout_snap(&self, idx: usize) {
        self.note_snap_event(idx, SnapEventKind::Changing);
        self.note_snap_event(idx, SnapEventKind::Change);
    }

    /// Baseline a never-seeded container's snap-event state against
    /// its resting position (CSS Scroll Snap 2 §5.1.4: the initial
    /// layout is a layout snap). The resting selection is the
    /// candidate whose valid range covers the live offset (the
    /// `resnap_axis` stay rule), merged over whatever the resnap
    /// recorded; a container resting on a target queues
    /// `scrollsnapchanging` + `scrollsnapchange` for it, a container
    /// resting unsnapped just seeds its slots to the all-null pair so
    /// a later scroll back to that state reports nothing.
    ///
    /// Initial snap on load: a mandatory container never rests
    /// unsnapped — snapping is executed when the page loads (web.dev
    /// "scroll snapping after layout changes": the re-snap rule of
    /// CSS Scroll Snap 1 applied to the initial layout), so the
    /// install-time re-snap in
    /// [`ScrollTree::resnap_after_relayout`] seats it on the nearest
    /// candidate, and the slots here are seeded with that real pair —
    /// never all-null for a container the re-snap seated — so a later
    /// identical relayout emits nothing.
    fn initial_snap_update(&self, idx: usize) {
        let Some(c) = self.containers.get(idx) else {
            return;
        };
        if c.snap.snap_type == SnapType::None {
            c.snap_changing_sent.set(Some(SnapTargets::default()));
            c.snap_change_sent.set(Some(SnapTargets::default()));
            c.snap_events_seeded.set(true);
            return;
        }
        let st = c.snap.snap_type;
        let (px, py, _) = snap_candidates(c);
        let live = c.state.offset();
        let (mn, mx) = (c.state.min_offset(), c.state.max_offset());
        let pick = |cands: &SnapCandidates, v: f32, range: (f32, f32)| {
            cands
                .iter()
                .find(|(_, cd)| v >= cd.cover_lo - SNAP_EPS && v <= cd.cover_hi + SNAP_EPS)
                .map(|&(ai, _)| SnapRest::of(&c.snap_areas, ai, clamp_snap_pos(v, range)))
        };
        let (crx, cry) = c.snap_rest.get();
        let rx = if snap_snaps_x(st) {
            pick(&px, live.0, (mn.0, mx.0))
        } else {
            None
        };
        let ry = if snap_snaps_y(st) {
            pick(&py, live.1, (mn.1, mx.1))
        } else {
            None
        };
        c.snap_rest.set((rx.or(crx), ry.or(cry)));
        if c.snap_targets() == SnapTargets::default() {
            c.snap_changing_sent.set(Some(SnapTargets::default()));
            c.snap_change_sent.set(Some(SnapTargets::default()));
        } else {
            self.note_layout_snap(idx);
        }
        c.snap_events_seeded.set(true);
    }

    /// Carry `scrollend` bookkeeping from `old` across a relayout: the
    /// rebuilt tree starts with zeroed ticks, which would silently drop
    /// a pending `scrollend` for a gesture in flight when layout re-runs
    /// mid-scroll. The tick counter and each surviving container's
    /// last-movement stamp (matched by DOM node) are copied over;
    /// containers that went away keep tick 0 and never fire a stale
    /// `scrollend`.
    ///
    /// In-flight smooth-scroll animations ride along too (matched the
    /// same way): without this a relayout mid-animation would visibly
    /// stall the scroll. The animation's offsets stay meaningful — the
    /// shared [`ScrollState`]s survive relayout, so `last` still matches
    /// the live offset — and each tick re-clamps into the new range.
    ///
    /// The containers' snapped state ([`ScrollContainer::snap_rest`])
    /// rides along the same way, and once everything is carried the
    /// fresh tree re-snaps against its new geometry
    /// ([`ScrollTree::resnap_after_relayout`]): a relayout that moved
    /// a snap target must not leave the scroller resting between
    /// positions.
    ///
    /// Returns the matched containers the re-snap moved, each with
    /// its pre-move offset (see
    /// [`ScrollTree::resnap_after_relayout`]): the caller re-exposes
    /// them to the scroll-event fold after the re-bake.
    pub fn carry_scroll_state(&mut self, old: &ScrollTree) -> Vec<(usize, (f32, f32))> {
        self.tick.set(old.tick.get());
        self.overscroll_tick.set(old.overscroll_tick.get());
        // Snap-event deliveries queued on the old tree but not yet
        // drained ride into the new tree: a relayout must not drop an
        // event the page has not seen. (This tree's own queue is
        // empty — `build` never queues; baselines happen below.)
        self.snap_events.borrow_mut().extend(old.take_snap_events());
        let Some(old_viewport) = old.containers.first() else {
            return Vec::new();
        };
        let Some(viewport) = self.containers.first() else {
            return Vec::new();
        };
        viewport.scroll_tick.set(old_viewport.scroll_tick.get());
        viewport.snap_base.set(old_viewport.snap_base.get());
        viewport.snap_rest.set(old_viewport.snap_rest.get());
        viewport
            .overscroll_delta
            .set(old_viewport.overscroll_delta.get());
        viewport
            .overscroll_vel
            .set(old_viewport.overscroll_vel.get());
        if let Some(anim) = old_viewport.smooth.take() {
            viewport.smooth.set(Some(anim));
        }
        // Snap-event baselines carry like `snap_rest`, but only from
        // a tree that was really built: the pre-layout placeholder's
        // viewport was seeded against no snap configuration, so the
        // first real tree baselines it via `initial_snap_update`
        // below instead of inheriting the placeholder's all-null pair.
        let mut matched = vec![false; self.containers.len()];
        if old.from_build {
            viewport
                .snap_changing_sent
                .set(old_viewport.snap_changing_sent.get());
            viewport
                .snap_change_sent
                .set(old_viewport.snap_change_sent.get());
            viewport
                .snap_events_seeded
                .set(old_viewport.snap_events_seeded.get());
            matched[0] = true;
        }
        for (idx, c) in self.containers.iter().enumerate().skip(1) {
            if let Some(node) = c.node
                && let Some(old_idx) = old.index_of(node)
            {
                c.scroll_tick.set(old.containers[old_idx].scroll_tick.get());
                c.snap_base.set(old.containers[old_idx].snap_base.get());
                c.snap_rest.set(old.containers[old_idx].snap_rest.get());
                c.overscroll_delta
                    .set(old.containers[old_idx].overscroll_delta.get());
                c.overscroll_vel
                    .set(old.containers[old_idx].overscroll_vel.get());
                c.snap_changing_sent
                    .set(old.containers[old_idx].snap_changing_sent.get());
                c.snap_change_sent
                    .set(old.containers[old_idx].snap_change_sent.get());
                c.snap_events_seeded
                    .set(old.containers[old_idx].snap_events_seeded.get());
                if let Some(anim) = old.containers[old_idx].smooth.take() {
                    c.smooth.set(Some(anim));
                }
                matched[idx] = true;
            }
        }
        // `scroll-initial-target` seating (CSS Scroll Snap 2
        // §3.1.2): a container whose shared state has never been
        // scrolled — still resting at its published origin when its
        // first carry runs — starts at its initial target instead of
        // the origin. The decision is recorded per target *value*
        // on the shared state (see
        // [`ScrollState::initial_target_seating`]), so it is made on
        // every first carry even when a state the shim adoption or a
        // real scroll already moved is left exactly where it is.
        //
        // A matched container whose freshly built target differs
        // from the recorded value — the property was added, moved
        // to another element, or removed after an earlier carry —
        // re-seats, subject to the same guard in spirit as the
        // first seat: it moves only while its current offset still
        // equals the previously seated target (or its published
        // origin when no target was recorded), so a user- or
        // script-scrolled container is never yanked. Removal
        // re-seats to the origin under that guard.
        //
        // "Evaluated" is distinguished from "applied": a matched
        // re-seat is recorded only when the guard passes and the
        // move is applied. When the guard rejects, the new value
        // is NOT recorded — the recorded value stays the previous
        // one, so the next carry re-evaluates the same change and
        // seats it if the container has returned to the guard
        // position in the meantime. While the container remains
        // scrolled away, each carry merely re-evaluates and
        // rejects without moving anything: cheap, and it cannot
        // spin, because this loop is a single pass over the
        // containers and recording was never needed for its
        // termination. An unchanged value never re-seats at all.
        //
        // A previously seated container with no old match (removed
        // and reinserted across a refresh, or an old tree that
        // never contained it) cannot be guard-checked against a
        // carried position, so its changed value is recorded
        // WITHOUT moving it: the change is not silently lost even
        // if no further rebuild occurs, and the recorded value
        // becomes the guard baseline for a later change. A never
        // seated state is distinct from that case — it takes the
        // first-carry path in the `None` arm above regardless of
        // matching. Runs before the resnap so the seated position
        // is the resnap's `pre`.
        for (idx, c) in self.containers.iter().enumerate() {
            match c.state.initial_target_seating() {
                None => {
                    if c.state.offset() == c.state.min_offset()
                        && let Some(t) = c.initial_target
                    {
                        c.state.set_offset(t.0, t.1);
                        // The seated position is the container's
                        // evaluated rest: the `mark_scrolled`
                        // backstop must not read it as unevaluated
                        // movement.
                        c.snap_base.set(c.state.offset());
                    }
                    c.state.record_initial_target_seating(c.initial_target);
                }
                Some(prev) if prev != c.initial_target => {
                    if matched.get(idx).copied().unwrap_or(false) {
                        let guard = prev.unwrap_or_else(|| c.state.min_offset());
                        if c.state.offset() == guard {
                            let dest = c.initial_target.unwrap_or_else(|| c.state.min_offset());
                            c.state.set_offset(dest.0, dest.1);
                            // As on the first seat, the seated position
                            // is the container's evaluated rest.
                            c.snap_base.set(c.state.offset());
                            c.state.record_initial_target_seating(c.initial_target);
                        }
                        // Guard rejected: do not record (see the
                        // comment above); a later carry retries.
                    } else {
                        // Seated but unmatched: record the changed
                        // value without moving the container.
                        c.state.record_initial_target_seating(c.initial_target);
                    }
                }
                Some(_) => {}
            }
        }
        let resnap_moved = self.resnap_after_relayout(old, &matched);
        // Containers with no carried baseline (first install, or
        // newly appeared in this layout) get the initial layout-snap
        // update: queue the resting pair, or seed the all-null pair.
        for idx in 0..self.containers.len() {
            if !self.containers[idx].snap_events_seeded.get() {
                self.initial_snap_update(idx);
            }
        }
        resnap_moved
    }

    /// Re-snap after a relayout (CSS Scroll Snap 1: re-snap after
    /// layout changes): reconcile every container's offset against
    /// its freshly built snap areas and snap configuration, following
    /// the targets its carried snapped state names. Called from
    /// [`ScrollTree::carry_scroll_state`] on every rebuild; see
    /// [`resnap_axis`] for the per-axis rules.
    ///
    /// Moves are instant (a layout change is not a scroll gesture)
    /// and cancel any in-flight smooth scroll on the moved container:
    /// the animation's target was computed against the old geometry,
    /// and letting it run would fight the re-seated position.
    ///
    /// Containers with an in-flight smooth scroll are skipped
    /// entirely instead: their live offset is only an intermediate
    /// point of the animation, not a resting position — re-seating
    /// from it would treat the container as unsnapped (the recorded
    /// rest position is the animation's target, not the live offset)
    /// and jump to whichever candidate is nearest the intermediate
    /// point. The animation's completion is snap-evaluated against
    /// the NEW tree's geometry (see
    /// [`ScrollTree::advance_smooth`]), so the container still
    /// settles on a snap position of the new layout.
    ///
    /// Convergence: the selection is a pure function of the new
    /// areas, the new config, and the carried state. When nothing
    /// relevant changed, the live offset is still a valid snap
    /// position and nothing moves — a second rebuild with identical
    /// geometry is a no-op — and a move lands on a valid position,
    /// so even a rebuild the move itself provokes is a no-op. Resnap
    /// never launches an animation and never dirties layout by
    /// itself, so it cannot oscillate (the failure mode behind the
    /// snap-after-relayout TIMEOUTs this replaces).
    ///
    /// Returns the matched containers the re-snap moved, each with
    /// its pre-move offset. A move is a real offset change — Chrome
    /// fires `scroll` for layout-induced moves — but the re-bake that
    /// follows re-seeds the fold's baselines (the `painted` report
    /// baseline and the `paint_base` rect shift base) at the post-move
    /// offsets, hiding the move from the next fold's drift check; the
    /// `scrollend` tick stamped below would then fire bare, with no
    /// preceding `scroll`. The caller resets only the report baseline
    /// to the pre-move offsets so the fold reports the move first
    /// (`scroll` before `scrollend`) while still diffing the rects
    /// against the correct baked shift base.
    fn resnap_after_relayout(
        &self,
        old: &ScrollTree,
        matched: &[bool],
    ) -> Vec<(usize, (f32, f32))> {
        let mut moved = Vec::new();
        for (idx, c) in self.containers.iter().enumerate() {
            // In-flight smooth scroll (carried over above): skip the
            // re-seat for this container and let the animation finish
            // (see the doc comment above).
            if c.smooth.get().is_some() {
                continue;
            }
            let old_c = if idx == 0 {
                old.containers.first()
            } else {
                c.node
                    .and_then(|n| old.index_of(n))
                    .map(|oi| &old.containers[oi])
            };
            let old_rest = old_c.map_or((None, None), |oc| oc.snap_rest.get());
            // The pair the container named before the re-snap: the
            // layout-snap events below fire only when the re-snap
            // changes it.
            let old_pair = c.targets_from_rests(old_rest.0, old_rest.1);
            let live = c.state.offset();
            let (px, py, (spw, sph)) = snap_candidates(c);
            let mandatory = snap_is_mandatory(c.snap.snap_type);
            let (mn, mx) = (c.state.min_offset(), c.state.max_offset());
            let (rest_x, tx) = if snap_snaps_x(c.snap.snap_type) {
                resnap_axis(
                    &c.snap_areas,
                    &px,
                    old_rest.0,
                    live.0,
                    mandatory,
                    proximity_threshold(spw),
                    (mn.0, mx.0),
                )
            } else {
                (None, live.0)
            };
            let (rest_y, ty) = if snap_snaps_y(c.snap.snap_type) {
                resnap_axis(
                    &c.snap_areas,
                    &py,
                    old_rest.1,
                    live.1,
                    mandatory,
                    proximity_threshold(sph),
                    (mn.1, mx.1),
                )
            } else {
                (None, live.1)
            };
            c.snap_rest.set((rest_x, rest_y));
            // Layout snap (CSS Scroll Snap 2 §5.1.4): a container
            // whose baseline carried over reports the pair change the
            // re-snap produced — `scrollsnapchanging` then
            // `scrollsnapchange`, no scroll involved — but only when
            // the pair actually changed. An unconditional report would
            // fire `scrollsnapchange` for the target of a
            // still-in-flight scroll (its change belongs to the
            // scroll's end, reported by `end_tick`), or re-report a
            // pair the container already settled on. Unmatched
            // containers baseline in `initial_snap_update` instead,
            // and in-flight smooth scrolls (skipped above) report at
            // their completion evaluation.
            let pair_changed = c.snap_targets() != old_pair;
            if matched.get(idx).copied().unwrap_or(false) && pair_changed {
                self.note_layout_snap(idx);
            }
            if (tx - live.0).abs() > SNAP_EPS || (ty - live.1).abs() > SNAP_EPS {
                c.smooth.set(None);
                c.state.set_offset(tx, ty);
                // The move is evaluated: the backstop in
                // `mark_scrolled` must not re-run it as a fresh
                // scroll and launch a follow-up animation.
                c.snap_base.set(c.state.offset());
                // The re-snap moved the scroller outside any scroll
                // gesture: stamp the `scrollend` tick so the next
                // quiet tick ends the move with a `scrollend` event.
                // The headless pump folds scroll movement before
                // `refresh` runs, so without this a layout-driven
                // scroll would never complete its event lifecycle
                // (the layout-change tests wait on `scrollend`).
                // Install-time (unmatched) moves are initial layout,
                // not a scroll, and stay unstamped.
                if matched.get(idx).copied().unwrap_or(false) {
                    c.scroll_tick.set(self.tick.get());
                    // Report the move to the caller (see the doc
                    // comment): only stamped moves are returned, so
                    // the next fold fires `scroll` before this tick's
                    // `scrollend` — an unstamped install-time move must
                    // not surface as a scroll.
                    moved.push((idx, live));
                }
            }
        }
        moved
    }

    /// Innermost scroll container that is `node` itself or an ancestor
    /// scroller of it (DOM ancestry), for keyboard scrolling: the
    /// viewport when `node` is outside every nested container.
    pub fn container_for_node(&self, node: NodeId, doc: &Document) -> usize {
        let mut cursor = Some(node);
        while let Some(n) = cursor {
            if let Some(&i) = self.index.get(&n) {
                return i;
            }
            cursor = doc.parent(n);
        }
        0
    }

    /// Document scroll range for `root`: `(origin, max)`. `origin` is the
    /// nearest content edge relative to the viewport origin — negative
    /// when content overflows above/left of it (the document's
    /// `scroll_origin`, `(0, 0)` otherwise); `max` is the farthest
    /// layout-box edge minus the viewport, floored at zero per axis.
    ///
    /// Descending into nested scroll containers is skipped (their box
    /// still counts): content clipped inside a nested scroller is only
    /// reachable through that scroller, never by scrolling the document.
    pub fn document_max_scroll(
        &self,
        root: &LayoutBox,
        viewport: bowser_layout::Size,
    ) -> ((f32, f32), (f32, f32)) {
        let (mut nx, mut ny) = (0.0f32, 0.0f32);
        let (mut ex, mut ey) = (0.0f32, 0.0f32);
        // CSS Overflow 3 §2.2: `position: fixed` boxes are viewport-frame
        // and never extend document scrollable overflow — a fixed
        // `top:-50px` box's negative rect would otherwise publish a bogus
        // negative origin, and tall fixed elements would inflate the max.
        let mut stack = vec![(root, false)];
        while let Some((b, fixed_above)) = stack.pop() {
            let fixed = fixed_above
                || b.style
                    .as_ref()
                    .is_some_and(|s| position_of(s) == Position::Fixed);
            if !fixed {
                nx = nx.min(b.rect.x);
                ny = ny.min(b.rect.y);
                ex = ex.max(b.rect.x + b.rect.w);
                ey = ey.max(b.rect.y + b.rect.h);
            }
            let is_nested = b.node.is_some_and(|n| self.index.contains_key(&n));
            if !is_nested {
                stack.extend(b.children.iter().map(|c| (c, fixed)));
            }
        }
        (
            (nx.min(0.0), ny.min(0.0)),
            ((ex - viewport.w).max(0.0), (ey - viewport.h).max(0.0)),
        )
    }

    /// Scroll `node` into view through its container chain: each
    /// container from the innermost up to the viewport moves just enough
    /// to bring the revealed rect's top to its scrollport's top (the
    /// historical `scrollIntoView()` backend); the horizontal offset is
    /// kept. For the horizontal path and the other alignments use
    /// [`reveal_node_with_options`](Self::reveal_node_with_options).
    ///
    /// Always instant: fragment navigation keeps its historical
    /// synchronous behavior even when CSS `scroll-behavior` is `smooth`.
    ///
    /// Returns the scrolled containers innermost-first (`None` =
    /// viewport), for `scroll` event dispatch. A node with no layout box
    /// scrolls nothing.
    pub fn reveal_node(&self, root: &LayoutBox, node: NodeId) -> Vec<Option<NodeId>> {
        self.reveal_node_with_options(
            root,
            node,
            RevealAlign::Start,
            RevealAlign::Keep,
            ScrollBehavior::Instant,
        )
    }

    /// Scroll `node` into view with per-axis alignment and scroll
    /// behavior: each container from the innermost up to the viewport
    /// moves just enough to satisfy `block`/`inline` (see
    /// [`reveal_target`]).
    ///
    /// `behavior` resolves per container against that container's CSS
    /// `scroll-behavior` default ([`ScrollBehavior::resolve`]): smooth
    /// containers animate (see [`ScrollTree::start_smooth_scroll`]),
    /// instant ones jump. An animated container is not reported as
    /// scrolled here — nothing moved yet; the animation's ticks report
    /// the movement through the normal scroll-event funnel.
    ///
    /// The revealed rect is the node's border box expanded by its
    /// `scroll-margin` (see [`scroll_margins`]).
    ///
    /// Returns the scrolled containers innermost-first (`None` =
    /// viewport), for `scroll` event dispatch. A node with no layout box
    /// scrolls nothing.
    pub fn reveal_node_with_options(
        &self,
        root: &LayoutBox,
        node: NodeId,
        block: RevealAlign,
        inline: RevealAlign,
        behavior: ScrollBehavior,
    ) -> Vec<Option<NodeId>> {
        // Viewport-unit bases for `scroll-margin` resolution: the
        // viewport scroller's scrollport is the viewport itself.
        let (vw, vh) = self
            .containers
            .first()
            .map(|c| (c.scrollport.w, c.scrollport.h))
            .unwrap_or((0.0, 0.0));
        // `rem` in `scroll-margin` resolves against the root element's
        // computed font size; a style-less root falls back to the CSS
        // initial value (`medium` = 16px).
        let root_font_size = root
            .style
            .as_ref()
            .map(|s| px_font_size(s, 16.0))
            .unwrap_or(16.0);
        let Some((border, style, chain)) = reveal_chain(self, root, node) else {
            return Vec::new();
        };
        let (mt, mr, mb, ml) = scroll_margins(style, root_font_size, vw, vh);
        let mut reveal = Rect {
            x: border.x - ml,
            y: border.y - mt,
            w: border.w + ml + mr,
            h: border.h + mt + mb,
        };
        // The revealed rect's frame: viewport coordinates when the
        // innermost container (or, with no nested container, the node's
        // own box) is viewport-anchored — layout reports fixed-subtree
        // rects in the viewport frame — page coordinates otherwise.
        // Crossing between frames translates by the LIVE viewport
        // offset (finding #1); a build-time translation would go stale
        // on the next document scroll.
        let mut reveal_viewport_frame = match chain.first() {
            Some(&inner) => self
                .containers
                .get(inner)
                .is_some_and(|c| c.viewport_anchored),
            None => style.is_some_and(|s| position_of(s) == Position::Fixed),
        };
        let mut scrolled = Vec::new();
        for &idx in chain.iter().chain(std::iter::once(&0)) {
            if idx >= self.containers.len() {
                break;
            }
            let c = &self.containers[idx];
            if c.viewport_anchored != reveal_viewport_frame {
                let (vox, voy) = self.containers[0].state.offset();
                // Page -> viewport subtracts the viewport offset;
                // viewport -> page adds it.
                let (dx, dy) = if c.viewport_anchored {
                    (-vox, -voy)
                } else {
                    (vox, voy)
                };
                reveal.x += dx;
                reveal.y += dy;
            }
            let (ox, oy) = c.state.offset();
            // The snapport: the scrollport inset by `scroll-padding`
            // (finding #9) — the standard fixed-header offset technique
            // is `scroll-padding-top` on the scroller, and reveal must
            // align to the padded viewport, not the raw scrollport.
            let (pad_t, pad_r, pad_b, pad_l) = c.snap.padding;
            let (sp_x, sp_y) = (c.scrollport.x + pad_l, c.scrollport.y + pad_t);
            let (sp_w, sp_h) = (
                (c.scrollport.w - pad_l - pad_r).max(0.0),
                (c.scrollport.h - pad_t - pad_b).max(0.0),
            );
            // Degenerate-safe clamp via [`ScrollState::set_offset`]
            // (same as `scroll_chain` above): a degenerate `origin >
            // max` publish collapses the range instead of panicking
            // `f32::clamp` the way a raw `.clamp(origin, max)` would.
            let (tx, ty) = (
                reveal_target(inline, reveal.x, reveal.w, sp_x, sp_w, ox),
                reveal_target(block, reveal.y, reveal.h, sp_y, sp_h, oy),
            );
            if behavior.resolve(c.css_behavior).is_smooth() {
                // Animated: the offset moves on `advance_smooth` ticks,
                // which report the movement themselves.
                self.start_smooth_scroll(idx, tx, ty);
            } else {
                c.state.set_offset(tx, ty);
                let (nx, ny) = c.state.offset();
                if nx != ox || ny != oy {
                    scrolled.push(c.node);
                    // Scroll snap (CSS Scroll Snap 1): a reveal is an
                    // absolute scroll operation on every container it
                    // moves — including outer containers the reveal
                    // only shifted as a side effect of chaining — so
                    // each one settles against its own snap areas, not
                    // just the innermost scroller's. The evaluation is
                    // non-relative: `scroll-snap-stop: always`
                    // degrades to `normal`, as for other absolute
                    // scrolls. Evaluating here (rather than leaving it
                    // to the caller's `mark_scrolled` funnel) keeps
                    // the guarantee inside the reveal itself: the
                    // funnel only sees the containers this returns,
                    // and the backstop would skip a container whose
                    // offset happens to equal its stale `snap_base`.
                    // After this call `live == snap_base`, so the
                    // funnel's backstop does not re-evaluate.
                    self.eval_snap(idx, (ox, oy), (nx, ny), false);
                }
            }
            reveal = c.scrollport;
            reveal_viewport_frame = c.viewport_anchored;
        }
        scrolled
    }

    /// Start (or replace: a new scroll command cancels the in-flight
    /// animation) a smooth scroll of container `idx` toward `(x, y)`,
    /// clamped into the container's `[origin, max]` with the same
    /// degenerate-safe clamp [`ScrollState::set_offset`] uses. The
    /// offset does not move here; [`ScrollTree::advance_smooth`]
    /// interpolates it on later ticks.
    pub fn start_smooth_scroll(&self, idx: usize, x: f32, y: f32) {
        let Some(c) = self.containers.get(idx) else {
            return;
        };
        let (mnx, mny) = c.state.min_offset();
        let (mxx, mxy) = c.state.max_offset();
        // Degenerate-safe clamp, mirroring `ScrollState::set_offset`: a
        // degenerate `min > max` publish collapses onto the max, and
        // `NaN` becomes the effective minimum.
        let clamp = |v: f32, mn: f32, mx: f32| {
            let lo = mn.min(mx);
            if v.is_nan() { lo } else { v.clamp(lo, mx) }
        };
        let from = c.state.offset();
        let to = (clamp(x, mnx, mxx), clamp(y, mny, mxy));
        // No-op targets must not kill a real in-flight animation or wake
        // the event loop for 300ms (finding #6).
        if to == from {
            return;
        }
        c.smooth.set(Some(SmoothScroll {
            from,
            to,
            last: from,
            start: Instant::now(),
        }));
    }

    /// Advance every in-flight smooth scroll to wall-clock `now`,
    /// interpolating the offsets with [`ease_in_out_cubic`] over
    /// [`SMOOTH_SCROLL_DURATION`]. An animation whose container moved
    /// externally since the last tick is cancelled instead of advanced
    /// (interruption rule: the newer scroll command wins); a finished
    /// animation lands exactly on its target and is dropped.
    ///
    /// Every completed or interrupted animation is snap re-evaluated
    /// afterwards ([`ScrollTree::eval_snap`]): completions as relative
    /// scrolls, interruptions as non-relative ones (the interrupter was
    /// an absolute write, or `scroll_chain` which already evaluated
    /// relatively).
    ///
    /// Returns the containers whose offset moved this tick,
    /// innermost-first (`None` = viewport), for `scroll` event dispatch.
    /// `scrollend` follows from the normal quiet-tick machinery: the
    /// last movement stamps the container scrolling, and the first tick
    /// with no movement ends it.
    pub fn advance_smooth(&self, now: Instant) -> Vec<Option<NodeId>> {
        let mut moved = Vec::new();
        // Containers whose animation finished or was interrupted this
        // tick, with the offset the completed/interrupted scroll started
        // from: snap evaluates after the movement loop so a follow-up
        // snap smooth never disturbs an in-flight animation's bookkeeping.
        let mut snapped: Vec<(usize, (f32, f32), bool)> = Vec::new();
        for (idx, c) in self.containers.iter().enumerate() {
            let mut anim = match c.smooth.take() {
                Some(a) => a,
                None => continue,
            };
            // Interruption: some other scroll command moved the container
            // since the last tick — it wins, the animation is dropped,
            // and the interrupting scroll gets its own snap evaluation
            // (non-relative: the interrupter was an absolute write, or
            // `scroll_chain` which already evaluated relatively —
            // re-evaluating here with `relative = true` would double-apply
            // `scroll-snap-stop: always`).
            if c.state.offset() != anim.last {
                snapped.push((idx, anim.last, false));
                continue;
            }
            let elapsed = now.saturating_duration_since(anim.start).as_secs_f32();
            let t = (elapsed / SMOOTH_SCROLL_DURATION.as_secs_f32()).min(1.0);
            let e = ease_in_out_cubic(t);
            let (fx, fy) = anim.from;
            let (tx, ty) = anim.to;
            let before = c.state.offset();
            if t >= 1.0 {
                // Land exactly on target (finding #7): the eased
                // interpolation `fx + (tx - fx) * e` with e = 1.0 is not
                // `tx` in f32, so a finished animation must write the
                // target directly.
                c.state.set_offset(tx, ty);
            } else {
                c.state.set_offset(fx + (tx - fx) * e, fy + (ty - fy) * e);
            }
            let after = c.state.offset();
            if after != before {
                moved.push(c.node);
            }
            if t < 1.0 {
                anim.last = after;
                c.smooth.set(Some(anim));
            } else {
                // Completed: the animation is dropped (the `take()` above
                // already removed it). A smooth scroll is a relative
                // scroll operation for `scroll-snap-stop: always`
                // purposes — the traversed interval is [from, to].
                snapped.push((idx, anim.from, true));
            }
            // `t >= 1.0`: the animation completed exactly on target and
            // is dropped (the `take()` above already removed it).
        }
        // Containers are stored parents-before-children; reverse for the
        // innermost-first order the `scroll` dispatches use.
        moved.reverse();
        for (idx, pre, relative) in snapped {
            let natural = self.containers[idx].state.offset();
            self.eval_snap(idx, pre, natural, relative);
        }
        moved
    }

    /// Instantly complete every in-flight smooth scroll: land each
    /// container exactly on its animation target, drop the animation,
    /// and run the completion snap re-evaluation — the headless
    /// equivalent of ticking [`ScrollTree::advance_smooth`] to
    /// completion. The headless pump has no animation ticks, so a
    /// smooth scroll the scroll-snap backstop launched (a script scroll
    /// of a snap container that did not land on a snap point) would
    /// otherwise sit frozen mid-flight forever: the stuck `Some` also
    /// suppresses every later backstop and resnap for that container
    /// (both skip in-flight smooth scrolls). Consistent with the
    /// documented smooth→instant headless degrade.
    ///
    /// Returns the containers whose offset moved, innermost-first
    /// (`None` = viewport), for `scroll` event dispatch. A completion
    /// evaluation that re-targets a moved snap point can launch a
    /// follow-up animation; the caller loops until this returns empty.
    pub fn complete_smooth_scrolls(&self) -> Vec<Option<NodeId>> {
        // A far-future timestamp drives every in-flight animation to
        // t = 1: `advance_smooth` lands them exactly on target, drops
        // them, and runs each completion's snap re-evaluation
        // (relative, like the windowed completion path), including the
        // interruption rule for containers something else moved.
        self.advance_smooth(Instant::now() + SMOOTH_SCROLL_DURATION * 2)
    }

    /// Whether any container has a smooth-scroll animation in flight.
    /// The window worker uses this to keep the event loop turning (each
    /// turn advances the animation) until every animation completes.
    pub fn has_smooth_in_flight(&self) -> bool {
        self.containers.iter().any(|c| c.smooth.get().is_some())
    }

    /// The container's transient overscroll (rubber-band) shift in CSS
    /// px: the paint-time translation the paint worker folds into
    /// `bake_paint_tree`/`sync_paint_offsets` on top of the
    /// scroll-offset shift for nested containers — never into layout.
    /// The viewport container's own shift (index 0) instead rides the
    /// per-frame GPU uniform, which moves the whole page including
    /// viewport-anchored boxes. `(0.0, 0.0)` for an unknown index. The
    /// layout offset, `scroll`/`scrollend` events, and `scroll_tick`
    /// bookkeeping are untouched by this shift.
    pub fn overscroll_shift(&self, container_idx: usize) -> (f32, f32) {
        self.containers
            .get(container_idx)
            .map(|c| c.overscroll_delta.get())
            .unwrap_or((0.0, 0.0))
    }

    /// The overscroll edge glow for one container (phase 2 of the
    /// overscroll affordance): the physical edge to glow and its
    /// strength in `[0, 1]`.
    ///
    /// Strength is proportional to the transient rubber-band shift
    /// magnitude, saturating at [`OVERSCROLL_MAX_SHIFT`]; the edge
    /// follows the shift's sign — the same press [`EdgeHit`] recorded
    /// (a positive shift pressed the max edge, a negative shift the
    /// origin edge), so a bottom/right press glows the bottom/right
    /// edge and a top/left press the top/left edge. When both axes are
    /// stretched, the larger magnitude wins.
    ///
    /// Returns `None` when the shift is zero, the index is unknown, or
    /// the pressed axis's `overscroll-behavior` is `none` — which
    /// suppresses the affordance on that axis (`contain` keeps it).
    /// Axes are independent: a `none` x-axis still glows on y.
    pub fn overscroll_glow(&self, container_idx: usize) -> Option<(Edge4, f32)> {
        let c = self.containers.get(container_idx)?;
        let (dx, dy) = c.overscroll_delta.get();
        let (obx, oby) = c.overscroll;
        // Per-axis (edge, magnitude), skipping `none`-suppressed axes.
        let x = (dx.is_finite() && dx != 0.0 && obx != OverscrollBehavior::None)
            .then(|| (if dx > 0.0 { Edge4::Right } else { Edge4::Left }, dx.abs()));
        let y = (dy.is_finite() && dy != 0.0 && oby != OverscrollBehavior::None)
            .then(|| (if dy > 0.0 { Edge4::Bottom } else { Edge4::Top }, dy.abs()));
        let (edge, mag) = match (x, y) {
            (Some((ex, mx)), Some((ey, my))) => {
                if mx >= my {
                    (ex, mx)
                } else {
                    (ey, my)
                }
            }
            (Some(v), None) | (None, Some(v)) => v,
            (None, None) => return None,
        };
        Some((edge, (mag / OVERSCROLL_MAX_SHIFT).clamp(0.0, 1.0)))
    }

    /// The container's scrollport rect in display-list (page) coordinates.
    ///
    /// Computed from live scroll state — the stored scrollport minus live
    /// ancestor offsets (the same frame
    /// [`ScrollTree::scrollbar_frame_rect`] paints the bars in) — rather
    /// than the baked paint tree: under the transform fast path the
    /// paint tree's box rects go stale by the accumulated fast-path
    /// deltas, while this stays current. It also matches the bars for
    /// `viewport_anchored` and CSS-transformed containers, where the
    /// paint-tree rect is in the wrong frame.
    pub fn painted_scrollport(&self, container_idx: usize) -> Option<Rect> {
        self.containers.get(container_idx)?;
        Some(self.scrollbar_frame_rect(container_idx))
    }

    /// Advance every container's overscroll spring toward zero by the
    /// wall-clock `dt` since the last call (clamped to
    /// [`OVERSCROLL_MAX_DT`]), using the exact closed-form solution of
    /// the critically-damped oscillator (`k = ω²`, `c = 2ω`) —
    /// unconditionally stable for any step size. Containers whose shift
    /// and velocity fall below the rest epsilons snap to exactly zero.
    ///
    /// Returns the containers whose shift changed this tick,
    /// innermost-first (`None` = viewport): the caller sets
    /// `needs_redraw` when this is non-empty and keeps the event loop
    /// turning while [`ScrollTree::has_overscroll_in_flight`] holds.
    /// The spring runs every event-loop turn, including while a gesture
    /// is still pushing (the band then holds a driven equilibrium
    /// instead of growing without bound); it never touches offsets or
    /// `scroll`/`scrollend` bookkeeping.
    pub fn advance_overscroll(&self, now: Instant) -> Vec<Option<NodeId>> {
        let dt = self
            .overscroll_tick
            .get()
            .map(|t| now.saturating_duration_since(t).as_secs_f32())
            .unwrap_or(0.0)
            .clamp(0.0, OVERSCROLL_MAX_DT);
        self.overscroll_tick.set(Some(now));
        let mut changed = Vec::new();
        if dt <= 0.0 {
            return changed;
        }
        for cont in &self.containers {
            let (x0, y0) = cont.overscroll_delta.get();
            if x0 == 0.0 && y0 == 0.0 {
                continue;
            }
            let (vx0, vy0) = cont.overscroll_vel.get();
            let (mut x, mut vx) = overscroll_spring_step(x0, vx0, dt);
            let (mut y, mut vy) = overscroll_spring_step(y0, vy0, dt);
            if x.abs() < OVERSCROLL_POS_EPS && vx.abs() < OVERSCROLL_VEL_EPS {
                x = 0.0;
                vx = 0.0;
            }
            if y.abs() < OVERSCROLL_POS_EPS && vy.abs() < OVERSCROLL_VEL_EPS {
                y = 0.0;
                vy = 0.0;
            }
            if (x, y) != cont.overscroll_delta.get() {
                cont.overscroll_delta.set((x, y));
                cont.overscroll_vel.set((vx, vy));
                changed.push(cont.node);
            }
        }
        // Containers are stored parents-before-children; reverse for
        // the innermost-first order the scroll dispatches use.
        changed.reverse();
        changed
    }

    /// Whether any container's transient overscroll shift is nonzero:
    /// the window worker folds this into its event-loop keep-alive
    /// (alongside [`ScrollTree::has_smooth_in_flight`]) so the spring
    /// settles even with no further input.
    pub fn has_overscroll_in_flight(&self) -> bool {
        self.containers
            .iter()
            .any(|c| c.overscroll_delta.get() != (0.0, 0.0))
    }

    /// Whether any *nested* (non-viewport) container's transient
    /// overscroll shift is nonzero. The window worker uses this to
    /// decide paint dirt while the spring settles: nested overscroll
    /// folds into the paint tree, while the viewport's own shift rides
    /// the per-frame GPU uniform ([`ScrollTree::overscroll_shift`])
    /// and needs no pipeline dirt — only the redraw the keep-alive
    /// already schedules.
    pub fn has_nested_overscroll_in_flight(&self) -> bool {
        self.containers
            .iter()
            .skip(1)
            .any(|c| c.overscroll_delta.get() != (0.0, 0.0))
    }

    /// Clone `root` with every nested container's current offset baked
    /// in: each container's descendants that scroll with it shift by
    /// `-offset` — `position: fixed` subtrees and `position: absolute`
    /// boxes whose containing block is outside the container are left
    /// alone (see [`child_shift`]) — while the container's own rect (the
    /// scrollport clip) stays put. The display list is built from this
    /// copy, so nested scrolling moves content without a relayout.
    ///
    /// With no nested scrollers the paint tree borrows `root` instead of
    /// cloning it ([`Cow::Borrowed`]): `sync_paint_offsets` is a no-op
    /// there, so the clone was pure waste on every paint of a page
    /// without nested scroll containers.
    ///
    /// Single pass: the walk clones each box exactly once, applying its
    /// accumulated shift while cloning. Container boxes resolve through
    /// the tree's node index (built once at discovery) instead of a
    /// per-bake `NodeId -> child-path` table plus one subtree walk per
    /// container, so there is no second tree walk and no per-bake index
    /// allocation (each box is still cloned once, with its own
    /// `style`/`text`/`fragments` allocations).
    pub fn bake_paint_tree<'a>(&self, root: &'a LayoutBox) -> Cow<'a, LayoutBox> {
        // Reset the transform fast-path state: the walk below recomputes
        // eligibility and clip rects from scratch. The wrap base tracks
        // what the rects hold, so it resets to the live offsets here
        // like `painted` does on container entry below.
        for (i, c) in self.containers.iter().enumerate() {
            c.transform_fast.set(false);
            c.wrap_item.set(None);
            c.oof_wraps.borrow_mut().clear();
            c.oof_wrap_items.borrow_mut().clear();
            c.sticky_wrap_items.borrow_mut().clear();
            c.clip_rect.set(None);
            c.wrap_base.set(c.state.offset());
            c.wrap_base_os.set(c.overscroll_delta.get());
            c.paint_base.set(c.state.offset());
            c.paint_base_os.set(c.overscroll_delta.get());
            if i == 0 {
                // The viewport has no paint-tree box entry, so the walk
                // below never seeds its report baseline: seed it here,
                // mirroring the per-container seeding at box entry.
                // `rebuild_after_layout` re-arms it to the pre-move
                // offset afterwards for resnap moves, so the next fold
                // still reports them.
                c.painted.set(c.state.offset());
                c.painted_os.set(c.overscroll_delta.get());
            }
        }
        // No nested scrollers: the paint tree is the layout tree.
        if self.containers.len() <= 1 {
            return Cow::Borrowed(root);
        }
        for c in &self.containers {
            c.transform_fast.set(true);
        }
        // First-match per container node, mirroring the old path index:
        // a repeated node shifts only its first box's subtree.
        let mut entered = vec![false; self.containers.len()];
        // Iterative post-order (the tree can be arbitrarily deep): each
        // `Visit` computes its children's shifts up front — container
        // offsets, `fixed`/outside-`absolute` skips — and the matching
        // `Assemble` pops the finished children and clones the box with
        // its own shift applied.
        //
        // The trailing `Visit` fields serve the transform fast-path
        // eligibility walk: the innermost strict scroll container
        // ancestor's index (`None` above all containers); the number of
        // innermost strict scroller ancestors whose offset terms the
        // box's shift keeps (see below); and the nearest positioned
        // ancestor box with its baked border-box rect (mirroring
        // paint's out-of-flow containing-block tracking, with no
        // scroller resets).
        //
        // Shift-keep count: a box's shift holds a scroller's `-offset`
        // term exactly when no `child_shift` reset arm fired between
        // that scroller's entry and the box — terms are only added at
        // container entry and only ever cleared all at once — so the
        // kept set is always the innermost `sk` strict scroller
        // ancestors.
        enum Work<'a> {
            Visit(
                &'a LayoutBox,
                f32,
                f32,
                bool,
                bool,
                Option<usize>,
                usize,
                Option<&'a LayoutBox>,
            ),
            Assemble(&'a LayoutBox, f32, f32, usize),
        }
        let mut stack = vec![Work::Visit(root, 0.0, 0.0, false, false, None, 0, None)];
        // Finished paint boxes; `Assemble(_, _, _, n)` pops the last `n`
        // as that box's children, in order.
        let mut out: Vec<LayoutBox> = Vec::new();
        while let Some(work) = stack.pop() {
            match work {
                Work::Visit(box_, dx, dy, cb_above, fixed_cb_inside, sci, sk, npa) => {
                    // Transform fast-path qualification for out-of-flow
                    // boxes: only the scrollers whose offset terms this
                    // box's baked shift actually keeps can be affected by
                    // it — the innermost `sk` strict scroller ancestors.
                    // A box that shifts with no scroller
                    // (viewport-anchored `position: fixed`, absolute
                    // against an outside containing block) needs nothing;
                    // an absolute box with `auto` z-index drains inside
                    // its shifting scrollers' clip ranges, covered by
                    // their own wraps. The rest are recorded per shifting
                    // scroller so the wrap pass can give the out-of-range
                    // ones their own transform pairs (see `OofWrap`).
                    if let Some(inner) = sci
                        && sk > 0
                    {
                        let pos = box_
                            .style
                            .as_deref()
                            .map(position_of)
                            .unwrap_or(Position::Static);
                        let oof = pos == Position::Absolute || pos == Position::Fixed;
                        // Certain in-range: the drain anchors at/below
                        // the containing block, and the containing block
                        // sits at/below every shifting scroller (a
                        // positioned box stands between the scroller's
                        // box and this box, else the shift would have
                        // reset) — so the items land inside each
                        // scroller's clip range, under its own wrap.
                        let certain_in_range = pos == Position::Absolute && box_.z_index.is_none();
                        if oof && !certain_in_range {
                            // `position: fixed` has no containing block;
                            // its drain never leaves a wrapper pair. The
                            // wrap pass resolves the nodes against the
                            // paint tree it built the list from.
                            let cb_node = if pos == Position::Absolute {
                                npa.and_then(|b| b.node)
                            } else {
                                None
                            };
                            let mut c = inner;
                            let mut kept = 0;
                            while c != 0 && kept < sk {
                                self.containers[c].oof_wraps.borrow_mut().push(OofWrap {
                                    node: box_.node,
                                    cb_node,
                                });
                                kept += 1;
                                c = self.parents[c];
                            }
                        }
                    }
                    // `(dx, dy)` is this box's own shift, already
                    // skip-adjusted by its parent; `(cdx, cdy)` is what
                    // its children accumulate.
                    //
                    // Arithmetic note: the old bake applied each
                    // container's shift in a separate `+=` pass over the
                    // subtree; this walk accumulates the shifts and adds
                    // once. The two roundings can differ by 1 ulp in f32
                    // for nested containers — sub-pixel, no visual
                    // impact. Intentional.
                    let (mut cdx, mut cdy) = (dx, dy);
                    let positioned = box_
                        .style
                        .as_ref()
                        .is_some_and(|s| position_of(s) != Position::Static);
                    let mut child_cb = positioned || cb_above;
                    // The fixed-CB flag restarts at each scroller: a
                    // fixed-CB trigger on the scroller itself is not a
                    // fixed containing block inside it (mirroring
                    // `scrollable_extent`), so only a trigger strictly
                    // below the scroller starts the flag for the boxes
                    // beneath it.
                    let mut child_fixed_cb = fixed_cb_inside;
                    // Strict container ancestor for the eligibility walk:
                    // the container's own box keeps the outer context;
                    // its children see this container.
                    let mut child_sci = sci;
                    // Shift-keep bump for children when this box is a
                    // scroll container entered for the first time: the
                    // container's own box is never shifted by its own
                    // offset, but its children accumulate the term.
                    let mut child_sk_bump = 0;
                    if let Some(node) = box_.node
                        && let Some(&ci) = self.index.get(&node)
                        && !entered[ci]
                    {
                        entered[ci] = true;
                        child_sk_bump = 1;
                        let (ox, oy) = self.containers[ci].state.offset();
                        // Record what the paint tree now holds so a
                        // later `sync_paint_offsets` diffs against the
                        // baked offsets instead of shifting by the full
                        // offset a second time.
                        self.containers[ci].painted.set((ox, oy));
                        self.containers[ci].paint_base.set((ox, oy));
                        cdx -= ox;
                        cdy -= oy;
                        // The transient overscroll (rubber-band) shift
                        // paints on top of the scroll-offset shift, in
                        // the push direction: the content follows the
                        // edge press, then the spring settles it back to
                        // zero.
                        let (osx, osy) = self.containers[ci].overscroll_delta.get();
                        self.containers[ci].painted_os.set((osx, osy));
                        self.containers[ci].paint_base_os.set((osx, osy));
                        cdx += osx;
                        cdy += osy;
                        // The baked border-box rect: the display-list
                        // builder copies it into the container's overflow
                        // clip, so the wrap pass can locate the content
                        // range. `(dx, dy)` is this box's own shift (the
                        // container is never shifted by its own offset).
                        self.containers[ci].clip_rect.set(Some(Rect {
                            x: box_.rect.x + dx,
                            y: box_.rect.y + dy,
                            w: box_.rect.w,
                            h: box_.rect.h,
                        }));
                        // The `cb_inside` seed, as the old subtree walk
                        // used: the container itself counts when it is
                        // positioned.
                        child_cb = positioned;
                        child_fixed_cb = false;
                        child_sci = Some(ci);
                    }
                    stack.push(Work::Assemble(box_, dx, dy, box_.children.len()));
                    // Nearest positioned ancestor for children (no
                    // scroller resets, mirroring paint's npa): this box
                    // when positioned, else the inherited one.
                    let child_npa = if positioned { Some(box_) } else { npa };
                    for child in box_.children.iter().rev() {
                        let (kdx, kdy, kcb, kfcb) =
                            child_shift(child, cdx, cdy, child_cb, child_fixed_cb);
                        // Mirror `child_shift`'s reset arms exactly: a
                        // fixed box with no fixed-CB trigger inside, or
                        // an absolute box with no positioned box above
                        // (both relative to the innermost scroller),
                        // drops the whole accumulated shift — and with it
                        // every scroller term.
                        let child_pos = child
                            .style
                            .as_deref()
                            .map(position_of)
                            .unwrap_or(Position::Static);
                        let keeps_shift = !((child_pos == Position::Fixed && !child_fixed_cb)
                            || (child_pos == Position::Absolute && !child_cb));
                        let ksk = if keeps_shift { sk + child_sk_bump } else { 0 };
                        stack.push(Work::Visit(
                            child, kdx, kdy, kcb, kfcb, child_sci, ksk, child_npa,
                        ));
                    }
                }
                Work::Assemble(box_, dx, dy, n) => {
                    let children = out.split_off(out.len() - n);
                    let mut rect = box_.rect;
                    rect.x += dx;
                    rect.y += dy;
                    out.push(LayoutBox {
                        rect,
                        children,
                        box_type: box_.box_type,
                        style: box_.style.clone(),
                        text: box_.text.clone(),
                        node: box_.node,
                        fragments: shift_fragments(&box_.fragments, dx, dy),
                        z_index: box_.z_index,
                        // Boxed by layout (see `LayoutBox::style`): clone
                        // the `Box`, not the array.
                        collapsed_borders: box_.collapsed_borders.clone(),
                        phantom_w: box_.phantom_w,
                        phantom_h: box_.phantom_h,
                    });
                }
            }
        }
        Cow::Owned(out.pop().expect("bake assembles the root box"))
    }

    /// Fold scrolls that happened since the last bake (script-driven
    /// `Element.scrollTop` writes, wheel/keyboard deltas) into an
    /// existing paint tree: for each nested container whose live offset
    /// differs from the baked one, shift its painted subtree by the
    /// difference. Returns the moved containers innermost-first
    /// (`Some(node)`; the viewport is never included — it is not baked
    /// into the paint tree), so the caller can mark paint dirty and fire
    /// `scroll` events.
    ///
    /// Single pass: the walk carries each subtree's accumulated delta
    /// shift and applies it in place, so the per-sync `NodeId ->
    /// child-path` index and the per-container subtree walks go away.
    /// The steady state (nothing moved) still costs only an
    /// O(containers) offset diff, with no tree walk at all.
    pub fn sync_paint_offsets(&self, paint_root: &mut LayoutBox) -> Vec<Option<NodeId>> {
        // The viewport (container 0) has no paint-tree rects to shift —
        // its offset applies at render time — so diff it against the
        // report baseline outside the walk below. Overscroll never
        // reports as moved (the rubber band fires no `scroll`) but
        // still re-arms its baseline so the drift check settles.
        let mut moved: Vec<Option<NodeId>> = Vec::new();
        let viewport_moved = match self.containers.first() {
            Some(c0) => {
                let moved_off = c0.state.offset() != c0.painted.get();
                if moved_off {
                    c0.painted.set(c0.state.offset());
                }
                c0.painted_os.set(c0.overscroll_delta.get());
                moved_off
            }
            None => false,
        };
        if viewport_moved {
            moved.push(None);
        }
        if self.containers.len() <= 1 {
            return moved;
        }
        // Cheap diff first: when nothing moved there is nothing to fold
        // in and the walk below is skipped entirely.
        let mut drifted = false;
        for (i, c) in self.containers.iter().enumerate() {
            if i == 0 {
                continue;
            }
            if c.state.offset() != c.painted.get() || c.overscroll_delta.get() != c.painted_os.get()
            {
                drifted = true;
                break;
            }
        }
        if !drifted {
            return moved;
        }
        // First-match per container node, mirroring the bake.
        let mut entered = vec![false; self.containers.len()];
        // Innermost-first: containers are discovered parents-before-
        // children, the walk below visits them outermost-first.
        moved.clear();
        // Iterative pre-order (the tree can be arbitrarily deep).
        struct Frame<'a> {
            box_: &'a mut LayoutBox,
            dx: f32,
            dy: f32,
            cb_above: bool,
            fixed_cb_inside: bool,
        }
        let mut stack = vec![Frame {
            box_: paint_root,
            dx: 0.0,
            dy: 0.0,
            cb_above: false,
            fixed_cb_inside: false,
        }];
        while let Some(Frame {
            box_,
            dx,
            dy,
            cb_above,
            fixed_cb_inside,
        }) = stack.pop()
        {
            let (mut cdx, mut cdy) = (dx, dy);
            let positioned = box_
                .style
                .as_ref()
                .is_some_and(|s| position_of(s) != Position::Static);
            let mut child_cb = positioned || cb_above;
            // Mirror the bake: the fixed-CB flag restarts at each
            // scroller (a trigger on the scroller itself is not a fixed
            // containing block inside it).
            let mut child_fixed_cb = fixed_cb_inside;
            if let Some(node) = box_.node
                && let Some(&ci) = self.index.get(&node)
                && !entered[ci]
            {
                entered[ci] = true;
                let actual = self.containers[ci].state.offset();
                let painted = self.containers[ci].painted.get();
                // A wrapped container's paint-tree rects hold
                // `-wrap_base`; an unwrapped container's rects hold
                // `-paint_base`. `painted` is the report baseline only —
                // the resnap reset re-arms it without touching rects,
                // so it must not seed the shift. The wrap's matrix is
                // stale after any shift here, so the wrap is cleared —
                // the caller rebuilds the display list and re-wraps.
                let wrapped = self.containers[ci].wrap_item.get().is_some();
                let (base, base_os) = if wrapped {
                    (
                        self.containers[ci].wrap_base.get(),
                        self.containers[ci].wrap_base_os.get(),
                    )
                } else {
                    (
                        self.containers[ci].paint_base.get(),
                        self.containers[ci].paint_base_os.get(),
                    )
                };
                let live_os = self.containers[ci].overscroll_delta.get();
                if actual != base || live_os != base_os {
                    // The paint tree holds `-base` (+`base_os`); move it
                    // to `-actual` (+`live_os`).
                    cdx += base.0 - actual.0;
                    cdy += base.1 - actual.1;
                    cdx += live_os.0 - base_os.0;
                    cdy += live_os.1 - base_os.1;
                    self.containers[ci].painted.set(actual);
                    self.containers[ci].wrap_base.set(actual);
                    self.containers[ci].paint_base.set(actual);
                    self.containers[ci].painted_os.set(live_os);
                    self.containers[ci].wrap_base_os.set(live_os);
                    self.containers[ci].paint_base_os.set(live_os);
                    self.containers[ci].wrap_item.set(None);
                    self.containers[ci].oof_wrap_items.borrow_mut().clear();
                    self.containers[ci].sticky_wrap_items.borrow_mut().clear();
                }
                // `moved` reports true offset changes since the last
                // fold (deduped via `painted`): advance the report
                // baseline here even when the rects needed no shift —
                // after a resnap reset the rects already hold the live
                // offset, but the move must still be reported exactly
                // once. Overscroll never reports as moved — it fires no
                // `scroll` events.
                if actual != painted {
                    moved.push(self.containers[ci].node);
                    self.containers[ci].painted.set(actual);
                }
                // Keep the recorded clip rect glued to the shifted box
                // rect: an ancestor's fallback fold moves this box, and
                // the wrap pass matches the clip by that rect.
                if (dx != 0.0 || dy != 0.0)
                    && let Some(r) = self.containers[ci].clip_rect.get()
                {
                    self.containers[ci].clip_rect.set(Some(Rect {
                        x: r.x + dx,
                        y: r.y + dy,
                        w: r.w,
                        h: r.h,
                    }));
                }
                child_cb = positioned;
                child_fixed_cb = false;
            }
            if dx != 0.0 || dy != 0.0 {
                box_.rect.x += dx;
                box_.rect.y += dy;
                for f in &mut box_.fragments {
                    f.rect.x += dx;
                    f.rect.y += dy;
                    f.baseline += dy;
                }
            }
            for child in box_.children.iter_mut().rev() {
                let (kdx, kdy, kcb, kfcb) = child_shift(child, cdx, cdy, child_cb, child_fixed_cb);
                stack.push(Frame {
                    box_: child,
                    dx: kdx,
                    dy: kdy,
                    cb_above: kcb,
                    fixed_cb_inside: kfcb,
                });
            }
        }
        moved.reverse();
        // The viewport reports last: innermost-first ordering, with the
        // viewport (`None`) after every nested container.
        if viewport_moved {
            moved.push(None);
        }
        moved
    }

    /// Re-arm the viewport's fold baselines at the live offsets, for a
    /// viewport move that reported through the direct path instead of
    /// the fold ([`sync_nested_paint`](crate::sync_nested_paint)
    /// callers): the viewport has no paint-tree rects to shift, so the
    /// fold would be a baseline advance only — but without it the next
    /// pump fold would re-report the already-reported movement as a
    /// second `scroll`.
    pub(crate) fn rearm_viewport_fold_baselines(&self) {
        if let Some(c0) = self.containers.first() {
            c0.painted.set(c0.state.offset());
            c0.painted_os.set(c0.overscroll_delta.get());
        }
    }

    /// Absolute scroll translation for a wrapped container: the
    /// display-list items are baked at [`ScrollContainer::wrap_base`]
    /// (plus [`ScrollContainer::wrap_base_os`]), so translating by
    /// `wrap_base - live_offset` (+ the overscroll-delta change) shows
    /// the live offset. Content moves opposite the scroll; the
    /// rubber-band shift moves with the push direction.
    fn wrap_translation(&self, idx: usize) -> [f32; 6] {
        let c = &self.containers[idx];
        let (ox, oy) = c.state.offset();
        let (bx, by) = c.wrap_base.get();
        let (osx, osy) = c.overscroll_delta.get();
        let (bsx, bsy) = c.wrap_base_os.get();
        [
            1.0,
            0.0,
            0.0,
            1.0,
            (bx - ox) + (osx - bsx),
            (by - oy) + (osy - bsy),
        ]
    }

    /// Wrap each transform-eligible nested container's scrolling content
    /// in a display-list `PushTransform`/`PopTransform` pair, so a
    /// scroll tick can move the content by patching the transform
    /// instead of rebuilding the display list (see
    /// [`ScrollTree::try_fast_fold`]).
    ///
    /// The wrapped range is the container's overflow clip range: the
    /// builder pushes `PushClip`/`PushRoundedClip` carrying the
    /// container's baked border box around its scrolling children, so
    /// the items between that clip and its matching pop are exactly
    /// the scrolling set. Out-of-flow descendants that shift with the
    /// scroll but drain outside that range ([`ScrollContainer::oof_wraps`])
    /// get their own extra pairs around the drain's wrapper clip pair,
    /// armed with the same translation — they are positioned relative
    /// to the scroller. Containers whose clip rect is ambiguous (shared
    /// with another clipping box, or missing), or whose out-of-flow
    /// boxes cannot be covered, stay unwrapped and keep the
    /// full-rebuild path.
    ///
    /// `paint` is the paint tree the display list was built from: the
    /// recorded out-of-flow boxes resolve their current baked rects
    /// against it, since rects recorded at bake time go stale when
    /// [`ScrollTree::sync_paint_offsets`] folds offsets into the paint
    /// tree between the bake and a later re-wrap.
    ///
    /// Runs after every full display-list build, before the epilogue
    /// (glow/scrollbars) is appended: the wraps sit below the glow
    /// split, so [`ScrollTree::refresh_epilogue`] preserves them.
    pub(crate) fn wrap_scroll_transforms(&self, paint: &LayoutBox, list: &mut DisplayList) {
        for c in &self.containers {
            c.wrap_item.set(None);
            c.oof_wrap_items.borrow_mut().clear();
            c.sticky_wrap_items.borrow_mut().clear();
        }
        if self.containers.len() <= 1 {
            return;
        }
        // Eligible containers and their baked clip rects.
        let mut targets: Vec<(usize, Rect)> = Vec::new();
        for (i, c) in self.containers.iter().enumerate().skip(1) {
            if c.transform_fast.get()
                && let Some(rect) = c.clip_rect.get()
            {
                targets.push((i, rect));
            }
        }
        if targets.is_empty() {
            return;
        }
        // Resolve out-of-flow candidates against the paint tree the
        // list was built from: rects recorded at bake time go stale
        // when `sync_paint_offsets` folds offsets into the paint tree
        // between the bake and a later re-wrap.
        let mut boxes: HashMap<NodeId, &LayoutBox> = HashMap::new();
        if self
            .containers
            .iter()
            .skip(1)
            .any(|c| !c.oof_wraps.borrow().is_empty())
            || self.sticky_bands.iter().any(|b| b.scroller != 0)
        {
            let mut stack = vec![paint];
            while let Some(b) = stack.pop() {
                if let Some(n) = b.node {
                    boxes.entry(n).or_insert(b);
                }
                stack.extend(b.children.iter());
            }
        }
        /// An out-of-flow candidate with its current baked rects.
        struct Resolved {
            d_rect: Rect,
            cb_rect: Option<Rect>,
            cb_clips: bool,
        }
        // Container index -> resolved candidates. Absent when a node
        // lookup failed (the container is disqualified below).
        let mut resolved: HashMap<usize, Vec<Resolved>> = HashMap::new();
        let mut resolve_failed: HashSet<usize> = HashSet::new();
        for (ci, c) in self.containers.iter().enumerate().skip(1) {
            let wraps = c.oof_wraps.borrow();
            if wraps.is_empty() {
                continue;
            }
            let mut out = Vec::with_capacity(wraps.len());
            let mut ok = true;
            for w in wraps.iter() {
                let d_rect = match w.node.and_then(|n| boxes.get(&n)) {
                    Some(b) => b.rect,
                    None => {
                        ok = false;
                        break;
                    }
                };
                let (cb_rect, cb_clips) = match w.cb_node.and_then(|n| boxes.get(&n)) {
                    Some(b) => (
                        Some(b.rect),
                        !std::ptr::eq(*b, paint) && b.style.as_deref().is_some_and(overflow_clips),
                    ),
                    None => (None, false),
                };
                out.push(Resolved {
                    d_rect,
                    cb_rect,
                    cb_clips,
                });
            }
            if ok {
                resolved.insert(ci, out);
            } else {
                resolve_failed.insert(ci);
            }
        }
        // Pair every clip push with its pop in one stack pass. The
        // builder keeps push/pop balanced; an unmatched pop is ignored
        // (defensive — it never happens).
        let mut stack: Vec<(usize, Rect)> = Vec::new();
        let mut ranges: Vec<(usize, usize, Rect)> = Vec::new();
        for (idx, item) in list.items.iter().enumerate() {
            match item {
                DisplayItem::PushClip { rect } | DisplayItem::PushRoundedClip { rect, .. } => {
                    stack.push((idx, *rect));
                }
                DisplayItem::PopClip => {
                    if let Some((push, rect)) = stack.pop() {
                        ranges.push((push, idx, rect));
                    }
                }
                _ => {}
            }
        }
        // Assign ranges to containers, one rect-group at a time: the
        // group's ranges must nest exactly (outer-to-inner), matching
        // the group's containers (parents before children). A single
        // container may additionally own extra same-rect pairs — its
        // out-of-flow drain wrappers (see `OofWrap`) — each attributable
        // to a recorded containing block; the wrap pass gives those
        // their own transform pairs below. Anything else — a rect shared
        // with a non-container clip, sibling same-rect containers, a
        // missing clip — leaves the group unwrapped (safe: the
        // full-rebuild path).
        let mut wrapped: Vec<(usize, usize, usize)> = Vec::new();
        let mut grouped = vec![false; targets.len()];
        for (ti, &(_, rect)) in targets.iter().enumerate() {
            if grouped[ti] {
                continue;
            }
            let mut group: Vec<(usize, usize)> = Vec::new();
            for (tj, &(cj, r)) in targets.iter().enumerate() {
                if !grouped[tj] && r == rect {
                    grouped[tj] = true;
                    group.push((tj, cj));
                }
            }
            let mut rr: Vec<(usize, usize)> = ranges
                .iter()
                .filter(|(_, _, r)| *r == rect)
                .map(|(s, e, _)| (*s, *e))
                .collect();
            rr.sort_by_key(|&(s, _)| s);
            group.sort_by_key(|&(_, cj)| cj);
            if group.len() == 1 && !rr.is_empty() {
                // The first pair anchors the container's own range; the
                // extras are out-of-flow drain wrappers. (Which pair is
                // "own" vs wrapper is immaterial: both take the same
                // translation.)
                let ci = group[0].1;
                let mut ok = !resolve_failed.contains(&ci);
                if ok {
                    match resolved.get(&ci) {
                        // Extra same-rect pairs are only plausible as
                        // drain wrappers when some recorded box's
                        // containing block has this rect.
                        Some(res) => {
                            if rr.len() > 1 && !res.iter().any(|r| r.cb_rect == Some(rect)) {
                                ok = false;
                            }
                        }
                        // Extras but no recorded boxes: genuinely
                        // ambiguous.
                        None => ok = rr.len() == 1,
                    }
                }
                if ok {
                    wrapped.push((ci, rr[0].0, rr[0].1));
                } else {
                    self.containers[ci].transform_fast.set(false);
                }
            } else {
                let nested = rr.len() == group.len()
                    && rr.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 > w[1].1);
                if nested {
                    for ((_, cj), &(s, e)) in group.iter().zip(rr.iter()) {
                        wrapped.push((*cj, s, e));
                    }
                } else {
                    for (_, cj) in group {
                        self.containers[cj].transform_fast.set(false);
                    }
                }
            }
        }
        if wrapped.is_empty() {
            return;
        }
        // Extra wraps for out-of-range out-of-flow descendants: wrap
        // every drain wrapper pair belonging to this container's
        // recorded boxes — disjoint from the container's own clip range
        // — with the container's scroll translation. A box with no
        // wrapper is covered when its containing block clips (an
        // out-of-range drain always leaves a wrapper then) or when its
        // own items sit inside the clip range; anything else
        // disqualifies the container (safe: the full-rebuild path).
        // Pairs already claimed as another container's own range are
        // never taken.
        let main_pairs: HashSet<(usize, usize)> = wrapped.iter().map(|&(_, s, e)| (s, e)).collect();
        // (container, wrapper push) pairs already extra-wrapped: one
        // pair can serve two nested scrollers, each needing its own
        // (nested) transform.
        let mut claimed: HashSet<(usize, usize)> = HashSet::new();
        // (container, wrapper push, wrapper pop).
        let mut oof: Vec<(usize, usize, usize)> = Vec::new();
        for &(ci, s, e) in &wrapped {
            let res = resolved.get(&ci);
            if res.is_none_or(Vec::is_empty) {
                continue;
            }
            let res = res.expect("checked non-empty");
            // All wrapper pairs for this container's containing blocks.
            let mut wanted: Vec<Rect> = Vec::new();
            for r in res {
                if let Some(cb) = r.cb_rect
                    && !wanted.contains(&cb)
                {
                    wanted.push(cb);
                }
            }
            let mut found: Vec<(usize, usize, Rect)> = Vec::new();
            for &rect in &wanted {
                for &(p, q, r) in &ranges {
                    if r == rect
                        && (q < s || p > e)
                        && !main_pairs.contains(&(p, q))
                        && !claimed.contains(&(ci, p))
                    {
                        claimed.insert((ci, p));
                        found.push((p, q, r));
                    }
                }
            }
            // Every recorded box must be covered: by a wrapper, by a
            // clipping containing block (in range, certain), or by its
            // own items inside the clip range.
            let mut ok = !resolve_failed.contains(&ci);
            if ok {
                for r in res {
                    let covered = match r.cb_rect {
                        Some(cb_rect) => {
                            found.iter().any(|&(_, _, fr)| fr == cb_rect) || r.cb_clips
                        }
                        // `position: fixed`: no wrapper pair ever.
                        None => false,
                    } || oof_box_in_range(&list.items, r.d_rect, s, e);
                    if !covered {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                for (p, q, _) in found {
                    oof.push((ci, p, q));
                }
            } else {
                self.containers[ci].transform_fast.set(false);
            }
        }
        wrapped.retain(|&(ci, _, _)| self.containers[ci].transform_fast.get());
        if wrapped.is_empty() {
            return;
        }
        // Dedicated sticky pairs: for each sticky band of a wrapped
        // container, isolate the box's items to a clean display-list
        // range and wrap it, so the tick can fold the exact sticky
        // shift (see `sticky_shifts_at`) on top of the scroll transform.
        // A box with no clean range gets no pair; the tick then falls
        // back to re-bake rather than mis-shifting content.
        // (container, node, range start, range end).
        let mut sticky: Vec<(usize, NodeId, usize, usize)> = Vec::new();
        for &(ci, s, e) in &wrapped {
            // The constraining scroller's box: foreign signatures come
            // from its subtree minus the sticky box's.
            let container_box = match self.containers[ci].node.and_then(|n| boxes.get(&n)) {
                Some(b) => b,
                None => continue,
            };
            for b in &self.sticky_bands {
                if b.scroller != ci {
                    continue;
                }
                let lb = match boxes.get(&b.node) {
                    Some(lb) => lb,
                    None => continue,
                };
                // Node→box mismatch guard: `boxes` maps the first box
                // built for the node, which need not be the
                // sticky-positioned one the band was built from.
                if !matches!(lb.style.as_deref().map(position_of), Some(Position::Sticky)) {
                    continue;
                }
                // Subtree shapes that defeat exact isolation (nested
                // sticky, hoisted positioned descendants) keep the
                // re-bake path: no pair, so the tick falls back rather
                // than mis-shifting.
                if !sticky_subtree_isolable(lb) {
                    continue;
                }
                if let Some((a, z)) = sticky_item_range(&list.items, lb, container_box, s, e) {
                    sticky.push((ci, b.node, a, z));
                }
            }
        }
        // Sticky ranges must not overlap each other (siblings paint
        // atomically, so overlap means mis-identification).
        {
            let mut ranges: Vec<(usize, usize)> =
                sticky.iter().map(|&(_, _, a, z)| (a, z)).collect();
            ranges.sort();
            let mut ok = true;
            for w in ranges.windows(2) {
                if w[0].1 >= w[1].0 {
                    ok = false;
                    break;
                }
            }
            if !ok {
                sticky.clear();
            }
        }
        // Splice the pairs in one pass. At equal indices the
        // PopTransform sorts before the PushTransform (close the earlier
        // range before opening the next); at equal (index, kind) outer
        // containers open first and close last, so nested extra wraps
        // compose outside-in. Sticky pairs nest inside their
        // container's own wrap: at equal (index, kind) the main pair
        // opens first / closes last.
        const POP: u8 = 0;
        const PUSH: u8 = 1;
        const SUB_MAIN: u8 = 0;
        const SUB_STICKY: u8 = 1;
        const SUB_OOF: u8 = 2;
        struct Ins {
            at: usize,
            kind: u8,
            ci: usize,
            sub: u8,
            node: Option<NodeId>,
            item: DisplayItem,
        }
        let identity = || DisplayItem::PushTransform {
            matrix: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        };
        let mut insertions: Vec<Ins> = Vec::new();
        for &(ci, s, e) in &wrapped {
            insertions.push(Ins {
                at: s,
                kind: PUSH,
                ci,
                sub: SUB_MAIN,
                node: None,
                item: identity(),
            });
            insertions.push(Ins {
                at: e + 1,
                kind: POP,
                ci,
                sub: SUB_MAIN,
                node: None,
                item: DisplayItem::PopTransform,
            });
        }
        for &(ci, p, q) in &oof {
            insertions.push(Ins {
                at: p,
                kind: PUSH,
                ci,
                sub: SUB_OOF,
                node: None,
                item: identity(),
            });
            insertions.push(Ins {
                at: q + 1,
                kind: POP,
                ci,
                sub: SUB_OOF,
                node: None,
                item: DisplayItem::PopTransform,
            });
        }
        for &(ci, node, a, z) in &sticky {
            insertions.push(Ins {
                at: a,
                kind: PUSH,
                ci,
                sub: SUB_STICKY,
                node: Some(node),
                item: identity(),
            });
            insertions.push(Ins {
                at: z + 1,
                kind: POP,
                ci,
                sub: SUB_STICKY,
                node: Some(node),
                item: DisplayItem::PopTransform,
            });
        }
        insertions.sort_by(|a, b| {
            a.at.cmp(&b.at)
                .then_with(|| a.kind.cmp(&b.kind))
                .then_with(|| {
                    if a.kind == POP {
                        // Close inner ranges first: deeper containers have
                        // larger indices, and sticky pairs nest inside
                        // the main pair.
                        b.ci.cmp(&a.ci).then_with(|| b.sub.cmp(&a.sub))
                    } else {
                        // Open outer ranges first.
                        a.ci.cmp(&b.ci).then_with(|| a.sub.cmp(&b.sub))
                    }
                })
        });
        // PushTransform arming: the k-th insertion in sorted order lands
        // at `at + k` — `k` insertions shift it right past the `at`
        // original items before it.
        let mut pushes: Vec<(usize, u8, Option<NodeId>, usize, usize)> = Vec::new();
        for (k, ins) in insertions.iter().enumerate() {
            if ins.kind == PUSH {
                pushes.push((ins.ci, ins.sub, ins.node, ins.at, k));
            }
        }
        let mut out = Vec::with_capacity(list.items.len() + insertions.len());
        let mut ins = insertions.into_iter().peekable();
        for (i, item) in list.items.drain(..).enumerate() {
            while ins.peek().is_some_and(|in_| in_.at == i) {
                out.push(ins.next().expect("peeked insertion").item);
            }
            out.push(item);
        }
        for in_ in ins {
            out.push(in_.item);
        }
        list.items = out;
        // Arm each wrap with the absolute scroll translation and record
        // its item index for the tick path. The out-of-range boxes ride
        // the same translation: they are positioned relative to this
        // scroller. Sticky pairs start as the identity: at wrap time the
        // offset is the band's bake offset, so the extra sticky shift is
        // zero; the tick patches it via `sticky_shifts_at`.
        for (ci, sub, node, at, k) in pushes {
            let idx = at + k;
            let matrix = if sub == SUB_STICKY {
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            } else {
                self.wrap_translation(ci)
            };
            match list.items.get_mut(idx) {
                Some(DisplayItem::PushTransform { matrix: m }) => {
                    *m = matrix;
                    if sub == SUB_STICKY {
                        self.containers[ci]
                            .sticky_wrap_items
                            .borrow_mut()
                            .push((node.expect("sticky pair carries its node"), idx));
                    } else if sub == SUB_OOF {
                        self.containers[ci].oof_wrap_items.borrow_mut().push(idx);
                    } else {
                        self.containers[ci].wrap_item.set(Some(idx));
                    }
                }
                // Unreachable: the splice just put it there. Stay on
                // the safe path rather than recording a bad index.
                _ => self.containers[ci].transform_fast.set(false),
            }
        }
    }

    /// Fold drifted nested-container offsets/overscroll into the
    /// wrapped scroll transforms, when every drifted container is
    /// wrapped.
    ///
    /// Returns `None` when a drifted container is not transform-wrapped
    /// (ineligible, ambiguous clip, or the wrap was invalidated): the
    /// caller falls back to [`ScrollTree::sync_paint_offsets`] + a full
    /// rebuild. Otherwise patches each wrapped `PushTransform` matrix in
    /// place to the absolute scroll translation — the container's own
    /// wrap and each armed [`ScrollContainer::oof_wrap_items`] pair,
    /// which rides the same translation — folds each exactly-foldable
    /// sticky box's extra shift ([`ScrollTree::sticky_shifts_at`]) into
    /// its dedicated [`ScrollContainer::sticky_wrap_items`] pair on top
    /// of the scroll transform, advances `painted`/`painted_os` for
    /// `scroll`-event dedup, and returns the drifted containers
    /// innermost-first (`Some(node)`). Overscroll-only drift fires no
    /// `scroll` events, mirroring [`ScrollTree::sync_paint_offsets`].
    ///
    /// The paint tree is deliberately left stale: it is a build
    /// artifact, and [`ScrollTree::painted_scrollport`] no longer reads
    /// it. Validation runs before any patch, so a `None` return leaves
    /// the display list untouched.
    pub(crate) fn try_fast_fold(&self, list: &mut DisplayList) -> Option<Vec<Option<NodeId>>> {
        if self.containers.len() <= 1 {
            // Viewport-only page: no wraps to patch, but the fold's
            // baseline advance + report must still run — the windowed
            // pump relies on the fold (not the pre-first-layout
            // fallback) once a paint tree exists.
            let mut moved = Vec::new();
            if let Some(c0) = self.containers.first() {
                if c0.state.offset() != c0.painted.get() {
                    moved.push(c0.node); // None
                }
                c0.painted.set(c0.state.offset());
                c0.painted_os.set(c0.overscroll_delta.get());
            }
            return Some(moved);
        }
        let mut drifted: Vec<usize> = Vec::new();
        // Per drifted container, the exact sticky shifts to fold, when
        // every in-band sticky box is exactly foldable.
        let mut sticky_folds: Vec<(usize, Vec<StickyShift>)> = Vec::new();
        for (i, c) in self.containers.iter().enumerate() {
            if i == 0 {
                // The viewport has no paint-tree rects to shift and no
                // wrap to patch — its offset applies at render time —
                // so it skips the wrap requirements below: drift still
                // re-arms the baselines and reports, mirroring the
                // paint-tree fold.
                if c.state.offset() != c.painted.get()
                    || c.overscroll_delta.get() != c.painted_os.get()
                {
                    drifted.push(i);
                }
                continue;
            }
            if c.state.offset() != c.painted.get() || c.overscroll_delta.get() != c.painted_os.get()
            {
                if !c.transform_fast.get() {
                    return None;
                }
                let wrapped = c.wrap_item.get().is_some_and(|idx| {
                    matches!(list.items.get(idx), Some(DisplayItem::PushTransform { .. }))
                });
                if !wrapped {
                    return None;
                }
                // Every armed extra out-of-flow wrap must still be a
                // live `PushTransform`.
                for idx in c.oof_wrap_items.borrow().iter() {
                    if !matches!(
                        list.items.get(*idx),
                        Some(DisplayItem::PushTransform { .. })
                    ) {
                        return None;
                    }
                }
                // Sticky: fold the exact per-box shifts when the regime
                // is exactly foldable; a box in a constraint transition
                // needs a re-bake.
                let (ox, oy) = c.state.offset();
                let shifts = self.sticky_shifts_at(i, ox, oy)?;
                // Every shifted box needs an armed dedicated pair holding
                // a live `PushTransform`.
                for sh in &shifts {
                    let armed = c.sticky_wrap_items.borrow().iter().any(|&(n, idx)| {
                        n == sh.node
                            && matches!(
                                list.items.get(idx),
                                Some(DisplayItem::PushTransform { .. })
                            )
                    });
                    if !armed {
                        return None;
                    }
                }
                // Every armed pair must still be a live `PushTransform`,
                // even for zero-shift boxes: the patch phase resets all
                // of them to identity before folding the current shifts,
                // and a clobbered pair invalidates the wrap.
                for &(_, idx) in c.sticky_wrap_items.borrow().iter() {
                    if !matches!(list.items.get(idx), Some(DisplayItem::PushTransform { .. })) {
                        return None;
                    }
                }
                if !shifts.is_empty() {
                    sticky_folds.push((i, shifts));
                }
                drifted.push(i);
            }
        }
        let mut moved: Vec<Option<NodeId>> = Vec::with_capacity(drifted.len());
        for &i in &drifted {
            let c = &self.containers[i];
            // The viewport has no wrap to patch — its offset applies at
            // render time — so skip the (always discarded) translation
            // computation for it.
            if i > 0 {
                let matrix = self.wrap_translation(i);
                if let Some(idx) = c.wrap_item.get()
                    && let Some(DisplayItem::PushTransform { matrix: m }) = list.items.get_mut(idx)
                {
                    *m = matrix;
                }
                // The out-of-range boxes are positioned relative to this
                // scroller, so they take the same translation.
                for idx in c.oof_wrap_items.borrow().iter() {
                    if let Some(DisplayItem::PushTransform { matrix: m }) = list.items.get_mut(*idx)
                    {
                        *m = matrix;
                    }
                }
            }
            if c.state.offset() != c.painted.get() {
                moved.push(c.node);
            }
            c.painted.set(c.state.offset());
            c.painted_os.set(c.overscroll_delta.get());
        }
        // Fold the sticky shifts on top of the scroll transforms: each
        // dedicated pair nests inside its container's wrap, so the box
        // renders at scroll translation + sticky extra shift. Reset every
        // armed pair to identity first: `sticky_shifts_at` omits
        // zero-shift boxes, so without the reset a pair would keep its
        // previous tick's matrix forever (stale shift after scrolling
        // back to the bake offset).
        const IDENTITY: [f32; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        for &i in &drifted {
            let pairs = self.containers[i].sticky_wrap_items.borrow();
            for &(_, idx) in pairs.iter() {
                // Validated live above; stay defensive.
                if let Some(DisplayItem::PushTransform { matrix: m }) = list.items.get_mut(idx) {
                    *m = IDENTITY;
                }
            }
        }
        for (i, shifts) in sticky_folds {
            let c = &self.containers[i];
            let pairs = c.sticky_wrap_items.borrow();
            for sh in &shifts {
                if let Some(&(_, idx)) = pairs.iter().find(|&&(n, _)| n == sh.node)
                    && let Some(DisplayItem::PushTransform { matrix: m }) = list.items.get_mut(idx)
                {
                    *m = [1.0, 0.0, 0.0, 1.0, sh.dx, sh.dy];
                }
            }
        }
        moved.reverse();
        Some(moved)
    }
}

// ---------------------------------------------------------------------------
// Scrollbars: geometry, hit-testing, drag interaction
// ---------------------------------------------------------------------------

/// Scrollbar thickness in CSS px: the vertical track's width and the
/// horizontal track's height.
pub const SCROLLBAR_THICKNESS: f32 = 12.0;
/// Minimum thumb length in CSS px: a huge scroll range must not shrink
/// the thumb below a grabbable size.
pub const SCROLLBAR_MIN_THUMB: f32 = 24.0;

/// Which scrollbar axis of a scroll container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollbarAxis {
    Vertical,
    Horizontal,
}

/// One axis' scrollbar geometry: the track rect and the thumb rect,
/// both in page coordinates (the display-list frame — the render
/// backend subtracts the viewport offset for these items like every
/// other item).
#[derive(Debug, Clone, Copy)]
pub struct ScrollbarAxisGeom {
    pub track: Rect,
    pub thumb: Rect,
}

/// A scroll container's scrollbars in page coordinates. An axis is
/// `None` when the container shows no scrollbar on it: the axis is not
/// user-scrollable (`overflow: hidden`), has no scrollable overflow, or
/// its track would be degenerate (a scrollport smaller than the
/// scrollbar thickness yields a zero/negative track length).
#[derive(Debug, Clone, Copy, Default)]
pub struct ScrollbarGeom {
    /// Index of the scroll container in the [`ScrollTree`].
    pub container: usize,
    pub vertical: Option<ScrollbarAxisGeom>,
    pub horizontal: Option<ScrollbarAxisGeom>,
}

/// Where a pointer landed on a scrollbar, in viewport coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScrollbarHit {
    /// Index of the hit scroll container in the [`ScrollTree`].
    pub container: usize,
    /// Which axis' scrollbar was hit.
    pub axis: ScrollbarAxis,
    /// True for the thumb (starts a drag), false for the track
    /// (pages toward the click).
    pub on_thumb: bool,
}

/// An in-progress scrollbar thumb drag, captured at press time.
///
/// The track geometry is snapshotted in viewport coordinates at
/// press. Each motion event rebases the snapshot against the live
/// tree ([`ScrollTree::scrollbar_drag_rebased`]): a rigid translation
/// of the track (an ancestor scrolled, including the viewport) shifts
/// `track_start` by the same delta with `grab` unchanged, so the
/// thumb stays under the cursor instead of jumping; anything else — a
/// relayout that resized the track or repositioned it relative to its
/// frame, or a container that is gone — ends the gesture. The
/// container's own scroll never moves its track (a drag writes the
/// container's offset directly, never chaining to ancestors), so the
/// mapping cannot feed back into itself. The container is identified
/// by DOM node — `None` for the viewport — and resolved per motion
/// event, so a relayout mid-drag ends the gesture instead of
/// scrolling a recycled index.
#[derive(Debug, Clone, Copy)]
pub struct ScrollbarDrag {
    /// Scrolled container's DOM node (`None` = viewport scroller).
    pub node: Option<NodeId>,
    /// Axis being scrolled.
    pub axis: ScrollbarAxis,
    /// Track start along the drag axis, in viewport coordinates.
    /// Rebased per motion event (see [`ScrollTree::scrollbar_drag_rebased`]).
    track_start: f32,
    /// Track length along the drag axis (px).
    track_len: f32,
    /// Thumb length along the drag axis (px).
    thumb_len: f32,
    /// Cursor position minus thumb start at press: the thumb follows
    /// the cursor without jumping to it.
    grab: f32,
    /// Scroll range on the drag axis (`origin`..=`max`).
    min: f32,
    max: f32,
    /// Press-time track rect in viewport coordinates, for the
    /// mid-gesture geometry check. Never rebased: every motion
    /// compares the live rect against this snapshot.
    press_track: Rect,
    /// Press-time scrollport frame rect in viewport coordinates, for
    /// the mid-gesture geometry check. A rigid translation of this
    /// frame (an ancestor scrolled) is the only motion the gesture
    /// survives; anything else ends it.
    press_frame: Rect,
}

impl ScrollbarDrag {
    /// Target scroll offset on the drag axis for a cursor at viewport
    /// point `(x, y)`: the thumb position under the cursor, mapped
    /// linearly onto `[min, max]`. The caller writes it through
    /// [`ScrollState::set_offset`](bowser_js::ScrollState::set_offset),
    /// whose clamp keeps it inside the range.
    pub fn offset_for(&self, x: f32, y: f32) -> f32 {
        let cursor = match self.axis {
            ScrollbarAxis::Vertical => y,
            ScrollbarAxis::Horizontal => x,
        };
        let travel = self.track_len - self.thumb_len;
        if travel <= 0.0 {
            return self.min;
        }
        let frac = (cursor - self.grab - self.track_start) / travel;
        self.min + frac * (self.max - self.min)
    }
}

/// Thumb rect inside `track`: `scrollport_len` of
/// `scrollport_len + (max - min)` content is visible, so the thumb
/// takes that fraction of the track length, floored at
/// [`SCROLLBAR_MIN_THUMB`] and capped at the track length, positioned
/// by the offset's fraction of the scroll range.
fn scrollbar_thumb_rect(
    track: Rect,
    scrollport_len: f32,
    min: f32,
    max: f32,
    offset: f32,
    vertical: bool,
) -> Rect {
    let range = (max - min).max(0.0);
    let track_len = if vertical { track.h } else { track.w };
    let mut thumb_len = if range > 0.0 && scrollport_len > 0.0 {
        track_len * scrollport_len / (scrollport_len + range)
    } else {
        track_len
    };
    // `clamp` would panic when `track_len < SCROLLBAR_MIN_THUMB`
    // (min > max); the two-sided min/max is degenerate-safe.
    thumb_len = thumb_len.max(SCROLLBAR_MIN_THUMB).min(track_len);
    let travel = (track_len - thumb_len).max(0.0);
    let frac = if range > 0.0 {
        ((offset - min) / range).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let thumb_start = (if vertical { track.y } else { track.x }) + frac * travel;
    if vertical {
        Rect {
            x: track.x,
            y: thumb_start,
            w: track.w,
            h: thumb_len,
        }
    } else {
        Rect {
            x: thumb_start,
            y: track.y,
            w: thumb_len,
            h: track.h,
        }
    }
}

fn rect_contains_point(r: &Rect, x: f32, y: f32) -> bool {
    x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h
}

/// The rect the vertical scrollbar's track occupies in the display-list
/// frame: the scrollport's inline-end strip ([`SCROLLBAR_THICKNESS`]
/// wide; the left edge under `direction: rtl`), shortened by the
/// horizontal scrollbar's thickness when one is shown (the corner stays
/// blank). Shared by [`ScrollTree::scrollbar_geom`] and the
/// `scrollbar-gutter: stable` reservation
/// ([`ScrollTree::gutter_rects`]): a shown scrollbar paints exactly the
/// reserved strip.
fn vertical_track_rect(frame: &Rect, h_present: bool, rtl: bool) -> Rect {
    let t = SCROLLBAR_THICKNESS;
    Rect {
        x: if rtl { frame.x } else { frame.x + frame.w - t },
        y: frame.y,
        w: t,
        h: frame.h - if h_present { t } else { 0.0 },
    }
}

impl ScrollTree {
    /// The viewport scroller's live offset, or `(0, 0)` before the
    /// first layout (the tree is empty then).
    fn scrollbar_viewport_offset(&self) -> (f32, f32) {
        self.containers
            .first()
            .map(|c| c.state.offset())
            .unwrap_or((0.0, 0.0))
    }

    /// Container index for `node` (`None` = viewport scroller), or
    /// `None` when the container is gone — a relayout mid-gesture must
    /// end the gesture, never scroll a recycled index.
    pub fn scrollbar_container_idx(&self, node: Option<NodeId>) -> Option<usize> {
        match node {
            None => {
                if self.containers.is_empty() {
                    None
                } else {
                    Some(0)
                }
            }
            Some(n) => self.index_of(n),
        }
    }

    /// Scrollport rect of container `idx` in the display-list frame
    /// (page coordinates, matching the baked paint tree the display
    /// list builds from).
    ///
    /// A container's own offset shifts only its content, never its box
    /// ([`ScrollTree::bake_paint_tree`]), and the viewport's offset
    /// rides the GPU uniform instead of the paint tree — so the frame
    /// rect is the layout scrollport shifted by the strict scroller
    /// ancestors' offsets, viewport excluded. The viewport scroller
    /// and viewport-anchored (`position: fixed`) containers are stored
    /// at/near the viewport origin instead and translate by the live
    /// viewport offset; [`ScrollContainer::unshifted`] absolute boxes
    /// skip the ancestor subtraction, mirroring the paint walk.
    ///
    /// Layout's scrollports are pre-transform: the baked
    /// [`ScrollContainer::transform_offset`] is added so the frame —
    /// and the scrollbar geometry and hit-testing built on it — sits
    /// at the visual location.
    fn scrollbar_frame_rect(&self, idx: usize) -> Rect {
        let c = &self.containers[idx];
        let (vox, voy) = self.scrollbar_viewport_offset();
        let (tx, ty) = c.transform_offset;
        if idx == 0 {
            // The viewport scroller's stored scrollport is the
            // viewport at offset zero; the visible scrollport rides
            // the live offset.
            return Rect {
                x: vox + tx,
                y: voy + ty,
                w: c.scrollport.w,
                h: c.scrollport.h,
            };
        }
        if c.viewport_anchored {
            // Stored in viewport coordinates (finding #1); the display
            // list is in page coordinates.
            return Rect {
                x: c.scrollport.x + vox + tx,
                y: c.scrollport.y + voy + ty,
                w: c.scrollport.w,
                h: c.scrollport.h,
            };
        }
        let mut r = c.scrollport;
        if !c.unshifted {
            let mut a = self.parents[idx];
            while a != 0 {
                let (ax, ay) = self.containers[a].state.offset();
                r.x -= ax;
                r.y -= ay;
                a = self.parents[a];
            }
        }
        r.x += tx;
        r.y += ty;
        r
    }

    /// Scrollbar presence on one axis: the axis is user-scrollable
    /// (`auto`/`scroll`, not `hidden`) and has scrollable overflow
    /// (`max` above `min`).
    fn scrollbar_axis_present(&self, idx: usize, axis: ScrollbarAxis) -> bool {
        let Some(c) = self.containers.get(idx) else {
            return false;
        };
        let (user, min, max) = match axis {
            ScrollbarAxis::Vertical => (
                c.user_scroll.1,
                c.state.min_offset().1,
                c.state.max_offset().1,
            ),
            ScrollbarAxis::Horizontal => (
                c.user_scroll.0,
                c.state.min_offset().0,
                c.state.max_offset().0,
            ),
        };
        user && max > min
    }

    /// One container's scrollbar geometry in page coordinates, for
    /// painting and hit-testing. Out-of-range `idx` yields no
    /// scrollbars.
    pub fn scrollbar_geom(&self, idx: usize) -> ScrollbarGeom {
        let mut geom = ScrollbarGeom {
            container: idx,
            vertical: None,
            horizontal: None,
        };
        let Some(c) = self.containers.get(idx) else {
            return geom;
        };
        let v_present = self.scrollbar_axis_present(idx, ScrollbarAxis::Vertical);
        let h_present = self.scrollbar_axis_present(idx, ScrollbarAxis::Horizontal);
        if !v_present && !h_present {
            return geom;
        }
        let frame = self.scrollbar_frame_rect(idx);
        // Tracks sit on the scrollport's inline-end edge (vertical)
        // and block-end edge (horizontal). Under `direction: rtl` the
        // inline-end edge is the left edge, so the vertical track
        // flips sides. When both show, each shrinks by the other's
        // thickness and the corner stays blank.
        let t = SCROLLBAR_THICKNESS;
        let vtrack = vertical_track_rect(&frame, h_present, c.rtl);
        let htrack = Rect {
            x: frame.x,
            y: frame.y + frame.h - t,
            w: frame.w - if v_present { t } else { 0.0 },
            h: t,
        };
        let (min_x, min_y) = c.state.min_offset();
        let (max_x, max_y) = c.state.max_offset();
        let (off_x, off_y) = c.state.offset();
        // A scrollport smaller than the scrollbar thickness yields a
        // zero/negative track length (e.g. a 10px-tall `overflow: auto`
        // box showing both bars); that axis gets no scrollbar rather
        // than degenerate rects in the display list.
        if v_present && vtrack.h > 0.0 {
            geom.vertical = Some(ScrollbarAxisGeom {
                track: vtrack,
                thumb: scrollbar_thumb_rect(vtrack, frame.h, min_y, max_y, off_y, true),
            });
        }
        if h_present && htrack.w > 0.0 {
            geom.horizontal = Some(ScrollbarAxisGeom {
                track: htrack,
                thumb: scrollbar_thumb_rect(htrack, frame.w, min_x, max_x, off_x, false),
            });
        }
        geom
    }

    /// Every container's scrollbar geometry in page coordinates, in
    /// tree order. The painter appends these above content.
    pub fn scrollbar_geoms(&self) -> Vec<ScrollbarGeom> {
        (0..self.containers.len())
            .map(|i| self.scrollbar_geom(i))
            .collect()
    }

    /// Reserved `scrollbar-gutter: stable` strips in page coordinates
    /// (the display-list frame): the inline-end strip, plus the
    /// inline-start strip under `stable both-edges`. The strips are
    /// reported whether or not a scrollbar is currently shown — that is
    /// the point of `stable`: the reserved space, and hence the
    /// content-box width, is identical with the scrollbar present or
    /// hidden, so no layout shift occurs when scrollbars appear or
    /// disappear. Each strip is the rect the vertical scrollbar's track
    /// would occupy ([`vertical_track_rect`]); a shown scrollbar paints
    /// exactly there ([`ScrollTree::scrollbar_geom`]).
    ///
    /// Empty unless the container's gutter mode is `Stable` and its
    /// block axis is a scroll-container axis (`overflow-y` of `auto`,
    /// `scroll`, or `hidden` — CSS Overflow 4 reserves the gutter
    /// whenever overflow is not `visible` on the axis).
    ///
    /// The reserved strips: layout keeps content out of them via
    /// `bowser_layout::scrollbar_gutter_reservation` (children lay out
    /// against the narrowed width), and the `scrollbar-gutter` property
    /// is registered in `bowser_css`. Paint does not paint these strips
    /// (it paints only live scrollbars). When the horizontal scrollbar
    /// is present the vertical strip is shortened by the corner blank;
    /// that shortening is an unconfirmed internal-consistency choice,
    /// not spec-settled text — treat it as provisional.
    pub fn gutter_rects(&self, idx: usize) -> Vec<Rect> {
        let Some(c) = self.containers.get(idx) else {
            return Vec::new();
        };
        let ScrollbarGutter::Stable { both_edges } = c.gutter else {
            return Vec::new();
        };
        if !c.axes.1 {
            return Vec::new();
        }
        let frame = self.scrollbar_frame_rect(idx);
        let h_present = self.scrollbar_axis_present(idx, ScrollbarAxis::Horizontal);
        let mut strips = Vec::with_capacity(2);
        let end = vertical_track_rect(&frame, h_present, c.rtl);
        if end.h > 0.0 {
            strips.push(end);
        }
        if both_edges {
            let start = vertical_track_rect(&frame, h_present, !c.rtl);
            if start.h > 0.0 {
                strips.push(start);
            }
        }
        strips
    }

    /// Append the overscroll edge glow items for every container whose
    /// rubber-band shift is in flight (see
    /// [`ScrollTree::overscroll_glow`]): each glow as a
    /// `PushClip`/`PopClip` pair at the scrollport around one
    /// [`DisplayItem::EdgeGlow`], so the glow paints above the scrolled
    /// content but never bleeds outside the scrollport.
    ///
    /// Rects are display-list (page) coordinates: nested containers use
    /// the live scrollport rect ([`ScrollTree::painted_scrollport`]),
    /// and the viewport glows the viewport rect translated into page
    /// coords at the live scroll minus the transient shift, which the
    /// backend's per-frame uniform subtracts back out — the glow stays
    /// glued to the screen edge while the spring settles. The caller
    /// re-pushes this epilogue whenever the shift moves (see
    /// [`ScrollTree::refresh_epilogue`]), so `strength` tracks the
    /// decaying spring without a full repaint.
    pub(crate) fn append_overscroll_glow(&self, list: &mut DisplayList, viewport: Size) {
        for idx in 0..self.containers.len() {
            let Some((edge, strength)) = self.overscroll_glow(idx) else {
                continue;
            };
            if strength <= 0.0 {
                continue;
            }
            let scrollport = if idx == 0 {
                let (sx, sy) = self.containers[0].state.offset();
                let (osx, osy) = self.containers[0].overscroll_delta.get();
                Rect {
                    x: sx - osx,
                    y: sy - osy,
                    w: viewport.w,
                    h: viewport.h,
                }
            } else {
                let Some(r) = self.painted_scrollport(idx) else {
                    continue;
                };
                r
            };
            let Some(band) = edge_glow_band(scrollport, edge) else {
                continue;
            };
            list.push(DisplayItem::PushClip { rect: scrollport });
            list.push(DisplayItem::EdgeGlow {
                edge,
                rect: band,
                strength,
            });
            list.push(DisplayItem::PopClip);
        }
    }

    /// Mark the scrollbar rects baked into the display list stale:
    /// the viewport scroller moved without a paint rebuild. Set by
    /// [`crate::window::sync_nested_paint`] on every viewport move
    /// (and by the script-driven scroll fold in the event loop, which
    /// bypasses that path). Idempotent; a coincident nested move
    /// dirties paint and the full repaint clears the flag itself.
    pub(crate) fn mark_scrollbars_dirty(&mut self) {
        self.scrollbars_dirty = true;
    }

    /// Whether the scrollbar rects baked into the display list are
    /// stale (viewport moved without a rebuild).
    pub(crate) fn scrollbars_dirty(&self) -> bool {
        self.scrollbars_dirty
    }

    /// Clear the stale-scrollbar flag without re-pushing: a full
    /// repaint re-pushes the bars itself.
    pub(crate) fn clear_scrollbars_dirty(&mut self) {
        self.scrollbars_dirty = false;
    }

    /// Record where the overscroll glow items start in the display
    /// list; called by the paint path right before appending the glow
    /// (glow sits between content and scrollbars).
    pub(crate) fn set_glow_split(&mut self, split: usize) {
        self.glow_split = split;
    }

    /// Mark the baked glow items stale: a container's transient
    /// overscroll shift moved without a paint rebuild. Idempotent;
    /// the next `refresh` re-pushes the epilogue (see
    /// [`ScrollTree::refresh_epilogue`]).
    pub(crate) fn mark_glow_dirty(&mut self) {
        self.glow_dirty = true;
    }

    /// Whether the baked glow items are stale (see
    /// [`ScrollTree::mark_glow_dirty`]).
    pub(crate) fn glow_dirty(&self) -> bool {
        self.glow_dirty
    }

    /// Clear the stale-glow flag without re-pushing: a full repaint
    /// re-pushes the glow itself.
    pub(crate) fn clear_glow_dirty(&mut self) {
        self.glow_dirty = false;
    }

    /// Truncate `list` back to the recorded glow split and re-push the
    /// epilogue — the live overscroll glow items, then the live
    /// scrollbar geometry: the scroll fast path that keeps the baked
    /// rects glued to the viewport edge without rebuilding the whole
    /// display list (the viewport rides the GPU uniform and nested
    /// containers ride their patched scroll transforms, so the content
    /// items stay valid), and the overscroll fast path that re-emits
    /// the glow at the spring's current strength/rect while it settles.
    /// A missing or out-of-range split is a no-op — it never truncates
    /// real content. Always clears both dirty flags. Allocation-free:
    /// walks the containers directly instead of collecting
    /// [`ScrollTree::scrollbar_geoms`].
    pub(crate) fn refresh_epilogue(&mut self, list: &mut DisplayList, viewport: Size) {
        let split = self.glow_split;
        self.scrollbars_dirty = false;
        self.glow_dirty = false;
        if split == usize::MAX || split > list.items.len() {
            return;
        }
        list.items.truncate(split);
        self.append_overscroll_glow(list, viewport);
        for idx in 0..self.containers.len() {
            let geom = self.scrollbar_geom(idx);
            if let Some(v) = geom.vertical {
                push_scrollbar(list, v.track, v.thumb);
            }
            if let Some(h) = geom.horizontal {
                push_scrollbar(list, h.track, h.thumb);
            }
        }
    }

    /// Hit-test scrollbars at viewport point `(x, y)`: the innermost
    /// container's thumb first, then its track, so a nested
    /// container's scrollbar wins over an outer one behind it. (The
    /// DOM hit test still runs first in production — see
    /// `wheel_start_container` — so a hit-tested overlay intercepts
    /// before this is consulted.)
    pub fn scrollbar_at(&self, x: f32, y: f32) -> Option<ScrollbarHit> {
        if self.containers.is_empty() {
            return None;
        }
        let (vox, voy) = self.scrollbar_viewport_offset();
        // Containers are parents-before-children: reverse for
        // innermost-first.
        for idx in (0..self.containers.len()).rev() {
            let geom = self.scrollbar_geom(idx);
            let axes = [
                (ScrollbarAxis::Vertical, geom.vertical),
                (ScrollbarAxis::Horizontal, geom.horizontal),
            ];
            // Thumbs before tracks: each thumb sits inside its track.
            for (axis, sb) in axes.iter().filter_map(|(a, s)| s.map(|s| (*a, s))) {
                let thumb = Rect {
                    x: sb.thumb.x - vox,
                    y: sb.thumb.y - voy,
                    w: sb.thumb.w,
                    h: sb.thumb.h,
                };
                if rect_contains_point(&thumb, x, y) {
                    return Some(ScrollbarHit {
                        container: idx,
                        axis,
                        on_thumb: true,
                    });
                }
            }
            for (axis, sb) in axes.iter().filter_map(|(a, s)| s.map(|s| (*a, s))) {
                let track = Rect {
                    x: sb.track.x - vox,
                    y: sb.track.y - voy,
                    w: sb.track.w,
                    h: sb.track.h,
                };
                if rect_contains_point(&track, x, y) {
                    return Some(ScrollbarHit {
                        container: idx,
                        axis,
                        on_thumb: false,
                    });
                }
            }
        }
        None
    }

    /// Capture a thumb drag for a press on `hit` at viewport point
    /// `(x, y)`. Returns `None` for track presses (those page instead)
    /// and stale hits.
    pub fn scrollbar_drag(&self, hit: ScrollbarHit, x: f32, y: f32) -> Option<ScrollbarDrag> {
        if !hit.on_thumb {
            return None;
        }
        let geom = self.scrollbar_geom(hit.container);
        let sb = match hit.axis {
            ScrollbarAxis::Vertical => geom.vertical?,
            ScrollbarAxis::Horizontal => geom.horizontal?,
        };
        let c = self.containers.get(hit.container)?;
        let (vox, voy) = self.scrollbar_viewport_offset();
        // Viewport-space snapshots for the mid-gesture geometry check
        // (see [`ScrollTree::scrollbar_drag_rebased`]).
        let to_viewport = |r: Rect| Rect {
            x: r.x - vox,
            y: r.y - voy,
            w: r.w,
            h: r.h,
        };
        let press_track = to_viewport(sb.track);
        let press_frame = to_viewport(self.scrollbar_frame_rect(hit.container));
        let (track_start, track_len, thumb_start, thumb_len, cursor, min, max) = match hit.axis {
            ScrollbarAxis::Vertical => (
                sb.track.y - voy,
                sb.track.h,
                sb.thumb.y - voy,
                sb.thumb.h,
                y,
                c.state.min_offset().1,
                c.state.max_offset().1,
            ),
            ScrollbarAxis::Horizontal => (
                sb.track.x - vox,
                sb.track.w,
                sb.thumb.x - vox,
                sb.thumb.w,
                x,
                c.state.min_offset().0,
                c.state.max_offset().0,
            ),
        };
        Some(ScrollbarDrag {
            node: c.node,
            axis: hit.axis,
            track_start,
            track_len,
            thumb_len,
            grab: cursor - thumb_start,
            min,
            max,
            press_track,
            press_frame,
        })
    }

    /// Rebase a press-time [`ScrollbarDrag`] against the live tree, or
    /// `None` when the gesture must end. The window worker calls this
    /// on every motion event before mapping the cursor.
    ///
    /// The live viewport-space track rect is compared against the
    /// press-time snapshot: when the scrollport frame translated
    /// rigidly and the track translated by the same delta — an
    /// ancestor scrolled (the viewport included) — `track_start`
    /// shifts by the axis delta with `grab` unchanged, so the thumb
    /// stays glued under the cursor instead of jumping. Anything else
    /// ends the gesture: the track resized, the track repositioned
    /// relative to its frame (a relayout moved or mirrored it, e.g. an
    /// LTR/RTL flip), the track went degenerate, or the container is
    /// gone. The container's own scroll never moves its frame (see
    /// [`ScrollTree::scrollbar_frame_rect`]), so a drag's own writes
    /// always rebase to a zero delta here.
    pub fn scrollbar_drag_rebased(&self, drag: ScrollbarDrag) -> Option<ScrollbarDrag> {
        let idx = self.scrollbar_container_idx(drag.node)?;
        let geom = self.scrollbar_geom(idx);
        let sb = match drag.axis {
            ScrollbarAxis::Vertical => geom.vertical,
            ScrollbarAxis::Horizontal => geom.horizontal,
        }?;
        let (vox, voy) = self.scrollbar_viewport_offset();
        let to_viewport = |r: Rect| Rect {
            x: r.x - vox,
            y: r.y - voy,
            w: r.w,
            h: r.h,
        };
        let frame = to_viewport(self.scrollbar_frame_rect(idx));
        let track = to_viewport(sb.track);
        // Geometry equality within float noise: identical layouts
        // produce identical values, so anything beyond this epsilon is
        // a real layout or scroll change.
        const EPS: f32 = 1e-3;
        let same = |a: f32, b: f32| (a - b).abs() <= EPS;
        let frame_dx = frame.x - drag.press_frame.x;
        let frame_dy = frame.y - drag.press_frame.y;
        let track_dx = track.x - drag.press_track.x;
        let track_dy = track.y - drag.press_track.y;
        let translated = same(frame.w, drag.press_frame.w)
            && same(frame.h, drag.press_frame.h)
            && same(track.w, drag.press_track.w)
            && same(track.h, drag.press_track.h)
            && same(track_dx, frame_dx)
            && same(track_dy, frame_dy);
        if !translated {
            return None;
        }
        let mut rebased = drag;
        rebased.track_start += match drag.axis {
            ScrollbarAxis::Vertical => track_dy,
            ScrollbarAxis::Horizontal => track_dx,
        };
        Some(rebased)
    }

    /// Target offset for a track click at viewport point `(x, y)` on
    /// `hit`'s track: one scrollport page toward the click side of the
    /// thumb, from the live offset. Returns `None` for thumb presses
    /// (those drag instead) and stale hits. The caller clamps through
    /// [`ScrollState::set_offset`](bowser_js::ScrollState::set_offset).
    pub fn scrollbar_page_target(&self, hit: ScrollbarHit, x: f32, y: f32) -> Option<f32> {
        if hit.on_thumb {
            return None;
        }
        let geom = self.scrollbar_geom(hit.container);
        let sb = match hit.axis {
            ScrollbarAxis::Vertical => geom.vertical?,
            ScrollbarAxis::Horizontal => geom.horizontal?,
        };
        let c = self.containers.get(hit.container)?;
        let frame = self.scrollbar_frame_rect(hit.container);
        let (vox, voy) = self.scrollbar_viewport_offset();
        let (cursor, thumb_start, thumb_end, page, offset) = match hit.axis {
            ScrollbarAxis::Vertical => (
                y,
                sb.thumb.y - voy,
                sb.thumb.y - voy + sb.thumb.h,
                frame.h,
                c.state.offset().1,
            ),
            ScrollbarAxis::Horizontal => (
                x,
                sb.thumb.x - vox,
                sb.thumb.x - vox + sb.thumb.w,
                frame.w,
                c.state.offset().0,
            ),
        };
        let dir = if cursor < thumb_start {
            -1.0
        } else if cursor > thumb_end {
            1.0
        } else {
            // On the thumb: a drag, not a page (unreachable through
            // `scrollbar_at`, which reports thumbs first).
            return None;
        };
        Some(offset + dir * page)
    }
}

/// Target offset for one axis of
/// [`ScrollTree::reveal_node_with_options`]: align the revealed
/// `[pos, pos + len]` (page coords) against the scrollport
/// `[sp, sp + sp_len]` (page coords) per `align`, given the axis's
/// current offset. The caller clamps the result into `[origin, max]`
/// (via [`ScrollState::set_offset`] or
/// [`ScrollTree::start_smooth_scroll`]).
fn reveal_target(
    align: RevealAlign,
    pos: f32,
    len: f32,
    sp: f32,
    sp_len: f32,
    current: f32,
) -> f32 {
    match align {
        RevealAlign::Keep => current,
        RevealAlign::Start => pos - sp,
        RevealAlign::End => (pos + len) - (sp + sp_len),
        RevealAlign::Center => (pos + len / 2.0) - (sp + sp_len / 2.0),
        RevealAlign::Nearest => {
            let vis_start = sp + current;
            let vis_end = vis_start + sp_len;
            if pos >= vis_start && pos + len <= vis_end {
                // Already fully visible: the minimal scroll is none.
                current
            } else if pos < vis_start {
                // Starts above/left of the visible range (or is larger
                // than the scrollport): align the leading edges.
                pos - sp
            } else {
                // Ends below/right of the visible range: align the
                // trailing edges.
                (pos + len) - (sp + sp_len)
            }
        }
    }
}

/// `(node border box, node computed style, container chain
/// innermost-first)` for revealing `node`: DFS over the layout tree
/// tracking scroller ancestry.
fn reveal_chain<'a>(
    tree: &ScrollTree,
    root: &'a LayoutBox,
    node: NodeId,
) -> Option<(Rect, Option<&'a bowser_css::ComputedStyle>, Vec<usize>)> {
    // Iterative preorder with explicit exit markers: the tree can be
    // arbitrarily deep (finding #4). On exit of a container box, pop it
    // from the ancestor chain.
    let mut ancestors: Vec<usize> = Vec::new();
    let mut stack: Vec<(&LayoutBox, bool)> = vec![(root, false)];
    while let Some((box_, exit)) = stack.pop() {
        if exit {
            if box_.node.and_then(|n| tree.index_of(n)).is_some() {
                ancestors.pop();
            }
            continue;
        }
        if box_.node == Some(node) {
            let mut chain = ancestors.clone();
            chain.reverse();
            return Some((box_.rect, box_.style.as_deref(), chain));
        }
        if let Some(n) = box_.node
            && let Some(i) = tree.index_of(n)
        {
            ancestors.push(i);
        }
        stack.push((box_, true));
        for child in box_.children.iter().rev() {
            stack.push((child, false));
        }
    }
    None
}

/// The node's `scroll-margin` longhands in px, as
/// `(top, right, bottom, left)`: `reveal_node` aligns the
/// scroll-margin-expanded rect, not the border box (CSS Scroll Snap §2).
///
/// `root_font_size` is the computed font size of the document root
/// (what `rem` resolves against); `viewport_w`/`h` are the `vw`/`vh`
/// bases.
///
/// Only `<length>` longhands resolve here: `auto` computes to zero and
/// percentages are invalid per CSS Scroll Snap §2, so anything else —
/// including a property `bowser_css` does not know — reads as zero.
/// Negative lengths are invalid at parse time (they keep the zero
/// initial); the `.max(0.0)` below is the second layer for values that
/// arrive negative through another path.
fn scroll_margins(
    style: Option<&bowser_css::ComputedStyle>,
    root_font_size: f32,
    viewport_w: f32,
    viewport_h: f32,
) -> (f32, f32, f32, f32) {
    let Some(style) = style else {
        return (0.0, 0.0, 0.0, 0.0);
    };
    // `em` basis: the element's own computed font size when it is a
    // resolved px length; the root font size otherwise (hand-built
    // styles outside the normal pipeline, which may carry no
    // `font-size` at all).
    let em_basis = px_font_size(style, root_font_size);
    let len = |prop: &str| -> f32 {
        match style.get(prop) {
            // scroll-margin longhands take `<length>` only (percentages
            // are invalid per CSS Scroll Snap §2): anything else —
            // including the not-yet-parsed property — is zero.
            Some(bowser_css::CssValue::Length(v, unit)) => {
                bowser_css::length_to_px(*v, unit, em_basis, root_font_size, viewport_w, viewport_h)
                    .max(0.0)
            }
            _ => 0.0,
        }
    };
    (
        len("scroll-margin-top"),
        len("scroll-margin-right"),
        len("scroll-margin-bottom"),
        len("scroll-margin-left"),
    )
}

/// The `font-size` when the computed style carries it as a resolved px
/// length (author wrote `px`), else `fallback`. The cascade keeps raw
/// values (`em`, `%`, keywords) — only layout can resolve those against
/// the parent — so a non-px `font-size` falls back here.
pub(crate) fn px_font_size(style: &bowser_css::ComputedStyle, fallback: f32) -> f32 {
    match style.get("font-size") {
        Some(bowser_css::CssValue::Length(v, bowser_css::LengthUnit::Px)) => *v,
        _ => fallback,
    }
}

/// Length-resolution bases for `transform: translate()`: everything
/// [`bowser_css::length_to_px`] needs beyond the value itself.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TranslateCtx {
    pub root_font_size: f32,
    pub viewport_w: f32,
    pub viewport_h: f32,
}

/// The used `translate()` offset of a box's computed `transform`, in
/// px: how far the box paints from its laid-out position.
///
/// Sums every `translate()`/`translatex()`/`translatey()` in the list
/// (the parser folds the single-axis forms into
/// [`bowser_css::TransformFunction::Translate`], and a space-separated
/// list into a `List` of them). Percentages resolve against the box's
/// own border-box size (CSS Transforms 1 §13.1).
///
/// Only translations are resolved. `scale()`/`rotate()` would need a
/// transform origin and matrix inversion in every hit-test path, and
/// the opaque `Other` functions (`matrix()`, `skew()`, 3D forms,
/// `calc()` arguments) cannot be evaluated at all — and none of them
/// are rendered by paint yet either, so resolving them as identity
/// keeps hit-testing exactly where rendering is instead of inventing
/// a second, disagreeing geometry.
pub(crate) fn box_translate(
    style: Option<&bowser_css::ComputedStyle>,
    box_w: f32,
    box_h: f32,
    ctx: TranslateCtx,
) -> (f32, f32) {
    use bowser_css::CssValue;
    let Some(style) = style else {
        return (0.0, 0.0);
    };
    // One `Transform` per function, or a `List` of them for a
    // space-separated list (see `has_non_none_transform`).
    let mut tx = 0.0f32;
    let mut ty = 0.0f32;
    match style.get("transform") {
        Some(CssValue::Transform(fns)) => {
            add_translate_fns(fns, style, box_w, box_h, ctx, &mut tx, &mut ty);
        }
        Some(CssValue::List(items)) => {
            for item in items {
                if let CssValue::Transform(fns) = item {
                    add_translate_fns(fns, style, box_w, box_h, ctx, &mut tx, &mut ty);
                }
            }
        }
        _ => {}
    }
    (tx, ty)
}

/// Sum the `translate()` functions of one function list into
/// `(tx, ty)`.
fn add_translate_fns(
    fns: &[bowser_css::TransformFunction],
    style: &bowser_css::ComputedStyle,
    box_w: f32,
    box_h: f32,
    ctx: TranslateCtx,
    tx: &mut f32,
    ty: &mut f32,
) {
    use bowser_css::TransformFunction;
    for f in fns {
        if let TransformFunction::Translate { x, y } = f {
            *tx += translate_len_px(x, box_w, style, ctx);
            *ty += translate_len_px(y, box_h, style, ctx);
        }
    }
}

/// One `translate()` argument in px: lengths resolve with the usual
/// bases, percentages against the axis' reference length, and the
/// unitless zero the parser keeps as `Number` is 0px. Anything else
/// cannot appear (the parser rejects it) — resolve as zero.
fn translate_len_px(
    value: &bowser_css::CssValue,
    pct_base: f32,
    style: &bowser_css::ComputedStyle,
    ctx: TranslateCtx,
) -> f32 {
    use bowser_css::CssValue;
    match value {
        CssValue::Length(n, unit) => {
            let em = px_font_size(style, ctx.root_font_size);
            bowser_css::length_to_px(
                *n,
                unit,
                em,
                ctx.root_font_size,
                ctx.viewport_w,
                ctx.viewport_h,
            )
        }
        CssValue::Percentage(p) => p / 100.0 * pct_base,
        CssValue::Number(n) => *n,
        _ => 0.0,
    }
}

/// The `(dx, dy)` shift, `cb_above` flag, and `fixed_cb_inside` flag
/// for a child box, given the shift accumulated for its parent's
/// scrolling content, whether the nearest positioned ancestor strictly
/// above the parent lies inside that content, and whether the nearest
/// fixed containing-block trigger strictly above the parent lies
/// strictly below the scroller.
///
/// Out-of-flow subtrees that do not scroll with a container are left
/// alone, mirroring the `cb_inside`/`fixed_cb_inside` rules
/// `bowser_layout` uses for scrollable overflow extent: a
/// `position: fixed` box is viewport-anchored unless a fixed-CB trigger
/// (non-`none` `transform`, `filter`, `perspective`, `will-change`,
/// `contain` — see [`bowser_layout::establishes_fixed_cb`]) between it
/// and the scroller makes it take its containing block from inside the
/// scroller, in which case it scrolls with the content (CSS Transforms
/// 1 §3); a `position: absolute` box whose containing block lies
/// outside the shifted content keeps its laid-out position. Letting
/// them shift would drag viewport-anchored content along with the
/// scroll (and disagree with the overflow extent, which already
/// excludes them).
///
/// The returned flags feed the child's own children: a positioned (or
/// fixed-CB-establishing) child that scrolls with the content is
/// itself a containing block inside it. A box with no computed style
/// resolves to the CSS initial value (`static`), exactly as layout's
/// own `position_of` callers treat it.
fn child_shift(
    child: &LayoutBox,
    dx: f32,
    dy: f32,
    cb_above: bool,
    fixed_cb_inside: bool,
) -> (f32, f32, bool, bool) {
    let pos = child
        .style
        .as_deref()
        .map(position_of)
        .unwrap_or(Position::Static);
    let fixed_cb_here = child.style.as_deref().is_some_and(establishes_fixed_cb);
    // A fixed box under a fixed-CB trigger inside the scroller takes
    // its containing block from that trigger and scrolls with the
    // content; otherwise it is viewport-anchored and neither the box
    // nor its subtree moves with this scroll. Either way the box is
    // positioned, so its descendants resolve their containing block
    // against it.
    if pos == Position::Fixed {
        if fixed_cb_inside {
            (dx, dy, true, true)
        } else {
            (0.0, 0.0, true, false)
        }
    // Absolutely positioned against a containing block outside the
    // shifted content: its laid-out position is independent of this
    // scroll, so the whole subtree stays.
    } else if pos == Position::Absolute && !cb_above {
        (0.0, 0.0, false, false)
    } else {
        (
            dx,
            dy,
            pos != Position::Static || cb_above || fixed_cb_here,
            fixed_cb_inside || fixed_cb_here,
        )
    }
}

/// Whether the box clips its children: any non-`visible` overflow
/// (`hidden`, `clip`, `scroll`, `auto`) clips. Mirrors
/// `bowser_paint`'s clip-frame rule, so the wrap pass knows when an
/// out-of-range out-of-flow drain must have left a wrapper clip pair
/// around its items (see [`OofCb::clips`]).
fn overflow_clips(style: &bowser_css::ComputedStyle) -> bool {
    let clips = |v: Option<&str>| {
        matches!(
            v,
            Some("hidden") | Some("clip") | Some("scroll") | Some("auto")
        )
    };
    clips(style.get_str_by_index(props::OVERFLOW))
        || clips(style.get_str_by_index(props::OVERFLOW_X))
        || clips(style.get_str_by_index(props::OVERFLOW_Y))
}

/// Whether the display list holds one of the box's own items — a
/// background, border, or shadow painted at its baked border box —
/// inside the item range `[s, e]`: locates an out-of-flow box's items
/// for the fast-path in-range check (see [`OofWrap`]). `false` when the
/// box paints no such item (transparent, borderless).
fn oof_box_in_range(items: &[DisplayItem], d_rect: Rect, s: usize, e: usize) -> bool {
    items.iter().enumerate().any(|(i, item)| {
        i >= s
            && i <= e
            && match item {
                DisplayItem::FillRect { rect, .. }
                | DisplayItem::Border { rect, .. }
                | DisplayItem::BoxShadow { rect, .. } => *rect == d_rect,
                _ => false,
            }
    })
}

/// Paint signatures a layout box's subtree can emit: the keys the
/// display-list builder stamps verbatim from layout, used to attribute
/// items to a sticky box exactly (no content heuristics).
///
/// The sets are exact, not over-approximated: every key below is both
/// necessary (a box that cannot emit the item contributes nothing) and
/// sufficient (every item the builder emits for the subtree matches a
/// key). Exactness matters because the disjointness gate below treats
/// any shared key as ambiguity and falls back to re-bake — phantom keys
/// would disable the fold on ordinary pages.
#[derive(Default)]
struct PaintSignatures {
    /// Border-box rects of element boxes: the key `Border`/`BoxShadow`/
    /// `Image` items stamp. Text boxes never emit these (all gated on
    /// the box's own style), so they contribute nothing here.
    rects: Vec<Rect>,
    /// `(rect, color)` for background fills: a box emits
    /// `FillRect(rect, color)` iff its effective background color has
    /// nonzero alpha — element boxes read their own style, text boxes
    /// the inherited parent style (the builder reads it through the
    /// inherited `style`).
    fills: Vec<(Rect, Color)>,
    /// Border-box rects of boxes with a background image: the key
    /// `RepeatImage`/`Gradient` stamps as its area. `url()` backgrounds
    /// read through the inherited style (the builder emits `RepeatImage`
    /// for text boxes too, so they contribute); gradients read the box's
    /// own style only (`background-image` is not inherited, so text
    /// boxes emit nothing and contribute nothing).
    bg_images: Vec<Rect>,
    /// One key per text fragment.
    frags: Vec<FragKey>,
}

/// The exact key a text item stamps: the fragment's `(rect, baseline,
/// text_range)` plus the originating text box's whole text. Text content
/// alone does not identify a box (identical text in two boxes), and
/// geometry alone does not either (two boxes' fragments can coincide in
/// the baked frame with different text); together they do, and any
/// residual collision forces a re-bake.
#[derive(Clone)]
struct FragKey {
    rect: Rect,
    baseline: f32,
    text_range: std::ops::Range<usize>,
    /// The originating [`LayoutBox::text`](bowser_layout::LayoutBox::text):
    /// verbatim what `TextRun.text` carries.
    text: String,
}

impl FragKey {
    fn matches(
        &self,
        rect: &Rect,
        baseline: f32,
        text_range: &std::ops::Range<usize>,
        text: &str,
    ) -> bool {
        self.rect == *rect
            && self.baseline == baseline
            && self.text_range == *text_range
            && self.text == text
    }
}

/// Effective background fill color, if the box emits a `FillRect` for
/// it: `Some(color)` when the computed `background-color` is a color
/// with nonzero alpha (mirrors the builder's gate exactly).
fn bg_fill_color(style: &bowser_css::ComputedStyle) -> Option<Color> {
    match style.get_by_index(props::BACKGROUND_COLOR) {
        Some(bowser_css::CssValue::Color(c)) if c.a > 0 => Some(Color::rgba8(c.r, c.g, c.b, c.a)),
        _ => None,
    }
}

/// Whether the box emits a `RepeatImage` for a `url()` background image:
/// the builder reads `background-image` through the inherited style, so
/// text boxes emit too — signatures use the effective style. (A resolver
/// miss just drops the item, which can only under-populate the list —
/// the range checks below then find no owned item and fall back to
/// re-bake, never misattribute.)
fn has_bg_url(style: &bowser_css::ComputedStyle) -> bool {
    matches!(
        style.get_by_index(props::BACKGROUND_IMAGE),
        Some(bowser_css::CssValue::Url(_))
    )
}

/// Whether the box emits a `Gradient` for a gradient background image:
/// `background-image` is not inherited, so the builder reads the box's
/// own style only and text boxes emit nothing — signatures use the own
/// style. (A parse failure drops the item, the same safe direction as
/// the resolver miss above.) Splitting the two cases keeps the
/// signature set exactly mirroring builder emission: a gradient text
/// rect must not linger here with no item (the ambiguity gate would
/// spuriously force re-bakes on coinciding rects), and a `url()` text
/// rect must not lose its key (its `RepeatImage` item would otherwise
/// attribute `Unknown` and force re-bakes on ordinary pages).
fn has_bg_gradient(style: &bowser_css::ComputedStyle) -> bool {
    matches!(
        style.get_by_index(props::BACKGROUND_IMAGE),
        Some(bowser_css::CssValue::Gradient { .. })
    )
}

/// Collect the paint signatures of `root`'s subtree. When `exclude` is
/// `Some`, the subtree rooted at that box is pruned (identity by
/// reference: both come from the same paint tree).
fn collect_paint_signatures(
    root: &LayoutBox,
    exclude: Option<&LayoutBox>,
    out: &mut PaintSignatures,
) {
    // (box, inherited parent style, for text-box background keys).
    let mut stack: Vec<(&LayoutBox, Option<&bowser_css::ComputedStyle>)> = vec![(root, None)];
    while let Some((b, parent_style)) = stack.pop() {
        if exclude.is_some_and(|x| std::ptr::eq(b, x)) {
            continue;
        }
        // The builder reads background properties through the inherited
        // style (own, else parent).
        let own = b.style.as_deref();
        let eff = own.or(parent_style);
        if b.box_type != BoxType::Text {
            out.rects.push(b.rect);
            // Collapsing table cells paint their border centered on the
            // grid lines: the rect is the border box grown by half the
            // winning widths (mirrors the builder's collapsing path).
            if let Some(winners) = b.collapsed_borders.as_deref() {
                let r = b.rect;
                out.rects.push(Rect {
                    x: r.x - winners[3].width / 2.0,
                    y: r.y - winners[0].width / 2.0,
                    w: r.w + (winners[3].width + winners[1].width) / 2.0,
                    h: r.h + (winners[0].width + winners[2].width) / 2.0,
                });
            }
        }
        if let Some(color) = eff.and_then(bg_fill_color) {
            out.fills.push((b.rect, color));
        }
        // Background-image keys mirror the builder's two arms exactly:
        // `url()` reads through the inherited style, gradients through
        // the box's own style only (see `has_bg_url`/`has_bg_gradient`).
        if eff.is_some_and(has_bg_url) || own.is_some_and(has_bg_gradient) {
            out.bg_images.push(b.rect);
        }
        if b.box_type == BoxType::Text
            && let Some(text) = b.text.as_deref()
        {
            for f in &b.fragments {
                out.frags.push(FragKey {
                    rect: f.rect,
                    baseline: f.baseline,
                    text_range: f.text_range.clone(),
                    text: text.to_owned(),
                });
            }
        }
        let child_parent = own.or(parent_style);
        stack.extend(b.children.iter().map(|c| (c, child_parent)));
    }
}

/// How a display-list item attributes against the owned/foreign
/// signature sets.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ItemAttribution {
    /// The item's key is in the sticky box's signature set.
    Owned,
    /// The item's key is in the foreign signature set.
    Foreign,
    /// Group push/pop: structural, no key.
    Structural,
    /// No attributable key, or a kind whose key cannot be derived here:
    /// the range is not exactly isolable.
    Unknown,
}

/// Attribute one display-list item against the owned (sticky box's
/// subtree) and foreign (container's subtree minus the sticky box's)
/// signature sets. `ListMarker` needs DOM list-item detection
/// unavailable here, `EdgeGlow` is appended after the wrap, and
/// per-tile `BackgroundImage`s are no longer emitted by the builder
/// (backgrounds go through `RepeatImage`): any of these in the range
/// forces a re-bake.
fn attribute_item(
    item: &DisplayItem,
    owned: &PaintSignatures,
    foreign: &PaintSignatures,
) -> ItemAttribution {
    // The caller guarantees the sets are disjoint; check owned first so
    // a (buggy) collision attributes to the box rather than silently
    // dropping one of its items.
    let rect_owner = |r: &Rect| {
        if owned.rects.contains(r) {
            ItemAttribution::Owned
        } else if foreign.rects.contains(r) {
            ItemAttribution::Foreign
        } else {
            ItemAttribution::Unknown
        }
    };
    let fill_owner = |r: &Rect, color: Color| {
        if owned.fills.contains(&(*r, color)) {
            ItemAttribution::Owned
        } else if foreign.fills.contains(&(*r, color)) {
            ItemAttribution::Foreign
        } else {
            ItemAttribution::Unknown
        }
    };
    let frag_owner = |r: &Rect, baseline: f32, text_range: &std::ops::Range<usize>, text: &str| {
        if owned
            .frags
            .iter()
            .any(|k| k.matches(r, baseline, text_range, text))
        {
            ItemAttribution::Owned
        } else if foreign
            .frags
            .iter()
            .any(|k| k.matches(r, baseline, text_range, text))
        {
            ItemAttribution::Foreign
        } else {
            ItemAttribution::Unknown
        }
    };
    match item {
        DisplayItem::FillRect { rect, color } => fill_owner(rect, *color),
        DisplayItem::Border { rect, .. }
        | DisplayItem::BoxShadow { rect, .. }
        | DisplayItem::Image { rect, .. } => rect_owner(rect),
        DisplayItem::RepeatImage { area, .. } | DisplayItem::Gradient { rect: area, .. } => {
            if owned.bg_images.contains(area) {
                ItemAttribution::Owned
            } else if foreign.bg_images.contains(area) {
                ItemAttribution::Foreign
            } else {
                ItemAttribution::Unknown
            }
        }
        DisplayItem::TextRun {
            text,
            rect,
            baseline,
            text_range,
            ..
        }
        | DisplayItem::TextShadow {
            text,
            rect,
            baseline,
            text_range,
            ..
        } => frag_owner(rect, *baseline, text_range, text.as_ref()),
        // Decorations stamp the fragment rect they draw over, with no
        // text key: attributable only when the rect is unambiguous
        // (present in exactly one set's fragment rects).
        DisplayItem::Decoration { rect, .. } => {
            let in_owned = owned.frags.iter().any(|k| k.rect == *rect);
            let in_foreign = foreign.frags.iter().any(|k| k.rect == *rect);
            match (in_owned, in_foreign) {
                (true, false) => ItemAttribution::Owned,
                (false, true) => ItemAttribution::Foreign,
                _ => ItemAttribution::Unknown,
            }
        }
        DisplayItem::PushClip { .. }
        | DisplayItem::PushRoundedClip { .. }
        | DisplayItem::PushClipPath { .. }
        | DisplayItem::PushOpacity { .. }
        | DisplayItem::PushBlend { .. }
        | DisplayItem::PushFilter { .. }
        | DisplayItem::PushBackdropFilter { .. }
        | DisplayItem::PushTransform { .. }
        | DisplayItem::PopClip
        | DisplayItem::PopOpacity
        | DisplayItem::PopBlend
        | DisplayItem::PopFilter
        | DisplayItem::PopBackdropFilter
        | DisplayItem::PopTransform => ItemAttribution::Structural,
        DisplayItem::BackgroundImage { .. }
        | DisplayItem::ListMarker { .. }
        | DisplayItem::EdgeGlow { .. } => ItemAttribution::Unknown,
    }
}

/// Whether a sticky box's subtree can be isolated to an exact
/// display-list range for the fast-path fold. Two subtree shapes defeat
/// exact isolation and keep the re-bake path (no pair is armed):
///
/// * a nested `position: sticky` descendant: its items would take both
///   the outer and the inner pair's shift, but the two fold models are
///   not compositional (the inner band is baked against the outer box's
///   baked position);
/// * a `position: relative` descendant with non-`auto` z-index: paint
///   hoists it to the nearest ancestor stacking context when its parent
///   is not one, so its items may drain outside the box's contiguous
///   item block — out of the container's clip range entirely — where the
///   range checks below cannot see them.
///
/// `position: absolute` descendants are fine: the sticky box is itself
/// positioned, so their containing block is the box or deeper and they
/// drain inside its block. `position: fixed` descendants drain at the
/// viewport (or a fixed-CB trigger) and correctly stay out of the sticky
/// shift.
fn sticky_subtree_isolable(sticky: &LayoutBox) -> bool {
    let mut stack: Vec<&LayoutBox> = Vec::new();
    stack.extend(sticky.children.iter());
    while let Some(b) = stack.pop() {
        if let Some(style) = b.style.as_deref() {
            match position_of(style) {
                Position::Sticky => return false,
                Position::Relative if !z_index_is_auto(style) => return false,
                _ => {}
            }
        }
        stack.extend(b.children.iter());
    }
    true
}

/// Whether `z-index` is `auto` (or missing/unrecognized): mirrors
/// `bowser_layout`'s `z_index_of`, which is crate-private there.
fn z_index_is_auto(style: &bowser_css::ComputedStyle) -> bool {
    !matches!(style.get("z-index"), Some(bowser_css::CssValue::Number(_)))
}

/// Display-list range of a `position: sticky` box's items, if exactly
/// isolable.
///
/// A sticky box paints atomically — its whole subtree at its Appendix E
/// slot — so its items are contiguous in the container's clip range
/// `(s, e)`. Attribution is by paint signature, not content: every box
/// in the sticky box's subtree contributes its border-box rect (the key
/// box-level items stamp) and every text fragment its
/// `(rect, baseline, text_range)` (the key text items stamp); the same
/// sets are collected for the foreign boxes — the container's subtree
/// minus the sticky box's. Any signature shared between the two sets
/// (e.g. identical text at the identical position in two boxes) is
/// ambiguous and forces a re-bake, as does any item kind with no
/// derivable key (`ListMarker`, `EdgeGlow`, per-tile
/// `BackgroundImage`). With disjoint sets the range is exact: it spans
/// the owned items, holds no foreign item, and leaves no owned item
/// outside.
///
/// Returns `None` when no exact range exists. The caller then leaves the
/// box unwrapped and the tick falls back to re-bake rather than
/// mis-shifting content.
fn sticky_item_range(
    items: &[DisplayItem],
    sticky: &LayoutBox,
    container: &LayoutBox,
    s: usize,
    e: usize,
) -> Option<(usize, usize)> {
    let mut owned = PaintSignatures::default();
    collect_paint_signatures(sticky, None, &mut owned);
    let mut foreign = PaintSignatures::default();
    collect_paint_signatures(container, Some(sticky), &mut foreign);
    // Ambiguity gate: a shared signature cannot be attributed.
    let frag_collision = owned.frags.iter().any(|o| {
        foreign
            .frags
            .iter()
            .any(|f| o.matches(&f.rect, f.baseline, &f.text_range, &f.text))
    });
    if owned.rects.iter().any(|r| foreign.rects.contains(r))
        || owned.fills.iter().any(|k| foreign.fills.contains(k))
        || owned
            .bg_images
            .iter()
            .any(|r| foreign.bg_images.contains(r))
        || frag_collision
    {
        return None;
    }
    // Item kinds with no derivable key cannot be attributed; their
    // presence in the range forces a re-bake.
    if items
        .iter()
        .take(e + 1)
        .skip(s)
        .any(|item| attribute_item(item, &owned, &foreign) == ItemAttribution::Unknown)
    {
        return None;
    }
    // The box's items are exactly the owned items in `(s, e)`; they are
    // contiguous, so the range is [first, last].
    let mut lo: Option<usize> = None;
    let mut hi: usize = 0;
    for (i, item) in items.iter().enumerate().take(e + 1).skip(s) {
        if attribute_item(item, &owned, &foreign) == ItemAttribution::Owned {
            lo.get_or_insert(i);
            hi = i;
        }
    }
    let lo = lo?;
    // Exactness: no foreign item inside [lo, hi] ...
    if items
        .iter()
        .take(hi + 1)
        .skip(lo)
        .any(|item| attribute_item(item, &owned, &foreign) == ItemAttribution::Foreign)
    {
        return None;
    }
    // ... and no owned item outside it (e.g. a hoisted out-of-flow
    // descendant painted elsewhere in the container's range).
    if items
        .iter()
        .enumerate()
        .take(e + 1)
        .skip(s)
        .any(|(i, item)| {
            (i < lo || i > hi) && attribute_item(item, &owned, &foreign) == ItemAttribution::Owned
        })
    {
        return None;
    }
    // Structural balance: groups opened inside the range must close
    // inside it, so splicing the transform pair around `[lo, hi]` nests
    // correctly.
    let mut depth: i32 = 0;
    for item in items.iter().take(hi + 1).skip(lo) {
        match item {
            DisplayItem::PushClip { .. }
            | DisplayItem::PushRoundedClip { .. }
            | DisplayItem::PushOpacity { .. }
            | DisplayItem::PushBlend { .. }
            | DisplayItem::PushTransform { .. } => depth += 1,
            DisplayItem::PopClip
            | DisplayItem::PopOpacity
            | DisplayItem::PopBlend
            | DisplayItem::PopTransform => {
                depth -= 1;
                if depth < 0 {
                    return None;
                }
            }
            _ => {}
        }
    }
    if depth != 0 {
        return None;
    }
    Some((lo, hi))
}

/// `fragments` shifted by `(dx, dy)`: rects and baselines move with the
/// box. The unshifted fast path is a plain copy, like `Clone`.
fn shift_fragments(
    fragments: &[bowser_layout::TextFragment],
    dx: f32,
    dy: f32,
) -> Vec<bowser_layout::TextFragment> {
    if dx == 0.0 && dy == 0.0 {
        return fragments.to_vec();
    }
    fragments
        .iter()
        .map(|f| {
            let mut rect = f.rect;
            rect.x += dx;
            rect.y += dy;
            bowser_layout::TextFragment {
                rect,
                baseline: f.baseline + dy,
                text_range: f.text_range.clone(),
            }
        })
        .collect()
}

/// One `position: sticky` box's constraint band: the range of its
/// constraining scroller's offsets over which the box may be stuck or
/// transitioning, i.e. scroll-dependent. Built once per layout by
/// [`ScrollTree::build`]; the per-scroll check is a few range tests.
///
/// The band over-approximates from the baked rect: a box is stuck or
/// transitioning only while its natural position is within about a
/// scrollport of the scrollport edges (inset by the sticky offsets),
/// so the band spans the baked position +/- scrollport size, the
/// box's own extent, the resolved insets, and float slack. A stuck
/// box's baked position tracks the scroll offset, so its band tracks
/// too across the re-bakes the band itself triggers; a box far above
/// or below falls out of band and its scrolls keep the fast path.
/// Sides whose inset is not resolvable app-side (percentages need the
/// containing block, em/rem need font metrics, calc() needs layout)
/// stay unbounded on that side: always dirty, exactly as before the
/// hysteresis. Missing a real transition would reintroduce the
/// stuck-box lag the dirtying exists to fix, so every approximation
/// here errs toward dirtying.
#[derive(Debug, Clone)]
struct StickyBand {
    /// Container index of the constraining scroller (0 = viewport):
    /// only this scroller's offset can move the box.
    scroller: usize,
    /// The sticky box's DOM node: identifies it for the fast-path fold.
    node: NodeId,
    /// Scroll-offset band per axis; `-inf`/`+inf` ends are unbounded.
    lo_x: f32,
    hi_x: f32,
    lo_y: f32,
    hi_y: f32,
    /// The scroller's offset when the band was built: the fold's base.
    bake_x: f32,
    bake_y: f32,
    /// The box's baked scrollport-relative position (`rect - scrollport`).
    pos_x: f32,
    pos_y: f32,
    /// Box size.
    w: f32,
    h: f32,
    /// Resolved sticky insets per side.
    top: StickyInset,
    bottom: StickyInset,
    left: StickyInset,
    right: StickyInset,
    /// Whether the containing-block candidate is the constraining
    /// scroller's own box: then the block's content end is exactly the
    /// scroller's maximum offset plus the scrollport, so the stuck
    /// regime's far edge is exactly provable. Otherwise a stuck box
    /// keeps the re-bake path (conservative: the far edge needs the
    /// block's height definiteness, which lives in layout's forest).
    cb_is_scroller: bool,
    /// Band-frame border-box end of the containing-block candidate
    /// per axis: when it is not the scroller's own box the fold uses
    /// these to exclude the block-end-clamped zone, where the free
    /// interval would be unsound. They are conservative (border box,
    /// not content box; an upper bound for indefinite heights), so the
    /// exclusion only ever errs toward re-baking. Ignored when
    /// `cb_is_scroller` (the fold uses `cb_end_*`).
    cb_hi_x: f32,
    cb_hi_y: f32,
    /// Exact band-frame content-box end of the containing block per
    /// axis, when the block is the scroller's own box and the end is
    /// provable app-side (`Some(INFINITY)` when the block height is
    /// indefinite, so layout applies no clamp). `None` when unprovable
    /// (non-px padding/borders, or an unresolvable height): the fold
    /// then stays conservative.
    cb_end_x: Option<f32>,
    cb_end_y: Option<f32>,
    /// Used far margin in px (`margin-bottom`/`margin-right`),
    /// `None` when not a simple length: the containing-block clamp
    /// applies to the margin box, so the stuck regime's far edge moves
    /// in by it, and an unknown margin makes that edge unprovable.
    margin_far_y: Option<f32>,
    margin_far_x: Option<f32>,
    /// Used near margin in px (`margin-top`/`margin-left`), `None` when
    /// not a simple length. The fold models border-box positions, but
    /// layout sticks the margin box: any nonzero (or unresolvable) used
    /// margin makes the model unsound, so [`ScrollTree::sticky_band_fold`]
    /// falls back to re-bake then (see `margins_unsound`).
    margin_near_y: Option<f32>,
    margin_near_x: Option<f32>,
    /// Resolved border widths `(top, right, bottom, left)` in px of the
    /// constraining scroller's box, `None` when any side is unresolvable
    /// app-side (or the scroller is the viewport, which never folds).
    /// The fold models positions in the border-box frame against the
    /// padding-box scrollport extent (w79: layout's sticky view
    /// rectangle size is the padding-box size), so any nonzero (or
    /// unresolvable) border makes the model unsound and
    /// [`ScrollTree::sticky_band_fold`] falls back to re-bake then
    /// (see `scroller_borders_unsound`).
    scroller_border: Option<(f32, f32, f32, f32)>,
    /// Resolved paddings `(top, right, bottom, left)` in px of the
    /// constraining scroller's box, `None` when any side is unresolvable
    /// app-side (or the scroller is the viewport, which never folds).
    /// Layout's sticky pins are measured from the scroller's content
    /// origin (border-box origin plus border plus near padding), while
    /// the fold's positions are border-box-frame, so the near padding
    /// offsets every pin and transition the fold computes (see
    /// [`ScrollTree::sticky_axis_fold`]). An unresolvable padding makes
    /// that offset unprovable and the fold falls back to re-bake then
    /// (see `scroller_padding_unsound`); a nonzero resolvable padding
    /// folds exactly.
    scroller_padding: Option<(f32, f32, f32, f32)>,
}

impl StickyBand {
    /// Whether any used margin is nonzero or unresolvable: the fold
    /// models the border box, but layout sticks the margin box, so the
    /// fold is unsound and must fall back to re-bake.
    fn margins_unsound(&self) -> bool {
        [
            self.margin_near_y,
            self.margin_far_y,
            self.margin_near_x,
            self.margin_far_x,
        ]
        .iter()
        .any(|m| !matches!(m, Some(0.0)))
    }

    /// Whether the scroller has nonzero (or unresolvable) borders: the
    /// fold models in the border-box frame against the padding-box
    /// scrollport extent, but a nonzero border separates the two
    /// frames, so the fold is unsound and must fall back to re-bake.
    fn scroller_borders_unsound(&self) -> bool {
        match self.scroller_border {
            Some((t, r, b, l)) => t != 0.0 || r != 0.0 || b != 0.0 || l != 0.0,
            // Unresolvable (or viewport scroller, which never folds):
            // conservative.
            None => true,
        }
    }

    /// Whether the scroller's padding is unresolvable app-side (or the
    /// scroller is the viewport, which never folds): the near-padding
    /// offset of every pin is then unprovable, so the fold must fall
    /// back to re-bake. A resolved padding of any size is sound.
    fn scroller_padding_unsound(&self) -> bool {
        self.scroller_padding.is_none()
    }
}

/// One sticky box's exact extra shift for a scroll tick: apply
/// `(dx, dy)` to the box's items on top of its scroller's scroll
/// transform. See [`ScrollTree::sticky_shifts_at`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct StickyShift {
    /// The sticky box's DOM node.
    pub node: NodeId,
    /// Extra shift in CSS px, relative to the band's bake offset.
    pub dx: f32,
    pub dy: f32,
}

/// Slack around each band edge: covers f32 rounding in the baked
/// positions and the band arithmetic. The band is already
/// viewport-wide conservative; this is pure float hygiene.
const STICKY_BAND_SLACK: f32 = 2.0;

/// A physical sticky inset's (`top`/`right`/`bottom`/`left`) effect on
/// the band, per side.
#[derive(Debug, Clone, Copy)]
enum StickyInset {
    /// `auto` or absent: no constraint from this side.
    None,
    /// Absolute length resolved to px.
    Px(f32),
    /// Not resolvable app-side: the band stays unbounded on this
    /// side (always dirty, as before the hysteresis).
    Unknown,
}

/// Resolve a physical sticky inset to px: absolute lengths directly,
/// `vw`/`vh` against the viewport. Percentages need the containing
/// block, em/rem need font metrics, and calc() needs layout — all
/// `Unknown`, which keeps the band conservative.
fn sticky_inset(style: &bowser_css::ComputedStyle, prop: &str, viewport: f32) -> StickyInset {
    match style.get(prop) {
        None => StickyInset::None,
        Some(bowser_css::CssValue::Keyword(k)) if k.eq_ignore_ascii_case("auto") => {
            StickyInset::None
        }
        // Unitless zero is a valid `<length>`; the parser keeps it as
        // a number.
        Some(bowser_css::CssValue::Number(n)) if *n == 0.0 => StickyInset::Px(0.0),
        Some(bowser_css::CssValue::Length(v, unit)) => {
            let px = match unit {
                bowser_css::LengthUnit::Px => *v,
                bowser_css::LengthUnit::Cm => *v * 96.0 / 2.54,
                bowser_css::LengthUnit::Mm => *v * 96.0 / 25.4,
                bowser_css::LengthUnit::In => *v * 96.0,
                bowser_css::LengthUnit::Pt => *v * 96.0 / 72.0,
                bowser_css::LengthUnit::Pc => *v * 16.0,
                bowser_css::LengthUnit::Q => *v * 96.0 / 2.54 / 40.0,
                bowser_css::LengthUnit::Vw | bowser_css::LengthUnit::Vh => *v / 100.0 * viewport,
                _ => return StickyInset::Unknown,
            };
            StickyInset::Px(px)
        }
        _ => StickyInset::Unknown,
    }
}

/// `Some(px)` for a resolvable inset, `Some(None)` for no constraint,
/// `None` when the inset is unresolvable app-side (the caller bails to
/// the re-bake path, keeping the band's conservative semantics).
fn sticky_inset_opt(inset: StickyInset) -> Option<Option<f32>> {
    match inset {
        StickyInset::Px(v) => Some(Some(v)),
        StickyInset::None => Some(None),
        StickyInset::Unknown => None,
    }
}

/// Used far margin in px (`margin-bottom`/`margin-right`), or `None`
/// when it is not a simple absolute length: the containing-block clamp
/// applies to the margin box, so an unknown margin makes the block end
/// unprovable and the fold stays conservative.
fn used_margin_px(style: &bowser_css::ComputedStyle, prop: &str) -> Option<f32> {
    match style.get(prop) {
        Some(bowser_css::CssValue::Length(v, unit)) if *unit == bowser_css::LengthUnit::Px => {
            Some(*v)
        }
        Some(bowser_css::CssValue::Number(n)) if *n == 0.0 => Some(0.0),
        _ => None,
    }
}

/// Strict px length: `Some(px)` for px lengths and unitless/absent
/// zero, `None` for anything else (percentages, em, calc, ...).
/// Border widths accept the CSS keywords (`thin`/`medium`/`thick`);
/// a `none`/`hidden` border style forces zero regardless of width.
fn strict_px(value: Option<&bowser_css::CssValue>) -> Option<f32> {
    match value {
        Some(bowser_css::CssValue::Length(v, u)) if *u == bowser_css::LengthUnit::Px => Some(*v),
        Some(bowser_css::CssValue::Number(n)) if *n == 0.0 => Some(0.0),
        Some(bowser_css::CssValue::Keyword(k)) if k.eq_ignore_ascii_case("thin") => Some(1.0),
        Some(bowser_css::CssValue::Keyword(k)) if k.eq_ignore_ascii_case("medium") => Some(3.0),
        Some(bowser_css::CssValue::Keyword(k)) if k.eq_ignore_ascii_case("thick") => Some(5.0),
        None => Some(0.0),
        _ => None,
    }
}

/// Used border width for one side in px: zero when the side's border
/// style is `none`/`hidden` (the initial `medium` width then paints
/// nothing), else the resolved width; `None` when unresolvable
/// app-side.
fn used_border_px(
    style: &bowser_css::ComputedStyle,
    width_prop: &str,
    style_prop: &str,
) -> Option<f32> {
    let style_none = style.get(style_prop).is_some_and(|v| {
        matches!(
            v,
            bowser_css::CssValue::Keyword(k)
                if k.eq_ignore_ascii_case("none") || k.eq_ignore_ascii_case("hidden")
        )
    });
    if style_none {
        return Some(0.0);
    }
    strict_px(style.get(width_prop))
}

/// Exact band-frame content-box end of the scroller's own box along one
/// axis, or `None` when unprovable app-side. `Some(INFINITY)` when the
/// size is indefinite (`auto`/absent): layout then applies no
/// containing-block clamp. The band frame's origin is the scrollport
/// frame's (border-box) origin and the scrollport extent is the
/// border-box extent, so the content-box end is the extent minus
/// only the FAR padding and the FAR border: the near border and
/// near padding sit between the frame origin and the content box
/// and are part of the end, not subtracted from it. (Pre-w79 this
/// subtracted both paddings, understating the end by the near
/// padding for padded scrollers and making the fold re-bake where
/// it could have folded exactly.) Any non-px component bails to
/// `None` (conservative).
fn scroller_cb_end(
    cb: &LayoutBox,
    scrollport_extent: f32,
    size_prop: &str,
    pad_after: &str,
    border_after: &str,
) -> Option<f32> {
    let s = cb.style.as_deref()?;
    let definite = match s.get(size_prop) {
        None => false,
        Some(bowser_css::CssValue::Keyword(k)) if k.eq_ignore_ascii_case("auto") => false,
        _ => true,
    };
    if !definite {
        return Some(f32::INFINITY);
    }
    let pb = strict_px(s.get(pad_after))?;
    let bb = strict_px(s.get(border_after))?;
    Some(scrollport_extent - pb - bb)
}

/// Mirror of `bowser_layout`'s sticky containing-block candidate test:
/// the nearest such ancestor's content box bounds the in-flow sticky
/// constraint. Used only to tell whether that ancestor is the
/// constraining scroller's own box (see
/// [`StickyBand::cb_is_scroller`]).
fn sticky_cb_candidate(box_: &LayoutBox) -> bool {
    !matches!(box_.box_type, BoxType::AnonymousBlock)
        && matches!(
            box_.box_type,
            BoxType::Block
                | BoxType::InlineBlock
                | BoxType::TableCell
                | BoxType::Grid
                | BoxType::Table
        )
}

impl ScrollTree {
    /// The constraint band for one sticky box, or `None` when the box
    /// cannot be scroll-dependent: with no inset on any side layout's
    /// scrollport clamp is the identity and its containing-block clamp
    /// is scroll-independent, so no scroll offset moves the box.
    fn sticky_band_for(
        &self,
        box_: &LayoutBox,
        root: &LayoutBox,
        scroller: usize,
        viewport: bowser_layout::Size,
        cb: Option<&LayoutBox>,
    ) -> Option<StickyBand> {
        let style = box_.style.as_ref()?;
        let node = box_.node?;
        let container = &self.containers[scroller];
        let cb_is_scroller =
            cb.and_then(|c| c.node).is_some() && cb.and_then(|c| c.node) == container.node;
        let top = sticky_inset(style, "top", viewport.h);
        let bottom = sticky_inset(style, "bottom", viewport.h);
        let left = sticky_inset(style, "left", viewport.w);
        let right = sticky_inset(style, "right", viewport.w);
        if matches!(top, StickyInset::None)
            && matches!(bottom, StickyInset::None)
            && matches!(left, StickyInset::None)
            && matches!(right, StickyInset::None)
        {
            return None;
        }
        let container = &self.containers[scroller];
        let (sp_x, sp_y, sp_w, sp_h) = (
            container.scrollport.x,
            container.scrollport.y,
            container.scrollport.w,
            container.scrollport.h,
        );
        // Band-frame border-box end of the containing-block
        // candidate (`INFINITY` = unbounded): the free fold uses it to
        // exclude the clamped zone when the block is not the
        // scroller's own box.
        let (cb_hi_x, cb_hi_y) = match cb {
            Some(c) => (c.rect.x + c.rect.w - sp_x, c.rect.y + c.rect.h - sp_y),
            None => (f32::INFINITY, f32::INFINITY),
        };
        // Exact content-box end of the scroller's own box (band frame),
        // when provable: the stuck fold's far edge and the at-end
        // detection need it.
        let (cb_end_x, cb_end_y) = if cb_is_scroller {
            match cb {
                Some(c) => (
                    scroller_cb_end(c, sp_w, "width", "padding-right", "border-right-width"),
                    scroller_cb_end(c, sp_h, "height", "padding-bottom", "border-bottom-width"),
                ),
                None => (None, None),
            }
        } else {
            (None, None)
        };
        let r = box_.rect;
        let (bake_x, bake_y) = container.state.offset();
        // The box sticks (top side) once the scroll offset passes
        // `natural - inset`, where `natural` is measured from the
        // scrollport origin (`r.y - sp.y`): for the viewport sp.y = 0,
        // but for nested scrollers the raw box rect overstates the band
        // (finding #3). `hi` covers the box's own extent plus the
        // bottom/right insets so a bottom-anchored stick transition is
        // never missed either.
        let lo_y = match top {
            StickyInset::Px(t) => r.y - sp_y - t - sp_h - STICKY_BAND_SLACK,
            StickyInset::None => r.y - sp_y - sp_h - STICKY_BAND_SLACK,
            StickyInset::Unknown => f32::NEG_INFINITY,
        };
        let hi_y = match bottom {
            StickyInset::Px(b) => r.y - sp_y + r.h + sp_h + b + STICKY_BAND_SLACK,
            StickyInset::None => r.y - sp_y + r.h + sp_h + STICKY_BAND_SLACK,
            StickyInset::Unknown => f32::INFINITY,
        };
        let lo_x = match left {
            StickyInset::Px(l) => r.x - sp_x - l - sp_w - STICKY_BAND_SLACK,
            StickyInset::None => r.x - sp_x - sp_w - STICKY_BAND_SLACK,
            StickyInset::Unknown => f32::NEG_INFINITY,
        };
        let hi_x = match right {
            StickyInset::Px(x) => r.x - sp_x + r.w + sp_w + x + STICKY_BAND_SLACK,
            StickyInset::None => r.x - sp_x + r.w + sp_w + STICKY_BAND_SLACK,
            StickyInset::Unknown => f32::INFINITY,
        };
        Some(StickyBand {
            scroller,
            node,
            lo_x,
            hi_x,
            lo_y,
            hi_y,
            bake_x,
            bake_y,
            pos_x: r.x - sp_x,
            pos_y: r.y - sp_y,
            w: r.w,
            h: r.h,
            top,
            bottom,
            left,
            right,
            cb_is_scroller,
            cb_hi_x,
            cb_hi_y,
            cb_end_x,
            cb_end_y,
            margin_far_y: used_margin_px(style, "margin-bottom"),
            margin_far_x: used_margin_px(style, "margin-right"),
            margin_near_y: used_margin_px(style, "margin-top"),
            margin_near_x: used_margin_px(style, "margin-left"),
            // Resolved border widths of the constraining scroller's box
            // (the fold models in the border-box frame against the
            // padding-box scrollport extent). `None` when any side is
            // unresolvable app-side — the fold then stays conservative.
            scroller_border: self.containers[scroller]
                .node
                .and_then(|n| find_box(root, n))
                .and_then(|scroller_box| {
                    let s = scroller_box.style.as_deref()?;
                    Some((
                        used_border_px(s, "border-top-width", "border-top-style")?,
                        used_border_px(s, "border-right-width", "border-right-style")?,
                        used_border_px(s, "border-bottom-width", "border-bottom-style")?,
                        used_border_px(s, "border-left-width", "border-left-style")?,
                    ))
                }),
            // Resolved paddings of the constraining scroller's box
            // (the fold's near-padding offset per axis). `None` when
            // any side is unresolvable app-side — the fold then stays
            // conservative. Same lookup as the border above.
            scroller_padding: self.containers[scroller]
                .node
                .and_then(|n| find_box(root, n))
                .and_then(|scroller_box| {
                    let s = scroller_box.style.as_deref()?;
                    Some((
                        strict_px(s.get("padding-top"))?,
                        strict_px(s.get("padding-right"))?,
                        strict_px(s.get("padding-bottom"))?,
                        strict_px(s.get("padding-left"))?,
                    ))
                }),
        })
    }

    /// Exact extra shift for one band axis at the new offset `o1`, or
    /// `None` when the shift is not exactly foldable (caller re-bakes).
    ///
    /// `pos`/`bake` are the baked scrollport-relative position and the
    /// bake-time offset; `size`/`sp` the box and scrollport extents;
    /// `near`/`far` the resolved near/far insets (`None` = no
    /// constraint on that side); `near_pad` the scroller's near
    /// padding on this axis; `marg_far` the used far margin in px
    /// (`None` = not a simple length, so the block end is unprovable);
    /// `cb_is_scroller` whether the containing block is the scroller's
    /// own box; `cb_end` the block's exact band-frame content-box end
    /// (`Some(INFINITY)` when indefinite, `None` when unprovable),
    /// used only when `cb_is_scroller`; `cb_hi` the block's band-frame
    /// border-box end (conservative upper bound), used only for foreign
    /// blocks.
    ///
    /// Frame note: `pos` is border-box-frame (baked rect minus the
    /// scroller's border-box origin), but layout's pins are measured
    /// from the scroller's content origin, which sits `near_pad`
    /// inside the border-box frame (borders are zero for every
    /// runnable band, so no border term appears). The near pin is
    /// therefore `near_pad + o1 + near` and the far pin
    /// `near_pad + o1 + sp - far - size` in this frame, and every
    /// detection, guard, and free-interval bound below carries that
    /// offset — except where it cancels: the block-start guards
    /// compare a pin against the block start, which is itself the
    /// content origin, so `near_pad` drops out of those two.
    ///
    /// A box stuck to the near edge at bake time rides that edge, so
    /// its extra shift is exactly the scroll delta — but only forward:
    /// scrolling back unsticks it at its natural position, which is
    /// unrecoverable app-side (the baked rect already includes the
    /// sticky shift), so backward motion re-bakes. Forward motion stays
    /// stuck until the containing-block end pushes the box off; with
    /// the scroller's own content as the block and a provable end that
    /// edge is exactly `cb_end - size - near - marg_far - near_pad`,
    /// otherwise it is unprovable and the fold bails. The far edge is
    /// symmetric (backward only). A box within float slack of the edge
    /// at bake time counts as stuck: misclassification that close errs
    /// by at most the slack, transiently.
    ///
    /// A box at the block end with the scroller's own content as the
    /// block is either stuck-but-clamped (the bake offset is past the
    /// clamp point, so the scrollport clamp would place the box past
    /// the block — the stuck checks miss it because clamping masks the
    /// stick position) or truly free at the end. The clamped box is
    /// pinned going forward and transitions going back; the free one
    /// uses the free interval.
    ///
    /// A free box needs no extra shift while the offset stays clear of
    /// both stick transitions: `[pos - near_pad + size + far - sp,
    /// pos - near_pad - near]`, shrunk by slack so the transition
    /// itself re-bakes. The interval is exact for a truly free box, and
    /// sound for a block-start-clamped one (there the interval's upper
    /// bound is exactly the unclamp point, and it only errs
    /// conservative going back). Within slack of a foreign block's end
    /// the box may be block-end-clamped — undetectable without the
    /// exact end — so that zone re-bakes.
    ///
    /// The stuck fold is band-independent: the plateau provably
    /// outlasts the band, so a box tracked past the band edge keeps
    /// folding instead of re-baking (and past the provable edge the
    /// gate re-bakes, fixing the old outrun-the-band lag).
    #[allow(clippy::too_many_arguments)]
    fn sticky_axis_fold(
        pos: f32,
        bake: f32,
        o1: f32,
        size: f32,
        sp: f32,
        near: Option<f32>,
        far: Option<f32>,
        near_pad: f32,
        marg_far: Option<f32>,
        cb_is_scroller: bool,
        cb_end: Option<f32>,
        cb_hi: f32,
    ) -> Option<f32> {
        if let Some(t) = near
            && (pos - (bake + near_pad + t)).abs() <= STICKY_BAND_SLACK
        {
            if o1 < bake - STICKY_BAND_SLACK {
                return None;
            }
            if cb_is_scroller {
                match (cb_end, marg_far) {
                    (Some(e), Some(m)) => {
                        // Already at the block end: pinned going
                        // forward (the end pushes back), not riding.
                        if pos >= e - size - m - STICKY_BAND_SLACK {
                            return Some(0.0);
                        }
                        // The near pin in this frame is
                        // `near_pad + o1 + t`; the block-end pin is
                        // `e - size - m`.
                        if o1 > e - size - t - m - near_pad + STICKY_BAND_SLACK {
                            return None;
                        }
                    }
                    // Unprovable block end (or unknown far margin):
                    // only the no-op fold is safe going forward.
                    _ => {
                        if o1 > bake + STICKY_BAND_SLACK {
                            return None;
                        }
                    }
                }
            } else if o1 > bake + STICKY_BAND_SLACK {
                return None;
            }
            return Some(o1 - bake);
        }
        // Far-edge stuck: symmetric. Impossible when the scrollport
        // clamp is over-constrained (`near + far + size > sp`): layout
        // resolves that start-wins, so the near edge always wins.
        // Skipped at the block start (pos ~ 0 with the scroller's own
        // content as the block): there the box may be natural-clamped
        // rather than far-stuck, and the start-clamped logic below
        // handles it exactly.
        let far_wins = match (near, far) {
            (Some(t), Some(b)) => t + b + size <= sp,
            _ => true,
        };
        // The block start in this frame is the content origin,
        // `near_pad` (borders are zero for runnable bands), not 0.
        let at_block_start = cb_is_scroller && pos <= near_pad + STICKY_BAND_SLACK;
        if !at_block_start
            && far_wins
            && let Some(b) = far
            && (pos - (bake + near_pad + sp - b - size)).abs() <= STICKY_BAND_SLACK
        {
            if o1 > bake + STICKY_BAND_SLACK {
                return None;
            }
            if cb_is_scroller {
                // The block's start is the scrollport origin here: the
                // far pin (`near_pad + o1 + sp - b - size`) crosses it
                // when `o1 = b + size - sp`, with `near_pad` cancelling
                // against the start. The guard errs toward re-baking
                // (no near-margin credit).
                if o1 < b + size - sp - STICKY_BAND_SLACK {
                    return None;
                }
            } else if o1 < bake - STICKY_BAND_SLACK {
                return None;
            }
            return Some(o1 - bake);
        }
        // At the block end with the scroller's own content as the
        // block (exact end): stuck-but-clamped vs truly free. Needs a
        // known far margin and a provable end.
        if cb_is_scroller {
            if let (Some(e), Some(m)) = (cb_end, marg_far) {
                let end = e - size - m;
                if (pos - end).abs() <= STICKY_BAND_SLACK {
                    // Stuck-but-clamped when the near pin at bake
                    // (`near_pad + bake + t`) already passes the block
                    // end.
                    if near.is_some_and(|t| bake > end - t - near_pad + STICKY_BAND_SLACK) {
                        // Stuck-but-clamped: pinned to the block end
                        // going forward, transition going back.
                        return if o1 >= bake - STICKY_BAND_SLACK {
                            Some(0.0)
                        } else {
                            None
                        };
                    }
                    // Truly free at the block end: the free interval is
                    // exact (the near-stick transition is at
                    // `pos - near_pad - near`).
                    return Self::sticky_free_fold(pos, o1, size, sp, near, far, near_pad);
                }
            }
            // Clamped to the block start: the far-stick position at
            // bake is below the start, so the box is pinned at the
            // content origin (`near_pad` in this frame). It stays
            // pinned until the far-stick rises above the start (at
            // `far + size - sp`, with `near_pad` cancelling against
            // the start, as in the far-stuck guard above). Any other
            // box at ~`near_pad` is ambiguous (far-stuck vs
            // natural-clamped need different folds), so it re-bakes
            // except for the no-op.
            // (Near-side start-clamp only happens at bake=0,t=0, which
            // the near-stuck check already classified.)
            if pos <= near_pad + STICKY_BAND_SLACK {
                if let Some(b) = far
                    && bake + sp - b - size < -STICKY_BAND_SLACK
                {
                    return if o1 <= b + size - sp + STICKY_BAND_SLACK {
                        Some(0.0)
                    } else {
                        None
                    };
                }
                return if (o1 - bake).abs() <= STICKY_BAND_SLACK {
                    Some(0.0)
                } else {
                    None
                };
            }
            // Strictly inside the block (or end unprovable): the free
            // interval below.
            return Self::sticky_free_fold(pos, o1, size, sp, near, far, near_pad);
        }
        // Foreign block: within slack of the block end the box may be
        // block-clamped (undetectable — the stored end is only an
        // upper bound), so that zone re-bakes.
        if pos >= cb_hi - size - STICKY_BAND_SLACK {
            return None;
        }
        Self::sticky_free_fold(pos, o1, size, sp, near, far, near_pad)
    }

    /// Zero shift for a free box while the offset stays inside the
    /// provable no-transition interval `[pos - near_pad + size + far -
    /// sp, pos - near_pad - near]` (unbounded on a side with no inset),
    /// shrunk by slack so the transition itself re-bakes; `None`
    /// outside it. The `near_pad` terms are the content-origin offset:
    /// layout's stick transitions are at `pos - near_pad - near` and
    /// `pos - near_pad + size + far - sp` in this frame, not at the
    /// unpadded positions (w79fix: the old interval was a full
    /// near-padding too high on both ends, so a padded scroller's free
    /// box folded zero shift past its near transition).
    fn sticky_free_fold(
        pos: f32,
        o1: f32,
        size: f32,
        sp: f32,
        near: Option<f32>,
        far: Option<f32>,
        near_pad: f32,
    ) -> Option<f32> {
        let lo = far.map_or(f32::NEG_INFINITY, |b| pos - near_pad + size + b - sp);
        let hi = near.map_or(f32::INFINITY, |t| pos - near_pad - t);
        if o1 >= lo + STICKY_BAND_SLACK && o1 <= hi - STICKY_BAND_SLACK {
            Some(0.0)
        } else {
            None
        }
    }

    /// Exact `(dx, dy)` extra shift for one band at a new scroller
    /// offset, or `None` when any axis needs a re-bake.
    fn sticky_band_fold(&self, b: &StickyBand, x: f32, y: f32) -> Option<(f32, f32)> {
        // Border-box vs margin-box/padding-box: layout sticks the margin
        // box against the scroller's padding box, and the fold below
        // models border-box positions in the border-box frame against
        // that same padding-box scrollport extent (w79: layout's
        // `sticky_scrollport_axis` size term is the padding-box size
        // per CSS Overflow 3 §2.3, and the fold's `sp` must match
        // it). The container's stored scrollport rect is its border
        // box, so the padding-box extent is derived here by
        // subtracting the scroller's borders. When the fold runs
        // those borders are zero (any nonzero — or unresolvable —
        // border bails to re-bake just below), so the derivation is
        // the identity for every runnable band and only makes the
        // model explicit; it does not change any folded value.
        // Matching the size term alone is not matching layout: layout's
        // pins are measured from the content origin, a near padding
        // inside the border-box frame, so the fold offsets every pin
        // and transition by the scroller's near padding per axis
        // (w79fix: the w79 claim that the fold matched layout was
        // false for padded scrollers, whose stuck boxes folded
        // through the free interval instead). Any nonzero (or
        // unresolvable) used margin on the box, any nonzero (or
        // unresolvable) border on the scroller, or any unresolvable
        // scroller padding makes the model unsound (sawtooth stutter
        // + layout thrash) — fall back to re-bake.
        if b.margins_unsound() || b.scroller_borders_unsound() || b.scroller_padding_unsound() {
            return None;
        }
        let top = sticky_inset_opt(b.top)?;
        let bottom = sticky_inset_opt(b.bottom)?;
        let left = sticky_inset_opt(b.left)?;
        let right = sticky_inset_opt(b.right)?;
        let c = &self.containers[b.scroller];
        let sp = c.scrollport;
        let (sp_w, sp_h) = match b.scroller_border {
            Some((t, r, bb, l)) => ((sp.w - l - r).max(0.0), (sp.h - t - bb).max(0.0)),
            // Unreachable: `scroller_borders_unsound` above already
            // bailed for `None` — kept as a defensive re-bake.
            None => return None,
        };
        let (pad_t, _pad_r, _pad_b, pad_l) = b.scroller_padding?;
        let dx = Self::sticky_axis_fold(
            b.pos_x,
            b.bake_x,
            x,
            b.w,
            sp_w,
            left,
            right,
            pad_l,
            b.margin_far_x,
            b.cb_is_scroller,
            b.cb_end_x,
            b.cb_hi_x,
        )?;
        let dy = Self::sticky_axis_fold(
            b.pos_y,
            b.bake_y,
            y,
            b.h,
            sp_h,
            top,
            bottom,
            pad_t,
            b.margin_far_y,
            b.cb_is_scroller,
            b.cb_end_y,
            b.cb_hi_y,
        )?;
        Some((dx, dy))
    }

    /// Collect one [`StickyBand`] per `position: sticky` box that takes
    /// scroll-driven constraints.
    ///
    /// Iterative: the tree can be arbitrarily deep. Mirrors
    /// `bowser_layout`'s sticky pass: a sticky box under a `fixed`
    /// ancestor takes no scroll-driven constraint (see
    /// `has_fixed_ancestor` there), so its whole subtree contributes
    /// no bands. Each box constrains against its nearest ancestor
    /// scroll container (layout's `sticky_scroller`), tracked through
    /// the tree index; the viewport is container 0. The nearest
    /// containing-block candidate ancestor is tracked alongside (see
    /// [`sticky_cb_candidate`]) so the fold knows when it is the
    /// constraining scroller's own box.
    fn collect_sticky_bands(
        &self,
        root: &LayoutBox,
        viewport: bowser_layout::Size,
    ) -> Vec<StickyBand> {
        let mut bands = Vec::new();
        // (box, constraining scroller, fixed ancestor above,
        //  nearest containing-block candidate ancestor)
        let mut stack = vec![(root, 0usize, false, None)];
        while let Some((box_, scroller, fixed_above, cb)) = stack.pop() {
            if fixed_above {
                continue;
            }
            let pos = box_
                .style
                .as_deref()
                .map(position_of)
                .unwrap_or(Position::Static);
            let mut child_scroller = scroller;
            let mut child_fixed = false;
            let child_cb: Option<&LayoutBox> = if sticky_cb_candidate(box_) {
                Some(box_)
            } else {
                cb
            };
            match pos {
                Position::Sticky => {
                    if let Some(band) = self.sticky_band_for(box_, root, scroller, viewport, cb) {
                        bands.push(band);
                    }
                }
                Position::Fixed => child_fixed = true,
                _ => {}
            }
            // The box itself still constrains against the outer
            // scroller; only its descendants see the new one.
            if let Some(node) = box_.node
                && let Some(&ci) = self.index.get(&node)
            {
                child_scroller = ci;
            }
            stack.extend(
                box_.children
                    .iter()
                    .map(|c| (c, child_scroller, child_fixed, child_cb)),
            );
        }
        bands
    }
}

// ---------------------------------------------------------------------------
// Scroll snap
// ---------------------------------------------------------------------------

/// Snap evaluations ignore 2D movement below this: sub-pixel jitter is
/// not a scroll worth snapping after (the same 1/4096px quantum the old
/// chain dust-kill used per axis).
const SNAP_EPS: f32 = 1.0 / 4096.0;

/// Whether a `scroll-snap-type` snaps unconditionally (vs.
/// proximity-gated).
fn snap_is_mandatory(t: SnapType) -> bool {
    matches!(
        t,
        SnapType::MandatoryX | SnapType::MandatoryY | SnapType::MandatoryBoth
    )
}

/// Whether a `scroll-snap-type` snaps the x axis.
fn snap_snaps_x(t: SnapType) -> bool {
    matches!(
        t,
        SnapType::ProximityX
            | SnapType::MandatoryX
            | SnapType::ProximityBoth
            | SnapType::MandatoryBoth
    )
}

/// Whether a `scroll-snap-type` snaps the y axis.
fn snap_snaps_y(t: SnapType) -> bool {
    matches!(
        t,
        SnapType::ProximityY
            | SnapType::MandatoryY
            | SnapType::ProximityBoth
            | SnapType::MandatoryBoth
    )
}

/// A snap container's target pair in logical axes (CSS Scroll Snap 2
/// §5): the snap target element of the block axis and of the inline
/// axis, each `None` when that axis is unsnapped (proximity miss, axis
/// uncovered by the snap type, or no snap type at all). The payload of
/// the `scrollsnapchanging` / `scrollsnapchange` events
/// (`SnapEvent.snapTargetBlock` / `snapTargetInline`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct SnapTargets {
    /// Snap target of the block axis, if any.
    pub block: Option<NodeId>,
    /// Snap target of the inline axis, if any.
    pub inline: Option<NodeId>,
}

/// Which snap event a [`SnapEventDelivery`] carries.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SnapEventKind {
    /// `scrollsnapchanging`: the eventual target pair changed during
    /// a scrolling operation (or a layout snap).
    Changing,
    /// `scrollsnapchange`: the settled target pair changed at scroll
    /// completion (or a layout snap).
    Change,
}

impl SnapEventKind {
    /// The DOM event type string.
    pub fn as_str(self) -> &'static str {
        match self {
            SnapEventKind::Changing => "scrollsnapchanging",
            SnapEventKind::Change => "scrollsnapchange",
        }
    }
}

/// One queued snap-event delivery: the container it fires at (`None`
/// = the viewport scroller, whose events dispatch at the `Document`),
/// the event kind, and the target pair captured when the pair changed
/// (CSS Scroll Snap 2 §5; a target removed between queueing and
/// dispatch is still reported — see `dispatch_snap_event`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SnapEventDelivery {
    /// Container node; `None` for the viewport scroller.
    pub container: Option<NodeId>,
    /// Which event to fire.
    pub kind: SnapEventKind,
    /// The target pair the event reports.
    pub targets: SnapTargets,
}

impl ScrollContainer {
    /// The container's current snap target pair (logical axes): the
    /// per-axis [`SnapRest`] node ids of [`ScrollContainer::snap_rest`],
    /// mapped from physical (x, y) through
    /// [`ScrollContainer::block_is_y`].
    pub fn snap_targets(&self) -> SnapTargets {
        let (rx, ry) = self.snap_rest.get();
        self.targets_from_rests(rx, ry)
    }

    /// The target pair a pair of per-axis rests (physical x, y) names.
    fn targets_from_rests(&self, rx: Option<SnapRest>, ry: Option<SnapRest>) -> SnapTargets {
        let x = rx.and_then(|r| r.node);
        let y = ry.and_then(|r| r.node);
        if self.block_is_y {
            SnapTargets {
                block: y,
                inline: x,
            }
        } else {
            SnapTargets {
                block: x,
                inline: y,
            }
        }
    }
}

/// A container's resolved scroll-snap configuration: `scroll-snap-type`
/// and `scroll-padding`. The padding doubles as the reveal inset
/// (finding #9), so it is resolved once and shared. `scroll-snap-stop`
/// is NOT part of this configuration: it is a property of each snap
/// *area* (the target element), carried per area in [`SnapAreaInfo`].
#[derive(Clone, Copy, Debug)]
struct SnapConfig {
    snap_type: SnapType,
    /// `(top, right, bottom, left)` in px.
    padding: (f32, f32, f32, f32),
}

impl SnapConfig {
    fn none() -> Self {
        SnapConfig {
            snap_type: SnapType::None,
            padding: (0.0, 0.0, 0.0, 0.0),
        }
    }
}

/// One snap area of a container: the layout query's rect and per-axis
/// alignment, plus the target element's own `scroll-snap-stop`.
/// [`snap_areas_for_engine`] recovers the stop — and the target's node
/// id, in `node` — with its own mirror walk of the container subtree,
/// matching each surviving query area against the participating target
/// boxes (page-frame rect and alignment). The walk shadow-retargets at
/// collection (`box_.node.map(|n| doc.shadow_host_for(n).unwrap_or(n))`),
/// so the snap event reports the host element, never a node inside a
/// shadow tree. `node` is `None` only when no participating target
/// matched (the area keeps the default stop and no identity).
///
/// (The layout query's own [`bowser_layout::SnapArea::target`] field is
/// not read here; it is unused in `bowser_app`.)
///
/// `rev_x` records that the target's inline/block progression on the
/// x axis runs right-to-left (`direction: rtl` in a horizontal writing
/// mode, or a `vertical-rl`/`sideways-rl` writing mode): the query's
/// `align_x` is the *logical* value, so the physical start/end edges
/// swap when [`ScrollTree::eval_snap`] turns it into an offset.
#[derive(Clone, Copy, Debug)]
struct SnapAreaInfo {
    rect: Rect,
    align_x: SnapAlign,
    align_y: SnapAlign,
    stop: SnapStop,
    rev_x: bool,
    node: Option<NodeId>,
}

/// One snap candidate on one axis (CSS Scroll Snap 1 §4): the scroll
/// offset that aligns a snap area per its `scroll-snap-align`, the
/// area's `scroll-snap-stop`, and the range of valid resting offsets
/// when the area covers the snapport on this axis (snap overflow, §4:
/// every offset in `[cover_lo, cover_hi]` is a valid snap position).
/// For a normal-sized area `cover_lo == cover_hi == pos`.
#[derive(Clone, Copy, Debug)]
struct SnapCandidate {
    pos: f32,
    always: bool,
    cover_lo: f32,
    cover_hi: f32,
}

/// A container's snapped state on one axis, recorded by
/// [`ScrollTree::eval_snap`] and consumed by
/// [`ScrollTree::resnap_after_relayout`]: which snap area the
/// container snapped to and the snap position (scroll offset) it
/// rests at. The area is keyed on its target's node id
/// ([`SnapAreaInfo::node`], recovered by
/// [`snap_areas_for_engine`]) whenever both the recorded rest and
/// the candidate area carry one: node identity survives the target
/// being removed/reordered among same-size, same-stop siblings,
/// where the older index+size+stop key silently followed whatever
/// area shifted into the recorded index. The index, size and stop
/// are kept as the fallback key for areas whose target box could
/// not be recovered (`node: None` on either side). Positions and
/// alignments are never part of the key: they legitimately change
/// across a relayout, and following them is what resnap does.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SnapRest {
    area: usize,
    node: Option<NodeId>,
    size: (f32, f32),
    stop: SnapStop,
    pos: f32,
}

impl SnapRest {
    /// Record the snap of `area` (index into `areas`) at `pos`.
    fn of(areas: &[SnapAreaInfo], area: usize, pos: f32) -> Self {
        let a = &areas[area];
        SnapRest {
            area,
            node: a.node,
            size: (a.rect.w, a.rect.h),
            stop: a.stop,
            pos,
        }
    }

    /// Whether `area` is this rest's target area in a rebuilt area
    /// list. With node ids on both sides the node alone decides (a
    /// different node at the recorded index is a different target,
    /// however similar its size and stop); without them, fall back
    /// to the index+size+stop key (a target that was removed shifts
    /// a differently sized/stopped area into its index, which reads
    /// as vanished).
    fn matches(&self, areas: &[SnapAreaInfo], area: usize) -> bool {
        let Some(a) = areas.get(area) else {
            return false;
        };
        if let (Some(node), Some(area_node)) = (self.node, a.node) {
            return node == area_node;
        }
        area == self.area
            && a.rect.w == self.size.0
            && a.rect.h == self.size.1
            && a.stop == self.stop
    }
}

/// One axis's snap candidates, each tagged with its snap area's
/// index in the container's `snap_areas` (see [`snap_candidates`]).
type SnapCandidates = Vec<(usize, SnapCandidate)>;

/// The point of candidate `cd` nearest `v`: its snap position, or
/// the edge of its cover range when `v` lies outside it (an offset
/// inside the range maps to itself — every such offset is valid).
fn candidate_nearest(cd: &SnapCandidate, v: f32) -> f32 {
    v.clamp(cd.cover_lo, cd.cover_hi)
}

/// Clamp a snap position into a container's reachable `[min, max]`
/// range on one axis (degenerate-safe, mirroring the clamp
/// [`ScrollTree::eval_snap`] applies before recording a rest): a
/// candidate past the range's end is an unreachable snap position,
/// and the container can only rest at the range bound.
fn clamp_snap_pos(v: f32, range: (f32, f32)) -> f32 {
    let lo = range.0.min(range.1);
    if v.is_nan() { lo } else { v.clamp(lo, range.1) }
}

/// The `scroll-padding` longhands in px as `(top, right, bottom, left)`,
/// for the snapport and reveal alignment (finding #9, task a).
/// Percentages resolve against the scrollport's corresponding dimension
/// (CSS Scroll Snap §2); `auto` and missing longhands are zero;
/// negatives clamp to zero. `rem`/`em`/`vw`/`vh` resolve like
/// [`scroll_margins`].
fn scroll_padding(
    style: Option<&bowser_css::ComputedStyle>,
    root_font_size: f32,
    viewport_w: f32,
    viewport_h: f32,
    scrollport_w: f32,
    scrollport_h: f32,
) -> (f32, f32, f32, f32) {
    let Some(style) = style else {
        return (0.0, 0.0, 0.0, 0.0);
    };
    // `em` basis: the element's own computed font size when it is a
    // resolved px length; the root font size otherwise (same caveat as
    // `scroll_margins`).
    let em_basis = px_font_size(style, root_font_size);
    let len = |prop: &str, pct_base: f32| -> f32 {
        match style.get(prop) {
            Some(bowser_css::CssValue::Length(v, unit)) => {
                bowser_css::length_to_px(*v, unit, em_basis, root_font_size, viewport_w, viewport_h)
                    .max(0.0)
            }
            Some(bowser_css::CssValue::Percentage(p)) => (p / 100.0 * pct_base).max(0.0),
            // `auto` and the not-yet-parsed property: zero.
            _ => 0.0,
        }
    };
    (
        len("scroll-padding-top", scrollport_h),
        len("scroll-padding-right", scrollport_w),
        len("scroll-padding-bottom", scrollport_h),
        len("scroll-padding-left", scrollport_w),
    )
}

/// First layout box built for `node`, iterative (finding #4).
fn find_box(root: &LayoutBox, node: NodeId) -> Option<&LayoutBox> {
    let mut stack = vec![root];
    while let Some(box_) = stack.pop() {
        if box_.node == Some(node) {
            return Some(box_);
        }
        stack.extend(box_.children.iter());
    }
    None
}

/// Resolve an absolute length to px, mirroring the private helper of the
/// same name inside `bowser_layout::snap_areas_for_container` (used here
/// only to recover the container's content origin in exactly the frame
/// the query's rects use): absolute units fold to px per CSS Values 4
/// §5.2, and everything else — relative units, percentages, `auto` —
/// resolves to zero.
fn abs_len_px(value: Option<&bowser_css::CssValue>) -> f32 {
    match value {
        Some(bowser_css::CssValue::Length(v, unit)) => match unit {
            bowser_css::LengthUnit::Px => *v,
            bowser_css::LengthUnit::Cm => *v * 96.0 / 2.54,
            bowser_css::LengthUnit::Mm => *v * 96.0 / 25.4,
            bowser_css::LengthUnit::In => *v * 96.0,
            bowser_css::LengthUnit::Pt => *v * 96.0 / 72.0,
            bowser_css::LengthUnit::Pc => *v * 16.0,
            bowser_css::LengthUnit::Q => *v * 96.0 / 101.6,
            _ => 0.0,
        },
        _ => 0.0,
    }
}

/// Approximate float equality for snap-area rect matching across query
/// frames: the two page-frame rects derive from the same box rect and
/// scroll-margin, so they agree up to a few ulps of the origin
/// round-trips.
fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-3 * a.abs().max(b.abs()).max(1.0)
}

/// Snap areas for `container_node` from the real layout query
/// ([`bowser_layout::snap_areas_for_container`]), adapted to the snap
/// engine's contract:
///
/// * Out-of-flow skipping and `scroll-margin` expansion come from the
///   query itself: `position: fixed` subtrees never participate, an
///   absolutely positioned box only participates when its containing
///   block is inside the container, and each rect is the border box
///   expanded by its `scroll-margin`, in the container's content
///   coordinates. (The query resolves relative-unit `scroll-margin`s to
///   zero — no font/viewport context at the query — where the old
///   stand-in resolved `em`/`rem`/`vw`/`vh`.)
/// * Axis realignment: the query maps `scroll_snap_align()`'s
///   `(block, inline)` pair onto `(align_x, align_y)` per the target's
///   own writing mode, but the values stay *logical*: whether the x
///   progression runs right-to-left (recorded per area as
///   [`SnapAreaInfo::rev_x`]) decides if start/end swap physical edges
///   in [`ScrollTree::eval_snap`].
/// * Per-area `scroll-snap-stop` and node id: the query reports no
///   node ids, so each surviving area is matched against the target
///   boxes of the container subtree (page-frame rect and alignment,
///   like the pruning pass) and takes its target's
///   `scroll-snap-stop` and node id (`Normal` / `None` when nothing
///   matches). The match set is restricted to boxes that actually
///   participate as snap areas of THIS container — the query's
///   fixed/abspos participation rules, and no descent into nested
///   containers (their own boxes participate here; their subtrees
///   belong to them) — so a coincident non-participating box cannot
///   donate its stop or reversal flag to this container's area.
///
/// One deliberate difference from the old stand-in: the stand-in pruned
/// a nested container's *own box* from the outer container too, losing
/// the area entirely (no query ever claims a container's own box). Here
/// it stays with the outer container, where it scrolls — matching the
/// query's "in-flow descendants" contract.
/// A container's `scroll-initial-target` position (CSS Scroll Snap 2
/// §3.1.2): among the container's descendants with
/// `scroll-initial-target: nearest` — skipping `position: fixed`
/// subtrees and nested containers' interiors, like the snap-area
/// walk — the target whose aligned offset is nearest the container's
/// initial offset (`origin`) wins, ties keep document order, and the
/// winning offset, clamped into `[origin, max]`, is the container's
/// initial scroll position. Alignment follows the target's
/// `scroll-snap-align`, defaulting to `start` on an axis that names
/// none (the spec's default initial alignment). Rects convert to
/// container content coordinates through the container's content
/// origin (the same border+padding recovery [`snap_areas_for_engine`]
/// performs), so the per-axis positions are the snap candidate
/// positions of [`snap_candidates`] verbatim. The production mirror
/// of the parse shim's `parse_initial_target` in bowser_js.
fn initial_target_for(
    root: &LayoutBox,
    container_node: NodeId,
    containers: &HashSet<NodeId>,
    scrollport: Rect,
    padding: (f32, f32, f32, f32),
    origin: (f32, f32),
    max: (f32, f32),
) -> Option<(f32, f32)> {
    let cbox = find_box(root, container_node)?;
    let (pad_t, pad_r, pad_b, pad_l) = padding;
    let spw = (scrollport.w - pad_l - pad_r).max(0.0);
    let sph = (scrollport.h - pad_t - pad_b).max(0.0);
    let (ox, oy) = {
        let style = cbox.style.as_deref();
        let get = |prop: &str| style.and_then(|s| s.get(prop));
        (
            cbox.rect.x + abs_len_px(get("border-left-width")) + abs_len_px(get("padding-left")),
            cbox.rect.y + abs_len_px(get("border-top-width")) + abs_len_px(get("padding-top")),
        )
    };
    let clamp = |v: f32, lo: f32, hi: f32| {
        let lo = lo.min(hi);
        if v.is_nan() { lo } else { v.clamp(lo, hi) }
    };
    // One axis of the candidate position, mirroring `snap_candidates`'
    // closure (reversal swaps start/end before positioning).
    let cand = |align: SnapAlign,
                start: f32,
                size: f32,
                port: f32,
                sp: f32,
                pad_start: f32,
                pad_end: f32,
                rev: bool|
     -> Option<f32> {
        let align = if rev {
            match align {
                SnapAlign::Start => SnapAlign::End,
                SnapAlign::End => SnapAlign::Start,
                a => a,
            }
        } else {
            align
        };
        Some(match align {
            SnapAlign::Start => start - pad_start,
            SnapAlign::End => start + size - port + pad_end,
            SnapAlign::Center => start + size * 0.5 - pad_start - sp * 0.5,
            SnapAlign::None => return None,
        })
    };
    let mut cands: Vec<(f32, f32)> = Vec::new();
    let mut walk: Vec<&LayoutBox> = cbox.children.iter().collect();
    while let Some(box_) = walk.pop() {
        if box_
            .style
            .as_ref()
            .is_some_and(|s| position_of(s) == Position::Fixed)
        {
            continue;
        }
        if box_
            .node
            .is_some_and(|n| n != container_node && containers.contains(&n))
        {
            continue; // a nested container's subtree is its own
        }
        if let Some(style) = box_.style.as_deref()
            && style
                .get_str("scroll-initial-target")
                .is_some_and(|v| v.eq_ignore_ascii_case("nearest"))
        {
            let (block, inline) = style.scroll_snap_align();
            let wm = style.get_str("writing-mode");
            // Same axis mapping as `snap_areas_for_engine`'s stops
            // walk (and the layout query's): anything vertical swaps
            // the (block, inline) pair.
            let vertical = wm.is_some_and(|w| {
                w.eq_ignore_ascii_case("vertical-lr")
                    || w.eq_ignore_ascii_case("vertical-rl")
                    || w.eq_ignore_ascii_case("sideways-lr")
                    || w.eq_ignore_ascii_case("sideways-rl")
            });
            let (mut ax, mut ay) = if vertical {
                (block, inline)
            } else {
                (inline, block)
            };
            if ax == SnapAlign::None {
                ax = SnapAlign::Start;
            }
            if ay == SnapAlign::None {
                ay = SnapAlign::Start;
            }
            let rev_x = match wm {
                Some(w)
                    if w.eq_ignore_ascii_case("vertical-rl")
                        || w.eq_ignore_ascii_case("sideways-rl") =>
                {
                    true
                }
                Some(w)
                    if w.eq_ignore_ascii_case("vertical-lr")
                        || w.eq_ignore_ascii_case("sideways-lr") =>
                {
                    false
                }
                _ => style
                    .get_str("direction")
                    .is_some_and(|d| d.eq_ignore_ascii_case("rtl")),
            };
            let px = cand(
                ax,
                box_.rect.x - ox,
                box_.rect.w,
                scrollport.w,
                spw,
                pad_l,
                pad_r,
                rev_x,
            );
            let py = cand(
                ay,
                box_.rect.y - oy,
                box_.rect.h,
                scrollport.h,
                sph,
                pad_t,
                pad_b,
                false,
            );
            if let (Some(px), Some(py)) = (px, py) {
                cands.push((clamp(px, origin.0, max.0), clamp(py, origin.1, max.1)));
            }
        }
        walk.extend(box_.children.iter());
    }
    // The walk is stack order (reverse document order): iterate it
    // backwards so ties keep the earliest target in document order.
    let mut best: Option<((f32, f32), f32)> = None;
    for pos in cands.into_iter().rev() {
        let d2 = (pos.0 - origin.0).powi(2) + (pos.1 - origin.1).powi(2);
        if best.is_none_or(|(_, bd)| d2 < bd) {
            best = Some((pos, d2));
        }
    }
    best.map(|(pos, _)| pos)
}

fn snap_areas_for_engine(
    root: &LayoutTree,
    container_node: NodeId,
    containers: &HashSet<NodeId>,
    doc: &Document,
) -> Vec<SnapAreaInfo> {
    let Some(cbox) = find_box(root, container_node) else {
        return Vec::new();
    };
    let areas = snap_areas_for_container(root, container_node, doc);
    if areas.is_empty() {
        return Vec::new();
    }
    // The query reports rects in container content coordinates (border
    // box minus border and padding): recover a container's content
    // origin to compare areas in the shared page frame.
    let origin_of = |b: &LayoutBox| -> (f32, f32) {
        let style = b.style.as_deref();
        let get = |prop: &str| style.and_then(|s| s.get(prop));
        (
            b.rect.x + abs_len_px(get("border-left-width")) + abs_len_px(get("padding-left")),
            b.rect.y + abs_len_px(get("border-top-width")) + abs_len_px(get("padding-top")),
        )
    };
    // Outermost nested container boxes (`position: fixed` subtrees are
    // skipped — the query emits no areas from them).
    let mut nested: Vec<&LayoutBox> = Vec::new();
    let mut stack: Vec<&LayoutBox> = cbox.children.iter().collect();
    while let Some(box_) = stack.pop() {
        if box_
            .style
            .as_ref()
            .is_some_and(|s| position_of(s) == Position::Fixed)
        {
            continue;
        }
        if box_
            .node
            .is_some_and(|n| n != container_node && containers.contains(&n))
        {
            nested.push(box_);
            continue;
        }
        stack.extend(box_.children.iter());
    }
    // The targets' stops and node ids: (page-frame rect expanded by
    // `scroll-margin`, physical alignment) -> (`scroll-snap-stop`,
    // reversal flag, target node). The query maps
    // `scroll_snap_align()`'s `(block, inline)` pair onto physical axes
    // per the target's own writing mode; the same mapping is reproduced
    // here off the serialized `writing-mode` (vertical-* swaps the
    // pair, everything else keeps it — mirroring the query's
    // `HorizontalTb` vs not split).
    //
    // The walk collects only boxes that participate as snap areas of
    // THIS container, tracking containing-block membership exactly
    // like the query (`cb_inside` / `fixed_cb_inside`, seeded from
    // the container box — with the query's viewport exception: when the
    // container is the tree root box, the ICB counts as inside): a
    // viewport-anchored fixed subtree and an
    // abspos box whose containing block lies outside the container
    // are skipped (subtree included), and a nested container's own
    // box is collected but not descended into — its subtree's areas
    // belong to the nested container. The previous looser walk let a
    // coincident non-participating box (a nested container's child
    // stacked exactly on the outer target, an abspos box against an
    // outside containing block) donate its `scroll-snap-stop: always`
    // and reversal flag to this container's area.
    let is_viewport = std::ptr::eq(cbox, root);
    let container_cb = is_viewport
        || cbox
            .style
            .as_ref()
            .is_some_and(|s| position_of(s) != Position::Static);
    let mut stops: Vec<(Rect, SnapAlign, SnapAlign, SnapStop, bool, Option<NodeId>)> = Vec::new();
    let mut walk: Vec<(&LayoutBox, bool, bool)> = cbox
        .children
        .iter()
        .map(|c| (c, container_cb, false))
        .collect();
    while let Some((box_, cb_inside, fixed_cb_inside)) = walk.pop() {
        let pos = box_
            .style
            .as_ref()
            .map(|s| position_of(s))
            .unwrap_or(Position::Static);
        if pos == Position::Fixed && !fixed_cb_inside {
            continue;
        }
        if pos == Position::Absolute && !cb_inside {
            continue;
        }
        if let Some(style) = box_.style.as_deref() {
            let (block, inline) = style.scroll_snap_align();
            if block != SnapAlign::None || inline != SnapAlign::None {
                let wm = style.get_str("writing-mode");
                // Mirror the query's `HorizontalTb` vs not split:
                // anything vertical swaps the (block, inline) pair.
                let vertical = wm.is_some_and(|w| {
                    w.eq_ignore_ascii_case("vertical-lr")
                        || w.eq_ignore_ascii_case("vertical-rl")
                        || w.eq_ignore_ascii_case("sideways-lr")
                        || w.eq_ignore_ascii_case("sideways-rl")
                });
                let (align_x, align_y) = if vertical {
                    (block, inline)
                } else {
                    (inline, block)
                };
                // The query's `align_x` is logical: the physical
                // start/end edges swap when the x progression runs
                // right-to-left (`vertical-rl`/`sideways-rl`, or
                // `direction: rtl` in a horizontal writing mode).
                let rev_x = match wm {
                    Some(w)
                        if w.eq_ignore_ascii_case("vertical-rl")
                            || w.eq_ignore_ascii_case("sideways-rl") =>
                    {
                        true
                    }
                    Some(w)
                        if w.eq_ignore_ascii_case("vertical-lr")
                            || w.eq_ignore_ascii_case("sideways-lr") =>
                    {
                        false
                    }
                    _ => style
                        .get_str("direction")
                        .is_some_and(|d| d.eq_ignore_ascii_case("rtl")),
                };
                let get = |prop: &str| style.get(prop);
                // `sane_len` mirrors the query's own sanitization so
                // the rect compares exactly against the query's areas
                // (shift-invariant for sane inputs: NaN/huge clamps
                // only).
                stops.push((
                    Rect {
                        x: bowser_layout::sane_len(
                            box_.rect.x - abs_len_px(get("scroll-margin-left")),
                        ),
                        y: bowser_layout::sane_len(
                            box_.rect.y - abs_len_px(get("scroll-margin-top")),
                        ),
                        w: bowser_layout::sane_len(
                            box_.rect.w
                                + abs_len_px(get("scroll-margin-left"))
                                + abs_len_px(get("scroll-margin-right")),
                        ),
                        h: bowser_layout::sane_len(
                            box_.rect.h
                                + abs_len_px(get("scroll-margin-top"))
                                + abs_len_px(get("scroll-margin-bottom")),
                        ),
                    },
                    align_x,
                    align_y,
                    style.scroll_snap_stop(),
                    rev_x,
                    // Shadow-descendant snap areas retarget to the host:
                    // the snap event must report the host element, never
                    // a node inside a shadow tree.
                    box_.node.map(|n| doc.shadow_host_for(n).unwrap_or(n)),
                ));
            }
        }
        // A nested container's subtree is its own: its areas are
        // collected when this function runs for that container.
        if box_
            .node
            .is_some_and(|n| n != container_node && containers.contains(&n))
        {
            continue;
        }
        let child_cb_inside = pos != Position::Static
            || cb_inside
            || box_.style.as_deref().is_some_and(establishes_fixed_cb);
        let child_fixed_cb_inside =
            fixed_cb_inside || box_.style.as_deref().is_some_and(establishes_fixed_cb);
        walk.extend(
            box_.children
                .iter()
                .map(|c| (c, child_cb_inside, child_fixed_cb_inside)),
        );
    }
    let (ox, oy) = origin_of(cbox);
    // Every area the nested containers claim for themselves, in the
    // page frame: an area of this container matching one of these
    // (rect and alignment) belongs to the nested subtree, not here.
    // Widths/heights are computed identically by both queries, so they
    // compare exactly; origins round-trip through floats, hence
    // `approx_eq` on the positions.
    //
    // The removal below is one-for-one (each owned area removes at
    // most one of this container's areas): a nested container's own
    // box can share a rect and alignment with the first area inside
    // it (a child that exactly fills it), and an any-match filter
    // removed BOTH — losing the container's own box as an area of
    // this container entirely. With one-for-one removal one twin
    // survives, which is all the snap engine needs: their snap
    // positions agree.
    let mut owned: Vec<(Rect, SnapAlign, SnapAlign)> = Vec::new();
    for n in &nested {
        let Some(nn) = n.node else { continue };
        let (nox, noy) = origin_of(n);
        owned.extend(
            snap_areas_for_container(root, nn, doc)
                .into_iter()
                .map(|a| {
                    (
                        Rect {
                            x: a.rect.x + nox,
                            y: a.rect.y + noy,
                            w: a.rect.w,
                            h: a.rect.h,
                        },
                        a.align_x,
                        a.align_y,
                    )
                }),
        );
    }
    let mut areas = areas;
    for (r, ax, ay) in &owned {
        if let Some(i) = areas.iter().position(|a| {
            *ax == a.align_x
                && *ay == a.align_y
                && r.w == a.rect.w
                && r.h == a.rect.h
                && approx_eq(r.x, a.rect.x + ox)
                && approx_eq(r.y, a.rect.y + oy)
        }) {
            areas.remove(i);
        }
    }
    areas
        .into_iter()
        .map(|a| {
            let (px, py) = (a.rect.x + ox, a.rect.y + oy);
            // Prefer `Always` on conflicting matches (identical rect
            // and alignment from two participating targets): the trap
            // then errs toward stopping, the spec-faithful direction.
            // The reversal flag and the node id ride the same match.
            let matched = stops
                .iter()
                .filter(|(r, ax, ay, _, _, _)| {
                    *ax == a.align_x
                        && *ay == a.align_y
                        && r.w == a.rect.w
                        && r.h == a.rect.h
                        && approx_eq(r.x, px)
                        && approx_eq(r.y, py)
                })
                .max_by(|x, y| (x.3 == SnapStop::Always).cmp(&(y.3 == SnapStop::Always)));
            let (stop, rev_x, node) = match matched {
                Some((_, _, _, s, r, n)) => (*s, *r, *n),
                None => (SnapStop::Normal, false, None),
            };
            SnapAreaInfo {
                rect: a.rect,
                align_x: a.align_x,
                align_y: a.align_y,
                stop,
                rev_x,
                node,
            }
        })
        .collect()
}

/// The proximity snap reach for one axis: how far from a snap position
/// a `proximity` scroll may rest and still be pulled onto it. CSS Scroll
/// Snap 1 §5.2 leaves the reach UA-defined; one third of the snapport
/// extent matches Chromium/Gecko practice and is pinned by WPT: at a
/// 600px snapport a rest 190px from the target snaps but 250px does not
/// (`scrollsnapchange-with-proximity-strictness`), at a 400px snapport a
/// rest 375px away does not snap (`scroll-snap-type`), and a rest 250px
/// from the nearest position in a 600px snapport stays put
/// (`not-resnap-outside-proximity-threshold`). The pre-wave-50 code used
/// the full snapport extent, which snapped nearly every proximity rest.
fn proximity_threshold(snapport_extent: f32) -> f32 {
    snapport_extent / 3.0
}

/// Pick the snap target on one axis (CSS Scroll Snap 1 §4–§5.3): the
/// scroll offset that aligns a snap area per its `scroll-snap-align`.
///
/// `candidates` are the per-axis candidates (see [`ScrollTree::eval_snap`]
/// for how alignment maps to offsets), `natural` the resting offset the
/// scroll would end at, `pre` the offset before the scroll. `mandatory`
/// snaps to the nearest candidate unconditionally; proximity snaps only
/// within `threshold` (the proximity reach — see
/// [`proximity_threshold`]).
///
/// `scroll-snap-stop: always` on a relative scroll must not pass over a
/// snap position (CSS Scroll Snap §5.3): it snaps to the first
/// `always` position encountered from `pre` toward `natural` instead.
/// With no `always` position in the traversed interval, normal
/// selection applies — but an area larger than the snapport makes its
/// whole covering range valid (snap overflow, §4): a `natural` inside
/// some area's `[cover_lo, cover_hi]` wins before the nearest-position
/// rule. That range is only valid when the snap positions geometrically
/// adjacent to it are farther apart than the snapport (CSS Scroll Snap
/// 1 §5.2.2's second condition): a range wedged between snap positions
/// closer than the snapport would trap the scroll past them, so the
/// nearest ordinary snap position wins instead. (A degenerate
/// cover — `cover_lo == cover_hi` — is the area's own snap position,
/// not a range, and is always valid.) On non-relative scrolls
/// (`relative = false`) the always rule has no effect at all.
///
/// Note: `scrollsnapchange-after-layout-change.tentative.html` subtest
/// 1 expects a `scrollTo(0, 190)` to land at 200 ("just below the inner
/// snap area") with snap positions at 0 and 100 and a 200px snapport.
/// No standard formula yields 200 there — Chromium's EndPositionStrategy
/// also predicts 190 — so the test encodes aspirational behavior; the
/// spec-grounded §5.2.2 answer is 100 (the outer area's cover range is
/// invalid because the adjacent positions are 100px apart, under the
/// 200px snapport, and 100 is the nearest valid position).
///
/// Returns `None` when no snapping applies on this axis.
fn snap_target_on_axis(
    candidates: &[SnapCandidate],
    natural: f32,
    pre: f32,
    mandatory: bool,
    threshold: f32,
    relative: bool,
    snapport: f32,
) -> Option<f32> {
    if candidates.is_empty() {
        return None;
    }
    let nearer = |a: &f32, b: &f32, to: f32| {
        (*a - to)
            .abs()
            .partial_cmp(&(*b - to).abs())
            .unwrap_or(Ordering::Equal)
    };
    if relative {
        let (lo, hi) = if pre <= natural {
            (pre, natural)
        } else {
            (natural, pre)
        };
        // The starting point itself does not count as passed over —
        // otherwise a scroll starting exactly on a snap position could
        // never leave it.
        let first = candidates
            .iter()
            .filter(|c| c.always && (c.pos - pre).abs() > SNAP_EPS && c.pos >= lo && c.pos <= hi)
            .min_by(|a, b| nearer(&a.pos, &b.pos, pre));
        if let Some(c) = first {
            return Some(c.pos);
        }
    }
    // A resting offset the snapport covers entirely inside an area is a
    // valid snap position in its own right (snap overflow): the scroll
    // stays there instead of being dragged to a position — unless the
    // range fails §5.2.2's second condition (see the doc comment).
    let cover_valid = |c: &SnapCandidate| -> bool {
        if c.cover_hi - c.cover_lo <= SNAP_EPS {
            return true;
        }
        // The snap positions geometrically adjacent to the cover
        // range's start edge: the nearest at-or-before it (for a
        // start-aligned area, its own snap position) and the nearest
        // after it. A missing endpoint is no constraint (an infinite
        // gap): the range then runs to the scrollable edge, as in the
        // spec's "ranges taller than the viewport" example.
        let mut prev = f32::NEG_INFINITY;
        let mut next = f32::INFINITY;
        for p in candidates.iter().map(|o| o.pos) {
            if p <= c.cover_lo + SNAP_EPS && p > prev {
                prev = p;
            }
            if p > c.cover_lo + SNAP_EPS && p < next {
                next = p;
            }
        }
        next - prev > snapport
    };
    if candidates.iter().any(|c| {
        natural >= c.cover_lo - SNAP_EPS && natural <= c.cover_hi + SNAP_EPS && cover_valid(c)
    }) {
        return Some(natural);
    }
    candidates
        .iter()
        .map(|c| c.pos)
        .min_by(|a, b| nearer(a, b, natural))
        .filter(|&p| mandatory || (p - natural).abs() <= threshold)
}

/// The viewport's `scroll-snap-type`: the root element's value, falling
/// back to the body's when the root's is `none` — mirroring
/// `viewport_scroll_behavior`'s root→body convention (the initial
/// `none` is unobservable in the computed style, so an explicit `none`
/// on the root falls back just like an absent one).
fn viewport_snap_type(doc: &Document, root: &LayoutBox) -> SnapType {
    let root_t = root
        .style
        .as_deref()
        .map(|s| s.scroll_snap_type())
        .unwrap_or(SnapType::None);
    if root_t != SnapType::None {
        return root_t;
    }
    body_style(doc, root)
        .map(|s| s.scroll_snap_type())
        .unwrap_or(SnapType::None)
}

/// Per-axis snap candidates of a container, each tagged with its
/// snap area's index in `c.snap_areas`, plus the snapport extents
/// `(w, h)` (the scrollport inset by `scroll-padding`).
///
/// Snap positions are scroll offsets, independent of the current
/// one: areas are stored in the offset-0 content frame (page minus
/// scrollport origin), so aligning edges is pure arithmetic against
/// the snapport geometry. The snapport's edges in the scrollport's
/// own coordinates are `pad_l` / `scrollport.w - pad_r` (and likewise
/// vertically), so:
///   start:  area_start - pad_start
///   end:    area_end - scrollport + pad_end
///   center: area_mid - pad_start - snapport_extent / 2
/// An area larger than the snapport on an axis covers it over a
/// whole range of offsets: from the offset aligning the area's
/// start edge with the snapport's start edge (area_start -
/// pad_start) up to the offset aligning the area's end edge with
/// the snapport's end edge (area_end - sp - pad_start) — the
/// candidate's valid resting range (snap overflow). On a reversed
/// axis (`rev_x`: `direction: rtl` / `vertical-rl` targets) the
/// logical start/end swap physical edges; center and the cover
/// range are direction-invariant.
///
/// Shared by [`ScrollTree::eval_snap`] and
/// [`ScrollTree::resnap_after_relayout`], so a resnap selects from
/// exactly the positions a scroll would.
fn snap_candidates(c: &ScrollContainer) -> (SnapCandidates, SnapCandidates, (f32, f32)) {
    let (pad_t, pad_r, pad_b, pad_l) = c.snap.padding;
    let spw = (c.scrollport.w - pad_l - pad_r).max(0.0);
    let sph = (c.scrollport.h - pad_t - pad_b).max(0.0);
    let cand = |align: SnapAlign,
                start: f32,
                size: f32,
                port: f32,
                sp: f32,
                pad_start: f32,
                pad_end: f32,
                always: bool,
                rev: bool|
     -> Option<SnapCandidate> {
        let align = if rev {
            match align {
                SnapAlign::Start => SnapAlign::End,
                SnapAlign::End => SnapAlign::Start,
                a => a,
            }
        } else {
            align
        };
        let pos = match align {
            SnapAlign::Start => start - pad_start,
            SnapAlign::End => start + size - port + pad_end,
            SnapAlign::Center => start + size * 0.5 - pad_start - sp * 0.5,
            SnapAlign::None => return None,
        };
        let (cover_lo, cover_hi) = if size > sp {
            (start - pad_start, start + size - sp - pad_start)
        } else {
            (pos, pos)
        };
        Some(SnapCandidate {
            pos,
            always,
            cover_lo,
            cover_hi,
        })
    };
    let mut px = Vec::new();
    let mut py = Vec::new();
    for (i, a) in c.snap_areas.iter().enumerate() {
        let always = a.stop == SnapStop::Always;
        if let Some(cd) = cand(
            a.align_x,
            a.rect.x,
            a.rect.w,
            c.scrollport.w,
            spw,
            pad_l,
            pad_r,
            always,
            a.rev_x,
        ) {
            px.push((i, cd));
        }
        if let Some(cd) = cand(
            a.align_y,
            a.rect.y,
            a.rect.h,
            c.scrollport.h,
            sph,
            pad_t,
            pad_b,
            always,
            false,
        ) {
            py.push((i, cd));
        }
    }
    (px, py, (spw, sph))
}

/// The snap area (index into `snap_areas`) whose candidate selected
/// position `p` on one axis. A candidate with a proper cover range
/// (an area larger than the snapport, `cover_lo < cover_hi`) that
/// contains `p` wins first: resting inside its range is the
/// snap-overflow stay, and that covering area is the one the scroll
/// actually settled on — a different area whose point position
/// merely coincides with `p` inside the range did not win anything.
/// Otherwise the first candidate whose position is `p`, else the
/// first whose (point) cover range contains `p`, else the nearest
/// candidate. Several areas can share a position (aligned targets):
/// the earliest — in `snap_areas` order, i.e. tree order — wins,
/// matching the re-snap preference for the first target in tree
/// order.
fn snap_winner(cands: &[(usize, SnapCandidate)], p: f32) -> Option<usize> {
    if let Some(&(i, _)) = cands.iter().find(|(_, cd)| {
        cd.cover_lo < cd.cover_hi && p >= cd.cover_lo - SNAP_EPS && p <= cd.cover_hi + SNAP_EPS
    }) {
        return Some(i);
    }
    if let Some(&(i, _)) = cands.iter().find(|(_, cd)| cd.pos == p) {
        return Some(i);
    }
    if let Some(&(i, _)) = cands
        .iter()
        .find(|(_, cd)| p >= cd.cover_lo - SNAP_EPS && p <= cd.cover_hi + SNAP_EPS)
    {
        return Some(i);
    }
    cands
        .iter()
        .min_by(|a, b| {
            (a.1.pos - p)
                .abs()
                .partial_cmp(&(b.1.pos - p).abs())
                .unwrap_or(Ordering::Equal)
        })
        .map(|&(i, _)| i)
}

/// One axis of [`ScrollTree::resnap_after_relayout`]: given the
/// freshly built candidates, the snapped state carried from the old
/// tree, and the live offset, return the new snapped state and the
/// offset the axis should rest at (the live offset when nothing
/// moves). `mandatory` is the axis's snap strictness under the NEW
/// snap type; `threshold` is the proximity reach (see
/// [`proximity_threshold`]); `range` is the axis's reachable `[min, max]` offset
/// range, into which every returned position is clamped before it
/// is recorded in a [`SnapRest`].
fn resnap_axis(
    areas: &[SnapAreaInfo],
    cands: &[(usize, SnapCandidate)],
    carried: Option<SnapRest>,
    live: f32,
    mandatory: bool,
    threshold: f32,
    range: (f32, f32),
) -> (Option<SnapRest>, f32) {
    if cands.is_empty() {
        return (None, live);
    }
    // The container counts as snapped only when it still rests at
    // the carried position: a user scroll (or an in-flight snap
    // animation) that left it elsewhere has already superseded the
    // state, and resnap must not chase a stale target.
    let snapped = carried.filter(|r| (r.pos - live).abs() <= SNAP_EPS);
    // Follow the carried target while it still exists: move to the
    // point of its new valid range nearest the live offset — the
    // live offset itself when the target still covers it (so a pure
    // `scroll-snap-align` change on a covering target does not
    // resnap), its new snap position otherwise.
    if let Some(r) = snapped
        && let Some(&(ai, cd)) = cands.iter().find(|&&(ai, _)| r.matches(areas, ai))
    {
        let p = clamp_snap_pos(candidate_nearest(&cd, live), range);
        if !mandatory && (p - live).abs() > threshold {
            // Proximity: the target moved out of reach — unsnap
            // rather than chase it across the container.
            return (None, live);
        }
        return (Some(SnapRest::of(areas, ai, p)), p);
    }
    // Resting inside some candidate's valid range is valid as-is:
    // stay. The covering candidate becomes the tracked target when
    // the container is (or was) snapped, so a later relayout can
    // follow it.
    if let Some(&(ai, _)) = cands
        .iter()
        .find(|(_, cd)| live >= cd.cover_lo - SNAP_EPS && live <= cd.cover_hi + SNAP_EPS)
    {
        let rest = if mandatory || snapped.is_some() {
            Some(SnapRest::of(areas, ai, live))
        } else {
            None
        };
        return (rest, live);
    }
    // Snap to the surviving candidate nearest the resting position
    // (its valid-range point nearest it; first in tree order on
    // ties): `mandatory` always — its resting offset must be a snap
    // position, so a snap-type change that newly covers the axis and
    // a persisted non-snap offset both re-seat here — while
    // `proximity` only re-seats a container that was snapped (its
    // target vanished) and only within the threshold.
    if mandatory || snapped.is_some() {
        let anchor = snapped.map_or(live, |r| r.pos);
        let mut best: Option<(usize, f32, f32)> = None;
        for &(ai, cd) in cands {
            let p = candidate_nearest(&cd, anchor);
            let d = (p - anchor).abs();
            if best.is_none_or(|(_, _, bd)| d < bd) {
                best = Some((ai, p, d));
            }
        }
        if let Some((ai, p, _)) = best {
            // Clamp before recording the rest: with the unclamped
            // position stored, `rest.pos` could never equal the
            // (clamped) live offset, so every later relayout would
            // re-enter this fallback, reset `snap_base`, and cancel
            // smooth scrolls indefinitely.
            let p = clamp_snap_pos(p, range);
            if !mandatory && (p - live).abs() > threshold {
                return (None, live);
            }
            return (Some(SnapRest::of(areas, ai, p)), p);
        }
    }
    (None, live)
}

/// The outcome of one pure snap selection ([`ScrollTree::select_snap`]):
/// the per-axis rests the scroll settles on (physical x, y) and the
/// resting offset — the selection clamped into the container's
/// `[origin, max]` range.
struct SnapSelection {
    rest_x: Option<SnapRest>,
    rest_y: Option<SnapRest>,
    rest_offset: (f32, f32),
}

impl ScrollTree {
    /// Scroll-snap evaluation for container `idx` after a scroll that
    /// moved it from `pre` to `natural` (CSS Scroll Snap 1 §4–§5).
    ///
    /// Computes the snapport (the scrollport inset by `scroll-padding`)
    /// in container content coordinates, gathers each snap area's snap
    /// position per axis from its `scroll-snap-align` (start/end/center
    /// map to the offsets that align the area's edge or midpoint with
    /// the snapport's), and picks the target per
    /// [`snap_target_on_axis`]. When the target differs from the
    /// natural resting offset, a follow-up [`SmoothScroll`] is launched
    /// via [`start_smooth_scroll`] — a normal animation, so the
    /// interruption rule cancels it on external moves and `scrollend`
    /// fires after it completes, never before (task b).
    ///
    /// `relative` marks relative scroll operations (wheel, keyboard,
    /// completed smooth scrolls); absolute writes (script-driven
    /// instant scrolls) pass `false`, degrading
    /// `scroll-snap-stop: always` to `normal` per spec.
    ///
    /// The offset after this evaluation becomes the next one's
    /// pre-scroll offset (`snap_base`, carried across relayouts) — the
    /// `scroll-snap-stop: always` interval base — even when nothing
    /// launches, so every movement is evaluated exactly once.
    ///
    /// The selection also records the container's snapped state
    /// ([`ScrollContainer::snap_rest`]): each axis's winning snap area
    /// and resting position, or `None` where nothing was selected.
    /// [`ScrollTree::resnap_after_relayout`] follows those areas when
    /// layout moves them. An axis the snap type does not cover has no
    /// valid snap positions (CSS Scroll Snap §5.2): snap points
    /// orthogonal to the snap axis are ignored, not proximity-gated.
    fn eval_snap(&self, idx: usize, pre: (f32, f32), natural: (f32, f32), relative: bool) {
        let Some(c) = self.containers.get(idx) else {
            return;
        };
        c.snap_base.set(natural);
        let snap = c.snap;
        if snap.snap_type == SnapType::None {
            c.snap_rest.set((None, None));
            // A container evaluated in the non-snapping state reports
            // the all-null pair (it can only differ from the baseline
            // when the type changed under it, which resnap reports —
            // this is the backstop for completeness).
            self.note_snap_event(idx, SnapEventKind::Changing);
            return;
        }
        // Sub-`SNAP_EPS` 2D dust is not a scroll worth snapping after:
        // no evaluation happens, so no event can fire either.
        let Some(sel) = self.select_snap(idx, pre, natural, relative) else {
            return;
        };
        let (rest_x, rest_y, (sx, sy)) = (sel.rest_x, sel.rest_y, sel.rest_offset);
        // `select_snap` returns the resting offset already clamped
        // into `[origin, max]` — exactly the target
        // `start_smooth_scroll` would launch toward — so a follow-up
        // animation launches iff it differs from the natural offset.
        if (sx - natural.0).abs() > SNAP_EPS || (sy - natural.1).abs() > SNAP_EPS {
            self.start_smooth_scroll(idx, sx, sy);
        }
        c.snap_rest.set((rest_x, rest_y));
        // `scrollsnapchanging`: after any evaluation the recorded
        // rest names the eventual target of the in-progress scroll.
        self.note_snap_event(idx, SnapEventKind::Changing);
    }

    /// Report the eventual snap pair of container `idx`'s in-progress
    /// direct manipulation (scrollbar thumb drag) without moving
    /// anything: run the same selection the release-time
    /// [`ScrollTree::eval_snap`] will run (from `snap_base` to the
    /// live offset, non-relative) and queue `scrollsnapchanging` if
    /// the resulting pair differs from the last reported one.
    /// `snap_rest`, `snap_base`, and the offset are all left
    /// untouched, so the drag itself stays unsnapped until release.
    fn peek_changing(&self, idx: usize) {
        let Some(c) = self.containers.get(idx) else {
            return;
        };
        if c.smooth.get().is_some() {
            return;
        }
        let live = c.state.offset();
        let base = c.snap_base.get();
        if live == base {
            return;
        }
        let Some(sel) = self.select_snap(idx, base, live, false) else {
            return;
        };
        let pair = c.targets_from_rests(sel.rest_x, sel.rest_y);
        self.note_snap_event_pair(c, SnapEventKind::Changing, pair);
    }

    /// The pure selection core of [`ScrollTree::eval_snap`]: given a
    /// scroll of container `idx` from `pre` to `natural`, return the
    /// per-axis rests the scroll settles on and the resting offset
    /// (the selection clamped into `[origin, max]`), or `None` when
    /// the movement is sub-[`SNAP_EPS`] dust and nothing is evaluated.
    /// Touches no state: `eval_snap` adds the follow-up animation and
    /// the `snap_rest` recording, and the direct-manipulation peek
    /// reports the pair without either.
    fn select_snap(
        &self,
        idx: usize,
        pre: (f32, f32),
        natural: (f32, f32),
        relative: bool,
    ) -> Option<SnapSelection> {
        let c = self.containers.get(idx)?;
        let snap = c.snap;
        if snap.snap_type == SnapType::None {
            return Some(SnapSelection {
                rest_x: None,
                rest_y: None,
                rest_offset: natural,
            });
        }
        if (natural.0 - pre.0).abs() < SNAP_EPS && (natural.1 - pre.1).abs() < SNAP_EPS {
            return None;
        }
        let (px, py, (spw, sph)) = snap_candidates(c);
        let plain =
            |cands: &[(usize, SnapCandidate)]| cands.iter().map(|&(_, cd)| cd).collect::<Vec<_>>();
        let st = snap.snap_type;
        let mandatory = snap_is_mandatory(st);
        let tx = if snap_snaps_x(st) {
            snap_target_on_axis(
                &plain(&px),
                natural.0,
                pre.0,
                mandatory,
                proximity_threshold(spw),
                relative,
                spw,
            )
        } else {
            None
        };
        let ty = if snap_snaps_y(st) {
            snap_target_on_axis(
                &plain(&py),
                natural.1,
                pre.1,
                mandatory,
                proximity_threshold(sph),
                relative,
                sph,
            )
        } else {
            None
        };
        let (mut sx, mut sy) = natural;
        if let Some(x) = tx {
            sx = x;
        }
        if let Some(y) = ty {
            sy = y;
        }
        // The offset this selection leaves the container resting at:
        // the selection clamped into `[origin, max]` — exactly the
        // offset `start_smooth_scroll` would launch toward (or already
        // is), since it uses the same degenerate-safe clamp
        // ("Unreachable Snap Positions" per spec §5.3).
        let (mnx, mny) = c.state.min_offset();
        let (mxx, mxy) = c.state.max_offset();
        let clamp = |v: f32, mn: f32, mx: f32| {
            let lo = mn.min(mx);
            if v.is_nan() { lo } else { v.clamp(lo, mx) }
        };
        let rest = |sel: Option<f32>, pos: f32, cands: &[(usize, SnapCandidate)]| {
            sel.and_then(|_| snap_winner(cands, pos))
                .map(|area| SnapRest::of(&c.snap_areas, area, pos))
        };
        let (cx, cy) = (clamp(sx, mnx, mxx), clamp(sy, mny, mxy));
        Some(SnapSelection {
            rest_x: rest(tx, cx, &px),
            rest_y: rest(ty, cy, &py),
            rest_offset: (cx, cy),
        })
    }
}

// ---------------------------------------------------------------------------
// Tab glue
// ---------------------------------------------------------------------------

/// Rebuild the tab's scroll state after layout: discover the containers,
/// publish the per-element maxima (offsets survive, re-clamped), publish
/// the document maximum (nested subtrees excluded), and re-bake the
/// paint tree. Called from the window worker's `refresh` layout stage.
///
/// Takes the render pipeline and the document directly (not `&Tab`):
/// the synchronous reflow a geometry getter triggers
/// ([`bowser_js::LayoutEnsurer`]) holds `&mut RenderPipeline` disjoint
/// from the installed bundle.
pub(crate) fn rebuild_after_layout(render: &mut RenderPipeline, doc: &RefCell<Document>) {
    let Some(root) = render.layout_root.as_ref() else {
        return;
    };
    // The document borrow ends before the tree is installed: `doc`
    // would otherwise hold `render` borrowed across the assignment below.
    let mut tree = {
        let doc = doc.borrow();
        ScrollTree::build(
            &doc,
            root,
            render.viewport,
            render.scroll(),
            render.element_scroll(),
        )
    };
    let (origin, max) = tree.document_max_scroll(root, render.viewport);
    // The viewport container's minimum offset: content above/left of the
    // viewport origin scrolls into view on negative offsets. The shared
    // `render.scroll()` state carries the same minimum so script-driven
    // scrolls clamp into `[origin, max]` exactly like wheel/keyboard.
    tree.containers[0].origin = origin;
    render.scroll().set_min_offset(origin);
    // Publish the new maximum BEFORE carrying scroll state over:
    // `carry_scroll_state` re-snaps against the fresh geometry, and
    // the viewport container's shared state would otherwise still
    // clamp that resnap to the PREVIOUS layout's max — stranding the
    // viewport at the stale max (a non-snap position) whenever a
    // snapped target moved past it.
    render.scroll().set_max_offset(max);
    // The viewport's CSS `scroll-behavior` default, for
    // `window.scrollTo()`/`scrollBy()` without an explicit `behavior`
    // (see `bowser_js::ScrollState::set_smooth_default`): the script
    // natives resolve it synchronously against the shared state.
    render
        .scroll()
        .set_smooth_default(tree.containers[0].css_behavior == ScrollBehavior::Smooth);
    // Keep a mid-scroll gesture's `scrollend` bookkeeping across the
    // rebuild: without this the fresh tree's zeroed ticks would silently
    // drop the pending `scrollend`.
    let resnap_moved = tree.carry_scroll_state(&render.scroll_tree);
    render.scroll_tree = tree;
    let (x, y) = render.scroll().offset();
    render.scroll().set_offset(x, y);
    render.paint_root = Some(render.scroll_tree.bake_paint_tree(root).into_owned());
    // Re-expose the resnap's moves to the next fold: the bake above
    // re-seeded `painted` at the post-move offsets, which would hide
    // the moves from the fold's drift check — two ticks later
    // `end_tick` would return a bare `scrollend` with no preceding
    // `scroll`. Resetting to the pre-move offsets makes the fold
    // report each move, so `scroll` fires before the stamped
    // `scrollend` (Chrome fires `scroll` for layout-induced offset
    // changes).
    for (idx, before) in resnap_moved {
        render.scroll_tree.containers[idx].painted.set(before);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tab::Tab;
    use bowser_css::parse_stylesheet;
    use bowser_dom::ShadowRootMode;
    use bowser_layout::{FixedFontMetrics, Size, layout};
    use bowser_paint::display_list::{
        DisplayItem, EDGE_GLOW_DEPTH, scrollbar_thumb_color, scrollbar_track_color,
    };

    /// Real layout tree for `html` with `style`: styles cascade for real,
    /// so discovery runs against what production sees.
    fn layout_html(html: &[u8], css: &str) -> (Document, LayoutBox) {
        let (doc, _) = Document::parse_html(html);
        let sheet = parse_stylesheet(css);
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        (doc, root)
    }

    fn build_tree(
        doc: &Document,
        root: &LayoutBox,
    ) -> (Rc<ScrollState>, Rc<ElementScrollStates>, ScrollTree) {
        let doc_state = Rc::new(ScrollState::new());
        let elements = Rc::new(ElementScrollStates::new());
        let tree = ScrollTree::build(
            doc,
            root,
            Size { w: 800.0, h: 600.0 },
            &doc_state,
            &elements,
        );
        (doc_state, elements, tree)
    }

    const SCROLL_HTML: &[u8] = b"<html><body>\
        <div id=\"outer\"><div id=\"tall\">x</div></div>\
        <div id=\"wide\"><div id=\"wide-inner\">y</div></div>\
        <div id=\"hidden\"><div id=\"hidden-inner\">z</div></div>\
        </body></html>";
    const SCROLL_CSS: &str = "
        #outer { overflow-y: auto; height: 100px; width: 200px; }
        #tall { height: 300px; }
        #wide { overflow-x: scroll; width: 100px; height: 50px; }
        #wide-inner { width: 400px; height: 10px; }
        #hidden { overflow: hidden; height: 100px; }
        #hidden-inner { height: 300px; }
    ";

    #[test]
    fn discover_finds_scroll_containers_including_hidden() {
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let found = discover(&doc, &root, Size { w: 800.0, h: 600.0 });
        let names: Vec<String> = found
            .iter()
            .map(|d| {
                doc.get_attribute(d.node, "id")
                    .unwrap_or_default()
                    .to_owned()
            })
            .collect();
        assert_eq!(names, vec!["outer", "wide", "hidden"]);
        // Maxima: content extent minus scrollport.
        let outer = &found[0];
        assert_eq!(outer.max, (0.0, 200.0));
        let wide = &found[1];
        assert_eq!(wide.max, (300.0, 0.0));
        // `overflow: hidden` also establishes a scroll container
        // (css-overflow-3): it gets a real maximum for programmatic
        // scrolling, but never consumes wheel/keyboard deltas.
        let hidden = &found[2];
        assert_eq!(hidden.max, (0.0, 200.0));
        assert_eq!(hidden.user_axes, (false, false));
        // The unspecified axis computes to `auto` when the other is
        // non-`visible` (css-overflow-3 §2), so `outer`/`wide` are
        // user-scrollable on both axes.
        assert_eq!(outer.user_axes, (true, true));
        assert_eq!(wide.user_axes, (true, true));
    }

    /// `bg_images` signatures mirror the builder's two background-image
    /// arms: a gradient on a div with a text child contributes exactly
    /// one key (the div's own rect — `background-image` is not
    /// inherited, so the text child emits no `Gradient`), while a
    /// `url()` background still keys the text child's rect too (the
    /// builder emits `RepeatImage` for it through the inherited style).
    /// A phantom gradient key would trip the ambiguity gate and force
    /// spurious sticky re-bakes; a missing `url()` key would leave the
    /// text box's `RepeatImage` unattributable (`Unknown`) with the same
    /// effect.
    #[test]
    fn bg_image_signatures_split_url_and_gradient() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"g\">hello</div><div id=\"u\">world</div></body></html>",
            "#g { background-image: linear-gradient(to right, red, blue); width: 100px; height: 60px; } \
             #u { background-image: url(a.png); width: 100px; height: 60px; }",
        );
        let g = doc.get_elements_by_tag_name("div").item(0).expect("div#g");
        let g_box = find_box(&root, g).expect("div#g box");
        let u = doc.get_elements_by_tag_name("div").item(1).expect("div#u");
        let u_box = find_box(&root, u).expect("div#u box");
        let mut sig = PaintSignatures::default();
        collect_paint_signatures(&root, None, &mut sig);
        assert!(
            sig.bg_images.contains(&g_box.rect),
            "gradient div keys its own rect"
        );
        assert!(
            sig.bg_images.contains(&u_box.rect),
            "url div keys its own rect"
        );
        assert_eq!(
            sig.bg_images.len(),
            3,
            "three keys: both divs plus the url text child; \
             the gradient text child contributes no key"
        );
    }

    #[test]
    fn discover_nests_containers_with_parent_links() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"a\"><div id=\"b\"><div id=\"c\">x</div></div></div></body></html>",
            "#a { overflow-y: auto; height: 200px; } \
             #b { overflow-y: auto; height: 100px; } \
             #c { height: 500px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert_eq!(tree.len(), 3, "viewport + two nested containers");
        let a = doc.get_element_by_id("a").expect("a");
        let b = doc.get_element_by_id("b").expect("b");
        let ia = tree.index_of(a).expect("a is a container");
        let ib = tree.index_of(b).expect("b is a container");
        assert_eq!(tree.parent(ib), ia, "b chains into a");
        assert_eq!(tree.parent(ia), 0, "a chains into the viewport");
        // Maxima account for nesting: b's overflowing content does not
        // extend a's range (only b's own 100px box counts).
        assert_eq!(tree.container(ib).state.max_offset(), (0.0, 400.0));
        assert_eq!(tree.container(ia).state.max_offset(), (0.0, 0.0));
    }

    #[test]
    fn discover_restores_preorder_for_parent_linking() {
        // A > B (child of A), C (sibling of A): deepest-first
        // confirmation runs B, A, C. The tree must be rebuilt in preorder
        // (A, B, C) — sorting by depth alone keeps B after C among equal
        // depths, and the depth stack would then link B to C instead of A.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"a\"><div id=\"b\"><div id=\"tall-b\">x</div></div></div>\
              <div id=\"c\"><div id=\"tall-c\">y</div></div></body></html>",
            "#a { overflow-y: auto; height: 200px; } \
             #b { overflow-y: auto; height: 100px; } \
             #c { overflow-y: auto; height: 100px; } \
             #tall-b { height: 500px; } #tall-c { height: 500px; }",
        );
        let found = discover(&doc, &root, Size { w: 800.0, h: 600.0 });
        let names: Vec<String> = found
            .iter()
            .map(|d| {
                doc.get_attribute(d.node, "id")
                    .unwrap_or_default()
                    .to_owned()
            })
            .collect();
        assert_eq!(names, vec!["a", "b", "c"], "discovery is in preorder");

        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let a = doc.get_element_by_id("a").expect("a");
        let b = doc.get_element_by_id("b").expect("b");
        let c = doc.get_element_by_id("c").expect("c");
        let ia = tree.index_of(a).expect("a container");
        let ib = tree.index_of(b).expect("b container");
        let ic = tree.index_of(c).expect("c container");
        assert!(ia < ib && ib < ic, "containers pushed in tree order");
        assert_eq!(tree.parent(ib), ia, "b chains into a, not c");
        assert_eq!(tree.parent(ia), 0, "a chains into the viewport");
        assert_eq!(tree.parent(ic), 0, "c chains into the viewport");
    }

    #[test]
    fn sibling_subtrees_at_different_depths_parent_to_viewport() {
        // F4 regression (reviewer probe 4): preorder `#a` (layout depth
        // 2), `#b` (layout depth 4) where `#b` lives in a SIBLING
        // subtree. The old depth stack parented `#b` to `#a`; both must
        // chain straight to the viewport.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"a\"><div id=\"ta\">x</div></div>\
              <section><div><div id=\"b\"><div id=\"tb\">y</div></div></div></section></body></html>",
            "#a { overflow: auto; height: 100px; } #ta { height: 200px; } \
             #b { overflow: auto; height: 100px; } #tb { height: 200px; }",
        );
        let found = discover(&doc, &root, Size { w: 800.0, h: 600.0 });
        // The probe's shape: preorder, different layout depths (depths
        // read from layout itself: `Discovered` carries no depth — depth
        // comparison is what mis-parented here).
        let depths: Vec<usize> = bowser_layout::scroll_containers_with_offset(&root, (0.0, 0.0))
            .iter()
            .map(|c| c.depth)
            .collect();
        assert_eq!(found.len(), 2);
        assert_eq!(depths.len(), 2);
        assert!(
            depths[0] < depths[1],
            "sibling subtrees at different layout depths: {depths:?}"
        );
        assert_eq!(found[0].parent, None, "a's scroller is the viewport");
        assert_eq!(found[1].parent, None, "b's scroller is the viewport");

        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let a = doc.get_element_by_id("a").expect("a");
        let b = doc.get_element_by_id("b").expect("b");
        let ia = tree.index_of(a).expect("a container");
        let ib = tree.index_of(b).expect("b container");
        assert_eq!(tree.parent(ia), 0, "a chains to the viewport");
        assert_eq!(tree.parent(ib), 0, "b chains to the viewport, not to a");

        // Behavioral: scrolling #a must not perturb hit-testing over #b
        // (the old tree accumulated #a's offset descending to #b).
        tree.container(ia).state.scroll_by(0.0, 95.0);
        let rb = tree.container(ib).scrollport;
        assert_eq!(
            tree.innermost_at(rb.x + 10.0, rb.y + 10.0),
            ib,
            "hit test over b is not offset by a's scroll"
        );

        // #b's chained remainder must reach the viewport, not #a: #b at
        // its max (100) consumes 100 of the 150, the rest chains past.
        let o = tree.scroll_chain(tree.innermost_at(rb.x + 10.0, rb.y + 10.0), 0.0, 150.0);
        assert_eq!(o.nodes(), vec![Some(b)]);
        assert_eq!(tree.container(ib).state.offset(), (0.0, 100.0));
        assert_eq!(
            tree.container(ia).state.offset(),
            (0.0, 95.0),
            "a untouched by b's chain"
        );
        assert_eq!(
            doc_state.offset(),
            (0.0, 0.0),
            "document has no room; the remainder dies at the viewport"
        );
    }

    #[test]
    fn nested_triple_chains_viewport_outer_middle_inner() {
        // Genuine nesting still chains innermost-first through every
        // level: viewport > outer > middle > inner.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"outer\"><div id=\"ot\">a</div>\
              <div id=\"middle\"><div id=\"mt\">b</div>\
              <div id=\"inner\"><div id=\"tall\">x</div></div></div></div></body></html>",
            "#outer { overflow-y: auto; height: 300px; } #ot { height: 400px; } \
             #middle { overflow-y: auto; height: 200px; } #mt { height: 300px; } \
             #inner { overflow-y: auto; height: 100px; } #tall { height: 500px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let middle = doc.get_element_by_id("middle").expect("middle");
        let inner = doc.get_element_by_id("inner").expect("inner");
        let io = tree.index_of(outer).expect("outer container");
        let im = tree.index_of(middle).expect("middle container");
        let ii = tree.index_of(inner).expect("inner container");
        assert_eq!(tree.parent(io), 0, "outer chains to the viewport");
        assert_eq!(tree.parent(im), io, "middle chains into outer");
        assert_eq!(tree.parent(ii), im, "inner chains into middle");

        // Maxima (in-flow siblings stack, so each level's extent reaches
        // the bottom of the nested box below its own filler): inner 400,
        // middle 200, outer 300.
        assert_eq!(tree.container(ii).state.max_offset(), (0.0, 400.0));
        assert_eq!(tree.container(im).state.max_offset(), (0.0, 200.0));
        assert_eq!(tree.container(io).state.max_offset(), (0.0, 300.0));

        // A scroll starting at the innermost box chains through all
        // three levels innermost-first (900 = 400 + 200 + 300).
        let o = tree.scroll_chain(ii, 0.0, 900.0);
        assert_eq!(o.nodes(), vec![Some(inner), Some(middle), Some(outer)]);
        assert_eq!(tree.container(ii).state.offset(), (0.0, 400.0));
        assert_eq!(tree.container(im).state.offset(), (0.0, 200.0));
        assert_eq!(tree.container(io).state.offset(), (0.0, 300.0));
    }

    #[test]
    fn wheel_chains_past_hidden_containers() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"h\"><div id=\"tall\">x</div></div><div id=\"doc-tall\">y</div></body></html>",
            "#h { overflow: hidden; height: 100px; } #tall { height: 500px; } \
             #doc-tall { height: 2000px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let h = doc.get_element_by_id("h").expect("h");
        let ih = tree.index_of(h).expect("hidden is a container");
        assert_eq!(tree.container(ih).user_scroll, (false, false));

        // Programmatic scrolling still works on `hidden` containers: JS
        // writes the shared state directly, bypassing the wheel mask.
        tree.container(ih).state.set_offset(0.0, 100.0);
        assert_eq!(tree.container(ih).state.offset(), (0.0, 100.0));
        tree.container(ih).state.set_offset(0.0, 0.0);

        // A wheel over the hidden box consumes nothing there: the full
        // delta chains to the viewport.
        let o = tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 120.0);
        assert_eq!(o.nodes(), vec![None]);
        assert_eq!(doc_state.offset(), (0.0, 120.0));
        assert_eq!(tree.container(ih).state.offset(), (0.0, 0.0));
    }

    #[test]
    fn viewport_scroller_elements_never_become_containers() {
        // `overflow` on <body> propagates to the viewport; it must not
        // also mint a nested container (the JS scrollTop mapping would
        // disagree with wheel dispatch).
        let (doc, root) = layout_html(
            b"<html><body><div id=\"tall\">x</div></body></html>",
            "body { overflow-y: auto; } #tall { height: 2000px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert!(tree.is_viewport_only());
    }

    #[test]
    fn innermost_at_routes_through_offsets() {
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");

        // Viewport point (10, 10) is inside #outer's scrollport.
        assert_eq!(tree.innermost_at(10.0, 10.0), io);
        // Far below it: the viewport.
        assert_eq!(tree.innermost_at(700.0, 500.0), 0);

        // Scroll #outer down 150: its content shifts up, so the point
        // that used to hit the container's content at y=10 now needs
        // y=160 in page terms — the scrollport itself is unmoved.
        tree.container(io).state.scroll_by(0.0, 150.0);
        assert_eq!(tree.innermost_at(10.0, 10.0), io);
        assert_eq!(tree.innermost_at(10.0, 99.0), io);
        // Past the scrollport's bottom edge and clear of #wide below it:
        // viewport again.
        assert_eq!(tree.innermost_at(700.0, 101.0), 0);
    }

    const TRANSLATE_HTML: &[u8] = b"<html><body>\
        <div id=\"wrap\"><div id=\"sc\"><div id=\"tall\">x</div></div></div>\
        </body></html>";
    const TRANSLATE_CSS: &str = "#wrap { transform: translate(100px, 50px); } \
        #sc { overflow-y: auto; height: 100px; width: 200px; } \
        #tall { height: 500px; }";

    #[test]
    fn innermost_at_hits_translated_scroller_at_visual_location() {
        let (doc, root) = layout_html(TRANSLATE_HTML, TRANSLATE_CSS);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");
        let port = tree.container(isc).scrollport;
        // The wrapper's translate is baked onto the container: layout's
        // scrollport stays pre-transform.
        assert_eq!(tree.container(isc).transform_offset, (100.0, 50.0));
        // The wheel-routing point is the scrollport's visual location...
        let (vx, vy) = (port.x + 100.0 + 10.0, port.y + 50.0 + 10.0);
        assert_eq!(tree.innermost_at(vx, vy), isc);
        // ...not its pre-transform one, which now belongs to the page
        // behind the translated scroller.
        assert_eq!(tree.innermost_at(port.x + 10.0, port.y + 10.0), 0);
    }

    #[test]
    fn scrollbar_hit_tests_translated_scroller_at_visual_location() {
        let (doc, root) = layout_html(TRANSLATE_HTML, TRANSLATE_CSS);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");
        // Viewport offset is zero here: viewport coords == page coords.
        let v = tree.scrollbar_geom(isc).vertical.expect("bar");
        let port = tree.container(isc).scrollport;
        // The geometry sits at the visual (translated) location...
        assert!(approx_eq(v.track.x, port.x + 100.0 + port.w - v.track.w));
        assert!(approx_eq(v.track.y, port.y + 50.0));
        // ...so the thumb hits there, and nothing hits at the
        // pre-transform spot anymore.
        let cx = v.thumb.x + v.thumb.w / 2.0;
        let cy = v.thumb.y + v.thumb.h / 2.0;
        let hit = tree
            .scrollbar_at(cx, cy)
            .expect("thumb hit at visual location");
        assert_eq!(hit.container, isc);
        assert!(hit.on_thumb);
        assert!(tree.scrollbar_at(cx - 100.0, cy - 50.0).is_none());
    }

    #[test]
    fn scroll_chain_chains_at_bounds() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div><div id=\"doc-tall\">y</div><div id=\"doc-wide\">z</div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 300px; } \
             #doc-tall { height: 2000px; } #doc-wide { width: 2000px; height: 10px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");
        let cstate = tree.container(isc).state.clone();

        // Cursor over the container: the container consumes first.
        let o = tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 120.0);
        assert!(o.moved());
        assert_eq!(cstate.offset(), (0.0, 120.0));
        assert_eq!(o.nodes(), vec![Some(sc)]);
        assert_eq!(doc_state.offset(), (0.0, 0.0));

        // Past the container's max (200): the remainder chains to the
        // document scroller in the same gesture.
        let o = tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 200.0);
        assert_eq!(cstate.offset(), (0.0, 200.0));
        assert_eq!(doc_state.offset(), (0.0, 120.0));
        assert_eq!(o.nodes(), vec![Some(sc), None]);

        // Cursor outside any container: straight to the viewport.
        let o = tree.scroll_chain(tree.innermost_at(700.0, 500.0), 0.0, 60.0);
        assert_eq!(o.nodes(), vec![None]);
        assert_eq!(doc_state.offset(), (0.0, 180.0));

        // Horizontal delta over a vertical-only container chains past it
        // untouched: the container consumes nothing on x.
        let o = tree.scroll_chain(tree.innermost_at(10.0, 10.0), 50.0, 0.0);
        assert_eq!(o.nodes(), vec![None]);
        assert_eq!(doc_state.offset(), (50.0, 180.0));
        assert_eq!(cstate.offset(), (0.0, 200.0));
    }

    #[test]
    fn overscroll_edge_hit_records_on_boundary_push() {
        // Container with room to scroll, unscrollable document: a push
        // past the container's max edge clamps the container, chains
        // the remainder to the viewport (also clamped, at its zero
        // max), and records an edge press on each — innermost-first.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 300px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");
        let cstate = tree.container(isc).state.clone();

        // Drive the container to its max edge (max = 200): no edge hit
        // while the container still moves.
        let o = tree.scroll_chain(isc, 0.0, 200.0);
        assert!(o.moved());
        assert!(!o.has_edge_hits());
        assert_eq!(cstate.offset(), (0.0, 200.0));

        // Past the max: the container clamps with residual left.
        let o = tree.scroll_chain(isc, 0.0, 100.0);
        assert_eq!(cstate.offset(), (0.0, 200.0));
        assert!(!o.moved());
        assert!(o.nodes().is_empty());
        assert!(o.has_edge_hits());
        assert_eq!(o.edge_hits.len(), 2);
        let hit = o.edge_hits[0];
        assert_eq!(hit.index, isc);
        assert_eq!(hit.node, Some(sc));
        assert_eq!(hit.axis, OverscrollAxis::Vertical);
        assert_eq!(hit.edge, OverscrollEdge::Max);
        assert_eq!(hit.residual, 100.0);
        // The viewport clamper records second, innermost-first.
        assert_eq!(o.edge_hits[1].index, 0);
        assert_eq!(o.edge_hits[1].node, None);
        assert_eq!(o.edge_hits[1].edge, OverscrollEdge::Max);
        // Transient shift only: resistance * residual, paint-time.
        assert_eq!(
            tree.overscroll_shift(isc),
            (0.0, 100.0 * OVERSCROLL_RESISTANCE)
        );
        assert_eq!(
            tree.overscroll_shift(0),
            (0.0, 100.0 * OVERSCROLL_RESISTANCE)
        );
        assert_eq!(tree.overscroll_shift(usize::MAX), (0.0, 0.0));
        // A clamped press arms no `scroll`/`scrollend` bookkeeping.
        assert!(tree.end_tick().is_empty());
    }

    #[test]
    fn overscroll_origin_edge_pushes_negative_shift() {
        // Pressing up past the top edge records an Origin hit and a
        // negative transient shift.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 300px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");

        let o = tree.scroll_chain(isc, 0.0, -60.0);
        assert!(!o.moved());
        let hit = o.edge_hits[0];
        assert_eq!(hit.index, isc);
        assert_eq!(hit.axis, OverscrollAxis::Vertical);
        assert_eq!(hit.edge, OverscrollEdge::Origin);
        assert_eq!(hit.residual, -60.0);
        assert_eq!(
            tree.overscroll_shift(isc),
            (0.0, -60.0 * OVERSCROLL_RESISTANCE)
        );
    }

    #[test]
    fn overscroll_partial_consume_records_no_hit() {
        // A container that consumed part of the delta is not at a
        // boundary press: its remainder chains outward and only the
        // terminal clamper records an edge hit.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 300px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");

        // 250 offered, 200 consumed: at the max edge, but moved.
        let o = tree.scroll_chain(isc, 0.0, 250.0);
        assert!(o.moved());
        assert_eq!(tree.container(isc).state.offset(), (0.0, 200.0));
        assert!(o.edge_hits.iter().all(|h| h.index != isc));
        // The remainder pressed the viewport instead.
        assert_eq!(o.edge_hits.len(), 1);
        assert_eq!(o.edge_hits[0].index, 0);
    }

    #[test]
    fn overscroll_spring_settles_to_zero_without_scroll_events() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 300px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");
        let cstate = tree.container(isc).state.clone();

        tree.scroll_chain(isc, 0.0, 200.0);
        tree.scroll_chain(isc, 0.0, 100.0);
        assert!(tree.has_overscroll_in_flight());

        // The first tick establishes the baseline; the spring
        // integrates from the second tick on.
        let mut now = Instant::now() + Duration::from_millis(16);
        let mut saw_motion = false;
        for _ in 0..600 {
            now += Duration::from_millis(16);
            if !tree.advance_overscroll(now).is_empty() {
                saw_motion = true;
            }
            if !tree.has_overscroll_in_flight() {
                break;
            }
        }
        assert!(saw_motion, "spring never moved");
        assert!(!tree.has_overscroll_in_flight());
        assert_eq!(tree.overscroll_shift(isc), (0.0, 0.0));
        assert_eq!(tree.overscroll_shift(0), (0.0, 0.0));
        // Settling never moved the offset or armed scrollend.
        assert_eq!(cstate.offset(), (0.0, 200.0));
        assert!(tree.end_tick().is_empty());
        // A settled spring is a no-op tick.
        assert!(
            tree.advance_overscroll(now + Duration::from_millis(16))
                .is_empty()
        );
    }

    #[test]
    fn overscroll_spring_stable_at_max_dt() {
        // Regression: the old semi-implicit Euler integrator diverged
        // for dt > ~30 ms (OVERSCROLL_MAX_DT is 50 ms), producing NaN and
        // a spring that never settled. The exact closed-form step is
        // unconditionally stable: 50 ms steps must stay finite, decay,
        // and reach rest.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 300px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");

        tree.scroll_chain(isc, 0.0, 200.0);
        tree.scroll_chain(isc, 0.0, 100.0);
        assert!(tree.has_overscroll_in_flight());

        let mut now = Instant::now() + Duration::from_millis(50);
        let mut prev = f32::INFINITY;
        let mut ticks = 0;
        while tree.has_overscroll_in_flight() {
            now += Duration::from_millis(50);
            tree.advance_overscroll(now);
            ticks += 1;
            let (x, y) = tree.overscroll_shift(isc);
            assert!(x.is_finite() && y.is_finite(), "spring went non-finite");
            let mag = x.abs() + y.abs();
            // Exact critically-damped decay: each 50 ms step multiplies
            // by e^(-1.4) ≈ 0.247, so magnitude must shrink fast.
            assert!(mag <= prev, "spring grew at max dt: {prev} -> {mag}");
            prev = mag;
            assert!(ticks < 600, "spring did not settle at max dt");
        }
        assert_eq!(tree.overscroll_shift(isc), (0.0, 0.0));
    }

    #[test]
    fn overscroll_none_suppresses_shift_but_records_hit() {
        // `overscroll-behavior-y: none`: the edge press is recorded
        // (it happened) but the transient shift stays zero, and the
        // remainder dies at the container — the viewport sees nothing.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; overscroll-behavior-y: none; } \
             #tall { height: 300px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");

        tree.scroll_chain(isc, 0.0, 200.0);
        let o = tree.scroll_chain(isc, 0.0, 100.0);
        assert!(o.has_edge_hits());
        assert_eq!(o.edge_hits.len(), 1);
        assert_eq!(o.edge_hits[0].index, isc);
        assert_eq!(o.edge_hits[0].edge, OverscrollEdge::Max);
        assert_eq!(tree.overscroll_shift(isc), (0.0, 0.0));
        assert!(!tree.has_overscroll_in_flight());
        assert_eq!(doc_state.offset(), (0.0, 0.0));
    }

    #[test]
    fn overscroll_rtl_negative_origin_reports_origin_edge() {
        // Bound-relative edge reporting: a push past a negative origin
        // under RTL reports Origin, never Max, despite the negative
        // coordinates.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"inner\"><div id=\"neg\">x</div><div id=\"tall\">y</div></div></body></html>",
            "#inner { overflow: scroll; height: 100px; width: 200px; direction: rtl; } \
             #neg { position: relative; left: -40px; top: -30px; height: 50px; } \
             #tall { height: 400px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let inner = doc.get_element_by_id("inner").expect("inner");
        let i = tree.index_of(inner).expect("inner container");
        assert!(tree.container(i).rtl);
        // `#inner` is `overflow: scroll` under RTL, so its forced
        // vertical strip sits on the left and the content origin shifts
        // past it: `#neg`'s `left: -40px` lands at -28px, not -40px.
        assert_eq!(tree.container(i).origin, (-28.0, -30.0));

        // First push reaches the negative origin (partial consume: no
        // hit for the container itself).
        let o = tree.scroll_chain(i, -100.0, 0.0);
        assert_eq!(tree.container(i).state.offset().0, -28.0);
        assert!(o.edge_hits.iter().all(|h| h.index != i));

        // Second push clamps against the origin edge.
        let o = tree.scroll_chain(i, -100.0, 0.0);
        let hit = o
            .edge_hits
            .iter()
            .find(|h| h.index == i)
            .expect("container edge hit");
        assert_eq!(hit.axis, OverscrollAxis::Horizontal);
        assert_eq!(hit.edge, OverscrollEdge::Origin);
        assert_eq!(hit.residual, -100.0);
        assert_eq!(
            tree.overscroll_shift(i),
            (-100.0 * OVERSCROLL_RESISTANCE, 0.0)
        );
    }

    #[test]
    fn overscroll_infinite_residual_clamps_to_finite_shift() {
        // `End` drives `scroll_chain` with +∞; at the max edge the press
        // must feed a finite, capped shift — never NaN/∞ — and the
        // saturated band still springs back to zero.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 300px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");

        tree.scroll_chain(isc, 0.0, 200.0);
        let o = tree.scroll_chain(isc, 0.0, f32::INFINITY);
        assert!(!o.moved());
        let hit = o.edge_hits[0];
        assert_eq!(hit.index, isc);
        assert_eq!(hit.edge, OverscrollEdge::Max);
        assert_eq!(hit.residual, EDGE_HIT_MAX_RESIDUAL);
        let (_, shift) = tree.overscroll_shift(isc);
        assert!(shift.is_finite());
        assert_eq!(shift, OVERSCROLL_MAX_SHIFT);

        let mut now = Instant::now() + Duration::from_millis(16);
        for _ in 0..1200 {
            now += Duration::from_millis(16);
            tree.advance_overscroll(now);
            if !tree.has_overscroll_in_flight() {
                break;
            }
        }
        assert!(!tree.has_overscroll_in_flight());
        assert_eq!(tree.overscroll_shift(isc), (0.0, 0.0));
    }

    #[test]
    fn overscroll_glow_strength_proportional_to_shift() {
        // Strength is |shift| / OVERSCROLL_MAX_SHIFT; the edge follows
        // the shift's sign (positive = pressed the max edge).
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 300px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");

        assert_eq!(tree.overscroll_glow(isc), None, "no shift, no glow");
        assert_eq!(tree.overscroll_glow(usize::MAX), None, "unknown index");

        // To the max edge, then 100px past it: shift = 100 * resistance.
        tree.scroll_chain(isc, 0.0, 200.0);
        tree.scroll_chain(isc, 0.0, 100.0);
        let (shift_x, shift_y) = tree.overscroll_shift(isc);
        assert_eq!((shift_x, shift_y), (0.0, 100.0 * OVERSCROLL_RESISTANCE));
        let (edge, strength) = tree.overscroll_glow(isc).expect("glow");
        assert_eq!(edge, Edge4::Bottom);
        let expected = (100.0 * OVERSCROLL_RESISTANCE) / OVERSCROLL_MAX_SHIFT;
        assert!(
            (strength - expected).abs() < 1e-6,
            "strength {strength} ∝ shift, expected {expected}"
        );

        // Origin press: negative shift glows the top edge.
        let c = tree.container(isc);
        c.overscroll_delta.set((0.0, -48.0));
        let (edge, strength) = tree.overscroll_glow(isc).expect("glow");
        assert_eq!(edge, Edge4::Top);
        assert!((strength - 48.0 / OVERSCROLL_MAX_SHIFT).abs() < 1e-6);
    }

    #[test]
    fn overscroll_glow_saturates_at_cap() {
        // A huge residual clamps the shift at OVERSCROLL_MAX_SHIFT, so
        // strength saturates at exactly 1.0.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 300px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");

        tree.scroll_chain(isc, 0.0, 200.0);
        tree.scroll_chain(isc, 0.0, f32::INFINITY);
        assert_eq!(tree.overscroll_shift(isc).1, OVERSCROLL_MAX_SHIFT);
        let (edge, strength) = tree.overscroll_glow(isc).expect("glow");
        assert_eq!(edge, Edge4::Bottom);
        assert_eq!(strength, 1.0, "saturated at the cap");
    }

    #[test]
    fn overscroll_glow_dominant_axis_wins() {
        // Both axes stretched: the larger magnitude picks the edge.
        // (Set directly: scroll_chain stretches one axis per press, and
        // diagonal presses are the gesture layer's job.)
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let iouter = tree.index_of(outer).expect("outer container");
        let c = tree.container(iouter);

        c.overscroll_delta.set((50.0, 20.0));
        assert_eq!(
            tree.overscroll_glow(iouter).map(|(e, _)| e),
            Some(Edge4::Right)
        );
        c.overscroll_delta.set((20.0, 50.0));
        assert_eq!(
            tree.overscroll_glow(iouter).map(|(e, _)| e),
            Some(Edge4::Bottom)
        );
        c.overscroll_delta.set((-20.0, -50.0));
        assert_eq!(
            tree.overscroll_glow(iouter).map(|(e, _)| e),
            Some(Edge4::Top)
        );
        c.overscroll_delta.set((-50.0, -20.0));
        assert_eq!(
            tree.overscroll_glow(iouter).map(|(e, _)| e),
            Some(Edge4::Left)
        );
    }

    #[test]
    fn overscroll_glow_none_suppresses_axis_independently() {
        // `overscroll-behavior: none` on an axis suppresses that axis's
        // glow (defense in depth: the press path already refuses to
        // stretch a `none` axis, so the shift is set directly here);
        // `contain` keeps the affordance, and the other axis is
        // independent.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div>\
               <div id=\"sc2\"><div id=\"tall2\">y</div></div></body></html>",
            "#sc { overflow: auto; height: 100px; width: 100px; overscroll-behavior-y: none; } \
             #tall { height: 300px; } \
             #sc2 { overflow: auto; height: 100px; width: 100px; overscroll-behavior: contain; } \
             #tall2 { height: 300px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");
        let sc2 = doc.get_element_by_id("sc2").expect("sc2");
        let isc2 = tree.index_of(sc2).expect("sc2 container");

        // y is `none`: no glow on y even with a (hypothetical) shift.
        let c = tree.container(isc);
        c.overscroll_delta.set((0.0, 42.0));
        assert_eq!(
            tree.overscroll_glow(isc),
            None,
            "`none` suppresses the y glow"
        );
        // x is `auto` on the same container: independent, still glows.
        c.overscroll_delta.set((24.0, 42.0));
        let (edge, strength) = tree.overscroll_glow(isc).expect("x glows");
        assert_eq!(edge, Edge4::Right);
        assert!((strength - 24.0 / OVERSCROLL_MAX_SHIFT).abs() < 1e-6);

        // `contain` keeps the affordance on both axes.
        let c2 = tree.container(isc2);
        c2.overscroll_delta.set((0.0, -42.0));
        let (edge, _) = tree.overscroll_glow(isc2).expect("contain glows");
        assert_eq!(edge, Edge4::Top);
    }

    #[test]
    fn overscroll_glow_fades_with_spring() {
        // The spring drives strength to zero: after settling, the glow
        // is gone (the window worker re-pushes the epilogue per tick,
        // so a stale full-strength glow can never linger).
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 300px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");

        tree.scroll_chain(isc, 0.0, 200.0);
        tree.scroll_chain(isc, 0.0, 100.0);
        let (_, s0) = tree.overscroll_glow(isc).expect("glow");
        assert!(s0 > 0.0);
        let mut now = Instant::now();
        let mut prev = s0;
        for _ in 0..1200 {
            now += Duration::from_millis(16);
            tree.advance_overscroll(now);
            match tree.overscroll_glow(isc) {
                Some((_, s)) => {
                    assert!(s <= prev + 1e-6, "strength decays monotonically");
                    prev = s;
                }
                None => break,
            }
            if !tree.has_overscroll_in_flight() {
                break;
            }
        }
        assert_eq!(tree.overscroll_glow(isc), None, "settled: no glow");
    }

    #[test]
    fn append_overscroll_glow_emits_clipped_edge_items() {
        // The epilogue helper wraps each glow in a PushClip/PopClip
        // pair at the scrollport: PushClip, EdgeGlow, PopClip, in that
        // order, with the band hugging the pressed edge.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 300px; }",
        );
        let (doc_state, _elements, mut tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");

        tree.scroll_chain(isc, 0.0, 200.0);
        tree.scroll_chain(isc, 0.0, 100.0);
        // The residual chains past the container to the viewport, so
        // both glow (viewport first, tree order).
        let _paint_root = tree.bake_paint_tree(&root);

        let mut list = DisplayList::new();
        list.push(DisplayItem::FillRect {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                w: 8.0,
                h: 8.0,
            },
            color: Color::TRANSPARENT,
        });
        tree.set_glow_split(list.items.len());
        tree.append_overscroll_glow(&mut list, Size { w: 800.0, h: 600.0 });
        // The real paint path appends scrollbars after the glow.
        for idx in 0..tree.len() {
            let geom = tree.scrollbar_geom(idx);
            if let Some(v) = geom.vertical {
                push_scrollbar(&mut list, v.track, v.thumb);
            }
            if let Some(h) = geom.horizontal {
                push_scrollbar(&mut list, h.track, h.thumb);
            }
        }
        // Content + (PushClip + EdgeGlow + PopClip) × 2 + #sc's bar.
        assert_eq!(list.items.len(), 9);

        // Viewport glow: glued to the screen — the page-coord rect is
        // the viewport translated by scroll minus the transient shift
        // (the backend's uniform subtracts it back out).
        let (_, vosy) = tree.overscroll_shift(0);
        assert!(matches!(
            list.items[1],
            DisplayItem::PushClip { rect }
                if approx_eq(rect.x, 0.0)
                    && approx_eq(rect.y, -vosy)
                    && approx_eq(rect.w, 800.0)
                    && approx_eq(rect.h, 600.0)
        ));
        match list.items[2] {
            DisplayItem::EdgeGlow {
                edge,
                rect,
                strength,
            } => {
                assert_eq!(edge, Edge4::Bottom);
                assert!((strength - vosy / OVERSCROLL_MAX_SHIFT).abs() < 1e-6);
                // Band hugs the viewport rect's bottom edge.
                assert!(approx_eq(rect.y + rect.h, 600.0 - vosy));
                assert!(approx_eq(rect.h, EDGE_GLOW_DEPTH.min(600.0)));
            }
            ref other => panic!("expected EdgeGlow, got {other:?}"),
        }
        assert!(matches!(list.items[3], DisplayItem::PopClip));

        // Nested container glow: clipped to the baked (post-shift)
        // scrollport, band hugging its bottom edge.
        let scrollport = tree.painted_scrollport(isc).expect("scrollport");
        assert!(matches!(
            list.items[4],
            DisplayItem::PushClip { rect } if rect == scrollport
        ));
        match list.items[5] {
            DisplayItem::EdgeGlow {
                edge,
                rect,
                strength,
            } => {
                assert_eq!(edge, Edge4::Bottom);
                assert!(strength > 0.0);
                assert!(approx_eq(rect.y + rect.h, scrollport.y + scrollport.h));
                assert!(approx_eq(rect.x, scrollport.x));
                assert!(approx_eq(rect.w, scrollport.w));
            }
            ref other => panic!("expected EdgeGlow, got {other:?}"),
        }
        assert!(matches!(list.items[6], DisplayItem::PopClip));

        // Epilogue re-push truncates back to the glow split and
        // re-emits glow + scrollbars identically: no item growth
        // across refreshes.
        tree.mark_glow_dirty();
        tree.refresh_epilogue(&mut list, Size { w: 800.0, h: 600.0 });
        assert_eq!(list.items.len(), 9, "truncate + re-push, no growth");
        assert!(!tree.glow_dirty(), "re-push clears the flag");
        // Scrollbars still come last, re-baked at the live geometry.
        let bar = tree.scrollbar_geom(isc).vertical.expect("sc bar");
        assert!(matches!(
            list.items[7],
            DisplayItem::FillRect { rect, .. } if rect == bar.track
        ));
        assert!(matches!(
            list.items[8],
            DisplayItem::FillRect { rect, .. } if rect == bar.thumb
        ));
    }

    #[test]
    fn bake_folds_overscroll_shift_into_painted_content() {
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer");
        let tall = doc.get_element_by_id("tall").expect("tall");
        let layout_outer = find_box(&root, outer).expect("outer in layout tree");
        let layout_tall = find_box(&root, tall).expect("tall in layout tree");

        // Park at the max edge, then press past it: the offset clamps
        // and the residual stretches the transient shift.
        tree.container(io).state.scroll_by(0.0, 200.0);
        let outcome = tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 500.0);
        assert!(outcome.has_edge_hits());
        let (osx, osy) = tree.overscroll_shift(io);
        assert_eq!(osx, 0.0);
        assert!(osy > 0.0, "edge press stretches the shift, got {osy}");

        let paint = tree.bake_paint_tree(&root).into_owned();
        // The scrollport box itself is unmoved...
        let paint_outer = find_box(&paint, outer).expect("outer in paint tree");
        assert_eq!(paint_outer.rect, layout_outer.rect);
        // ...while the content carries the offset shift plus the
        // overscroll shift, in the push direction.
        let paint_tall = find_box(&paint, tall).expect("tall in paint tree");
        assert!(
            (paint_tall.rect.y - (layout_tall.rect.y - 200.0 + osy)).abs() < 0.001,
            "painted {}, expected {}",
            paint_tall.rect.y,
            layout_tall.rect.y - 200.0 + osy,
        );
        assert_eq!(paint_tall.rect.x, layout_tall.rect.x);
    }

    #[test]
    fn sync_folds_overscroll_drift_without_scroll_events() {
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer");
        let tall = doc.get_element_by_id("tall").expect("tall");
        let layout_tall = find_box(&root, tall).expect("tall in layout tree");

        // Bake clean, then stretch: the drift is overscroll-only, no
        // offset movement.
        tree.container(io).state.scroll_by(0.0, 200.0);
        let mut paint = tree.bake_paint_tree(&root).into_owned();
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 500.0);
        let (_, osy) = tree.overscroll_shift(io);
        assert!(osy > 0.0);
        assert!(tree.has_nested_overscroll_in_flight());

        // The fold moves the painted content but reports no container
        // as moved: overscroll fires no `scroll` events.
        assert!(tree.sync_paint_offsets(&mut paint).is_empty());
        let paint_tall = find_box(&paint, tall).expect("tall");
        assert!(
            (paint_tall.rect.y - (layout_tall.rect.y - 200.0 + osy)).abs() < 0.001,
            "painted {}, expected {}",
            paint_tall.rect.y,
            layout_tall.rect.y - 200.0 + osy,
        );

        // Settle the spring to zero: the fold tracks it back, still
        // reporting no movement.
        let mut now = Instant::now() + Duration::from_millis(16);
        for _ in 0..1200 {
            now += Duration::from_millis(16);
            tree.advance_overscroll(now);
            if !tree.has_overscroll_in_flight() {
                break;
            }
        }
        assert!(!tree.has_overscroll_in_flight());
        assert!(!tree.has_nested_overscroll_in_flight());
        assert!(tree.sync_paint_offsets(&mut paint).is_empty());
        let paint_tall = find_box(&paint, tall).expect("tall");
        assert!(
            (paint_tall.rect.y - (layout_tall.rect.y - 200.0)).abs() < 0.001,
            "painted {}, expected {}",
            paint_tall.rect.y,
            layout_tall.rect.y - 200.0,
        );
    }

    #[test]
    fn viewport_overscroll_never_folds_into_paint_tree() {
        // The viewport's own shift rides the per-frame GPU uniform, so
        // `sync_paint_offsets` must not walk (or report) for it: the
        // cheap diff skips container 0.
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let mut paint = tree.bake_paint_tree(&root).into_owned();
        tree.scroll_chain(0, 0.0, 500.0);
        let (_, osy) = tree.overscroll_shift(0);
        assert!(osy > 0.0, "viewport edge press stretches, got {osy}");
        assert!(tree.has_overscroll_in_flight());
        assert!(!tree.has_nested_overscroll_in_flight());
        assert!(tree.sync_paint_offsets(&mut paint).is_empty());
    }

    #[test]
    fn fixed_under_fixed_cb_trigger_scrolls_with_scroller() {
        // A `position: fixed` box under a fixed-CB trigger (here a
        // non-`none` `transform`) inside a scroller takes its containing
        // block from that trigger and scrolls with the scroller:
        // layout's `scrollable_extent` counts its overflow toward the
        // range, and the paint bake shifts it with the content.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"t\"><div id=\"f\">x</div></div>\
              <div id=\"tall\">y</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; width: 200px; } \
             #t { transform: translate(0, 0); } \
             #f { position: fixed; top: 150px; left: 10px; width: 50px; height: 80px; } \
             #tall { height: 50px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");
        let f = doc.get_element_by_id("f").expect("f");
        let f_box = find_box(&root, f).expect("f in layout tree");

        // Extent agreement: the in-flow content is only 50px tall, so
        // any scroll range at all comes from the fixed box's overflow —
        // and the range matches its overflow below the scrollport.
        let max = tree.container(isc).state.max_offset();
        assert_eq!(max.0, 0.0);
        assert!(max.1 > 0.0, "fixed overflow extends the range, got {max:?}");
        let port = tree.container(isc).scrollport;
        let overflow = (f_box.rect.y + f_box.rect.h - (port.y + port.h)).max(0.0);
        assert!(
            (max.1 - overflow).abs() < 1.0,
            "range matches the fixed overflow: max {max:?}, overflow {overflow}"
        );

        // Bake agreement: a scroll moves the fixed box with the content.
        tree.container(isc).state.scroll_by(0.0, 50.0);
        let paint = tree.bake_paint_tree(&root).into_owned();
        let paint_f = find_box(&paint, f).expect("f in paint tree");
        assert!(
            (paint_f.rect.y - (f_box.rect.y - 50.0)).abs() < 0.001,
            "painted {}, expected {}",
            paint_f.rect.y,
            f_box.rect.y - 50.0,
        );
        assert_eq!(paint_f.rect.x, f_box.rect.x);

        // `sync_paint_offsets` folds later scrolls the same way, without
        // a rebake — reporting the container as moved, like any scroll.
        let mut paint = tree.bake_paint_tree(&root).into_owned();
        tree.container(isc).state.scroll_by(0.0, 25.0);
        assert_eq!(tree.sync_paint_offsets(&mut paint), vec![Some(sc)]);
        let paint_f = find_box(&paint, f).expect("f in paint tree");
        assert!(
            (paint_f.rect.y - (f_box.rect.y - 75.0)).abs() < 0.001,
            "painted {}, expected {}",
            paint_f.rect.y,
            f_box.rect.y - 75.0,
        );
    }

    #[test]
    fn viewport_anchored_fixed_ignores_nested_scroll() {
        // A `position: fixed` box with no fixed-CB trigger between it and
        // the scroller is viewport-anchored: layout's `scrollable_extent`
        // excludes its overflow from the range, and the paint bake leaves
        // it unshifted.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"f\">x</div>\
              <div id=\"tall\">y</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; width: 200px; } \
             #f { position: fixed; top: 150px; left: 10px; width: 50px; height: 80px; } \
             #tall { height: 300px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");
        let f = doc.get_element_by_id("f").expect("f");
        let tall = doc.get_element_by_id("tall").expect("tall");

        // Extent agreement: only the in-flow 300px counts (range 200);
        // the fixed box's overflow past the scrollport is excluded.
        assert_eq!(tree.container(isc).state.max_offset(), (0.0, 200.0));

        // Bake agreement: scrolled content moves, the fixed box does not.
        tree.container(isc).state.scroll_by(0.0, 50.0);
        let paint = tree.bake_paint_tree(&root).into_owned();
        let f_box = find_box(&root, f).expect("f in layout tree");
        let paint_f = find_box(&paint, f).expect("f in paint tree");
        assert_eq!(paint_f.rect, f_box.rect);
        let layout_tall = find_box(&root, tall).expect("tall in layout tree");
        let paint_tall = find_box(&paint, tall).expect("tall in paint tree");
        assert_eq!(paint_tall.rect.y, layout_tall.rect.y - 50.0);
    }

    #[test]
    fn scroll_chain_is_noop_without_layout() {
        let doc_state = Rc::new(ScrollState::new());
        let tree = ScrollTree::for_viewport(&doc_state, Size { w: 800.0, h: 600.0 }, (true, true));
        let o = tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 120.0);
        assert!(!o.moved());
        assert_eq!(doc_state.offset(), (0.0, 0.0));
    }

    #[test]
    fn document_max_scroll_excludes_nested_subtrees() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 5000px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        // The 5000px lives inside the nested scroller: the document's own
        // range only covers the 100px container box.
        assert_eq!(
            tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 }),
            ((0.0, 0.0), (0.0, 0.0))
        );
    }

    #[test]
    fn document_max_scroll_reports_negative_origin() {
        // Content above/left of the viewport origin: the document's
        // minimum offset goes negative while the max stays floored at
        // zero.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"neg\">x</div></body></html>",
            "#neg { position: relative; left: -30px; top: -50px; \
             width: 100px; height: 100px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let (origin, max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        assert_eq!(origin, (-30.0, -50.0));
        assert_eq!(max, (0.0, 0.0));
    }

    #[test]
    fn container_origin_plumbs_layout_scroll_origin() {
        // `ScrollContainer::origin` carries layout's `scroll_origin`:
        // negative when the container's content overflows above/left of
        // its scrollport, `(0, 0)` otherwise.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"inner\"><div id=\"neg\">x</div>\
              <div id=\"tall\">y</div></div></body></html>",
            "#inner { overflow: scroll; height: 100px; width: 200px; } \
             #neg { position: relative; left: -40px; top: -30px; height: 50px; } \
             #tall { height: 400px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let inner = doc.get_element_by_id("inner").expect("inner");
        let i = tree.index_of(inner).expect("inner container");
        assert_eq!(tree.container(i).origin, (-40.0, -30.0));
        // `relative` shifts #neg visually but not its flow footprint:
        // #tall still runs 400px from y=50, so max is 350.
        assert_eq!(tree.container(i).state.max_offset(), (0.0, 350.0));
    }

    #[test]
    fn negative_origin_publishes_min_and_reaches_negative_offsets() {
        // Wave 26 wiring: `ScrollTree::build` publishes each container's
        // scroll origin onto its shared `ScrollState`, so `scroll_chain`
        // can reach negative offsets and script-driven writes clamp into
        // `[origin, max]` exactly like the app's wheel/keyboard paths.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"inner\"><div id=\"neg\">x</div>\
              <div id=\"tall\">y</div></div></body></html>",
            "#inner { overflow: scroll; height: 100px; width: 200px; } \
             #neg { position: relative; left: -40px; top: -30px; height: 50px; } \
             #tall { height: 400px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let inner = doc.get_element_by_id("inner").expect("inner");
        let i = tree.index_of(inner).expect("inner container");
        let state = tree.container(i).state.clone();
        // The negative origin is published as the minimum bound.
        assert_eq!(state.min_offset(), (-40.0, -30.0));
        assert_eq!(state.max_offset(), (0.0, 350.0));

        // A negative wheel delta reaches a negative offset, clamped at
        // the origin rather than at zero.
        let o = tree.scroll_chain(i, 0.0, -50.0);
        assert_eq!(o.nodes(), vec![Some(inner)]);
        assert_eq!(state.offset(), (0.0, -30.0));

        // Script-driven writes clamp into `[origin, max]` the same way:
        // below the origin floors at the origin, past the max at the max.
        state.set_offset(-100.0, -100.0);
        assert_eq!(state.offset(), (-40.0, -30.0));
        state.set_offset(10.0, 400.0);
        assert_eq!(state.offset(), (0.0, 350.0));
    }

    #[test]
    fn nested_offsets_keys_each_scroller_by_dom_node() {
        // Two nested scrollers: `#outer` scrolls `#inner`'s 300px box,
        // `#inner` scrolls its 500px content. The map `refresh` feeds
        // layout must carry each container's live offset keyed by its
        // DOM node, viewport excluded, zeros omitted.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"outer\"><div id=\"inner\">\
              <div id=\"tall\">x</div></div></div></body></html>",
            "#outer { overflow-y: auto; height: 200px; } \
             #inner { overflow-y: auto; height: 300px; } \
             #tall { height: 500px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let inner = doc.get_element_by_id("inner").expect("inner");
        let oi = tree.index_of(outer).expect("outer is a container");
        let ii = tree.index_of(inner).expect("inner is a container");
        // Everything idle at zero: the map is empty (layout constrains
        // absent entries at zero, so this is exact, not a shortcut).
        assert!(tree.nested_offsets().is_empty());
        tree.container(oi).state.set_offset(0.0, 30.0);
        tree.container(ii).state.set_offset(0.0, 70.0);
        let map = tree.nested_offsets();
        assert_eq!(map.len(), 2);
        assert_eq!(map[&outer], (0.0, 30.0));
        assert_eq!(map[&inner], (0.0, 70.0));
        // Back to zero: the entry drops out again.
        tree.container(oi).state.set_offset(0.0, 0.0);
        let map = tree.nested_offsets();
        assert_eq!(map.len(), 1);
        assert_eq!(map[&inner], (0.0, 70.0));
    }

    #[test]
    fn build_records_whether_tree_has_sticky() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sticky\">s</div><div id=\"tall\">x</div></body></html>",
            "#sticky { position: sticky; top: 0; height: 50px; } #tall { height: 2000px; }",
        );
        let (_, _, tree) = build_tree(&doc, &root);
        assert!(tree.has_sticky());

        let (doc, root) = layout_html(
            b"<html><body><div id=\"tall\">x</div></body></html>",
            "#tall { height: 2000px; }",
        );
        let (_, _, tree) = build_tree(&doc, &root);
        assert!(!tree.has_sticky());

        // Pre-first-layout tree: no layout ran, so no sticky boxes.
        let state = Rc::new(ScrollState::new());
        let tree = ScrollTree::for_viewport(&state, Size { w: 800.0, h: 600.0 }, (true, true));
        assert!(!tree.has_sticky());
    }

    #[test]
    fn sticky_bands_skip_fixed_ancestor_subtrees() {
        // The only sticky box sits under a `fixed` ancestor: layout's
        // sticky pass applies no scroll-driven constraint there, so no
        // band — scrolling must not dirty layout for it.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"fix\"><div id=\"sticky\">s</div></div><div id=\"tall\">x</div></body></html>",
            "#fix { position: fixed; } #sticky { position: sticky; top: 0; height: 50px; } #tall { height: 2000px; }",
        );
        let (_, _, tree) = build_tree(&doc, &root);
        assert!(!tree.has_sticky());
        assert!(!tree.sticky_constrained_at(0, 0.0, 500.0));
    }

    #[test]
    fn sticky_band_absent_when_no_inset_can_stick() {
        // All insets `auto`: the scrollport clamp is the identity, so
        // no scroll offset can move the box — no band.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sticky\">s</div><div id=\"tall\">x</div></body></html>",
            "#sticky { position: sticky; height: 50px; } #tall { height: 2000px; }",
        );
        let (_, _, tree) = build_tree(&doc, &root);
        assert!(!tree.has_sticky());
    }

    #[test]
    fn sticky_band_covers_box_and_releases_far_away() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"pad\">x</div><div id=\"sticky\">s</div></body></html>",
            "#pad { height: 5000px; } #sticky { position: sticky; top: 0; height: 50px; }",
        );
        let (_, _, tree) = build_tree(&doc, &root);
        assert!(tree.has_sticky());
        // At the top of the page the 5000px-distant box is nowhere near
        // its constraint band: no re-bake needed.
        assert!(!tree.sticky_constrained_at(0, 0.0, 120.0));
        // Near the box (a scrollport above its natural position) the
        // band opens: the transition must be preceded by dirtying.
        assert!(tree.sticky_constrained_at(0, 0.0, 4600.0));
        // Far past it (containing block ended, box parked): released.
        assert!(!tree.sticky_constrained_at(0, 0.0, 20000.0));
    }

    #[test]
    fn reveal_node_scrolls_container_chain() {
        let (doc, root) = layout_html(
            b"<html><body><div style=\"height:700px\"></div>\
              <div id=\"sc\"><div id=\"tall\">x</div><div id=\"target\">t</div></div>\
              <div style=\"height:700px\"></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 300px; } \
             #target { height: 120px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let target = doc.get_element_by_id("target").expect("target");
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");

        let scrolled = tree.reveal_node(&root, target);
        // Innermost first: the nested container, then the viewport.
        assert_eq!(scrolled, vec![Some(sc), None]);
        // #target's top (y=1000) aligns to #sc's scrollport top (y=700).
        assert_eq!(tree.container(isc).state.offset(), (0.0, 300.0));
        // #sc's box top (y=700) aligns to the viewport top.
        assert_eq!(doc_state.offset(), (0.0, 700.0));

        // Unknown node: nothing scrolls.
        let (doc2, _) = Document::parse_html(b"<html><body></body></html>");
        let ghost = doc2.get_elements_by_tag_name("body").item(0).expect("body");
        assert!(tree.reveal_node(&root, ghost).is_empty());
    }

    #[test]
    fn container_for_node_walks_dom_ancestry() {
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let tall = doc.get_element_by_id("tall").expect("tall");
        let outer = doc.get_element_by_id("outer").expect("outer");
        assert_eq!(
            tree.container_for_node(tall, &doc),
            tree.index_of(outer).expect("outer")
        );
        let body = doc.get_elements_by_tag_name("body").item(0).expect("body");
        assert_eq!(tree.container_for_node(body, &doc), 0);
    }

    #[test]
    fn bake_and_sync_shift_painted_subtrees() {
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer");
        tree.container(io).state.scroll_by(0.0, 50.0);

        let paint = tree.bake_paint_tree(&root).into_owned();
        // The container's own box is unmoved (the scrollport is fixed)...
        let paint_outer = find_box(&paint, outer).expect("outer in paint tree");
        let layout_outer = find_box(&root, outer).expect("outer in layout tree");
        assert_eq!(paint_outer.rect, layout_outer.rect);
        // ...while its content shifted up by the offset.
        let tall = doc.get_element_by_id("tall").expect("tall");
        let paint_tall = find_box(&paint, tall).expect("tall in paint tree");
        let layout_tall = find_box(&root, tall).expect("tall in layout tree");
        assert_eq!(paint_tall.rect.y, layout_tall.rect.y - 50.0);
        assert_eq!(paint_tall.rect.x, layout_tall.rect.x);

        // A later script-driven write folds in via sync without a rebake.
        tree.container(io).state.scroll_by(0.0, 25.0);
        let mut paint = paint;
        assert_eq!(tree.sync_paint_offsets(&mut paint), vec![Some(outer)]);
        let paint_tall = find_box(&paint, tall).expect("tall");
        assert_eq!(paint_tall.rect.y, layout_tall.rect.y - 75.0);
        // Steady state: nothing left to fold.
        assert!(tree.sync_paint_offsets(&mut paint).is_empty());
    }

    #[test]
    fn publish_survives_relayout_and_prunes_gone_containers() {
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let elements = Rc::new(ElementScrollStates::new());
        let doc_state = Rc::new(ScrollState::new());
        let tree = ScrollTree::build(
            &doc,
            &root,
            Size { w: 800.0, h: 600.0 },
            &doc_state,
            &elements,
        );
        let outer = doc.get_element_by_id("outer").expect("outer");
        let state_before = tree
            .index_of(outer)
            .map(|i| tree.container(i).state.clone())
            .expect("outer state");
        state_before.scroll_by(0.0, 42.0);

        // Relayout: the same node keeps its shared state and offset.
        let tree2 = ScrollTree::build(
            &doc,
            &root,
            Size { w: 800.0, h: 600.0 },
            &doc_state,
            &elements,
        );
        let state_after = tree2
            .index_of(outer)
            .map(|i| tree2.container(i).state.clone())
            .expect("outer state after relayout");
        assert!(Rc::ptr_eq(&state_before, &state_after));
        assert_eq!(state_after.offset(), (0.0, 42.0));

        // The wide container's box goes away (new tree without it):
        // its state is pruned.
        let (doc3, root3) = layout_html(
            b"<html><body><div id=\"outer\"><div id=\"tall\">x</div></div></body></html>",
            "#outer { overflow-y: auto; height: 100px; } #tall { height: 300px; }",
        );
        let wide = doc.get_element_by_id("wide").expect("wide");
        assert!(elements.state_for(wide).is_some());
        let _tree3 = ScrollTree::build(
            &doc3,
            &root3,
            Size { w: 800.0, h: 600.0 },
            &doc_state,
            &elements,
        );
        assert!(elements.state_for(wide).is_none());
        // NodeIds are document-scoped, so the survivor is looked up in
        // the new document.
        let outer3 = doc3.get_element_by_id("outer").expect("outer in doc3");
        assert!(elements.state_for(outer3).is_some());
    }

    #[test]
    fn viewport_user_axes_honors_propagated_overflow_hidden() {
        // `body { overflow: hidden }` propagates to the viewport (CSS
        // Overflow 3 §2.4): wheel/keyboard must not scroll the document —
        // the scroll-lock pattern — while programmatic scrolls still work.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"tall\">x</div></body></html>",
            "body { overflow: hidden; } #tall { height: 2000px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        assert!(tree.is_viewport_only());
        assert_eq!(tree.container(0).user_scroll, (false, false));
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        assert_ne!(doc_state.max_offset(), (0.0, 0.0));

        // A wheel over the page moves nothing and reports nothing.
        let o = tree.scroll_chain(tree.innermost_at(400.0, 300.0), 0.0, 120.0);
        assert!(!o.moved());
        assert_eq!(doc_state.offset(), (0.0, 0.0));

        // Programmatic scrolling bypasses the mask: `hidden` only stops
        // user scrolls.
        doc_state.set_offset(0.0, 300.0);
        assert_eq!(doc_state.offset(), (0.0, 300.0));
    }

    #[test]
    fn viewport_user_axes_root_overflow_wins_over_body() {
        // The root element's non-`visible` values propagate instead of the
        // body's: `overflow-x: hidden` on `<html>` masks the x axis only.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"tallwide\">x</div></body></html>",
            "html { overflow-x: hidden; } body { overflow-y: auto; } \
             #tallwide { height: 2000px; width: 2000px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        assert!(tree.is_viewport_only());
        assert_eq!(tree.container(0).user_scroll, (false, true));
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);

        // A diagonal wheel: x is masked at the viewport, y scrolls it.
        let o = tree.scroll_chain(tree.innermost_at(400.0, 300.0), 50.0, 120.0);
        assert_eq!(o.nodes(), vec![None]);
        assert_eq!(doc_state.offset(), (0.0, 120.0));
    }

    #[test]
    fn wheel_chains_per_axis_past_masked_axes() {
        // `#m` is user-scrollable on y only (`overflow-x: hidden`): a
        // diagonal wheel's x component must chain past it untouched while
        // y is consumed — per-axis chaining through mixed `user_axes`.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"o\"><div id=\"m\"><div id=\"tw\">x</div></div>\
              <div id=\"ot\">y</div></div><div id=\"doctall\">z</div></body></html>",
            "#o { overflow: auto; height: 150px; width: 300px; } \
             #m { overflow-x: hidden; overflow-y: auto; height: 100px; width: 100px; } \
             #tw { height: 300px; width: 400px; } \
             #ot { height: 200px; } \
             #doctall { height: 2000px; width: 2000px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let m = doc.get_element_by_id("m").expect("m");
        let o = doc.get_element_by_id("o").expect("o");
        let im = tree.index_of(m).expect("m container");
        let io = tree.index_of(o).expect("o container");
        assert_eq!(tree.parent(im), io, "m chains into o");
        assert_eq!(tree.container(im).user_scroll, (false, true));
        assert_eq!(tree.container(im).state.max_offset(), (300.0, 200.0));
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);

        // Wheel over #m's scrollport: y is consumed by #m, x chains past
        // it (and past #o, which has no x room) to the viewport.
        let r = tree.container(im).scrollport;
        let (px, py) = (r.x + 10.0, r.y + 10.0);
        let out = tree.scroll_chain(tree.innermost_at(px, py), 50.0, 120.0);
        assert_eq!(out.nodes(), vec![Some(m), None]);
        assert_eq!(tree.container(im).state.offset(), (0.0, 120.0));
        assert_eq!(tree.container(io).state.offset(), (0.0, 0.0));
        assert_eq!(doc_state.offset(), (50.0, 0.0));

        // Pure-x wheel over #m: nothing consumes it until the viewport.
        let out = tree.scroll_chain(tree.innermost_at(px, py), 50.0, 0.0);
        assert_eq!(out.nodes(), vec![None]);
        assert_eq!(doc_state.offset(), (100.0, 0.0));
    }

    #[test]
    fn overscroll_value_maps_keywords_to_behaviors() {
        use bowser_css::{CssValue, LengthUnit};
        let kw = |s: &str| CssValue::Keyword(s.into());
        assert_eq!(
            overscroll_value(Some(&kw("auto"))),
            OverscrollBehavior::Auto
        );
        assert_eq!(
            overscroll_value(Some(&kw("contain"))),
            OverscrollBehavior::Contain
        );
        assert_eq!(
            overscroll_value(Some(&kw("none"))),
            OverscrollBehavior::None
        );
        // CSS keywords are ASCII case-insensitive.
        assert_eq!(
            overscroll_value(Some(&kw("NONE"))),
            OverscrollBehavior::None
        );
        // Missing or unparseable: the initial value, `auto`.
        assert_eq!(overscroll_value(None), OverscrollBehavior::Auto);
        assert_eq!(
            overscroll_value(Some(&CssValue::Length(1.0, LengthUnit::Px))),
            OverscrollBehavior::Auto
        );
    }

    #[test]
    fn overscroll_behavior_kills_chaining_per_axis() {
        // Every container defaults to `auto`/`auto` (chaining
        // propagates); `overscroll-behavior-y: none` on #m must kill the
        // y remainder at #m while x (still `auto`) chains past it to the
        // viewport.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"o\"><div id=\"m\"><div id=\"tw\">x</div></div>\
              <div id=\"ot\">y</div></div><div id=\"doctall\">z</div></body></html>",
            "#o { overflow: auto; height: 150px; width: 300px; } \
             #m { overflow-x: hidden; overflow-y: auto; height: 100px; width: 100px; \
                  overscroll-behavior-y: none; } \
             #tw { height: 300px; width: 400px; } \
             #ot { height: 200px; } \
             #doctall { height: 2000px; width: 2000px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let m = doc.get_element_by_id("m").expect("m");
        let im = tree.index_of(m).expect("m container");
        assert_eq!(
            tree.container(im).overscroll,
            (OverscrollBehavior::Auto, OverscrollBehavior::None),
            "longhand parsed from the container's own style"
        );
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);

        // #m consumes to its y max (200) and the 300px y remainder dies
        // instead of chaining to #o and the viewport, while x chains
        // past (auto) to the viewport.
        let r = tree.container(im).scrollport;
        let out = tree.scroll_chain(tree.innermost_at(r.x + 10.0, r.y + 10.0), 50.0, 500.0);
        assert_eq!(out.nodes(), vec![Some(m), None]);
        assert_eq!(tree.container(im).state.offset(), (0.0, 200.0));
        assert_eq!(doc_state.offset(), (50.0, 0.0), "y remainder died at #m");
    }

    #[test]
    fn overscroll_behavior_contain_matches_none_on_chaining() {
        // `contain` agrees with `none` on the chaining decision (the
        // distinction is only for the overscroll affordance): the
        // shorthand sets both axes, so both remainders die at #m.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"o\"><div id=\"m\"><div id=\"tw\">x</div></div>\
              <div id=\"ot\">y</div></div><div id=\"doctall\">z</div></body></html>",
            "#o { overflow: auto; height: 150px; width: 300px; } \
             #m { overflow-x: hidden; overflow-y: auto; height: 100px; width: 100px; \
                  overscroll-behavior: contain; } \
             #tw { height: 300px; width: 400px; } \
             #ot { height: 200px; } \
             #doctall { height: 2000px; width: 2000px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let m = doc.get_element_by_id("m").expect("m");
        let im = tree.index_of(m).expect("m container");
        assert_eq!(
            tree.container(im).overscroll,
            (OverscrollBehavior::Contain, OverscrollBehavior::Contain),
            "shorthand expands to both longhands"
        );
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);

        let r = tree.container(im).scrollport;
        let out = tree.scroll_chain(tree.innermost_at(r.x + 10.0, r.y + 10.0), 50.0, 500.0);
        assert_eq!(out.nodes(), vec![Some(m)]);
        assert_eq!(tree.container(im).state.offset(), (0.0, 200.0));
        assert_eq!(doc_state.offset(), (0.0, 0.0), "both remainders died at #m");
    }

    #[test]
    fn degenerate_origin_above_max_does_not_panic() {
        // A degenerate publish (`origin`/`min` above `max`) must not
        // panic: `scroll_chain` and `reveal_node_with_options` used to
        // clamp with raw `f32::clamp(origin, max)`, which panics on
        // `min > max`. Both now route through `ScrollState::set_offset`'s
        // degenerate-safe clamp, which collapses the range onto `max`.
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_doc_state, _elements, mut tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        let state = tree.container(io).state.clone();
        assert_eq!(state.max_offset(), (0.0, 200.0));
        // Publish the degenerate range: minimum above the maximum. The
        // re-clamp inside `set_min_offset` already collapses the offset
        // onto `max` (0, 200).
        state.set_min_offset((500.0, 500.0));
        tree.containers[io].origin = (500.0, 500.0);
        assert!(state.min_offset().1 > state.max_offset().1);
        assert_eq!(state.offset(), (0.0, 200.0));

        // `scroll_chain`: no panic; nothing moves on a collapsed range.
        let o = tree.scroll_chain(io, 0.0, 120.0);
        assert!(o.nodes().is_empty(), "nothing moves on a collapsed range");
        assert_eq!(state.offset(), (0.0, 200.0));

        // The reveal path clamps through the same logic: no panic, and
        // the offset stays on the collapsed range.
        let tall = doc.get_element_by_id("tall").expect("tall");
        tree.reveal_node(&root, tall);
        assert_eq!(state.offset(), (0.0, 200.0));
    }

    #[test]
    fn viewport_overscroll_falls_back_to_body_when_root_is_auto() {
        // CSS Overscroll Behavior §2: the viewport uses the root
        // element's `overscroll-behavior`, falling back per-axis to the
        // body's computed value when the root's is `auto`.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"tall\">x</div></body></html>",
            "body { overscroll-behavior-y: none; } #tall { height: 2000px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert_eq!(
            tree.container(0).overscroll,
            (OverscrollBehavior::Auto, OverscrollBehavior::None),
            "y's none propagates from body when the root is auto"
        );
    }

    #[test]
    fn viewport_overscroll_root_wins_over_body() {
        // The root's own non-`auto` value wins over the body's.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"tall\">x</div></body></html>",
            "html { overscroll-behavior: contain; } body { overscroll-behavior-y: none; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert_eq!(
            tree.container(0).overscroll,
            (OverscrollBehavior::Contain, OverscrollBehavior::Contain),
            "root's contain wins over body's none"
        );
    }

    #[test]
    fn scrolled_viewport_nested_box_routes_wheel_to_inner_first() {
        // The layout-to-app wiring: a nested `overflow: scroll` box
        // laid out inside a *scrolled* viewport produces two scrollers
        // (viewport + inner) with correct scrollport/overflow geometry
        // from `bowser_layout::scroll_containers`, and a wheel over the
        // inner box hits the inner scroller first.
        let html = b"<html><body><div id=\"inner\"><div id=\"tall\">x</div></div></body></html>";
        let (doc, _) = Document::parse_html(html);
        let sheet = parse_stylesheet(
            "#inner { overflow: scroll; height: 100px; width: 200px; } #tall { height: 400px; }",
        );
        let root = bowser_layout::layout_with_scroll(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
            (0.0, 50.0),
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert_eq!(tree.len(), 2, "viewport + one nested scroller");
        let inner = doc.get_element_by_id("inner").expect("inner");
        let idx = tree.index_of(inner).expect("inner is a container");
        let c = tree.container(idx);
        assert_eq!((c.scrollport.w, c.scrollport.h), (200.0, 100.0));
        assert_eq!(c.state.max_offset(), (0.0, 300.0));
        // Wheel over the inner box: the inner scroller consumes first.
        let outcome = tree.scroll_chain(
            tree.innermost_at(c.scrollport.x + 10.0, c.scrollport.y + 10.0),
            0.0,
            50.0,
        );
        assert!(outcome.moved());
        assert_eq!(outcome.nodes(), vec![Some(inner)]);
        assert_eq!(c.state.offset(), (0.0, 50.0));
    }

    #[test]
    fn build_keeps_fixed_container_scrollport_in_viewport_frame() {
        // A `position: fixed` container is laid out in viewport
        // coordinates; `ScrollTree::build` keeps its scrollport in that
        // frame (marked `viewport_anchored`) instead of translating it
        // once at build — the build-time translation went stale on the
        // next document scroll, mis-hit-testing by
        // (live − build-time) (finding #1). Query-time code translates
        // by the live viewport offset instead.
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"tall\">x</div><div id=\"fx\"><div id=\"in\">y</div></div></body></html>",
        );
        let sheet = parse_stylesheet(
            "#tall { height: 2000px; } \
             #fx { position: fixed; top: 10px; left: 10px; width: 100px; height: 100px; overflow: auto; } \
             #in { height: 500px; }",
        );
        let root = bowser_layout::layout_with_scroll(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
            (0.0, 0.0),
        );
        let fx = doc.get_element_by_id("fx").expect("fx");
        let doc_state = Rc::new(ScrollState::new());
        doc_state.set_max_offset((0.0, 1400.0));
        let elements = Rc::new(ElementScrollStates::new());
        let tree = ScrollTree::build(
            &doc,
            &root,
            Size { w: 800.0, h: 600.0 },
            &doc_state,
            &elements,
        );
        let idx = tree.index_of(fx).expect("fx is a container");
        let c = tree.container(idx);
        assert!(c.viewport_anchored, "fixed container is viewport-anchored");
        assert_eq!(
            (c.scrollport.x, c.scrollport.y),
            (10.0, 10.0),
            "fixed scrollport stays in viewport frame"
        );
        // The document scrolls later without a relayout (the stale
        // scenario): hit-testing must still find the fixed container
        // under its viewport rect.
        doc_state.set_offset(0.0, 500.0);
        assert_eq!(tree.innermost_at(50.0, 50.0), idx);
        // And a point outside its viewport rect misses it even though
        // the page-frame descent point moved.
        assert_eq!(tree.innermost_at(700.0, 500.0), 0);
        // Scrolling the fixed container itself does not move its
        // scrollport, and hit-testing still routes through its own
        // offset, not the document's.
        c.state.scroll_by(0.0, 200.0);
        assert_eq!(tree.innermost_at(50.0, 50.0), idx);
    }

    #[test]
    fn bake_and_sync_leave_fixed_descendants_unmoved() {
        // A `position: fixed` box inside a scrolling container is
        // viewport-anchored: baking the container's scroll into the
        // paint tree must not move it, while in-flow content shifts.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div>\
              <div id=\"fx\">f</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 300px; } \
             #fx { position: fixed; top: 5px; left: 5px; width: 50px; height: 50px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let fx = doc.get_element_by_id("fx").expect("fx");
        let tall = doc.get_element_by_id("tall").expect("tall");
        let isc = tree.index_of(sc).expect("sc container");
        // The fixed box really is a box-tree descendant of the
        // container here (out-of-flow boxes stay under their flow
        // parent), so the skip logic is what keeps it put.
        let layout_sc = find_box(&root, sc).expect("sc in layout tree");
        assert!(
            find_box(layout_sc, fx).is_some(),
            "fixed box is a box-tree descendant of the scroller"
        );
        let layout_fx = find_box(&root, fx).expect("fx in layout tree").rect;
        let layout_tall = find_box(&root, tall).expect("tall in layout tree").rect;

        tree.container(isc).state.scroll_by(0.0, 50.0);
        let paint = tree.bake_paint_tree(&root).into_owned();
        assert_eq!(
            find_box(&paint, fx).expect("fx in paint tree").rect,
            layout_fx,
            "fixed box does not scroll with its container"
        );
        assert_eq!(
            find_box(&paint, tall).expect("tall in paint tree").rect.y,
            layout_tall.y - 50.0,
            "in-flow content shifts with the scroll"
        );

        // A later script-driven write folds in via sync the same way.
        tree.container(isc).state.scroll_by(0.0, 25.0);
        let mut paint = paint;
        assert_eq!(tree.sync_paint_offsets(&mut paint), vec![Some(sc)]);
        assert_eq!(
            find_box(&paint, fx).expect("fx").rect,
            layout_fx,
            "fixed box still unmoved after sync"
        );
        assert_eq!(
            find_box(&paint, tall).expect("tall").rect.y,
            layout_tall.y - 75.0
        );
    }

    #[test]
    fn bake_skips_abspos_with_outside_containing_block() {
        // `cb_inside` rule, mirrored from `scrollable_extent`: an
        // abspos box whose containing block is outside the scroller
        // keeps its laid-out position; one whose containing block is
        // inside (via the positioned `#rel`) scrolls with the content.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div>\
              <div id=\"ap\">a</div>\
              <div id=\"rel\"><div id=\"ap2\">b</div></div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 300px; } \
             #ap { position: absolute; top: 0; left: 0; width: 50px; height: 20px; } \
             #rel { position: relative; height: 60px; } \
             #ap2 { position: absolute; top: 0; left: 0; width: 50px; height: 20px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let ap = doc.get_element_by_id("ap").expect("ap");
        let ap2 = doc.get_element_by_id("ap2").expect("ap2");
        let tall = doc.get_element_by_id("tall").expect("tall");
        let isc = tree.index_of(sc).expect("sc container");
        let layout_of = |n| find_box(&root, n).expect("box in layout tree").rect;

        tree.container(isc).state.scroll_by(0.0, 50.0);
        let paint = tree.bake_paint_tree(&root).into_owned();
        let paint_of = |n| find_box(&paint, n).expect("box in paint tree").rect;
        assert_eq!(
            paint_of(ap),
            layout_of(ap),
            "abspos against a CB outside the scroller does not move"
        );
        assert_eq!(
            paint_of(ap2).y,
            layout_of(ap2).y - 50.0,
            "abspos against a CB inside the scroller moves with content"
        );
        assert_eq!(paint_of(tall).y, layout_of(tall).y - 50.0);

        // Same through the sync path.
        tree.container(isc).state.scroll_by(0.0, 25.0);
        let mut paint = paint;
        tree.sync_paint_offsets(&mut paint);
        let paint_of = |n| find_box(&paint, n).expect("box in paint tree").rect;
        assert_eq!(paint_of(ap), layout_of(ap));
        assert_eq!(paint_of(ap2).y, layout_of(ap2).y - 75.0);
    }

    #[test]
    fn reveal_node_keeps_horizontal_offset_by_default() {
        // The historical `reveal_node` is vertical-only even when the
        // target sits far to the right of the scrollport.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"wide\">x</div>\
              <div id=\"target\">t</div></div></body></html>",
            "#sc { overflow: auto; width: 100px; height: 50px; } \
             #wide { width: 400px; height: 10px; } \
             #target { width: 50px; height: 10px; margin-left: 300px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let target = doc.get_element_by_id("target").expect("target");
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");
        assert_eq!(tree.container(isc).state.max_offset(), (300.0, 0.0));

        let scrolled = tree.reveal_node(&root, target);
        assert!(scrolled.is_empty(), "nothing to reveal vertically");
        assert_eq!(
            tree.container(isc).state.offset(),
            (0.0, 0.0),
            "horizontal offset kept"
        );
    }

    #[test]
    fn reveal_node_with_inline_start_scrolls_horizontally() {
        // `scrollIntoView({ inline: "start" })`: the horizontal path.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"wide\">x</div>\
              <div id=\"target\">t</div></div></body></html>",
            "#sc { overflow: auto; width: 100px; height: 50px; } \
             #wide { width: 400px; height: 10px; } \
             #target { width: 50px; height: 10px; margin-left: 300px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let target = doc.get_element_by_id("target").expect("target");
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");

        let scrolled = tree.reveal_node_with_options(
            &root,
            target,
            RevealAlign::Keep,
            RevealAlign::Start,
            ScrollBehavior::Instant,
        );
        assert_eq!(scrolled, vec![Some(sc)]);
        // The target's left edge is >= 300px right of the scrollport's;
        // the container's horizontal max is exactly 300.
        assert_eq!(tree.container(isc).state.offset(), (300.0, 0.0));
        // `block: Keep` left the vertical offset alone.
        assert_eq!(doc_state.offset(), (0.0, 0.0));
    }

    #[test]
    fn scroll_margins_resolve_parsed_longhands() {
        // `bowser_css` parses `scroll-margin`: the reader resolves the
        // longhands against the font-size and viewport bases, clamped at
        // zero (negative lengths are invalid at parse time and fall back
        // to the zero initial; the `.max(0.0)` in the reader is the
        // second layer, e.g. for hand-built styles).
        let (doc, root) = layout_html(
            b"<html><body><div id=\"t\">x</div><div id=\"neg\">y</div></body></html>",
            "html { font-size: 20px; } \
             #t { scroll-margin: 10px 20px; } \
             #neg { scroll-margin: -5px 0; }",
        );
        let style_of = |id: &str| {
            let n = doc.get_element_by_id(id).expect("target");
            find_box(&root, n).expect("box").style.as_deref()
        };
        assert_eq!(
            scroll_margins(style_of("t"), 20.0, 800.0, 600.0),
            (10.0, 20.0, 10.0, 20.0)
        );
        // Negative is unparseable: the longhands keep their zero initial.
        assert_eq!(
            scroll_margins(style_of("neg"), 20.0, 800.0, 600.0),
            (0.0, 0.0, 0.0, 0.0)
        );
        assert_eq!(
            scroll_margins(None, 20.0, 800.0, 600.0),
            (0.0, 0.0, 0.0, 0.0)
        );
        // `rem` resolves against the root element's computed font size:
        // the reader takes it as a parameter rather than hardcoding
        // 16.0, and `reveal_node_with_options` derives it from the
        // root's computed style the same way.
        let root_style = root.style.as_ref().expect("root style");
        assert_eq!(px_font_size(root_style, 16.0), 20.0);
    }

    #[test]
    fn scrollend_fires_after_a_quiet_tick() {
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_, _, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        // A fresh tree: quiet ticks end nothing.
        assert!(tree.end_tick().is_empty());
        // Mark the nested container and the viewport as moved this tick.
        tree.mark_scrolled(&[Some(outer), None]);
        // Same tick: still scrolling, nothing ends.
        assert!(tree.end_tick().is_empty());
        // Next quiet tick: both end, innermost-first.
        assert_eq!(tree.end_tick(), vec![Some(outer), None]);
        // Drained: further quiet ticks stay quiet.
        assert!(tree.end_tick().is_empty());
    }

    #[test]
    fn scrollend_mark_skips_unknown_nodes() {
        // A stale node id marks nothing: it must not panic and must not
        // be misattributed to the viewport scroller.
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_, _, tree) = build_tree(&doc, &root);
        let tall = doc.get_element_by_id("tall").expect("tall");
        assert!(tree.index_of(tall).is_none(), "tall is not a container");
        tree.mark_scrolled(&[Some(tall)]);
        assert!(tree.end_tick().is_empty());
        assert!(tree.end_tick().is_empty());
    }

    #[test]
    fn reveal_node_with_options_end_and_center() {
        let (doc, root) = layout_html(
            b"<html><body><div style=\"height:700px\"></div>\
              <div id=\"sc\"><div id=\"tall\">x</div><div id=\"target\">t</div></div>\
              <div style=\"height:700px\"></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 300px; } \
             #target { height: 120px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let target = doc.get_element_by_id("target").expect("target");
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");
        // #target spans y=1000..1120; #sc's scrollport is y=700..800.

        // `block: "end"`: the target's bottom edge aligns to the
        // scrollport's bottom edge: 1120-800 = 320 (the container max).
        let scrolled = tree.reveal_node_with_options(
            &root,
            target,
            RevealAlign::End,
            RevealAlign::Keep,
            ScrollBehavior::Instant,
        );
        assert_eq!(scrolled, vec![Some(sc), None]);
        assert_eq!(tree.container(isc).state.offset(), (0.0, 320.0));
        // The viewport aligns #sc's scrollport bottom (800) to its own
        // bottom (600): 200.
        assert_eq!(doc_state.offset(), (0.0, 200.0));

        // Reset; `block: "center"`: (1000+60)-(700+50) = 310 on #sc,
        // (700+50)-(0+300) = 450 on the viewport.
        tree.container(isc).state.set_offset(0.0, 0.0);
        doc_state.set_offset(0.0, 0.0);
        let scrolled = tree.reveal_node_with_options(
            &root,
            target,
            RevealAlign::Center,
            RevealAlign::Keep,
            ScrollBehavior::Instant,
        );
        assert_eq!(scrolled, vec![Some(sc), None]);
        assert_eq!(tree.container(isc).state.offset(), (0.0, 310.0));
        assert_eq!(doc_state.offset(), (0.0, 450.0));
    }

    #[test]
    fn reveal_nearest_scrolls_minimally() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div style=\"height:30px\"></div>\
              <div id=\"t2\">a</div><div id=\"t3\">b</div>\
              <div id=\"pad\"></div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } \
             #t2 { height: 40px; } #t3 { height: 40px; } #pad { height: 200px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let t2 = doc.get_element_by_id("t2").expect("t2");
        let t3 = doc.get_element_by_id("t3").expect("t3");
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");
        // #t2 spans y=30..70, #t3 y=70..110; #sc's scrollport is
        // y=0..100 (max 210). Scroll down 50: the visible range is
        // 50..150.
        tree.container(isc).state.set_offset(0.0, 50.0);

        // #t2 starts above the visible range: the leading edges align
        // (30-0 = 30).
        let scrolled = tree.reveal_node_with_options(
            &root,
            t2,
            RevealAlign::Nearest,
            RevealAlign::Keep,
            ScrollBehavior::Instant,
        );
        assert_eq!(scrolled, vec![Some(sc)]);
        assert_eq!(tree.container(isc).state.offset(), (0.0, 30.0));

        // #t3 (70..110) is fully inside 30..130 now: no movement.
        let scrolled = tree.reveal_node_with_options(
            &root,
            t3,
            RevealAlign::Nearest,
            RevealAlign::Keep,
            ScrollBehavior::Instant,
        );
        assert!(scrolled.is_empty());
        assert_eq!(tree.container(isc).state.offset(), (0.0, 30.0));

        // #t2 (30..70) is fully inside too: no movement.
        let scrolled = tree.reveal_node_with_options(
            &root,
            t2,
            RevealAlign::Nearest,
            RevealAlign::Keep,
            ScrollBehavior::Instant,
        );
        assert!(scrolled.is_empty());
    }

    #[test]
    fn smooth_scroll_animates_to_target_over_time() {
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        // #outer's max is (0, 200).
        tree.start_smooth_scroll(io, 0.0, 200.0);
        assert!(tree.has_smooth_in_flight());

        // Well before the 300ms duration: interpolating, not there yet.
        let t0 = Instant::now();
        let moved = tree.advance_smooth(t0 + Duration::from_millis(10));
        assert_eq!(moved, vec![Some(outer)]);
        let (x, y) = tree.container(io).state.offset();
        assert_eq!(x, 0.0);
        assert!(y > 0.0 && y < 200.0, "mid-animation offset: {y}");

        // Past the duration: lands exactly on target, animation done.
        let moved = tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(moved, vec![Some(outer)]);
        assert_eq!(tree.container(io).state.offset(), (0.0, 200.0));
        assert!(!tree.has_smooth_in_flight());

        // Steady state: no more movement.
        let moved = tree.advance_smooth(t0 + Duration::from_secs(20));
        assert!(moved.is_empty());
    }

    #[test]
    fn smooth_scroll_clamps_target_into_range() {
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        // Past the max (200): the target clamps, the animation still
        // completes.
        tree.start_smooth_scroll(io, 0.0, 9999.0);
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (0.0, 200.0));
        assert!(!tree.has_smooth_in_flight());
    }

    #[test]
    fn smooth_scroll_cancelled_by_external_write() {
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        tree.start_smooth_scroll(io, 0.0, 200.0);
        let t0 = Instant::now();
        let moved = tree.advance_smooth(t0 + Duration::from_millis(10));
        assert_eq!(moved, vec![Some(outer)]);

        // An instant scroll command wins: the next tick cancels the
        // animation instead of advancing it.
        tree.container(io).state.set_offset(0.0, 150.0);
        let moved = tree.advance_smooth(t0 + Duration::from_secs(10));
        assert!(
            moved.is_empty(),
            "a cancelled animation reports no movement"
        );
        assert!(!tree.has_smooth_in_flight());
        assert_eq!(tree.container(io).state.offset(), (0.0, 150.0));
    }

    #[test]
    fn smooth_scroll_replaced_by_newer_command() {
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        tree.start_smooth_scroll(io, 0.0, 200.0);
        // A newer smooth command on the same container replaces the
        // in-flight animation.
        tree.start_smooth_scroll(io, 0.0, 50.0);
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (0.0, 50.0));
        assert!(!tree.has_smooth_in_flight());
    }

    #[test]
    fn smooth_scroll_completion_feeds_scrollend() {
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        tree.start_smooth_scroll(io, 0.0, 200.0);
        let t0 = Instant::now();

        // Mid-animation tick: the movement stamps the container as
        // scrolling.
        let moved = tree.advance_smooth(t0 + Duration::from_millis(10));
        tree.mark_scrolled(&moved);
        assert!(
            tree.end_tick().is_empty(),
            "moved this tick: no scrollend yet"
        );

        // Completion tick: the last movement stamps again...
        let moved = tree.advance_smooth(t0 + Duration::from_secs(10));
        tree.mark_scrolled(&moved);
        assert!(tree.end_tick().is_empty());

        // ...and the first quiet tick fires `scrollend`.
        assert_eq!(tree.end_tick(), vec![Some(outer)]);
        assert!(tree.end_tick().is_empty());
    }

    #[test]
    fn css_scroll_behavior_defaults_containers() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
            "html { scroll-behavior: smooth; } \
             #sc { overflow-y: auto; height: 100px; scroll-behavior: smooth; } \
             #tall { height: 300px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert_eq!(tree.container(0).css_behavior, ScrollBehavior::Smooth);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");
        assert_eq!(tree.container(isc).css_behavior, ScrollBehavior::Smooth);
    }

    #[test]
    fn css_scroll_behavior_viewport_falls_back_to_body() {
        // Like `overflow`, the viewport takes the body's
        // `scroll-behavior` when the root element's is `auto`.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"tall\">x</div></body></html>",
            "body { scroll-behavior: smooth; } #tall { height: 2000px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert_eq!(tree.container(0).css_behavior, ScrollBehavior::Smooth);
    }

    #[test]
    fn css_scroll_behavior_defaults_to_auto() {
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert_eq!(tree.container(0).css_behavior, ScrollBehavior::Auto);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        assert_eq!(tree.container(io).css_behavior, ScrollBehavior::Auto);
    }

    #[test]
    fn reveal_with_smooth_behavior_animates_instead_of_jumping() {
        let (doc, root) = layout_html(
            b"<html><body><div style=\"height:700px\"></div>\
              <div id=\"sc\"><div id=\"tall\">x</div><div id=\"target\">t</div></div>\
              <div style=\"height:700px\"></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 300px; } \
             #target { height: 120px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let target = doc.get_element_by_id("target").expect("target");
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");

        // `behavior: "smooth"`: nothing moves synchronously; both
        // containers animate instead.
        let scrolled = tree.reveal_node_with_options(
            &root,
            target,
            RevealAlign::Start,
            RevealAlign::Keep,
            ScrollBehavior::Smooth,
        );
        assert!(scrolled.is_empty());
        assert!(tree.has_smooth_in_flight());
        assert_eq!(tree.container(isc).state.offset(), (0.0, 0.0));
        assert_eq!(doc_state.offset(), (0.0, 0.0));

        // Past the duration both land on the instant reveal's targets
        // (300 on #sc, 700 on the viewport), innermost-first.
        let t0 = Instant::now();
        let moved = tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(moved, vec![Some(sc), None]);
        assert_eq!(tree.container(isc).state.offset(), (0.0, 300.0));
        assert_eq!(doc_state.offset(), (0.0, 700.0));
        assert!(!tree.has_smooth_in_flight());
    }

    #[test]
    fn reveal_auto_behavior_defers_to_css_smooth_per_container() {
        let (doc, root) = layout_html(
            b"<html><body><div style=\"height:700px\"></div>\
              <div id=\"sc\"><div id=\"tall\">x</div><div id=\"target\">t</div></div>\
              <div style=\"height:700px\"></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; scroll-behavior: smooth; } \
             #tall { height: 300px; } #target { height: 120px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let target = doc.get_element_by_id("target").expect("target");
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");

        // `behavior: "auto"` defers per scroller: #sc is CSS-smooth
        // (animates), the viewport is CSS-auto (jumps).
        let scrolled = tree.reveal_node_with_options(
            &root,
            target,
            RevealAlign::Start,
            RevealAlign::Keep,
            ScrollBehavior::Auto,
        );
        assert_eq!(scrolled, vec![None]);
        assert_eq!(doc_state.offset(), (0.0, 700.0));
        assert!(tree.has_smooth_in_flight());
        assert_eq!(tree.container(isc).state.offset(), (0.0, 0.0));

        let t0 = Instant::now();
        let moved = tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(moved, vec![Some(sc)]);
        assert_eq!(tree.container(isc).state.offset(), (0.0, 300.0));
        assert!(!tree.has_smooth_in_flight());
    }

    #[test]
    fn scrollend_state_survives_rebuild() {
        // A relayout mid-gesture must not silently drop the pending
        // `scrollend`: the rebuilt tree carries the tick bookkeeping.
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (doc_state, elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        tree.mark_scrolled(&[Some(outer)]);
        let mut rebuilt = ScrollTree::build(
            &doc,
            &root,
            Size { w: 800.0, h: 600.0 },
            &doc_state,
            &elements,
        );
        rebuilt.carry_scroll_state(&tree);
        // The carried-over tick is still "this tick": nothing ends yet.
        assert!(rebuilt.end_tick().is_empty());
        // The quiet tick after the rebuild fires the gesture's `scrollend`.
        assert_eq!(rebuilt.end_tick(), vec![Some(outer)]);
    }

    const INITIAL_TARGET_HTML: &[u8] = b"<html><body>\
        <div id=\"scroller\"><div class=\"spacer\"></div>\
        <div class=\"box\" style=\"top:0;left:0\"></div>\
        <div class=\"box target\" style=\"top:200px;left:200px\"></div>\
        </div></body></html>";
    const INITIAL_TARGET_CSS: &str = "
        #scroller { width: 300px; height: 300px; overflow: scroll; position: absolute; }
        .spacer { width: 1000px; height: 1000px; }
        .box { position: absolute; width: 200px; height: 200px; }
        .target { scroll-initial-target: nearest; }
    ";

    #[test]
    fn initial_target_seats_once_on_first_carry() {
        // `scroll-initial-target: nearest` (CSS Scroll Snap 2
        // §3.1.2): the first carry seats a never-scrolled container
        // at its initial target (the center box at (200, 200),
        // aligned start) instead of leaving it at the origin — and
        // a later rebuild must not re-seat it after the user moves.
        let (doc, root) = layout_html(INITIAL_TARGET_HTML, INITIAL_TARGET_CSS);
        let scroller = doc.get_element_by_id("scroller").expect("scroller");
        let (doc_state, elements, mut tree) = build_tree(&doc, &root);
        let old = ScrollTree::for_viewport(&doc_state, Size { w: 800.0, h: 600.0 }, (true, true));
        tree.carry_scroll_state(&old);
        let state = elements.state_for(scroller).expect("state");
        assert_eq!(state.offset(), (200.0, 200.0));
        assert!(state.initial_target_seated());
        // A scroll moves it; a rebuild keeps the user's position.
        state.set_offset(50.0, 60.0);
        let mut rebuilt = ScrollTree::build(
            &doc,
            &root,
            Size { w: 800.0, h: 600.0 },
            &doc_state,
            &elements,
        );
        rebuilt.carry_scroll_state(&tree);
        assert_eq!(state.offset(), (50.0, 60.0));
    }

    #[test]
    fn initial_target_yields_to_pre_carry_scroll() {
        // A state moved before its first carry (a script scroll the
        // shim adopted, or an early user scroll) is not the
        // container's initial position anymore: the carry records
        // the seating decision without moving it.
        let (doc, root) = layout_html(INITIAL_TARGET_HTML, INITIAL_TARGET_CSS);
        let scroller = doc.get_element_by_id("scroller").expect("scroller");
        let (doc_state, elements, mut tree) = build_tree(&doc, &root);
        let state = elements.state_for(scroller).expect("state");
        state.set_offset(30.0, 40.0);
        let old = ScrollTree::for_viewport(&doc_state, Size { w: 800.0, h: 600.0 }, (true, true));
        tree.carry_scroll_state(&old);
        assert_eq!(state.offset(), (30.0, 40.0));
        assert!(state.initial_target_seated());
    }

    const DYN_INITIAL_TARGET_HTML: &[u8] = b"<html><body>\
        <div id=\"scroller\"><div class=\"spacer\"></div>\
        <div id=\"a\" class=\"box\" style=\"top:100px;left:100px\"></div>\
        <div id=\"b\" class=\"box\" style=\"top:300px;left:300px\"></div>\
        </div></body></html>";
    const DYN_INITIAL_TARGET_CSS: &str = "
        #scroller { width: 300px; height: 300px; overflow: scroll; position: absolute; }
        .spacer { width: 1000px; height: 1000px; }
        .box { position: absolute; width: 200px; height: 200px; }
    ";

    fn dyn_initial_target_css(target: Option<&str>) -> String {
        match target {
            Some(id) => {
                format!("{DYN_INITIAL_TARGET_CSS} #{id} {{ scroll-initial-target: nearest; }}")
            }
            None => DYN_INITIAL_TARGET_CSS.to_string(),
        }
    }

    fn dyn_initial_target_tree(
        doc: &Document,
        css: &str,
        doc_state: &Rc<ScrollState>,
        elements: &Rc<ElementScrollStates>,
    ) -> ScrollTree {
        let sheet = parse_stylesheet(css);
        let root = layout(
            doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        ScrollTree::build(doc, &root, Size { w: 800.0, h: 600.0 }, doc_state, elements)
    }

    /// First carry (from the pre-layout placeholder) for a document
    /// whose scroller's initial target is `target` (`"a"` at
    /// (100, 100), `"b"` at (300, 300), or no target).
    fn dyn_initial_target_setup(
        target: Option<&str>,
    ) -> (
        Document,
        Rc<ScrollState>,
        Rc<ElementScrollStates>,
        ScrollTree,
        Rc<ScrollState>,
    ) {
        let (doc, _) = Document::parse_html(DYN_INITIAL_TARGET_HTML);
        let doc_state = Rc::new(ScrollState::new());
        let elements = Rc::new(ElementScrollStates::new());
        let mut tree =
            dyn_initial_target_tree(&doc, &dyn_initial_target_css(target), &doc_state, &elements);
        let old = ScrollTree::for_viewport(&doc_state, Size { w: 800.0, h: 600.0 }, (true, true));
        tree.carry_scroll_state(&old);
        let scroller = doc.get_element_by_id("scroller").expect("scroller");
        let state = elements.state_for(scroller).expect("state");
        (doc, doc_state, elements, tree, state)
    }

    /// Rebuild with `target` as the scroller's initial target and
    /// carry `old`'s state into the fresh tree.
    fn dyn_initial_target_rebuild(
        doc: &Document,
        target: Option<&str>,
        doc_state: &Rc<ScrollState>,
        elements: &Rc<ElementScrollStates>,
        old: &ScrollTree,
    ) -> ScrollTree {
        let mut tree =
            dyn_initial_target_tree(doc, &dyn_initial_target_css(target), doc_state, elements);
        tree.carry_scroll_state(old);
        tree
    }

    #[test]
    fn initial_target_added_dynamically_seats_on_next_carry() {
        // No target at the first carry: the container rests at its
        // origin and the decision records `None`. Adding
        // `scroll-initial-target: nearest` afterwards re-seats it.
        let (doc, doc_state, elements, tree, state) = dyn_initial_target_setup(None);
        assert_eq!(state.offset(), (0.0, 0.0));
        assert_eq!(state.initial_target_seating(), Some(None));
        dyn_initial_target_rebuild(&doc, Some("a"), &doc_state, &elements, &tree);
        assert_eq!(state.offset(), (100.0, 100.0));
        assert_eq!(state.initial_target_seating(), Some(Some((100.0, 100.0))));
    }

    #[test]
    fn initial_target_moved_to_other_element_reseats() {
        // The property moves from #a to #b: the container still
        // rests at the old seated target, so it re-seats to the new
        // one.
        let (doc, doc_state, elements, tree, state) = dyn_initial_target_setup(Some("a"));
        assert_eq!(state.offset(), (100.0, 100.0));
        dyn_initial_target_rebuild(&doc, Some("b"), &doc_state, &elements, &tree);
        assert_eq!(state.offset(), (300.0, 300.0));
        assert_eq!(state.initial_target_seating(), Some(Some((300.0, 300.0))));
    }

    #[test]
    fn initial_target_removed_reseats_to_min_under_guard() {
        // The property is removed while the container still rests
        // at its seated target: it returns to its origin.
        let (doc, doc_state, elements, tree, state) = dyn_initial_target_setup(Some("a"));
        assert_eq!(state.offset(), (100.0, 100.0));
        dyn_initial_target_rebuild(&doc, None, &doc_state, &elements, &tree);
        assert_eq!(state.offset(), (0.0, 0.0));
        assert_eq!(state.offset(), state.min_offset());
        assert_eq!(state.initial_target_seating(), Some(None));
    }

    #[test]
    fn initial_target_change_does_not_yank_scrolled_container() {
        // The user scrolled away from the seated target before the
        // property moved: the changed target must not yank the
        // container. The rejected change is NOT recorded (the
        // recorded value stays the old target), so later rebuilds
        // re-evaluate it — but while the container remains scrolled
        // away they still must not yank it.
        let (doc, doc_state, elements, tree, state) = dyn_initial_target_setup(Some("a"));
        assert_eq!(state.offset(), (100.0, 100.0));
        state.set_offset(50.0, 60.0);
        let tree = dyn_initial_target_rebuild(&doc, Some("b"), &doc_state, &elements, &tree);
        assert_eq!(state.offset(), (50.0, 60.0));
        assert_eq!(state.initial_target_seating(), Some(Some((100.0, 100.0))));
        dyn_initial_target_rebuild(&doc, Some("b"), &doc_state, &elements, &tree);
        assert_eq!(state.offset(), (50.0, 60.0));
        assert_eq!(state.initial_target_seating(), Some(Some((100.0, 100.0))));
    }

    #[test]
    fn initial_target_rejected_reseat_seats_after_return_to_guard() {
        // A -> B is rejected while the user is scrolled away (not
        // recorded); when the container returns to exactly the old
        // seated target A — the guard position — the next carry
        // re-evaluates the still-unrecorded change and seats B.
        let (doc, doc_state, elements, tree, state) = dyn_initial_target_setup(Some("a"));
        assert_eq!(state.offset(), (100.0, 100.0));
        state.set_offset(50.0, 60.0);
        let tree = dyn_initial_target_rebuild(&doc, Some("b"), &doc_state, &elements, &tree);
        assert_eq!(state.offset(), (50.0, 60.0));
        assert_eq!(state.initial_target_seating(), Some(Some((100.0, 100.0))));
        state.set_offset(100.0, 100.0);
        dyn_initial_target_rebuild(&doc, Some("b"), &doc_state, &elements, &tree);
        assert_eq!(state.offset(), (300.0, 300.0));
        assert_eq!(state.initial_target_seating(), Some(Some((300.0, 300.0))));
    }

    #[test]
    fn initial_target_rejected_reseat_reevaluated_without_moving() {
        // A rejected change is re-evaluated on every subsequent
        // carry without ever moving the scrolled-away container
        // and without being recorded.
        let (doc, doc_state, elements, mut tree, state) = dyn_initial_target_setup(Some("a"));
        assert_eq!(state.offset(), (100.0, 100.0));
        state.set_offset(50.0, 60.0);
        for _ in 0..3 {
            tree = dyn_initial_target_rebuild(&doc, Some("b"), &doc_state, &elements, &tree);
            assert_eq!(state.offset(), (50.0, 60.0));
            assert_eq!(state.initial_target_seating(), Some(Some((100.0, 100.0))));
        }
    }

    /// Rebuild with `target` and carry from the pre-layout
    /// placeholder instead of a real tree: the scroller has no old
    /// match in the placeholder, so it is unmatched for this carry.
    fn dyn_initial_target_unmatched_rebuild(
        doc: &Document,
        target: Option<&str>,
        doc_state: &Rc<ScrollState>,
        elements: &Rc<ElementScrollStates>,
    ) -> ScrollTree {
        let mut tree =
            dyn_initial_target_tree(doc, &dyn_initial_target_css(target), doc_state, elements);
        let old = ScrollTree::for_viewport(doc_state, Size { w: 800.0, h: 600.0 }, (true, true));
        tree.carry_scroll_state(&old);
        tree
    }

    #[test]
    fn initial_target_unmatched_seated_records_without_moving() {
        // Previously seated at A, then a carry in which the
        // scroller has no old match (here the old tree is the
        // pre-layout placeholder) sees the target change to B: the
        // new value is recorded but the container is not moved.
        // Recording B makes it the guard baseline, so a subsequent
        // matched carry — with the container resting at B, the
        // guard position for the recorded value, and the target
        // changed again (back to A) — seats that new target. (A
        // matched carry with the *same* value B cannot re-seat:
        // by design an unchanged, already-recorded value never
        // re-seats.)
        let (doc, doc_state, elements, _tree, state) = dyn_initial_target_setup(Some("a"));
        assert_eq!(state.offset(), (100.0, 100.0));
        assert_eq!(state.initial_target_seating(), Some(Some((100.0, 100.0))));
        let unmatched =
            dyn_initial_target_unmatched_rebuild(&doc, Some("b"), &doc_state, &elements);
        assert_eq!(state.offset(), (100.0, 100.0));
        assert_eq!(state.initial_target_seating(), Some(Some((300.0, 300.0))));
        // Subsequent matched carry (the unmatched tree is a real
        // built tree containing the scroller, so the next rebuild
        // matches it) at the guard position (the recorded target
        // B) with a further target change seats that new target.
        state.set_offset(300.0, 300.0);
        dyn_initial_target_rebuild(&doc, Some("a"), &doc_state, &elements, &unmatched);
        assert_eq!(state.offset(), (100.0, 100.0));
        assert_eq!(state.initial_target_seating(), Some(Some((100.0, 100.0))));
    }

    #[test]
    fn initial_target_unchanged_across_carries_does_not_reseat() {
        // An unchanged target never re-seats: a rebuild while
        // resting at the target is a no-op, and a rebuild after the
        // user moved away leaves the user's position untouched.
        let (doc, doc_state, elements, tree, state) = dyn_initial_target_setup(Some("a"));
        assert_eq!(state.offset(), (100.0, 100.0));
        let tree = dyn_initial_target_rebuild(&doc, Some("a"), &doc_state, &elements, &tree);
        assert_eq!(state.offset(), (100.0, 100.0));
        state.set_offset(50.0, 60.0);
        dyn_initial_target_rebuild(&doc, Some("a"), &doc_state, &elements, &tree);
        assert_eq!(state.offset(), (50.0, 60.0));
    }

    // ------------------------------------------------------------------
    // Scroll snap (CSS Scroll Snap 1)
    // ------------------------------------------------------------------

    const SNAP_HTML: &[u8] = b"<html><body>\
        <div id=\"sc\"><div id=\"a\">x</div><div id=\"b\">y</div></div>\
        </body></html>";
    const SNAP_CSS: &str = "\
        #sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
        #a { height: 100px; scroll-snap-align: start; } \
        #b { height: 100px; scroll-snap-align: start; } \
    ";

    /// `#sc` from `SNAP_HTML`/`SNAP_CSS`: 100px scrollport, snap areas
    /// at content y=0 and y=100, max y=100.
    fn snap_tree() -> (Document, LayoutBox, NodeId, usize, ScrollTree) {
        let (doc, root) = layout_html(SNAP_HTML, SNAP_CSS);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let io = tree.index_of(sc).expect("sc container");
        (doc, root, sc, io, tree)
    }

    /// Point candidates (no snap overflow) with a shared stop value.
    fn point_cands(positions: &[f32], stop: SnapStop) -> Vec<SnapCandidate> {
        positions
            .iter()
            .map(|&p| SnapCandidate {
                pos: p,
                always: stop == SnapStop::Always,
                cover_lo: p,
                cover_hi: p,
            })
            .collect()
    }

    #[test]
    fn snap_areas_for_engine_retargets_shadow_descendants_to_host() {
        // A snap-aligned element inside a (closed) shadow root is
        // discovered as a snap area of the outer scroller, but the
        // engine reports the host as the target node — the snap event's
        // `snapTargetBlock` must be the host, never a node inside the
        // shadow tree.
        let (mut doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"host\"></div></div></body></html>",
        );
        let host = doc.get_element_by_id("host").expect("host");
        let frag = doc
            .attach_shadow_root(host, ShadowRootMode::Closed)
            .expect("attach");
        let kid = doc.create_element("div");
        doc.set_attribute(kid, "id", "kid");
        doc.append_child(frag, kid);
        let sheet = parse_stylesheet(
            "body { margin: 0; } \
             #sc { overflow-y: auto; height: 100px; } \
             #kid { height: 50px; scroll-snap-align: start; }",
        );
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let sc = doc.get_element_by_id("sc").expect("sc");
        let containers: HashSet<NodeId> = [sc].into_iter().collect();
        let areas = snap_areas_for_engine(&root, sc, &containers, &doc);
        assert_eq!(areas.len(), 1, "the shadow kid is a snap area");
        assert_eq!(areas[0].node, Some(host));
    }

    #[test]
    fn snap_target_proximity_nearest_within_threshold() {
        // Nearest candidate within one snapport dimension: snap.
        assert_eq!(
            snap_target_on_axis(
                &point_cands(&[100.0, 300.0], SnapStop::Normal),
                120.0,
                0.0,
                false,
                600.0,
                true,
                1800.0
            ),
            Some(100.0)
        );
    }

    #[test]
    fn snap_target_proximity_ignores_distant() {
        // The only candidate is 680px away, past the 600px snapport:
        // no snap.
        assert_eq!(
            snap_target_on_axis(
                &point_cands(&[800.0], SnapStop::Normal),
                120.0,
                0.0,
                false,
                600.0,
                true,
                1800.0
            ),
            None
        );
    }

    #[test]
    fn snap_target_mandatory_snaps_despite_distance() {
        assert_eq!(
            snap_target_on_axis(
                &point_cands(&[800.0], SnapStop::Normal),
                120.0,
                0.0,
                true,
                600.0,
                true,
                1800.0
            ),
            Some(800.0)
        );
    }

    #[test]
    fn snap_target_always_first_in_interval() {
        // `scroll-snap-stop: always` on a relative scroll must not pass
        // over a snap position: the first encountered from `pre` wins,
        // not the nearest to `natural`.
        assert_eq!(
            snap_target_on_axis(
                &point_cands(&[100.0, 200.0, 300.0], SnapStop::Always),
                250.0,
                0.0,
                true,
                600.0,
                true,
                1800.0
            ),
            Some(100.0)
        );
    }

    #[test]
    fn snap_target_always_excludes_start_point() {
        // Starting exactly on a snap position must not pin the scroll
        // there: the start point does not count as passed over.
        assert_eq!(
            snap_target_on_axis(
                &point_cands(&[100.0, 200.0, 300.0], SnapStop::Always),
                250.0,
                100.0,
                true,
                600.0,
                true,
                1800.0
            ),
            Some(200.0)
        );
    }

    #[test]
    fn snap_target_always_degrades_to_normal_when_interval_empty() {
        // No candidate in the traversed interval: normal selection.
        assert_eq!(
            snap_target_on_axis(
                &point_cands(&[200.0], SnapStop::Always),
                50.0,
                0.0,
                true,
                600.0,
                true,
                1800.0
            ),
            Some(200.0)
        );
        // ... including the proximity gate.
        assert_eq!(
            snap_target_on_axis(
                &point_cands(&[200.0], SnapStop::Always),
                50.0,
                0.0,
                false,
                600.0,
                true,
                1800.0
            ),
            Some(200.0)
        );
        assert_eq!(
            snap_target_on_axis(
                &point_cands(&[200.0], SnapStop::Always),
                50.0,
                0.0,
                false,
                100.0,
                true,
                300.0
            ),
            None
        );
    }

    #[test]
    fn snap_target_always_inert_for_absolute_scrolls() {
        // `scroll-snap-stop: always` has no effect on non-relative
        // scrolling operations: plain nearest selection.
        assert_eq!(
            snap_target_on_axis(
                &point_cands(&[100.0, 200.0, 300.0], SnapStop::Always),
                250.0,
                0.0,
                true,
                600.0,
                false,
                1800.0
            ),
            Some(200.0)
        );
    }

    #[test]
    fn snap_target_none_without_positions() {
        assert_eq!(
            snap_target_on_axis(&[], 120.0, 0.0, true, 600.0, true, 1800.0),
            None
        );
    }

    #[test]
    fn snap_target_stop_is_per_area() {
        // WPT scroll-snap-stop-001 shape: candidates 0 (normal),
        // 100 (always), 200 (normal); a relative scroll from 0 to 300
        // must stop at 100 — the per-area `always` traps even though
        // 200 is nearer the destination (the old container-level stop
        // read the *container's* `scroll-snap-stop: normal` and let
        // the scroll land on 200).
        let cands = vec![
            SnapCandidate {
                pos: 0.0,
                always: false,
                cover_lo: 0.0,
                cover_hi: 0.0,
            },
            SnapCandidate {
                pos: 100.0,
                always: true,
                cover_lo: 100.0,
                cover_hi: 100.0,
            },
            SnapCandidate {
                pos: 200.0,
                always: false,
                cover_lo: 200.0,
                cover_hi: 200.0,
            },
        ];
        assert_eq!(
            snap_target_on_axis(&cands, 300.0, 0.0, true, 400.0, true, 1200.0),
            Some(100.0)
        );
        // The same scroll as an absolute `scrollTo`: nearest wins.
        assert_eq!(
            snap_target_on_axis(&cands, 300.0, 0.0, true, 400.0, false, 1200.0),
            Some(200.0)
        );
    }

    #[test]
    fn snap_target_always_beyond_destination_does_not_trap() {
        // WPT scroll-snap-stop-002 scroller1 shape: candidates 0, 110
        // (normal), 120 (always); scrollBy(100) from 0 lands on 110 —
        // the `always` position is past the destination, so it never
        // traps, and the closest position to the destination wins.
        let cands = vec![
            SnapCandidate {
                pos: 0.0,
                always: false,
                cover_lo: 0.0,
                cover_hi: 0.0,
            },
            SnapCandidate {
                pos: 110.0,
                always: false,
                cover_lo: 110.0,
                cover_hi: 110.0,
            },
            SnapCandidate {
                pos: 120.0,
                always: true,
                cover_lo: 120.0,
                cover_hi: 120.0,
            },
        ];
        assert_eq!(
            snap_target_on_axis(&cands, 100.0, 0.0, true, 400.0, true, 1200.0),
            Some(110.0)
        );
    }

    #[test]
    fn snap_target_cover_range_keeps_destination() {
        // WPT scroll-snap-stop-002 scroller3 shape: an area larger
        // than the snapport (snap position 100, covering range
        // [100, 200]) makes the destination itself valid: a relative
        // scroll to 150 stays at 150, even with an `always` position
        // far beyond the destination.
        let cands = vec![
            SnapCandidate {
                pos: 0.0,
                always: false,
                cover_lo: 0.0,
                cover_hi: 0.0,
            },
            SnapCandidate {
                pos: 100.0,
                always: false,
                cover_lo: 100.0,
                cover_hi: 200.0,
            },
            SnapCandidate {
                pos: 600.0,
                always: false,
                cover_lo: 600.0,
                cover_hi: 600.0,
            },
            SnapCandidate {
                pos: 700.0,
                always: true,
                cover_lo: 700.0,
                cover_hi: 700.0,
            },
        ];
        // The covering range [100, 200] stays valid under §5.2.2's
        // second condition: the geometrically adjacent snap positions
        // are 100 and 600, 500px apart — more than the 400px snapport.
        assert_eq!(
            snap_target_on_axis(&cands, 150.0, 0.0, true, 400.0, true, 400.0),
            Some(150.0)
        );
    }

    #[test]
    fn snap_target_always_cover_area_traps_at_its_position() {
        // WPT scroll-snap-stop-002 scroller5 shape: the covering area
        // itself is `always`, so a relative scroll from 0 to 150 stops
        // at the area's snap position (100) instead of riding into the
        // covering range; once resting at 100, a further scroll to 150
        // is NOT trapped (the start point is not passed over) and the
        // covering destination is valid — the adjacent snap positions
        // (100, 600) are 500px apart, more than the 400px snapport, so
        // §5.2.2's second condition holds.
        let cands = vec![
            SnapCandidate {
                pos: 0.0,
                always: false,
                cover_lo: 0.0,
                cover_hi: 0.0,
            },
            SnapCandidate {
                pos: 100.0,
                always: true,
                cover_lo: 100.0,
                cover_hi: 200.0,
            },
            SnapCandidate {
                pos: 600.0,
                always: false,
                cover_lo: 600.0,
                cover_hi: 600.0,
            },
            SnapCandidate {
                pos: 700.0,
                always: true,
                cover_lo: 700.0,
                cover_hi: 700.0,
            },
        ];
        assert_eq!(
            snap_target_on_axis(&cands, 150.0, 0.0, true, 400.0, true, 400.0),
            Some(100.0)
        );
        assert_eq!(
            snap_target_on_axis(&cands, 150.0, 100.0, true, 400.0, true, 400.0),
            Some(150.0)
        );
    }

    #[test]
    fn snap_target_cover_range_invalid_when_positions_closer_than_snapport() {
        // CSS Scroll Snap 1 §5.2.2's second condition: the outer area's
        // covering range [0, 800] is invalid because the geometrically
        // adjacent snap positions (0 and 100) are 100px apart — less
        // than the 200px snapport. A rest at 190 therefore falls back
        // to the nearest ordinary snap position (100), not the natural
        // rest. This is the `scrollsnapchange-after-layout-change`
        // subtest 1 shape; that tentative test expects 200, which no
        // standard formula yields (Chromium's EndPositionStrategy also
        // predicts 190) — 100 is the spec-grounded answer.
        let cands = vec![
            SnapCandidate {
                pos: 0.0,
                always: false,
                cover_lo: 0.0,
                cover_hi: 800.0,
            },
            SnapCandidate {
                pos: 100.0,
                always: false,
                cover_lo: 100.0,
                cover_hi: 100.0,
            },
        ];
        assert_eq!(
            snap_target_on_axis(&cands, 190.0, 0.0, true, 200.0, false, 200.0),
            Some(100.0)
        );
    }

    #[test]
    fn snap_target_cover_range_valid_when_positions_farther_than_snapport() {
        // Same shape, but the adjacent snap positions (0 and 600) are
        // 600px apart — more than the 200px snapport — so §5.2.2's
        // second condition holds and the rest at 190 stays put.
        let cands = vec![
            SnapCandidate {
                pos: 0.0,
                always: false,
                cover_lo: 0.0,
                cover_hi: 800.0,
            },
            SnapCandidate {
                pos: 600.0,
                always: false,
                cover_lo: 600.0,
                cover_hi: 600.0,
            },
        ];
        assert_eq!(
            snap_target_on_axis(&cands, 190.0, 0.0, true, 200.0, false, 200.0),
            Some(190.0)
        );
    }

    #[test]
    fn snap_wheel_mandatory_snaps_to_start() {
        let (_doc, _root, sc, io, tree) = snap_tree();
        // Wheel 30px down a mandatory snap container: the natural rest
        // (30) is nearest snap area #a at 0, so a follow-up smooth
        // scroll launches back to 0.
        let o = tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 30.0);
        assert!(o.moved());
        assert_eq!(tree.container(io).state.offset(), (0.0, 30.0));
        assert!(tree.has_smooth_in_flight());

        // Completing the snap lands exactly on the snap position, and
        // no further snap launches (the target is the natural rest).
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (0.0, 0.0));
        assert!(!tree.has_smooth_in_flight());
        assert_eq!(o.nodes(), vec![Some(sc)]);
    }

    #[test]
    fn snap_proximity_no_followup_when_far() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div>\
              <div id=\"gap\"></div><div id=\"b\">y</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; scroll-snap-type: y proximity; } \
             #a { height: 100px; scroll-snap-align: start; } \
             #gap { height: 300px; } \
             #b { height: 100px; scroll-snap-align: start; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let io = tree.index_of(sc).expect("sc container");
        // Snap positions at 0 and 400; resting at 200 is 200px from
        // both — past the 100px snapport — so proximity does not snap.
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 200.0);
        assert_eq!(tree.container(io).state.offset(), (0.0, 200.0));
        assert!(!tree.has_smooth_in_flight());
    }

    /// Wave-50 regression (WPT
    /// `snap-after-relayout/not-resnap-outside-proximity-threshold`):
    /// the proximity reach is one third of the snapport, not the whole
    /// snapport. In a 600px snapport with snap positions at 0 and 500,
    /// a rest at 250 is 250px from the nearest position — outside the
    /// 200px reach — so neither the scroll evaluation nor a relayout
    /// resnap may move it; a rest at 190 is inside the reach and snaps.
    #[test]
    fn proximity_reach_is_a_third_of_the_snapport() {
        assert_eq!(proximity_threshold(600.0), 200.0);
        assert_eq!(proximity_threshold(300.0), 100.0);
        const CSS: &str = "\
            #sc { overflow: hidden; height: 600px; width: 600px; \
                  scroll-snap-type: y proximity; } \
            #a { height: 300px; scroll-snap-align: start; } \
            #gap { height: 200px; } \
            #b { height: 300px; scroll-snap-align: start; } \
            #tail { height: 1200px; }";
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div>\
              <div id=\"gap\"></div><div id=\"b\">y</div>\
              <div id=\"tail\"></div></div></body></html>",
            CSS,
        );
        let (doc_state, elements, tree) = build_tree(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let io = tree.index_of(sc).expect("sc container");
        // Scroll evaluation: resting at 250 stays (250 > 200 reach).
        tree.container(io).state.set_offset(0.0, 250.0);
        tree.eval_snap(io, (0.0, 0.0), (0.0, 250.0), false);
        assert_eq!(tree.container(io).state.offset(), (0.0, 250.0));
        assert!(!tree.has_smooth_in_flight());
        assert!(tree.container(io).snap_rest.get().1.is_none());
        // Relayout resnap: the unsnapped rest at 250 survives an
        // identical rebuild (the WPT file mutates a snap area's `top`
        // and asserts the offset does not move).
        let tree = snap_relayout(&doc, CSS, &doc_state, &elements, &tree);
        let io = tree.index_of(sc).expect("sc container");
        assert_eq!(tree.container(io).state.offset(), (0.0, 250.0));
        assert!(tree.container(io).snap_rest.get().1.is_none());
        // Inside the reach: resting at 190 snaps back to 0.
        tree.container(io).state.set_offset(0.0, 190.0);
        tree.eval_snap(io, (0.0, 250.0), (0.0, 190.0), false);
        assert!(tree.has_smooth_in_flight());
        tree.advance_smooth(Instant::now() + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (0.0, 0.0));
    }

    #[test]
    fn snap_mandatory_snaps_despite_distance() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div>\
              <div id=\"gap\"></div><div id=\"b\">y</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
             #a { height: 100px; scroll-snap-align: start; } \
             #gap { height: 300px; } \
             #b { height: 100px; scroll-snap-align: start; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let io = tree
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc container");
        // Resting at 210: nearest snap position is 400 (190 < 210).
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 210.0);
        assert!(tree.has_smooth_in_flight());
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (0.0, 400.0));
        assert!(!tree.has_smooth_in_flight());
    }

    #[test]
    fn snap_center_align_targets_midpoint() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"sp\"></div>\
              <div id=\"a\">x</div><div id=\"tail\"></div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
             #sp { height: 140px; } \
             #a { height: 60px; scroll-snap-align: center; } \
             #tail { height: 200px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let io = tree
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc container");
        // Center alignment: the offset putting #a's midpoint (140+30)
        // on the snapport's midpoint (50) is 120.
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 150.0);
        assert!(tree.has_smooth_in_flight());
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (0.0, 120.0));
    }

    #[test]
    fn snap_interruption_drops_snap_animation() {
        let (_doc, _root, _sc, io, tree) = snap_tree();
        // A wheel starts a snap follow-up back toward 0...
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 30.0);
        assert!(tree.has_smooth_in_flight());
        // ...then an external write moves the container: the in-flight
        // snap is dropped (interruption rule) and the interrupting
        // scroll is snap-evaluated itself — resting at 80 snaps to 100.
        tree.container(io).state.set_offset(0.0, 80.0);
        let t0 = Instant::now();
        let moved = tree.advance_smooth(t0 + Duration::from_secs(10));
        assert!(moved.is_empty(), "dropped animation moves nothing");
        assert!(tree.has_smooth_in_flight(), "re-snap launched");
        tree.advance_smooth(t0 + Duration::from_secs(20));
        assert_eq!(tree.container(io).state.offset(), (0.0, 100.0));
        assert!(!tree.has_smooth_in_flight());
    }

    #[test]
    fn snap_scrollend_waits_for_snap_completion() {
        let (_doc, _root, sc, io, tree) = snap_tree();
        // Turn 1: the wheel gesture; the snap follow-up is in flight.
        let o = tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 30.0);
        tree.mark_scrolled(&o.nodes());
        assert!(tree.end_tick().is_empty());

        // Turn 2: the snap animation is still moving — no scrollend yet.
        let t0 = Instant::now();
        let moved = tree.advance_smooth(t0 + Duration::from_millis(10));
        tree.mark_scrolled(&moved);
        assert!(tree.end_tick().is_empty());

        // Turn 3: the snap completes.
        let moved = tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (0.0, 0.0));
        tree.mark_scrolled(&moved);
        assert!(tree.end_tick().is_empty());

        // Turn 4: quiet — now the scrollend fires, after the snap.
        assert!(tree.advance_smooth(t0 + Duration::from_secs(20)).is_empty());
        assert_eq!(tree.end_tick(), vec![Some(sc)]);
    }

    #[test]
    fn snap_mark_scrolled_covers_script_writes() {
        let (_doc, _root, sc, io, tree) = snap_tree();
        // A script-driven instant write bypasses `scroll_chain`; the
        // `mark_scrolled` backstop still snap-evaluates it (as a
        // non-relative scroll): resting at 60 snaps to 100.
        tree.container(io).state.set_offset(0.0, 60.0);
        tree.mark_scrolled(&[Some(sc)]);
        assert!(tree.has_smooth_in_flight());
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (0.0, 100.0));
    }

    #[test]
    fn snap_padding_shifts_snapport() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"sp\"></div>\
              <div id=\"a\">x</div><div id=\"tail\"></div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; \
                     scroll-padding-top: 30px; } \
             #sp { height: 100px; } \
             #a { height: 100px; scroll-snap-align: start; } \
             #tail { height: 100px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let io = tree
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc container");
        // The 30px top padding moves the snapport's start edge down:
        // start-aligning #a (content y=100) targets 70, not 100.
        assert_eq!(tree.container(io).snap.padding, (30.0, 0.0, 0.0, 0.0));
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 30.0);
        assert!(tree.has_smooth_in_flight());
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (0.0, 70.0));
    }

    #[test]
    fn viewport_snap_areas_feed_eval_snap() {
        // The viewport scroller (container 0) is not part of
        // `discovered`, but its snap areas must still be collected:
        // without them `window.scrollTo()` never snaps.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"t1\">x</div><div id=\"t2\">y</div></body></html>",
            "html { scroll-snap-type: y mandatory; } \
             #t1 { height: 500px; scroll-snap-align: start; } \
             #t2 { height: 500px; scroll-snap-align: start; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let areas = &tree.container(0).snap_areas;
        assert_eq!(areas.len(), 2);
        assert_eq!(areas[0].align_y, SnapAlign::Start);
        // The document is 1000px tall against a 600px viewport (max
        // 400): a script-driven write resting at 300 snaps toward 500,
        // clamped into the real [origin, max] on the way (spec §5.3's
        // "unreachable snap positions" rule: snap as much as
        // possible), landing on the maximum 400.
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        doc_state.set_offset(0.0, 300.0);
        tree.mark_scrolled(&[None]);
        assert!(tree.has_smooth_in_flight());
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(doc_state.offset(), (0.0, 400.0));
    }

    #[test]
    fn snap_stop_always_read_from_target_style() {
        // WPT scroll-snap-stop-001's shape: the `always` lives on the
        // *target* (`.always-stop`), not the container. A relative
        // scroll from 0 by 300 must stop at 100.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"space\"></div>\
              <div class=\"target origin\"></div>\
              <div class=\"target always-stop\"></div><div class=\"target closer\"></div>\
              </div></body></html>",
            "div { position: absolute; } \
             #sc { width: 400px; height: 400px; overflow: scroll; \
                   scroll-snap-type: both mandatory; position: absolute; } \
             #space { left: 0px; top: 0px; width: 2100px; height: 2100px; } \
             .target { width: 50px; height: 50px; scroll-snap-align: start; } \
             .origin { left: 0px; top: 0px; } \
             .always-stop { left: 100px; top: 0px; scroll-snap-stop: always; } \
             .closer { left: 200px; top: 0px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let io = tree
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc container");
        let always = tree
            .container(io)
            .snap_areas
            .iter()
            .find(|a| a.rect.x == 100.0);
        assert_eq!(
            always.map(|a| a.stop),
            Some(SnapStop::Always),
            "the 100px target's stop is recovered"
        );
        // Relative scroll (wheel/keyboard semantics) from 0 by 300:
        // must not pass the `always` position.
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 300.0, 0.0);
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (100.0, 0.0));
        // The same travel as an absolute write: nearest wins.
        tree.container(io).state.set_offset(0.0, 0.0);
        tree.eval_snap(io, (0.0, 0.0), (300.0, 0.0), false);
        tree.advance_smooth(t0 + Duration::from_secs(20));
        assert_eq!(tree.container(io).state.offset(), (200.0, 0.0));
    }

    #[test]
    fn snap_end_center_use_scrollport_edges_with_padding() {
        // `scroll-padding` insets the snapport: end/center positions
        // measure against its edges, not the scrollport's (the old
        // formulas dropped the padding on those alignments).
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div><div id=\"tail\"></div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; \
                   scroll-padding-top: 30px; scroll-padding-bottom: 10px; } \
             #a { height: 50px; margin-top: 200px; scroll-snap-align: end; } \
             #tail { height: 300px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let io = tree
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc container");
        // Snapport height 60 (100 - 30 - 10); the area's end edge at
        // 250 aligns with the snapport's end edge at offset o when
        // 250 - o = 100 - 10, i.e. o = 160.
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 140.0);
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (0.0, 160.0));
    }

    #[test]
    fn snap_cover_range_destination_stays() {
        // WPT scroll-snap-stop-002 scroller3 shape, driven end to end:
        // a 500px-wide area in a 400px snapport covers it over offsets
        // [100, 200], so a relative scroll resting at 150 stays at 150
        // (snap overflow) instead of being dragged to a position.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"space\"></div>\
              <div class=\"t\" style=\"left:0px\"></div>\
              <div class=\"t big\" style=\"left:100px\"></div>\
              <div class=\"t\" style=\"left:600px\"></div>\
              <div class=\"t\" style=\"left:700px\"></div></div></body></html>",
            "div { position: absolute; } \
             #sc { width: 400px; height: 100px; overflow: scroll; \
                   scroll-snap-type: x mandatory; position: absolute; } \
             #space { left: 0px; top: 0px; width: 2100px; height: 50px; } \
             .t { width: 50px; height: 50px; top: 0px; scroll-snap-align: start; } \
             .big { width: 500px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let io = tree
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc container");
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 150.0, 0.0);
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (150.0, 0.0));
        assert!(!tree.has_smooth_in_flight());
    }

    #[test]
    fn snap_reversed_axis_rtl_swaps_start_end() {
        // Under `direction: rtl` the query's logical `start` is the
        // physical right edge: aligning it with the snapport's start
        // (right) edge is the LTR end-edge formula.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"t\">x</div></div></body></html>",
            "#sc { overflow-x: auto; width: 400px; height: 100px; direction: rtl; \
                   scroll-snap-type: x mandatory; position: absolute; } \
             #t { width: 600px; height: 50px; position: absolute; left: -500px; \
                 scroll-snap-align: start; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let io = tree
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc container");
        assert!(
            tree.container(io).snap_areas.iter().any(|a| a.rev_x),
            "the target's area is marked reversed"
        );
        // Start-aligning the target's right edge (-500 + 600 = 100)
        // with the snapport's right edge (offset + 400): o = -300.
        // The 600px area covers the 400px snapport over [-500, -300];
        // resting outside that range, at -250, must be pulled to the
        // covering edge -300 (without the start/end swap the engine
        // would aim at the left-edge position -500 instead).
        tree.container(io).state.set_offset(-250.0, 0.0);
        tree.eval_snap(io, (0.0, 0.0), (-250.0, 0.0), true);
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (-300.0, 0.0));
    }

    #[test]
    fn snap_dust_does_not_launch() {
        let (_doc, _root, _sc, io, tree) = snap_tree();
        // Sub-`SNAP_EPS` movement is not a scroll worth snapping after,
        // but it is still recorded as evaluated.
        tree.eval_snap(io, (0.0, 0.0), (0.0, 1.0 / 8192.0), true);
        assert!(!tree.has_smooth_in_flight());
        assert_eq!(tree.container(io).snap_base.get(), (0.0, 1.0 / 8192.0));
    }

    #[test]
    fn snap_areas_expand_by_scroll_margin() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
             #a { height: 100px; scroll-snap-align: start; scroll-margin-top: 20px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let io = tree
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc container");
        let areas = &tree.container(io).snap_areas;
        assert_eq!(areas.len(), 1);
        // #a sits at the top of the content (content y=0): the 20px
        // top scroll-margin expands the area upward and grows it.
        assert_eq!(areas[0].rect.y, -20.0);
        assert_eq!(areas[0].rect.h, 120.0);
        assert_eq!(areas[0].align_y, SnapAlign::Start);
    }

    #[test]
    fn snap_areas_pruned_at_nested_containers() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"o\"><div id=\"i\"><div id=\"deep\">x</div>\
              </div></div></body></html>",
            "#o { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
             #i { overflow-y: auto; height: 80px; } \
             #deep { height: 200px; scroll-snap-align: start; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let oi = tree
            .index_of(doc.get_element_by_id("o").expect("o"))
            .expect("o container");
        let ii = tree
            .index_of(doc.get_element_by_id("i").expect("i"))
            .expect("i container");
        // #deep's subtree belongs to the inner container: the outer
        // container sees no snap areas from it.
        assert!(tree.container(oi).snap_areas.is_empty());
        assert_eq!(tree.container(ii).snap_areas.len(), 1);
    }

    #[test]
    fn snap_real_layout_query_feeds_snapped_end_position() {
        // The areas below come from the real
        // `bowser_layout::snap_areas_for_container` query (border box
        // expanded by `scroll-margin`, in container content coordinates,
        // with `(align_x, align_y)` already in physical axes per the
        // target's writing mode) and drive `eval_snap` to a snapped
        // scroll end position.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"sp\"></div>\
              <div id=\"a\">x</div><div id=\"tail\"></div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
             #sp { height: 100px; } \
             #a { height: 100px; scroll-snap-align: start end; scroll-margin-top: 30px; } \
             #tail { height: 100px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let io = tree
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc container");
        // One area: #a's border box (content y=100..200) expanded by the
        // 30px top scroll-margin. `scroll-snap-align: start end` is
        // (block, inline); the engine works in physical axes, so it sees
        // (x, y) = (End, Start).
        let areas = &tree.container(io).snap_areas;
        assert_eq!(areas.len(), 1);
        assert_eq!(areas[0].rect.y, 70.0);
        assert_eq!(areas[0].rect.h, 130.0);
        assert_eq!(areas[0].align_x, SnapAlign::End);
        assert_eq!(areas[0].align_y, SnapAlign::Start);
        // A wheel scroll resting at y=30 snaps to the area's start edge
        // (70), not the natural 30.
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 30.0);
        assert!(tree.has_smooth_in_flight());
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (0.0, 70.0));
    }

    #[test]
    fn viewport_snap_type_root_and_body_fallback() {
        // Root-specified: used directly.
        let (doc, root) = layout_html(
            b"<html><body></body></html>",
            "html { scroll-snap-type: y mandatory; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert_eq!(tree.container(0).snap.snap_type, SnapType::MandatoryY);

        // Root `none` (the initial): the body's value applies, mirroring
        // `viewport_scroll_behavior`.
        let (doc, root) = layout_html(
            b"<html><body></body></html>",
            "body { scroll-snap-type: x proximity; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert_eq!(tree.container(0).snap.snap_type, SnapType::ProximityX);
    }

    #[test]
    fn viewport_rtl_root_and_body_fallback() {
        // Root-specified: used directly.
        let (doc, root) = layout_html(b"<html><body></body></html>", "html { direction: rtl; }");
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert!(
            tree.container(0).rtl,
            "root direction: rtl marks the viewport rtl"
        );

        // The common authoring pattern: `direction` only on the body —
        // the body's value propagates to the viewport, mirroring
        // `viewport_scroll_behavior` and the `writing-mode` fallback.
        let (doc, root) = layout_html(b"<html><body></body></html>", "body { direction: rtl; }");
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert!(
            tree.container(0).rtl,
            "body direction: rtl propagates to the viewport"
        );

        // Default: ltr.
        let (doc, root) = layout_html(b"<html><body></body></html>", "");
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert!(!tree.container(0).rtl);

        // Root explicitly `ltr`, body `rtl`: the body's explicit `rtl`
        // wins. Computed style cannot distinguish "root specified ltr"
        // from "root unspecified" (both read `ltr`), and the body's
        // computed `rtl` against an `ltr` root means the body specified
        // it — so the fallback treats any non-`rtl` root as "the root
        // didn't ask for rtl".
        let (doc, root) = layout_html(
            b"<html><body></body></html>",
            "html { direction: ltr; } body { direction: rtl; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert!(
            tree.container(0).rtl,
            "body-specified rtl wins over a non-rtl root"
        );
    }

    #[test]
    fn viewport_snap_areas_include_abspos_against_icb() {
        // The `snapevents-at-document-bubble-to-window` shape: the snap
        // type lives on `:root` and the snap points are
        // `position: absolute` divs with no positioned ancestor (their
        // containing block is the ICB). The viewport scroller must list
        // them as snap areas — previously both the layout query and
        // this engine mirror seeded `cb_inside` false for the static
        // root box and dropped them all, so `select_snap` had no
        // targets and no snap events were ever queued. The node ids
        // prove the mirror walk visited the boxes too (not just the
        // query): an unvisited box would surface with `node: None`.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"a\">x</div><div id=\"b\">y</div></body></html>",
            "html { scroll-snap-type: y mandatory; } \
             body { margin: 0; } \
             #a { position: absolute; top: 10px; left: 0; width: 50px; height: 50px; scroll-snap-align: start; } \
             #b { position: absolute; top: 700px; left: 0; width: 50px; height: 50px; scroll-snap-align: start; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let areas = &tree.container(0).snap_areas;
        assert_eq!(
            areas.len(),
            2,
            "abspos snap points against the ICB are viewport snap areas"
        );
        let a = doc.get_element_by_id("a").expect("a");
        let b = doc.get_element_by_id("b").expect("b");
        assert_eq!(areas[0].node, Some(a));
        assert_eq!(areas[1].node, Some(b));
    }

    /// Rebuild the scroll tree against a fresh layout of the same
    /// document (as a relayout after a style/content change produces),
    /// carrying scroll state over the way `rebuild_after_layout`
    /// does. The publication order mirrors production exactly: the
    /// viewport's min AND max are published BEFORE
    /// `carry_scroll_state`, because the resnap inside the carry
    /// clamps against the shared state — publishing the max after
    /// the carry (the old production order) stranded the viewport at
    /// the stale max while this helper's order masked it in tests.
    fn snap_relayout(
        doc: &Document,
        css: &str,
        doc_state: &Rc<ScrollState>,
        elements: &Rc<ElementScrollStates>,
        old: &ScrollTree,
    ) -> ScrollTree {
        let sheet = parse_stylesheet(css);
        let root = layout(
            doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let mut tree =
            ScrollTree::build(doc, &root, Size { w: 800.0, h: 600.0 }, doc_state, elements);
        let (origin, max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        tree.containers[0].origin = origin;
        doc_state.set_min_offset(origin);
        doc_state.set_max_offset(max);
        tree.carry_scroll_state(old);
        tree
    }

    /// H1 (production path): the viewport resnap inside
    /// `rebuild_after_layout` must clamp against the NEW layout's
    /// scroll range, not the previous layout's. Snapped at 0 to `#a`,
    /// the content grows so `#a` moves to 500 — past the old max of
    /// 400 — and the viewport must rest at 500, a real snap position,
    /// not stranded at the stale max.
    #[test]
    fn rebuild_after_layout_resnap_uses_new_viewport_max() {
        const CSS1: &str = "\
            html { scroll-snap-type: y mandatory; } \
            #sp { height: 0px; } \
            #a { height: 600px; scroll-snap-align: start; } \
            #tail { height: 400px; }";
        const CSS2: &str = "\
            html { scroll-snap-type: y mandatory; } \
            #sp { height: 500px; } \
            #a { height: 600px; scroll-snap-align: start; } \
            #tail { height: 400px; }";
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sp\"></div><div id=\"a\">x</div>\
              <div id=\"tail\"></div></body></html>",
        );
        let mut tab = Tab::new(Size { w: 800.0, h: 600.0 });
        // The tab must be the document's sole strong owner: layouts
        // are built through its `document_ref` borrow instead.
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        let lay = |tab: &Tab, css: &str| {
            let sheet = parse_stylesheet(css);
            let doc = tab.document_ref().expect("installed document");
            layout(
                &doc,
                &[&sheet],
                Size { w: 800.0, h: 600.0 },
                &FixedFontMetrics,
            )
        };
        tab.layout_root = Some(Rc::new(lay(&tab, CSS1)));
        tab.rebuild_scroll_after_layout();
        assert_eq!(tab.scroll().max_offset(), (0.0, 400.0));
        // Rest snapped to #a at 0: a 30px wheel gesture snaps back.
        tab.scroll_tree
            .scroll_chain(tab.scroll_tree.innermost_at(10.0, 10.0), 0.0, 30.0);
        tab.scroll_tree
            .advance_smooth(Instant::now() + Duration::from_secs(10));
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
        let rest = tab
            .scroll_tree
            .container(0)
            .snap_rest
            .get()
            .1
            .expect("snapped to #a");
        assert_eq!(rest.pos, 0.0);
        // The spacer grows 500px: #a's snap position moves to 500 and
        // the new max is 900. Resnap must follow #a to 500.
        tab.layout_root = Some(Rc::new(lay(&tab, CSS2)));
        tab.rebuild_scroll_after_layout();
        assert_eq!(tab.scroll().max_offset(), (0.0, 900.0));
        assert_eq!(
            tab.scroll().offset(),
            (0.0, 500.0),
            "resnap follows #a to its new position, not the stale max"
        );
        let rest = tab
            .scroll_tree
            .container(0)
            .snap_rest
            .get()
            .1
            .expect("still snapped to #a");
        assert_eq!(rest.pos, 500.0);
    }

    /// The viewport analogue of `resnap_move_fires_scroll_before_scrollend`:
    /// after a relayout whose resnap moves the viewport (0 -> 500 here),
    /// the move is stamped AND the next fold reports it as `None` — the
    /// `scroll` event's source — so `scrollend` never fires bare.
    #[test]
    fn viewport_resnap_move_reports_scroll_before_scrollend() {
        const CSS1: &str = "\
            html { scroll-snap-type: y mandatory; } \
            #sp { height: 0px; } \
            #a { height: 600px; scroll-snap-align: start; } \
            #tail { height: 400px; }";
        const CSS2: &str = "\
            html { scroll-snap-type: y mandatory; } \
            #sp { height: 500px; } \
            #a { height: 600px; scroll-snap-align: start; } \
            #tail { height: 400px; }";
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sp\"></div><div id=\"a\">x</div>\
              <div id=\"tail\"></div></body></html>",
        );
        let mut tab = Tab::new(Size { w: 800.0, h: 600.0 });
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        let lay = |tab: &Tab, css: &str| {
            let sheet = parse_stylesheet(css);
            let doc = tab.document_ref().expect("installed document");
            layout(
                &doc,
                &[&sheet],
                Size { w: 800.0, h: 600.0 },
                &FixedFontMetrics,
            )
        };
        tab.layout_root = Some(Rc::new(lay(&tab, CSS1)));
        tab.rebuild_scroll_after_layout();
        // Rest snapped to #a at 0: a 30px wheel gesture snaps back.
        tab.scroll_tree
            .scroll_chain(tab.scroll_tree.innermost_at(10.0, 10.0), 0.0, 30.0);
        tab.scroll_tree
            .advance_smooth(Instant::now() + Duration::from_secs(10));
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
        // The spacer grows 500px: #a's snap position moves to 500 and
        // the resnap follows it, stamping the `scrollend` tick.
        tab.layout_root = Some(Rc::new(lay(&tab, CSS2)));
        tab.rebuild_scroll_after_layout();
        assert_eq!(tab.scroll().offset(), (0.0, 500.0));
        assert_eq!(
            tab.scroll_tree.container(0).scroll_tick.get(),
            tab.scroll_tree.tick.get(),
            "resnap move stamps the scrollend tick"
        );
        // The fold reports the move as a viewport `scroll`: without the
        // fold-level report the stamped tick would end in a bare
        // `scrollend`.
        let paint = tab.render.paint_root.as_mut().expect("paint root");
        assert_eq!(
            tab.render.scroll_tree.sync_paint_offsets(paint),
            vec![None],
            "resnap move surfaces as a viewport scroll before scrollend"
        );
        // Idempotent: the fold re-armed the baseline, so a second fold
        // is a no-op.
        assert!(tab.render.scroll_tree.sync_paint_offsets(paint).is_empty());
        // Reporting the movement re-stamps the tick (the
        // `report_scroll_movement` path), so `scrollend` waits for a
        // quiet tick: `scroll` precedes `scrollend`, never a bare
        // `scrollend` for an unreported move.
        tab.scroll_tree.mark_scrolled(&[None]);
        assert!(tab.scroll_tree.end_tick().is_empty());
        assert_eq!(tab.scroll_tree.end_tick(), vec![None]);
    }

    /// H3: a snap candidate outside the container's reachable range
    /// (an unreachable snap position) resnaps to the clamped offset,
    /// and the recorded rest position is the clamped one too — so a
    /// second identical relayout is a strict no-op instead of
    /// re-entering the fallback forever.
    #[test]
    fn resnap_unreachable_candidate_clamps_rest_and_converges() {
        let css = |t: &str| {
            format!(
                "#sc {{ overflow-y: auto; height: 100px; scroll-snap-type: {t}; }} \
                 #a {{ height: 100px; scroll-snap-align: start; }} \
                 #mid {{ height: 150px; }} \
                 #b {{ height: 50px; scroll-snap-align: start; }}"
            )
        };
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div>\
              <div id=\"mid\"></div><div id=\"b\">y</div></div></body></html>",
        );
        let doc_state = Rc::new(ScrollState::new());
        let elements = Rc::new(ElementScrollStates::new());
        let sheet = parse_stylesheet(&css("none"));
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let tree = ScrollTree::build(
            &doc,
            &root,
            Size { w: 800.0, h: 600.0 },
            &doc_state,
            &elements,
        );
        let sc = doc.get_element_by_id("sc").expect("sc");
        let io = tree.index_of(sc).expect("sc container");
        // #b's snap position is 250 but the max offset is 200
        // (300px of content in a 100px scrollport): unreachable.
        // Rest at 190 under `none`, then turn `mandatory` on.
        tree.container(io).state.set_offset(0.0, 190.0);
        tree.mark_scrolled(&[Some(sc)]);
        assert_eq!(tree.container(io).state.offset(), (0.0, 190.0));
        let tree2 = snap_relayout(&doc, &css("y mandatory"), &doc_state, &elements, &tree);
        assert_eq!(
            tree2.container(io).state.offset(),
            (0.0, 200.0),
            "the unreachable candidate clamps to the max offset"
        );
        let rest = tree2
            .container(io)
            .snap_rest
            .get()
            .1
            .expect("snapped to #b");
        assert_eq!(rest.pos, 200.0, "the recorded rest position is clamped");
        // A second identical relayout is a strict no-op: offset and
        // snap_base both untouched (a sentinel snap_base proves the
        // resnap move path never ran).
        tree2.container(io).snap_base.set((0.0, 123.0));
        let tree3 = snap_relayout(&doc, &css("y mandatory"), &doc_state, &elements, &tree2);
        assert_eq!(tree3.container(io).state.offset(), (0.0, 200.0));
        assert_eq!(
            tree3.container(io).snap_base.get(),
            (0.0, 123.0),
            "second relayout does not touch snap_base"
        );
        let rest = tree3
            .container(io)
            .snap_rest
            .get()
            .1
            .expect("still snapped to #b");
        assert_eq!(rest.pos, 200.0);
    }

    /// `eval_snap` records which area it selected: the container is
    /// "snapped" exactly when it still rests at the recorded
    /// position.
    #[test]
    fn snap_eval_records_snapped_state() {
        let (_doc, _root, _sc, io, tree) = snap_tree();
        // A wheel resting at 30 snaps toward 0: the follow-up is in
        // flight, and the recorded state names the target.
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 30.0);
        let rest = tree.container(io).snap_rest.get().1.expect("snapped on y");
        assert_eq!(rest.pos, 0.0);
        assert_eq!(rest.area, 0, "the first snap area in tree order");
        // Completing the animation leaves the container resting at
        // the recorded position: it counts as snapped.
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        let (live_x, live_y) = tree.container(io).state.offset();
        assert_eq!((live_x, live_y), (0.0, 0.0));
        let rest = tree.container(io).snap_rest.get().1.expect("still snapped");
        assert!((rest.pos - live_y).abs() <= SNAP_EPS);
        // A proximity miss records nothing: nothing to follow.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div>\
              <div id=\"gap\"></div><div id=\"b\">y</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; scroll-snap-type: y proximity; } \
             #a { height: 100px; scroll-snap-align: start; } \
             #gap { height: 300px; } \
             #b { height: 100px; scroll-snap-align: start; }",
        );
        let (_ds, _el, tree) = build_tree(&doc, &root);
        let io = tree
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc container");
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 200.0);
        assert_eq!(tree.container(io).state.offset(), (0.0, 200.0));
        assert!(
            tree.container(io).snap_rest.get().1.is_none(),
            "resting between targets is not snapped"
        );
    }

    /// Install a built tree the way production does
    /// (`rebuild_after_layout`): build, then carry state over from
    /// the pre-layout placeholder tree — the carry baselines the
    /// snap-event state (initial layout-snap update).
    fn snap_installed(
        doc: &Document,
        root: &LayoutBox,
    ) -> (Rc<ScrollState>, Rc<ElementScrollStates>, ScrollTree) {
        let (doc_state, elements, mut tree) = build_tree(doc, root);
        let old = ScrollTree::for_viewport(&doc_state, Size { w: 800.0, h: 600.0 }, (true, true));
        tree.carry_scroll_state(&old);
        (doc_state, elements, tree)
    }

    /// One drained snap-event delivery as a comparable tuple:
    /// `(container, kind, snapTargetBlock, snapTargetInline)`.
    type SnapDeliveryRow = (
        Option<NodeId>,
        SnapEventKind,
        Option<NodeId>,
        Option<NodeId>,
    );

    /// Drain a tree's snap-event queue into comparable tuples.
    fn snap_deliveries(tree: &ScrollTree) -> Vec<SnapDeliveryRow> {
        tree.take_snap_events()
            .into_iter()
            .map(|d| (d.container, d.kind, d.targets.block, d.targets.inline))
            .collect()
    }

    #[test]
    fn snap_events_initial_build() {
        // A tree installed resting on #a queues the initial
        // layout-snap pair (changing, then change) for #a and seeds
        // the slots: an identical relayout queues nothing more.
        let (doc, root) = layout_html(SNAP_HTML, SNAP_CSS);
        let (doc_state, elements, tree) = snap_installed(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let a = doc.get_element_by_id("a").expect("a");
        assert_eq!(
            snap_deliveries(&tree),
            vec![
                (Some(sc), SnapEventKind::Changing, Some(a), None),
                (Some(sc), SnapEventKind::Change, Some(a), None),
            ]
        );
        let tree2 = snap_relayout(&doc, SNAP_CSS, &doc_state, &elements, &tree);
        assert_eq!(
            snap_deliveries(&tree2),
            vec![],
            "slots carried across the rebuild: no re-fire"
        );

        // A proximity container installed resting between targets
        // queues nothing and seeds the all-null pair.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div>\
              <div id=\"gap\"></div><div id=\"b\">y</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; scroll-snap-type: y proximity; } \
             #a { height: 100px; scroll-snap-align: start; } \
             #gap { height: 300px; } \
             #b { height: 100px; scroll-snap-align: start; }",
        );
        let (_ds, elements, mut tree) = build_tree(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let io = tree.index_of(sc).expect("sc container");
        elements
            .state_for(sc)
            .expect("sc state")
            .set_offset(0.0, 200.0);
        let old = ScrollTree::for_viewport(&_ds, Size { w: 800.0, h: 600.0 }, (true, true));
        tree.carry_scroll_state(&old);
        assert_eq!(snap_deliveries(&tree), vec![]);
        assert_eq!(
            tree.container(io).snap_change_sent.get(),
            Some(SnapTargets::default()),
            "unsnapped baseline is seeded, not left never-fired"
        );
    }

    #[test]
    fn snap_events_eval_queues_changing_once() {
        let (doc, root) = layout_html(SNAP_HTML, SNAP_CSS);
        let (_ds, _el, tree) = snap_installed(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let b = doc.get_element_by_id("b").expect("b");
        let _ = snap_deliveries(&tree); // initial pair
        // Wheel toward #b: exactly one `Changing` for #b.
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 60.0);
        assert_eq!(
            snap_deliveries(&tree),
            vec![(Some(sc), SnapEventKind::Changing, Some(b), None)]
        );
        // A second wheel still toward #b queues nothing.
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 10.0);
        assert_eq!(snap_deliveries(&tree), vec![]);
        // Completing the follow-up animation queues no duplicate.
        tree.advance_smooth(Instant::now() + Duration::from_secs(10));
        assert_eq!(snap_deliveries(&tree), vec![]);
    }

    #[test]
    fn snap_events_change_at_end_tick_only() {
        let (doc, root) = layout_html(SNAP_HTML, SNAP_CSS);
        let (_ds, _el, tree) = snap_installed(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let b = doc.get_element_by_id("b").expect("b");
        let _ = snap_deliveries(&tree);
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 60.0);
        let _ = snap_deliveries(&tree);
        tree.advance_smooth(Instant::now() + Duration::from_secs(10));
        // No `Change` until the scroll completes (quiet tick).
        assert_eq!(snap_deliveries(&tree), vec![]);
        tree.mark_scrolled(&[Some(sc)]);
        assert_eq!(tree.end_tick(), Vec::<Option<NodeId>>::new());
        assert_eq!(snap_deliveries(&tree), vec![]);
        assert_eq!(tree.end_tick(), vec![Some(sc)]);
        assert_eq!(
            snap_deliveries(&tree),
            vec![(Some(sc), SnapEventKind::Change, Some(b), None)]
        );
        // A further quiet tick queues nothing.
        assert_eq!(tree.end_tick(), Vec::<Option<NodeId>>::new());
        assert_eq!(snap_deliveries(&tree), vec![]);
    }

    #[test]
    fn snap_events_snaps_back_fires_nothing() {
        let (doc, root) = layout_html(SNAP_HTML, SNAP_CSS);
        let (_ds, _el, tree) = snap_installed(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let _ = snap_deliveries(&tree);
        // A small wheel that snaps back to the initial target: the
        // eventual pair is the initial pair throughout, so neither
        // event ever queues.
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 30.0);
        tree.advance_smooth(Instant::now() + Duration::from_secs(10));
        tree.mark_scrolled(&[Some(sc)]);
        let _ = tree.end_tick();
        let _ = tree.end_tick();
        assert_eq!(snap_deliveries(&tree), vec![]);
    }

    #[test]
    fn snap_events_proximity_unsnap_fires_change_with_null() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div>\
              <div id=\"gap\"></div><div id=\"b\">y</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; scroll-snap-type: y proximity; } \
             #a { height: 100px; scroll-snap-align: start; } \
             #gap { height: 300px; } \
             #b { height: 100px; scroll-snap-align: start; }",
        );
        let (_ds, _el, tree) = snap_installed(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let a = doc.get_element_by_id("a").expect("a");
        assert_eq!(
            snap_deliveries(&tree),
            vec![
                (Some(sc), SnapEventKind::Changing, Some(a), None),
                (Some(sc), SnapEventKind::Change, Some(a), None),
            ]
        );
        // Scrolling past every target's proximity range unsnaps: the
        // pair goes to all-null, which is a change on both events.
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 200.0);
        assert_eq!(
            snap_deliveries(&tree),
            vec![(Some(sc), SnapEventKind::Changing, None, None)]
        );
        tree.mark_scrolled(&[Some(sc)]);
        let _ = tree.end_tick();
        let _ = tree.end_tick();
        assert_eq!(
            snap_deliveries(&tree),
            vec![(Some(sc), SnapEventKind::Change, None, None)]
        );
    }

    #[test]
    fn snap_events_both_axes_independent_pair() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"content\">\
              <div id=\"x1\" class=\"t\"></div><div id=\"x2\" class=\"t\"></div>\
              <div id=\"x3\" class=\"t\"></div><div id=\"x4\" class=\"t\"></div>\
              </div></div></body></html>",
            "#sc { overflow: auto; width: 100px; height: 100px; scroll-snap-type: both mandatory; } \
             #content { position: relative; width: 200px; height: 200px; } \
             .t { position: absolute; width: 100px; height: 100px; scroll-snap-align: start; } \
             #x1 { left: 0; top: 0; } #x2 { left: 100px; top: 0; } \
             #x3 { left: 0; top: 100px; } #x4 { left: 100px; top: 100px; }",
        );
        let (_ds, _el, tree) = snap_installed(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let x2 = doc.get_element_by_id("x2").expect("x2");
        let x3 = doc.get_element_by_id("x3").expect("x3");
        let _ = snap_deliveries(&tree);
        // A diagonal scroll: the x winner and the y winner are
        // different areas, and the pair carries both, axis-mapped
        // (block = y winner, inline = x winner).
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 60.0, 60.0);
        assert_eq!(
            snap_deliveries(&tree),
            vec![(Some(sc), SnapEventKind::Changing, Some(x3), Some(x2))]
        );
    }

    #[test]
    fn snap_events_resnap_diffs() {
        const CSS1: &str = "\
            #sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
            #a { height: 100px; scroll-snap-align: start; } \
            #b { height: 100px; scroll-snap-align: start; } \
            #c { height: 100px; scroll-snap-align: start; }";
        const CSS2: &str = "\
            #sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
            #a { height: 150px; scroll-snap-align: start; } \
            #b { height: 100px; scroll-snap-align: start; } \
            #c { height: 100px; scroll-snap-align: start; }";
        const CSS3: &str = "\
            #sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
            #a { height: 100px; scroll-snap-align: start; } \
            #c { height: 100px; scroll-snap-align: start; }";
        const CSS4: &str = "\
            #sc { overflow-y: auto; height: 100px; scroll-snap-type: none; } \
            #a { height: 100px; scroll-snap-align: start; } \
            #c { height: 100px; scroll-snap-align: start; }";
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div>\
              <div id=\"b\">y</div><div id=\"c\">z</div></div></body></html>",
            CSS1,
        );
        let (doc_state, elements, tree) = snap_installed(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let b = doc.get_element_by_id("b").expect("b");
        let c = doc.get_element_by_id("c").expect("c");
        let io = tree.index_of(sc).expect("sc container");
        let _ = snap_deliveries(&tree);
        // Settle on #b and drain its events.
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 100.0);
        tree.mark_scrolled(&[Some(sc)]);
        let _ = tree.end_tick();
        let _ = tree.end_tick();
        assert_eq!(tree.container(io).state.offset(), (0.0, 100.0));
        let _ = snap_deliveries(&tree);
        // #a grows: #b moves to 150 and resnap follows it — the
        // target did not change, so no events fire.
        let tree = snap_relayout(&doc, CSS2, &doc_state, &elements, &tree);
        assert_eq!(tree.container(io).state.offset(), (0.0, 150.0));
        assert_eq!(snap_deliveries(&tree), vec![]);
        // #b is removed: the nearest survivor (#c, now at 100) wins,
        // reported by both events.
        let tree = snap_relayout(&doc, CSS3, &doc_state, &elements, &tree);
        assert_eq!(
            snap_deliveries(&tree),
            vec![
                (Some(sc), SnapEventKind::Changing, Some(c), None),
                (Some(sc), SnapEventKind::Change, Some(c), None),
            ]
        );
        // Snap type -> none: both events fire with the all-null pair.
        let tree = snap_relayout(&doc, CSS4, &doc_state, &elements, &tree);
        assert_eq!(
            snap_deliveries(&tree),
            vec![
                (Some(sc), SnapEventKind::Changing, None, None),
                (Some(sc), SnapEventKind::Change, None, None),
            ]
        );
        // none -> mandatory: both events name the resting target.
        let tree = snap_relayout(&doc, CSS3, &doc_state, &elements, &tree);
        assert_eq!(
            snap_deliveries(&tree),
            vec![
                (Some(sc), SnapEventKind::Changing, Some(c), None),
                (Some(sc), SnapEventKind::Change, Some(c), None),
            ]
        );
        let _ = b;
    }

    /// The full interrupted-scroll lifecycle at unit level
    /// (`scrollsnapchange-on-interrupted-scroll.tentative.html`): a
    /// smooth `scrollTo(250)` toward area2 is interrupted by a
    /// `scroll` listener's smooth `scrollTo(0)`, and the scroll
    /// settles back on area1. The headless pump folds each drain's
    /// movement through `mark_scrolled` (the snap backstop) and
    /// completes in-flight animations via `complete_smooth_scrolls`,
    /// whose interruption branch re-evaluates the scroll — so the
    /// `scrollsnapchange` baseline (`snap_change_sent`) still names
    /// area1 when the scroll ends, and `end_tick` queues nothing.
    /// `scrollsnapchanging` fires for each redirected target (area2,
    /// then area1); `scrollsnapchange` never does.
    #[test]
    fn interrupted_smooth_scroll_settling_on_same_target_emits_no_change() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"area1\">x</div>\
              <div id=\"area2\">y</div><div id=\"area3\">z</div></div></body></html>",
            "#sc { overflow-y: scroll; height: 500px; width: 300px; \
             scroll-snap-type: y mandatory; } \
             #area1 { height: 400px; scroll-snap-align: start; } \
             #area2 { height: 400px; scroll-snap-align: start; } \
             #area3 { height: 400px; scroll-snap-align: start; }",
        );
        let (_doc_state, _elements, tree) = snap_installed(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let a1 = doc.get_element_by_id("area1").expect("area1");
        let a2 = doc.get_element_by_id("area2").expect("area2");
        let io = tree.index_of(sc).expect("sc container");
        // Install seats area1 and seeds the sent slots; drain those
        // deliveries so only the scroll's own events are observed.
        assert_eq!(
            snap_deliveries(&tree),
            vec![
                (Some(sc), SnapEventKind::Changing, Some(a1), None),
                (Some(sc), SnapEventKind::Change, Some(a1), None),
            ]
        );
        // Tick past zero so the scroll stamps read as a real gesture.
        assert_eq!(tree.end_tick(), Vec::<Option<NodeId>>::new());
        // Pump iteration 1: the headless drain applies the smooth
        // scrollTo(250) instantly; the fold's backstop snap-evaluates
        // and launches the follow-up smooth scroll to area2's 400.
        tree.container(io).state.set_offset(0.0, 250.0);
        tree.mark_scrolled(&[Some(sc)]);
        assert!(tree.has_smooth_in_flight());
        assert_eq!(
            snap_deliveries(&tree),
            vec![(Some(sc), SnapEventKind::Changing, Some(a2), None)]
        );
        // The `scroll` listener's reset(): the headless drain applies
        // the smooth scrollTo(0) instantly, interrupting the in-flight
        // animation; completing it re-evaluates the interruption.
        tree.container(io).state.set_offset(0.0, 0.0);
        let moved = tree.complete_smooth_scrolls();
        tree.mark_scrolled(&moved);
        assert!(tree.end_tick().is_empty(), "scroll still in flight");
        // Pump iteration 2: quiet — the scroll ends on area1.
        assert_eq!(tree.end_tick(), vec![Some(sc)]);
        // Settling back on the same target reports the redirection
        // via `scrollsnapchanging` only: no `scrollsnapchange`.
        assert_eq!(
            snap_deliveries(&tree),
            vec![(Some(sc), SnapEventKind::Changing, Some(a1), None)]
        );
        assert_eq!(tree.container(io).state.offset(), (0.0, 0.0));
    }

    /// The interrupted-scroll scenario at unit level
    /// (`scrollsnapchange-on-interrupted-scroll.tentative.html`): a
    /// scroll toward #b completed its snap animation (offset at #b,
    /// `scrollsnapchanging` reported) but has not ended —
    /// `scrollsnapchange` still baselines at #a. A relayout whose
    /// re-snap stays on #b must not report the in-flight target; the
    /// scroll's own end does that.
    #[test]
    fn resnap_midscroll_pair_unchanged_queues_no_change() {
        let (doc, root) = layout_html(SNAP_HTML, SNAP_CSS);
        let (doc_state, elements, tree) = snap_installed(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let b = doc.get_element_by_id("b").expect("b");
        let io = tree.index_of(sc).expect("sc container");
        let _ = snap_deliveries(&tree);
        // Tick past zero so the scroll stamps read as a real gesture.
        assert_eq!(tree.end_tick(), Vec::<Option<NodeId>>::new());
        // Script write toward #b: the backstop snap-evaluates and the
        // follow-up animation completes headless-instantly.
        tree.container(io).state.set_offset(0.0, 60.0);
        tree.mark_scrolled(&[Some(sc)]);
        assert!(tree.has_smooth_in_flight());
        let t0 = Instant::now();
        let moved = tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (0.0, 100.0));
        tree.mark_scrolled(&moved);
        assert_eq!(
            snap_deliveries(&tree),
            vec![(Some(sc), SnapEventKind::Changing, Some(b), None)]
        );
        // Relayout with identical geometry: the re-snap stays on #b.
        // Queuing `scrollsnapchange` here would report a target whose
        // scroll never ended — and corrupt the baseline when the
        // scroll settles back on #a, as in the interrupted-scroll
        // WPT test.
        let tree2 = snap_relayout(&doc, SNAP_CSS, &doc_state, &elements, &tree);
        assert_eq!(tree2.container(io).state.offset(), (0.0, 100.0));
        assert_eq!(snap_deliveries(&tree2), vec![]);
        // The scroll still ends normally, reporting #b exactly once.
        assert!(tree2.end_tick().is_empty());
        assert_eq!(tree2.end_tick(), vec![Some(sc)]);
        assert_eq!(
            snap_deliveries(&tree2),
            vec![(Some(sc), SnapEventKind::Change, Some(b), None)]
        );
    }

    /// A re-snap that moves the scroller stamps the `scrollend` tick
    /// (`scrollsnapchange-same-targets-after-layout-changed.html`
    /// subtest 2): the headless pump folds scroll movement before
    /// `refresh` runs, so without the stamp a layout-driven scroll
    /// would never complete its event lifecycle and the test's
    /// `scrollend` wait would time out.
    #[test]
    fn resnap_move_stamps_scrollend_tick() {
        const CSS1: &str = "\
            #sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
            #a { height: 100px; scroll-snap-align: start; } \
            #b { height: 100px; scroll-snap-align: start; }";
        const CSS2: &str = "\
            #sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
            #a { height: 150px; scroll-snap-align: start; } \
            #b { height: 100px; scroll-snap-align: start; }";
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div>\
              <div id=\"b\">y</div></div></body></html>",
            CSS1,
        );
        let (doc_state, elements, tree) = snap_installed(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let io = tree.index_of(sc).expect("sc container");
        let _ = snap_deliveries(&tree);
        // Settle on #b and end the scroll, draining its events.
        assert_eq!(tree.end_tick(), Vec::<Option<NodeId>>::new());
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 100.0);
        tree.mark_scrolled(&[Some(sc)]);
        let t0 = Instant::now();
        let moved = tree.advance_smooth(t0 + Duration::from_secs(10));
        tree.mark_scrolled(&moved);
        assert!(tree.end_tick().is_empty());
        assert_eq!(tree.end_tick(), vec![Some(sc)]);
        let _ = snap_deliveries(&tree);
        assert_eq!(tree.container(io).state.offset(), (0.0, 100.0));
        // #a grows: #b's snap position moves 100 -> 150 and the
        // re-snap follows it without changing the pair.
        let tree2 = snap_relayout(&doc, CSS2, &doc_state, &elements, &tree);
        assert_eq!(tree2.container(io).state.offset(), (0.0, 150.0));
        assert_eq!(
            snap_deliveries(&tree2),
            vec![],
            "pair unchanged: no snap events"
        );
        // The layout-driven move stamped the `scrollend` tick, so the
        // next quiet tick ends it: `scrollend` fires for the move,
        // with no duplicate `scrollsnapchange`.
        assert_ne!(
            tree2.container(io).scroll_tick.get(),
            0,
            "resnap move stamps scroll_tick"
        );
        assert!(tree2.end_tick().is_empty());
        assert_eq!(tree2.end_tick(), vec![Some(sc)]);
        assert_eq!(snap_deliveries(&tree2), vec![]);
    }

    /// A resnap move surfaces as a `scroll` before its `scrollend`:
    /// `rebuild_after_layout` re-bakes the paint tree at the post-move
    /// offsets, which would hide the move from the next fold's drift
    /// check and leave a bare `scrollend` for a move the page never
    /// heard about. Resetting the `painted` baselines to the pre-move
    /// offsets makes the fold report the move first (Chrome fires
    /// `scroll` for layout-induced offset changes).
    #[test]
    fn resnap_move_fires_scroll_before_scrollend() {
        const CSS1: &str = "\
            #sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
            #a { height: 100px; scroll-snap-align: start; } \
            #b { height: 100px; scroll-snap-align: start; }";
        const CSS2: &str = "\
            #sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
            #a { height: 150px; scroll-snap-align: start; } \
            #b { height: 100px; scroll-snap-align: start; }";
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div>\
              <div id=\"b\">y</div></div></body></html>",
        );
        let mut tab = Tab::new(Size { w: 800.0, h: 600.0 });
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        let lay = |tab: &Tab, css: &str| {
            let sheet = parse_stylesheet(css);
            let doc = tab.document_ref().expect("installed document");
            layout(
                &doc,
                &[&sheet],
                Size { w: 800.0, h: 600.0 },
                &FixedFontMetrics,
            )
        };
        tab.layout_root = Some(Rc::new(lay(&tab, CSS1)));
        tab.rebuild_scroll_after_layout();
        let sc = tab
            .document_ref()
            .expect("installed document")
            .get_element_by_id("sc")
            .expect("sc");
        let io = tab.scroll_tree.index_of(sc).expect("sc container");
        let _ = snap_deliveries(&tab.scroll_tree);
        // Settle on #b and end the scroll, draining its events — the
        // same prelude as `resnap_move_stamps_scrollend_tick`.
        assert_eq!(tab.scroll_tree.end_tick(), Vec::<Option<NodeId>>::new());
        tab.scroll_tree
            .scroll_chain(tab.scroll_tree.innermost_at(10.0, 10.0), 0.0, 100.0);
        tab.scroll_tree.mark_scrolled(&[Some(sc)]);
        let t0 = Instant::now();
        let moved = tab.scroll_tree.advance_smooth(t0 + Duration::from_secs(10));
        tab.scroll_tree.mark_scrolled(&moved);
        assert!(tab.scroll_tree.end_tick().is_empty());
        assert_eq!(tab.scroll_tree.end_tick(), vec![Some(sc)]);
        let _ = snap_deliveries(&tab.scroll_tree);
        assert_eq!(tab.scroll_tree.container(io).state.offset(), (0.0, 100.0));
        // #a grows: #b's snap position moves 100 -> 150 and the
        // re-snap follows it without changing the pair.
        tab.layout_root = Some(Rc::new(lay(&tab, CSS2)));
        tab.rebuild_scroll_after_layout();
        assert_eq!(
            tab.scroll_tree.container(io).state.offset(),
            (0.0, 150.0),
            "resnap follows #b to its new position"
        );
        // The next fold's drift check (the following pump iteration)
        // reports the resnap move: the `scroll` event's source. Without
        // the baseline reset the re-bake hides the move and the fold
        // reports nothing.
        let b = tab
            .document_ref()
            .expect("installed document")
            .get_element_by_id("b")
            .expect("b");
        let layout_b = find_box(tab.layout_root.as_ref().expect("layout root"), b)
            .expect("b in layout tree")
            .rect;
        let paint = tab.render.paint_root.as_mut().expect("paint root");
        let folded = tab.render.scroll_tree.sync_paint_offsets(paint);
        assert_eq!(
            folded,
            vec![Some(sc)],
            "resnap move surfaces as a scroll before scrollend"
        );
        // The fold reports the move but must not mis-shift the
        // paint-tree rects: the bake seeded them at `layout − post`
        // and the re-armed report baseline (pre) is not the shift
        // base — #b stays at `layout_b.y − 150`, not shifted by
        // (pre − post) down to `layout_b.y − 200`.
        let paint_b = find_box(paint, b).expect("b in paint tree").rect;
        assert_eq!(paint_b.y, layout_b.y - 150.0);
        assert_eq!(paint_b.x, layout_b.x);
        // Idempotent: the fold advanced the shift base to the live
        // offset, so a second fold is a no-op.
        assert!(tab.render.scroll_tree.sync_paint_offsets(paint).is_empty());
        // Reporting the movement re-stamps the tick (the
        // `report_scroll_movement` path), so `scrollend` waits for a
        // quiet tick: `scroll` precedes `scrollend`, never a bare
        // `scrollend` for an unreported move.
        tab.scroll_tree.mark_scrolled(&folded);
        assert!(tab.scroll_tree.end_tick().is_empty());
        assert_eq!(tab.scroll_tree.end_tick(), vec![Some(sc)]);
    }

    /// The resnap baseline reset re-arms only the report baseline:
    /// after `rebuild_after_layout` the wrapped fast path still folds
    /// the resnap move with `try_fast_fold`, writing the absolute
    /// matrix from the never-reset `wrap_base` while `painted` is
    /// still at the pre-move offset.
    #[test]
    fn resnap_reset_keeps_fast_fold_on_wrap_base() {
        const CSS1: &str = "\
            #sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
            #a { height: 100px; scroll-snap-align: start; } \
            #b { height: 100px; scroll-snap-align: start; }";
        const CSS2: &str = "\
            #sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
            #a { height: 150px; scroll-snap-align: start; } \
            #b { height: 100px; scroll-snap-align: start; }";
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div>\
              <div id=\"b\">y</div></div></body></html>",
        );
        let mut tab = Tab::new(Size { w: 800.0, h: 600.0 });
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        let lay = |tab: &Tab, css: &str| {
            let sheet = parse_stylesheet(css);
            let doc = tab.document_ref().expect("installed document");
            layout(
                &doc,
                &[&sheet],
                Size { w: 800.0, h: 600.0 },
                &FixedFontMetrics,
            )
        };
        tab.layout_root = Some(Rc::new(lay(&tab, CSS1)));
        tab.rebuild_scroll_after_layout();
        let sc = tab
            .document_ref()
            .expect("installed document")
            .get_element_by_id("sc")
            .expect("sc");
        let io = tab.scroll_tree.index_of(sc).expect("sc container");
        let _ = snap_deliveries(&tab.scroll_tree);
        // Settle on #b and end the scroll, draining its events — the
        // same prelude as `resnap_move_fires_scroll_before_scrollend`.
        assert_eq!(tab.scroll_tree.end_tick(), Vec::<Option<NodeId>>::new());
        tab.scroll_tree
            .scroll_chain(tab.scroll_tree.innermost_at(10.0, 10.0), 0.0, 100.0);
        tab.scroll_tree.mark_scrolled(&[Some(sc)]);
        let t0 = Instant::now();
        let moved = tab.scroll_tree.advance_smooth(t0 + Duration::from_secs(10));
        tab.scroll_tree.mark_scrolled(&moved);
        assert!(tab.scroll_tree.end_tick().is_empty());
        assert_eq!(tab.scroll_tree.end_tick(), vec![Some(sc)]);
        let _ = snap_deliveries(&tab.scroll_tree);
        assert_eq!(tab.scroll_tree.container(io).state.offset(), (0.0, 100.0));
        // #a grows: #b's snap position moves 100 -> 150 and the
        // re-snap follows it without changing the pair.
        tab.layout_root = Some(Rc::new(lay(&tab, CSS2)));
        tab.rebuild_scroll_after_layout();
        assert_eq!(
            tab.scroll_tree.container(io).state.offset(),
            (0.0, 150.0),
            "resnap follows #b to its new position"
        );
        // The reset re-armed only the report baseline; the shift bases
        // stayed at the post-move offsets the bake seeded.
        assert_eq!(tab.scroll_tree.container(io).painted.get(), (0.0, 100.0));
        assert_eq!(tab.scroll_tree.container(io).wrap_base.get(), (0.0, 150.0));
        assert_eq!(tab.scroll_tree.container(io).paint_base.get(), (0.0, 150.0));
        // Rebuild the display list and wrap: the container is
        // transform-eligible, so the fast path must take the fold.
        let paint = tab.paint_root.as_ref().expect("paint root");
        let doc_ref = tab.document_ref().expect("installed document");
        let mut list = display_list_of(paint, &doc_ref);
        tab.scroll_tree.wrap_scroll_transforms(paint, &mut list);
        let wrap_at = tab
            .scroll_tree
            .container(io)
            .wrap_item
            .get()
            .expect("wrapped");
        let folded = tab.render.scroll_tree.try_fast_fold(&mut list);
        assert_eq!(
            folded,
            Some(vec![Some(sc)]),
            "fast path folds the resnap move"
        );
        // The wrap matrix is the absolute translation from the
        // un-reset `wrap_base` — (0, 0) here since base == live — not
        // from the re-armed `painted`, which would write (0, -50).
        let (bx, by) = tab.scroll_tree.container(io).wrap_base.get();
        let (ox, oy) = tab.scroll_tree.container(io).state.offset();
        let m = transform_matrix_at(&list, wrap_at);
        assert!(approx_eq(m[4], bx - ox));
        assert!(approx_eq(m[5], by - oy));
        // The fold advanced the report baseline to the live offset.
        assert_eq!(tab.scroll_tree.container(io).painted.get(), (0.0, 150.0));
    }

    /// Initial snap on load for a mandatory container whose nearest
    /// candidate is unreachable: install seats it at the clamped max
    /// and seeds the sent slots with the real pair — a later
    /// identical relayout emits nothing.
    #[test]
    fn initial_snap_update_seeds_resnap_seated_pair() {
        // 300px of content in a 100px scrollport (max 200); #b's snap
        // position is 250, unreachable — the install-time re-snap
        // seats the container at 200.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"mid\"></div>\
              <div id=\"b\">y</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
             #mid { height: 250px; } \
             #b { height: 50px; scroll-snap-align: start; }",
        );
        let (doc_state, elements, tree) = snap_installed(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let b = doc.get_element_by_id("b").expect("b");
        let io = tree.index_of(sc).expect("sc container");
        assert_eq!(
            tree.container(io).state.offset(),
            (0.0, 200.0),
            "install snaps to the clamped nearest candidate"
        );
        // The slots are seeded with the real pair (not all-null), via
        // the initial layout-snap deliveries.
        assert_eq!(
            snap_deliveries(&tree),
            vec![
                (Some(sc), SnapEventKind::Changing, Some(b), None),
                (Some(sc), SnapEventKind::Change, Some(b), None),
            ]
        );
        let pair = tree.container(io).snap_targets();
        assert_eq!(pair.block, Some(b));
        assert_eq!(tree.container(io).snap_change_sent.get(), Some(pair));
        // An identical relayout: no move, no events.
        let css = "#sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
             #mid { height: 250px; } \
             #b { height: 50px; scroll-snap-align: start; }";
        let tree2 = snap_relayout(&doc, css, &doc_state, &elements, &tree);
        assert_eq!(tree2.container(io).state.offset(), (0.0, 200.0));
        assert_eq!(snap_deliveries(&tree2), vec![]);
    }

    #[test]
    fn snap_events_direct_peek() {
        let (doc, root) = layout_html(SNAP_HTML, SNAP_CSS);
        let (_ds, _el, tree) = snap_installed(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let a = doc.get_element_by_id("a").expect("a");
        let b = doc.get_element_by_id("b").expect("b");
        let io = tree.index_of(sc).expect("sc container");
        let _ = snap_deliveries(&tree);
        // A thumb drag to 60: the peek reports #b as the eventual
        // target, but the offset, the recorded rest, and the smooth
        // state are all untouched.
        tree.container(io).state.set_offset(0.0, 60.0);
        tree.mark_scrolled_direct(&[Some(sc)]);
        assert_eq!(
            snap_deliveries(&tree),
            vec![(Some(sc), SnapEventKind::Changing, Some(b), None)]
        );
        assert_eq!(tree.container(io).state.offset(), (0.0, 60.0));
        assert_eq!(
            tree.container(io).snap_rest.get().1.map(|r| r.node),
            Some(Some(a)),
            "the peek does not record the peeked rest"
        );
        assert!(!tree.has_smooth_in_flight());
        // The release evaluation settles on the same target: no
        // duplicate `Changing`.
        tree.snap_after_direct_manipulation(Some(sc));
        assert_eq!(snap_deliveries(&tree), vec![]);
    }

    /// Resnap follows a moved snap target, then stays put on an
    /// identical rebuild (convergence: the second relayout is a
    /// no-op).
    #[test]
    fn resnap_follows_moved_target_and_is_idempotent() {
        const CSS1: &str = "\
            #sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
            #a { height: 100px; scroll-snap-align: start; } \
            #b { height: 100px; scroll-snap-align: start; } \
            #c { height: 100px; scroll-snap-align: start; }";
        const CSS2: &str = "\
            #sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
            #a { height: 150px; scroll-snap-align: start; } \
            #b { height: 100px; scroll-snap-align: start; } \
            #c { height: 100px; scroll-snap-align: start; }";
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div>\
              <div id=\"b\">y</div><div id=\"c\">z</div></div></body></html>",
        );
        let doc_state = Rc::new(ScrollState::new());
        let elements = Rc::new(ElementScrollStates::new());
        let sheet = parse_stylesheet(CSS1);
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let tree = ScrollTree::build(
            &doc,
            &root,
            Size { w: 800.0, h: 600.0 },
            &doc_state,
            &elements,
        );
        let io = tree
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc container");
        // Wheel 60px: snaps to #b at 100.
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 60.0);
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (0.0, 100.0));
        assert!(
            tree.container(io).snap_rest.get().1.is_some(),
            "snapped to #b"
        );
        // Relayout grows #a: #b's target moves to 150. Resnap follows
        // it instantly — no follow-up animation, so no oscillation.
        let tree2 = snap_relayout(&doc, CSS2, &doc_state, &elements, &tree);
        let io2 = tree2
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc");
        assert_eq!(
            tree2.container(io2).state.offset(),
            (0.0, 150.0),
            "resnap follows the moved target"
        );
        assert!(
            !tree2.has_smooth_in_flight(),
            "resnap moves directly, never via animation"
        );
        // An identical second rebuild changes nothing: the position
        // is already valid, so resnap is a no-op (convergence).
        let tree3 = snap_relayout(&doc, CSS2, &doc_state, &elements, &tree2);
        assert_eq!(tree3.container(io2).state.offset(), (0.0, 150.0));
        assert!(!tree3.has_smooth_in_flight());
        // And the container still counts as snapped to #b's area.
        let rest = tree3.container(io2).snap_rest.get().1.expect("snapped");
        assert_eq!(rest.pos, 150.0);
        assert_eq!(rest.area, 1, "still #b's area, now at its new position");
    }

    /// H2: a relayout mid-snap-animation must not re-seat the
    /// container: the in-flight smooth scroll's target is the
    /// recorded rest position, and the live offset is only an
    /// intermediate point of the animation. Resnap skips the
    /// container; the animation finishes at its target and its
    /// completion evaluation snaps against the new geometry.
    #[test]
    fn resnap_does_not_cancel_in_flight_smooth_scroll() {
        const CSS1: &str = "\
            #sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
            #a { height: 100px; scroll-snap-align: start; } \
            #b { height: 100px; scroll-snap-align: start; } \
            #c { height: 100px; scroll-snap-align: start; }";
        const CSS2: &str = "\
            #sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
            #a { height: 150px; scroll-snap-align: start; } \
            #b { height: 100px; scroll-snap-align: start; } \
            #c { height: 100px; scroll-snap-align: start; }";
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div>\
              <div id=\"b\">y</div><div id=\"c\">z</div></div></body></html>",
        );
        let doc_state = Rc::new(ScrollState::new());
        let elements = Rc::new(ElementScrollStates::new());
        let sheet = parse_stylesheet(CSS1);
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let tree = ScrollTree::build(
            &doc,
            &root,
            Size { w: 800.0, h: 600.0 },
            &doc_state,
            &elements,
        );
        let io = tree
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc container");
        // Wheel 60px: natural rest 60, snap target #b at 100 — the
        // follow-up smooth scroll is in flight, the live offset is
        // the intermediate 60, and the rest position is 100.
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 60.0);
        assert_eq!(tree.container(io).state.offset(), (0.0, 60.0));
        assert!(tree.has_smooth_in_flight());
        assert_eq!(
            tree.container(io).snap_rest.get().1.map(|r| r.pos),
            Some(100.0)
        );
        // Relayout grows #a (#b moves to 150) mid-animation: no
        // re-seat, no cancellation.
        let tree2 = snap_relayout(&doc, CSS2, &doc_state, &elements, &tree);
        let io2 = tree2
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc");
        assert_eq!(
            tree2.container(io2).state.offset(),
            (0.0, 60.0),
            "resnap leaves the mid-animation offset alone"
        );
        assert!(
            tree2.has_smooth_in_flight(),
            "the in-flight smooth scroll survives the relayout"
        );
        // The animation lands on its target (100); its completion
        // evaluation against the new geometry launches the follow-up
        // to #b's new position (150), which then completes there.
        let t0 = Instant::now();
        tree2.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree2.container(io2).state.offset(), (0.0, 100.0));
        assert!(tree2.has_smooth_in_flight());
        tree2.advance_smooth(t0 + Duration::from_secs(20));
        assert_eq!(tree2.container(io2).state.offset(), (0.0, 150.0));
        assert!(!tree2.has_smooth_in_flight());
    }

    /// A removed snap target resnaps to the surviving candidate
    /// nearest the resting position.
    #[test]
    fn resnap_target_removed_snaps_to_nearest_surviving() {
        const CSS1: &str = "\
            #sc { overflow-y: auto; height: 100px; position: relative; \
                   scroll-snap-type: y mandatory; } \
            #a { position: absolute; top: 0px; height: 100px; \
                 scroll-snap-align: start; } \
            #b { position: absolute; top: 100px; height: 150px; \
                 scroll-snap-align: start; } \
            #c { position: absolute; top: 250px; height: 100px; \
                 scroll-snap-align: start; } \
            #space { height: 400px; }";
        const CSS2: &str = "\
            #sc { overflow-y: auto; height: 100px; position: relative; \
                   scroll-snap-type: y mandatory; } \
            #a { position: absolute; top: 0px; height: 100px; \
                 scroll-snap-align: start; } \
            #b { display: none; } \
            #c { position: absolute; top: 250px; height: 100px; \
                 scroll-snap-align: start; } \
            #space { height: 400px; }";
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div>\
              <div id=\"b\">y</div><div id=\"c\">z</div><div id=\"space\"></div>\
              </div></body></html>",
        );
        let doc_state = Rc::new(ScrollState::new());
        let elements = Rc::new(ElementScrollStates::new());
        let sheet = parse_stylesheet(CSS1);
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let tree = ScrollTree::build(
            &doc,
            &root,
            Size { w: 800.0, h: 600.0 },
            &doc_state,
            &elements,
        );
        let io = tree
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc container");
        tree.container(io).state.set_offset(0.0, 60.0);
        tree.mark_scrolled(&[Some(doc.get_element_by_id("sc").expect("sc"))]);
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (0.0, 100.0));
        // #b vanishes: of the survivors (0, 250), 0 is nearest the
        // resting 100.
        let tree2 = snap_relayout(&doc, CSS2, &doc_state, &elements, &tree);
        assert_eq!(tree2.container(io).state.offset(), (0.0, 0.0));
        let rest = tree2.container(io).snap_rest.get().1.expect("snapped");
        assert_eq!(rest.pos, 0.0);
    }

    /// M8: the snapped state is keyed on the target's node identity,
    /// not its index in the area list. Reordering two same-size,
    /// same-stop targets (here a DOM move of `#a` to the end, which
    /// also moves `#b` from index 1 / position 100 to index 0 /
    /// position 0) makes resnap follow `#b` the node — the old
    /// index+size+stop identity matched `#c`, the area that shifted
    /// into index 1, and stayed at 100.
    #[test]
    fn resnap_follows_node_identity_across_reorder() {
        const CSS: &str = "\
            #sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
            #a { height: 100px; scroll-snap-align: start; } \
            #b { height: 100px; scroll-snap-align: start; } \
            #c { height: 100px; scroll-snap-align: start; }";
        let (mut doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div>\
              <div id=\"b\">y</div><div id=\"c\">z</div></div></body></html>",
        );
        let doc_state = Rc::new(ScrollState::new());
        let elements = Rc::new(ElementScrollStates::new());
        let sheet = parse_stylesheet(CSS);
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let tree = ScrollTree::build(
            &doc,
            &root,
            Size { w: 800.0, h: 600.0 },
            &doc_state,
            &elements,
        );
        let sc = doc.get_element_by_id("sc").expect("sc");
        let a = doc.get_element_by_id("a").expect("a");
        let b = doc.get_element_by_id("b").expect("b");
        let io = tree.index_of(sc).expect("sc container");
        // Snap to #b at 100.
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 60.0);
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (0.0, 100.0));
        let rest = tree.container(io).snap_rest.get().1.expect("snapped");
        assert_eq!(rest.node, Some(b), "the rest records #b's node id");
        // Move #a to the end of #sc: order becomes #b, #c, #a, so
        // #b's area moves to index 0 and its position to 0.
        doc.append_child(sc, a);
        let tree2 = snap_relayout(&doc, CSS, &doc_state, &elements, &tree);
        let io2 = tree2.index_of(sc).expect("sc");
        assert_eq!(
            tree2.container(io2).state.offset(),
            (0.0, 0.0),
            "resnap follows #b to its new position"
        );
        let rest = tree2.container(io2).snap_rest.get().1.expect("snapped");
        assert_eq!(rest.node, Some(b));
        assert_eq!(rest.area, 0, "#b's area is now first in the list");
        assert_eq!(rest.pos, 0.0);
    }

    /// L4: `scroll-snap-stop` / reversal recovery for a container
    /// only considers that container's own participating snap areas.
    /// Here the outer container's area (the nested container's own
    /// box, `stop: normal`) is stacked exactly on top of two
    /// `always` boxes that do NOT participate in the outer
    /// container — the nested container's child, and an abspos box
    /// whose containing block lies outside the outer container —
    /// and must not inherit `always` from either.
    #[test]
    fn snap_stop_not_stolen_from_coincident_non_participating_targets() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\">\
                <div id=\"inner\"><div id=\"kid\">x</div><div id=\"kidtail\"></div></div>\
                <div id=\"ghost\"></div>\
                <div id=\"tail\"></div>\
              </div></body></html>",
            "body { margin: 0px; } \
             #sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
             #inner { overflow-y: auto; height: 100px; scroll-snap-align: start; } \
             #kid { height: 100px; scroll-snap-align: start; scroll-snap-stop: always; } \
             #kidtail { height: 100px; } \
             #ghost { position: absolute; left: 0px; top: 0px; width: 800px; \
                      height: 100px; scroll-snap-align: start; \
                      scroll-snap-stop: always; } \
             #tail { height: 200px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let inner = doc.get_element_by_id("inner").expect("inner");
        let kid = doc.get_element_by_id("kid").expect("kid");
        let oi = tree
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc container");
        let ii = tree.index_of(inner).expect("inner container");
        // The outer container has exactly one area (the nested
        // container's own box; its in-flow twin inside the nested
        // container is pruned one-for-one).
        let areas = &tree.container(oi).snap_areas;
        assert_eq!(areas.len(), 1);
        assert_eq!(
            areas[0].stop,
            SnapStop::Normal,
            "the outer target stays normal, not the coincident always"
        );
        assert_eq!(areas[0].node, Some(inner));
        // The nested container's own area keeps its real `always`.
        let inner_areas = &tree.container(ii).snap_areas;
        assert_eq!(inner_areas.len(), 1);
        assert_eq!(inner_areas[0].stop, SnapStop::Always);
        assert_eq!(inner_areas[0].node, Some(kid));
    }

    /// M8 (winner side): when the resting position lies inside a
    /// covering area's range, the covering area is the winner — not
    /// a different area whose point position happens to coincide.
    #[test]
    fn snap_winner_prefers_covering_area() {
        let point = SnapCandidate {
            pos: 150.0,
            always: false,
            cover_lo: 150.0,
            cover_hi: 150.0,
        };
        let covering = SnapCandidate {
            pos: 100.0,
            always: false,
            cover_lo: 100.0,
            cover_hi: 200.0,
        };
        assert_eq!(snap_winner(&[(0, point), (1, covering)], 150.0), Some(1));
        // Without a covering area, the exact point match still wins.
        let other_point = SnapCandidate {
            pos: 0.0,
            always: false,
            cover_lo: 0.0,
            cover_hi: 0.0,
        };
        assert_eq!(snap_winner(&[(0, point), (1, other_point)], 150.0), Some(0));
    }

    /// A `proximity` container resting between targets is not snapped:
    /// a relayout that moves the targets must not move it.
    #[test]
    fn resnap_proximity_unsnapped_position_stays() {
        const CSS1: &str = "\
            #sc { overflow-y: auto; height: 100px; scroll-snap-type: y proximity; } \
            #a { height: 100px; scroll-snap-align: start; } \
            #gap { height: 300px; } \
            #b { height: 100px; scroll-snap-align: start; }";
        const CSS2: &str = "\
            #sc { overflow-y: auto; height: 100px; scroll-snap-type: y proximity; } \
            #a { height: 100px; scroll-snap-align: start; } \
            #gap { height: 350px; } \
            #b { height: 100px; scroll-snap-align: start; }";
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div>\
              <div id=\"gap\"></div><div id=\"b\">y</div></div></body></html>",
        );
        let doc_state = Rc::new(ScrollState::new());
        let elements = Rc::new(ElementScrollStates::new());
        let sheet = parse_stylesheet(CSS1);
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let tree = ScrollTree::build(
            &doc,
            &root,
            Size { w: 800.0, h: 600.0 },
            &doc_state,
            &elements,
        );
        let io = tree
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc container");
        // Between the targets (200px from both, past the 100px
        // snapport): no snap, no snapped state.
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 200.0);
        assert_eq!(tree.container(io).state.offset(), (0.0, 200.0));
        assert!(tree.container(io).snap_rest.get().1.is_none());
        // The targets move 50px down: the unsnapped position stays.
        let tree2 = snap_relayout(&doc, CSS2, &doc_state, &elements, &tree);
        assert_eq!(tree2.container(io).state.offset(), (0.0, 200.0));
        assert!(!tree2.has_smooth_in_flight());
    }

    /// A `proximity` container that is snapped follows a target that
    /// stays within reach; one that moved farther unsnaps it instead
    /// of chasing it.
    #[test]
    fn resnap_proximity_follows_within_threshold_and_unsnaps_beyond() {
        const BASE: &str = "\
            #sc { overflow-y: auto; height: 100px; scroll-snap-type: y proximity; } \
            #b { height: 100px; scroll-snap-align: start; } \
            #c { height: 100px; scroll-snap-align: start; }";
        let css = |a: u32| format!("#a {{ height: {a}px; scroll-snap-align: start; }} {BASE}");
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div>\
              <div id=\"b\">y</div><div id=\"c\">z</div></div></body></html>",
        );
        let build = |css: &str, doc_state: &Rc<ScrollState>, elements: &Rc<ElementScrollStates>| {
            let sheet = parse_stylesheet(css);
            let root = layout(
                &doc,
                &[&sheet],
                Size { w: 800.0, h: 600.0 },
                &FixedFontMetrics,
            );
            ScrollTree::build(
                &doc,
                &root,
                Size { w: 800.0, h: 600.0 },
                doc_state,
                elements,
            )
        };
        let doc_state = Rc::new(ScrollState::new());
        let elements = Rc::new(ElementScrollStates::new());
        let tree = build(&css(100), &doc_state, &elements);
        let io = tree
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc container");
        // A 70px scroll ends 30px from #b's position: inside the
        // proximity reach (a third of the 100px snapport), so the
        // container snaps to it.
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 70.0);
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (0.0, 100.0));
        assert!(tree.container(io).snap_rest.get().1.is_some());
        // #b's target moves 30px (to 130): still within the proximity
        // reach, so resnap follows it.
        let tree2 = snap_relayout(&doc, &css(130), &doc_state, &elements, &tree);
        assert_eq!(tree2.container(io).state.offset(), (0.0, 130.0));
        // #b's target jumps 300px (to 400): out of reach — the
        // container unsnaps and stays where it was.
        let tree3 = snap_relayout(&doc, &css(400), &doc_state, &elements, &tree2);
        assert_eq!(tree3.container(io).state.offset(), (0.0, 130.0));
        assert!(
            tree3.container(io).snap_rest.get().1.is_none(),
            "target out of reach unsnaps"
        );
    }

    /// A pure `scroll-snap-align` change on a covering target does not
    /// resnap: the resting offset stays a valid snap position of the
    /// same target (its cover range is alignment-invariant).
    #[test]
    fn resnap_covering_target_alignment_change_stays() {
        const CSS1: &str = "\
            #sc { overflow-y: auto; height: 200px; scroll-snap-type: y mandatory; } \
            #big { height: 300px; margin-top: 100px; scroll-snap-align: start; } \
            #space { height: 600px; }";
        const CSS2: &str = "\
            #sc { overflow-y: auto; height: 200px; scroll-snap-type: y mandatory; } \
            #big { height: 300px; margin-top: 100px; scroll-snap-align: end; } \
            #space { height: 600px; }";
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"big\">x</div>\
              <div id=\"space\"></div></div></body></html>",
        );
        let doc_state = Rc::new(ScrollState::new());
        let elements = Rc::new(ElementScrollStates::new());
        let sheet = parse_stylesheet(CSS1);
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let tree = ScrollTree::build(
            &doc,
            &root,
            Size { w: 800.0, h: 600.0 },
            &doc_state,
            &elements,
        );
        let io = tree
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc container");
        // #big covers the snapport over offsets [100, 200]: resting
        // at 150 is valid and records #big as the target.
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 150.0);
        assert_eq!(tree.container(io).state.offset(), (0.0, 150.0));
        assert!(!tree.has_smooth_in_flight());
        assert!(tree.container(io).snap_rest.get().1.is_some());
        // The alignment flips start->end: #big still covers 150, so
        // no resnap — notably NOT the target's new end position (200).
        let tree2 = snap_relayout(&doc, CSS2, &doc_state, &elements, &tree);
        assert_eq!(tree2.container(io).state.offset(), (0.0, 150.0));
    }

    /// Changing a container's snap type takes effect across a
    /// rebuild: `none` -> `mandatory` re-seats a persisted
    /// non-snap offset; `mandatory` -> `none` stops snapping.
    #[test]
    fn resnap_snap_type_change_takes_effect() {
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div>\
              <div id=\"b\">y</div><div id=\"c\">z</div></div></body></html>",
        );
        let css = |t: &str| {
            format!(
                "#sc {{ overflow-y: auto; height: 100px; scroll-snap-type: {t}; }} \
                 #a {{ height: 100px; scroll-snap-align: start; }} \
                 #b {{ height: 100px; scroll-snap-align: start; }} \
                 #c {{ height: 100px; scroll-snap-align: start; }}"
            )
        };
        let doc_state = Rc::new(ScrollState::new());
        let elements = Rc::new(ElementScrollStates::new());
        let sheet = parse_stylesheet(&css("none"));
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let tree = ScrollTree::build(
            &doc,
            &root,
            Size { w: 800.0, h: 600.0 },
            &doc_state,
            &elements,
        );
        let sc = doc.get_element_by_id("sc").expect("sc");
        let io = tree.index_of(sc).expect("sc container");
        // No snap type: an absolute write rests where put.
        tree.container(io).state.set_offset(0.0, 60.0);
        tree.mark_scrolled(&[Some(sc)]);
        assert!(!tree.has_smooth_in_flight());
        assert_eq!(tree.container(io).state.offset(), (0.0, 60.0));
        // Type becomes `y mandatory`: the non-snap offset re-seats to
        // the nearest snap position (100).
        let tree2 = snap_relayout(&doc, &css("y mandatory"), &doc_state, &elements, &tree);
        assert_eq!(tree2.container(io).state.offset(), (0.0, 100.0));
        // And subsequent scrolls snap under the new type.
        tree2.container(io).state.set_offset(0.0, 160.0);
        tree2.mark_scrolled(&[Some(sc)]);
        assert!(tree2.has_smooth_in_flight());
        let t0 = Instant::now();
        tree2.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree2.container(io).state.offset(), (0.0, 200.0));
        // Type back to `none`: nothing snaps anymore, and a relayout
        // never moves the offset.
        let tree3 = snap_relayout(&doc, &css("none"), &doc_state, &elements, &tree2);
        tree3.container(io).state.set_offset(0.0, 60.0);
        tree3.mark_scrolled(&[Some(sc)]);
        assert!(!tree3.has_smooth_in_flight());
        assert_eq!(tree3.container(io).state.offset(), (0.0, 60.0));
        let tree4 = snap_relayout(&doc, &css("none"), &doc_state, &elements, &tree3);
        assert_eq!(tree4.container(io).state.offset(), (0.0, 60.0));
    }

    /// Changing `scroll-snap-type` on the root element takes effect
    /// across a rebuild: the viewport re-snaps under the new type and
    /// stops snapping when it becomes `none`.
    #[test]
    fn root_snap_type_change_takes_effect_across_rebuilds() {
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"t1\">x</div><div id=\"t2\">y</div></body></html>",
        );
        let css = |t: &str| {
            format!(
                "html {{ scroll-snap-type: {t}; }} \
                 #t1 {{ height: 500px; scroll-snap-align: start; }} \
                 #t2 {{ height: 500px; scroll-snap-align: start; }}"
            )
        };
        let doc_state = Rc::new(ScrollState::new());
        let elements = Rc::new(ElementScrollStates::new());
        let sheet = parse_stylesheet(&css("none"));
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let tree = ScrollTree::build(
            &doc,
            &root,
            Size { w: 800.0, h: 600.0 },
            &doc_state,
            &elements,
        );
        let (origin, max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_min_offset(origin);
        doc_state.set_max_offset(max);
        assert_eq!(tree.container(0).snap.snap_type, SnapType::None);
        // No snap type: an instant write rests where put.
        doc_state.set_offset(0.0, 240.0);
        tree.mark_scrolled(&[None]);
        assert!(!tree.has_smooth_in_flight());
        assert_eq!(doc_state.offset(), (0.0, 240.0));
        // Root type becomes `y mandatory`: the 240 offset re-seats to
        // the nearest snap position (0) on rebuild.
        let tree2 = snap_relayout(&doc, &css("y mandatory"), &doc_state, &elements, &tree);
        assert_eq!(tree2.container(0).snap.snap_type, SnapType::MandatoryY);
        assert_eq!(doc_state.offset(), (0.0, 0.0));
        // And the viewport now snaps after script scrolls.
        doc_state.set_offset(0.0, 240.0);
        tree2.mark_scrolled(&[None]);
        assert!(tree2.has_smooth_in_flight());
        let t0 = Instant::now();
        tree2.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(doc_state.offset(), (0.0, 0.0));
        // Root type back to `none`: the viewport no longer snaps, and
        // relayouts leave the offset alone.
        let tree3 = snap_relayout(&doc, &css("none"), &doc_state, &elements, &tree2);
        assert_eq!(tree3.container(0).snap.snap_type, SnapType::None);
        doc_state.set_offset(0.0, 240.0);
        tree3.mark_scrolled(&[None]);
        assert!(!tree3.has_smooth_in_flight());
        assert_eq!(doc_state.offset(), (0.0, 240.0));
        let tree4 = snap_relayout(&doc, &css("none"), &doc_state, &elements, &tree3);
        assert_eq!(tree4.container(0).snap.snap_type, SnapType::None);
        assert_eq!(doc_state.offset(), (0.0, 240.0));
    }

    /// A scroll chain that moves several nested snap containers
    /// re-evaluates each of them against its own areas (the outer's
    /// areas exclude the inner subtree by construction), so every
    /// level settles on a snap position.
    #[test]
    fn nested_chain_snaps_every_moved_container() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"outer\">\
                <div id=\"a\">a</div>\
                <div id=\"inner\">\
                  <div id=\"i1\">x</div><div id=\"i2\">y</div><div id=\"i3\">z</div>\
                </div>\
                <div id=\"b\">b</div>\
              </div></body></html>",
            "#outer { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
             #a { height: 100px; scroll-snap-align: start; } \
             #inner { overflow-y: auto; height: 100px; scroll-snap-align: start; \
                      scroll-snap-type: y mandatory; } \
             #i1 { height: 100px; scroll-snap-align: start; } \
             #i2 { height: 100px; scroll-snap-align: start; } \
             #i3 { height: 100px; scroll-snap-align: start; } \
             #b { height: 100px; scroll-snap-align: start; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let inner = doc.get_element_by_id("inner").expect("inner");
        let io = tree.index_of(outer).expect("outer container");
        let ii = tree.index_of(inner).expect("inner container");
        // Outer candidates: #a at 0, #inner's own box at 100, #b at
        // 200. Inner candidates: 0, 100, 200. (#inner's box is an
        // area of the OUTER container; #i1..3 prune into the inner.)
        assert_eq!(tree.container(io).snap_areas.len(), 3);
        assert_eq!(tree.container(ii).snap_areas.len(), 3);
        // A 260px scroll on the inner: inner consumes 200 (to its max),
        // the outer takes the 60 remainder.
        let o = tree.scroll_chain(ii, 0.0, 260.0);
        assert!(o.moved());
        // Inner rests on its 200 position (no follow-up); outer rests
        // at 60, nearest its 100 position — a follow-up launches.
        assert_eq!(tree.container(ii).state.offset(), (0.0, 200.0));
        assert_eq!(tree.container(io).state.offset(), (0.0, 60.0));
        assert!(tree.has_smooth_in_flight());
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(ii).state.offset(), (0.0, 200.0));
        assert_eq!(tree.container(io).state.offset(), (0.0, 100.0));
        assert!(!tree.has_smooth_in_flight());
        // Both containers count as snapped on their own targets.
        let inner_rest = tree.container(ii).snap_rest.get().1.expect("inner snapped");
        assert_eq!(inner_rest.pos, 200.0);
        let outer_rest = tree.container(io).snap_rest.get().1.expect("outer snapped");
        assert_eq!(outer_rest.pos, 100.0);
    }

    /// `reveal_node` through a nested snap container leaves every
    /// moved container snapped against its own areas — not just the
    /// innermost scroller's.
    #[test]
    fn reveal_through_nested_snap_containers_snaps_each() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"outer\">\
                <div id=\"a\">a</div>\
                <div id=\"inner\">\
                  <div id=\"i1\">x</div><div id=\"i2\">y</div><div id=\"i3\">z</div>\
                </div>\
                <div id=\"f\"></div>\
                <div id=\"b\">b</div>\
              </div></body></html>",
            "#outer { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
             #a { height: 100px; scroll-snap-align: start; } \
             #inner { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
             #i1 { height: 100px; scroll-snap-align: start; } \
             #i2 { height: 100px; scroll-snap-align: start; } \
             #f { height: 50px; } \
             #b { height: 100px; scroll-snap-align: start; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let inner = doc.get_element_by_id("inner").expect("inner");
        let i2 = doc.get_element_by_id("i2").expect("i2");
        let io = tree.index_of(outer).expect("outer container");
        let ii = tree.index_of(inner).expect("inner container");
        // Outer candidates: #a at 0, #b at 250 (#inner carries no
        // `scroll-snap-align`, so it contributes none of its own).
        assert_eq!(tree.container(io).snap_areas.len(), 2);
        // Revealing #i2 (content y=100) aligns the inner scroller at
        // 100 — its own snap position — and moves the outer scroller
        // so the inner's scrollport top (outer content y=100) meets
        // the outer scrollport's top: 100 is no snap position of the
        // outer (0/250), so it settles on 0, the nearer one.
        let scrolled = tree.reveal_node(&root, i2);
        assert!(scrolled.contains(&Some(inner)));
        assert!(scrolled.contains(&Some(outer)));
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(ii).state.offset(), (0.0, 100.0));
        assert_eq!(
            tree.container(io).state.offset(),
            (0.0, 0.0),
            "the outer container settles on its own snap position too"
        );
    }

    /// Snap points orthogonal to the snap axis are ignored (CSS Scroll
    /// Snap §5.2): on `x mandatory`, the target's y position never
    /// moves the y axis (the old proximity-gated fallback on the
    /// uncovered axis dragged it to the target's y).
    #[test]
    fn snap_points_orthogonal_to_snap_axis_ignored() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"t\">x</div>\
              <div id=\"area\"></div></div></body></html>",
            "#sc { height: 500px; width: 500px; overflow: hidden; \
                   position: relative; scroll-snap-type: x mandatory; } \
             #t { position: absolute; width: 300px; height: 300px; \
                  left: 100px; top: 100px; scroll-snap-align: start; } \
             #area { width: 2000px; height: 2000px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let io = tree
            .index_of(doc.get_element_by_id("sc").expect("sc"))
            .expect("sc container");
        // An absolute scroll to (50, 30) snaps x to the target's 100
        // and leaves y exactly where put.
        tree.container(io).state.set_offset(50.0, 30.0);
        tree.eval_snap(io, (0.0, 0.0), (50.0, 30.0), false);
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (100.0, 30.0));
    }

    #[test]
    fn bake_paint_tree_borrows_without_nested_scrollers() {
        let (doc, root) = layout_html(b"<html><body><div>x</div></body></html>", "");
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert!(tree.is_viewport_only());
        // Fast path: no clone, the paint tree borrows the layout tree.
        assert!(matches!(tree.bake_paint_tree(&root), Cow::Borrowed(_)));
    }

    #[test]
    fn bake_paint_tree_clones_with_nested_scrollers() {
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert!(!tree.is_viewport_only());
        assert!(matches!(tree.bake_paint_tree(&root), Cow::Owned(_)));
    }

    // ------------------------------------------------------------------
    // Harsh-review findings
    // ------------------------------------------------------------------

    #[test]
    fn document_max_scroll_ignores_fixed_subtrees() {
        // CSS Overflow 3 §2.2: fixed boxes never extend document
        // scrollable overflow — the `top: -50px` rect must not publish
        // a bogus negative origin, and the 2000px height must not
        // inflate the max.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"fx\">x</div></body></html>",
            "#fx { position: fixed; top: -50px; left: 0; width: 100px; height: 2000px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert_eq!(
            tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 }),
            ((0.0, 0.0), (0.0, 0.0))
        );
    }

    #[test]
    fn sticky_band_subtracts_nested_scrollport_origin() {
        // #outer's scrollport starts at page y=500; the sticky band for
        // #st must measure from the scrollport origin, not the page.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"pad\">x</div><div id=\"outer\">\
              <div id=\"st\">y</div><div id=\"tall\">z</div></div></body></html>",
            "#pad { height: 500px; } \
             #outer { overflow-y: auto; height: 100px; } \
             #st { position: sticky; top: 0; height: 20px; } \
             #tall { height: 500px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        let band = tree
            .sticky_bands
            .iter()
            .find(|b| b.scroller == io)
            .expect("sticky band for #st");
        // lo_y = (r.y - sp.y) - top - sp_h - slack
        //      = (500 - 500) - 0 - 100 - 2 = -102.
        assert_eq!(band.lo_y, -100.0 - STICKY_BAND_SLACK);
    }

    /// Layout with a nested scroller pre-scrolled to `(x, y)`: mirrors
    /// what the window worker feeds layout after a scroll, so a tree
    /// rebuilt over it bakes its sticky bands at the live offset.
    fn layout_html_nested_at(
        html: &[u8],
        css: &str,
        scroller_id: &str,
        x: f32,
        y: f32,
    ) -> (Document, LayoutBox) {
        let (doc, _) = Document::parse_html(html);
        let sheet = parse_stylesheet(css);
        let node = doc.get_element_by_id(scroller_id).expect("scroller");
        let mut nested = HashMap::new();
        nested.insert(node, (x, y));
        let root = bowser_layout::layout_with_scroll_and_offsets(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
            (0.0, 0.0),
            &nested,
        );
        (doc, root)
    }

    /// Build a tree whose sticky bands bake at a live nested offset:
    /// first build publishes the shared states, then the container is
    /// moved to the layout's offset and the tree is rebuilt — offsets
    /// survive relayout through the shared state, like the window
    /// worker's transition re-bake.
    fn build_tree_nested_at(
        doc: &Document,
        root: &LayoutBox,
        scroller_id: &str,
        x: f32,
        y: f32,
    ) -> (Rc<ScrollState>, Rc<ElementScrollStates>, ScrollTree) {
        let doc_state = Rc::new(ScrollState::new());
        let elements = Rc::new(ElementScrollStates::new());
        let size = Size { w: 800.0, h: 600.0 };
        let tree = ScrollTree::build(doc, root, size, &doc_state, &elements);
        let scroller = doc.get_element_by_id(scroller_id).expect("scroller");
        let io = tree.index_of(scroller).expect("scroller container");
        tree.container(io).state.set_offset(x, y);
        let tree = ScrollTree::build(doc, root, size, &doc_state, &elements);
        (doc_state, elements, tree)
    }

    const STICKY_NESTED_HTML: &[u8] = b"<html><body><div id=\"outer\">\
        <div id=\"pad\">x</div><div id=\"st\">y</div><div id=\"tall\">z</div>\
        </div></body></html>";
    const STICKY_NESTED_CSS: &str = "#outer { overflow-y: auto; height: 400px; } \
        #pad { height: 100px; } \
        #st { position: sticky; top: 0; height: 50px; } \
        #tall { height: 2000px; }";

    #[test]
    fn sticky_nested_stuck_plateau_folds_without_rebake() {
        // Sticky header in a nested scroller: once the transition's
        // re-bake has baked the bands at a stuck offset, further ticks
        // along the plateau must stay on the fast path — no re-bake,
        // exact fold — instead of a full display-list rebuild per tick.
        // (#st natural at 100, stuck from 100 to the block end 350.)
        let (doc, root) =
            layout_html_nested_at(STICKY_NESTED_HTML, STICKY_NESTED_CSS, "outer", 0.0, 150.0);
        let (_doc_state, _elements, tree) = build_tree_nested_at(&doc, &root, "outer", 0.0, 150.0);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let st = doc.get_element_by_id("st").expect("st");
        let io = tree.index_of(outer).expect("outer container");
        let sp_y = tree.container(io).scrollport.y;
        // Sanity: the band baked stuck, against the scroller's content.
        let band = tree
            .sticky_bands
            .iter()
            .find(|b| b.scroller == io)
            .expect("band for #st");
        assert!(band.cb_is_scroller, "cb is #outer's own box");
        assert!(approx_eq(band.pos_y, 150.0), "pos_y={}", band.pos_y);
        assert!(approx_eq(band.bake_y, 150.0), "bake_y={}", band.bake_y);
        // Ticks further down the plateau: no re-bake, and the extra
        // shift is exactly the scroll delta.
        for oy in [200.0, 250.0, 300.0] {
            assert!(
                !tree.sticky_constrained_at(io, 0.0, oy),
                "stuck plateau must not re-bake at {oy}"
            );
            let shifts = tree.sticky_shifts_at(io, 0.0, oy).expect("foldable");
            assert_eq!(shifts.len(), 1, "exactly the sticky box");
            assert_eq!(shifts[0].node, st);
            assert!(approx_eq(shifts[0].dx, 0.0));
            assert!(
                approx_eq(shifts[0].dy, oy - 150.0),
                "extra shift is exactly the scroll delta at {oy}"
            );
            // Behavior equivalence: an independent layout at `oy`
            // agrees the box is stuck exactly at the scrollport top.
            let (ref_doc, ref_root) =
                layout_html_nested_at(STICKY_NESTED_HTML, STICKY_NESTED_CSS, "outer", 0.0, oy);
            let ref_st = ref_doc.get_element_by_id("st").expect("ref st");
            let ref_box = find_box(&ref_root, ref_st).expect("#st box");
            assert!(
                approx_eq(ref_box.rect.y - sp_y, oy),
                "layout agrees the box is stuck at {oy}"
            );
        }
        // Past the block end (350): re-bake to re-track.
        assert!(tree.sticky_constrained_at(io, 0.0, 400.0));
        assert!(tree.sticky_shifts_at(io, 0.0, 400.0).is_none());
    }

    #[test]
    fn sticky_nested_transitions_still_rebake() {
        // Same page baked stuck at 150: leaving the plateau needs the
        // re-bake path — the transition is not exactly foldable.
        let (doc, root) =
            layout_html_nested_at(STICKY_NESTED_HTML, STICKY_NESTED_CSS, "outer", 0.0, 150.0);
        let (_doc_state, _elements, tree) = build_tree_nested_at(&doc, &root, "outer", 0.0, 150.0);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        // Scrolling back below the stick threshold (100): the box unsticks.
        assert!(tree.sticky_constrained_at(io, 0.0, 50.0));
        assert!(tree.sticky_shifts_at(io, 0.0, 50.0).is_none());
        // Scrolling back while still stuck: conservative re-bake (the
        // unstick point needs the unrecoverable natural position).
        assert!(tree.sticky_constrained_at(io, 0.0, 120.0));
        assert!(tree.sticky_shifts_at(io, 0.0, 120.0).is_none());
        // Scrolling past the containing-block end (350): re-bake to re-track.
        assert!(tree.sticky_constrained_at(io, 0.0, 400.0));
        assert!(tree.sticky_shifts_at(io, 0.0, 400.0).is_none());
    }

    #[test]
    fn sticky_nested_free_interval_folds_without_rebake() {
        // Baked at offset 0 with the box still free (natural 100):
        // ticks that provably cannot reach a transition fold to zero
        // shift instead of re-baking. The provable free interval is
        // [100+50-400, 100] = [-250, 100], shrunk by slack.
        let (doc, root) = layout_html(STICKY_NESTED_HTML, STICKY_NESTED_CSS);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        assert!(!tree.sticky_constrained_at(io, 0.0, 50.0));
        let shifts = tree.sticky_shifts_at(io, 0.0, 50.0).expect("foldable");
        assert!(shifts.is_empty(), "free box needs no extra shift");
        // Approaching the stick transition: re-bake.
        assert!(tree.sticky_constrained_at(io, 0.0, 99.0));
        // Far away (a fling past the transition): re-bake — the box
        // may have stuck, so the fold cannot claim zero shift.
        assert!(tree.sticky_constrained_at(io, 0.0, 500.0));
        assert!(tree.sticky_shifts_at(io, 0.0, 500.0).is_none());
    }

    #[test]
    fn sticky_viewport_keeps_rebake_gate() {
        // The viewport scroller has no per-box fold mechanism (its
        // shift rides the GPU uniform): in-band ticks still re-bake.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"pad\">x</div><div id=\"st\">s</div>\
              <div id=\"tall\">x</div></body></html>",
            "#pad { height: 5000px; } #st { position: sticky; top: 0; height: 50px; } \
             #tall { height: 5000px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert!(tree.has_sticky());
        assert!(tree.sticky_constrained_at(0, 0.0, 4600.0));
        assert!(tree.sticky_shifts_at(0, 0.0, 4600.0).is_none());
        assert!(!tree.sticky_constrained_at(0, 0.0, 120.0));
    }

    #[test]
    fn sticky_nested_foreign_cb_stays_conservative() {
        // Sticky box inside a definite-height wrapper: the
        // containing-block end is unprovable app-side, so a stuck box
        // keeps the re-bake path (no behavior change, just no fold).
        const HTML: &[u8] = b"<html><body><div id=\"outer\"><div id=\"wrap\">\
            <div id=\"st\">y</div><div id=\"tall\">z</div></div></div></body></html>";
        const CSS: &str = "#outer { overflow-y: auto; height: 100px; } \
            #wrap { height: 150px; } \
            #st { position: sticky; top: 0; height: 20px; } \
            #tall { height: 2000px; }";
        let (doc, root) = layout_html_nested_at(HTML, CSS, "outer", 0.0, 50.0);
        let (_doc_state, _elements, tree) = build_tree_nested_at(&doc, &root, "outer", 0.0, 50.0);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        let band = tree
            .sticky_bands
            .iter()
            .find(|b| b.scroller == io)
            .expect("band for #st");
        assert!(!band.cb_is_scroller, "cb is #wrap, not #outer");
        // Stuck (natural 0, baked at 50) but the far edge is unprovable.
        assert!(tree.sticky_constrained_at(io, 0.0, 80.0));
        assert!(tree.sticky_shifts_at(io, 0.0, 80.0).is_none());
    }

    #[test]
    fn deep_tree_walks_do_not_overflow() {
        // 30k nested divs: every tree walk in this file is iterative,
        // so discovery, `body_style`, `document_max_scroll` and
        // `reveal_chain` must survive. Runs on a 64MB-stack thread:
        // `LayoutBox`/`Document` drop recurses through the box spine
        // (sibling crates, outside this file's ownership), so teardown
        // of a 30k-deep tree needs more than the default 2MB test
        // stack even though every walk above it is iterative.
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(|| {
                let depth = 30_000;
                let mut html = b"<html><body>".to_vec();
                for i in 0..depth {
                    html.extend(format!("<div id=\"d{i}\">").bytes());
                }
                html.extend(b"x");
                for _ in 0..depth {
                    html.extend(b"</div>");
                }
                html.extend(b"</body></html>");
                let (doc, root) = layout_html(&html, "");
                let (_doc_state, _elements, tree) = build_tree(&doc, &root);
                let _ = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
                let deep = doc
                    .get_element_by_id(&format!("d{}", depth - 1))
                    .expect("deepest div");
                // Nothing scrollable: the point is surviving the walk.
                let scrolled = tree.reveal_node_with_options(
                    &root,
                    deep,
                    RevealAlign::Start,
                    RevealAlign::Start,
                    ScrollBehavior::Instant,
                );
                assert!(scrolled.is_empty());
            })
            .expect("spawn deep-tree thread")
            .join()
            .expect("deep-tree thread survived");
    }

    #[test]
    fn start_smooth_scroll_noop_keeps_in_flight() {
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        // A real animation is in flight toward 200.
        tree.start_smooth_scroll(io, 0.0, 200.0);
        assert!(tree.has_smooth_in_flight());
        // A no-op command (target == current offset) must neither kill
        // it nor replace it with a 300ms no-op animation (finding #6).
        let from = tree.container(io).state.offset();
        tree.start_smooth_scroll(io, from.0, from.1);
        assert!(tree.has_smooth_in_flight());
        let anim = tree.container(io).smooth.get().expect("still in flight");
        assert_eq!(anim.to, (0.0, 200.0));
        // And it still completes to its real target.
        let t0 = Instant::now();
        tree.advance_smooth(t0 + Duration::from_secs(10));
        assert_eq!(tree.container(io).state.offset(), (0.0, 200.0));
    }

    #[test]
    fn scroll_chain_preserves_subpixel_residue() {
        // Inner max is exactly 200; the 1/8192px residue past it must
        // chain to the viewport, not die as float dust (finding #8).
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div>\
              <div id=\"doc-tall\">y</div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 300px; } \
             #doc-tall { height: 2000px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (_, doc_max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_max_offset(doc_max);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");
        let cstate = tree.container(isc).state.clone();
        tree.scroll_chain(tree.innermost_at(10.0, 10.0), 0.0, 200.0 + 1.0 / 8192.0);
        assert_eq!(cstate.offset(), (0.0, 200.0));
        assert_eq!(doc_state.offset(), (0.0, 1.0 / 8192.0));
    }

    #[test]
    fn reveal_respects_scroll_padding() {
        // The standard fixed-header offset technique:
        // `scroll-padding-top` insets the snapport reveal aligns to.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"sp\"></div>\
              <div id=\"a\">x</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; scroll-padding-top: 30px; } \
             #sp { height: 100px; } \
             #a { height: 100px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let a = doc.get_element_by_id("a").expect("a");
        let sc = doc.get_element_by_id("sc").expect("sc");
        let isc = tree.index_of(sc).expect("sc container");
        let scrolled = tree.reveal_node_with_options(
            &root,
            a,
            RevealAlign::Start,
            RevealAlign::Start,
            ScrollBehavior::Instant,
        );
        // #a at content y=100 aligns to the padded snapport top (30):
        // target 70, not 100.
        assert_eq!(scrolled, vec![Some(sc)]);
        assert_eq!(tree.container(isc).state.offset(), (0.0, 70.0));
    }

    #[test]
    fn reveal_into_fixed_container_uses_viewport_frame() {
        // The fixed container's scrollport is viewport-frame; with the
        // document scrolled to 500 the reveal target must still be
        // computed against the viewport frame (finding #1).
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"tall\">x</div><div id=\"fx\">\
              <div id=\"sp\"></div><div id=\"in\">y</div></div></body></html>",
        );
        let sheet = parse_stylesheet(
            "#tall { height: 2000px; } \
             #fx { position: fixed; top: 10px; left: 10px; width: 100px; \
                   height: 100px; overflow: auto; } \
             #sp { height: 300px; } #in { height: 100px; }",
        );
        let root = bowser_layout::layout_with_scroll(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
            (0.0, 500.0),
        );
        let doc_state = Rc::new(ScrollState::new());
        doc_state.set_max_offset((0.0, 1400.0));
        doc_state.set_offset(0.0, 500.0);
        let elements = Rc::new(ElementScrollStates::new());
        let tree = ScrollTree::build(
            &doc,
            &root,
            Size { w: 800.0, h: 600.0 },
            &doc_state,
            &elements,
        );
        let fx = doc.get_element_by_id("fx").expect("fx");
        let ifx = tree.index_of(fx).expect("fx container");
        let inner = doc.get_element_by_id("in").expect("in");
        tree.reveal_node_with_options(
            &root,
            inner,
            RevealAlign::Start,
            RevealAlign::Start,
            ScrollBehavior::Instant,
        );
        // #in at viewport y=310, scrollport top at viewport y=10:
        // target 300. A stale page-frame translation would have
        // produced 310 - 510 < 0, clamped to 0.
        assert_eq!(tree.container(ifx).state.offset(), (0.0, 300.0));
    }

    // ------------------------------------------------------------------
    // Scrollbars
    // ------------------------------------------------------------------

    /// Real tree for [`SCROLL_HTML`]/[`SCROLL_CSS`]: `#outer`
    /// overflows vertically (100px scrollport, 300px content), `#wide`
    /// horizontally, `#hidden` is `overflow: hidden`.
    fn scrollbar_tree() -> (Document, LayoutBox, ScrollTree) {
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_, _, tree) = build_tree(&doc, &root);
        (doc, root, tree)
    }

    #[test]
    fn scrollbar_presence_needs_user_axis_and_overflow() {
        let (doc, _root, tree) = scrollbar_tree();
        // `#outer`: vertical overflow only.
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        let g = tree.scrollbar_geom(io);
        assert!(g.vertical.is_some(), "outer overflows vertically");
        assert!(g.horizontal.is_none(), "outer has no horizontal overflow");
        // `#wide`: horizontal overflow only.
        let wide = doc.get_element_by_id("wide").expect("wide");
        let iw = tree.index_of(wide).expect("wide container");
        let gw = tree.scrollbar_geom(iw);
        assert!(gw.horizontal.is_some(), "wide overflows horizontally");
        assert!(gw.vertical.is_none(), "wide has no vertical overflow");
        // `overflow: hidden` shows no scrollbar even when overflowing.
        let hidden = doc.get_element_by_id("hidden").expect("hidden");
        let ih = tree.index_of(hidden).expect("hidden container");
        let gh = tree.scrollbar_geom(ih);
        assert!(
            gh.vertical.is_none() && gh.horizontal.is_none(),
            "overflow:hidden shows no scrollbar"
        );
        // The viewport fits the 800x600 viewport: no bars.
        let gv = tree.scrollbar_geom(0);
        assert!(
            gv.vertical.is_none() && gv.horizontal.is_none(),
            "viewport does not overflow"
        );
        // Out-of-range index: no bars, no panic.
        let go = tree.scrollbar_geom(tree.len() + 7);
        assert!(go.vertical.is_none() && go.horizontal.is_none());
    }

    #[test]
    fn scrollbar_track_sits_on_scrollport_edges() {
        let (doc, _root, tree) = scrollbar_tree();
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        let g = tree.scrollbar_geom(io);
        let v = g.vertical.expect("vertical bar");
        let sp = tree.container(io).scrollport;
        // Inline-end edge, full scrollport height (no horizontal bar to
        // share the corner with).
        assert!(approx_eq(v.track.x, sp.x + sp.w - SCROLLBAR_THICKNESS));
        assert!(approx_eq(v.track.y, sp.y));
        assert!(approx_eq(v.track.w, SCROLLBAR_THICKNESS));
        assert!(approx_eq(v.track.h, sp.h));
        // The thumb sits inside its track.
        assert!(v.thumb.y >= v.track.y);
        assert!(v.thumb.y + v.thumb.h <= v.track.y + v.track.h + 1e-3);
        assert!(approx_eq(v.thumb.x, v.track.x));
        assert!(approx_eq(v.thumb.w, v.track.w));
    }

    #[test]
    fn scrollbar_vertical_track_flips_under_rtl() {
        // `direction: rtl` mirrors the inline-end edge: the vertical
        // track sits on the scrollport's left edge instead of the
        // right. The horizontal track is unaffected (block-end edge).
        let (doc, root) = layout_html(
            b"<html><body><div id=\"rtl\"><div id=\"tall\">x</div></div></body></html>",
            "#rtl { overflow-y: auto; height: 100px; width: 200px; direction: rtl; } \
             #tall { height: 300px; }",
        );
        let (_, _, tree) = build_tree(&doc, &root);
        let rtl = doc.get_element_by_id("rtl").expect("rtl");
        let ir = tree.index_of(rtl).expect("rtl container");
        assert!(
            tree.container(ir).rtl,
            "computed direction: rtl is recorded"
        );
        let g = tree.scrollbar_geom(ir);
        let v = g.vertical.expect("vertical bar");
        let sp = tree.container(ir).scrollport;
        assert!(
            approx_eq(v.track.x, sp.x),
            "rtl: track on the left edge, got x={} scrollport.x={}",
            v.track.x,
            sp.x
        );
        assert!(approx_eq(v.track.w, SCROLLBAR_THICKNESS));
        assert!(approx_eq(v.track.h, sp.h));
        // The thumb sits inside its track.
        assert!(v.thumb.y >= v.track.y);
        assert!(v.thumb.y + v.thumb.h <= v.track.y + v.track.h + 1e-3);
        // LTR control: the shared fixture's track stays on the right.
        let (doc2, _root2, tree2) = scrollbar_tree();
        let outer = doc2.get_element_by_id("outer").expect("outer");
        let io = tree2.index_of(outer).expect("outer container");
        assert!(!tree2.container(io).rtl);
        let v2 = tree2.scrollbar_geom(io).vertical.expect("bar");
        let sp2 = tree2.container(io).scrollport;
        assert!(approx_eq(v2.track.x, sp2.x + sp2.w - SCROLLBAR_THICKNESS));
    }

    #[test]
    fn forced_scrollbar_content_stops_at_painted_track() {
        // Layout/paint seam for `overflow: scroll`: layout narrows the
        // content box by the forced strip
        // (`bowser_layout::forced_scrollbar_reservation`) and paint
        // puts the track exactly there
        // (`vertical_track_rect`) — content must not underlap the
        // scrollbar. The child's border box ends where the track
        // begins, which is also what WPT
        // `scrollbar-gutter-root.html`'s `offsetWidth` emulation
        // measures.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"forced\"><div id=\"kid\">x</div></div></body></html>",
            "body { margin: 0; } \
             #forced { overflow: scroll; width: 100px; height: 100px; } \
             #kid { width: 100%; height: 200%; }",
        );
        let (_, _, tree) = build_tree(&doc, &root);
        let forced = doc.get_element_by_id("forced").expect("forced");
        let kid = doc.get_element_by_id("kid").expect("kid");
        let idx = tree.index_of(forced).expect("forced container");
        let v = tree.scrollbar_geom(idx).vertical.expect("forced bar");
        let kid_box = find_box(&root, kid).expect("kid box");
        assert!(
            approx_eq(kid_box.rect.x + kid_box.rect.w, v.track.x),
            "child right edge {} != track x {}",
            kid_box.rect.x + kid_box.rect.w,
            v.track.x
        );
        assert!(
            approx_eq(v.track.w, SCROLLBAR_THICKNESS),
            "track width {} != 12",
            v.track.w
        );
    }

    #[test]
    fn gutter_mode_parses_stable_keywords() {
        // The property is registered and validated in `bowser_css`, so
        // the computed value is `auto`, `stable`, or the `stable
        // both-edges` keyword list (either order): map those shapes
        // directly. Anything else reserves nothing.
        use bowser_css::CssValue;
        let mode = |v: &Option<CssValue>| gutter_mode_from_value(v.as_ref());
        let kw = |s: &str| Some(CssValue::Keyword(s.into()));
        let list = |words: &[&str]| {
            Some(CssValue::List(
                words
                    .iter()
                    .map(|w| CssValue::Keyword((*w).into()))
                    .collect(),
            ))
        };
        assert_eq!(mode(&None), ScrollbarGutter::Auto);
        assert_eq!(mode(&kw("auto")), ScrollbarGutter::Auto);
        assert_eq!(
            mode(&kw("stable")),
            ScrollbarGutter::Stable { both_edges: false }
        );
        assert_eq!(
            mode(&list(&["stable", "both-edges"])),
            ScrollbarGutter::Stable { both_edges: true }
        );
        assert_eq!(
            mode(&list(&["both-edges", "stable"])),
            ScrollbarGutter::Stable { both_edges: true },
            "keyword order is irrelevant"
        );
        // `both-edges` without `stable` is not a valid value and
        // reserves nothing on its own; neither does garbage.
        assert_eq!(mode(&kw("both-edges")), ScrollbarGutter::Auto);
        assert_eq!(mode(&kw("bogus")), ScrollbarGutter::Auto);
        assert_eq!(mode(&Some(CssValue::Number(1.0))), ScrollbarGutter::Auto);
        assert_eq!(
            mode(&list(&["stable"])),
            ScrollbarGutter::Auto,
            "a lone `stable` in a list is not the registered shape"
        );
    }

    #[test]
    fn gutter_reads_registered_property_from_stylesheet() {
        // `scrollbar-gutter` is a registered computed property, so the
        // cascade keeps the declaration and the scroll tree reads
        // `Stable` when the stylesheet says `stable`.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"g\"><div id=\"g-inner\">x</div></div></body></html>",
            "#g { overflow-y: auto; height: 100px; width: 200px; scrollbar-gutter: stable; } \
             #g-inner { height: 300px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let g = doc.get_element_by_id("g").expect("g");
        let ig = tree.index_of(g).expect("g is a container");
        assert_eq!(
            tree.container(ig).gutter,
            ScrollbarGutter::Stable { both_edges: false }
        );
        assert!(!tree.gutter_rects(ig).is_empty());
    }

    #[test]
    fn touch_action_flows_from_stylesheet_to_container() {
        // `touch-action` is registered in `bowser_css`, so the cascade
        // keeps the declaration and the scroll tree maps it onto the
        // container's pan mask.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"t\"><div id=\"t-inner\">x</div></div></body></html>",
            "#t { overflow-y: auto; height: 100px; width: 200px; touch-action: pan-x; } \
             #t-inner { height: 300px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let t = doc.get_element_by_id("t").expect("t");
        let it = tree.index_of(t).expect("t is a container");
        assert_eq!(
            tree.container(it).touch_action,
            TouchAction {
                x_neg: true,
                x_pos: true,
                y_neg: false,
                y_pos: false,
            }
        );
        // `touch-action: none` on the root propagates to the viewport
        // container (index 0).
        let (doc, root) = layout_html(
            b"<html><body><div style=\"height: 2000px;\">x</div></body></html>",
            "html { touch-action: none; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert_eq!(tree.container(0).touch_action, TouchAction::default());
        // No `touch-action` anywhere: the `auto` initial pans freely.
        let (doc, root) = layout_html(
            b"<html><body><div style=\"height: 2000px;\">x</div></body></html>",
            "",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        assert_eq!(tree.container(0).touch_action, TouchAction::all());
    }

    /// Pointer Events 3 §5.2 on a real nested tree: outer
    /// `touch-action: pan-x`, inner `auto`. A vertical touch delta that
    /// exhausts the inner scroller must not pan the outer vertically —
    /// the touch chain re-gates the residual through each container's
    /// mask as it ascends — while the ungated wheel path
    /// ([`ScrollTree::scroll_chain`]) still chains the same residual
    /// into the outer scroller (`touch-action` does not apply to it).
    #[test]
    fn touch_chain_regates_residual_per_container() {
        let html = b"<html><body>\
            <div id=\"outer\"><div id=\"inner\"><div id=\"tall\">x</div></div>\
            <div id=\"spacer\">y</div></div></body></html>";
        let css = "#outer { overflow-y: auto; height: 200px; touch-action: pan-x; } \
                   #inner { overflow-y: auto; height: 100px; } \
                   #tall { height: 400px; } \
                   #spacer { height: 300px; }";
        let (doc, root) = layout_html(html, css);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = tree
            .index_of(doc.get_element_by_id("outer").expect("outer"))
            .expect("outer is a container");
        let inner = tree
            .index_of(doc.get_element_by_id("inner").expect("inner"))
            .expect("inner is a container");
        let offset = |idx: usize| tree.container(idx).state.offset();
        // Sanity: the masks the test means to exercise.
        assert_eq!(tree.container(inner).touch_action, TouchAction::all());
        assert_ne!(tree.container(outer).touch_action, TouchAction::all());
        // Park the inner scroller at its vertical max so the next
        // downward delta is pure residual.
        let o = tree.scroll_chain(inner, 0.0, 300.0);
        assert!(o.moved());
        assert_eq!(offset(inner), (0.0, 300.0));
        assert_eq!(offset(outer), (0.0, 0.0));
        // Touch path: the exhausted inner passes the vertical
        // remainder up, but the outer `pan-x` mask drops it.
        let o = tree.scroll_chain_touch(inner, 0.0, 100.0);
        assert_eq!(offset(inner), (0.0, 300.0), "inner stays at its edge");
        assert_eq!(offset(outer), (0.0, 0.0), "outer must not pan vertically");
        assert!(
            o.deltas
                .iter()
                .all(|&(node, _)| node != tree.container(outer).node),
            "no outcome delta for the outer container"
        );
        // Wheel path (ungated): the identical residual pans the outer
        // scroller, proving `touch-action` gating is touch-only.
        let o = tree.scroll_chain(inner, 0.0, 100.0);
        assert!(o.moved());
        assert_eq!(offset(outer), (0.0, 100.0), "wheel chains ungated");
    }

    /// Outer `touch-action: none`: a touch-chained residual dies at the
    /// outer container on every axis, even one the inner exhausted.
    #[test]
    fn touch_chain_none_outer_drops_residual() {
        let html = b"<html><body>\
            <div id=\"outer\"><div id=\"inner\"><div id=\"tall\">x</div></div>\
            <div id=\"spacer\">y</div></div></body></html>";
        let css = "#outer { overflow-y: auto; height: 200px; touch-action: none; } \
                   #inner { overflow-y: auto; height: 100px; } \
                   #tall { height: 400px; } \
                   #spacer { height: 300px; }";
        let (doc, root) = layout_html(html, css);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = tree
            .index_of(doc.get_element_by_id("outer").expect("outer"))
            .expect("outer is a container");
        let inner = tree
            .index_of(doc.get_element_by_id("inner").expect("inner"))
            .expect("inner is a container");
        assert_eq!(tree.container(outer).touch_action, TouchAction::default());
        let o = tree.scroll_chain(inner, 0.0, 300.0);
        assert!(o.moved());
        // Touch path: inner exhausted, residual ascends, outer `none`
        // drops it wholesale.
        let o = tree.scroll_chain_touch(inner, 0.0, 100.0);
        assert!(!o.moved(), "nothing may move past a `none` outer");
        assert_eq!(
            tree.container(outer).state.offset(),
            (0.0, 0.0),
            "outer stays put"
        );
    }

    /// Pointer Events 3 §5.2 pre-gate on a real tree: the touched
    /// element's `touch-action` and its non-scrolling ancestors' are
    /// intersected up to but excluding the anchored container. A
    /// `touch-action: none` element inside a `pan-y` scroller fences
    /// the gesture off — the container-only masks consulted before
    /// could not see it — while a plain sibling stays permissive.
    #[test]
    fn touch_action_pregate_intersects_touched_element_and_ancestors() {
        let html = b"<html><body>\
            <div id=\"scroller\"><div id=\"mid\"><div id=\"leaf\">x</div></div>\
            <div id=\"fenced\">y</div><div id=\"plain\">z</div>\
            <div id=\"tall\">w</div></div></body></html>";
        let css = "#scroller { overflow-y: auto; height: 100px; touch-action: pan-y; } \
                   #mid { touch-action: pan-x; } \
                   #fenced { touch-action: none; } \
                   #tall { height: 400px; }";
        let (doc, root) = layout_html(html, css);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let scroller = doc.get_element_by_id("scroller").expect("scroller");
        let anchor = Some(scroller);
        // Sanity: the scroller's own mask is `pan-y` (live-gated
        // elsewhere, so the pre-gate must not include it).
        let isc = tree.index_of(scroller).expect("scroller is a container");
        let pan_y = tree.container(isc).touch_action;
        assert!(pan_y.y_neg && pan_y.y_pos && !pan_y.x_neg && !pan_y.x_pos);
        // `none` on the touched element: the whole gesture is fenced.
        let fenced = doc.get_element_by_id("fenced").expect("fenced");
        assert_eq!(
            touch_action_pregate(&doc, &root, Some(fenced), anchor),
            TouchAction::default(),
            "touch-action: none element fences the gesture"
        );
        // Non-scrolling ancestor's mask intersects the touched
        // element's: `pan-x` on #mid narrows the `auto` leaf to pan-x.
        let leaf = doc.get_element_by_id("leaf").expect("leaf");
        assert_eq!(
            touch_action_pregate(&doc, &root, Some(leaf), anchor),
            TouchAction {
                x_neg: true,
                x_pos: true,
                y_neg: false,
                y_pos: false,
            },
            "non-scrolling ancestor's mask intersects"
        );
        // Plain touched element under the same scroller: permissive.
        let plain = doc.get_element_by_id("plain").expect("plain");
        assert_eq!(
            touch_action_pregate(&doc, &root, Some(plain), anchor),
            TouchAction::all()
        );
        // The anchored container's own element is not part of the
        // pre-gate (its mask stays live): hitting the scroller itself
        // yields the permissive initial, not its `pan-y`.
        assert_eq!(
            touch_action_pregate(&doc, &root, anchor, anchor),
            TouchAction::all()
        );
        // No hit node (the `innermost_at` fallback): permissive.
        assert_eq!(
            touch_action_pregate(&doc, &root, None, anchor),
            TouchAction::all()
        );
    }

    /// Pointer Events 3 §5.2 intersects over the flat tree: a touch
    /// inside a shadow tree still intersects the shadow host's (and
    /// outer ancestors') `touch-action`. The shadow-root fragment is
    /// unlinked, so a plain light-DOM parent walk would stop there and
    /// miss the host's fence.
    #[test]
    fn touch_action_pregate_crosses_shadow_boundary_to_host() {
        let html = b"<html><body><div id=\"host\"></div></body></html>";
        let css = "#host { touch-action: none; }";
        let (mut doc, _) = Document::parse_html(html);
        let host = doc.get_element_by_id("host").expect("host");
        let frag = doc
            .attach_shadow_root(host, ShadowRootMode::Open)
            .expect("shadow root attaches");
        let inner = doc.create_element("div");
        doc.append_child(frag, inner);
        let sheet = parse_stylesheet(css);
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        // The host's `touch-action: none` fences a touch on the shadow
        // child: the walk crossed the fragment boundary to the host
        // instead of stopping permissive at the fragment.
        assert_eq!(
            touch_action_pregate(&doc, &root, Some(inner), None),
            TouchAction::default(),
            "shadow host's touch-action: none fences shadow-descendant touch"
        );
    }

    /// `display: contents` wrappers generate no box, so their computed
    /// `touch-action` is not recoverable by the pre-gate (documented
    /// §5.2 deviation on [`touch_action_pregate`]): a `touch-action:
    /// none` on such a wrapper stays permissive instead of fencing the
    /// gesture. This pins that behavior so a future style-lookup fix
    /// updates it deliberately.
    #[test]
    fn touch_action_pregate_display_contents_wrapper_stays_permissive() {
        let html = b"<html><body>\
            <div id=\"scroller\"><div id=\"wrap\"><div id=\"leaf\">x</div></div>\
            <div id=\"tall\">w</div></div></body></html>";
        let css = "#scroller { overflow-y: auto; height: 100px; } \
                   #wrap { display: contents; touch-action: none; } \
                   #tall { height: 400px; }";
        let (doc, root) = layout_html(html, css);
        let scroller = doc.get_element_by_id("scroller").expect("scroller");
        let leaf = doc.get_element_by_id("leaf").expect("leaf");
        assert_eq!(
            touch_action_pregate(&doc, &root, Some(leaf), Some(scroller)),
            TouchAction::all(),
            "documented deviation: display:contents wrapper's touch-action is not intersected"
        );
    }

    /// The chain publishes what it could not consume: a delta that fits
    /// leaves `(0, 0)`; a delta clamped at the container's max leaves
    /// the signed leftover in `terminal_residual`.
    #[test]
    fn scroll_chain_publishes_terminal_residual() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"t\"><div id=\"tall\">x</div></div></body></html>",
            "#t { overflow-y: auto; height: 100px; } #tall { height: 400px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let t = doc.get_element_by_id("t").expect("t");
        let it = tree.index_of(t).expect("t is a container");
        // Fully consumed: no residual.
        let o = tree.scroll_chain(it, 0.0, 50.0);
        assert!(o.moved());
        assert_eq!(o.terminal_residual, (0.0, 0.0));
        // Park at the max (300 px), then push past it: the clamped
        // leftover is published.
        let o = tree.scroll_chain(it, 0.0, 250.0);
        assert!(o.moved());
        assert_eq!(tree.container(it).state.offset(), (0.0, 300.0));
        let o = tree.scroll_chain(it, 0.0, 100.0);
        assert!(!o.moved(), "at the edge, nothing moves");
        assert_eq!(o.terminal_residual, (0.0, 100.0));
        assert!(!o.edge_hits.is_empty(), "the press still records");
    }

    #[test]
    fn stable_gutter_reserves_identical_space_with_and_without_scrollbar() {
        // Two `overflow-y: auto` boxes, one overflowing (vertical
        // scrollbar shown) and one not (scrollbar hidden). With
        // `scrollbar-gutter: stable` the reserved strip must be
        // identical in both: the content-box width cannot shift when
        // the scrollbar appears or disappears.
        //
        // The property is registered, so the stylesheet declaration
        // flows through the cascade into `recover_parents`, exercising
        // the reservation geometry end to end through a real layout.
        let (doc, root) = layout_html(
            b"<html><body>\
              <div id=\"tall\"><div id=\"tall-inner\">x</div></div>\
              <div id=\"short\"><div id=\"short-inner\">y</div></div>\
              </body></html>",
            "#tall { overflow-y: auto; height: 100px; width: 200px; scrollbar-gutter: stable; } \
             #tall-inner { height: 300px; } \
             #short { overflow-y: auto; height: 100px; width: 200px; scrollbar-gutter: stable; } \
             #short-inner { height: 10px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let tall = doc.get_element_by_id("tall").expect("tall");
        let short = doc.get_element_by_id("short").expect("short");
        let it = tree.index_of(tall).expect("tall is a container");
        let is = tree.index_of(short).expect("short is a container");
        // Premise: the scrollbar is shown for the overflowing box and
        // hidden for the other.
        assert!(tree.scrollbar_geom(it).vertical.is_some());
        assert!(tree.scrollbar_geom(is).vertical.is_none());
        // The cascade resolved `stable` into both containers.
        assert_eq!(
            tree.container(it).gutter,
            ScrollbarGutter::Stable { both_edges: false }
        );
        assert_eq!(
            tree.container(is).gutter,
            ScrollbarGutter::Stable { both_edges: false }
        );
        let reserved_tall = tree.gutter_rects(it);
        let reserved_short = tree.gutter_rects(is);
        assert_eq!(reserved_tall.len(), 1);
        assert_eq!(reserved_short.len(), 1);
        // Same strip position and width, scrollbar or not: zero layout
        // shift. (The y origins differ — the boxes stack vertically —
        // so compare the inline geometry, not whole rects.)
        assert!(approx_eq(reserved_tall[0].x, reserved_short[0].x));
        assert!(approx_eq(reserved_tall[0].w, SCROLLBAR_THICKNESS));
        assert!(approx_eq(reserved_short[0].w, SCROLLBAR_THICKNESS));
        // The strip is the scrollport's inline-end edge — exactly where
        // the shown scrollbar's track paints.
        let port = tree.container(it).scrollport;
        assert!(approx_eq(
            reserved_tall[0].x,
            port.x + port.w - SCROLLBAR_THICKNESS
        ));
        assert_eq!(
            reserved_tall[0],
            tree.scrollbar_geom(it).vertical.expect("shown").track
        );
        // Content-box width available to content is identical both ways.
        let content_w = |idx: usize| {
            tree.container(idx).scrollport.w
                - tree.gutter_rects(idx).iter().map(|r| r.w).sum::<f32>()
        };
        assert!(approx_eq(content_w(it), content_w(is)));
        assert!(approx_eq(content_w(it), 200.0 - SCROLLBAR_THICKNESS));
    }

    #[test]
    fn stable_gutter_both_edges_reserves_mirrored_strip() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"b\"><div id=\"b-inner\">x</div></div></body></html>",
            "#b { overflow-y: auto; height: 100px; width: 200px; } #b-inner { height: 300px; }",
        );
        let (_doc_state, _elements, mut tree) = build_tree(&doc, &root);
        let b = doc.get_element_by_id("b").expect("b");
        let ib = tree.index_of(b).expect("b is a container");
        tree.containers[ib].gutter = ScrollbarGutter::Stable { both_edges: true };
        let strips = tree.gutter_rects(ib);
        assert_eq!(strips.len(), 2, "inline-end plus inline-start strips");
        let port = tree.container(ib).scrollport;
        // Inline-end strip first (right edge in ltr), then the mirrored
        // inline-start strip.
        assert!(approx_eq(
            strips[0].x,
            port.x + port.w - SCROLLBAR_THICKNESS
        ));
        assert!(approx_eq(strips[1].x, port.x));
        for s in &strips {
            assert!(approx_eq(s.w, SCROLLBAR_THICKNESS));
        }
    }

    #[test]
    fn stable_gutter_flips_sides_under_rtl() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"r\"><div id=\"r-inner\">x</div></div></body></html>",
            "#r { overflow-y: auto; height: 100px; width: 200px; direction: rtl; } \
             #r-inner { height: 300px; }",
        );
        let (_doc_state, _elements, mut tree) = build_tree(&doc, &root);
        let r = doc.get_element_by_id("r").expect("r");
        let ir = tree.index_of(r).expect("r is a container");
        assert!(tree.container(ir).rtl);
        tree.containers[ir].gutter = ScrollbarGutter::Stable { both_edges: false };
        let strips = tree.gutter_rects(ir);
        assert_eq!(strips.len(), 1);
        // Under rtl the inline-end edge (and the scrollbar) is the left
        // edge: the reserved strip follows it, like the track does.
        assert!(approx_eq(strips[0].x, tree.container(ir).scrollport.x));
        assert!(approx_eq(strips[0].w, SCROLLBAR_THICKNESS));
        assert_eq!(
            strips[0],
            tree.scrollbar_geom(ir).vertical.expect("shown").track
        );
    }

    #[test]
    fn stable_gutter_applies_to_hidden_overflow() {
        // `overflow: hidden` boxes are scroll containers without user
        // scroll; CSS Overflow 4 still reserves the gutter for them when
        // overflow is not `visible`.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"h\"><div id=\"h-inner\">x</div></div></body></html>",
            "#h { overflow: hidden; height: 100px; width: 200px; } #h-inner { height: 300px; }",
        );
        let (_doc_state, _elements, mut tree) = build_tree(&doc, &root);
        let h = doc.get_element_by_id("h").expect("h");
        let ih = tree.index_of(h).expect("h is a container");
        assert_eq!(tree.container(ih).user_scroll, (false, false));
        assert!(tree.container(ih).axes.1, "hidden is a container axis");
        tree.containers[ih].gutter = ScrollbarGutter::Stable { both_edges: false };
        let strips = tree.gutter_rects(ih);
        assert_eq!(strips.len(), 1);
        assert!(approx_eq(strips[0].w, SCROLLBAR_THICKNESS));
        // ...while no scrollbar is ever shown for it.
        assert!(tree.scrollbar_geom(ih).vertical.is_none());
    }

    #[test]
    fn stable_gutter_skipped_when_block_axis_not_scrollable() {
        // No vertical scrollbar can ever appear (`overflow-y: clip`
        // clips without becoming a scroll container), so no inline-end
        // gutter is reserved even under `stable`.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"c\"><div id=\"c-inner\">x</div></div></body></html>",
            "#c { overflow-x: hidden; overflow-y: clip; height: 100px; width: 200px; } \
             #c-inner { height: 300px; }",
        );
        let (_doc_state, _elements, mut tree) = build_tree(&doc, &root);
        let c = doc.get_element_by_id("c").expect("c");
        let ic = tree.index_of(c).expect("c is a container");
        assert!(!tree.container(ic).axes.1);
        tree.containers[ic].gutter = ScrollbarGutter::Stable { both_edges: false };
        assert!(tree.gutter_rects(ic).is_empty());
    }

    #[test]
    fn scrollbar_degenerate_track_yields_no_bar() {
        // A 10px scrollport is smaller than the 12px scrollbar
        // thickness: with both axes overflowing, each track would
        // have negative length. The geometry reports no bars instead
        // of emitting degenerate rects into the display list.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"tiny\"><div id=\"big\">x</div></div></body></html>",
            "#tiny { overflow: auto; height: 10px; width: 10px; } \
             #big { height: 100px; width: 100px; }",
        );
        let (_, _, tree) = build_tree(&doc, &root);
        let tiny = doc.get_element_by_id("tiny").expect("tiny");
        let it = tree.index_of(tiny).expect("tiny is a container");
        let g = tree.scrollbar_geom(it);
        assert!(
            g.vertical.is_none(),
            "negative-length vertical track is no bar"
        );
        assert!(
            g.horizontal.is_none(),
            "negative-length horizontal track is no bar"
        );
    }

    #[test]
    fn scrollbar_thumb_length_is_visible_fraction() {
        let (doc, _root, tree) = scrollbar_tree();
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        // 100px of 300px visible: the thumb takes a third of the track.
        let v = tree.scrollbar_geom(io).vertical.expect("vertical bar");
        assert!(approx_eq(v.thumb.h, v.track.h * 100.0 / 300.0));
    }

    #[test]
    fn scrollbar_thumb_honors_minimum_size() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"big\"><div id=\"bigin\">x</div></div></body></html>",
            "#big { overflow-y: auto; height: 100px; width: 200px; } #bigin { height: 10000px; }",
        );
        let (_, _, tree) = build_tree(&doc, &root);
        let big = doc.get_element_by_id("big").expect("big");
        let ib = tree.index_of(big).expect("big container");
        let v = tree.scrollbar_geom(ib).vertical.expect("vertical bar");
        // 100px of 10000px would be a 1px thumb: floored at the minimum.
        assert!(approx_eq(v.thumb.h, SCROLLBAR_MIN_THUMB));
        assert!(v.thumb.h <= v.track.h);
    }

    #[test]
    fn scrollbar_thumb_position_tracks_offset() {
        let (doc, _root, tree) = scrollbar_tree();
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        let c = tree.container(io);
        // max is 200 (300px content - 100px scrollport).
        c.state.set_offset(0.0, 0.0);
        let top = tree.scrollbar_geom(io).vertical.expect("bar");
        assert!(approx_eq(top.thumb.y, top.track.y));
        c.state.set_offset(0.0, 200.0);
        let bottom = tree.scrollbar_geom(io).vertical.expect("bar");
        assert!(approx_eq(
            bottom.thumb.y + bottom.thumb.h,
            bottom.track.y + bottom.track.h
        ));
        c.state.set_offset(0.0, 100.0);
        let mid = tree.scrollbar_geom(io).vertical.expect("bar");
        assert!(approx_eq(
            mid.thumb.y,
            mid.track.y + (mid.track.h - mid.thumb.h) / 2.0
        ));
    }

    #[test]
    fn scrollbar_hit_prefers_thumb_over_track() {
        let (doc, _root, tree) = scrollbar_tree();
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        // Viewport offset is zero here: viewport coords == page coords.
        let v = tree.scrollbar_geom(io).vertical.expect("bar");
        let cx = v.thumb.x + v.thumb.w / 2.0;
        // Thumb center: a drag hit.
        let cy = v.thumb.y + v.thumb.h / 2.0;
        let hit = tree.scrollbar_at(cx, cy).expect("thumb hit");
        assert_eq!(hit.container, io);
        assert_eq!(hit.axis, ScrollbarAxis::Vertical);
        assert!(hit.on_thumb);
        // Track below the thumb: a page hit.
        let ty = v.thumb.y + v.thumb.h + 5.0;
        assert!(ty < v.track.y + v.track.h, "test point inside track");
        let hit2 = tree.scrollbar_at(cx, ty).expect("track hit");
        assert_eq!(hit2.container, io);
        assert_eq!(hit2.axis, ScrollbarAxis::Vertical);
        assert!(!hit2.on_thumb);
        // Content area: no scrollbar hit.
        assert!(tree.scrollbar_at(5.0, 5.0).is_none());
    }

    #[test]
    fn scrollbar_drag_maps_cursor_to_offset() {
        let (doc, _root, tree) = scrollbar_tree();
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        let v = tree.scrollbar_geom(io).vertical.expect("bar");
        let cx = v.thumb.x + v.thumb.w / 2.0;
        // Press mid-thumb: the grab is half the thumb.
        let press_y = v.thumb.y + v.thumb.h / 2.0;
        let hit = tree.scrollbar_at(cx, press_y).expect("thumb hit");
        let drag = tree.scrollbar_drag(hit, cx, press_y).expect("drag begins");
        assert_eq!(drag.node, tree.container(io).node);
        assert_eq!(drag.axis, ScrollbarAxis::Vertical);
        // Cursor at track top (+ grab): offset 0.
        assert!(approx_eq(
            drag.offset_for(cx, v.track.y + (press_y - v.thumb.y)),
            0.0
        ));
        // Thumb bottom at track bottom: max (200).
        let end_y = v.track.y + v.track.h - v.thumb.h + (press_y - v.thumb.y);
        assert!(approx_eq(drag.offset_for(cx, end_y), 200.0));
        // Past the ends the drag asks beyond the range; the shared
        // state's clamp (what production writes through) holds it.
        let c = tree.container(io);
        c.state.set_offset(0.0, drag.offset_for(cx, 1e6));
        assert_eq!(c.state.offset().1, 200.0);
        c.state.set_offset(0.0, drag.offset_for(cx, -1e6));
        assert_eq!(c.state.offset().1, 0.0);
        // No jump: pressing at a different thumb point and moving by
        // the same cursor delta moves the offset by the same amount.
        let press_top = v.thumb.y + 2.0;
        let drag2 = tree.scrollbar_drag(hit, cx, press_top).expect("drag2");
        let d1 = drag.offset_for(cx, press_y + 10.0) - drag.offset_for(cx, press_y);
        let d2 = drag2.offset_for(cx, press_top + 10.0) - drag2.offset_for(cx, press_top);
        assert!(d1 > 0.0);
        assert!(approx_eq(d1, d2));
        // A track press never starts a drag.
        let track_hit = tree
            .scrollbar_at(cx, v.thumb.y + v.thumb.h + 5.0)
            .expect("track hit");
        assert!(!track_hit.on_thumb);
        assert!(tree.scrollbar_drag(track_hit, cx, press_y).is_none());
    }

    /// Nested scroll containers for the drag-rebase tests: `#outer`
    /// scrolls vertically (200px scrollport, 500px content) and holds
    /// `#inner`, which scrolls vertically too (100px scrollport, 300px
    /// content).
    fn nested_scrollbar_tree() -> (Document, LayoutBox, ScrollTree) {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"outer\"><div id=\"inner\">\
              <div id=\"tall\">x</div></div><div id=\"pad\">y</div>\
              </div></body></html>",
            "#outer { overflow-y: auto; height: 200px; width: 300px; } \
             #inner { overflow-y: auto; height: 100px; width: 200px; } \
             #tall { height: 300px; } \
             #pad { height: 400px; }",
        );
        let (_, _, tree) = build_tree(&doc, &root);
        (doc, root, tree)
    }

    #[test]
    fn scrollbar_drag_rebases_when_ancestor_scrolls() {
        let (doc, _root, tree) = nested_scrollbar_tree();
        let inner = doc.get_element_by_id("inner").expect("inner");
        let outer = doc.get_element_by_id("outer").expect("outer");
        let ii = tree.index_of(inner).expect("inner container");
        let io = tree.index_of(outer).expect("outer container");
        // Viewport offset is zero: viewport coords == page coords.
        let v = tree.scrollbar_geom(ii).vertical.expect("bar");
        let cx = v.track.x + v.track.w / 2.0;
        let press_y = v.thumb.y + v.thumb.h / 2.0;
        let hit = tree.scrollbar_at(cx, press_y).expect("thumb hit");
        assert!(hit.on_thumb);
        let drag = tree.scrollbar_drag(hit, cx, press_y).expect("drag");
        // No intervening change: the rebase is a no-op.
        let same = tree.scrollbar_drag_rebased(drag).expect("drag alive");
        assert!(approx_eq(same.track_start, drag.track_start));
        assert!(approx_eq(
            same.offset_for(cx, press_y),
            drag.offset_for(cx, press_y)
        ));
        // Scroll the ancestor 60px: the inner frame (and its track)
        // translate rigidly in viewport coords.
        tree.container(io).state.set_offset(0.0, 60.0);
        let live = tree.scrollbar_geom(ii).vertical.expect("bar");
        assert!(approx_eq(live.track.y, v.track.y - 60.0));
        let rebased = tree
            .scrollbar_drag_rebased(drag)
            .expect("pure translation rebases, not cancels");
        // The mapping moved with the track ...
        assert!(approx_eq(rebased.track_start, live.track.y));
        // ... so the thumb stays glued under the stationary cursor
        // instead of jumping 60px with the stale frozen mapping.
        let stale_thumb_y = live.track.y
            + (drag.offset_for(cx, press_y) - drag.min) / (drag.max - drag.min)
                * (drag.track_len - drag.thumb_len);
        let fresh_thumb_y = live.track.y
            + (rebased.offset_for(cx, press_y) - rebased.min) / (rebased.max - rebased.min)
                * (rebased.track_len - rebased.thumb_len);
        assert!(
            (stale_thumb_y - (press_y - drag.grab)).abs() > 30.0,
            "stale mapping would jump the thumb"
        );
        assert!(
            approx_eq(fresh_thumb_y, press_y - rebased.grab),
            "rebased mapping keeps the thumb under the cursor"
        );
    }

    #[test]
    fn scrollbar_drag_cancelled_when_track_resized_or_gone() {
        // One DOM, laid out twice: the drag is captured from the
        // first tree, then rebased against a relaid-out tree.
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"box\"><div id=\"tall\">x</div></div></body></html>",
        );
        let css = |h: &str| {
            parse_stylesheet(&format!(
                "#box {{ overflow-y: auto; height: {h}; width: 200px; }} #tall {{ height: 300px; }}"
            ))
        };
        let relayout = |sheet: &bowser_css::Stylesheet| {
            let root = layout(
                &doc,
                &[sheet],
                Size { w: 800.0, h: 600.0 },
                &FixedFontMetrics,
            );
            let (_, _, tree) = build_tree(&doc, &root);
            (root, tree)
        };
        let sheet_a = css("100px");
        let (_root_a, tree_a) = relayout(&sheet_a);
        let box_ = doc.get_element_by_id("box").expect("box");
        let ib = tree_a.index_of(box_).expect("box container");
        let v = tree_a.scrollbar_geom(ib).vertical.expect("bar");
        let cx = v.track.x + v.track.w / 2.0;
        let press_y = v.thumb.y + v.thumb.h / 2.0;
        let hit = tree_a.scrollbar_at(cx, press_y).expect("thumb hit");
        let drag = tree_a.scrollbar_drag(hit, cx, press_y).expect("drag");
        // Same layout: the gesture survives.
        assert!(tree_a.scrollbar_drag_rebased(drag).is_some());
        // Relayout resizes the scrollport (100px -> 150px): the track
        // resizes, so the gesture ends instead of scrolling through
        // stale geometry.
        let sheet_b = css("150px");
        let (_root_b, tree_b) = relayout(&sheet_b);
        assert!(
            tree_b.scrollbar_drag_rebased(drag).is_none(),
            "resized track ends the gesture"
        );
        // Relayout removes the container entirely: the gesture ends.
        let sheet_c = parse_stylesheet("#box { height: 100px; } #tall { height: 300px; }");
        let (_root_c, tree_c) = relayout(&sheet_c);
        assert!(tree_c.index_of(box_).is_none(), "box is no container now");
        assert!(
            tree_c.scrollbar_drag_rebased(drag).is_none(),
            "vanished container ends the gesture"
        );
    }

    #[test]
    fn scrollbar_drag_suppresses_snap_backstop_until_release() {
        // The plain backstop snap-evaluates a moved container ...
        let (_doc, _root, sc, io, tree) = snap_tree();
        tree.container(io).state.set_offset(0.0, 40.0);
        tree.mark_scrolled(&[Some(sc)]);
        assert!(
            tree.has_smooth_in_flight(),
            "backstop snaps script-style writes"
        );
        // ... but the direct-manipulation path (thumb drag) skips it
        // per-motion and settles once on release.
        let (_doc2, _root2, sc2, io2, tree2) = snap_tree();
        tree2.container(io2).state.set_offset(0.0, 40.0);
        tree2.mark_scrolled_direct(&[Some(sc2)]);
        assert!(
            !tree2.has_smooth_in_flight(),
            "drag motions must not snap mid-gesture"
        );
        // `scrollend` bookkeeping still stamps under the direct path:
        // the container ends on the first quiet tick.
        assert!(tree2.end_tick().is_empty());
        assert_eq!(tree2.end_tick(), vec![Some(sc2)]);
        // Release: one non-relative evaluation settles the gesture —
        // offset 40 snaps back to the y=0 area (mandatory).
        tree2.snap_after_direct_manipulation(Some(sc2));
        assert!(
            tree2.has_smooth_in_flight(),
            "release runs one snap evaluation"
        );
    }

    #[test]
    fn scrollbar_page_target_pages_one_scrollport() {
        let (doc, _root, tree) = scrollbar_tree();
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        let v = tree.scrollbar_geom(io).vertical.expect("bar");
        let cx = v.thumb.x + v.thumb.w / 2.0;
        // Click below the thumb: page down one scrollport (100px).
        let below = v.thumb.y + v.thumb.h + 5.0;
        let hit = tree.scrollbar_at(cx, below).expect("track hit");
        assert!(!hit.on_thumb);
        assert!(approx_eq(
            tree.scrollbar_page_target(hit, cx, below)
                .expect("page target"),
            100.0
        ));
        // From the bottom, a click above the thumb pages back up.
        let c = tree.container(io);
        c.state.set_offset(0.0, 200.0);
        let v2 = tree.scrollbar_geom(io).vertical.expect("bar");
        let above = v2.thumb.y - 5.0;
        let hit2 = tree.scrollbar_at(cx, above).expect("track hit");
        assert!(!hit2.on_thumb);
        assert!(approx_eq(
            tree.scrollbar_page_target(hit2, cx, above)
                .expect("page target"),
            100.0
        ));
        // A thumb press is a drag, not a page.
        let hit3 = tree.scrollbar_at(cx, v2.thumb.y + 2.0).expect("thumb hit");
        assert!(hit3.on_thumb);
        assert!(
            tree.scrollbar_page_target(hit3, cx, v2.thumb.y + 2.0)
                .is_none()
        );
    }

    #[test]
    fn scrollbar_viewport_bar_rides_viewport_offset() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"tallpage\">x</div></body></html>",
            "#tallpage { height: 2000px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        // Publish the viewport range the way `rebuild_after_layout` does.
        let (origin, max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_min_offset(origin);
        doc_state.set_max_offset(max);
        assert_eq!(max, (0.0, 1400.0));
        let v = tree.scrollbar_geom(0).vertical.expect("viewport bar");
        assert!(approx_eq(v.track.x, 800.0 - SCROLLBAR_THICKNESS));
        assert!(approx_eq(v.track.y, 0.0));
        assert!(approx_eq(v.track.h, 600.0));
        // Scrolled: the track rides the offset in page coords, so the
        // backend's scroll subtraction keeps it glued to the viewport
        // edge — and the hit test still finds it there.
        doc_state.set_offset(0.0, 500.0);
        let v2 = tree.scrollbar_geom(0).vertical.expect("viewport bar");
        assert!(approx_eq(v2.track.x, 800.0 - SCROLLBAR_THICKNESS));
        assert!(approx_eq(v2.track.y, 500.0));
        let hit = tree
            .scrollbar_at(800.0 - SCROLLBAR_THICKNESS / 2.0, 300.0)
            .expect("viewport bar hit");
        assert_eq!(hit.container, 0);
        assert_eq!(hit.axis, ScrollbarAxis::Vertical);
    }

    /// Track [`DisplayItem::FillRect`]s in `list`, in order.
    fn fill_rects(list: &DisplayList) -> Vec<(Rect, Color)> {
        list.items
            .iter()
            .filter_map(|item| match *item {
                DisplayItem::FillRect { rect, color } => Some((rect, color)),
                _ => None,
            })
            .collect()
    }

    /// Regression (wave-34 review): a viewport scroll without a
    /// repaint must not leave the baked scrollbar rects stale. This
    /// drives the exact sequence `refresh` runs — record the split,
    /// push the bars, scroll the viewport, re-push through
    /// [`ScrollTree::refresh_epilogue`] — and asserts the
    /// painted track still hugs the viewport edge and agrees with
    /// the hit test.
    #[test]
    fn scrollbar_repush_after_viewport_scroll_keeps_bar_on_edge() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"tallpage\">x</div></body></html>",
            "#tallpage { height: 2000px; }",
        );
        let (doc_state, _elements, mut tree) = build_tree(&doc, &root);
        let (origin, max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_min_offset(origin);
        doc_state.set_max_offset(max);

        // Emulate the full-repaint path: content items first, then the
        // bars appended last with the split recorded.
        let mut list = DisplayList::new();
        let content = Rect {
            x: 0.0,
            y: 0.0,
            w: 800.0,
            h: 2000.0,
        };
        list.push(DisplayItem::FillRect {
            rect: content,
            color: scrollbar_thumb_color(),
        });
        tree.set_glow_split(list.items.len());
        tree.mark_scrollbars_dirty();
        tree.refresh_epilogue(&mut list, Size { w: 800.0, h: 600.0 });
        assert!(!tree.scrollbars_dirty(), "re-push clears the flag");
        // One axis' bar: track + thumb.
        assert_eq!(list.items.len(), 3);
        let stale_track = fill_rects(&list)
            .into_iter()
            .find(|(_, c)| *c == scrollbar_track_color())
            .expect("track pushed")
            .0;
        assert!(approx_eq(stale_track.y, 0.0));

        // Viewport-only scroll: no rebuild, just the stale-bar flag —
        // the GPU-uniform fast path. Without the re-push the baked
        // track would sit at the old page coords while the backend
        // subtracts the new offset (visually sliding off the edge).
        doc_state.set_offset(0.0, 500.0);
        tree.mark_scrollbars_dirty();
        tree.refresh_epilogue(&mut list, Size { w: 800.0, h: 600.0 });

        // Item count is stable: truncate + re-push, never growth.
        assert_eq!(list.items.len(), 3);
        // The content item before the split is untouched.
        let rects = fill_rects(&list);
        assert!(approx_eq(rects[0].0.x, content.x));
        assert!(approx_eq(rects[0].0.h, content.h));
        // The re-pushed track rode the offset in page coords, so the
        // backend's scroll subtraction keeps it glued to the viewport
        // edge — and it moved exactly with the scroll (no stale rect).
        let track = rects
            .iter()
            .find(|(_, c)| *c == scrollbar_track_color())
            .expect("track re-pushed")
            .0;
        assert!(
            approx_eq(track.y, stale_track.y + 500.0),
            "track re-baked at the live offset"
        );
        let (vox, voy) = (0.0, 500.0);
        assert!(approx_eq(track.x - vox, 800.0 - SCROLLBAR_THICKNESS));
        assert!(approx_eq(track.y - voy, 0.0));
        assert!(approx_eq(track.h, 600.0));
        // Paint and hit-test agree: the hit test finds the bar exactly
        // where the painted track sits on screen, and a point just
        // left of it is not a bar hit.
        let hit = tree
            .scrollbar_at(track.x - vox + 2.0, 300.0)
            .expect("painted bar is hit-testable");
        assert_eq!(hit.container, 0);
        assert_eq!(hit.axis, ScrollbarAxis::Vertical);
        assert!(
            tree.scrollbar_at(track.x - vox - 1.0, 300.0).is_none(),
            "content left of the painted bar is not swallowed"
        );
    }

    /// The re-push also covers `position: fixed` containers, whose
    /// frame bakes the viewport offset: after a viewport scroll their
    /// bars must stay glued to the fixed box on screen.
    #[test]
    fn scrollbar_repush_after_viewport_scroll_moves_fixed_bars() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"tallpage\">x</div><div id=\"fx\"><div id=\"fxin\">y</div></div></body></html>",
            "#tallpage { height: 2000px; } \
             #fx { position: fixed; top: 10px; left: 10px; width: 100px; height: 100px; overflow-y: auto; } \
             #fxin { height: 500px; }",
        );
        let (doc_state, _elements, mut tree) = build_tree(&doc, &root);
        let (origin, max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_min_offset(origin);
        doc_state.set_max_offset(max);
        let fx = doc.get_element_by_id("fx").expect("fx");
        let ifx = tree.index_of(fx).expect("fx container");
        assert!(tree.container(ifx).viewport_anchored);

        let mut list = DisplayList::new();
        tree.set_glow_split(list.items.len());
        tree.refresh_epilogue(&mut list, Size { w: 800.0, h: 600.0 });
        // Viewport bar + fixed box's bar, track + thumb each.
        assert_eq!(list.items.len(), 4);

        doc_state.set_offset(0.0, 500.0);
        tree.mark_scrollbars_dirty();
        tree.refresh_epilogue(&mut list, Size { w: 800.0, h: 600.0 });
        assert_eq!(list.items.len(), 4, "truncate + re-push, no growth");

        // The re-pushed items are exactly the live geometry, in tree
        // order: the fixed box's baked track rode the offset, so on
        // screen (minus the live offset) it still sits on the fixed
        // box's inline-end edge.
        let mut items = fill_rects(&list).into_iter();
        for idx in 0..tree.len() {
            let geom = tree.scrollbar_geom(idx);
            for axis in [geom.vertical, geom.horizontal].into_iter().flatten() {
                let (track, _) = items.next().expect("track item");
                let (thumb, _) = items.next().expect("thumb item");
                assert!(approx_eq(track.x, axis.track.x));
                assert!(approx_eq(track.y, axis.track.y));
                assert!(approx_eq(thumb.x, axis.thumb.x));
                assert!(approx_eq(thumb.y, axis.thumb.y));
            }
        }
        let g = tree.scrollbar_geom(ifx).vertical.expect("fx bar");
        let sp = tree.container(ifx).scrollport;
        assert!(approx_eq(g.track.y - 500.0, sp.y));
        assert!(approx_eq(g.track.x, sp.x + sp.w - SCROLLBAR_THICKNESS));
    }

    /// A scrollbar-only refresh before the first paint (split never
    /// recorded) must not truncate content — and still clears the flag.
    #[test]
    fn scrollbar_repush_without_split_is_noop() {
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_doc_state, _elements, mut tree) = build_tree(&doc, &root);
        let mut list = DisplayList::new();
        list.push(DisplayItem::FillRect {
            rect: Rect {
                x: 1.0,
                y: 2.0,
                w: 3.0,
                h: 4.0,
            },
            color: scrollbar_thumb_color(),
        });
        tree.mark_scrollbars_dirty();
        tree.refresh_epilogue(&mut list, Size { w: 800.0, h: 600.0 });
        assert!(!tree.scrollbars_dirty());
        assert_eq!(list.items.len(), 1, "content item survives");
    }

    #[test]
    fn scrollbar_anchored_container_translates_by_viewport_offset() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"tallpage\">x</div><div id=\"fx\"><div id=\"fxin\">y</div></div></body></html>",
            "#tallpage { height: 2000px; } \
             #fx { position: fixed; top: 10px; left: 10px; width: 100px; height: 100px; overflow-y: auto; } \
             #fxin { height: 500px; }",
        );
        let (doc_state, _elements, tree) = build_tree(&doc, &root);
        let (origin, max) = tree.document_max_scroll(&root, Size { w: 800.0, h: 600.0 });
        doc_state.set_min_offset(origin);
        doc_state.set_max_offset(max);
        let fx = doc.get_element_by_id("fx").expect("fx");
        let ifx = tree.index_of(fx).expect("fx container");
        assert!(tree.container(ifx).viewport_anchored);
        doc_state.set_offset(0.0, 500.0);
        // The scrollport is stored in viewport coordinates; painting
        // translates it by the live viewport offset into page coords.
        let sp = tree.container(ifx).scrollport;
        let g = tree.scrollbar_geom(ifx).vertical.expect("fx bar");
        assert!(approx_eq(g.track.x, sp.x + sp.w - SCROLLBAR_THICKNESS));
        assert!(approx_eq(g.track.y, sp.y + 500.0));
    }

    #[test]
    fn scrollbar_frame_subtracts_scrolled_ancestor_offsets() {
        // A scroller inside a scroller: the inner bar paints at the
        // inner box's painted position (layout position minus the
        // outer container's offset), matching the baked paint tree.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"outer\"><div id=\"pad\"></div><div id=\"inner\"><div id=\"tall\">x</div></div></div></body></html>",
            "#outer { overflow-y: auto; height: 200px; width: 300px; } \
             #pad { height: 400px; } \
             #inner { overflow-y: auto; height: 100px; width: 200px; } \
             #tall { height: 300px; }",
        );
        let (_, _, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let inner = doc.get_element_by_id("inner").expect("inner");
        let io = tree.index_of(outer).expect("outer container");
        let ii = tree.index_of(inner).expect("inner container");
        assert_eq!(tree.parent(ii), io);
        // Scroll the outer container: the inner box paints 80px higher.
        tree.container(io).state.set_offset(0.0, 80.0);
        let paint = tree.bake_paint_tree(&root).into_owned();
        let paint_inner = find_box(&paint, inner).expect("inner in paint tree");
        let inner_sp = tree.container(ii).scrollport;
        assert!(approx_eq(paint_inner.rect.y, inner_sp.y - 80.0));
        let g = tree.scrollbar_geom(ii).vertical.expect("inner bar");
        assert!(approx_eq(g.track.y, inner_sp.y - 80.0));
        assert!(approx_eq(
            g.track.x,
            inner_sp.x + inner_sp.w - SCROLLBAR_THICKNESS
        ));
        // The hit test agrees with the painted position: below the
        // thumb (offset 0), inside the shifted track.
        let v = g;
        let below = v.track.y + v.thumb.h + 5.0;
        let hit = tree
            .scrollbar_at(v.track.x + SCROLLBAR_THICKNESS / 2.0, below)
            .expect("inner track hit");
        assert_eq!(hit.container, ii);
        assert!(!hit.on_thumb);
    }

    #[test]
    fn scrollbar_pointer_drag_drives_offsets() {
        // The pointer-event pipeline `window.rs` drives: hit-test the
        // press point, capture the drag, map motion to offsets, fold
        // the movement into the paint tree without a relayout.
        let (doc, root) = layout_html(SCROLL_HTML, SCROLL_CSS);
        let (_, _, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        let mut paint = tree.bake_paint_tree(&root).into_owned();

        // pointerdown on the thumb (viewport offset is zero: viewport
        // coords == page coords here).
        let g = tree.scrollbar_geom(io).vertical.expect("vertical bar");
        let cx = g.thumb.x + g.thumb.w / 2.0;
        let press_y = g.thumb.y + g.thumb.h / 2.0;
        let hit = tree.scrollbar_at(cx, press_y).expect("thumb hit");
        assert!(hit.on_thumb);
        let drag = tree.scrollbar_drag(hit, cx, press_y).expect("drag");

        // pointermove: drag the thumb to the bottom of the track.
        let target = drag.offset_for(
            cx,
            g.track.y + g.track.h - g.thumb.h + (press_y - g.thumb.y),
        );
        let c = tree.container(io);
        let (ox, _) = c.state.offset();
        c.state.set_offset(ox, target);
        assert!(approx_eq(c.state.offset().1, 200.0));

        // pointerup: the movement folds into the paint tree, no relayout.
        assert_eq!(tree.sync_paint_offsets(&mut paint), vec![Some(outer)]);
        assert!(tree.sync_paint_offsets(&mut paint).is_empty());
    }

    /// Build a display list from a paint root, with dummy font/image
    /// resolvers (tests use no real text shaping or images).
    fn display_list_of(paint: &LayoutBox, doc: &Document) -> DisplayList {
        use bowser_paint::display_list::{FontKey, build_display_list};
        build_display_list(
            paint,
            doc,
            &mut |_| FontKey(0),
            &mut |_| None,
            Size { w: 800.0, h: 600.0 },
        )
    }

    /// Pair `PushTransform`/`PopTransform` items by stack order.
    fn transform_ranges(list: &DisplayList) -> Vec<(usize, usize)> {
        let mut stack = Vec::new();
        let mut out = Vec::new();
        for (i, item) in list.items.iter().enumerate() {
            match item {
                DisplayItem::PushTransform { .. } => stack.push(i),
                DisplayItem::PopTransform => {
                    if let Some(p) = stack.pop() {
                        out.push((p, i));
                    }
                }
                _ => {}
            }
        }
        out
    }

    /// Index of the background `FillRect` painted at `rect`, if any.
    fn fill_rect_at(list: &DisplayList, rect: Rect) -> Option<usize> {
        list.items.iter().position(|item| match item {
            DisplayItem::FillRect { rect: r, .. } => *r == rect,
            _ => false,
        })
    }

    /// Matrix of the `PushTransform` at `idx`.
    fn transform_matrix_at(list: &DisplayList, idx: usize) -> [f32; 6] {
        match list.items[idx] {
            DisplayItem::PushTransform { matrix } => matrix,
            _ => panic!("item {idx} is not a PushTransform"),
        }
    }

    /// A scroll container with an absolutely-positioned child (a
    /// dropdown) stays on the transform fast path: the child shifts
    /// with the scroll but drains inside the container's clip range,
    /// so the container's own wrap covers it.
    #[test]
    fn fast_path_keeps_abspos_child() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div><div id=\"dd\">menu</div></div></body></html>",
            "#sc { position: relative; overflow-y: auto; height: 100px; width: 200px; } \
             #tall { height: 300px; } \
             #dd { position: absolute; top: 10px; left: 5px; width: 50px; height: 20px; background: red; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let dd = doc.get_element_by_id("dd").expect("dd");
        let ci = tree.index_of(sc).expect("sc container");
        let paint = tree.bake_paint_tree(&root).into_owned();
        // `auto` z-index: certain in-range, not even recorded.
        assert!(tree.container(ci).transform_fast.get());
        assert!(tree.container(ci).oof_wraps.borrow().is_empty());
        let mut list = display_list_of(&paint, &doc);
        tree.wrap_scroll_transforms(&paint, &mut list);
        let wrap_at = tree.container(ci).wrap_item.get().expect("wrapped");
        // Scroll: the fast path must take it (no rebuild).
        tree.container(ci).state.scroll_by(0.0, 50.0);
        let moved = tree.try_fast_fold(&mut list);
        assert!(moved.is_some(), "abspos child stays on the fast path");
        assert_eq!(moved.expect("folded"), vec![Some(sc)]);
        // The wrap carries the scroll translation...
        let m = transform_matrix_at(&list, wrap_at);
        assert!(approx_eq(m[4], 0.0));
        assert!(approx_eq(m[5], -50.0));
        // ...and the dropdown's items sit inside that wrap's range, so
        // they ride the translation.
        let dd_rect = find_box(&paint, dd).expect("dd box").rect;
        let dd_item = fill_rect_at(&list, dd_rect).expect("dropdown fill");
        let (_, wrap_pop) = transform_ranges(&list)
            .into_iter()
            .find(|&(p, _)| p == wrap_at)
            .expect("wrap range");
        assert!(
            dd_item > wrap_at && dd_item < wrap_pop,
            "dropdown items inside the scroll transform"
        );
    }

    /// A scroll container with a viewport-anchored fixed overlay child
    /// stays on the transform fast path: the overlay shifts with no
    /// scroller, so it needs no wrap and disqualifies nothing.
    #[test]
    fn fast_path_keeps_fixed_overlay() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div><div id=\"ov\">overlay</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; width: 200px; } \
             #tall { height: 300px; } \
             #ov { position: fixed; top: 10px; left: 5px; width: 50px; height: 20px; background: blue; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let ov = doc.get_element_by_id("ov").expect("ov");
        let ci = tree.index_of(sc).expect("sc container");
        let paint = tree.bake_paint_tree(&root).into_owned();
        assert!(tree.container(ci).transform_fast.get());
        assert!(tree.container(ci).oof_wraps.borrow().is_empty());
        let mut list = display_list_of(&paint, &doc);
        tree.wrap_scroll_transforms(&paint, &mut list);
        let wrap_at = tree.container(ci).wrap_item.get().expect("wrapped");
        tree.container(ci).state.scroll_by(0.0, 50.0);
        let moved = tree.try_fast_fold(&mut list);
        assert!(moved.is_some(), "fixed overlay stays on the fast path");
        assert_eq!(moved.expect("folded"), vec![Some(sc)]);
        // The overlay drains at the root, outside the container's wrap:
        // it must not move with the scroll.
        let ov_rect = find_box(&paint, ov).expect("ov box").rect;
        let ov_item = fill_rect_at(&list, ov_rect).expect("overlay fill");
        let (_, wrap_pop) = transform_ranges(&list)
            .into_iter()
            .find(|&(p, _)| p == wrap_at)
            .expect("wrap range");
        assert!(
            ov_item < wrap_at || ov_item > wrap_pop,
            "fixed overlay outside the scroll transform"
        );
    }

    /// A z-indexed abspos child hoisted above the scroller (by an
    /// ancestor stacking context) gets its own transform pair: the
    /// container stays on the fast path and the tick patches both
    /// matrices.
    #[test]
    fn fast_path_extra_wrap_for_hoisted_abspos() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"outer\"><div id=\"sc\"><div id=\"tall\">x</div><div id=\"dd\">menu</div></div></div></body></html>",
            "#outer { transform: translate(0px, 0px); } \
             #sc { position: relative; overflow-y: auto; height: 100px; width: 200px; } \
             #tall { height: 300px; } \
             #dd { position: absolute; z-index: 5; top: 10px; left: 5px; width: 50px; height: 20px; background: red; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let dd = doc.get_element_by_id("dd").expect("dd");
        let ci = tree.index_of(sc).expect("sc container");
        let paint = tree.bake_paint_tree(&root).into_owned();
        // Recorded: z-indexed, shifts with the scroller.
        assert_eq!(tree.container(ci).oof_wraps.borrow().len(), 1);
        assert!(tree.container(ci).transform_fast.get());
        let mut list = display_list_of(&paint, &doc);
        tree.wrap_scroll_transforms(&paint, &mut list);
        assert!(tree.container(ci).wrap_item.get().is_some());
        let oof_at = tree.container(ci).oof_wrap_items.borrow()[0];
        // The dropdown's items sit inside the extra wrap's range but
        // outside the container's own wrap range.
        let dd_rect = find_box(&paint, dd).expect("dd box").rect;
        let dd_item = fill_rect_at(&list, dd_rect).expect("dropdown fill");
        let ranges = transform_ranges(&list);
        let (_, oof_pop) = ranges
            .iter()
            .find(|&&(p, _)| p == oof_at)
            .expect("oof range");
        assert!(
            dd_item > oof_at && dd_item < *oof_pop,
            "dropdown items inside the extra wrap"
        );
        let main_at = tree.container(ci).wrap_item.get().expect("wrapped");
        let (_, main_pop) = ranges
            .iter()
            .find(|&&(p, _)| p == main_at)
            .expect("main range");
        assert!(
            dd_item < main_at || dd_item > *main_pop,
            "dropdown items outside the main wrap (out of range)"
        );
        // Scroll: the fast path patches both matrices.
        tree.container(ci).state.scroll_by(0.0, 50.0);
        let moved = tree.try_fast_fold(&mut list);
        assert!(moved.is_some(), "hoisted abspos stays on the fast path");
        assert_eq!(moved.expect("folded"), vec![Some(sc)]);
        for at in [main_at, oof_at] {
            let m = transform_matrix_at(&list, at);
            assert!(approx_eq(m[4], 0.0), "no x drift");
            assert!(approx_eq(m[5], -50.0), "both wraps track the scroll");
        }
        // Behavior equivalence: the fast-folded dropdown renders where
        // a full rebuild at the new offset puts it.
        let fresh_paint = tree.bake_paint_tree(&root).into_owned();
        let fresh = display_list_of(&fresh_paint, &doc);
        let dd_rect2 = find_box(&fresh_paint, dd).expect("dd box").rect;
        let dd_slow = fill_rect_at(&fresh, dd_rect2).expect("dropdown fill, rebuilt");
        let slow_rect = match fresh.items[dd_slow] {
            DisplayItem::FillRect { rect, .. } => rect,
            _ => unreachable!(),
        };
        let m = transform_matrix_at(&list, oof_at);
        assert!(approx_eq(dd_rect.y + m[5], slow_rect.y));
        assert!(approx_eq(dd_rect.x + m[4], slow_rect.x));
    }

    /// A fixed box under a fixed-CB trigger that is not a stacking
    /// context (`will-change`) leaves no locatable wrapper: the
    /// container keeps the safe full-rebuild path rather than going
    /// stale. (The trigger must sit strictly below the scroller: a
    /// trigger on the scroller itself does not start the fixed-CB flag
    /// for the boxes beneath it.)
    #[test]
    fn fast_path_falls_back_for_unlocatable_fixed() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div><div id=\"trig\"><div id=\"ov\">overlay</div></div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; width: 200px; } \
             #tall { height: 300px; } \
             #trig { will-change: transform; } \
             #ov { position: fixed; top: 10px; left: 5px; width: 50px; height: 20px; background: blue; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let ci = tree.index_of(sc).expect("sc container");
        let paint = tree.bake_paint_tree(&root).into_owned();
        // Recorded (shifts with the scroller via the trigger)...
        assert_eq!(tree.container(ci).oof_wraps.borrow().len(), 1);
        let mut list = display_list_of(&paint, &doc);
        tree.wrap_scroll_transforms(&paint, &mut list);
        // ...but disqualified: no wrapper pair, items out of range.
        assert!(!tree.container(ci).transform_fast.get());
        tree.container(ci).state.scroll_by(0.0, 50.0);
        assert!(
            tree.try_fast_fold(&mut list).is_none(),
            "unlocatable fixed box keeps the rebuild path"
        );
    }

    /// A sticky box in a nested scroller stays on the fast path when
    /// its shift is exactly foldable: the wrap arms a dedicated pair
    /// for the box's items, and the tick folds `sticky_shifts_at`'s
    /// extra shift on top of the scroll transform instead of re-baking.
    /// Baked stuck on the plateau (like the sibling's plateau test),
    /// then ticked further along it.
    #[test]
    fn fast_path_folds_sticky_shift() {
        let (doc, root) =
            layout_html_nested_at(STICKY_NESTED_HTML, STICKY_NESTED_CSS, "outer", 0.0, 150.0);
        let (_doc_state, _elements, tree) = build_tree_nested_at(&doc, &root, "outer", 0.0, 150.0);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let st = doc.get_element_by_id("st").expect("st");
        let io = tree.index_of(outer).expect("outer container");
        let paint = tree.bake_paint_tree(&root).into_owned();
        assert!(tree.container(io).transform_fast.get());
        let mut list = display_list_of(&paint, &doc);
        tree.wrap_scroll_transforms(&paint, &mut list);
        // Dedicated sticky pair armed for the box's items.
        assert_eq!(tree.container(io).sticky_wrap_items.borrow().len(), 1);
        let (node, st_at) = tree.container(io).sticky_wrap_items.borrow()[0];
        assert_eq!(node, st);
        // The sticky pair nests inside the container's own wrap.
        let main_at = tree.container(io).wrap_item.get().expect("wrapped");
        let ranges = transform_ranges(&list);
        let (_, main_pop) = ranges
            .iter()
            .find(|&&(p, _)| p == main_at)
            .expect("main range");
        let (_, st_pop) = ranges
            .iter()
            .find(|&&(p, _)| p == st_at)
            .expect("sticky range");
        assert!(
            st_at > main_at && *st_pop < *main_pop,
            "sticky pair nests inside the main wrap"
        );
        // Tick along the plateau: exactly foldable, shift is the scroll
        // delta.
        tree.container(io).state.scroll_by(0.0, 50.0);
        let shifts = tree.sticky_shifts_at(io, 0.0, 200.0).expect("foldable");
        assert_eq!(shifts.len(), 1);
        assert_eq!(shifts[0].node, st);
        assert!(approx_eq(shifts[0].dy, 50.0));
        let moved = tree.try_fast_fold(&mut list);
        assert!(moved.is_some(), "sticky fold stays on the fast path");
        assert_eq!(moved.expect("folded"), vec![Some(outer)]);
        // The sticky pair carries the extra shift; the main wrap carries
        // the scroll translation.
        let m = transform_matrix_at(&list, st_at);
        assert!(approx_eq(m[4], 0.0));
        assert!(approx_eq(m[5], 50.0));
        let mm = transform_matrix_at(&list, main_at);
        assert!(approx_eq(mm[4], 0.0));
        assert!(approx_eq(mm[5], -50.0));
        // Behavior: the box is stuck, so the scroll translation and the
        // sticky extra shift cancel — it renders at the scrollport top,
        // exactly where the bake put it.
        assert!(approx_eq(mm[5] + m[5], 0.0));
    }

    /// A sticky box in a constraint transition (not exactly foldable)
    /// keeps the safe re-bake path: `try_fast_fold` returns `None`.
    /// Baked stuck at 150, scrolled back to 50 (unstick transition).
    #[test]
    fn fast_path_sticky_transition_rebakes() {
        let (doc, root) =
            layout_html_nested_at(STICKY_NESTED_HTML, STICKY_NESTED_CSS, "outer", 0.0, 150.0);
        let (_doc_state, _elements, tree) = build_tree_nested_at(&doc, &root, "outer", 0.0, 150.0);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        let paint = tree.bake_paint_tree(&root).into_owned();
        let mut list = display_list_of(&paint, &doc);
        tree.wrap_scroll_transforms(&paint, &mut list);
        tree.container(io).state.set_offset(0.0, 50.0);
        assert!(
            tree.sticky_shifts_at(io, 0.0, 50.0).is_none(),
            "unstick transition is not foldable"
        );
        assert!(
            tree.try_fast_fold(&mut list).is_none(),
            "transition keeps the re-bake path"
        );
    }

    /// First text fragment rect under the element's box (test helper).
    fn first_frag_rect(paint: &LayoutBox, node: NodeId) -> Rect {
        let b = find_box(paint, node).expect("box");
        let mut stack = vec![b];
        while let Some(x) = stack.pop() {
            if x.box_type == BoxType::Text
                && let Some(f) = x.fragments.first()
            {
                return f.rect;
            }
            stack.extend(x.children.iter());
        }
        panic!("no text fragment");
    }

    /// Scrolling back to the bake offset resets the sticky pair to
    /// identity: `sticky_shifts_at` omits zero-shift boxes, but the pair
    /// must not keep its previous tick's matrix (stale-shift
    /// corruption).
    #[test]
    fn fast_fold_resets_sticky_pair_when_shift_returns_to_zero() {
        let (doc, root) =
            layout_html_nested_at(STICKY_NESTED_HTML, STICKY_NESTED_CSS, "outer", 0.0, 150.0);
        let (_doc_state, _elements, tree) = build_tree_nested_at(&doc, &root, "outer", 0.0, 150.0);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        let paint = tree.bake_paint_tree(&root).into_owned();
        let mut list = display_list_of(&paint, &doc);
        tree.wrap_scroll_transforms(&paint, &mut list);
        let st_at = tree.container(io).sticky_wrap_items.borrow()[0].1;
        // Tick forward: the sticky shift is applied to the pair.
        tree.container(io).state.scroll_by(0.0, 50.0);
        assert!(tree.try_fast_fold(&mut list).is_some());
        assert!(approx_eq(transform_matrix_at(&list, st_at)[5], 50.0));
        // Tick back to the bake offset: the shift returns to zero and
        // the pair must be identity again, not the stale +50.
        tree.container(io).state.set_offset(0.0, 150.0);
        assert!(tree.try_fast_fold(&mut list).is_some());
        assert_eq!(
            transform_matrix_at(&list, st_at),
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "stale sticky matrix must reset to identity"
        );
    }

    #[test]
    fn fast_fold_viewport_only_page_reports_and_rearms() {
        // Viewport-only page (the most common shape) with a display
        // list present: the fold has no wraps to patch, but a
        // script-style offset write must still report `Some(vec![None])`
        // and advance the baselines — the windowed pump relies on the
        // fold, not the pre-first-layout fallback, once a paint tree
        // exists.
        let (doc, root) = layout_html(
            b"<html><body><div id=\"tall\">x</div></body></html>",
            "#tall { height: 1200px; }",
        );
        let mut tab = Tab::new(Size { w: 800.0, h: 600.0 });
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        assert!(tab.scroll_tree.is_viewport_only(), "viewport-only page");
        let mut list = {
            let doc = tab.document_ref().expect("installed document");
            let paint = tab.paint_root.as_ref().expect("paint root");
            display_list_of(paint, &doc)
        };
        // Script-style write: the bindings mutate the shared scroll
        // state directly, behind the tree's back.
        tab.scroll().set_offset(0.0, 100.0);
        assert_eq!(
            tab.render.scroll_tree.try_fast_fold(&mut list),
            Some(vec![None]),
            "viewport drift reports exactly once"
        );
        // The fold re-armed the baseline: a second fold is a no-op, so
        // the same movement can never report twice.
        assert_eq!(
            tab.render.scroll_tree.try_fast_fold(&mut list),
            Some(Vec::new()),
            "re-armed fold stays quiet"
        );
    }

    /// Two boxes with identical text: only the sticky box's run is
    /// wrapped. The old content heuristic matched any identical text
    /// inside the box's rect, so an overlapping absolutely-positioned
    /// sibling's run got wrapped and shifted (visual corruption).
    #[test]
    fn sticky_wrap_isolates_only_the_sticky_text_run() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"st\">Hello</div>\
              <div id=\"ov\">Hello</div><div id=\"tall\"></div></div></body></html>",
            "#sc { position: relative; overflow-y: auto; height: 100px; } \
             #st { position: sticky; top: 0; height: 60px; background: yellow; } \
             #ov { position: absolute; top: 35px; left: 0; } \
             #tall { height: 300px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let st = doc.get_element_by_id("st").expect("st");
        let ov = doc.get_element_by_id("ov").expect("ov");
        let io = tree.index_of(sc).expect("sc container");
        let paint = tree.bake_paint_tree(&root).into_owned();
        let mut list = display_list_of(&paint, &doc);
        tree.wrap_scroll_transforms(&paint, &mut list);
        // Exactly one sticky pair armed, for #st.
        assert_eq!(tree.container(io).sticky_wrap_items.borrow().len(), 1);
        let (node, at) = tree.container(io).sticky_wrap_items.borrow()[0];
        assert_eq!(node, st);
        let (_, pop) = transform_ranges(&list)
            .into_iter()
            .find(|&(p, _)| p == at)
            .expect("sticky range");
        // The pair's range holds #st's run and not #ov's identical run.
        let st_frag = first_frag_rect(&paint, st);
        let ov_frag = first_frag_rect(&paint, ov);
        assert_ne!(st_frag, ov_frag, "runs at distinct rects");
        let hello_runs: Vec<Rect> = list.items[at..=pop]
            .iter()
            .filter_map(|item| match item {
                DisplayItem::TextRun {
                    text,
                    text_range,
                    rect,
                    ..
                } if text.get(text_range.clone()) == Some("Hello") => Some(*rect),
                _ => None,
            })
            .collect();
        assert_eq!(
            hello_runs,
            vec![st_frag],
            "only the sticky box's run is wrapped"
        );
    }

    /// A decorated sticky box is fully covered: text-decoration and
    /// text-shadow items belong to the box and must sit inside its
    /// pair's range. The old rect/text heuristic truncated the range
    /// before them, leaving them unshifted while the text moved.
    #[test]
    fn sticky_wrap_covers_decoration_and_text_shadow() {
        let (doc, root) = layout_html(
            b"<html><body><div id=\"sc\"><div id=\"st\">Hello</div>\
              <div id=\"tall\"></div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; } \
             #st { position: sticky; top: 0; text-decoration: underline; \
                   text-shadow: 2px 2px 0 red; } \
             #tall { height: 300px; }",
        );
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let sc = doc.get_element_by_id("sc").expect("sc");
        let st = doc.get_element_by_id("st").expect("st");
        let io = tree.index_of(sc).expect("sc container");
        let paint = tree.bake_paint_tree(&root).into_owned();
        let mut list = display_list_of(&paint, &doc);
        tree.wrap_scroll_transforms(&paint, &mut list);
        assert_eq!(tree.container(io).sticky_wrap_items.borrow().len(), 1);
        let (node, at) = tree.container(io).sticky_wrap_items.borrow()[0];
        assert_eq!(node, st);
        let (_, pop) = transform_ranges(&list)
            .into_iter()
            .find(|&(p, _)| p == at)
            .expect("sticky range");
        let st_frag = first_frag_rect(&paint, st);
        let mut saw_run = false;
        let mut saw_decoration = false;
        let mut saw_shadow = false;
        for item in &list.items[at..=pop] {
            match item {
                DisplayItem::TextRun { rect, .. } if *rect == st_frag => saw_run = true,
                DisplayItem::Decoration { rect, .. } if *rect == st_frag => saw_decoration = true,
                DisplayItem::TextShadow { rect, .. } if *rect == st_frag => saw_shadow = true,
                _ => {}
            }
        }
        assert!(saw_run, "text run is wrapped");
        assert!(saw_decoration, "text-decoration is wrapped");
        assert!(saw_shadow, "text-shadow is wrapped");
    }

    /// A sticky box with nonzero used margins, or a scroller with
    /// nonzero borders, is not foldable: layout sticks the margin box
    /// against the padding box, but the fold models border-box positions
    /// in the border-box frame. Both fall back to re-bake.
    #[test]
    fn sticky_fold_bails_on_margins_and_scroller_borders() {
        // Margined sticky box: baked stuck at 150, ticking to 200 would
        // fold on the plateau — but the margin makes it unsound.
        let css = "#outer { overflow-y: auto; height: 400px; } \
            #pad { height: 100px; } \
            #st { position: sticky; top: 0; height: 50px; margin-top: 10px; } \
            #tall { height: 2000px; }";
        let (doc, root) = layout_html_nested_at(STICKY_NESTED_HTML, css, "outer", 0.0, 150.0);
        let (_doc_state, _elements, tree) = build_tree_nested_at(&doc, &root, "outer", 0.0, 150.0);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        let band = tree
            .sticky_bands
            .iter()
            .find(|b| b.scroller == io)
            .expect("band for #st");
        assert!(band.margins_unsound(), "nonzero margin is unsound");
        assert!(
            tree.sticky_shifts_at(io, 0.0, 200.0).is_none(),
            "margined box is not foldable"
        );
        // Bordered scroller: same fallback.
        let css = "#outer { overflow-y: auto; height: 400px; border: 5px solid black; } \
            #pad { height: 100px; } \
            #st { position: sticky; top: 0; height: 50px; } \
            #tall { height: 2000px; }";
        let (doc, root) = layout_html_nested_at(STICKY_NESTED_HTML, css, "outer", 0.0, 150.0);
        let (_doc_state, _elements, tree) = build_tree_nested_at(&doc, &root, "outer", 0.0, 150.0);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        let band = tree
            .sticky_bands
            .iter()
            .find(|b| b.scroller == io)
            .expect("band for #st");
        assert!(
            band.scroller_borders_unsound(),
            "nonzero scroller border is unsound"
        );
        assert!(
            tree.sticky_shifts_at(io, 0.0, 200.0).is_none(),
            "bordered scroller is not foldable"
        );
    }

    /// Near-stuck folding in a near-padded scroller (w79fix): layout's
    /// near pin is the content origin plus the offset, so a scroller
    /// with 20px top padding sticks a `top: 0` box at offset + 20, not
    /// at the offset. Pre-fix the fold's near detection (`pos ~=
    /// bake + t`) missed the baked stuck box by exactly the padding
    /// and the box fell through to the free interval, which folded
    /// zero shift past the near transition. Geometry: content origin
    /// 20, `#st` natural at 120, stuck from 100; baked stuck at 150,
    /// so the baked border-box-frame position is 170.
    #[test]
    fn sticky_fold_near_stuck_with_scroller_padding() {
        let css = "#outer { overflow-y: auto; height: 400px; padding-top: 20px; } \
            #pad { height: 100px; } \
            #st { position: sticky; top: 0; height: 50px; } \
            #tall { height: 2000px; }";
        let (doc, root) = layout_html_nested_at(STICKY_NESTED_HTML, css, "outer", 0.0, 150.0);
        let (_doc_state, _elements, tree) = build_tree_nested_at(&doc, &root, "outer", 0.0, 150.0);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let st = doc.get_element_by_id("st").expect("st");
        let io = tree.index_of(outer).expect("outer container");
        let sp_y = tree.container(io).scrollport.y;
        let band = tree
            .sticky_bands
            .iter()
            .find(|b| b.scroller == io)
            .expect("band for #st");
        assert!(approx_eq(band.pos_y, 170.0), "pos_y={}", band.pos_y);
        assert!(approx_eq(band.bake_y, 150.0), "bake_y={}", band.bake_y);
        assert_eq!(band.scroller_padding, Some((20.0, 0.0, 0.0, 0.0)));
        // Tick further along the plateau: exactly foldable, shift is
        // the scroll delta (pre-fix this re-baked: no stuck match).
        let shifts = tree.sticky_shifts_at(io, 0.0, 200.0).expect("foldable");
        assert_eq!(shifts.len(), 1, "exactly the sticky box");
        assert_eq!(shifts[0].node, st);
        assert!(approx_eq(shifts[0].dy, 50.0));
        // Behavior equivalence: independent layout at 200 agrees the
        // box is stuck at the content origin plus the offset.
        let (ref_doc, ref_root) =
            layout_html_nested_at(STICKY_NESTED_HTML, css, "outer", 0.0, 200.0);
        let ref_st = ref_doc.get_element_by_id("st").expect("ref st");
        let ref_box = find_box(&ref_root, ref_st).expect("#st box");
        assert!(
            approx_eq(ref_box.rect.y - sp_y, 220.0),
            "layout agrees the box is stuck at 220, got {}",
            ref_box.rect.y - sp_y
        );
        // Scrolling back from the plateau is the conservative re-bake
        // path, not a zero shift: pre-fix the free interval folded
        // `Some(0)` here while the box was still stuck.
        assert!(tree.sticky_shifts_at(io, 0.0, 120.0).is_none());
        // Free-baked twin: baked at 0 with the box still free
        // (natural 120), the near transition is at 100, not 120.
        let (doc, root) = layout_html(STICKY_NESTED_HTML, css);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        assert!(!tree.sticky_constrained_at(io, 0.0, 50.0));
        assert!(
            tree.sticky_shifts_at(io, 0.0, 50.0)
                .expect("foldable")
                .is_empty(),
            "free box needs no extra shift"
        );
        assert!(
            tree.sticky_shifts_at(io, 0.0, 110.0).is_none(),
            "past the padded near transition must re-bake, not fold zero"
        );
    }

    /// Far-stuck folding in a near-padded scroller with no far padding
    /// (w79fix): layout's far pin is `near_pad + bake + sp_pad - far
    /// - size`. Pre-fix the fold's far detection missed the baked
    /// stuck box by the near padding and the box fell through to the
    /// free interval, which folded zero shift on forward ticks while
    /// the box was still stuck. Geometry: content origin 20,
    /// padding-box height 220, `#st` natural at 1020, far pin at bake
    ///   500 is 690; ticking back rides the far edge.
    #[test]
    fn sticky_fold_far_stuck_with_scroller_padding() {
        let css = "#outer { overflow-y: auto; height: 200px; padding-top: 20px; } \
            #pad { height: 1000px; } \
            #st { position: sticky; bottom: 0; height: 50px; } \
            #tall { height: 2000px; }";
        let (doc, root) = layout_html_nested_at(STICKY_NESTED_HTML, css, "outer", 0.0, 500.0);
        let (_doc_state, _elements, tree) = build_tree_nested_at(&doc, &root, "outer", 0.0, 500.0);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let st = doc.get_element_by_id("st").expect("st");
        let io = tree.index_of(outer).expect("outer container");
        let sp_y = tree.container(io).scrollport.y;
        let band = tree
            .sticky_bands
            .iter()
            .find(|b| b.scroller == io)
            .expect("band for #st");
        assert!(approx_eq(band.pos_y, 690.0), "pos_y={}", band.pos_y);
        assert!(approx_eq(band.bake_y, 500.0), "bake_y={}", band.bake_y);
        assert_eq!(band.scroller_padding, Some((20.0, 0.0, 0.0, 0.0)));
        // Ticking back rides the far edge: shift is the scroll delta
        // (pre-fix this re-baked: no far-stuck match).
        let shifts = tree.sticky_shifts_at(io, 0.0, 450.0).expect("foldable");
        assert_eq!(shifts.len(), 1, "exactly the sticky box");
        assert_eq!(shifts[0].node, st);
        assert!(approx_eq(shifts[0].dy, -50.0));
        // Behavior equivalence: independent layout at 450 agrees the
        // box is far-stuck at the content origin plus offset plus
        // padding-box height minus the box.
        let (ref_doc, ref_root) =
            layout_html_nested_at(STICKY_NESTED_HTML, css, "outer", 0.0, 450.0);
        let ref_st = ref_doc.get_element_by_id("st").expect("ref st");
        let ref_box = find_box(&ref_root, ref_st).expect("#st box");
        assert!(
            approx_eq(ref_box.rect.y - sp_y, 640.0),
            "layout agrees the box is far-stuck at 640, got {}",
            ref_box.rect.y - sp_y
        );
        // Forward ticks from a far-stuck bake are the conservative
        // re-bake path, not a zero shift: pre-fix the free interval
        // folded `Some(0)` at 530 (its lower bound was
        // `bake + near_pad`), letting the stuck box scroll away.
        assert!(tree.sticky_shifts_at(io, 0.0, 530.0).is_none());
    }

    /// Near-stuck folding on the horizontal axis (w79fix round 2):
    /// the mirror of `sticky_fold_near_stuck_with_scroller_padding`,
    /// which covered only `padding-top`. The fold passes `pad_l`
    /// on the x-axis, and the near pin is the content origin plus
    /// the offset, so a `pad_l`/`pad_r` mix-up is caught only when
    /// the two paddings differ: here left is 20 and right is 30, so
    /// swapping them would pin at 180, not 170, and the stuck match
    /// would miss. Geometry: content origin 20, `#pad` is an
    /// inline-block 100 wide so `#st`'s natural x is 120, stuck
    /// from 100; baked stuck at 150, so the baked
    /// border-box-frame position is 170.
    #[test]
    fn sticky_fold_near_stuck_with_scroller_padding_horizontal() {
        let css = "#outer { overflow-x: auto; width: 400px; padding-left: 20px; padding-right: 30px; white-space: nowrap; } \
            #pad { display: inline-block; width: 100px; height: 10px; } \
            #st { position: sticky; left: 0; width: 50px; height: 50px; display: inline-block; } \
            #tall { width: 2000px; height: 10px; }";
        let (doc, root) = layout_html_nested_at(STICKY_NESTED_HTML, css, "outer", 150.0, 0.0);
        let (_doc_state, _elements, tree) = build_tree_nested_at(&doc, &root, "outer", 150.0, 0.0);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let st = doc.get_element_by_id("st").expect("st");
        let io = tree.index_of(outer).expect("outer container");
        let sp_x = tree.container(io).scrollport.x;
        let band = tree
            .sticky_bands
            .iter()
            .find(|b| b.scroller == io)
            .expect("band for #st");
        assert!(approx_eq(band.pos_x, 170.0), "pos_x={}", band.pos_x);
        assert!(approx_eq(band.bake_x, 150.0), "bake_x={}", band.bake_x);
        assert_eq!(band.scroller_padding, Some((0.0, 30.0, 0.0, 20.0)));
        // Tick further along the plateau: exactly foldable, shift is
        // the scroll delta.
        let shifts = tree.sticky_shifts_at(io, 200.0, 0.0).expect("foldable");
        assert_eq!(shifts.len(), 1, "exactly the sticky box");
        assert_eq!(shifts[0].node, st);
        assert!(approx_eq(shifts[0].dx, 50.0));
        assert!(approx_eq(shifts[0].dy, 0.0));
        // Behavior equivalence: independent layout at 200 agrees the
        // box is stuck at the content origin plus the offset.
        let (ref_doc, ref_root) =
            layout_html_nested_at(STICKY_NESTED_HTML, css, "outer", 200.0, 0.0);
        let ref_st = ref_doc.get_element_by_id("st").expect("ref st");
        let ref_box = find_box(&ref_root, ref_st).expect("#st box");
        assert!(
            approx_eq(ref_box.rect.x - sp_x, 220.0),
            "layout agrees the box is stuck at 220, got {}",
            ref_box.rect.x - sp_x
        );
        // Scrolling back from the plateau is the conservative re-bake
        // path, not a zero shift.
        assert!(tree.sticky_shifts_at(io, 120.0, 0.0).is_none());
        // Free-baked twin: baked at 0 with the box still free
        // (natural 120), the near transition is at 100, not 120.
        let (doc, root) = layout_html(STICKY_NESTED_HTML, css);
        let (_doc_state, _elements, tree) = build_tree(&doc, &root);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let io = tree.index_of(outer).expect("outer container");
        assert!(!tree.sticky_constrained_at(io, 50.0, 0.0));
        assert!(
            tree.sticky_shifts_at(io, 50.0, 0.0)
                .expect("foldable")
                .is_empty(),
            "free box needs no extra shift"
        );
        assert!(
            tree.sticky_shifts_at(io, 110.0, 0.0).is_none(),
            "past the padded near transition must re-bake, not fold zero"
        );
    }

    /// Far-stuck folding on the horizontal axis (w79fix round 2):
    /// the mirror of `sticky_fold_far_stuck_with_scroller_padding`.
    /// Geometry: content origin 20, padding-box width 220, `#st`
    /// natural at 1020 (`nowrap` keeps the 1000-wide inline-block
    /// `#pad` on the same line), far pin at bake 500 is 690;
    /// ticking back rides the far edge.
    #[test]
    fn sticky_fold_far_stuck_with_scroller_padding_horizontal() {
        let css = "#outer { overflow-x: auto; width: 200px; padding-left: 20px; white-space: nowrap; } \
            #pad { display: inline-block; width: 1000px; height: 10px; } \
            #st { position: sticky; right: 0; width: 50px; height: 50px; display: inline-block; } \
            #tall { width: 2000px; height: 10px; }";
        let (doc, root) = layout_html_nested_at(STICKY_NESTED_HTML, css, "outer", 500.0, 0.0);
        let (_doc_state, _elements, tree) = build_tree_nested_at(&doc, &root, "outer", 500.0, 0.0);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let st = doc.get_element_by_id("st").expect("st");
        let io = tree.index_of(outer).expect("outer container");
        let sp_x = tree.container(io).scrollport.x;
        let band = tree
            .sticky_bands
            .iter()
            .find(|b| b.scroller == io)
            .expect("band for #st");
        assert!(approx_eq(band.pos_x, 690.0), "pos_x={}", band.pos_x);
        assert!(approx_eq(band.bake_x, 500.0), "bake_x={}", band.bake_x);
        assert_eq!(band.scroller_padding, Some((0.0, 0.0, 0.0, 20.0)));
        // Ticking back rides the far edge: shift is the scroll delta.
        let shifts = tree.sticky_shifts_at(io, 450.0, 0.0).expect("foldable");
        assert_eq!(shifts.len(), 1, "exactly the sticky box");
        assert_eq!(shifts[0].node, st);
        assert!(approx_eq(shifts[0].dx, -50.0));
        // Behavior equivalence: independent layout at 450 agrees the
        // box is far-stuck at the content origin plus offset plus
        // padding-box width minus the box.
        let (ref_doc, ref_root) =
            layout_html_nested_at(STICKY_NESTED_HTML, css, "outer", 450.0, 0.0);
        let ref_st = ref_doc.get_element_by_id("st").expect("ref st");
        let ref_box = find_box(&ref_root, ref_st).expect("#st box");
        assert!(
            approx_eq(ref_box.rect.x - sp_x, 640.0),
            "layout agrees the box is far-stuck at 640, got {}",
            ref_box.rect.x - sp_x
        );
        // Forward ticks from a far-stuck bake are the conservative
        // re-bake path, not a zero shift.
        assert!(tree.sticky_shifts_at(io, 530.0, 0.0).is_none());
    }

    /// Far-stuck folding with far padding on the horizontal axis
    /// (w79fix round 2): layout's hybrid interval is the content
    /// origin plus the padding-box size, and the padding-box size
    /// includes the far padding (CSS Overflow 3 §2.3: the scrollport
    /// is the padding box, borders only are subtracted). With
    /// `padding-right: 30px` the padding-box width is 250 (200 +
    /// 20 + 30), so the far pin at bake 500 is 20 + 500 + 250 - 50
    /// = 720. Deriving `sp` from the content box instead would pin
    /// at 690, and swapping `pad_l`/`pad_r` would pin at 730, so
    /// both errors fail here.
    #[test]
    fn sticky_fold_far_stuck_with_scroller_far_padding_horizontal() {
        let css = "#outer { overflow-x: auto; width: 200px; padding-left: 20px; padding-right: 30px; white-space: nowrap; } \
            #pad { display: inline-block; width: 1000px; height: 10px; } \
            #st { position: sticky; right: 0; width: 50px; height: 50px; display: inline-block; } \
            #tall { width: 2000px; height: 10px; }";
        let (doc, root) = layout_html_nested_at(STICKY_NESTED_HTML, css, "outer", 500.0, 0.0);
        let (_doc_state, _elements, tree) = build_tree_nested_at(&doc, &root, "outer", 500.0, 0.0);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let st = doc.get_element_by_id("st").expect("st");
        let io = tree.index_of(outer).expect("outer container");
        let sp_x = tree.container(io).scrollport.x;
        let sp_w = tree.container(io).scrollport.w;
        assert!(approx_eq(sp_w, 250.0), "padding-box width sp_w={}", sp_w);
        let band = tree
            .sticky_bands
            .iter()
            .find(|b| b.scroller == io)
            .expect("band for #st");
        assert!(approx_eq(band.pos_x, 720.0), "pos_x={}", band.pos_x);
        assert!(approx_eq(band.bake_x, 500.0), "bake_x={}", band.bake_x);
        assert_eq!(band.scroller_padding, Some((0.0, 30.0, 0.0, 20.0)));
        let shifts = tree.sticky_shifts_at(io, 450.0, 0.0).expect("foldable");
        assert_eq!(shifts.len(), 1, "exactly the sticky box");
        assert_eq!(shifts[0].node, st);
        assert!(approx_eq(shifts[0].dx, -50.0));
        let (ref_doc, ref_root) =
            layout_html_nested_at(STICKY_NESTED_HTML, css, "outer", 450.0, 0.0);
        let ref_st = ref_doc.get_element_by_id("st").expect("ref st");
        let ref_box = find_box(&ref_root, ref_st).expect("#st box");
        assert!(
            approx_eq(ref_box.rect.x - sp_x, 670.0),
            "layout agrees the box is far-stuck at 670, got {}",
            ref_box.rect.x - sp_x
        );
        assert!(tree.sticky_shifts_at(io, 530.0, 0.0).is_none());
    }

    /// Far-stuck folding with far padding on the vertical axis
    /// (w79fix round 2): the vertical twin of the horizontal
    /// far-padding case above. Padding-box height is 250 (200 + 20
    /// + 30), so the far pin at bake 500 is 20 + 500 + 250 - 50 =
    ///   720; a content-box `sp` would pin at 690.
    #[test]
    fn sticky_fold_far_stuck_with_scroller_far_padding() {
        let css = "#outer { overflow-y: auto; height: 200px; padding-top: 20px; padding-bottom: 30px; } \
            #pad { height: 1000px; } \
            #st { position: sticky; bottom: 0; height: 50px; } \
            #tall { height: 2000px; }";
        let (doc, root) = layout_html_nested_at(STICKY_NESTED_HTML, css, "outer", 0.0, 500.0);
        let (_doc_state, _elements, tree) = build_tree_nested_at(&doc, &root, "outer", 0.0, 500.0);
        let outer = doc.get_element_by_id("outer").expect("outer");
        let st = doc.get_element_by_id("st").expect("st");
        let io = tree.index_of(outer).expect("outer container");
        let sp_y = tree.container(io).scrollport.y;
        let sp_h = tree.container(io).scrollport.h;
        assert!(approx_eq(sp_h, 250.0), "padding-box height sp_h={}", sp_h);
        let band = tree
            .sticky_bands
            .iter()
            .find(|b| b.scroller == io)
            .expect("band for #st");
        assert!(approx_eq(band.pos_y, 720.0), "pos_y={}", band.pos_y);
        assert!(approx_eq(band.bake_y, 500.0), "bake_y={}", band.bake_y);
        assert_eq!(band.scroller_padding, Some((20.0, 0.0, 30.0, 0.0)));
        let shifts = tree.sticky_shifts_at(io, 0.0, 450.0).expect("foldable");
        assert_eq!(shifts.len(), 1, "exactly the sticky box");
        assert_eq!(shifts[0].node, st);
        assert!(approx_eq(shifts[0].dy, -50.0));
        let (ref_doc, ref_root) =
            layout_html_nested_at(STICKY_NESTED_HTML, css, "outer", 0.0, 450.0);
        let ref_st = ref_doc.get_element_by_id("st").expect("ref st");
        let ref_box = find_box(&ref_root, ref_st).expect("#st box");
        assert!(
            approx_eq(ref_box.rect.y - sp_y, 670.0),
            "layout agrees the box is far-stuck at 670, got {}",
            ref_box.rect.y - sp_y
        );
        assert!(tree.sticky_shifts_at(io, 0.0, 530.0).is_none());
    }
}
