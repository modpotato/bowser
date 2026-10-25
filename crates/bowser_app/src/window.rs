//! Windowed event loop: winit window, WebGPU presentation, input routing.
//!
//! [`run_windowed`] owns the windowed UI thread. It creates a winit 0.30
//! window, brings up a wgpu 30 surface on it, and pumps the event loop:
//!
//! * **Render loop.** [`refresh`] re-runs the style/layout/paint pipeline
//!   stages whose [`DirtyFlags`](crate::tab::DirtyFlags) are set, then the
//!   tab's cached display list is presented with
//!   [`WgpuBackend::render_to_surface`]. The viewport scroll offset rides
//!   in the per-frame uniform; nested scroll containers' offsets are
//!   baked into the paint tree the display list builds from, so nested
//!   scrolling rebuilds the display list without a relayout.
//! * **Input routing.** Mouse motion updates per-node hover state
//!   ([`bowser_css::ElementState`]); mouse press/release drives
//!   active/focus state and clicks (link navigation or the JS dispatch
//!   hook); wheel dispatches to the innermost scroll container under the
//!   cursor, chaining the remainder outward to the viewport; keys go to
//!   the focused document via the JS dispatch hook.
//! * **Resize.** The tab viewport follows the window's logical size and
//!   the layout + paint stages are invalidated.
//! * **Script pump.** Page scripts register timers on their
//!   [`PageVm`](crate::script::PageVm)'s event loop, which only advances
//!   when something pumps it: every event-loop turn (`about_to_wait`)
//!   pumps the active tab's realm so `setTimeout` callbacks fire while
//!   the window is open. v1 note: a pump cannot change what paints (page
//!   scripts cannot mutate the DOM yet), so it never requests a redraw;
//!   the bindings worker will mark render stages dirty when scripts start
//!   affecting the page.
//! * **Loading.** No network ever runs on the event thread. Document
//!   loads (the initial URL and link clicks) run on detached worker
//!   threads; completed loads are picked up in `about_to_wait` via a
//!   channel. Only one load is in flight at a time in v1: a new navigation
//!   supersedes the previous pending one, whose worker bails out through
//!   the [`LoadControl`] generation counter (cooperative cancellation) and
//!   whose cookie jar is handed to the replacement worker, so a
//!   superseded navigation neither stalls nor orphans the cookies of the
//!   load replacing it. Each worker owns a private jar swapped out of the
//!   profile mutex — the profile lock is never held across a load — so an
//!   abandoned load can never block the new one on the jar. Closing the
//!   window just stops listening for reports: workers are detached, never
//!   joined, so a stalled server can't hang browser exit.
//!
//! The style stage currently re-runs together with layout: selector
//! matching and the cascade live inside
//! [`bowser_layout::layout_with_intrinsics_and_scroll_and_offsets`] (there is no separate
//! style-only pass in the tree yet). The per-node [`CssElementState`]
//! map is maintained here and marks style dirty on change; threading it
//! into matching belongs to the style worker.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use bowser_css::{
    ComputedStyle, CssValue, ElementState as CssElementState, LengthUnit, Stylesheet,
    selectors_are_subtree_local,
};
use bowser_dom::{Document, NodeId, NodeKind};
use bowser_layout::{
    LayoutBox, Position, Size, layout_subtree_with_intrinsics_and_scroll_and_offsets,
    layout_with_intrinsics_and_scroll_and_offsets, position_of, ua_stylesheet,
};
use bowser_net::cookies::CookieJar;
use bowser_net::fetch::FetchTrace;
use bowser_paint::WgpuBackend;
use bowser_paint::display_list::{
    DecodedImage, DisplayList, DisplayListBuilder, ImageId, build_display_list_reused,
    push_scrollbar,
};
use bowser_paint::fonts::{FontCache, FontKey, SwashFontMetrics};
use bowser_paint::render::{css_font_family, css_font_style, css_font_weight};
use bowser_url::Url;

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{
    ElementState as WinitElementState, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::Window;

use crate::browser::{Browser, TabArena, TabId};
use crate::loader::{LoadError, LoadedDocument, cached_image_for_src};
use crate::script::PageEventDispatch;
use crate::scroll::{
    self, ScrollTree, ScrollbarAxis, ScrollbarDrag, ScrollbarHit, SnapEventDelivery, SnapTargets,
    TranslateCtx, box_translate, px_font_size,
};
use crate::scroll_touch::TouchScrollState;
use crate::tab::{DirtyFlags, InstalledDocument, RenderPipeline, Tab};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Every way [`run_windowed`] can fail before or during the event loop.
#[derive(Debug)]
pub enum WindowError {
    /// The winit event loop could not be created or pumped (no display,
    /// platform backend missing, ...).
    EventLoop(String),
    /// Window creation failed.
    Window(String),
    /// wgpu surface creation failed.
    Surface(String),
    /// No usable GPU adapter.
    ///
    /// This build creates a Vulkan-only wgpu instance ([`init_gpu`] passes
    /// `Backends::VULKAN`), so adapter request fails on machines without a
    /// Vulkan driver/loader even when other graphics APIs would work.
    /// There is no software fallback in the windowed backend; use
    /// `--headless` to render offscreen without a GPU.
    NoAdapter,
    /// Logical device creation failed.
    Device(String),
}

impl fmt::Display for WindowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EventLoop(e) => write!(f, "event loop error: {e}"),
            Self::Window(e) => write!(f, "window creation failed: {e}"),
            Self::Surface(e) => write!(f, "surface creation failed: {e}"),
            Self::NoAdapter => f.write_str(
                "no usable GPU adapter found (windowed mode is Vulkan-only in this build; \
                 use --headless to render without a GPU)",
            ),
            Self::Device(e) => write!(f, "GPU device creation failed: {e}"),
        }
    }
}

impl std::error::Error for WindowError {}

// ---------------------------------------------------------------------------
// Built-in page
// ---------------------------------------------------------------------------

/// Built-in welcome page rendered for `about:blank` (the default URL).
pub const WELCOME_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>welcome to bowser</title>
<style>
  body { font-family: sans-serif; background: #0b0b12; color: #e8e8f2; margin: 0;
         display: flex; align-items: center; justify-content: center; height: 100vh; }
  main { text-align: center; }
  h1 { font-size: 48px; margin: 0 0 8px; letter-spacing: 2px; }
  p { color: #9a9ab0; }
  code { background: #1a1a28; padding: 2px 8px; border-radius: 6px; color: #cfcfea; }
</style>
</head>
<body>
<main>
  <h1>bowser</h1>
  <p>a tiny, fast browser.</p>
  <p><code>bowser https://example.com --headless &gt; out.rgba</code></p>
</main>
</body>
</html>
"#;

// ---------------------------------------------------------------------------
// JS dispatch hook
// ---------------------------------------------------------------------------

/// Host-side entry point for dispatching UI events into page scripts.
///
/// The production implementation ([`crate::script::PageEventDispatch`])
/// runs the target's inline handler attribute and the listeners the
/// page registered through `addEventListener` on the tab's real JS
/// realm, then reports whether the page prevented the default action —
/// so the window keeps owning default actions (link navigation on
/// click) until the page cancels them. [`run_windowed`] installs the
/// production dispatcher; [`run_windowed_with_js`] is the install point
/// for test doubles and future bindings work.
pub trait JsDispatch {
    /// Dispatch a `click` event at `target` in `tab`'s document.
    /// Returns true when the page prevented the default action (the
    /// window then skips link navigation).
    fn dispatch_click(&mut self, tab: &mut Tab, target: NodeId) -> bool;
    /// Dispatch a key event for `key` (see [`key_label`]) to the focused
    /// document. Returns true when the page prevented the default
    /// action.
    fn dispatch_key(&mut self, tab: &mut Tab, key: &str) -> bool;
    /// Dispatch a `wheel` event at `target` in `tab`'s document before
    /// the window scrolls: `delta_x`/`delta_y` are the DOM `WheelEvent`
    /// deltas and `delta_mode` its `deltaMode` (0 = pixels, 1 = lines,
    /// 2 = pages). Returns true when a non-passive listener prevented
    /// the default action (the window then skips the scroll).
    ///
    /// The default does nothing and reports nothing prevented, so hosts
    /// without a JS bridge keep the pre-dispatch behavior; the
    /// production dispatcher
    /// ([`PageEventDispatch`](crate::script::PageEventDispatch))
    /// overrides this.
    fn dispatch_wheel(
        &mut self,
        tab: &mut Tab,
        target: NodeId,
        delta_x: f64,
        delta_y: f64,
        delta_mode: u32,
    ) -> bool {
        let _ = (tab, target, delta_x, delta_y, delta_mode);
        false
    }
    /// Dispatch a non-bubbling `scroll` event at `target` in `tab`'s
    /// document (a nested scroll container whose offset moved).
    fn dispatch_scroll(&mut self, tab: &mut Tab, target: NodeId);
    /// Dispatch a non-bubbling `scrollend` event at `target` in `tab`'s
    /// document (a nested scroll container whose scroll gesture ended:
    /// a full event-loop tick passed with no movement).
    ///
    /// The default does nothing, so hosts without a JS bridge keep
    /// working; the production dispatcher
    /// ([`PageEventDispatch`](crate::script::PageEventDispatch))
    /// overrides this.
    fn dispatch_scrollend(&mut self, tab: &mut Tab, target: NodeId) {
        let _ = (tab, target);
    }
    /// Dispatch a `scrollsnapchanging` / `scrollsnapchange` event
    /// (CSS Scroll Snap 2 §5) at `target` in `tab`'s document, carrying
    /// the snap target pair `targets`. `target` is `None` for the
    /// viewport scroller, whose snap events dispatch at the `Document`
    /// (and bubble to `window`).
    ///
    /// The default does nothing, so hosts without a JS bridge keep
    /// working; the production dispatcher
    /// ([`PageEventDispatch`](crate::script::PageEventDispatch))
    /// overrides this.
    fn dispatch_snap_event(
        &mut self,
        tab: &mut Tab,
        target: Option<NodeId>,
        event_type: &str,
        targets: SnapTargets,
    ) {
        let _ = (tab, target, event_type, targets);
    }
}

// ---------------------------------------------------------------------------
// Input state (pure, window-independent)
// ---------------------------------------------------------------------------

/// Per-window UI state: hover/active/focus plus the CSS [`CssElementState`]
/// map that feeds `:hover`/`:active`/`:focus` matching.
#[derive(Debug, Default)]
struct InputState {
    /// Node currently under the cursor, if any.
    hover: Option<NodeId>,
    /// Node the primary button was pressed on (cleared on release).
    active: Option<NodeId>,
    /// Node holding keyboard focus.
    focused: Option<NodeId>,
    /// Dynamic pseudo-class state per element. The renderer fills this in;
    /// the style worker reads it when matching.
    element_states: HashMap<NodeId, CssElementState>,
    /// Last cursor position in CSS px (winit reports button events without
    /// a position, so the motion position is remembered).
    cursor: (f32, f32),
}

fn state_for(input: &mut InputState, node: NodeId) -> &mut CssElementState {
    input.element_states.entry(node).or_default()
}

fn rect_contains(rect: &bowser_layout::Rect, x: f32, y: f32) -> bool {
    x >= rect.x && x < rect.x + rect.w && y >= rect.y && y < rect.y + rect.h
}

/// Intersection of two rects; `None` when they do not overlap with
/// positive area. A zero-area intersection clips everything out: a
/// zero-size scrollport shows no content, so nothing inside it is
/// hittable.
fn rect_intersect(a: &bowser_layout::Rect, b: &bowser_layout::Rect) -> Option<bowser_layout::Rect> {
    let x = a.x.max(b.x);
    let y = a.y.max(b.y);
    let right = (a.x + a.w).min(b.x + b.w);
    let bottom = (a.y + a.h).min(b.y + b.h);
    if right > x && bottom > y {
        Some(bowser_layout::Rect {
            x,
            y,
            w: right - x,
            h: bottom - y,
        })
    } else {
        None
    }
}

/// Deepest layout box containing `(x, y)` (page px), or `None`.
///
/// Children are visited last-to-first: later siblings paint on top, so
/// they win the hit test. Boxes with no DOM node (anonymous boxes) are
/// transparent to hits — the search continues into their children, and
/// the nearest ancestor *with* a node wins.
///
/// `shift` is the accumulated scroll offset of the current box's
/// scroller ancestors: content inside a nested scroll container paints
/// shifted by its container's offset, so the hit test compares against
/// the shifted rects. A scroll container's own box is not shifted by its
/// own offset — only its descendants are.
///
/// `clip` is the accumulated scrollport clip (`None` = unbounded):
/// each scroll container intersects it with its own scrollport for its
/// subtree. Content scrolled or clipped out of a nested container is
/// painted clipped and must not be hittable — without the clip the
/// shifted rects alone would still match it, and the wheel chain would
/// start at the wrong scroller. The scrollport is stored in page
/// coordinates, so it is shifted by the same accumulated `shift` the
/// container's own box was tested with.
///
/// `tshift` is the accumulated `transform: translate()` of the current
/// box's ancestors (the box's own translate joins before its own box
/// is tested): layout rects are pre-transform, so the hit test
/// compares against the translated rects. Translations commute with
/// the scroll-offset `shift`, so both accumulate as plain offsets.
/// Only translations are carried (see
/// [`crate::scroll::box_translate`]); the baked per-container offset
/// [`crate::scroll::ScrollTree::innermost_at`] uses is the same
/// accumulation, so the two agree.
///
/// The eight parameters are the natural recursion state of the walk
/// (query point, two accumulated offsets, translate context, clip);
/// bundling them would just add indirection at every level.
#[allow(clippy::too_many_arguments)]
fn hit_test(
    tree: &ScrollTree,
    root: &LayoutBox,
    x: f32,
    y: f32,
    shift: (f32, f32),
    tshift: (f32, f32),
    tctx: TranslateCtx,
    clip: Option<bowser_layout::Rect>,
) -> Option<NodeId> {
    let (ttx, tty) = box_translate(root.style.as_deref(), root.rect.w, root.rect.h, tctx);
    let tshift = (tshift.0 + ttx, tshift.1 + tty);
    let rect = bowser_layout::Rect {
        x: root.rect.x - shift.0 + tshift.0,
        y: root.rect.y - shift.1 + tshift.1,
        w: root.rect.w,
        h: root.rect.h,
    };
    if !rect_contains(&rect, x, y) {
        return None;
    }
    if clip.is_some_and(|c| !rect_contains(&c, x, y)) {
        return None;
    }
    let (child_shift, child_clip) = match root.node.and_then(|n| tree.index_of(n)) {
        Some(i) => {
            let container = tree.container(i);
            let (ox, oy) = container.state.offset();
            let port = bowser_layout::Rect {
                x: container.scrollport.x - shift.0 + tshift.0,
                y: container.scrollport.y - shift.1 + tshift.1,
                w: container.scrollport.w,
                h: container.scrollport.h,
            };
            let clipped = match clip {
                Some(c) => rect_intersect(&c, &port),
                None => Some(port),
            };
            // An empty intersection clips the whole subtree out: no
            // descendant can be hit. The container's own box was already
            // tested above and still wins.
            let Some(clipped) = clipped else {
                return root.node;
            };
            ((shift.0 + ox, shift.1 + oy), Some(clipped))
        }
        None => (shift, clip),
    };
    for child in root.children.iter().rev() {
        if let Some(node) = hit_test(tree, child, x, y, child_shift, tshift, tctx, child_clip) {
            return Some(node);
        }
    }
    root.node
}

/// Hit-test target at a viewport position (CSS px), adding the scroll
/// offset to reach page coordinates and accounting for nested scroll
/// containers' offsets and `transform: translate()` ancestors.
fn target_at(tab: &Tab, x: f32, y: f32) -> Option<NodeId> {
    let (sx, sy) = tab.scroll().offset();
    tab.layout_root.as_deref().and_then(|root| {
        let tctx = TranslateCtx {
            root_font_size: root
                .style
                .as_deref()
                .map(|s| px_font_size(s, 16.0))
                .unwrap_or(16.0),
            viewport_w: tab.viewport.w,
            viewport_h: tab.viewport.h,
        };
        hit_test(
            &tab.scroll_tree,
            root,
            x + sx,
            y + sy,
            (0.0, 0.0),
            (0.0, 0.0),
            tctx,
            // Unbounded: `target_at` only ever queries points inside the
            // viewport, so the viewport clip would be a no-op; nested
            // scrollports still clip their subtrees via `hit_test`.
            None,
        )
    })
}

/// Update hover state for a cursor move. Returns true when the hover
/// target changed (the caller should mark style dirty and redraw).
fn update_hover(input: &mut InputState, tab: &Tab, x: f32, y: f32) -> bool {
    input.cursor = (x, y);
    let target = target_at(tab, x, y);
    if target == input.hover {
        return false;
    }
    if let Some(old) = input.hover.take() {
        state_for(input, old).hover = false;
    }
    if let Some(node) = target {
        state_for(input, node).hover = true;
    }
    input.hover = target;
    true
}

/// Primary-button press at a viewport position (CSS px). Sets the active
/// and focused nodes; returns the pressed node, if any.
fn press_at(input: &mut InputState, tab: &Tab, x: f32, y: f32) -> Option<NodeId> {
    input.cursor = (x, y);
    let target = target_at(tab, x, y);
    if let Some(old) = input.focused
        && Some(old) != target
    {
        state_for(input, old).focus = false;
    }
    if let Some(node) = target {
        let st = state_for(input, node);
        st.active = true;
        st.focus = true;
    }
    input.active = target;
    input.focused = target;
    target
}

/// Primary-button release at a viewport position (CSS px).
///
/// Returns the clicked node: the release target when it matches the press
/// target. Active state is always cleared.
fn release_at(input: &mut InputState, tab: &Tab, x: f32, y: f32) -> Option<NodeId> {
    input.cursor = (x, y);
    let target = target_at(tab, x, y);
    let pressed = input.active.take();
    if let Some(node) = pressed {
        state_for(input, node).active = false;
    }
    match (pressed, target) {
        (Some(p), Some(t)) if p == t => Some(t),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Scroll / resize (pure)
// ---------------------------------------------------------------------------

/// Fold a scroll gesture's nested-container movement into the tab's
/// paint tree and invalidate paint. The viewport scroller rides the
/// per-frame uniform, so viewport-only scrolls invalidate no pipeline
/// stage — unless a `position: sticky` box is within its constraint
/// band of the new offset, in which case layout is dirtied too (see
/// [`dirty_layout_for_sticky`]).
///
/// Nested containers ride the transform fast path when every moved
/// container is wrapped ([`ScrollTree::try_fast_fold`]): the tick then
/// patches the display-list scroll transforms in place and only the
/// epilogue (scrollbars + glow) is stale. Otherwise the movement folds
/// into the paint tree ([`ScrollTree::sync_paint_offsets`]) and the
/// display list rebuilds (and re-wraps).
///
/// Shared with the navigation module: the fragment fast path
/// ([`crate::navigation::scroll_to_fragment`]) moves nested containers
/// through [`crate::scroll::ScrollTree::reveal_node`] and must fold them
/// in exactly like the wheel/keyboard/`scrollIntoView()` paths.
pub(crate) fn sync_nested_paint(tab: &mut Tab, scrolled: &[Option<NodeId>]) {
    let viewport_moved = scrolled.iter().any(Option::is_none);
    if scrolled.iter().any(Option::is_some) {
        // Transform fast path first: patch the wrapped scroll
        // transforms in place. The paint tree is deliberately left
        // stale (it is a build artifact); only the chrome — scrollbar
        // thumbs track the new offset — and the glow are stale, and the
        // next `refresh` re-pushes just the epilogue.
        let fast = match tab.display_list.as_mut() {
            Some(list) => tab.render.scroll_tree.try_fast_fold(list).is_some(),
            // No display list yet: the next build wraps from scratch.
            None => false,
        };
        if fast {
            tab.scroll_tree.mark_scrollbars_dirty();
            tab.scroll_tree.mark_glow_dirty();
        } else {
            if let Some(paint) = tab.render.paint_root.as_mut() {
                tab.render.scroll_tree.sync_paint_offsets(paint);
            }
            tab.dirty.paint = true;
        }
    } else if viewport_moved {
        // Viewport-only: no paint-tree rects to shift and no wrap to
        // patch, so the fold is a baseline advance only — but it must
        // still happen. The caller reported this movement through the
        // direct path, and without the re-arm the next pump fold would
        // re-report it as a second `scroll`.
        tab.scroll_tree.rearm_viewport_fold_baselines();
    }
    if viewport_moved {
        // The viewport scroller moved. Its own bars and
        // `position: fixed` containers' bars bake the live viewport
        // offset into page coordinates, so without a repaint those
        // baked rects would go stale — visually sliding while
        // `scrollbar_at` hit-tests the live offset. Mark them stale
        // instead of forcing a repaint: the next `refresh` re-pushes
        // just the bar items, preserving the GPU-uniform fast path.
        tab.scroll_tree.mark_scrollbars_dirty();
    }
    dirty_layout_for_sticky(tab, scrolled);
}

/// Dirty layout after a scroll moved something, when a `position:
/// sticky` box could be stuck or transitioning at the new offset.
///
/// Sticky constraints are baked at layout time for the offsets live
/// then; wheel scrolls ride the GPU uniform and nested scrolls fold
/// into the paint tree, so without a re-layout a stuck box would lag
/// behind its scroll-driven position until something else triggers
/// layout. Re-baking on scroll is exact — it reuses layout's own
/// constraint pass with the live offsets — and costs nothing on pages
/// without sticky boxes.
///
/// Hysteresis: each sticky box carries a scroll-offset constraint
/// band ([`ScrollTree::sticky_constrained_at`], built once per
/// layout), and only the moved scroller's bands are consulted — a
/// nested scroll cannot move a viewport-constrained sticky box and
/// vice versa. A sticky box far from the viewport keeps the no-layout
/// fast path; `refresh` coalesces a storm of wheel ticks into one
/// layout per frame.
///
/// A per-frame sticky re-evaluation without layout would be cheaper
/// still, but it cannot be exact from the app side: the constraint
/// needs each sticky box's in-flow position and its containing block's
/// definiteness, both baked over by layout and unrecoverable from the
/// [`LayoutBox`] tree (the layout forest is internal to
/// `bowser_layout`).
fn dirty_layout_for_sticky(tab: &mut Tab, scrolled: &[Option<NodeId>]) {
    if scrolled.is_empty() || !tab.scroll_tree.has_sticky() {
        return;
    }
    for target in scrolled {
        // Only the moved scroller's bands matter. An unresolvable
        // node stays conservative: dirty, as before the hysteresis.
        let idx = match target {
            None => 0,
            Some(node) => match tab.scroll_tree.index_of(*node) {
                Some(i) => i,
                None => {
                    // Sticky re-bake is non-DOM dirt that can move
                    // boxes outside any retained subtree: full layout
                    // via the helper (F1).
                    tab.mark_layout_dirty_full();
                    return;
                }
            },
        };
        let (ox, oy) = tab.scroll_tree.container(idx).state.offset();
        if tab.scroll_tree.sticky_constrained_at(idx, ox, oy) {
            tab.mark_layout_dirty_full();
            return;
        }
    }
}

/// Start container for a wheel gesture at viewport point `(x, y)`.
///
/// The DOM hit test runs first so a hit-tested overlay — a fixed modal
/// over a scroller, a later sibling, ... — intercepts the gesture: the
/// chain starts at the hit node's innermost scroll container instead of
/// the geometrically innermost container under the cursor. Falls back to
/// [`ScrollTree::innermost_at`] when nothing is hit (or no document is
/// installed).
fn wheel_start_container(tab: &Tab, x: f32, y: f32) -> usize {
    match target_at(tab, x, y).zip(tab.document_ref()) {
        Some((node, doc)) => tab.scroll_tree.container_for_node(node, &doc),
        None => tab.scroll_tree.innermost_at(x, y),
    }
}

/// Scrollbar hit for a press at viewport point `(x, y)`, with overlay
/// interception: the DOM hit test runs first (like
/// [`wheel_start_container`]), so a positioned overlay covering a
/// scrollbar gets the press instead of the covered container's bar.
/// The bar wins when the hit node's scroll container is the bar's
/// container or nested inside it — the press is on the container's
/// own chrome, which paints above its content (the viewport's own
/// bars likewise paint above page content, matching browser UI).
fn press_scrollbar_at(tab: &Tab, x: f32, y: f32) -> Option<ScrollbarHit> {
    let hit = tab.scroll_tree.scrollbar_at(x, y)?;
    // No DOM node under the cursor: no overlay can intercept.
    let (node, doc) = target_at(tab, x, y).zip(tab.document_ref())?;
    // Walk the hit node's scroller ancestry: reaching the bar's
    // container means the press is inside it (its chrome paints above
    // that content); reaching the viewport first means an overlay
    // from outside covers the bar.
    let mut c = tab.scroll_tree.container_for_node(node, &doc);
    loop {
        if c == hit.container {
            return Some(hit);
        }
        if c == 0 {
            return None;
        }
        c = tab.scroll_tree.parent(c);
    }
}

/// Middle-click autoscroll gesture state: the press anchors at its
/// point, and motion beyond a dead zone scrolls the latched container.
/// Release (or Escape) ends the gesture.
/// `Copy` so the event handlers can snapshot it without borrow fights.
#[derive(Clone, Copy, Debug)]
struct AutoscrollState {
    /// Press point in CSS px (viewport coordinates).
    anchor: (f32, f32),
    /// Scroller latched at press, by DOM node (`None` = viewport):
    /// the gesture scrolls this one container for its whole lifetime
    /// (the wheel-gesture latch rule), re-resolved per motion event so
    /// a relayout mid-gesture ends the gesture instead of scrolling a
    /// recycled tree index.
    node: Option<NodeId>,
}

/// Dead-zone radius (CSS px) around the autoscroll anchor: motion
/// inside it never scrolls.
const AUTOSCROLL_DEADZONE: f32 = 8.0;
/// Per-motion-event scroll px per px of cursor travel beyond the dead
/// zone: the gesture's per-event velocity, proportional to distance.
const AUTOSCROLL_RATE: f32 = 0.25;

/// Scroll delta for one middle-click-autoscroll motion from `anchor` to
/// `cursor` (CSS px): zero inside the dead zone, otherwise proportional
/// to the distance beyond it, pointing away from the anchor — the
/// cursor below/right of the anchor scrolls down/right.
fn autoscroll_delta(anchor: (f32, f32), cursor: (f32, f32)) -> (f32, f32) {
    let (vx, vy) = (cursor.0 - anchor.0, cursor.1 - anchor.1);
    let dist = vx.hypot(vy);
    if dist <= AUTOSCROLL_DEADZONE {
        return (0.0, 0.0);
    }
    let k = (dist - AUTOSCROLL_DEADZONE) * AUTOSCROLL_RATE / dist;
    (vx * k, vy * k)
}

/// Maximum gap between wheel events belonging to one scroll gesture: a
/// wheel event arriving later starts a fresh gesture (and re-resolves
/// its scroller from the cursor). Touchpad drivers bracket a gesture
/// with `TouchPhase::Started`/`Ended`, which reset the latch
/// explicitly; discrete mouse wheels report `Moved` only, so the
/// timeout is the gesture boundary there.
const WHEEL_GESTURE_TIMEOUT: Duration = Duration::from_millis(500);

/// Latched start container for the in-flight wheel-scroll gesture.
/// Every wheel event in one gesture chains from the scroller resolved
/// when the gesture began, so a fast fling over a nested scroller
/// keeps its target when the content — or the cursor — crosses a
/// scroller boundary mid-gesture (the touch-drag anchor rule, applied
/// to the wheel).
///
/// The latch keys the scroller by DOM node (`None` = viewport) and
/// re-resolves it per wheel event: a relayout mid-gesture re-keys the
/// tree, and chaining from a stale index would scroll the wrong
/// container. A container that vanished mid-gesture ends the latch —
/// the event scrolls nothing, and the next event re-latches fresh. The
/// tab is keyed too, so a tab switch mid-gesture starts over.
#[derive(Clone, Copy, Debug)]
struct WheelLatch {
    tab: TabId,
    node: Option<NodeId>,
    last: Instant,
}

/// Resolve a latched scroller identity against the live scroll tree:
/// `None` (the viewport latch) is container 0, `Some(node)` looks the
/// node's container up in the current tree. `None` when the tree is
/// empty or the container is gone — the caller ends the gesture
/// instead of chaining from a recycled index.
fn resolve_latched_node(tree: &ScrollTree, node: Option<NodeId>) -> Option<usize> {
    match node {
        None => (!tree.is_empty()).then_some(0),
        Some(n) => tree.index_of(n),
    }
}

/// Dispatch a `(dx, dy)` CSS px scroll through the chain starting at
/// `start` (see [`ScrollTree::scroll_chain`]): the innermost container
/// scrolls first, the unconsumed remainder chaining outward to the
/// viewport. Folds the movement into the paint tree without a relayout.
///
/// Returns the scrolled containers innermost-first (`None` = viewport);
/// callers fire `scroll` listeners only when it is non-empty.
fn apply_scroll_chain(tab: &mut Tab, start: usize, dx: f32, dy: f32) -> Vec<Option<NodeId>> {
    let outcome = tab.scroll_tree.scroll_chain(start, dx, dy);
    let scrolled = outcome.nodes();
    sync_nested_paint(tab, &scrolled);
    scrolled
}

/// Dispatch a wheel delta of `(dx, dy)` CSS px at viewport point
/// `(x, y)`: the innermost scroll container under the cursor scrolls
/// first, the unconsumed remainder chaining outward to the viewport
/// (scroll chaining). Each container clamps into `[origin, max]` per axis
/// (published by [`refresh`] after each layout; `(0, 0)` before the
/// first layout, so pre-layout wheels are no-ops).
///
/// Test-only: production wheel input goes through [`WindowApp::on_wheel`],
/// which latches the gesture's scroller ([`WheelLatch`]) instead of
/// re-resolving per event. This keeps the unlatched per-event path for
/// unit tests of the chain itself.
///
/// Returns the scrolled containers innermost-first (`None` = viewport);
/// callers fire `scroll` listeners only when it is non-empty.
#[cfg(test)]
fn apply_wheel(tab: &mut Tab, x: f32, y: f32, dx: f32, dy: f32) -> Vec<Option<NodeId>> {
    // Route through the DOM hit test first: an overlay (e.g. a fixed
    // modal) over a scroll container must intercept the wheel instead of
    // scrolling the covered container.
    let start = wheel_start_container(tab, x, y);
    apply_scroll_chain(tab, start, dx, dy)
}

/// Write one scrollbar gesture's target offset into the container's
/// shared [`ScrollState`](bowser_js::ScrollState) — whose own clamp
/// keeps it inside `[origin, max]` — and fold the movement into the
/// paint tree without a relayout ([`sync_nested_paint`]). The container
/// is resolved by DOM node per call, so a container that vanished
/// mid-gesture ends the gesture instead of scrolling a recycled tree
/// index.
///
/// Returns the moved containers innermost-first (`None` = viewport);
/// empty when the container is gone or the offset did not change.
/// Callers fire `scroll` listeners only when it is non-empty.
fn apply_scrollbar_offset(
    tab: &mut Tab,
    node: Option<NodeId>,
    axis: ScrollbarAxis,
    target: f32,
) -> Vec<Option<NodeId>> {
    let Some(idx) = tab.scroll_tree.scrollbar_container_idx(node) else {
        return Vec::new();
    };
    let container = tab.scroll_tree.container(idx);
    let (ox, oy) = container.state.offset();
    let (nx, ny) = match axis {
        ScrollbarAxis::Vertical => (ox, target),
        ScrollbarAxis::Horizontal => (target, oy),
    };
    let node = container.node;
    container.state.set_offset(nx, ny);
    if container.state.offset() == (ox, oy) {
        return Vec::new();
    }
    let scrolled = vec![node];
    sync_nested_paint(tab, &scrolled);
    scrolled
}

/// Apply a scrollbar thumb-drag motion at viewport point `(x, y)`:
/// rebase the press-time track mapping against the live tree, then
/// map the cursor onto the container's scroll range
/// ([`ScrollbarDrag::offset_for`]).
///
/// Returns the rebased drag plus the moved containers (as
/// [`apply_scrollbar_offset`]), or `None` when the gesture ended
/// mid-motion: the container is gone, or a relayout resized the
/// track or repositioned it relative to its frame. The caller clears
/// its drag state on `None` and settles scroll-snap once, like a
/// release — never scrolling through stale geometry.
fn apply_scrollbar_drag(
    tab: &mut Tab,
    drag: &ScrollbarDrag,
    x: f32,
    y: f32,
) -> Option<(ScrollbarDrag, Vec<Option<NodeId>>)> {
    let rebased = tab.scroll_tree.scrollbar_drag_rebased(*drag)?;
    let target = rebased.offset_for(x, y);
    let scrolled = apply_scrollbar_offset(tab, rebased.node, rebased.axis, target);
    Some((rebased, scrolled))
}

/// Apply a scrollbar track click at viewport point `(x, y)`: page the
/// container one scrollport toward the click side of the thumb.
fn apply_scrollbar_page(tab: &mut Tab, hit: ScrollbarHit, x: f32, y: f32) -> Vec<Option<NodeId>> {
    let Some(target) = tab.scroll_tree.scrollbar_page_target(hit, x, y) else {
        return Vec::new();
    };
    // The hit was built from this same tree in this same handler — no
    // relayout can intervene — so the index is still valid; the offset
    // write below re-resolves by node defensively anyway.
    let node = tab.scroll_tree.container(hit.container).node;
    apply_scrollbar_offset(tab, node, hit.axis, target)
}

/// Keyboard scrolling fallback (pure): when the page did not prevent
/// the key's default action, scrolling keys pan the focused element's
/// scroll container — arrows by a line step, PageUp/PageDown/Space by
/// one page of the target container's own scrollport, Home/End to the
/// top/bottom edge — or the viewport when nothing (or nothing
/// scrollable) has focus. Returns the scrolled containers
/// innermost-first (`None` = viewport); empty when nothing moved.
/// `shift`: Shift held — Shift+Space pages up (like browsers); the
/// other scrolling keys ignore it. Both the windowed key handler
/// (live modifier state) and the synthetic testdriver key path
/// ([`PageEventDispatch::dispatch_key_down_at`](crate::script::PageEventDispatch::dispatch_key_down_at),
/// own `KeyModifiers`) funnel through here, so both apply the same
/// default action.
pub(crate) fn apply_key_scroll_with_shift(
    tab: &mut Tab,
    focused: Option<NodeId>,
    label: &str,
    shift: bool,
) -> Vec<Option<NodeId>> {
    const LINE: f32 = 40.0;
    // The innermost scroll container that is the focused node itself or
    // one of its scroller ancestors; the viewport when the focus sits
    // outside every nested container.
    let target = focused
        .and_then(|n| {
            tab.document_ref()
                .as_ref()
                .map(|doc| tab.scroll_tree.container_for_node(n, doc))
        })
        .unwrap_or(0);
    // One page is the target container's own scrollport height — a
    // PageDown inside a 100px nested scroller moves 100px, not a full
    // viewport — matching the wheel `deltaMode: 2` scaling.
    let page = if tab.scroll_tree.is_empty() {
        0.0
    } else {
        tab.scroll_tree.container(target).scrollport.h
    };
    let scrolled = match label {
        "ArrowDown" => tab.scroll_tree.scroll_chain(target, 0.0, LINE).nodes(),
        "ArrowUp" => tab.scroll_tree.scroll_chain(target, 0.0, -LINE).nodes(),
        "ArrowLeft" => tab.scroll_tree.scroll_chain(target, -LINE, 0.0).nodes(),
        "ArrowRight" => tab.scroll_tree.scroll_chain(target, LINE, 0.0).nodes(),
        "PageDown" => tab.scroll_tree.scroll_chain(target, 0.0, page).nodes(),
        "PageUp" => tab.scroll_tree.scroll_chain(target, 0.0, -page).nodes(),
        // Shift+Space pages up, like browsers; plain Space pages down.
        " " => tab
            .scroll_tree
            .scroll_chain(target, 0.0, if shift { -page } else { page })
            .nodes(),
        // Home/End ride the same chaining path as the other scrolling
        // keys: an unbounded delta drives the target container to its
        // edge, the `user_scroll` mask keeps `overflow: hidden` axes
        // from consuming it (the old direct write scrolled hidden
        // containers, unlike the arrows), and the remainder chains
        // outward — End at the bottom of a nested scroller scrolls the
        // page, the way the arrows already did.
        "Home" => tab
            .scroll_tree
            .scroll_chain(target, 0.0, f32::NEG_INFINITY)
            .nodes(),
        "End" => tab
            .scroll_tree
            .scroll_chain(target, 0.0, f32::INFINITY)
            .nodes(),
        _ => Vec::new(),
    };
    sync_nested_paint(tab, &scrolled);
    scrolled
}

// ---------------------------------------------------------------------------
// Sequential focus navigation (pure)
// ---------------------------------------------------------------------------

/// Parsed `tabindex` attribute: `None` when absent or unparseable
/// (unparseable behaves as absent per HTML).
fn tab_index(doc: &Document, node: NodeId) -> Option<i32> {
    doc.get_attribute(node, "tabindex")?.trim().parse().ok()
}

/// Whether `node` participates in sequential focus navigation (the
/// WHATWG HTML "focusable" subset that Tab reaches):
/// - any rendered element with a non-negative `tabindex` (negative is
///   script-focusable only and skipped),
/// - `<a>`/`<area>` with `href`,
/// - `<button>`, `<input>` (except `type=hidden`), `<select>`,
///   `<textarea>` — unless `disabled`.
///
/// Mouse-press focus ([`press_at`]) is hit-test driven and can land on
/// anything with a box; this is the only enumeration of focusables, and
/// it walks the DOM in document order rather than the box tree.
fn is_sequentially_focusable(doc: &Document, node: NodeId) -> bool {
    let NodeKind::Element(data) = doc.kind(node) else {
        return false;
    };
    let tag = &data.tag_name;
    // Never focusable, `tabindex` or not: not rendered.
    if tag == "input"
        && matches!(doc.get_attribute(node, "type"), Some(t) if t.eq_ignore_ascii_case("hidden"))
    {
        return false;
    }
    if let Some(index) = tab_index(doc, node) {
        return index >= 0;
    }
    if tag == "a" || tag == "area" {
        return doc.has_attribute(node, "href");
    }
    if tag == "button" || tag == "select" || tag == "textarea" {
        return !doc.has_attribute(node, "disabled");
    }
    if tag == "input" {
        return !doc.has_attribute(node, "disabled");
    }
    false
}

/// Sequential focus navigation order (WHATWG HTML): elements with a
/// positive `tabindex` first in ascending order (stable on ties), then
/// the zero/`tabindex`-less focusables in document order. Negative
/// `tabindex` elements never appear.
fn focus_order(doc: &Document) -> Vec<NodeId> {
    let mut positive: Vec<(i32, usize, NodeId)> = Vec::new();
    let mut zero: Vec<NodeId> = Vec::new();
    for (i, node) in doc.descendants(doc.root()).iter().enumerate() {
        if !is_sequentially_focusable(doc, *node) {
            continue;
        }
        match tab_index(doc, *node) {
            Some(index) if index > 0 => positive.push((index, i, *node)),
            _ => zero.push(*node),
        }
    }
    // Stable: ascending `tabindex`, document order on ties.
    positive.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    positive
        .into_iter()
        .map(|(_, _, node)| node)
        .chain(zero)
        .collect()
}

/// Move keyboard focus one step through the sequential focus order and
/// reveal the newly focused element (`Nearest`/`Nearest`, instant — the
/// same [`reveal_node_with_options`](crate::scroll::ScrollTree::reveal_node_with_options)
/// path the `scrollIntoView()` drain uses, via
/// [`scroll_to_node`](crate::navigation::scroll_to_node)).
///
/// `backward` is Shift+Tab. Wraps at both ends, like browsers. Returns
/// the scrolled containers innermost-first (`None` = viewport), for
/// `scroll` event dispatch; empty when nothing moved (no focusables, or
/// the single focusable already has focus).
fn apply_focus_nav(input: &mut InputState, tab: &mut Tab, backward: bool) -> Vec<Option<NodeId>> {
    let order = match tab.document_ref() {
        Some(document) => focus_order(&document),
        None => return Vec::new(),
    };
    if order.is_empty() {
        return Vec::new();
    }
    let position = input
        .focused
        .and_then(|focused| order.iter().position(|&node| node == focused));
    let next = match position {
        Some(i) => {
            let len = order.len();
            order[if backward {
                (i + len - 1) % len
            } else {
                (i + 1) % len
            }]
        }
        // Focus sits outside the order (nothing focused, or the focused
        // node was removed): Tab starts at the first focusable,
        // Shift+Tab at the last, like browsers.
        None => order[if backward { order.len() - 1 } else { 0 }],
    };
    if Some(next) == input.focused {
        return Vec::new();
    }
    if let Some(old) = input.focused.take() {
        state_for(input, old).focus = false;
    }
    state_for(input, next).focus = true;
    input.focused = Some(next);
    let scrolled = crate::navigation::scroll_to_node(
        tab,
        next,
        crate::scroll::RevealAlign::Nearest,
        crate::scroll::RevealAlign::Nearest,
        bowser_js::ScrollBehavior::Auto,
    );
    sync_nested_paint(tab, &scrolled);
    scrolled
}

/// Resize the viewport to `(w, h)` CSS px, invalidating layout and paint
/// (style is unaffected by a pure size change in v1: no media queries).
fn apply_resize(tab: &mut Tab, w: f32, h: f32) {
    tab.viewport = Size { w, h };
    // A resize re-bases every box: full layout via the helper, so a
    // same-turn retained text subtree can never splice against the
    // old viewport's geometry and discard this dirt (F1).
    tab.mark_layout_dirty_full();
    tab.dirty.paint = true;
}

// ---------------------------------------------------------------------------
// Keyboard labels (pure)
// ---------------------------------------------------------------------------

/// Human-readable key name for dispatch: the typed character, or the
/// [`NamedKey`] debug name (`"Enter"`, `"F5"`, ...). `None` for keys with
/// no meaningful label.
fn key_label(key: &Key) -> Option<String> {
    match key {
        Key::Character(s) => Some(s.as_str().to_string()),
        Key::Named(named) => Some(format!("{named:?}")),
        Key::Unidentified(_) | Key::Dead(_) => None,
    }
}

// ---------------------------------------------------------------------------
// Partial (incremental) layout: size-stable subtree splice (wave 51)
// ---------------------------------------------------------------------------

/// Geometry tolerance for the partial path's size/position gates: the
/// subtree pass and the full pass run the same code, so a truly
/// size-stable subtree agrees to float noise; anything larger is a
/// real geometry change and takes the full path.
const PARTIAL_GEOMETRY_EPS: f32 = 0.01;

fn partial_approx(a: f32, b: f32) -> bool {
    (a - b).abs() <= PARTIAL_GEOMETRY_EPS
}

/// Child-index path from `root` to the first box (preorder) whose DOM
/// node is `node`, or `None` when no box carries it (display: contents
/// / none, text promoted into an ancestor's run, detached, ...).
fn find_box_path(root: &LayoutBox, node: NodeId) -> Option<Vec<usize>> {
    if root.node == Some(node) {
        return Some(Vec::new());
    }
    let mut stack: Vec<(&LayoutBox, Vec<usize>)> = vec![(root, Vec::new())];
    while let Some((box_, path)) = stack.pop() {
        for (i, child) in box_.children.iter().enumerate() {
            let mut child_path = path.clone();
            child_path.push(i);
            if child.node == Some(node) {
                return Some(child_path);
            }
            stack.push((child, child_path));
        }
    }
    None
}

fn box_at_path<'a>(root: &'a LayoutBox, path: &[usize]) -> &'a LayoutBox {
    let mut box_ = root;
    for &i in path {
        box_ = &box_.children[i];
    }
    box_
}

/// Number of boxes in a tree (the splice counter's unit).
fn count_boxes(root: &LayoutBox) -> usize {
    let mut count = 0;
    let mut stack = vec![root];
    while let Some(box_) = stack.pop() {
        count += 1;
        stack.extend(box_.children.iter());
    }
    count
}

/// Translate a whole box tree by `(dx, dy)`: border-box rects plus the
/// text fragments' rects and baselines (the only other absolute
/// geometry a [`LayoutBox`] carries).
fn translate_tree(root: &mut LayoutBox, dx: f32, dy: f32) {
    let mut stack = vec![root];
    while let Some(box_) = stack.pop() {
        box_.rect.x += dx;
        box_.rect.y += dy;
        for frag in &mut box_.fragments {
            frag.rect.x += dx;
            frag.rect.y += dy;
            frag.baseline += dy;
        }
        stack.extend(box_.children.iter_mut());
    }
}

fn value_has_viewport_unit(value: &CssValue) -> bool {
    match value {
        CssValue::Length(_, unit) => matches!(
            unit,
            LengthUnit::Vw | LengthUnit::Vh | LengthUnit::Vmin | LengthUnit::Vmax
        ),
        CssValue::List(items) => items.iter().any(value_has_viewport_unit),
        // Grid track structures: recurse into both `minmax()` sides
        // and every `repeat()` track (the pre-rework scan fell through
        // to `false` here, so `grid-template-columns: 1fr 25vw 100px`
        // passed the gate).
        CssValue::MinMax(min, max) => value_has_viewport_unit(min) || value_has_viewport_unit(max),
        CssValue::Repeat { tracks, .. } => tracks.iter().any(value_has_viewport_unit),
        // A raw `Math` tree cannot be inspected from here
        // (`bowser_css` does not export `MathExpr`), so fail closed:
        // any surviving math on a computed style is treated as
        // viewport-suspect. Raw `Math` is documented as never
        // surviving the cascade (it evaluates or defers as
        // `MathDeferred`, whose viewport leaves were already folded
        // against the real viewport into the `px` coefficient), so
        // this conservative arm costs nothing on real styles.
        CssValue::Math(_) => true,
        CssValue::Var(_, fallback) => fallback.as_deref().is_some_and(value_has_viewport_unit),
        CssValue::Shadows(layers) => layers.iter().any(|layer| {
            value_has_viewport_unit(&layer.offset_x)
                || value_has_viewport_unit(&layer.offset_y)
                || value_has_viewport_unit(&layer.blur)
                || value_has_viewport_unit(&layer.spread)
        }),
        CssValue::Transform(functions) => functions.iter().any(|f| match f {
            bowser_css::TransformFunction::Translate { x, y } => {
                value_has_viewport_unit(x) || value_has_viewport_unit(y)
            }
            _ => false,
        }),
        _ => false,
    }
}

/// Whether any geometry-bearing property of `style` uses a viewport
/// unit. Defense in depth: since the F3 structural fix the subtree
/// pass resolves viewport units against the real viewport (see
/// [`layout_subtree_with_intrinsics_and_scroll_and_offsets`]), so
/// this scan is no longer load-bearing — it stays as a cheap refusal
/// for a class the differential corpus pins as correct either way.
/// (`bowser_css` exposes no iterator over a computed style's present
/// properties — `COMPUTED_PROPS` is crate-private and
/// `ComputedStyle::get_by_index` panics past its length — so the scan
/// names the geometry-bearing properties explicitly; the list
/// includes the grid track, spacing, and column properties the
/// pre-rework scan omitted.)
fn style_has_viewport_unit(style: &ComputedStyle) -> bool {
    const PROPS: &[&str] = &[
        "width",
        "min-width",
        "max-width",
        "height",
        "min-height",
        "max-height",
        "margin-top",
        "margin-right",
        "margin-bottom",
        "margin-left",
        "padding-top",
        "padding-right",
        "padding-bottom",
        "padding-left",
        "border-top-width",
        "border-right-width",
        "border-bottom-width",
        "border-left-width",
        "top",
        "right",
        "bottom",
        "left",
        "font-size",
        "line-height",
        "flex-basis",
        "flex-grow",
        "flex-shrink",
        "gap",
        "row-gap",
        "column-gap",
        "grid-template-columns",
        "grid-template-rows",
        "grid-auto-columns",
        "grid-auto-rows",
        "grid-column-gap",
        "grid-row-gap",
        "text-indent",
        "letter-spacing",
        "word-spacing",
        "column-width",
        "column-gap",
        "border-spacing",
        "outline-offset",
        "translate",
        "inset-inline-start",
        "inset-inline-end",
        "inset-block-start",
        "inset-block-end",
    ];
    PROPS
        .iter()
        .any(|prop| style.get(prop).is_some_and(value_has_viewport_unit))
}

fn box_floats(box_: &LayoutBox) -> bool {
    box_.style.as_deref().is_some_and(|style| {
        matches!(
            style.get_str("float"),
            Some(v) if v.eq_ignore_ascii_case("left") || v.eq_ignore_ascii_case("right")
        )
    })
}

/// Whether `style`'s `width` is `auto` (or absent, which computes to
/// the initial `auto` on the properties read here).
fn width_is_auto(style: &ComputedStyle) -> bool {
    match style.get("width") {
        None => true,
        Some(CssValue::Keyword(k)) => k.eq_ignore_ascii_case("auto"),
        _ => false,
    }
}

/// Whether `style`'s `height` is `auto` (see [`width_is_auto`]).
fn height_is_auto(style: &ComputedStyle) -> bool {
    match style.get("height") {
        None => true,
        Some(CssValue::Keyword(k)) => k.eq_ignore_ascii_case("auto"),
        _ => false,
    }
}

/// Whether `style`'s sizing on `prop` (`width`/`min-width`/`max-width`)
/// is an intrinsic keyword (`fit-content`/`min-content`/`max-content`):
/// such a size is derived from the box's contents, so it can change
/// when a descendant's text does even at an unchanged containing
/// width.
fn has_intrinsic_size_keyword(style: &ComputedStyle, prop: &str) -> bool {
    matches!(
        style.get(prop),
        Some(CssValue::Keyword(k))
            if k.eq_ignore_ascii_case("fit-content")
                || k.eq_ignore_ascii_case("min-content")
                || k.eq_ignore_ascii_case("max-content")
    )
}

/// Whether `style`'s `min-width`/`min-height` (`prop`) is `auto` (or
/// absent, which computes to the initial `auto` — see
/// [`width_is_auto`]). On a flex or grid item this is the *automatic
/// minimum*: it computes to the min-content size and floors the used
/// size, so growing content can widen the box even when the main size
/// is definite.
fn min_size_is_auto(style: &ComputedStyle, prop: &str) -> bool {
    match style.get(prop) {
        None => true,
        Some(CssValue::Keyword(k)) => k.eq_ignore_ascii_case("auto"),
        _ => false,
    }
}

/// Whether `style`'s `flex-basis` is `auto` or `content` (or absent,
/// the initial `auto`): the flex base size then resolves from the
/// item's main size or its content rather than from a definite
/// length.
fn flex_basis_is_content_driven(style: &ComputedStyle) -> bool {
    match style.get("flex-basis") {
        None => true,
        Some(CssValue::Keyword(k)) => {
            k.eq_ignore_ascii_case("auto") || k.eq_ignore_ascii_case("content")
        }
        _ => false,
    }
}

/// Whether a flex/grid item's used main size can be content-driven
/// even when its main size property is definite. The automatic
/// minimum (`min-width`/`min-height: auto`, which computes to
/// min-content on flex and grid items) floors the used size at the
/// content size, and a content-driven `flex-basis` sizes the flex
/// base from the content — either lets growing text widen the item
/// at an unchanged containing width (the hole the wave-51 F2 probe
/// hit: `width: 200px` + `min-width: auto` + nowrap text). The
/// subtree pass replays the ancestor's old size as the containing
/// width and the splice gates only the candidate's size, so the
/// walk must refuse here. Conservative by design: over-refusing
/// costs one full pass, under-refusing splices a stale tree.
fn flex_grid_item_main_size_can_grow(style: &ComputedStyle, main_is_width: bool) -> bool {
    if if main_is_width {
        width_is_auto(style)
    } else {
        height_is_auto(style)
    } {
        return true;
    }
    let min_prop = if main_is_width {
        "min-width"
    } else {
        "min-height"
    };
    min_size_is_auto(style, min_prop) || flex_basis_is_content_driven(style)
}

/// Whether ancestor box `box_` (with box parent `parent`, `None` at
/// the tree root) is shrink-to-fit / intrinsic-sized, so a text edit
/// in a descendant can change *its* used size — a change the partial
/// path cannot see, because it only re-lays the candidate subtree and
/// only gates the *candidate's* size/position (wave 51 rework F2;
/// probe: a `float:left` wrapper around a `white-space:nowrap`
/// paragraph widened 19.2→307.2px in a full pass while the subtree
/// pass at the old width returned the paragraph unchanged and the
/// splice kept the stale float width).
///
/// Detectable classes, all read off the previous tree's
/// [`LayoutBox`]es (computed style + box type — the box carries no
/// `StyleData`, so these are the signals that exist app-side):
/// floated (style `float`); absolutely positioned with `auto` width
/// (style `position` + `width`); `display: inline-block`
/// ([`BoxType::InlineBlock`], blockified floats already caught by the
/// float arm); table, table-cell, table-row, and table-row-group
/// boxes (their widths come from table column sizing over all cells'
/// contents — box types); a flex or grid *item* whose main size can be
/// content-driven (auto main size, or a definite one with the
/// automatic minimum `min-width`/`min-height: auto` or a
/// content-driven `flex-basis` — the item's parent box is a flex
/// container — its style `display` is `flex`/`inline-flex` — with the
/// main axis from the parent's `flex-direction`, or a grid container —
/// [`BoxType::Grid`] or `display: grid`/`inline-grid` — where either
/// axis can be content-driven through track sizing); and an intrinsic
/// width keyword on
/// `width`/`min-width`/`max-width`. Anonymous ancestors carry no
/// style: their box types still expose the table classes, and an
/// anonymous block's sizing follows its (styled) parent formatting
/// context, whose own box is walked separately — no listed class is
/// *only* expressible on an anonymous box, so nothing in the list is
/// undetectable from the previous tree.
fn ancestor_is_shrink_to_fit(box_: &LayoutBox, parent: Option<&LayoutBox>) -> bool {
    if box_floats(box_) {
        return true;
    }
    if matches!(
        box_.box_type,
        bowser_layout::BoxType::InlineBlock
            | bowser_layout::BoxType::Table
            | bowser_layout::BoxType::TableCell
            | bowser_layout::BoxType::TableRow
            | bowser_layout::BoxType::TableRowGroup
    ) {
        return true;
    }
    let Some(style) = box_.style.as_deref() else {
        return false;
    };
    if has_intrinsic_size_keyword(style, "width")
        || has_intrinsic_size_keyword(style, "min-width")
        || has_intrinsic_size_keyword(style, "max-width")
    {
        return true;
    }
    if position_of(style) == Position::Absolute && width_is_auto(style) {
        return true;
    }
    // Flex/grid item whose main size can be content-driven: the item's
    // used size is decided by its container's item/track sizing over
    // its content, not by the containing width the partial path
    // replays. A definite main size is not enough to allow the pass:
    // the automatic minimum (`min-width`/`min-height: auto`) and a
    // content-driven `flex-basis` can still grow the box with its
    // content (see `flex_grid_item_main_size_can_grow`).
    if let Some(parent) = parent {
        let parent_display = parent
            .style
            .as_deref()
            .and_then(|s| s.get_str("display"))
            .unwrap_or("");
        if parent_display.eq_ignore_ascii_case("flex")
            || parent_display.eq_ignore_ascii_case("inline-flex")
        {
            let column = parent.style.as_deref().is_some_and(|s| {
                matches!(
                    s.get_str("flex-direction"),
                    Some(d) if d.eq_ignore_ascii_case("column")
                        || d.eq_ignore_ascii_case("column-reverse")
                )
            });
            if flex_grid_item_main_size_can_grow(style, !column) {
                return true;
            }
        }
        if parent.box_type == bowser_layout::BoxType::Grid
            || parent_display.eq_ignore_ascii_case("grid")
            || parent_display.eq_ignore_ascii_case("inline-grid")
        {
            // Track sizing can be content-based on either axis, so a
            // content-driven size on either axis refuses.
            if flex_grid_item_main_size_can_grow(style, true)
                || flex_grid_item_main_size_can_grow(style, false)
            {
                return true;
            }
        }
    }
    false
}

/// Conservative blockers for splicing a subtree laid out standalone:
/// any out-of-flow or sticky box inside it (their containing blocks /
/// constraint bands can live outside the subtree), a floated subtree
/// root (its placement is decided by its parent's float context), or a
/// viewport-unit length anywhere in it. Checked on both the previous
/// subtree (cheap pre-check) and the freshly laid-out one (a mutation
/// can introduce any of these).
///
/// Floated non-root descendants are deliberately not covered here:
/// character-data dirt cannot change any box's float state, so the
/// root-only check is enough on that path, while style-sourced dirt
/// gets the whole-subtree [`subtree_has_float`] check in
/// [`try_partial_layout`].
fn subtree_has_partial_blocker(root: &LayoutBox) -> bool {
    let mut stack = vec![root];
    while let Some(box_) = stack.pop() {
        if let Some(style) = box_.style.as_deref() {
            if matches!(
                position_of(style),
                Position::Absolute | Position::Fixed | Position::Sticky
            ) {
                return true;
            }
            if style_has_viewport_unit(style) {
                return true;
            }
        }
        stack.extend(box_.children.iter());
    }
    box_floats(root)
}

/// Whether any box in the subtree rooted at `root` (inclusive) is
/// floated. Only consulted for style-sourced dirt (see
/// [`try_partial_layout`]): a style change can add or remove a float
/// on any box in the candidate subtree, and a float's layout effects —
/// intrusion into following siblings' line boxes, float stacking —
/// reach outside the subtree. The standalone subtree pass cannot
/// reproduce those effects, and neither the rect gate nor
/// [`style_change_breaks_splice`] (which diffs only the candidate's
/// own style) sees them, so the splice must fail closed. Checked on
/// both the previous subtree (a removed float leaves stale wrapping
/// in the outside content) and the freshly laid-out one (an added
/// float is missing from the outside content's wrapping).
///
/// Character-data dirt provably cannot change any box's float state —
/// the box tree and every computed style are unchanged — so that path
/// keeps the cheaper root-only float check inside
/// [`subtree_has_partial_blocker`].
fn subtree_has_float(root: &LayoutBox) -> bool {
    let mut stack = vec![root];
    while let Some(box_) = stack.pop() {
        if box_floats(box_) {
            return true;
        }
        stack.extend(box_.children.iter());
    }
    false
}

/// Strict px length for the partial path's containing-block math:
/// `Some` only for px lengths and unitless zero; anything else
/// (percentages, em, calc) makes the geometry unprovable app-side and
/// the caller falls back to the full path.
fn partial_strict_px(value: Option<&CssValue>) -> Option<f32> {
    match value {
        Some(CssValue::Length(v, unit)) if *unit == LengthUnit::Px => Some(*v),
        Some(CssValue::Number(n)) if *n == 0.0 => Some(0.0),
        Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("thin") => Some(1.0),
        Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("medium") => Some(3.0),
        Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("thick") => Some(5.0),
        None => Some(0.0),
        _ => None,
    }
}

fn partial_used_border_px(
    style: &ComputedStyle,
    width_prop: &str,
    style_prop: &str,
) -> Option<f32> {
    let style_none = style.get(style_prop).is_some_and(|v| {
        matches!(v, CssValue::Keyword(k) if k.eq_ignore_ascii_case("none") || k.eq_ignore_ascii_case("hidden"))
    });
    if style_none {
        return Some(0.0);
    }
    partial_strict_px(style.get(width_prop))
}

/// The containing-block geometry a parent box offered its children:
/// `(content_origin_x, content_origin_y, content_width)`, or `None`
/// when any border/padding component is not a strict px length.
/// Anonymous boxes have no style and no decorations: their content box
/// is their border box.
fn parent_content_geometry(parent: &LayoutBox) -> Option<(f32, f32, f32)> {
    let Some(style) = parent.style.as_deref() else {
        return Some((parent.rect.x, parent.rect.y, parent.rect.w));
    };
    let bl = partial_used_border_px(style, "border-left-width", "border-left-style")?;
    let br = partial_used_border_px(style, "border-right-width", "border-right-style")?;
    let bt = partial_used_border_px(style, "border-top-width", "border-top-style")?;
    let pl = partial_strict_px(style.get("padding-left"))?;
    let pr = partial_strict_px(style.get("padding-right"))?;
    let pt = partial_strict_px(style.get("padding-top"))?;
    Some((
        parent.rect.x + bl + pl,
        parent.rect.y + bt + pt,
        parent.rect.w - bl - br - pl - pr,
    ))
}

/// Properties whose change on the splice candidate can move boxes
/// *outside* the candidate's subtree while leaving the candidate's own
/// border box (and box type) identical — the splice validation below
/// cannot see them, so any such change fails closed to the full path:
///
/// * `display`: `block` → `inline-block` with an identical rect changes
///   in-flow sequencing of following siblings.
/// * `position`: relative offsets and out-of-flow promotion are not all
///   visible in the compared border box.
/// * `float`, `clear`: in-flow sequencing around floats.
/// * `margin-*`: negative margins and margin collapsing move siblings
///   and ancestors without moving the candidate's own border box.
/// * `vertical-align`: moves line-mates of an inline-block candidate.
/// * `font-size`, `font-family`, `font-weight`, `font-style`: these
///   anchor `em`/`ex`/`ch`-relative *used* values of the properties
///   above (computed values keep relative units unresolved), so a font
///   change can move a sibling through an `em`-relative margin while
///   every blacklisted computed value above still compares equal.
/// * `break-before`, `break-after`, `break-inside`: fragmentation
///   breaks re-place boxes outside the candidate — e.g. a
///   `break-after: column` on the candidate pushes its following
///   siblings into the next column — while the candidate's own border
///   box is unchanged, so the rect gate below cannot see the change.
///
/// Everything else either only affects painting (`color`,
/// `background-*`, `opacity`, `z-index`, `transform`, `visibility` —
/// transforms never move other boxes' layout), only affects the
/// subtree's interior — which the subtree pass rebuilds from the
/// current DOM (`line-height`, `text-align`, flex/grid properties,
/// `content`, `overflow`) — or changes the candidate's own size, which
/// the rect validation below catches (`width`, `height`, `padding-*`,
/// `border-*`).
///
/// Only consulted for style-sourced dirt: for character-data dirt the
/// candidate's style provably did not change. When the candidate is a
/// proper ancestor of the dirty root its style provably did not change
/// either (the restyle scope is the dirty root's subtree), so this is a
/// no-op there; it bites exactly when the style mutation may have
/// changed the candidate's own style.
///
/// A missing style on either side fails closed: without both computed
/// styles there is nothing trustworthy to compare.
///
/// Residual (documented, accepted): exotic relative units anchored to
/// something other than these properties (e.g. `lh`/`rlh`, should they
/// ever be supported) could move a sibling through an external
/// property without tripping this check or the rect check. That needs
/// used-value comparison — a future audit.
fn style_change_breaks_splice(old: Option<&ComputedStyle>, new: Option<&ComputedStyle>) -> bool {
    const EXTERNAL_EFFECT: &[&str] = &[
        "display",
        "position",
        "float",
        "clear",
        "break-before",
        "break-after",
        "break-inside",
        "margin-top",
        "margin-right",
        "margin-bottom",
        "margin-left",
        "vertical-align",
        "font-size",
        "font-family",
        "font-weight",
        "font-style",
    ];
    let (Some(old), Some(new)) = (old, new) else {
        return true;
    };
    EXTERNAL_EFFECT.iter().any(|p| old.get(p) != new.get(p))
}

/// Try the partial layout path for `tab`'s retained dirty report:
/// re-lay out the single dirtied subtree standalone and splice it into
/// the previous layout tree in place. Returns `true` when the splice
/// happened (the caller then runs the normal post-layout steps);
/// `false` on *any* doubt, in which case the tab is left exactly as
/// it was found — including the JS bindings' published share of the
/// previous tree, which the unique-ownership step below drops and
/// every post-drop failure return re-publishes (F6) — and the caller
/// takes the full path.
///
/// Gates (each one is a full-path fallback, per the wave-51 brief):
/// the report forced the full fallback; the report holds other than
/// exactly one coalesced root; style dirt is present but not every
/// style-affecting mutation was a subtree-localized attribute write on
/// the single root, or some stylesheet uses sibling combinators or
/// `:has()` (see `bowser_css::selectors_are_subtree_local`); the tab's
/// live
/// viewport differs from the viewport the previous tree was laid out
/// at (F1 belt-and-braces, see [`Tab::last_layout_viewport`]); there
/// is no previous tree; the root resolves to the document element /
/// `body` / the tree root; the root has no previous box (or only an
/// inline-level one — the search then promotes to the nearest
/// block-level ancestor box, e.g. a text edit inside a `<span>`
/// re-lays its `<p>`); the subtree (old or new) contains an
/// out-of-flow, sticky, floated-root, or viewport-unit box (see
/// [`subtree_has_partial_blocker`]); for style-sourced dirt, any
/// floated box anywhere in the old or new candidate subtree (see
/// [`subtree_has_float`]); any ancestor of the root is
/// shrink-to-fit / intrinsic-sized (see
/// [`ancestor_is_shrink_to_fit`], F2); a preceding sibling floats (its
/// intrusion into the subtree's line boxes is decided outside it);
/// the root's height is percentage-based (the subtree pass's
/// containing-block height is the viewport's, not the parent's); the
/// parent's content geometry is not strict-px provable; the previous
/// tree cannot be made uniquely owned; or the new subtree's
/// border-box size or containing-block-relative position differs from
/// the old box's — a size/position change can move every following
/// sibling, which only a full pass can re-place.
///
/// Style-dirt re-audit (each gate above was re-examined for a
/// style-mutation-sourced root; all pass unchanged): the restyle scope
/// is confined to the dirty subtree by the style gate, so every box
/// outside the rebuilt subtree keeps its computed style — the
/// shrink-to-fit-ancestor (F2), floated-preceding-sibling, and
/// percentage-height gates, which all reason about boxes outside the
/// subtree, see the same styles as the full pass did. The out-of-flow /
/// sticky / viewport-unit blockers are re-checked on the *new* box as
/// well as the old, so a style change that introduces one fails
/// closed; floats get a stricter whole-subtree rule — any float in the
/// old or new candidate subtree refuses the splice (see
/// [`subtree_has_float`]) — because a float's layout effects (line-box
/// intrusion into following content, float stacking) reach outside the
/// subtree while leaving the candidate's own rect and style unchanged.
/// The one genuinely new hazard — a style change to the candidate's
/// own external-effect properties (margins, display, break-after, ...)
/// that leaves its rect identical — is caught by
/// [`style_change_breaks_splice`] in the validation below.
fn try_partial_layout(
    render: &mut RenderPipeline,
    doc: &RefCell<Document>,
    ctx: Option<u32>,
    fonts: &Mutex<FontCache>,
) -> bool {
    if render.dirty_full_fallback || render.dirty_subtrees.len() != 1 {
        return false;
    }
    // Style dirt used to force the full path on its own: any attribute
    // write re-ran the style stage over the whole document. Attribute
    // writes now record their restyle scope (`Document::set_attribute`
    // → `DirtyReport::style_subtrees`), so when every style-affecting
    // mutation in the batch was such a write confined to the single
    // dirty root — the localized roots coalesce to exactly the retained
    // root — and no stylesheet can observe the mutation from outside
    // that subtree (see `bowser_css::selectors_are_subtree_local`), the
    // subtree pass below re-matches the subtree from the current DOM
    // and the splice is as sound as for character-data dirt. Other
    // style dirt — text emptiness toggles, shadow DOM writes,
    // sibling-combinator or `:has()` stylesheets, and structural
    // changes on a root with no localized write — fails closed to the
    // full path, exactly as before: such dirt sets `dirty.style`
    // without recording a `style_subtrees` entry, so the root-set
    // equality below refuses. Note the gate compares root *sets*, not
    // dirt provenance: a structural mutation sharing its coalesced
    // root with a localized attribute write (same turn, or across
    // retained syncs) satisfies the equality and takes the splice
    // path — the subtree pass rebuilds from the current DOM, so the
    // structural change itself is serviced correctly.
    //
    // The stylesheet scan runs only on this path (a style-dirty,
    // single-root refresh): it is linear in the rule count and never on
    // a hot frame.
    let style_sourced = render.dirty.style;
    if style_sourced {
        if render.style_subtrees != render.dirty_subtrees {
            return false;
        }
        let author: Vec<&Stylesheet> = render.author.iter().collect();
        if !selectors_are_subtree_local(ua_stylesheet(), &author) {
            return false;
        }
    }
    // Belt-and-braces (F1): the previous tree was laid out at
    // `last_layout_viewport` by the last full pass; a live viewport
    // that differs (a resize whose dirt somehow reached here without
    // forcing the full fallback) invalidates every retained subtree.
    // `None` (no full pass recorded for this tree) refuses too.
    if render.last_layout_viewport != Some(render.viewport) {
        return false;
    }
    // ---- Plan against the document and the previous tree (shared).
    let plan = (|| {
        let old_root = render.layout_root.as_deref()?;
        let doc = doc.borrow();
        let raw = render.dirty_subtrees[0];
        // Candidate roots, nearest first: a text node's parent element
        // (text boxes live in their parent's formatting context), the
        // element itself for element dirt, then element ancestors for
        // the inline/display:contents promotion below.
        let start = match doc.kind(raw) {
            NodeKind::Text(_) => doc.parent(raw)?,
            NodeKind::Element(_) => raw,
            _ => return None,
        };
        let mut candidates = vec![start];
        let mut cursor = doc.parent(start);
        while let Some(parent) = cursor {
            if matches!(doc.kind(parent), NodeKind::Element(_)) {
                candidates.push(parent);
            }
            cursor = doc.parent(parent);
        }
        for candidate in candidates {
            let NodeKind::Element(data) = doc.kind(candidate) else {
                continue;
            };
            if data.tag_name == "html" || data.tag_name == "body" {
                return None;
            }
            let Some(path) = find_box_path(old_root, candidate) else {
                continue;
            };
            if path.is_empty() {
                return None;
            }
            let target_box = box_at_path(old_root, &path);
            if !matches!(
                target_box.box_type,
                bowser_layout::BoxType::Block
                    | bowser_layout::BoxType::InlineBlock
                    | bowser_layout::BoxType::Table
                    | bowser_layout::BoxType::Grid
            ) {
                continue;
            }
            // Percentage heights resolve against the parent's height
            // in a full pass but the viewport's in a subtree pass.
            if let Some(style) = target_box.style.as_deref()
                && ["height", "min-height", "max-height"]
                    .iter()
                    .any(|p| matches!(style.get(p), Some(CssValue::Percentage(_))))
            {
                return None;
            }
            if subtree_has_partial_blocker(target_box) {
                return None;
            }
            // Style-sourced dirt can add or remove a float on any box
            // in the candidate subtree (see `subtree_has_float`): the
            // previous tree may carry float wrapping the new subtree
            // no longer has, or vice versa. Character-data dirt cannot
            // change float state, so it keeps the cheaper root-only
            // check above.
            if style_sourced && subtree_has_float(target_box) {
                return None;
            }
            // F2: no ancestor of the candidate may be shrink-to-fit /
            // intrinsic-sized. The subtree pass replays the
            // candidate's old containing width and the splice gates
            // only the candidate's own size/position, so an ancestor
            // whose used size is derived from its contents (a float
            // wrapping the candidate, an abspos box with auto width,
            // a table cell, ...) would keep its stale size while its
            // content changed underneath it.
            for depth in 0..path.len() {
                let ancestor = box_at_path(old_root, &path[..depth]);
                let ancestor_parent =
                    (depth > 0).then(|| box_at_path(old_root, &path[..depth - 1]));
                if ancestor_is_shrink_to_fit(ancestor, ancestor_parent) {
                    return None;
                }
            }
            let parent_path = &path[..path.len() - 1];
            let parent_box = box_at_path(old_root, parent_path);
            // A floated preceding sibling intrudes into the target's
            // line boxes from outside the subtree: not reproducible
            // standalone.
            let target_index = *path.last()?;
            if parent_box.children[..target_index].iter().any(box_floats) {
                return None;
            }
            let (cb_x, cb_y, containing_width) = parent_content_geometry(parent_box)?;
            return Some((
                candidate,
                path,
                target_box.rect,
                target_box.style.clone(),
                cb_x,
                cb_y,
                containing_width,
            ));
        }
        None
    })();
    let Some((target, path, old_rect, old_style, cb_x, cb_y, containing_width)) = plan else {
        return false;
    };

    // ---- The splice mutates the previous tree in place, so it must
    // be uniquely owned. The only other strong owner is the JS
    // bindings' published share; drop it (the post-layout publish
    // re-shares the spliced tree before any script can run). F6:
    // dropping it is a mutation of observable state (the JS geometry
    // getters read the published share), so every failure return
    // after this point re-publishes the untouched previous tree
    // first — a failed partial attempt leaves the tab exactly as it
    // found it.
    let mut dropped_share = false;
    if render
        .layout_root
        .as_ref()
        .is_some_and(|rc| Rc::strong_count(rc) > 1)
        && let Some(ctx) = ctx
    {
        bowser_js::clear_layout_tree(ctx);
        dropped_share = true;
    }
    if render
        .layout_root
        .as_ref()
        .is_some_and(|rc| Rc::strong_count(rc) != 1)
    {
        if dropped_share {
            publish_layout_tree_parts(ctx, &render.layout_root);
        }
        return false;
    }

    // ---- Lay out only the subtree, mirroring the full call's context.
    let nested_offsets = render.scroll_tree.nested_offsets();
    let new_box = {
        let doc = doc.borrow();
        let author: Vec<&Stylesheet> = render.author.iter().collect();
        let fonts = fonts.lock().expect("font cache lock poisoned");
        let metrics = SwashFontMetrics::new(&fonts);
        layout_subtree_with_intrinsics_and_scroll_and_offsets(
            &doc,
            &author,
            render.viewport,
            &metrics,
            &render.image_intrinsics,
            render.scroll().offset(),
            &nested_offsets,
            target,
            containing_width,
        )
    };
    let Some(new_box) = new_box else {
        if dropped_share {
            publish_layout_tree_parts(ctx, &render.layout_root);
        }
        return false;
    };

    // ---- Validate: same node, same size, same position relative to
    // the containing block the full pass gave it — and, for
    // style-sourced dirt, no change to a property with layout effects
    // outside the candidate's border box (see
    // `style_change_breaks_splice`).
    if new_box.node != Some(target)
        || subtree_has_partial_blocker(&new_box)
        || (style_sourced
            && (subtree_has_float(&new_box)
                || style_change_breaks_splice(old_style.as_deref(), new_box.style.as_deref())))
        || !partial_approx(new_box.rect.w, old_rect.w)
        || !partial_approx(new_box.rect.h, old_rect.h)
        || !partial_approx(new_box.rect.x, old_rect.x - cb_x)
        || !partial_approx(new_box.rect.y, old_rect.y - cb_y)
    {
        if dropped_share {
            publish_layout_tree_parts(ctx, &render.layout_root);
        }
        return false;
    }
    let spliced = count_boxes(&new_box);
    let mut new_box = new_box;
    let (dx, dy) = (old_rect.x - new_box.rect.x, old_rect.y - new_box.rect.y);
    translate_tree(&mut new_box, dx, dy);

    // ---- Splice in place of the old box.
    let Some(rc) = render.layout_root.as_mut() else {
        if dropped_share {
            publish_layout_tree_parts(ctx, &render.layout_root);
        }
        return false;
    };
    let Some(root_box) = Rc::get_mut(rc) else {
        if dropped_share {
            publish_layout_tree_parts(ctx, &render.layout_root);
        }
        return false;
    };
    let mut slot = root_box;
    for &i in &path {
        slot = &mut slot.children[i];
    }
    *slot = new_box;
    render.boxes_spliced += spliced as u64;
    true
}

// ---------------------------------------------------------------------------
// Pipeline refresh (pure)
// ---------------------------------------------------------------------------

// The DisplayListBuilder reused across display-list builds (see
// build_display_list_reused): retaining it kills the per-frame
// Vec<DisplayItem> regrow ramp and keeps the gradient/filter parse caches
// warm across frames (epoch invalidation on restyle - see
// DisplayListBuilder::begin_reused_build).
//
// One builder serves every tab: the caches are keyed by their complete
// resolution inputs (gradient source text; filter text + length context +
// currentcolor), so entries are never document-dependent and sharing only
// improves hit rates. The recycled *list* storage stays per-tab
// (tab.display_list is taken and rebuilt in place each frame).
//
// Thread-local because `refresh` runs on the window worker thread; tests
// each get a fresh builder on first use, which is fine (behavior is
// identical to a fresh builder per build).
thread_local! {
    static REUSED_BUILDER: std::cell::RefCell<Option<DisplayListBuilder>> =
        const { std::cell::RefCell::new(None) };
}

/// Re-run the pipeline stages whose dirty flags are set, then clear the
/// flags serviced.
///
/// * style or layout dirty: rebuild the box tree with
///   [`layout_with_intrinsics_and_scroll_and_offsets`] (matching +
///   cascade happen inside it; there is no separate style-only pass
///   yet). The tab's current scroll offset goes into layout, and each
///   nested scroll container's live offset rides along in the
///   `nested_offsets` map (keyed by the scroller's DOM node — see
///   [`scroll::ScrollTree::nested_offsets`]) — that is what makes
///   `position: sticky` stick, in the viewport and inside nested
///   scrollers alike (see `bowser_layout::layout_with_scroll`). Layout
///   always invalidates paint.
/// * paint dirty: rebuild the display list from the layout tree, resolving
///   text-run fonts through the window's [`FontCache`] and `<img>`/CSS
///   image URLs through the loader's decoded-image cache (same `ImageId`
///   for every use of one `src`).
/// * scrollbar rects stale but nothing else dirty (viewport scrolled
///   without a rebuild): truncate the display list back to the
///   recorded scrollbar split and re-push just the bar items — the
///   content items stay valid because the viewport rides the GPU
///   uniform.
///
/// After a layout rebuild the scroll containers are re-discovered and
/// their maxima republished into the tab's shared scroll states
/// ([`scroll::rebuild_after_layout`]), so the wheel, keyboard,
/// fragment, and JS scroll paths all clamp against the same bounds and
/// a shrunken document cannot keep a stale, overscrolled offset.
///
/// `fonts` is the window's font cache, shared with the paint backend (see
/// [`init_gpu`]). Text *measures* with [`SwashFontMetrics`] over that same
/// cache — real shaped advances (kerning, ligatures) from the faces paint
/// resolves — so laid-out line breaks and fragment widths agree with the
/// painted glyphs.
/// Style + layout + scroll-tree rebuild + publish stages of [`refresh`],
/// factored so a geometry getter can force them synchronously
/// ([`bowser_js::LayoutEnsurer`]): folds the document's dirt, re-runs
/// the dirty stages, and republishes the tree. Paint stays dirty for the
/// next frame (the Blink forced-sync-layout contract); this never runs
/// script.
///
/// The `render`/`doc`/`ctx` split (instead of `&Tab`) is the
/// disjoint-borrow contract: a page re-entry holds `&mut
/// RenderPipeline` for this while the worker holds `&mut` the installed
/// bundle's page VM, and the borrow checker proves them disjoint.
///
/// Does NOT sync dirty flags from the document: callers sync beforehand
/// (pumps call `sync_dirty_flags` before `refresh`; the ensurer syncs
/// before forcing). This preserves the base `refresh` contract where
/// tests manage the flags manually.
///
/// Returns `true` when a style/layout pass ran (the published tree may
/// have changed).
pub(crate) fn refresh_style_layout(
    render: &mut RenderPipeline,
    doc: &RefCell<Document>,
    ctx: Option<u32>,
    fonts: &Mutex<FontCache>,
) -> bool {
    if !(render.dirty.style || render.dirty.layout) {
        return false;
    }
    // Up to two passes. The first bakes sticky constraints at the
    // pre-layout offsets; the rebuild then publishes script-stashed
    // scroll writes (`pending_script`: written to not-yet-published
    // scrollers), clamped into the freshly discovered ranges. When the
    // live offsets moved under the bake, a second pass re-bakes at the
    // live offsets. The stash is consumed by the first rebuild and
    // resnap is idempotent, so the second pass always converges.
    for _ in 0..2 {
        let offsets_before = render.scroll_tree.nested_offsets();
        // Incremental path first (wave 51 step 2): when the retained
        // dirty report names exactly one subtree and every gate in
        // `try_partial_layout` passes, splice a standalone re-layout of
        // that subtree into the previous tree and run only the normal
        // post-layout steps. Any gate failure leaves the tab untouched
        // and the full path below runs exactly as before.
        if try_partial_layout(render, doc, ctx, fonts) {
            render.layout_partial_passes += 1;
            // The spliced tree is a new tree for scroll purposes:
            // re-discover containers, republish maxima, re-bake the
            // paint tree (see the full path below for why this is not
            // skipped: scroll correctness reads the whole tree).
            scroll::rebuild_after_layout(render, doc);
            publish_layout_tree_parts(ctx, &render.layout_root);
            render.clear_dirty_report();
            render.dirty.style = false;
            render.dirty.layout = false;
            render.dirty.paint = true;
        } else {
            let root = {
                let doc = doc.borrow();
                let author: Vec<&Stylesheet> = render.author.iter().collect();
                // Layout measures text with the window's font cache
                // (`SwashFontMetrics`): real shaped advances from the faces
                // paint resolves, so line-breaking is decided with the widths
                // paint will use. The guard is scoped to the layout call; the
                // paint stage below re-locks for display-list building.
                //
                // The scroll offset goes into layout (not zero): the sticky
                // constraints are baked for the offset at layout time. The
                // nested containers' live offsets ride along too, keyed by
                // each scroller's DOM node: the scroll tree already keys
                // containers by [`NodeId`], so [`ScrollTree::nested_offsets`]
                // needs no translation. The shared per-element scroll states
                // survive relayout, so the pre-layout tree's offsets are the
                // live ones for the layout being computed; containers at
                // zero are omitted (layout constrains absent entries at
                // zero).
                //
                // Between layouts a scroll would leave a stuck box lagging:
                // wheel scrolling rides the GPU uniform and nested scrolling
                // folds into the paint tree, neither re-baking the sticky
                // constraints. So every scroll path dirties layout when a
                // sticky box is within its constraint band of the new
                // offset (see [`dirty_layout_for_sticky`]) and the next
                // `refresh` re-bakes the constraints at the live offsets.
                // Pages without sticky boxes — and sticky boxes far from
                // the viewport — keep the no-layout fast path.
                //
                // A per-frame sticky re-evaluation without layout was
                // considered and rejected: the constraint needs each sticky
                // box's in-flow position and its containing block's
                // definiteness, both baked over by layout and unrecoverable
                // from the [`LayoutBox`] tree, so an app-side recompute
                // could not be exact at stick/unstick transitions.
                let nested_offsets = render.scroll_tree.nested_offsets();
                let fonts = fonts.lock().expect("font cache lock poisoned");
                let metrics = SwashFontMetrics::new(&fonts);
                layout_with_intrinsics_and_scroll_and_offsets(
                    &doc,
                    &author,
                    render.viewport,
                    &metrics,
                    &render.image_intrinsics,
                    render.scroll().offset(),
                    &nested_offsets,
                )
            };
            render.layout_root = Some(Rc::new(root));
            // Record the viewport this tree was laid out at: the
            // partial path's belt-and-braces gate compares the live
            // viewport against it (F1).
            render.last_layout_viewport = Some(render.viewport);
            render.layout_full_passes += 1;
            // New tree: re-discover the scroll containers, publish the
            // per-element and document maxima (offsets survive,
            // re-clamped), and re-bake the paint tree the display list
            // builds from.
            scroll::rebuild_after_layout(render, doc);
            // Publish the fresh tree to the page's web APIs so the
            // CSSOM View geometry getters measure the current layout.
            publish_layout_tree_parts(ctx, &render.layout_root);
            // The retained dirty report is serviced by this pass.
            render.clear_dirty_report();
            render.dirty.style = false;
            render.dirty.layout = false;
            render.dirty.paint = true;
        }
        if render.scroll_tree.nested_offsets() == offsets_before {
            break;
        }
        // The rebuild moved a container's offset under the bake (a
        // stashed script write clamped on publish, or a resnap):
        // re-bake once at the live offsets.
        render.dirty.layout = true;
        render.dirty_full_fallback = true;
    }
    true
}

fn refresh(tab: &mut Tab, fonts: &Mutex<FontCache>) {
    if !tab.has_document() {
        tab.dirty = DirtyFlags::default();
        tab.clear_dirty_report();
        return;
    }
    // Cache-epoch signal for the reused display-list builder (see
    // `REUSED_BUILDER`): captured before the style stage below clears
    // the flag. Every restyle in this pipeline also rebuilds the
    // display list in the same tick (`dirty.paint` is set alongside),
    // so the flag is exact — a scroll-only frame keeps warm caches.
    let style_changed = tab.dirty.style;
    // Style + layout + scroll-tree rebuild + publish, factored so a
    // geometry getter can force the same stages synchronously (see
    // `refresh_style_layout`): the disjoint `&mut render` / installed
    // split is what the reflow entry point reuses.
    {
        let (installed, render) = tab.split_for_reentry();
        let installed = installed.as_ref().expect("document checked above");
        let ctx = installed.page_vm().web_api_context();
        refresh_style_layout(render, installed.document(), ctx, fonts);
    }
    if tab.dirty.paint {
        // The display list builds from the paint tree: the layout tree
        // with nested scroll offsets baked in, so nested scrolling moves
        // content without a relayout. Before the first bake it falls back
        // to the layout tree.
        //
        // Borrow granularity: `root` borrows `tab.render` (shared) while
        // `display_list` below needs `&mut tab.display_list` — disjoint
        // direct fields, so name `tab.render` explicitly instead of
        // going through the `Deref` (which would borrow all of `*tab`).
        let root = tab
            .render
            .paint_root
            .as_ref()
            .or(tab.render.layout_root.as_deref());
        if let Some(root) = root {
            // No-clone discipline: the tab owns the sole strong `Rc`,
            // so the document is reached through the scoped
            // `with_document` handle — the strong handle lives only for
            // the closure call and can never be stored, so the tab stays
            // the only persistent owner and closing the tab still frees
            // the DOM. The display list is built inside the closure and
            // returned; `display_list` is assigned after the borrow ends.
            // Recycle the previous frame's list storage into the build:
            // `take` must happen outside `with_document` (the closure
            // runs under `installed()`'s shared borrow of the tab). The
            // retained builder (see `REUSED_BUILDER`) is reused across
            // frames; `style_changed` drives its cache epoch.
            let mut list = tab.display_list.take().unwrap_or_default();
            // Borrow the URL after the `take`: the mutable borrow ends
            // there, so no per-frame `Url` clone is needed.
            let url = tab.current_url();
            let display_list = tab
                .installed()
                .expect("document checked above")
                .with_document(|cell| {
                    let doc = cell.borrow();
                    let mut images = ImageIdMap::new(&doc, url);
                    let mut resolve_images =
                        |src: &str| -> Option<(ImageId, DecodedImage)> { images.resolve(src) };
                    let mut fonts = fonts.lock().expect("font cache lock poisoned");
                    let mut resolve_fonts = |style: &ComputedStyle| -> FontKey {
                        font_key_for_style(&mut fonts, style)
                    };
                    REUSED_BUILDER.with(|cell| {
                        let mut slot = cell.borrow_mut();
                        let builder = slot.get_or_insert_with(DisplayListBuilder::new);
                        build_display_list_reused(
                            builder,
                            &mut list,
                            root,
                            &doc,
                            &mut resolve_fonts,
                            &mut resolve_images,
                            tab.render.viewport,
                            style_changed,
                        );
                    });
                    list
                })
                .expect("tab owns document");
            tab.display_list = Some(display_list);
            // Wrap each eligible nested container's scrolling content in
            // a `PushTransform`/`PopTransform` pair before the epilogue
            // split is recorded: scroll ticks then patch the transform
            // instead of rebuilding (see
            // [`ScrollTree::wrap_scroll_transforms`]).
            //
            // Borrow granularity: `root` borrows `tab.render.paint_root` /
            // `tab.render.layout_root`, `list` borrows `tab.display_list`
            // mutably — so the scroll tree and viewport go through their
            // own disjoint field projections, never a whole-`render`
            // borrow.
            if let Some(list) = tab.display_list.as_mut() {
                tab.render.scroll_tree.wrap_scroll_transforms(root, list);
                tab.render.scroll_tree.set_glow_split(list.items.len());
                // Overscroll edge glow (phase 2): appended after the
                // display list is built from the baked paint tree, so
                // the glow paints above scrolled content; each glow is
                // wrapped in a PushClip/PopClip pair at its scrollport
                // (see `append_overscroll_glow`), so it never bleeds
                // outside. Scrollbars paint above all content: solid
                // track + thumb rects from the scroll tree's live
                // geometry, in page coordinates like every other
                // display-list item. The backend subtracts the viewport
                // scroll for them too, so the viewport's own bars stay
                // glued to the viewport edge while nested bars ride their
                // containers' painted boxes. Bars are always appended
                // last (after the overscroll glow); the splits are
                // recorded so a scroll tick or overscroll spring tick can
                // truncate back to the glow split and re-push just the
                // epilogue without a full rebuild (see the epilogue
                // branch below).
                let viewport = tab.render.viewport;
                tab.render
                    .scroll_tree
                    .append_overscroll_glow(list, viewport);
                for geom in tab.render.scroll_tree.scrollbar_geoms() {
                    if let Some(v) = geom.vertical {
                        push_scrollbar(list, v.track, v.thumb);
                    }
                    if let Some(h) = geom.horizontal {
                        push_scrollbar(list, h.track, h.thumb);
                    }
                }
                tab.render.scroll_tree.clear_scrollbars_dirty();
                tab.render.scroll_tree.clear_glow_dirty();
            }
        }
        tab.dirty.paint = false;
    } else if tab.render.scroll_tree.scrollbars_dirty() || tab.render.scroll_tree.glow_dirty() {
        // Scroll since the last rebuild, or an overscroll spring tick:
        // the content items are still valid (the viewport rides the GPU
        // uniform, nested containers ride their patched scroll
        // transforms), but the scrollbar rects and/or glow items are
        // stale. Truncate back to the recorded glow split and re-push
        // just the epilogue (glow, then bars) — no full rebuild, so the
        // fast paths survive. Hit-testing already uses the live offsets,
        // so this re-syncs paint with hit-test.
        //
        // Borrow granularity (see above): `list` borrows
        // `tab.display_list` mutably, so the scroll tree and viewport go
        // through `tab.render`, disjoint from it.
        match tab.display_list.as_mut() {
            Some(list) => {
                let render = &mut tab.render;
                let viewport = render.viewport;
                render.scroll_tree.refresh_epilogue(list, viewport);
            }
            None => {
                tab.render.scroll_tree.clear_scrollbars_dirty();
                tab.render.scroll_tree.clear_glow_dirty();
            }
        }
    }
}

/// Publish the tab's freshly laid-out box tree to the page's web APIs
/// ([`bowser_js::share_layout_tree`]): the CSSOM View geometry getters
/// (`offsetWidth`/`offsetHeight`/`clientWidth`/`clientHeight`/
/// `offsetLeft`/`offsetTop`/`offsetParent`) measure against this tree.
/// Called after every layout pass ([`refresh`]); until the first publish
/// — and when the web APIs never installed, so there is no context to
/// publish to — the getters report 0/`null`, the honest no-layout
/// answer. When a context exists but the tab has no laid-out tree yet,
/// any previously published tree is cleared ([`bowser_js::clear_layout_tree`])
/// so the getters cannot measure a stale tree from a prior document.
#[cfg(test)]
fn publish_layout_tree(tab: &Tab) {
    publish_layout_tree_parts(
        tab.page_vm().and_then(|vm| vm.web_api_context()),
        &tab.layout_root,
    );
}

/// [`publish_layout_tree`] over explicit parts: the synchronous reflow a
/// geometry getter triggers ([`bowser_js::LayoutEnsurer`]) has the web-API
/// context and the render pipeline but not the [`Tab`].
fn publish_layout_tree_parts(ctx: Option<u32>, layout_root: &Option<Rc<LayoutBox>>) {
    let Some(ctx) = ctx else {
        return;
    };
    let Some(root) = layout_root.as_ref() else {
        // A context exists but nothing has been laid out yet: clear any
        // previously published tree so the geometry getters measure the
        // no-layout answer (0/`null`) instead of a dead tree.
        bowser_js::clear_layout_tree(ctx);
        return;
    };
    // Zero-copy publish: `layout_root` already holds the tree as
    // `Rc<LayoutBox>`, so sharing it with the bindings is one refcount
    // bump — no deep clone per layout pass.
    bowser_js::share_layout_tree(ctx, Rc::clone(root));
}

/// Resolve a text run's font through the window's [`FontCache`].
///
/// Delegates to `bowser_paint::render`'s `css_font_*` extraction so the
/// family/weight/style reading lives in exactly one place; behavior is
/// identical to the former local copy.
///
/// Falls back to [`FontKey(0)`] when the system has no matching face
/// (fontless CI).
pub fn font_key_for_style(fonts: &mut FontCache, style: &ComputedStyle) -> FontKey {
    fonts
        .resolve(
            css_font_family(style),
            css_font_weight(style),
            css_font_style(style),
        )
        .unwrap_or(FontKey(0))
}

// ---------------------------------------------------------------------------
// Image identity (shared by the windowed and headless paint paths)
// ---------------------------------------------------------------------------

/// Per-`src` [`ImageId`] minting for one document's display-list build,
/// resolving through the loader's decoded-image cache.
///
/// Both display-list builders — the windowed [`refresh`] path here and the
/// headless dump (`render_headless_pixels` in main.rs) — need one
/// `ImageId` per distinct `src`. This is the shared implementation both
/// paths use.
pub struct ImageIdMap<'a> {
    document: &'a Document,
    document_url: Option<&'a Url>,
    ids: HashMap<String, ImageId>,
    next: u32,
}

impl<'a> ImageIdMap<'a> {
    /// New mapper for `document`, whose own URL is `document_url`
    /// (`None` when the tab has no current URL: every `src` then misses).
    pub fn new(document: &'a Document, document_url: Option<&'a Url>) -> Self {
        Self {
            document,
            document_url,
            ids: HashMap::new(),
            next: 0,
        }
    }

    /// Resolve `src` to its [`ImageId`] and decoded pixels: `None` when
    /// `src` does not resolve against the document base or the decode is
    /// not cached. Repeated calls with the same `src` return the same
    /// `ImageId`.
    pub fn resolve(&mut self, src: &str) -> Option<(ImageId, DecodedImage)> {
        let decoded = cached_image_for_src(self.document, self.document_url?, src)?;
        let id = *self.ids.entry(src.to_owned()).or_insert_with(|| {
            let id = ImageId(self.next);
            self.next += 1;
            id
        });
        Some((id, decoded))
    }
}

// ---------------------------------------------------------------------------
// Link helpers (pure)
// ---------------------------------------------------------------------------

/// `href` of `node` when it is an `<a>` element, else `None`.
fn link_href(doc: &Document, node: NodeId) -> Option<String> {
    let NodeKind::Element(data) = doc.kind(node) else {
        return None;
    };
    if data.tag_name != "a" {
        return None;
    }
    doc.get_attribute(node, "href").map(str::to_owned)
}

/// The base URL for resolving a clicked link's href.
///
/// Delegates to the loader's [`effective_base_url`](crate::loader::effective_base_url)
/// — the document's first `<base href>` in document order, else the tab's
/// current URL — so the rule lives in exactly one place instead of being
/// duplicated here with a "keep in sync" comment.
fn link_base_url(tab: &Tab) -> Option<Url> {
    let document_url = tab.current_url()?.clone();
    Some(match tab.document_ref() {
        Some(document) => crate::loader::effective_base_url(&document, &document_url),
        None => document_url,
    })
}

// ---------------------------------------------------------------------------
// Background loads
// ---------------------------------------------------------------------------

/// How long a replacement worker waits for a superseded worker's cookie
/// jar before giving up and starting from a fresh jar.
///
/// The handoff only pays off when the old worker already finished (the
/// common rapid-renavigation case: it bailed on the generation check in
/// microseconds). One event-loop turn plus thread-scheduling slack is
/// plenty; the bound is what keeps a superseded-but-stuck load from
/// delaying its replacement.
const HANDOFF_TIMEOUT: Duration = Duration::from_millis(100);

/// Grace period [`WindowApp::shutdown`] waits for in-flight loader
/// workers to hand their cookie jars back before giving up.
///
/// Each worker took the profile jar out at spawn
/// ([`crate::navigation::take_profile_jar`]), leaving a fresh empty jar
/// behind, so its report is the only remaining copy of the profile's
/// real cookies. Draining the report channels non-blockingly drops the
/// jar whenever close beats the worker's report: the profile keeps the
/// empty swap jar and the worker's send fails once the receiver is gone,
/// losing every stored cookie. A short bounded wait lets a worker that
/// is already finishing hand its jar back instead. A worker still stuck
/// (in a socket read, ...) when the budget expires keeps its jar —
/// reclaiming it would mean joining, which shutdown never does — but
/// close still returns promptly (see
/// `close_does_not_wait_for_stalled_load`).
const SHUTDOWN_DRAIN_TIMEOUT: Duration = Duration::from_millis(250);

/// Cancellation state shared with loader workers.
///
/// `SeqCst` throughout: loads are rare (navigations), so the strongest
/// ordering is the clearest.
#[derive(Debug, Default)]
struct LoadControl {
    /// Generation of the latest spawned load. A worker whose generation
    /// no longer matches was superseded and must not start (or report).
    generation: AtomicU64,
    /// Set when the window closes: workers bail instead of loading.
    shutdown: AtomicBool,
}

impl LoadControl {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Claim the next generation id for a new load.
    fn next_generation(&self) -> u64 {
        self.generation.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Whether the worker spawned as `generation` should still run: not
    /// superseded and the window still open.
    fn is_live(&self, generation: u64) -> bool {
        !self.shutdown.load(Ordering::SeqCst)
            && self.generation.load(Ordering::SeqCst) == generation
    }
}

/// Where a loader worker gets the cookie jar it loads with.
///
/// Detached workers cannot borrow the profile, so the jar is always
/// owned: either swapped out of the profile mutex by the event thread
/// (no load in flight) or handed over by the superseded worker.
enum JarSource {
    /// Event thread swapped the profile jar at spawn; the profile lock
    /// was held only for the swap, never across the load.
    Owned(CookieJar),
    /// A worker is still out holding the profile's cookies: wait
    /// (bounded by [`HANDOFF_TIMEOUT`]) for it to hand its jar back
    /// instead of orphaning those cookies with a fresh jar.
    Handoff(Receiver<CookieJar>),
}

/// What a loader worker sends back.
///
/// The jar always comes back: the event loop restores it into the
/// profile (current load) or forwards it down the handoff chain
/// (superseded load). Only a worker that died without sending — or one
/// abandoned on window close — takes its jar to the grave, the same
/// documented tradeoff as navigation's detached worker on timeout.
struct LoadReport {
    /// The load outcome, or `None` when the worker bailed before loading
    /// (superseded or closed): nothing to install, jar untouched.
    result: Option<Result<LoadedDocument, LoadError>>,
    /// The jar the worker took out, with whatever cookies the load set.
    jar: CookieJar,
    /// True when the worker waited on [`JarSource::Handoff`] and the
    /// superseded worker's jar never arrived within [`HANDOFF_TIMEOUT`]:
    /// the worker started from a fresh default jar instead. Such a jar
    /// is not in the cookie lineage, so the event loop must not promote
    /// it as the profile jar when it is empty — "empty jars never win",
    /// the same rule the shutdown drain applies.
    handoff_missed: bool,
}

/// An in-flight document load: the worker thread reports through `rx`;
/// `push_history` decides whether the final URL is committed to session
/// history on completion (link navigations yes, the initial load no — its
/// entry was already recorded by `Browser::new_tab`). `traverse` moves
/// the history index on completion instead of pushing: back/forward
/// traversals load their target entry's URL without recording a new one.
struct PendingLoad {
    rx: Receiver<LoadReport>,
    push_history: bool,
    traverse: Option<TraverseDir>,
}

/// Session-history traversal direction for an in-flight load.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TraverseDir {
    Back,
    Forward,
}

/// A load superseded before it reported. Its document is never
/// installed, but its jar is drained in [`WindowApp::poll_pending`] and
/// forwarded to the newest handoff waiter so the cookies it carried out
/// are not lost when a navigation is superseded mid-flight. When the
/// waiter already moved on (its handoff timed out), the jar is restored
/// to the profile instead — never dropped.
struct StaleLoad {
    rx: Receiver<LoadReport>,
}

// ---------------------------------------------------------------------------
// Headless diagnostics
// ---------------------------------------------------------------------------

/// Install the built-in welcome page into `tab` for `about:blank`:
/// no network involved, just parse [`WELCOME_HTML`]. Shared by the
/// windowed initial load, [`load_tab_headless`], and the blocking
/// [`navigate`](crate::navigation::navigate) path. `user_agent` seeds
/// `navigator.userAgent` like any other install.
pub(crate) fn install_welcome_page(
    tab: &mut Tab,
    url: &Url,
    user_agent: &str,
    fonts: &Arc<Mutex<FontCache>>,
    cookie_jar: &Arc<Mutex<CookieJar>>,
) {
    let (document, _quirks) = Document::parse_html(WELCOME_HTML.as_bytes());
    crate::navigation::install_loaded_document(
        tab,
        LoadedDocument {
            url: url.clone(),
            trace: FetchTrace {
                hops: vec![url.clone()],
            },
            document,
            stylesheets: Vec::new(),
            image_sizes: HashMap::new(),
            script_sources: HashMap::new(),
            body: WELCOME_HTML.as_bytes().to_vec(),
            webfonts: Vec::new(),
        },
        user_agent,
        fonts,
        cookie_jar,
    );
}

/// Wall-clock budget for advancing a headless page's JS event loop after
/// install (see [`pump_headless`]).
///
/// Exported so the `--check` contract in the binary's help text can name
/// the bound without duplicating it.
pub const HEADLESS_PUMP_BUDGET: Duration = Duration::from_secs(2);

/// Wall-clock bound for [`pump_headless`] on a page with the WPT harness
/// reporter armed (see
/// [`HARNESS_COMPLETION_FLAG`](crate::script::HARNESS_COMPLETION_FLAG)):
/// async tests may need well past [`HEADLESS_PUMP_BUDGET`] for the
/// testharness.js completion callback to fire, so the pump keeps driving
/// the event loop until the completion flag fires or this elapses. The
/// outer `--timeout` deadline (the worker thread the binary runs the
/// load on) and the driver's per-test kill remain the real backstops;
/// this only guarantees the pump itself terminates.
const HEADLESS_HARNESS_WAIT_BUDGET: Duration = Duration::from_secs(30);

/// Cap on the tasks one [`pump_headless`] iteration runs through the
/// page's event loop (see
/// [`PageVm::pump_bounded`](crate::script::PageVm::pump_bounded)): a page
/// that keeps scheduling due work (a `setInterval` with a 0 ms delay)
/// can never spin a single pump forever — the iteration returns, the
/// completion check runs, and the loop continues until the deadline.
const HEADLESS_PUMP_TASK_BUDGET: usize = 10_000;

/// Pacing between [`pump_headless`] iterations: how often it re-checks
/// for newly-due timers while the budget lasts.
const HEADLESS_PUMP_INTERVAL: Duration = Duration::from_millis(10);

/// Synthetic frame period for headless `requestAnimationFrame`: the
/// headless pump has no vsync, so each [`pump_headless_with_budget`]
/// iteration advances one 60 Hz frame. Monotonic by construction, which
/// is all the [`bowser_js::pump_animation_frame`] frame-pump contract
/// requires of the timestamp.
const HEADLESS_FRAME_MS: f64 = 1000.0 / 60.0;

/// Wall-clock bound for the `reftest-wait`/`TestRendered` handshake
/// ([`settle_reftest_wait`]) inside one render: after load, the page gets
/// this long to drop the `reftest-wait` class. Well under the outer
/// `--timeout` (one-shot child) and driver-side `render_timeout`
/// (session child), which remain the hard bounds for the whole render.
///
/// Exported so both render paths share the bound without duplicating it.
pub const REFTEST_SETTLE_BUDGET: Duration = Duration::from_secs(10);

/// Slice of event-loop pumping between `reftest-wait` class re-checks in
/// [`settle_reftest_wait`]: short enough that a page dropping the class
/// mid-settle doesn't idle long, long enough to let timers, rAF, and
/// macrotasks run each slice.
const REFTEST_SETTLE_SLICE: Duration = Duration::from_millis(50);

// ---------------------------------------------------------------------------
// Tab-level scroll-event lifecycle
// ---------------------------------------------------------------------------
//
// The `WindowApp` scroll-event methods (`fire_scroll_listeners_for`,
// `fire_scroll_listeners_for_direct`, `fire_snap_events_for`,
// `fire_scrollend_for`) are factored into these free functions so the
// headless pump — which owns a `Tab` but no `WindowApp` — can run the
// same lifecycle the windowed `WindowApp::pump_scripts` runs. The
// `WindowApp` methods are thin wrappers preserving their exact behavior
// (including the no-JS-bridge bookkeeping); the headless path passes a
// locally constructed [`PageEventDispatch`].

/// Tab-level core of
/// [`WindowApp::fire_scroll_listeners_for`]: stamp the moved containers
/// as scrolling ([`ScrollTree::mark_scrolled`] — the snap backstop plus
/// the `scroll_tick` stamp `scrollend` bookkeeping reads), drain and
/// fire any queued `scrollsnapchanging` deliveries BEFORE the `scroll`
/// dispatches (a changing event always precedes the `scroll` of the
/// movement that caused it), then dispatch `scroll` — non-bubbling at
/// the element for nested scrollers, the window dispatch for the
/// viewport (`None`).
///
/// `js` is `None` only on hosts without a JS bridge: the bookkeeping
/// (mark + drain) still runs — the drain happens even with no JS
/// installed, so the queue cannot accumulate — and only the element
/// dispatches are skipped; the window dispatch needs no bridge.
/// `direct` selects the direct-manipulation variant (see
/// [`WindowApp::fire_scroll_listeners_for_direct`]).
fn fire_scroll_listeners_for_tab(
    mut js: Option<&mut dyn JsDispatch>,
    tab: &mut Tab,
    scrolled: &[Option<NodeId>],
    direct: bool,
) {
    if direct {
        tab.scroll_tree.mark_scrolled_direct(scrolled);
    } else {
        tab.scroll_tree.mark_scrolled(scrolled);
    }
    // Snap events (CSS Scroll Snap 2 §5): the mark above queued any
    // `scrollsnapchanging` the movement produced (the snap backstop's
    // evaluation, or the direct-manipulation peek); drain and fire them
    // BEFORE the `scroll` dispatches below.
    let deliveries = tab.scroll_tree.take_snap_events();
    if !deliveries.is_empty()
        && let Some(js) = js.as_deref_mut()
    {
        fire_snap_events_for_tab(js, tab, &deliveries);
    }
    for target in scrolled {
        match target {
            Some(node) => {
                if let Some(js) = js.as_mut() {
                    js.dispatch_scroll(tab, *node);
                }
            }
            None => fire_window_scroll_tab(tab),
        }
    }
}

/// Touch-drag default pan for synthetic touch input (the testdriver
/// Actions executor): the headless equivalent of [`WindowApp::on_touch`]'s
/// drag arm. Feeds the inverted drag delta into the scroll chain from
/// the gesture's anchored container and reports the movement through
/// the scroll-listener path — stamping the `scrollend` tick and running
/// the snap backstop exactly like a real finger drag — so a testdriver
/// touch scroll completes the scroll-event lifecycle (`scroll` →
/// `scrollsnapchanging` → `scrollsnapchange` → `scrollend`) instead of
/// moving nothing and leaving the page's `scrollend` wait hanging.
///
/// `dx`/`dy` are scroll-offset deltas (CSS px), already inverted from
/// finger motion: finger down means the offset decreases. Zero deltas
/// (touch jitter, duplicate events) skip the chain entirely, mirroring
/// [`TouchScrollState::moved`](crate::scroll_touch::TouchScrollState::moved).
pub(crate) fn touch_drag_pan(
    js: &mut dyn JsDispatch,
    tab: &mut Tab,
    container: usize,
    dx: f32,
    dy: f32,
) {
    if dx == 0.0 && dy == 0.0 {
        return;
    }
    let scrolled = tab.scroll_tree.scroll_chain(container, dx, dy).nodes();
    if scrolled.is_empty() {
        return;
    }
    sync_nested_paint(tab, &scrolled);
    fire_scroll_listeners_for_tab(Some(js), tab, &scrolled, false);
}

/// Report scrolled containers to the page: invalidate paint for nested
/// containers (the fold shifted the baked paint tree, so the display
/// list rebuilds from it on the next refresh — windowed parity with
/// the fallback fold in `pump_scripts`), run the sticky re-bake hook
/// (script-driven scrolls bypass the wheel/keyboard paths, so they need
/// it directly), and dispatch the `scroll` listeners (whose snap
/// backstop may launch smooth scrolls the caller then completes).
/// `moved` is innermost-first, viewport (`None`) last. No-op when
/// nothing moved.
fn report_scroll_movement(
    dispatch: &mut Box<dyn JsDispatch>,
    tab: &mut Tab,
    moved: &[Option<NodeId>],
) {
    if moved.is_empty() {
        return;
    }
    if moved.iter().any(Option::is_some) {
        tab.dirty.paint = true;
    }
    dirty_layout_for_sticky(tab, moved);
    fire_scroll_listeners_for_tab(Some(dispatch.as_mut()), tab, moved, false);
}

/// Complete in-flight snap-backstop smooth scrolls instantly and report
/// each completion's movement through [`report_scroll_movement`]: the
/// headless smooth→instant degrade, shared by the headless pump loop
/// and the reftest settle's trailing fold. Bounded: a completion's snap
/// re-evaluation settles on the already-reached target, so the second
/// pass finds nothing in flight in practice.
fn complete_and_report_smooth_scrolls(dispatch: &mut Box<dyn JsDispatch>, tab: &mut Tab) {
    for _ in 0..8 {
        let settled = tab.scroll_tree.complete_smooth_scrolls();
        if settled.is_empty() {
            break;
        }
        let mut again: Vec<Option<NodeId>> = Vec::new();
        if let Some(paint) = tab.render.paint_root.as_mut() {
            again = tab.render.scroll_tree.sync_paint_offsets(paint);
        }
        if again.is_empty() {
            // No paint tree (or a stale one that missed the container):
            // the offsets still moved, so listeners still fire for what
            // the completion reported.
            again = settled;
        }
        report_scroll_movement(dispatch, tab, &again);
    }
}

/// Tab-level core of [`WindowApp::fire_snap_events_for`]: fire queued
/// snap-event deliveries for the tab, in queue order — each delivery
/// goes through the JS bridge's snap dispatch (which resolves the
/// viewport's `None` container to a `Document` dispatch). The caller has
/// already drained the queue, so deliveries never accumulate.
fn fire_snap_events_for_tab(
    js: &mut dyn JsDispatch,
    tab: &mut Tab,
    deliveries: &[SnapEventDelivery],
) {
    for d in deliveries {
        js.dispatch_snap_event(tab, d.container, d.kind.as_str(), d.targets);
    }
}

/// Tab-level core of [`WindowApp::fire_scrollend_for`]: fire `scrollend`
/// events for the containers whose scroll gesture ended this tick,
/// innermost-first (`None` = viewport) — a non-bubbling dispatch at the
/// element for nested scrollers, the window dispatch for the viewport.
/// `js` is `None` only on hosts without a JS bridge: only the element
/// dispatches are skipped; the window dispatch needs no bridge.
fn fire_scrollend_for_tab(
    mut js: Option<&mut Box<dyn JsDispatch>>,
    tab: &mut Tab,
    ended: &[Option<NodeId>],
) {
    for target in ended {
        match target {
            Some(node) => {
                if let Some(js) = js.as_mut() {
                    js.dispatch_scrollend(tab, *node);
                }
            }
            None => fire_window_scrollend_tab(tab),
        }
    }
}

/// Tab-level `scroll` dispatch for the viewport (the former
/// `WindowApp::fire_window_scroll`, factored out so the headless pump
/// can fire it without a `WindowApp`): dispatch a `scroll` event to the
/// tab's page after the viewport scrolled — the page's
/// `window`/`document` `scroll` listeners observe it, just like a
/// script-initiated `scrollTo`. No-ops when the tab has no page realm;
/// when the bindings install failed or was removed (then no listener
/// could have registered) the `eval` failure is recorded as a script
/// error — it is NOT swallowed, and it fails `--check` verdicts. Mirrors
/// [`PageEventDispatch::dispatch_key`](crate::script::PageEventDispatch::dispatch_key):
/// a host `eval` of `dispatchEvent` (the scroll event neither bubbles
/// nor is cancelable), listener throws swallowed by the bindings'
/// dispatch, DOM dirt the handlers left behind folded into the render
/// flags.
fn fire_window_scroll_tab(tab: &mut Tab) {
    let Some(installed) = tab.installed_mut() else {
        return;
    };
    let root = Some(installed.document().borrow().root());
    // Reentrancy gate (see `InstalledDocument::assert_no_live_borrow`):
    // the `dispatchEvent` eval and the pump below re-enter the page VM,
    // whose natives re-borrow the document through the registry `Weak` —
    // so no `Ref`/`RefMut` may be alive across either.
    InstalledDocument::assert_no_live_borrow(installed.document());
    let page_vm = installed.page_vm_mut();
    if let Err(e) = page_vm.eval("window.dispatchEvent({type:'scroll'})") {
        // The global install failed or was removed: no listener could
        // have registered, so this is only bookkeeping.
        if let Some(node) = root {
            page_vm.record_error(crate::script::ScriptError::eval_error(node, None, e));
        }
    }
    page_vm.pump();
    tab.sync_dirty_flags();
}

/// Tab-level `scrollend` dispatch for the viewport (the former
/// `WindowApp::fire_window_scrollend`, factored out so the headless pump
/// can fire it without a `WindowApp`): dispatch a `scrollend` event to
/// the tab's page after the viewport's scroll gesture ended — the mirror
/// of [`fire_window_scroll_tab`]: the page's `window`/`document`
/// `scrollend` listeners observe it. No-ops when the tab has no page
/// realm, or the bindings install failed.
fn fire_window_scrollend_tab(tab: &mut Tab) {
    let Some(installed) = tab.installed_mut() else {
        return;
    };
    let root = Some(installed.document().borrow().root());
    // Reentrancy gate (see `InstalledDocument::assert_no_live_borrow`):
    // the `dispatchEvent` eval and the pump below re-enter the page VM,
    // whose natives re-borrow the document through the registry `Weak` —
    // so no `Ref`/`RefMut` may be alive across either.
    InstalledDocument::assert_no_live_borrow(installed.document());
    let page_vm = installed.page_vm_mut();
    if let Err(e) = page_vm.eval("window.dispatchEvent({type:'scrollend'})") {
        // The global install failed or was removed: no listener could
        // have registered, so this is only bookkeeping.
        if let Some(node) = root {
            page_vm.record_error(crate::script::ScriptError::eval_error(node, None, e));
        }
    }
    page_vm.pump();
    tab.sync_dirty_flags();
}

/// Advance the tab's page event loop — timers, macrotasks, microtasks —
/// after a headless install.
///
/// Headless has no event-loop thread: installing the document runs the
/// page's classic scripts synchronously, but nothing ever calls
/// [`PageVm::pump`](crate::script::PageVm::pump) the way
/// `WindowApp::about_to_wait` does every windowed turn. Without this, a
/// `setTimeout` callback scheduled during initial script execution never
/// fires under `--headless`/`--check`/`--dump-display-list`, and its
/// uncaught throws never reach the [`ScriptReport`](crate::script::ScriptReport)
/// that `--check` verdicts on.
///
/// Each pump iteration runs at most [`HEADLESS_PUMP_TASK_BUDGET`] tasks
/// ([`PageVm::pump_bounded`](crate::script::PageVm::pump_bounded)) and
/// never waits for future timers, so the sleep paces re-checks:
/// short-delay timers (the common deferred-init pattern) fire within the
/// budget. (Microtasks are already drained between scripts by
/// [`PageVm::run`](crate::script::PageVm::run); the pump is for timers
/// and other macrotasks.) Each iteration also drives one synthetic
/// 60 Hz animation frame, so `requestAnimationFrame` callbacks fire in
/// every headless load — `--check`, `--dump-display-list`, `--headless`,
/// and the WPT session alike. That is a behavior change: rAF callbacks
/// previously never ran headless. The `TestRendered` settle handshake
/// ([`settle_reftest_wait`]) is the only piece scoped to the render
/// paths; it does not run for `--check`/`--dump-display-list`.
///
/// Two wait modes: an ordinary page pumps for up to
/// [`HEADLESS_PUMP_BUDGET`]; a page with the WPT harness reporter armed
/// (the driver injected `REPORTER_JS`, which initializes
/// [`HARNESS_COMPLETION_FLAG`](crate::script::HARNESS_COMPLETION_FLAG))
/// pumps until the testharness.js completion callback fires or
/// [`HEADLESS_HARNESS_WAIT_BUDGET`] elapses, so `async_test`s and
/// `promise_test`s that need more event-loop time than the fixed budget
/// still produce their completion report instead of `SubtestsUnknown`.
///
/// Skipped outright when the install ran no scripts at all (nothing
/// executed, nothing skipped, no errors): with no JS execution nothing
/// could have scheduled work, so the pump would only burn the budget.
fn pump_headless(tab: &mut Tab, fonts: &Mutex<FontCache>) {
    let no_scripts = tab.script_report().is_some_and(|report| {
        report.executed == 0 && report.skipped == 0 && report.errors.is_empty()
    });
    if no_scripts {
        return;
    }
    // The reporter arms the flag during install (it runs as a classic
    // script before the pump), so one check up front decides the wait.
    let harness_armed = tab
        .page_vm()
        .is_some_and(|page_vm| page_vm.harness_reporter_armed());
    let budget = if harness_armed {
        HEADLESS_HARNESS_WAIT_BUDGET
    } else {
        HEADLESS_PUMP_BUDGET
    };
    // Fresh frame clock per load: the rAF timestamps start at one 60 Hz
    // frame and stay monotonic for the life of this pump.
    let mut frames_pumped = 0u64;
    pump_headless_with_budget(
        tab,
        fonts,
        budget,
        HEADLESS_PUMP_TASK_BUDGET,
        &mut frames_pumped,
    );
}

/// Drive the tab's page event loop in task-budgeted pump iterations
/// until the harness completion flag fires or `budget` elapses.
///
/// `task_budget` caps the tasks a single
/// [`PageVm::pump_bounded`](crate::script::PageVm::pump_bounded) call
/// runs, so a page that keeps scheduling due work cannot spin one pump
/// forever: the iteration returns, the completion check runs, and the
/// loop continues until the deadline — the page is busy, not hung, and
/// a page that never completes still terminates at the timeout.
///
/// `frames_pumped` is the caller's synthetic frame clock for
/// `requestAnimationFrame`: the headless pump has no vsync, so each
/// iteration advances one 60 Hz frame, and the counter is caller-owned
/// so the clock stays monotonic across calls (the reftest settle loop
/// below slices its budget into 50 ms pumps; a per-call counter would
/// sawtooth the timestamp back to 16.7 ms on every slice, violating the
/// frame-pump contract). Monotonic by construction, which is all the
/// [`bowser_js::pump_animation_frame_in_vm`] contract requires of the
/// timestamp.
///
/// `fonts` is the caller's font cache, reused for every refresh this
/// pump runs: constructing a [`FontCache`] walks the system font
/// directories, so a throwaway per dirty iteration would make
/// scroll-event-heavy pages crawl — each `scroll` dispatch dirties via
/// its temporary dispatch id, dirtying layout on every scroll.
fn pump_headless_with_budget(
    tab: &mut Tab,
    fonts: &Mutex<FontCache>,
    budget: Duration,
    task_budget: usize,
    frames_pumped: &mut u64,
) {
    let deadline = Instant::now() + budget;
    // Once the frame driver reports a torn-down context, stop paying for
    // the pump: re-driving a dead context is a wasted host-table probe
    // per iteration.
    let mut raf_alive = true;
    // The headless pump has no `WindowApp`/`JsDispatch` host: construct
    // the production event dispatcher once and reuse it across
    // iterations for the scroll-event lifecycle below. Its dispatches
    // no-op when the tab has no installed document.
    let mut dispatch: Box<dyn JsDispatch> = Box::new(PageEventDispatch::new());
    loop {
        // Check the deadline before doing work: an already-past deadline
        // runs zero iterations instead of one.
        if Instant::now() >= deadline {
            return;
        }
        // Snap events queued outside a scroll funnel — the layout-snap
        // deliveries the previous iteration's `refresh`/resnap queued
        // (initial snap, target moved/removed, snap-type change): fire
        // them at the top of the iteration, before the timers and rAF
        // callbacks observe the new state (windowed `pump_scripts`
        // parity — there they fire at the top of each tab's turn).
        let top_deliveries = tab.scroll_tree.take_snap_events();
        if !top_deliveries.is_empty() {
            fire_snap_events_for_tab(&mut *dispatch, tab, &top_deliveries);
        }
        // Snapshot the viewport offset before script runs: the bindings
        // mutate the shared scroll cells directly (`window.scrollTo`,
        // `el.scrollTop = ...`), bypassing the event paths, so the
        // lifecycle below diffs against this.
        let viewport_before = tab.scroll().offset();
        let completed = {
            // Page re-entry: installs the forced-sync-layout slot so a
            // geometry getter inside the pump observes the task's
            // mutations, and runs the document-borrow reentrancy gate.
            let Some((_guard, page_vm)) = tab.reenter_page(fonts) else {
                return;
            };
            page_vm.pump_bounded(task_budget);
            // rAF callbacks never fire from the event-loop pump alone
            // (the bindings_timers frame-pump contract): drive one
            // synthetic frame per iteration so double-rAF waits (WPT
            // `waitForCompositorCommit`, reftest `takeScreenshot`) settle
            // instead of wedging on a promise that never resolves. Timers
            // ran above, so a rAF queued from a timer callback fires on
            // this iteration's frame — the browser's task-then-render
            // ordering — while a rAF callback registered during the frame
            // pump waits for the next iteration (snapshot semantics).
            // The driver evals a stub calling the hidden pump native: the
            // callback invocation needs the engine's ACTIVE interpreter,
            // which a direct pump-side call cannot provide (see
            // `pump_animation_frame_in_vm`). Callback throws are
            // report-and-continue inside the drain, so there is nothing
            // to fold into the script report here.
            *frames_pumped += 1;
            // Skip the eval entirely when the page has no web-API
            // context: there is no hidden pump native to call.
            if raf_alive && page_vm.web_api_context().is_some() {
                // The frame timestamp derives from the context's time
                // origin — the same clock as `performance.now()` — so
                // rAF timestamps and `performance.now()` agree (falls
                // back to the synthetic frame count if the context died
                // between the check above and here).
                let raf_timestamp = page_vm
                    .web_api_context()
                    .and_then(bowser_js::web_apis_now_ms)
                    .unwrap_or(*frames_pumped as f64 * HEADLESS_FRAME_MS);
                raf_alive = bowser_js::pump_animation_frame_in_vm(page_vm.vm_mut(), raf_timestamp);
            }
            // The harness completion callback ran during this pump (the
            // reporter sets the flag right after printing the report
            // marker line): the report is on its way out, stop pumping.
            page_vm.harness_completed()
        };
        // Timer callbacks can mutate the shared DOM: fold the dirt into
        // the tab's render flags so the headless render below picks it up.
        tab.sync_dirty_flags();
        // Smooth `Element.scrollTo()`/`scrollBy()` requests the bindings
        // recorded: the headless pump has no animation ticks, so drain
        // them here as instant scrolls (smooth→instant degrade, a
        // documented headless policy) rather than leaving them silently
        // unapplied — the configuration every WPT run uses. Drained
        // before the fold below so the fold reports their movement.
        drain_pending_element_smooth_scroll_headless(tab);
        // Smooth `window.scrollTo()`/`scrollBy()` requests the bindings
        // recorded: same smooth→instant degrade for the viewport — the
        // headless pump has no animation ticks, so without this drain
        // the request would stay queued forever with the offset never
        // moving. Drained before the fold below so the fold reports the
        // movement.
        drain_pending_smooth_scroll_headless(tab);
        // JS `FontFace.load()` installs since the last iteration: the
        // `refresh` check below sees `dirty.layout` and reflows (full
        // path) in this same iteration, before the rasterize reads
        // the tree.
        drain_pending_font_load_invalidation_headless(tab);
        // Scroll-event lifecycle (the windowed `pump_scripts` equivalent):
        // diff the live scroll offsets against the last-folded baseline
        // and fold script-driven movement into the paint tree — the
        // headless rasterize builds from the paint tree, so the
        // display-list transform fast path would leave the pixels stale —
        // reporting the moved containers innermost-first with the
        // viewport (`None`) last.
        //
        // This runs BEFORE the layout/style refresh below (windowed
        // order: `pump_scripts` folds and fires listeners, `refresh`
        // runs later in `handle_redraw`): every layout/style refresh
        // re-bakes the paint tree and re-seeds the folded baselines at
        // the live offsets, so folding after it would see no drift and
        // silently drop the `scroll` events for a task that both
        // scrolled and dirtied style (e.g. `el.scrollTop = 40;
        // el.classList.add('x')`). Folding into the pre-refresh tree is
        // harmless: the re-bake discards the fold's shifts, and listener
        // dirt the dispatches leave behind is serviced by the same
        // iteration's refresh.
        let mut moved: Vec<Option<NodeId>> = Vec::new();
        if let Some(paint) = tab.render.paint_root.as_mut() {
            moved = tab.render.scroll_tree.sync_paint_offsets(paint);
        } else if tab.scroll().offset() != viewport_before {
            // Pre-first-layout fallback: there is no paint tree yet,
            // so no folded baselines — diff the viewport offset
            // directly. Once the first layout bakes, the fold above
            // reports the viewport and this check would double-report.
            moved.push(None);
        }
        report_scroll_movement(&mut dispatch, tab, &moved);
        // Snap-backstop smooth scrolls the listeners launched above
        // (`mark_scrolled` → `eval_snap` → `start_smooth_scroll`) never
        // tick headless — `advance_smooth` only runs in windowed
        // `pump_scripts` — so complete them instantly here
        // (smooth→instant degrade, the documented headless policy) and
        // report each completion's movement through the same path.
        complete_and_report_smooth_scrolls(&mut dispatch, tab);
        // `scrollend`: one quiet iteration after the last movement.
        // `end_tick` advances the tick and returns the containers whose
        // scroll gesture ended (stamped on an earlier tick), queueing
        // their `scrollsnapchange`; fire those BEFORE `scrollend` (CSS
        // Scroll Snap 2 §5.1.2 — a scroll that changes the snap target
        // always reports the change before its scrollend). A container
        // that moved this iteration was stamped above, so it is never
        // ended in the same tick.
        let ended = tab.scroll_tree.end_tick();
        let deliveries = tab.scroll_tree.take_snap_events();
        if !deliveries.is_empty() {
            fire_snap_events_for_tab(&mut *dispatch, tab, &deliveries);
        }
        if !ended.is_empty() {
            fire_scrollend_for_tab(Some(&mut dispatch), tab, &ended);
        }
        // testdriver.js automation: run any clicks/key presses/bless
        // grants the page queued via `window.__bowser_testdriver`
        // (see `crate::testdriver`). The actions need layout (hit
        // testing via elementsFromPoint), so ensure the tab is laid
        // out before draining. Each dispatch pumps and folds its own
        // dirt; the loop below re-pumps so promise continuations run
        // before the harness-completion check. The font cache is the
        // caller's, shared across iterations (see the `fonts` doc
        // above).
        //
        // Runs after the scroll-event lifecycle above (windowed order):
        // the refresh re-bakes the paint tree and re-seeds the folded
        // scroll baselines, so it must not run before the fold — and
        // listener dirt the dispatches left behind is serviced here, in
        // the same iteration.
        //
        // Late-image drain (headless): decode `data:`-scheme images
        // inline — the headless driver has no background fetch threads —
        // then refresh below picks up their geometry changes
        // (`note_image_loaded` invalidates layout when an intrinsic
        // size changed). The `load`/`error` DOM events fire after the
        // refresh so handlers observe the post-layout tree. Network URLs
        // stay pending: no synchronous network in the pump.
        let image_events = drain_late_image_loads(tab);
        // w76fix M4-redrain: the scroll/snap/scrollend listeners
        // dispatched above run after this iteration's font drain, so
        // a data-URL install from one must be folded into this
        // refresh, not left for the next iteration.
        drain_pending_font_load_invalidation_headless(tab);
        if tab.dirty.layout || tab.dirty.style {
            refresh(tab, fonts);
        }
        for (node, loaded_ok) in &image_events {
            if let Some(installed) = tab.installed_mut() {
                // Reentrancy gate: listener callbacks re-enter the page
                // VM and re-borrow the document through the registry
                // `Weak`, so no `Ref`/`RefMut` may be alive here.
                InstalledDocument::assert_no_live_borrow(installed.document());
                let event_type = if *loaded_ok { "load" } else { "error" };
                installed
                    .page_vm_mut()
                    .dispatch_image_event(*node, event_type);
            }
        }
        if !image_events.is_empty() {
            // Listener callbacks can mutate the shared DOM: fold the
            // dirt into the tab's render flags for the next iteration.
            tab.sync_dirty_flags();
        }
        // `Element.scrollIntoView()` from page script: the bindings only
        // record the target node and options (they cannot see layout);
        // resolve them here against the freshly-refreshed tree — after
        // `refresh`, never before. The script turn above may have mutated
        // the DOM, so the pre-refresh layout is stale: resolving against
        // it would degrade to scroll-to-top (`drain_scroll_into_view`
        // leaves the request queued on dirty layout as defense-in-depth,
        // and `refresh` just cleared it). Movement folds into the same
        // lifecycle the script-scroll fold above uses (windowed
        // `pump_scripts` parity: there the drain runs in the script turn
        // and listeners fire at its end). A smooth-behavior request starts
        // animations the headless pump cannot tick — they are completed
        // instantly just below (the smooth→instant degrade). Note the
        // drain reports only instant-path movement: `drain_scroll_into_view`
        // pushes nothing for the smooth path, so the completion's own
        // report is what fires the smooth scroll's listeners.
        let scrolled = drain_scroll_into_view(tab);
        if !scrolled.is_empty() {
            report_scroll_movement(&mut dispatch, tab, &scrolled);
        }
        // Unconditional: a pure-smooth drain pushes nothing to `scrolled`,
        // so gating on it would leave the animations in flight until the
        // *next* iteration's scroll-event fold — or forever, if the pump
        // budget expires on this iteration and no next one runs. A no-op
        // when nothing is in flight (`complete_smooth_scrolls` returns
        // empty on its first pass).
        complete_and_report_smooth_scrolls(&mut dispatch, tab);
        crate::testdriver::drain_testdriver_requests(tab);
        // w76fix M4-redrain: image-event listeners and testdriver
        // actions above run after the refresh check, so re-drain and
        // refresh here — otherwise a page completing in this
        // iteration would return with the install unreflowed.
        drain_pending_font_load_invalidation_headless(tab);
        if tab.dirty.layout || tab.dirty.style {
            refresh(tab, fonts);
        }
        if completed {
            return;
        }
        std::thread::sleep(HEADLESS_PUMP_INTERVAL);
    }
}

/// Load `url_str` into a fresh tab and run the full style/layout/paint
/// pipeline synchronously, returning the tab with its display list built.
///
/// Backs `--dump-display-list`. `about:blank` renders the built-in welcome
/// page; anything else goes through [`crate::navigation::navigate`], so
/// the dump sees exactly the same history/timeout/jar semantics as the
/// browser's own loads.
///
/// After install, the page's event loop is advanced for up to
/// [`HEADLESS_PUMP_BUDGET`] ([`pump_headless`]): `setTimeout` callbacks
/// and other macrotasks scheduled during initial script execution run
/// here, the way the windowed event loop would run them — so `--check`
/// (which verdicts on the script report) and `--headless` (which
/// rasterizes after this returns) both observe deferred script effects.
pub fn load_tab_headless(
    url_str: &str,
    viewport: Size,
) -> Result<Tab, crate::navigation::NavError> {
    let mut tab = Tab::new(viewport);
    let jar = Arc::new(Mutex::new(CookieJar::new()));
    let url = bowser_url::parse(url_str, None).map_err(crate::navigation::NavError::Url)?;
    // Defer the window `load` event until after the first layout
    // publishes: `load` handlers must observe geometry (WPT
    // scrollable-overflow), and no layout has run at install time.
    // The install records `pending_load`; this function fires it
    // after the first `refresh` below.
    tab.defer_load = true;
    // Headless dump path: no window owns a font cache here, so use a
    // throwaway one for the pump and the display-list build below —
    // shared, not per-refresh (constructing it walks the system font
    // directories). Install the test fonts (Ahem) so WPT runs measure
    // with the same faces the WPT driver's render path uses: layout
    // must see Ahem, not just paint.
    let fonts = Arc::new(Mutex::new({
        let mut fonts = FontCache::new();
        fonts.install_test_fonts();
        fonts
    }));
    if url.serialize() == "about:blank" {
        // Headless has no `BrowserConfig`: the welcome page carries the
        // default UA, like every other headless install.
        install_welcome_page(&mut tab, &url, bowser_net::fetch::USER_AGENT, &fonts, &jar);
        tab.history.push(url);
    } else {
        crate::navigation::navigate(
            &mut tab,
            &jar,
            url_str,
            bowser_net::fetch::USER_AGENT,
            &fonts,
        )?;
    }
    // First layout+publish BEFORE the deferred `load`: `refresh`
    // runs the full pipeline (style, layout, publish) from the
    // `mark_all_dirty` the install left, so by the time the `load`
    // handlers below run, the JS engine has a published layout tree
    // to measure.
    refresh(&mut tab, &fonts);
    if tab.pending_load {
        tab.pending_load = false;
        if let Some(page_vm) = tab.page_vm_mut()
            && let Err(e) = page_vm.vm_mut().dispatch_window_load()
        {
            eprintln!("bowser: load event dispatch failed ({e}); page runs without load handlers");
        }
    }
    // Headless has no event-loop thread pumping the page's timers (only
    // `WindowApp::about_to_wait` does, windowed): advance them here so
    // deferred callbacks run before the pipeline below observes the tab.
    pump_headless(&mut tab, &fonts);
    refresh(&mut tab, &fonts);
    Ok(tab)
}

/// Whether `document.documentElement` still carries the `reftest-wait`
/// class token. Reads the `class` content attribute via `getAttribute`:
/// production element wrappers have no `classList`/`className` (those
/// are parse-shim-only), so the check goes through the attribute the
/// page's markup actually set. A failed eval (no page realm, or the
/// realm is gone) reads as "not waiting", so a broken page still
/// rasterizes instead of burning the settle budget.
fn reftest_wait_present(tab: &mut Tab) -> bool {
    let Some(page_vm) = tab.page_vm_mut() else {
        return false;
    };
    let class = page_vm.eval("document.documentElement.getAttribute('class')");
    match class {
        Ok(bowser_js::Value::Str(s)) => s
            .as_wtf8()
            .to_utf8_lossy()
            .split_whitespace()
            .any(|token| token == "reftest-wait"),
        _ => false,
    }
}

/// wptrunner's `test-wait.js` handshake for reftests, minus the harness
/// completion callback: if the page opted into `reftest-wait` (the class
/// on `<html>`), dispatch `TestRendered` on the root element — the page's
/// listener runs synchronously inside `dispatchEvent` and usually drops
/// the class and mutates style on the spot — then pump the event loop
/// until the class is gone or `deadline` passes, and re-layout so the
/// listener's mutations are in the rasterized frame.
///
/// Called from the two render paths only (the session pipeline's `render`
/// and the one-shot `--headless` pixel path): NOT from
/// [`load_tab_headless`] itself, which also backs `--check` and
/// `--dump-display-list` — firing `TestRendered` there would run page
/// mutation code those modes must not trigger.
///
/// A throwing listener records into the script report like any other page
/// error; the settle loop still observes whatever class/style state the
/// page reached. `deadline` is bounded by the caller's
/// [`REFTEST_SETTLE_BUDGET`]; the outer `--timeout` / driver-side
/// `render_timeout` remains the hard bound for the whole render.
///
/// One-shot by design: the dispatch happens once, synchronously, before
/// the settle loop. A page that registers its `TestRendered` listener
/// lazily — after load, from a timer or a `requestAnimationFrame`
/// callback — misses the dispatch and settles on the class check alone.
pub fn settle_reftest_wait(tab: &mut Tab, fonts: &Mutex<FontCache>, deadline: Instant) {
    if !reftest_wait_present(tab) {
        return;
    }
    // Page re-entry: installs the forced-sync-layout slot for geometry
    // getters in the listener, and runs the document-borrow reentrancy
    // gate.
    if let Some((_slot, page_vm)) = tab.reenter_page(fonts)
        && let Err(e) = page_vm.eval(
            "document.documentElement.dispatchEvent(new Event('TestRendered',{bubbles:true}))",
        )
    {
        // A throwing listener is a page error like any other: surface
        // it in the script report instead of swallowing it (precedent:
        // `pump_bounded` drains task errors into the report).
        page_vm.record_error(crate::script::ScriptError::task_error(e));
    }
    // The listener runs synchronously inside `dispatchEvent` above and
    // its mutations (DOM/style writes, scroll offsets) leave dirt in the
    // document and the scroll cells, not in the tab's render flags: fold
    // it now. The settle loop below only runs when the class survived
    // the dispatch (it folds dirt per slice itself), so without this a
    // listener that drops the class synchronously — the common case —
    // would leave the trailing `refresh` a no-op and rasterize the
    // pre-listener frame (e.g. dynamic-004's `scrollbarGutter` flip).
    tab.sync_dirty_flags();
    // The listener usually drops the class synchronously above; loop for
    // the deferred case (the class removal waits on timers, rAF, or
    // promises). Each slice pumps the event loop — timers, the headless
    // rAF frame pump, macrotasks — and folds dirt, then the class is
    // re-checked; the loop is bounded by `deadline` no matter what the
    // page does. The frame clock is shared across slices so the synthetic
    // rAF timestamps stay monotonic for the whole settle.
    let mut frames_pumped = 0u64;
    while reftest_wait_present(tab) && Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        pump_headless_with_budget(
            tab,
            fonts,
            remaining.min(REFTEST_SETTLE_SLICE),
            HEADLESS_PUMP_TASK_BUDGET,
            &mut frames_pumped,
        );
    }
    // Re-layout with the caller's font cache so the listener's mutations
    // (e.g. dynamic-004 removing the root's `scrollbar-gutter`) are in
    // the rasterized frame; `refresh` is a no-op when nothing is dirty.
    //
    // Fold any scroll offsets the listener moved into the paint tree
    // first: a listener that drops the class synchronously never runs
    // the settle loop above, so the pump's fold never saw its scrolls,
    // and the headless rasterize builds from the paint tree. The fold is
    // idempotent (no drift → no-op) when the loop already ran.
    //
    // When the fold finds drift, run the same report path the pump loop
    // uses: a `scroll` listener that moved offsets while dropping the
    // class synchronously must observe its scrolls exactly like one
    // whose timers kept the loop alive — otherwise identical listener
    // scrolls produce events or nothing depending on timing. Exactly
    // once: drift implies the loop never reported this movement, and
    // listener dirt the dispatches leave behind is serviced by the
    // `refresh` below. `scrollend` is deliberately not fired here: it
    // needs a quiet tick after the last movement, and no more ticks
    // come on this path.
    let mut moved: Vec<Option<NodeId>> = Vec::new();
    if let Some(paint) = tab.render.paint_root.as_mut() {
        moved = tab.render.scroll_tree.sync_paint_offsets(paint);
    }
    if !moved.is_empty() {
        let mut dispatch: Box<dyn JsDispatch> = Box::new(PageEventDispatch::new());
        report_scroll_movement(&mut dispatch, tab, &moved);
        complete_and_report_smooth_scrolls(&mut dispatch, tab);
    }
    refresh(tab, fonts);
}

/// Install a finished non-push load (the initial load, reloads): the
/// document installs exactly as a navigation's would, then the recorded
/// history entry is rewritten to the final URL after redirects. `start_initial_load`
/// records the entry with the pre-redirect URL at tab creation, so without
/// this rewrite the async initial load would install the final document
/// while history still pointed at the pre-redirect URL — the entry and
/// the document (and relative URL resolution against the entry as base)
/// would disagree.
fn install_final_document(
    tab: &mut Tab,
    loaded: LoadedDocument,
    user_agent: &str,
    fonts: &Arc<Mutex<FontCache>>,
    cookie_jar: &Arc<Mutex<CookieJar>>,
) {
    // Webfonts are installed centrally by `install_loaded_document`
    // (before first layout), so every load path — initial, push, and
    // traversal — gets them; this function only adds the history-entry
    // rewrite below.
    let final_url = loaded.url.clone();
    crate::navigation::install_loaded_document(tab, loaded, user_agent, fonts, cookie_jar);
    // Same rewrite rule back/forward traversals apply, so the entry and
    // the installed document can never disagree.
    let current = tab.history.index();
    if tab.current_url().is_some_and(|u| *u != final_url) {
        crate::navigation::replace_history_entry(tab, current, final_url);
    }
}

/// GPU state for one window: surface, its configuration, and the paint
/// backend that renders display lists into it.
struct Gpu {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    backend: WgpuBackend,
}

/// Bring up wgpu on `window`: Vulkan instance, adapter, device, surface,
/// and the paint backend. `block_on` is used for the adapter/device
/// requests (init only, never per-frame).
///
/// The `fonts` argument is the window's font cache: the backend's text
/// rasterizer reads it through an `Arc`-wrapped closure (the resolver must
/// be `Send + Sync + 'static`, so it goes through the shared `Mutex`;
/// only the event thread ever locks it).
fn init_gpu(window: &Arc<Window>, fonts: &Arc<Mutex<FontCache>>) -> Result<Gpu, WindowError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    // `Arc<Window>` is `'static`, so the surface can outlive this call.
    let surface = instance
        .create_surface(window.clone())
        .map_err(|e| WindowError::Surface(e.to_string()))?;
    // Adapter + device come from paint's single bring-up choke point; the
    // `Some(surface)` policy wants the high-performance adapter with no
    // software fallback. `block_on` is used for the requests (init only,
    // never per-frame).
    let (adapter, device, queue) =
        WgpuBackend::request_device(Some(&surface)).ok_or(WindowError::NoAdapter)?;
    let size = window.inner_size();
    let mut config = surface
        .get_default_config(&adapter, size.width.max(1), size.height.max(1))
        .ok_or(WindowError::NoAdapter)?;
    // Fifo is always supported; v1 does not need mailbox latency wins.
    config.present_mode = wgpu::PresentMode::Fifo;
    let mut backend = WgpuBackend::new(&device, &queue);
    let fonts_for_backend = Arc::clone(fonts);
    backend.set_font_data(Arc::new(move |key| {
        fonts_for_backend
            .lock()
            .ok()
            .and_then(|cache| cache.font_data(key))
    }));
    Ok(Gpu {
        surface,
        config,
        backend,
    })
}

// ---------------------------------------------------------------------------
// The application
// ---------------------------------------------------------------------------

/// winit application driving one window for one tab.
///
/// `tabs` is a disjoint borrow of the [`crate::browser::Browser`]
/// (mutably) and `jar` a clone of its profile cookie jar handle.
/// Document loads
/// run on detached worker threads that own a private cookie jar swapped
/// out of `jar` — the profile lock is never held across a load — and
/// report back through a channel the event loop polls in `about_to_wait`.
/// A new navigation supersedes the pending one: the old worker bails via
/// the [`LoadControl`] generation counter and its jar is handed to the
/// replacement, so the new load never blocks on the old. Closing the
/// window drops the report channels and sets the shutdown flag; the
/// detached workers are never joined, so a stalled server can't hang
/// browser exit.
struct WindowApp<'a> {
    tabs: &'a mut TabArena,
    /// The profile's cookie jar handle: a clone of the profile's
    /// `Arc`, so the loader swap below and the JS contexts installed
    /// from this window lock the same jar (see
    /// [`crate::navigation::install_loaded_document`]).
    jar: Arc<Mutex<CookieJar>>,
    tab: TabId,
    /// Logical viewport (CSS px) the window was configured with.
    viewport: (u32, u32),
    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,
    input: InputState,
    /// Latest keyboard modifier state, from `WindowEvent::ModifiersChanged`
    /// (winit 0.30 delivers modifiers as a separate event; `Key` carries
    /// none). Only the Shift bit is read today, for Shift+Tab and
    /// Shift+Space.
    modifiers: ModifiersState,
    /// Touch drag + fling-inertia state (single active gesture; extra
    /// fingers ignored). Input gestures always target the visible tab.
    touch: TouchScrollState,
    /// Active scrollbar thumb drag: the last primary-button press
    /// landed on a scrollbar thumb. Motion scrolls the grabbed
    /// container; release ends the gesture without click dispatch.
    scrollbar_drag: Option<ScrollbarDrag>,
    /// Active middle-click autoscroll gesture: the last middle-button
    /// press anchored it at the press point. Motion beyond the dead
    /// zone scrolls the anchored container; release ends it.
    autoscroll: Option<AutoscrollState>,
    /// Latched wheel-gesture scroller (see [`WheelLatch`]): `None`
    /// between gestures.
    wheel_latch: Option<WheelLatch>,
    pending: Option<PendingLoad>,
    /// Loads superseded before they reported: drained in `poll_pending`
    /// for their jars, never installed.
    stale: Vec<StaleLoad>,
    /// Handoff channel of the newest worker waiting for a superseded
    /// worker's jar. Stale drains send there; when the send fails — the
    /// waiter timed out and moved on — the jar falls back to the profile
    /// instead of being dropped (see `poll_pending`).
    handoff_tx: Option<mpsc::Sender<CookieJar>>,
    /// Generation counter + shutdown flag shared with loader workers
    /// (cooperative cancellation).
    load_control: Arc<LoadControl>,
    /// Page-script event dispatcher. [`run_windowed`] installs the
    /// production [`crate::script::PageEventDispatch`]; [`run_windowed_with_js`]
    /// installs one for the bindings worker (and for tests of the dispatch path).
    js: Option<Box<dyn JsDispatch>>,
    /// The browser's configured user agent: sent on every loader fetch
    /// and seeded into `navigator.userAgent` on every install (welcome
    /// page included).
    user_agent: String,
    /// The window's font cache: [`refresh`] resolves text-run fonts
    /// through it, and the paint backend reads it for rasterization (see
    /// [`init_gpu`]).
    fonts: Arc<Mutex<FontCache>>,
    /// Current window scale factor (physical px per CSS px).
    scale: f64,
    needs_redraw: bool,
    /// Init failures cannot propagate out of `ApplicationHandler`
    /// methods, so they are stashed here and returned after `run_app`.
    init_error: Option<WindowError>,
}

impl<'a> WindowApp<'a> {
    fn tab_mut(&mut self) -> Option<&mut Tab> {
        self.tabs.get_mut(self.tab)
    }

    /// Physical px to CSS px at the current scale factor.
    fn to_css(&self, x: f64, y: f64) -> (f32, f32) {
        ((x / self.scale) as f32, (y / self.scale) as f32)
    }

    /// Kick off the tab's initial load: `about:blank` (or no URL) renders
    /// the built-in welcome page synchronously — no network involved — and
    /// anything else loads on a worker thread.
    fn start_initial_load(&mut self) {
        let url = self
            .tabs
            .get(self.tab)
            .and_then(|tab| tab.current_url().cloned());
        let Some(url) = url else { return };
        if url.serialize() == "about:blank" {
            // `tab_mut` borrows `self` mutably, so clone the UA and the
            // fonts handle first.
            let user_agent = self.user_agent.clone();
            let fonts = Arc::clone(&self.fonts);
            let cookie_jar = Arc::clone(&self.jar);
            if let Some(tab) = self.tab_mut() {
                install_welcome_page(tab, &url, &user_agent, &fonts, &cookie_jar);
            }
            self.needs_redraw = true;
        } else {
            self.spawn_load(&url, false, None);
        }
    }

    /// Run `load_document_with_user_agent` on a detached worker thread; the result is
    /// picked up in `about_to_wait`. Only one load's result can be
    /// installed in v1: a new navigation supersedes the previous pending
    /// one. The old worker bails on the generation check before touching
    /// the network, and its jar is handed to the replacement through the
    /// handoff channel, so the new load never blocks on the old — not on
    /// the jar lock (each worker owns a private jar; the profile lock is
    /// held only for the swap) and not on the old load's network either.
    fn spawn_load(&mut self, url: &Url, push_history: bool, traverse: Option<TraverseDir>) {
        if self.load_control.shutdown.load(Ordering::SeqCst) {
            return;
        }
        // Collect any load that finished since the last poll first: if
        // the previous worker already reported, this isn't a supersede
        // and the new worker can take the profile jar directly.
        self.poll_pending();
        let generation = self.load_control.next_generation();
        let (tx, rx) = mpsc::channel();
        let url = url.clone();
        let control = Arc::clone(&self.load_control);
        let user_agent = self.user_agent.clone();
        // The worker must own everything it touches: detached threads
        // can't borrow the profile, so the cookie jar is swapped out of
        // the profile mutex ([`crate::navigation::take_profile_jar`]: the
        // profile lock is held only for the swap, never across the load)
        // and handed over.
        let jar_source = if self.pending.is_some() || !self.stale.is_empty() {
            // A worker is still out there holding the profile's cookies:
            // wait (bounded) for it to hand its jar back instead of
            // starting from an empty jar and orphaning those cookies.
            let (jar_tx, jar_rx) = mpsc::channel();
            self.handoff_tx = Some(jar_tx);
            JarSource::Handoff(jar_rx)
        } else {
            JarSource::Owned(crate::navigation::take_profile_jar(&self.jar))
        };
        thread::spawn(move || {
            // Closing: never touch the network — but still hand the jar
            // back. The shutdown drain listens on the report channel for
            // a bounded grace period, and this worker owns the only copy
            // of the profile's cookies from here on (see `JarSource`):
            // returning without sending would leave the profile holding
            // the empty swap jar. The handoff resolves without waiting;
            // a superseded worker's jar is drained from its own report
            // channel instead.
            if control.shutdown.load(Ordering::SeqCst) {
                let (jar, handoff_missed) = match jar_source {
                    JarSource::Owned(jar) => (jar, false),
                    JarSource::Handoff(rx) => match rx.try_recv() {
                        Ok(jar) => (jar, false),
                        // The superseded worker never handed its jar
                        // over; this worker owns nothing of the cookie
                        // lineage.
                        Err(_) => (CookieJar::default(), true),
                    },
                };
                // The drain may already be gone; a failed send just
                // discards the jar with the worker.
                let _ = tx.send(LoadReport {
                    result: None,
                    jar,
                    handoff_missed,
                });
                return;
            }
            let (jar, handoff_missed) = match jar_source {
                JarSource::Owned(jar) => (jar, false),
                JarSource::Handoff(rx) => match rx.recv_timeout(HANDOFF_TIMEOUT) {
                    Ok(jar) => (jar, false),
                    Err(_) => {
                        eprintln!(
                            "bowser: cookie handoff timed out; \
                             superseding load starts from an empty jar"
                        );
                        (CookieJar::default(), true)
                    }
                },
            };
            // Cooperative cancellation: superseded (or closed) before the
            // load started — never touch the network, and hand the jar
            // back so the drain can forward it down the handoff chain.
            if !control.is_live(generation) {
                let _ = tx.send(LoadReport {
                    result: None,
                    jar,
                    handoff_missed,
                });
                return;
            }
            // The shared worker body: runs the loader with panic capture
            // and reclaims the jar (see `crate::navigation::run_loader`).
            // A loader panic is reported instead of taking down the
            // browser (the old scoped model re-panicked on join).
            let outcome = crate::navigation::run_loader(&url, jar, &user_agent);
            let result = match outcome.result {
                Ok(result) => Some(result),
                Err(_) => {
                    eprintln!("bowser: loader thread panicked; dropping its load");
                    None
                }
            };
            // After a supersede the receiver moved to `stale` (drained
            // for its jar); after close it's dropped. Either way a failed
            // send is fine — the jar is just discarded with the worker.
            let _ = tx.send(LoadReport {
                result,
                jar: outcome.jar,
                handoff_missed,
            });
        });
        if let Some(old) = self.pending.take() {
            self.stale.push(StaleLoad { rx: old.rx });
        }
        self.pending = Some(PendingLoad {
            rx,
            push_history,
            traverse,
        });
    }

    /// Pick up finished background loads, if any: the current one installs
    /// (document + jar), superseded ones only hand their jars to the
    /// newest handoff waiter — or, when no waiter claims the jar, back
    /// to the profile, so a superseded load's cookies are never dropped
    /// on the floor (see the handoff-miss rule below).
    fn poll_pending(&mut self) {
        let handoff_tx = &self.handoff_tx;
        let jar = &self.jar;
        // A superseded worker's jar that no handoff waiter claimed (the
        // waiter timed out and moved on, or never existed): the newest
        // such jar is kept and restored to the profile below, unless the
        // profile already holds a newer jar.
        let mut orphaned: Option<CookieJar> = None;
        self.stale.retain(|stale| match stale.rx.try_recv() {
            Ok(report) => {
                let mut jar = Some(report.jar);
                let forwarded = match handoff_tx {
                    Some(tx) => match tx.send(jar.take().expect("stale jar")) {
                        Ok(()) => true,
                        // No waiter: the replacement moved on (its
                        // handoff timed out) or never waited. Take the
                        // jar back instead of dropping it — it still
                        // carries the cookie lineage the replacement
                        // never received.
                        Err(e) => {
                            jar = Some(e.0);
                            false
                        }
                    },
                    None => false,
                };
                if !forwarded
                    && let Some(j) = jar
                    && !j.is_empty()
                {
                    // Drain order is oldest-first, so the newest stale
                    // jar wins.
                    orphaned = Some(j);
                }
                false
            }
            Err(TryRecvError::Empty) => true,
            Err(TryRecvError::Disconnected) => {
                eprintln!("bowser: loader thread died");
                false
            }
        });
        // Restore the orphaned stale jar only when the profile still
        // holds the empty swap jar: a newer non-empty jar (the
        // replacement's own report, restored below) is never clobbered
        // by a stale one, and empty jars never win — the same rule as
        // the shutdown drain.
        if let Some(j) = orphaned
            && jar.lock().map(|guard| guard.is_empty()).unwrap_or(false)
        {
            crate::navigation::restore_profile_jar(jar, j);
        }
        let (push_history, traverse, report) = match self.pending.as_ref() {
            Some(pending) => match pending.rx.try_recv() {
                Ok(report) => (pending.push_history, pending.traverse, report),
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    eprintln!("bowser: loader thread died");
                    self.pending = None;
                    // The jar went down with the worker; the profile keeps
                    // the fresh jar left by the swap.
                    return;
                }
            },
            None => return,
        };
        self.pending = None;
        // Return the jar to the profile, with whatever cookies the load
        // set — unless the worker missed its cookie handoff and reports
        // an empty jar. That jar is not in the cookie lineage (the
        // superseded worker still owns the real one, drained above), so
        // promoting it would wipe the profile's cookies; the stale jar
        // stays the profile jar until the replacement reports a jar of
        // its own. Empty jars never win — the same rule as the shutdown
        // drain.
        if report.handoff_missed && report.jar.is_empty() {
            eprintln!(
                "bowser: superseding load missed its cookie handoff; \
                 keeping the superseded jar"
            );
        } else {
            crate::navigation::restore_profile_jar(&self.jar, report.jar);
        }
        match report.result {
            Some(Ok(loaded)) => {
                // `tab_mut` borrows `self` mutably, so clone the UA and
                // the fonts handle first.
                let user_agent = self.user_agent.clone();
                let fonts = Arc::clone(&self.fonts);
                let cookie_jar = Arc::clone(&self.jar);
                if let Some(tab) = self.tab_mut() {
                    if push_history {
                        // One completion rule for both load paths: push
                        // the final URL to history and install the
                        // document.
                        crate::navigation::complete_navigation(
                            tab,
                            loaded,
                            true,
                            &user_agent,
                            &fonts,
                            &cookie_jar,
                        );
                    } else {
                        // Traversals move the history index to the loaded
                        // entry first, so the entry rewrite inside
                        // `install_final_document` lands on the target
                        // entry — the same rule the blocking traversals
                        // apply. Initial loads and reloads keep their
                        // recorded history entry instead.
                        match traverse {
                            Some(TraverseDir::Back) => {
                                tab.history.back();
                            }
                            Some(TraverseDir::Forward) => {
                                tab.history.forward();
                            }
                            None => {}
                        }
                        install_final_document(tab, loaded, &user_agent, &fonts, &cookie_jar);
                    }
                }
                self.needs_redraw = true;
            }
            // Bailed before loading (superseded/closed): nothing to install.
            None => {}
            Some(Err(e)) => eprintln!("bowser: load failed: {e}"),
        }
    }

    /// Window close: cancel in-flight loads and forget them.
    ///
    /// Workers are detached, so there is nothing to join: each one either
    /// bails on the shutdown flag (before touching the network) or
    /// finishes its load in the background and finds its report channel
    /// gone. Either way [`run_windowed`] returns without waiting for the
    /// network.
    ///
    /// Reports are drained for a bounded grace period
    /// ([`SHUTDOWN_DRAIN_TIMEOUT`]) before the channels are dropped: each
    /// worker took the profile jar out at spawn, so its report is the
    /// only remaining copy of the profile's real cookies — dropping the
    /// channel while the worker is still in flight would leave the
    /// profile holding the fresh empty swap jar, and the worker's send
    /// fails once the receiver is gone, so the real jar (with every
    /// stored cookie) would be dropped with the worker. The newest jar in
    /// the lineage wins the restore: the pending worker owns the latest
    /// cookies when it either took the profile jar directly or received
    /// the stale chain's jar through the handoff. A worker that missed
    /// its handoff reports a fresh default jar, which is *not* in the
    /// lineage — empty jars never win, so one cannot clobber the real
    /// cookies a stale worker handed back (skipping an empty restore is
    /// a no-op: the profile holds the fresh empty swap jar while any
    /// load is outstanding). Workers still stuck when the budget
    /// expires keep their jars; reclaiming those would mean joining,
    /// which shutdown never does.
    fn shutdown(&mut self) {
        self.load_control.shutdown.store(true, Ordering::SeqCst);
        // One budget shared by every drained receiver: a rapid series of
        // superseded loads must not stack a full grace period each.
        let deadline = Instant::now() + SHUTDOWN_DRAIN_TIMEOUT;
        // Stale first, pending last: each received jar overwrites the
        // previous, so the pending (newest) jar wins the restore.
        let mut receivers: Vec<Receiver<LoadReport>> =
            self.stale.drain(..).map(|stale| stale.rx).collect();
        if let Some(pending) = self.pending.take() {
            receivers.push(pending.rx);
        }
        // Stale first, pending last: each received jar overwrites the
        // previous, so the pending (newest) jar wins the restore —
        // unless it is empty. A worker that missed its handoff reports
        // a fresh default jar, which is not in the cookie lineage, and
        // restoring it would drop the real cookies a stale worker handed
        // back, so empty jars never win.
        let mut jar: Option<CookieJar> = None;
        for rx in receivers {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let report = if remaining.is_zero() {
                // Budget spent: one last non-blocking peek.
                rx.try_recv().ok()
            } else {
                rx.recv_timeout(remaining).ok()
            };
            if let Some(report) = report.filter(|report| !report.jar.is_empty()) {
                jar = Some(report.jar);
            }
        }
        if let Some(jar) = jar {
            crate::navigation::restore_profile_jar(&self.jar, jar);
        }
        self.handoff_tx = None;
    }

    /// Cursor motion in CSS px.
    fn on_cursor_moved(&mut self, x: f32, y: f32) {
        // A scrollbar thumb drag owns the cursor: motion scrolls the
        // grabbed container instead of updating hover.
        if let Some(drag) = self.scrollbar_drag {
            self.input.cursor = (x, y);
            let outcome = match self.tabs.get_mut(self.tab) {
                Some(tab) => apply_scrollbar_drag(tab, &drag, x, y),
                None => None,
            };
            match outcome {
                Some((rebased, scrolled)) => {
                    self.scrollbar_drag = Some(rebased);
                    if !scrolled.is_empty() {
                        // Direct manipulation: skip the snap backstop
                        // (it would fight the cursor mid-gesture); the
                        // gesture settles with one `eval_snap` on
                        // release.
                        self.fire_scroll_listeners_for_direct(self.tab, &scrolled);
                    }
                }
                None => {
                    // The gesture ended mid-motion (the container is
                    // gone, or a relayout resized/repositioned the
                    // track): clear it and settle scroll-snap once,
                    // like a release.
                    self.scrollbar_drag = None;
                    if let Some(tab) = self.tabs.get(self.tab) {
                        tab.scroll_tree.snap_after_direct_manipulation(drag.node);
                    }
                }
            }
            self.needs_redraw = true;
            return;
        }
        // A middle-click autoscroll owns the cursor too: motion beyond
        // the dead zone scrolls the container latched at press through
        // the scroll chain.
        if let Some(auto) = self.autoscroll {
            self.input.cursor = (x, y);
            let (dx, dy) = autoscroll_delta(auto.anchor, (x, y));
            // The gesture latched its scroller at press; the node
            // re-resolves per motion against the live tree (the
            // scrollbar-drag rule), so a relayout mid-gesture cannot
            // scroll through a stale tree index. A container that
            // vanished mid-gesture ends the gesture. Resolved before the
            // mutable tab borrow below (disjoint fields either way).
            let start = self
                .tabs
                .get(self.tab)
                .and_then(|tab| resolve_latched_node(&tab.scroll_tree, auto.node));
            let scrolled = if dx == 0.0 && dy == 0.0 {
                Vec::new()
            } else if let Some(start) = start {
                match self.tab_mut() {
                    Some(tab) => apply_scroll_chain(tab, start, dx, dy),
                    None => Vec::new(),
                }
            } else {
                // No live tab, or the latched container is gone: end the
                // gesture rather than scrolling a recycled index.
                self.autoscroll = None;
                Vec::new()
            };
            if !scrolled.is_empty() {
                self.fire_scroll_listeners_for(self.tab, &scrolled);
            }
            self.needs_redraw = true;
            return;
        }
        let changed = match self.tabs.get(self.tab) {
            Some(tab) => update_hover(&mut self.input, tab, x, y),
            None => false,
        };
        if changed {
            if let Some(tab) = self.tab_mut() {
                // Justified direct style write: style dirt is itself
                // a partial-layout blocker (`try_partial_layout`
                // refuses while `dirty.style` is set), so no retained
                // subtree can be spliced under a pending restyle; the
                // layout stage re-runs in full via the style branch of
                // `refresh`. No layout flag is written here by design.
                tab.dirty.style = true;
            }
            self.needs_redraw = true;
        }
    }

    /// Mouse-button press/release dispatch: the primary button drives
    /// press/active/focus/click; the middle button anchors and ends a
    /// middle-click autoscroll gesture. Other buttons are ignored.
    fn on_mouse_button(&mut self, pressed: bool, button: MouseButton) {
        let (x, y) = self.input.cursor;
        match button {
            MouseButton::Left => self.on_primary_button(pressed, x, y),
            MouseButton::Middle => self.on_middle_button(pressed, x, y),
            _ => {}
        }
    }

    /// Primary-button press/release. Press sets active + focus; release on
    /// the press target is a click.
    fn on_primary_button(&mut self, pressed: bool, x: f32, y: f32) {
        if pressed {
            // Scrollbars intercept the press ahead of the DOM: a thumb
            // press starts a drag, a track press pages the container.
            // Either way the DOM press/active/focus path is skipped —
            // clicking a scrollbar must not focus content beneath it.
            // A positioned overlay covering the bar keeps the press
            // (`press_scrollbar_at` runs the DOM hit test first).
            let hit = match self.tabs.get(self.tab) {
                Some(tab) => press_scrollbar_at(tab, x, y),
                None => None,
            };
            if let Some(hit) = hit {
                let scrolled = match self.tabs.get_mut(self.tab) {
                    Some(tab) => {
                        if hit.on_thumb {
                            self.scrollbar_drag = tab.scroll_tree.scrollbar_drag(hit, x, y);
                            Vec::new()
                        } else {
                            apply_scrollbar_page(tab, hit, x, y)
                        }
                    }
                    None => Vec::new(),
                };
                if !scrolled.is_empty() {
                    self.fire_scroll_listeners_for(self.tab, &scrolled);
                }
                self.needs_redraw = true;
                return;
            }
            let hit = match self.tabs.get(self.tab) {
                Some(tab) => press_at(&mut self.input, tab, x, y),
                None => None,
            };
            if hit.is_some() {
                if let Some(tab) = self.tab_mut() {
                    // Style-only dirt: a partial-layout blocker in
                    // itself (see the hover site's justification).
                    tab.dirty.style = true;
                }
                self.needs_redraw = true;
            }
        } else {
            // A scrollbar drag ends on release; it never dispatches a
            // click. The drag suppressed the snap backstop per-motion,
            // so settle scroll-snap once here (non-relative).
            if let Some(drag) = self.scrollbar_drag.take()
                && let Some(tab) = self.tabs.get(self.tab)
            {
                tab.scroll_tree.snap_after_direct_manipulation(drag.node);
            }
            let had_active = self.input.active.is_some();
            let clicked = match self.tabs.get(self.tab) {
                Some(tab) => release_at(&mut self.input, tab, x, y),
                None => None,
            };
            if had_active || clicked.is_some() {
                if let Some(tab) = self.tab_mut() {
                    // Style-only dirt: a partial-layout blocker in
                    // itself (see the hover site's justification).
                    tab.dirty.style = true;
                }
                self.needs_redraw = true;
            }
            if let Some(node) = clicked {
                self.on_click(node);
            }
        }
    }

    /// Middle-button press/release: press anchors an autoscroll gesture
    /// at the press point; release ends it.
    fn on_middle_button(&mut self, pressed: bool, x: f32, y: f32) {
        if pressed {
            // The gesture latches the scroller under the press point
            // for its whole lifetime (the wheel-gesture latch rule);
            // the node re-resolves per motion, so the press only stores
            // the point and the node — a relayout mid-gesture ends it
            // instead of scrolling a recycled tree index.
            if let Some(tab) = self.tabs.get(self.tab) {
                let start = wheel_start_container(tab, x, y);
                let node = if tab.scroll_tree.is_empty() {
                    None
                } else {
                    tab.scroll_tree.container(start).node
                };
                self.autoscroll = Some(AutoscrollState {
                    anchor: (x, y),
                    node,
                });
                self.needs_redraw = true;
            }
        } else {
            self.end_autoscroll();
        }
    }

    /// End an active middle-click autoscroll gesture. The per-motion
    /// scroll chains already settled snap targets as relative scrolls,
    /// so nothing else is needed.
    fn end_autoscroll(&mut self) {
        if self.autoscroll.take().is_some() {
            self.needs_redraw = true;
        }
    }

    /// Click dispatch: page scripts run first through the [`JsDispatch`]
    /// implementation; when the page did not prevent the default action,
    /// `<a href>` navigates on a worker thread (the pre-dispatcher
    /// fallback).
    fn on_click(&mut self, node: NodeId) {
        // Disjoint field borrows (`js` vs `tabs`): `tab_mut()` would
        // take `&mut self` and conflict.
        let prevented = match self.js.as_mut() {
            Some(js) => match self.tabs.get_mut(self.tab) {
                Some(tab) => js.dispatch_click(tab, node),
                None => false,
            },
            None => false,
        };
        self.needs_redraw = true;
        if prevented {
            return;
        }
        let href = self
            .tabs
            .get(self.tab)
            .and_then(|tab| tab.document_ref())
            .and_then(|doc| link_href(&doc, node));
        let Some(href) = href else { return };
        self.navigate_link(&href);
    }

    /// Traverse session history without blocking the event thread.
    ///
    /// Same-document (fragment-only) targets scroll synchronously via the
    /// shared [`crate::navigation::traverse_fragment`] fast path;
    /// anything else loads on a worker thread through [`spawn_load`]
    /// with `push_history = false`, and the history index moves when the
    /// load completes (see [`poll_pending`](Self::poll_pending)).
    ///
    /// Routing through the async loader is what keeps this race-free:
    /// the blocking [`crate::navigation::go_back`]/[`go_forward`]
    /// take the profile cookie jar for their worker, which would steal
    /// the empty swap jar out from under an in-flight async load and
    /// orphan its cookies on restore. `spawn_load`'s jar-handoff chain
    /// hands the in-flight jar to the traversal worker instead, so the
    /// cookie lineage stays intact.
    fn traverse(&mut self, back: bool) {
        let (can, index) = match self.tabs.get(self.tab) {
            Some(tab) => {
                let can = if back {
                    tab.history.can_go_back()
                } else {
                    tab.history.can_go_forward()
                };
                (can, tab.history.index())
            }
            None => return,
        };
        if !can {
            return;
        }
        // `can` guarantees the target index exists: no underflow.
        let target = if back { index - 1 } else { index + 1 };
        let url = match self
            .tabs
            .get(self.tab)
            .and_then(|tab| tab.history.entry(target).cloned())
        {
            Some(url) => url,
            None => return,
        };
        // The fragment fast path scrolls (nested containers included):
        // the page's `scroll` listeners observe only an actual move.
        let outcome = match self.tab_mut() {
            Some(tab) => crate::navigation::traverse_fragment(tab, target, &url),
            None => None,
        };
        if let Some(scrolled) = outcome {
            if !scrolled.is_empty() {
                self.fire_scroll_listeners_for(self.tab, &scrolled);
            }
            self.needs_redraw = true;
            return;
        }
        let dir = if back {
            TraverseDir::Back
        } else {
            TraverseDir::Forward
        };
        self.spawn_load(&url, false, Some(dir));
    }

    /// Follow a link href: same-document fragment navigations scroll
    /// without refetching (the shared [`crate::navigation::navigate_fragment`]
    /// fast path, one implementation for both load paths); anything else
    /// loads on a worker thread.
    ///
    /// The href resolves against the document's effective base URL (the
    /// first `<base href>`, else the tab's current URL) — the same rule
    /// the loader and paint use for subresources — not against the
    /// current URL alone.
    fn navigate_link(&mut self, href: &str) {
        let base = self.tabs.get(self.tab).and_then(link_base_url);
        let Ok(url) = bowser_url::parse(href, base.as_ref()) else {
            return;
        };
        let outcome = match self.tab_mut() {
            Some(tab) => crate::navigation::navigate_fragment(tab, &url),
            None => None,
        };
        if let Some(scrolled) = outcome {
            if !scrolled.is_empty() {
                self.fire_scroll_listeners_for(self.tab, &scrolled);
            }
            self.needs_redraw = true;
            return;
        }
        self.spawn_load(&url, true, None);
    }

    /// Start container for a wheel event at `(x, y)` (CSS px),
    /// honoring the per-gesture latch: the first event of a gesture
    /// resolves the scroller from the cursor
    /// ([`wheel_start_container`]) and latches it; later events in the
    /// same gesture re-resolve the latched node against the live tree
    /// ([`resolve_latched_node`]).
    ///
    /// `TouchPhase::Started` always begins a fresh gesture (touchpad
    /// drivers bracket one); `TouchPhase::Ended` still scrolls with the
    /// latch, then clears it so it cannot leak into the next gesture.
    /// An `Ended` with no live latch — none latched, expired, or on a
    /// different tab — closes a gesture that never started and scrolls
    /// nothing (no phantom delta); `TouchPhase::Cancelled` clears the
    /// latch and scrolls nothing. A gap longer than
    /// [`WHEEL_GESTURE_TIMEOUT`] — or a tab switch — ends the gesture
    /// between events (discrete wheels report `Moved` only).
    ///
    /// Returns the start index plus the page unit for this event — the
    /// latched container's own scrollport height — or `None` when there
    /// is no live tab or the latched container vanished mid-gesture
    /// (the event scrolls nothing; the next event re-latches fresh).
    fn wheel_gesture_start(
        &mut self,
        x: f32,
        y: f32,
        phase: TouchPhase,
        now: Instant,
    ) -> Option<(usize, f32)> {
        // Disjoint field borrows (`tabs` vs `wheel_latch`).
        let tab = self.tabs.get(self.tab)?;
        // A cancelled gesture is dead: the latch must not survive it —
        // the next unrelated wheel flick within the timeout window
        // would otherwise chain from the wrong scroller. The event
        // scrolls nothing.
        if matches!(phase, TouchPhase::Cancelled) {
            self.wheel_latch = None;
            return None;
        }
        let live = self.wheel_latch.is_some_and(|latch| {
            latch.tab == self.tab && now.duration_since(latch.last) <= WHEEL_GESTURE_TIMEOUT
        });
        // An `Ended` with no live latch closes a gesture that never
        // started (or already timed out): clear and return without
        // taking the fresh path, which would scroll one phantom delta.
        if matches!(phase, TouchPhase::Ended) && !live {
            self.wheel_latch = None;
            return None;
        }
        let fresh = matches!(phase, TouchPhase::Started) || !live;
        let result = if fresh {
            let start = wheel_start_container(tab, x, y);
            // `wheel_start_container` falls back to `innermost_at`,
            // which is 0 on an empty tree — guard the indexing.
            let (node, page) = if tab.scroll_tree.is_empty() {
                (None, 0.0)
            } else {
                let c = tab.scroll_tree.container(start);
                (c.node, c.scrollport.h)
            };
            self.wheel_latch = Some(WheelLatch {
                tab: self.tab,
                node,
                last: now,
            });
            Some((start, page))
        } else {
            let latch = self.wheel_latch.expect("a non-fresh gesture holds a latch");
            let Some(start) = resolve_latched_node(&tab.scroll_tree, latch.node) else {
                // The latched container vanished mid-gesture: end the
                // latch. The event scrolls nothing; the next event
                // re-latches fresh.
                self.wheel_latch = None;
                return None;
            };
            self.wheel_latch = Some(WheelLatch {
                tab: self.tab,
                node: latch.node,
                last: now,
            });
            let page = tab.scroll_tree.container(start).scrollport.h;
            Some((start, page))
        };
        if matches!(phase, TouchPhase::Ended) {
            // The driver closed the gesture: the latch must not leak
            // into the next one. The event above still scrolled with it.
            self.wheel_latch = None;
        }
        result
    }

    /// Mouse wheel: DOM deltas (`dx`, `dy`) in `delta_mode` units
    /// (`WheelEvent.deltaMode`: 0 = CSS px, 1 = lines, 2 = pages) at the
    /// last known cursor position (winit reports wheel events without
    /// one). Page scripts run first: the `wheel` event dispatches at the
    /// hit node, and when a non-passive listener prevents the default
    /// action the scroll is skipped. Otherwise the delta scrolls the
    /// gesture's latched scroll container — the scroller under the
    /// cursor when the gesture began, re-resolved per event
    /// ([`Self::wheel_gesture_start`]) — chaining the remainder outward
    /// to the viewport ([`apply_scroll_chain`]): lines at 40 CSS px
    /// each, pages at one page of the latched container's own scrollport
    /// each. Viewport scrolling rides the per-frame uniform; nested
    /// scrolling folds into the paint tree and invalidates paint.
    /// Fires the page's `scroll` listeners for every container that
    /// moved.
    fn on_wheel(&mut self, dx: f32, dy: f32, delta_mode: u32, phase: TouchPhase) {
        // winit reports wheel events without a position: the last known
        // cursor position is the dispatch point.
        let (x, y) = self.input.cursor;
        let now = Instant::now();
        // Whether the page prevented the wheel's default action; without
        // a dispatcher installed nothing is prevented.
        let mut prevented = false;
        if let Some(js) = self.js.as_mut() {
            // Disjoint field borrows (`js` vs `tabs`): `tab_mut()` would
            // take `&mut self` and conflict.
            if let Some(tab) = self.tabs.get_mut(self.tab) {
                // Same hit test `wheel_start_container` uses: the DOM
                // target is the node the scroll chain would start from.
                if let Some(target) = target_at(tab, x, y) {
                    prevented = js.dispatch_wheel(tab, target, dx as f64, dy as f64, delta_mode);
                }
            }
        }
        // `deltaMode: 1` (lines) scrolls at 40 CSS px per line (the old
        // `LineDelta` x 40 behavior); `deltaMode: 2` (pages) scrolls one
        // page of the latched container's own scrollport per unit (see
        // [`Self::wheel_gesture_start`]); pixel deltas are already CSS
        // px. The DOM `WheelEvent` keeps reporting the original
        // `deltaMode`.
        debug_assert!(delta_mode <= 2, "unknown wheel deltaMode: {delta_mode}");
        // The gesture latch resolves the start container (and its page
        // unit) before any scrolling below; a vanished latch target
        // scrolls nothing this event.
        let scaled = self
            .wheel_gesture_start(x, y, phase, now)
            .map(|(start, page)| {
                let (sx, sy) = match delta_mode {
                    1 => (dx * 40.0, dy * 40.0),
                    2 => (dx * page, dy * page),
                    _ => (dx, dy),
                };
                (start, sx, sy)
            });
        // Same pattern as `on_key`: a prevented default skips the
        // fallback action (here, the scroll chain).
        let scrolled = if !prevented {
            match (scaled, self.tab_mut()) {
                (Some((start, sx, sy)), Some(tab)) => apply_scroll_chain(tab, start, sx, sy),
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };
        if !scrolled.is_empty() {
            self.fire_scroll_listeners_for(self.tab, &scrolled);
        }
        self.needs_redraw = true;
    }

    /// Touch-screen gesture phase at viewport position `(x, y)` (CSS px).
    ///
    /// This is the scroll pipeline only — DOM `TouchEvent` dispatch to
    /// page scripts is a sibling's job (the bindings worker). A finger
    /// drag scrolls the anchored container (inverted deltas through the
    /// scroll chain); a fast release starts fling inertia, ticked in
    /// [`Self::pump_scripts`]. One finger owns the gesture; extra
    /// fingers are ignored (no pinch).
    fn on_touch(&mut self, phase: TouchPhase, id: u64, x: f32, y: f32) {
        let now = Instant::now();
        match phase {
            TouchPhase::Started => {
                // Disjoint field borrows (`tabs` vs `touch`): `tab_mut()`
                // would take `&mut self` and conflict.
                if let Some(tab) = self.tabs.get(self.tab) {
                    // Same anchor rule as the wheel path: the DOM hit
                    // test runs first so a hit-tested overlay intercepts
                    // the gesture instead of the covered container. The
                    // hit node is kept (not just the container index) so
                    // the gesture can latch its Pointer Events 3 §5.2
                    // `touch-action` context below. The container's DOM
                    // node is captured alongside the index so a fling
                    // started at release can re-resolve the anchor by
                    // node after a relayout re-keys the tree indices.
                    let (container, hit) = match target_at(tab, x, y).zip(tab.document_ref()) {
                        Some((node, doc)) => {
                            (tab.scroll_tree.container_for_node(node, &doc), Some(node))
                        }
                        None => (tab.scroll_tree.innermost_at(x, y), None),
                    };
                    let node = (!tab.scroll_tree.is_empty())
                        .then(|| tab.scroll_tree.container(container).node)
                        .flatten();
                    // §5.2 context, latched at touch start: the touched
                    // element's `touch-action` intersected with its
                    // non-scrolling ancestors' up to the anchored
                    // container — a `touch-action: none` element inside
                    // a `pan-y` scroller fences the gesture off.
                    // Without a document or layout tree there is no
                    // style to intersect, so the gate stays permissive.
                    let touch_hit = match (tab.document_ref(), tab.layout_root.as_deref()) {
                        (Some(doc), Some(root)) => crate::scroll_touch::TouchHit {
                            pregate: crate::scroll::touch_action_pregate(&doc, root, hit, node),
                        },
                        _ => crate::scroll_touch::TouchHit::permissive(),
                    };
                    self.touch.started(id, x, y, now, node, touch_hit);
                }
            }
            TouchPhase::Moved => {
                let scrolled = match self.tabs.get_mut(self.tab) {
                    Some(tab) => {
                        let moved = self.touch.moved(&tab.scroll_tree, id, x, y, now);
                        sync_nested_paint(tab, &moved);
                        moved
                    }
                    None => Vec::new(),
                };
                if !scrolled.is_empty() {
                    self.fire_scroll_listeners_for(self.tab, &scrolled);
                }
            }
            TouchPhase::Ended => {
                self.touch.ended(id, x, y, now);
            }
            TouchPhase::Cancelled => {
                self.touch.cancelled(id);
            }
        }
        self.needs_redraw = true;
    }

    /// Key press: F5 reloads; browser back/forward keys traverse session
    /// history through the async loader (never the blocking traversals —
    /// see [`traverse`](Self::traverse)); everything else goes to the JS
    /// dispatcher when one is installed, and — when the page did not
    /// prevent the key's default action — Tab/Shift+Tab move keyboard
    /// focus through the sequential focus order ([`apply_focus_nav`])
    /// while the scrolling keys fall back to panning the focused
    /// element's scroll container, else the viewport ([`apply_key_scroll`]).
    fn on_key(&mut self, key: &Key, pressed: bool) {
        if !pressed {
            return;
        }
        if *key == Key::Named(NamedKey::Escape) && self.autoscroll.is_some() {
            // Escape ends an active middle-click autoscroll gesture; when
            // no gesture is active it keeps its normal path below.
            self.end_autoscroll();
            return;
        }
        if *key == Key::Named(NamedKey::F5) {
            let url = self
                .tabs
                .get(self.tab)
                .and_then(|tab| tab.current_url().cloned());
            if let Some(url) = url
                && url.serialize() != "about:blank"
            {
                self.spawn_load(&url, false, None);
            }
            return;
        }
        if *key == Key::Named(NamedKey::BrowserBack) {
            self.traverse(true);
            return;
        }
        if *key == Key::Named(NamedKey::BrowserForward) {
            self.traverse(false);
            return;
        }
        let Some(label) = key_label(key) else { return };
        // Whether the page prevented the key's default action; without a
        // dispatcher installed nothing is prevented.
        let mut prevented = false;
        if let Some(js) = self.js.as_mut() {
            // Disjoint field borrows (`js` vs `tabs`): `tab_mut()` would
            // take `&mut self` and conflict.
            if let Some(tab) = self.tabs.get_mut(self.tab) {
                prevented = js.dispatch_key(tab, &label);
            }
            self.needs_redraw = true;
        }
        // Sequential focus navigation, then the keyboard scrolling
        // fallback: Tab/Shift+Tab move `input.focused` through the focus
        // order and reveal the newly focused element
        // ([`apply_focus_nav`]); arrows/PageUp/PageDown/Space/Home/End
        // pan the focused element's scroll container (else the viewport)
        // — all only when the page didn't prevent the key's default.
        let focused = self.input.focused;
        let shift = self.modifiers.shift_key();
        let scrolled = if prevented {
            Vec::new()
        } else if label == "Tab" {
            // Disjoint field borrows (`input` vs `tabs`): `tab_mut()`
            // would take `&mut self` and conflict with `&mut self.input`.
            match self.tabs.get_mut(self.tab) {
                Some(tab) => apply_focus_nav(&mut self.input, tab, shift),
                None => Vec::new(),
            }
        } else {
            match self.tab_mut() {
                Some(tab) => apply_key_scroll_with_shift(tab, focused, &label, shift),
                None => Vec::new(),
            }
        };
        if !scrolled.is_empty() {
            self.fire_scroll_listeners_for(self.tab, &scrolled);
            self.needs_redraw = true;
        }
    }

    /// Fire `scroll` events for the containers a scroll gesture moved,
    /// innermost-first (`None` = viewport): wheel, keyboard, fragment
    /// navigation, and the `scrollIntoView()` drain all funnel through
    /// here. `scroll` does not bubble, so each moved container gets its
    /// own event — a non-bubbling dispatch at the element for nested
    /// scrollers, the window dispatch below for the viewport. Takes an
    /// explicit tab id because [`pump_scripts`](Self::pump_scripts) fires
    /// for background tabs too.
    ///
    /// Also stamps the moved containers as scrolling for `scrollend`
    /// bookkeeping ([`ScrollTree::mark_scrolled`]): `pump_scripts` fires
    /// `scrollend` for containers that then see a full event-loop tick
    /// with no movement.
    fn fire_scroll_listeners_for(&mut self, id: TabId, scrolled: &[Option<NodeId>]) {
        self.fire_scroll_listeners_impl(id, scrolled, false);
    }

    /// Direct-manipulation variant of
    /// [`fire_scroll_listeners_for`](Self::fire_scroll_listeners_for):
    /// scrollbar thumb-drag motions stamp the `scrollend` tick and
    /// fire `scroll` listeners but skip `mark_scrolled`'s snap
    /// backstop, which would launch a smooth scroll toward the snap
    /// point and fight the cursor mid-gesture. The gesture settles
    /// with one non-relative `eval_snap` on release (see
    /// [`ScrollTree::snap_after_direct_manipulation`]).
    fn fire_scroll_listeners_for_direct(&mut self, id: TabId, scrolled: &[Option<NodeId>]) {
        self.fire_scroll_listeners_impl(id, scrolled, true);
    }

    fn fire_scroll_listeners_impl(&mut self, id: TabId, scrolled: &[Option<NodeId>], direct: bool) {
        // Disjoint field borrows (`js` vs `tabs`): `tab_mut()` would
        // take `&mut self` and conflict.
        let js = self.js.as_mut();
        let Some(tab) = self.tabs.get_mut(id) else {
            return;
        };
        fire_scroll_listeners_for_tab(js.map(|b| &mut **b as _), tab, scrolled, direct);
    }

    /// Fire queued snap-event deliveries for tab `id`, in queue
    /// order: the mirror of
    /// [`fire_scrollend_for`](Self::fire_scrollend_for) — each delivery
    /// goes through the JS bridge's snap dispatch (which resolves the
    /// viewport's `None` container to a `Document` dispatch). No-op
    /// when no JS bridge is installed (the caller has already drained
    /// the queue, so deliveries never accumulate headless).
    fn fire_snap_events_for(&mut self, id: TabId, deliveries: &[SnapEventDelivery]) {
        // Disjoint field borrows (`js` vs `tabs`): `tab_mut()` would
        // take `&mut self` and conflict.
        let (Some(js), Some(tab)) = (self.js.as_mut(), self.tabs.get_mut(id)) else {
            return;
        };
        fire_snap_events_for_tab(&mut **js, tab, deliveries);
    }

    /// Fire `scrollend` events for the containers whose scroll gesture
    /// ended this tick, innermost-first (`None` = viewport): the mirror
    /// of [`fire_scroll_listeners_for`](Self::fire_scroll_listeners_for)
    /// — a non-bubbling dispatch at the element for nested scrollers,
    /// the window dispatch below for the viewport.
    fn fire_scrollend_for(&mut self, id: TabId, ended: &[Option<NodeId>]) {
        // Disjoint field borrows (`js` vs `tabs`): `tab_mut()` would
        // take `&mut self` and conflict.
        let js = self.js.as_mut();
        let Some(tab) = self.tabs.get_mut(id) else {
            return;
        };
        fire_scrollend_for_tab(js, tab, ended);
    }

    /// Window resize to `(w, h)` physical px: re-derive the CSS viewport
    /// at the current scale, invalidate layout + paint, and resize the
    /// swap chain.
    fn on_resize(&mut self, w: u32, h: u32) {
        let (lw, lh) = self.to_css(f64::from(w), f64::from(h));
        if let Some(tab) = self.tab_mut() {
            apply_resize(tab, lw.max(1.0), lh.max(1.0));
        }
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.config.width = w.max(1);
            gpu.config.height = h.max(1);
        }
        self.needs_redraw = true;
    }

    /// Whether any animation work is in flight on any live tab: a
    /// smooth scroll or overscroll spring on any tab, or touch fling
    /// momentum. The event loop must keep turning while this is true —
    /// each turn advances the animations — until every one completes.
    ///
    /// Background tabs' smooth scrolls are advanced in
    /// [`Self::pump_scripts`] too, so they must keep the loop alive as
    /// well: checking only the active tab stalls a background smooth
    /// scroll mid-flight (its wall-clock `start` keeps running) and it
    /// jumps to its end on the next wake. The overscroll spring ticks
    /// in the same pump, so it joins the same keep-alive.
    fn animations_in_flight(&self) -> bool {
        self.tabs.live_ids().into_iter().any(|id| {
            self.tabs.get(id).is_some_and(|tab| {
                tab.scroll_tree.has_smooth_in_flight() || tab.scroll_tree.has_overscroll_in_flight()
            })
        }) || self.touch.touch_has_animation_in_flight()
    }

    /// Fire every live tab's pending JS timers and macrotasks — not just
    /// the active tab's. A background tab's `setTimeout` callbacks would
    /// otherwise never run until the tab is foregrounded.
    ///
    /// Page scripts register timers on the VM's own event loop, which
    /// only advances when something pumps it: this is the window
    /// worker's pump, called from `about_to_wait` every event-loop turn
    /// (see [`crate::script::PageVm::pump`]). A tab with no page VM yet
    /// (no document installed) is a no-op.
    ///
    /// Timer callbacks run here and can mutate the shared DOM through the
    /// bindings; their dirt is folded into each tab's render flags, but
    /// only the active tab's paint dirt schedules a repaint — background
    /// tabs repaint when foregrounded.
    fn pump_scripts(&mut self) {
        let active = self.tab;
        let mut active_paint_dirty = false;
        // Tabs whose `scrollIntoView()` request moved a scroller this
        // tick, with the moved containers: fired after the loop (borrow
        // discipline).
        let mut scrolled_tabs: Vec<(TabId, Vec<Option<NodeId>>)> = Vec::new();
        // Snap events queued outside a scroll funnel — the layout
        // snap deliveries a previous turn's `refresh`/resnap queued
        // (initial snap, target moved/removed, snap-type change):
        // fire them at the top of each tab's turn, before its timers
        // and rAF callbacks observe the new state.
        for id in self.tabs.live_ids() {
            let deliveries = match self.tabs.get(id) {
                Some(tab) => tab.scroll_tree.take_snap_events(),
                None => Vec::new(),
            };
            if !deliveries.is_empty() {
                self.fire_snap_events_for(id, &deliveries);
                self.needs_redraw = true;
            }
        }
        for id in self.tabs.live_ids() {
            let Some(tab) = self.tabs.get_mut(id) else {
                continue;
            };
            // Tick in-flight smooth scrolls first, before the
            // script-driven fold below: their movement is reported via
            // `smooth_moved` only — the fold would otherwise see the
            // same movement and double-report it.
            let smooth_moved = tab.scroll_tree.advance_smooth(Instant::now());
            // Touch fling inertia ticks on the same per-turn cadence, for
            // the active tab only (input gestures always target the
            // visible tab). Disjoint field borrows (`tabs` vs `touch`).
            let touch_moved = if id == active {
                self.touch.advance_inertia(&tab.scroll_tree, Instant::now())
            } else {
                Vec::new()
            };
            // Overscroll spring: rubber-band shifts settle back to zero
            // on the same per-turn cadence. Nested containers fold into
            // the paint tree (paint dirt below); the viewport's own shift
            // rides the per-frame GPU uniform, so it needs no pipeline
            // dirt — the keep-alive at the end of the pump schedules the
            // redraw that renders it. The viewport's glow lives in the
            // display-list epilogue (page coords at the live shift), so
            // while its spring moves the epilogue re-pushes (cheap
            // truncate + re-emit) and strength/rect track the settling
            // band without a full repaint.
            let os_changed = tab.scroll_tree.advance_overscroll(Instant::now());
            if os_changed.iter().any(|n| n.is_none()) {
                tab.scroll_tree.mark_glow_dirty();
            }
            // Nested overscroll: the rubber-band shift is a paint-time
            // translation, so wrapped containers ride the transform fast
            // path (patch the matrix, re-push the epilogue); anything
            // unwrapped falls back to the paint-tree fold + rebuild.
            // The fold covers any offset drift too (smooth/inertia ticks
            // ran just above), so both epilogue halves are marked.
            if os_changed.iter().any(|n| n.is_some()) {
                let fast = match tab.display_list.as_mut() {
                    Some(list) => tab.render.scroll_tree.try_fast_fold(list).is_some(),
                    None => false,
                };
                if fast {
                    tab.scroll_tree.mark_scrollbars_dirty();
                    tab.scroll_tree.mark_glow_dirty();
                } else {
                    if let Some(paint) = tab.render.paint_root.as_mut() {
                        tab.render.scroll_tree.sync_paint_offsets(paint);
                    }
                    tab.dirty.paint = true;
                    if id == active {
                        active_paint_dirty = true;
                    }
                }
            }
            // Snapshot before script runs: the bindings mutate the
            // shared scroll cells directly (`window.scrollTo`,
            // `el.scrollTop = ...`), bypassing the wheel/keyboard paths.
            let viewport_before = tab.scroll().offset();
            // Page re-entry: installs the forced-sync-layout slot so a
            // geometry getter inside the pump observes the task's
            // mutations, and runs the document-borrow reentrancy gate.
            if let Some((_guard, page_vm)) = tab.reenter_page(&self.fonts) {
                page_vm.pump();
            }
            // Smooth `window.scrollTo()`/`scrollBy()` requests the
            // bindings recorded: start the viewport animation (instant
            // scrolls applied synchronously in the natives).
            drain_pending_smooth_scroll(tab);
            // Smooth `Element.scrollTo()`/`scrollBy()` for nested scroll
            // containers: same idea, resolved through the scroll tree.
            drain_pending_element_smooth_scroll(tab);
            // JS `FontFace.load()` installs since the last turn: new
            // text metrics under the published tree, so the turn's
            // `sync_dirty_flags` + active-paint check below fold this
            // into the next `refresh` (full path) and redraw.
            drain_pending_font_load_invalidation(tab);
            // Fold script-driven scrolls: the transform fast path first
            // (wrapped containers patch their display-list transforms
            // in place, no rebuild), falling back to the paint-tree
            // sync + full rebuild. Queue `scroll` events for the moved
            // containers (innermost-first, viewport last).
            let mut script_scrolled: Vec<Option<NodeId>> = Vec::new();
            let mut script_needs_paint = false;
            if let Some(paint) = tab.render.paint_root.as_mut() {
                let fast = match tab.display_list.as_mut() {
                    Some(list) => tab.render.scroll_tree.try_fast_fold(list),
                    // No display list yet: the next build wraps from
                    // scratch, so fold into the paint tree below.
                    None => None,
                };
                match fast {
                    Some(moved) => {
                        if !moved.is_empty() {
                            // Content rides the patched transforms; only
                            // the chrome (scrollbars) and glow are stale.
                            tab.scroll_tree.mark_scrollbars_dirty();
                            tab.scroll_tree.mark_glow_dirty();
                        }
                        script_scrolled = moved;
                    }
                    None => {
                        script_scrolled = tab.render.scroll_tree.sync_paint_offsets(paint);
                        script_needs_paint = script_scrolled.iter().any(Option::is_some);
                    }
                }
            } else if tab.scroll().offset() != viewport_before {
                // Pre-first-layout fallback: there is no paint tree
                // yet, so no folded baselines — diff the viewport
                // offset directly. Once the first layout bakes, the
                // folds above report the viewport and this check would
                // double-report.
                script_scrolled.push(None);
            }
            // Merge the smooth/inertia-tick movement, deduplicated: a
            // script may have written the same container after the tick
            // (which cancels that animation on its next tick).
            for node in smooth_moved.into_iter().chain(touch_moved) {
                if !script_scrolled.contains(&node) {
                    script_scrolled.push(node);
                }
            }
            if !script_scrolled.is_empty() {
                // A fallback fold shifted the baked paint tree, so the
                // display list rebuilds (fast-pathed moves need no
                // repaint — the patched transforms already show them).
                // A viewport-only scroll applies at render and needs no
                // repaint — but it does stale the baked scrollbar rects,
                // which the next `refresh` re-pushes.
                if script_needs_paint {
                    tab.dirty.paint = true;
                }
                if script_scrolled.iter().any(Option::is_none) {
                    tab.scroll_tree.mark_scrollbars_dirty();
                }
                // Script-driven scrolls bypass the wheel/keyboard paths,
                // so they need the sticky re-bake hook directly.
                dirty_layout_for_sticky(tab, &script_scrolled);
                scrolled_tabs.push((id, script_scrolled));
            }
            tab.sync_dirty_flags();
            if id == active && tab.dirty.paint {
                active_paint_dirty = true;
            }
            // `Element.scrollIntoView()` from page script: the bindings
            // only record the target node (they cannot see layout); the
            // embedder resolves it against the layout tree here, once per
            // event-loop tick.
            let scrolled = drain_scroll_into_view(tab);
            if !scrolled.is_empty() {
                scrolled_tabs.push((id, scrolled));
            }
            // testdriver.js automation: run any clicks/key presses/bless
            // grants the page queued via `window.__bowser_testdriver`
            // (see `crate::testdriver`). Drained every event-loop turn
            // so a page driving the bridge on timers cannot grow the
            // queue without bound. Unarmed requests (no vendor shim in
            // flight) are refused by the drain itself; layout-dependent
            // steps degrade gracefully on dirty layout, as in headless.
            crate::testdriver::drain_testdriver_requests(tab);
        }
        if active_paint_dirty {
            self.needs_redraw = true;
        }
        // Any animation work in flight — smooth scrolls on any live
        // tab, touch fling momentum — keeps the event loop turning:
        // each turn advances the animations until every one completes.
        if self.animations_in_flight() {
            self.needs_redraw = true;
        }
        for (id, scrolled) in scrolled_tabs {
            self.fire_scroll_listeners_for(id, &scrolled);
            self.needs_redraw = true;
        }
        // `scrollend`: one event-loop tick after the last movement. Each
        // live tab drains its own tree: containers stamped scrolling on
        // an earlier tick but with no movement this tick get `scrollend`
        // and leave the scrolling state; containers that moved this tick
        // (stamped above, or by a wheel/keyboard event earlier this turn)
        // stay marked. This runs after the `scroll` dispatches so a
        // container that moved this tick is never ended in the same tick.
        for id in self.tabs.live_ids() {
            let ended = match self.tabs.get(id) {
                Some(tab) => tab.scroll_tree.end_tick(),
                None => Vec::new(),
            };
            // `end_tick` queued `scrollsnapchange` for the containers
            // that just settled: fire those BEFORE `scrollend` (CSS
            // Scroll Snap 2 §5.1.2 — a scroll that changes the snap
            // target always reports the change before its scrollend).
            let deliveries = match self.tabs.get(id) {
                Some(tab) => tab.scroll_tree.take_snap_events(),
                None => Vec::new(),
            };
            if !deliveries.is_empty() {
                self.fire_snap_events_for(id, &deliveries);
                self.needs_redraw = true;
            }
            if !ended.is_empty() {
                self.fire_scrollend_for(id, &ended);
                self.needs_redraw = true;
            }
        }
        // w76fix M4-redrain: the `scroll`/`scrollend`/snap listeners
        // fired above — and the testdriver actions drained inside the
        // per-tab loop — all run after that loop's font drain, so a
        // data-URL install synchronously triggered from one would
        // otherwise paint stale metrics for a frame: re-drain here so
        // the full reflow is marked before `handle_redraw`'s refresh.
        // Generation-based drain: a repeated drain in the same
        // iteration reports nothing new, so the re-drain is free
        // unless another install landed meanwhile.
        let mut redrain_paint = false;
        for id in self.tabs.live_ids() {
            let Some(tab) = self.tabs.get_mut(id) else {
                continue;
            };
            drain_pending_font_load_invalidation(tab);
            if id == active && tab.dirty.paint {
                redrain_paint = true;
            }
        }
        if redrain_paint {
            self.needs_redraw = true;
        }
    }

    /// One frame: refresh dirty pipeline stages, then present the tab's
    /// display list (empty until the first load lands).
    fn handle_redraw(&mut self) {
        // Disjoint field borrows (`tabs`/`fonts` vs `gpu`): `tab_mut()`
        // would take `&mut self` and conflict with the borrows below.
        let fonts = Arc::clone(&self.fonts);
        let tab = match self.tabs.get_mut(self.tab) {
            Some(tab) => tab,
            None => return,
        };
        refresh(tab, &fonts);
        let gpu = match self.gpu.as_mut() {
            Some(gpu) => gpu,
            None => return,
        };
        let empty;
        let list = match tab.display_list.as_ref() {
            Some(list) => list,
            None => {
                empty = DisplayList::new();
                &empty
            }
        };
        // The viewport's transient overscroll (rubber-band) shift rides
        // the per-frame uniform exactly like the offset: the whole page
        // — viewport-anchored boxes included — follows the edge press,
        // and the spring settles it back. Subtracted, like the offset:
        // a positive push (past the max edge) moves content with the
        // push direction.
        let (sx, sy) = tab.scroll().offset();
        let (osx, osy) = tab.scroll_tree.overscroll_shift(0);
        gpu.backend.render_to_surface(
            list,
            tab.viewport,
            (sx - osx, sy - osy),
            &gpu.surface,
            &gpu.config,
        );
    }
}

/// Drain one tab's pending `scrollIntoView()` request: the bindings
/// only record the target node and the parsed options (they cannot see
/// the layout tree); the embedder resolves them against the layout tree
/// here. A dirty layout root is stale — script mutated the DOM in the
/// same tick and layout has not re-run yet (it runs in `handle_redraw`,
/// after this pump) — so the request is left queued: resolving now
/// would consume it and degrade to scroll-to-top. The next tick
/// resolves it against the fresh tree. `scroll_to_node` keeps its own
/// guard as defense-in-depth. Returns the scrolled containers
/// innermost-first (`None` = viewport); empty when nothing moved (the
/// caller fires `scroll` listeners only then). A smooth request starts
/// animations instead of moving: their ticks report the movement.
fn drain_scroll_into_view(tab: &mut Tab) -> Vec<Option<NodeId>> {
    if tab.dirty.layout {
        return Vec::new();
    }
    let ctx = tab.page_vm().and_then(|p| p.web_api_context());
    let Some(req) = ctx.and_then(bowser_js::take_scroll_into_view_request) else {
        return Vec::new();
    };
    let scrolled = crate::navigation::scroll_to_node(
        tab,
        req.node,
        req.block.into(),
        req.inline.into(),
        req.behavior,
    );
    sync_nested_paint(tab, &scrolled);
    scrolled
}

/// Resolve one tab's pending smooth `window.scrollTo()`/`scrollBy()`
/// request to the absolute viewport target, resolved against the
/// viewport's live offset at drain time. `None` when no request is
/// pending. A missing member keeps the current offset (`scrollTo`);
/// `scrollBy` members are deltas that default to zero.
fn take_pending_viewport_smooth_scroll_target(tab: &mut Tab) -> Option<(f32, f32)> {
    let ctx = tab.page_vm().and_then(|p| p.web_api_context());
    let req = ctx.and_then(bowser_js::take_pending_smooth_scroll)?;
    let (cx, cy) = tab.scroll().offset();
    Some(if req.relative {
        (cx + req.x.unwrap_or(0.0), cy + req.y.unwrap_or(0.0))
    } else {
        (req.x.unwrap_or(cx), req.y.unwrap_or(cy))
    })
}

/// Drain one tab's pending smooth `window.scrollTo()`/`scrollBy()`
/// request: the bindings only record the target (they cannot animate);
/// the embedder starts the viewport animation here, once per
/// event-loop tick. Instant scrolls never reach this drain — the
/// natives apply them synchronously and cancel any pending smooth
/// request first. Windowed event loop only; the headless pump uses
/// [`drain_pending_smooth_scroll_headless`].
fn drain_pending_smooth_scroll(tab: &mut Tab) {
    let Some((tx, ty)) = take_pending_viewport_smooth_scroll_target(tab) else {
        return;
    };
    // A new scroll command cancels the in-flight animation; the pending
    // slot holds at most one request (newest wins), so there is nothing
    // older to preserve.
    tab.scroll_tree.start_smooth_scroll(0, tx, ty);
}

/// Drain one tab's pending JS font-load invalidation signal: a JS
/// `FontFace.load()` installed bytes into the shared cache since the
/// last drain, so text advances changed under the published tree.
/// Routed through [`Tab::mark_layout_dirty_full`] (not a bare
/// `dirty.layout = true`): a new font moves every box, so no retained
/// subtree report may be served incrementally against it — the next
/// `refresh` must take the full path. Paint is dirtied alongside so
/// the windowed pump's redraw flag and the headless rasterize pick
/// up the new glyphs. The signal persists until drained, so a
/// data-URL install during initial scripts (before the first pump
/// iteration) is still caught here. Shared by both pumps: the
/// windowed pump calls this directly, and the headless pump calls it
/// through [`drain_pending_font_load_invalidation_headless`]
/// (which is a plain delegation, not a separate implementation).
///
/// Generation-based (not a bare bool drain): the bindings report
/// `Some(gen)` only when the shared-cache generation advanced past
/// the last generation they reported, so this reflows exactly once
/// per install batch and the reported generation identifies which
/// snapshot went stale. The last-seen baseline lives per context
/// in the bindings (see `WebApis::take_pending_font_load_generation`),
/// not on the tab, so a new page's context can never inherit a
/// stale baseline from a dead one.
fn drain_pending_font_load_invalidation(tab: &mut Tab) {
    let ctx = tab.page_vm().and_then(|p| p.web_api_context());
    if ctx
        .and_then(bowser_js::take_pending_font_load_generation)
        .is_some()
    {
        tab.mark_layout_dirty_full();
        tab.dirty.paint = true;
    }
}

/// Resolve one tab's pending smooth `Element.scrollTo()`/`scrollBy()`
/// request: the scroll-tree index plus the absolute target, resolved
/// against the target container's live offset at drain time. `None`
/// when no request is pending or the container is gone.
fn take_pending_element_smooth_scroll_target(tab: &mut Tab) -> Option<(usize, f32, f32)> {
    let ctx = tab.page_vm().and_then(|p| p.web_api_context());
    let (node, req) = ctx.and_then(bowser_js::take_pending_element_smooth_scroll)?;
    let idx = tab.scroll_tree.index_of(node)?;
    let (cx, cy) = tab.scroll_tree.container(idx).state.offset();
    let (tx, ty) = if req.relative {
        (cx + req.x.unwrap_or(0.0), cy + req.y.unwrap_or(0.0))
    } else {
        (req.x.unwrap_or(cx), req.y.unwrap_or(cy))
    };
    Some((idx, tx, ty))
}

/// Drain one tab's pending smooth `Element.scrollTo()`/`scrollBy()`
/// request: the element variant of `drain_pending_smooth_scroll`,
/// started through the scroll tree (which clamps into the container's
/// range and snap-evaluates on completion). Windowed event loop only;
/// the headless pump uses
/// [`drain_pending_element_smooth_scroll_headless`].
fn drain_pending_element_smooth_scroll(tab: &mut Tab) {
    let Some((idx, tx, ty)) = take_pending_element_smooth_scroll_target(tab) else {
        return;
    };
    tab.scroll_tree.start_smooth_scroll(idx, tx, ty);
}

/// Headless drain for pending smooth `Element.scrollTo()`/`scrollBy()`
/// requests: the headless pump has no animation ticks
/// (`advance_smooth` only runs in the windowed event loop), so starting
/// the animation would leave the scroll silently unapplied. Degrade
/// smooth→instant instead — the script-observable offset lands on the
/// target immediately. Documented policy, not a bug: headless runs
/// have no frames to animate over, and WPT tests assert end states,
/// not motion.
fn drain_pending_element_smooth_scroll_headless(tab: &mut Tab) {
    let Some((idx, tx, ty)) = take_pending_element_smooth_scroll_target(tab) else {
        return;
    };
    // Same degenerate-safe clamp `start_smooth_scroll` uses; the shared
    // cell is what `scrollTop` reads and what the next headless render
    // paints from.
    tab.scroll_tree.container(idx).state.set_offset(tx, ty);
}

/// Headless drain for pending late `<img>` loads: decode `data:`-scheme
/// images inline and record their intrinsics via
/// [`Tab::note_image_loaded`]. Returns one `(node, loaded_ok)` pair per
/// newly completed load so the pump can fire the matching `load` /
/// `error` DOM event.
///
/// Only `data:` URLs drain here: the headless driver runs no background
/// fetch threads, and a synchronous network fetch in the pump would
/// hang the test driver on the network. Network-URL images stay pending
/// (and `complete` stays false) until the windowed fetch path runs —
/// a documented headless gap.
///
/// Load-time images the scan seeds as complete (they never produce a
/// `PendingLateImage`) get their `load` event here too: the drain diffs
/// the shared completed set around the scan, so every newly completed
/// node — seeded or freshly decoded — is reported exactly once.
fn drain_late_image_loads(tab: &mut Tab) -> Vec<(NodeId, bool)> {
    let completed_before: HashSet<NodeId> = tab.completed_image_loads().borrow().clone();
    let mut failed: HashSet<NodeId> = HashSet::new();
    let pendings = tab.pending_late_images();
    for pending in pendings {
        let dims = crate::loader::decode_data_image_url(&pending.url);
        let applied = tab.note_image_loaded(pending.node, pending.generation, dims);
        if applied && dims.is_none() {
            failed.insert(pending.node);
        }
    }
    let mut fired: Vec<(NodeId, bool)> = tab
        .completed_image_loads()
        .borrow()
        .iter()
        .filter(|n| !completed_before.contains(n))
        .map(|n| (*n, !failed.contains(n)))
        .collect();
    // Deterministic event order: the arena index is document order for
    // parser-inserted images.
    fired.sort_by_key(|(node, _)| node.index());
    fired
}

/// Headless drain for pending smooth `window.scrollTo()`/`scrollBy()`
/// requests: the viewport variant of
/// [`drain_pending_element_smooth_scroll_headless`]. The headless pump
/// has no animation ticks (`advance_smooth` only runs in the windowed
/// event loop), so starting the animation would leave the request
/// queued forever with the offset never moving. Degrade smooth→instant
/// instead — the script-observable offset lands on the target
/// immediately (the documented headless policy: headless runs have no
/// frames to animate over, and WPT tests assert end states, not
/// motion). The shared cell's `set_offset` clamps into the live range
/// (the same degenerate-safe clamp `start_smooth_scroll` uses); the
/// drain runs before the pump's scroll-event fold, so the fold reports
/// the movement.
fn drain_pending_smooth_scroll_headless(tab: &mut Tab) {
    let Some((tx, ty)) = take_pending_viewport_smooth_scroll_target(tab) else {
        return;
    };
    tab.scroll().set_offset(tx, ty);
}

/// Headless twin of [`drain_pending_font_load_invalidation`]: a
/// newly reported generation means the same full reflow here, and
/// the same iteration's
/// `if tab.dirty.layout || tab.dirty.style { refresh(tab, fonts); }`
/// check picks it up, so a font installed by initial scripts reflows
/// before the headless rasterize reads the tree.
fn drain_pending_font_load_invalidation_headless(tab: &mut Tab) {
    drain_pending_font_load_invalidation(tab);
}

impl<'a> ApplicationHandler for WindowApp<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = match event_loop.create_window(
            Window::default_attributes()
                .with_title("bowser")
                .with_inner_size(PhysicalSize::new(self.viewport.0, self.viewport.1)),
        ) {
            Ok(window) => Arc::new(window),
            Err(e) => {
                self.init_error = Some(WindowError::Window(e.to_string()));
                event_loop.exit();
                return;
            }
        };
        match init_gpu(&window, &self.fonts) {
            Ok(gpu) => self.gpu = Some(gpu),
            Err(e) => {
                self.init_error = Some(e);
                event_loop.exit();
                return;
            }
        }
        self.scale = window.scale_factor();
        // Sync the tab viewport to the real logical size (the platform
        // may have adjusted the requested inner size).
        let size = window.inner_size();
        self.on_resize(size.width, size.height);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                self.shutdown();
                event_loop.exit();
            }
            WindowEvent::Resized(size) => self.on_resize(size.width, size.height),
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scale = scale_factor;
                if let Some(window) = self.window.as_ref() {
                    let size = window.inner_size();
                    self.on_resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => self.handle_redraw(),
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = self.to_css(position.x, position.y);
                self.on_cursor_moved(x, y);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.on_mouse_button(state == WinitElementState::Pressed, button)
            }
            WindowEvent::MouseWheel { delta, phase, .. } => {
                // DOM `WheelEvent.deltaMode`: `LineDelta` reports lines
                // (mode 1), `PixelDelta` CSS px (mode 0). The raw deltas
                // go to page scripts; `on_wheel` scales lines to px for
                // the scroll chain. The phase brackets touchpad scroll
                // gestures for the per-gesture scroller latch.
                let (dx, dy, delta_mode) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (x, y, 1),
                    MouseScrollDelta::PixelDelta(p) => {
                        let (x, y) = self.to_css(p.x, p.y);
                        (x, y, 0)
                    }
                };
                self.on_wheel(dx, dy, delta_mode, phase);
            }
            WindowEvent::Touch(touch) => {
                // Touch locations are physical px, like cursor positions.
                let (x, y) = self.to_css(touch.location.x, touch.location.y);
                self.on_touch(touch.phase, touch.id, x, y);
            }
            WindowEvent::ModifiersChanged(mods) => {
                self.modifiers = mods.state();
            }
            WindowEvent::KeyboardInput { event, .. } => self.on_key(
                &event.logical_key,
                event.state == WinitElementState::Pressed,
            ),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.poll_pending();
        // The script pump: fire the active tab's due JS timers and
        // macrotasks (`setTimeout` et al.) once per event-loop turn.
        self.pump_scripts();
        if self.needs_redraw {
            self.needs_redraw = false;
            if let Some(window) = self.window.as_ref() {
                window.request_redraw();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run the windowed event loop for `tab` until the window closes.
///
/// The browser is split into disjoint borrows — the tab arena mutably for
/// the event loop, the profile cookie jar shared for the loader workers.
/// Document loads run on detached worker threads (never the event
/// thread); each worker owns the jar it swapped out of the profile, so
/// loads never block each other on the profile lock. Returns when the
/// window closes without waiting for in-flight loads: closing sets the
/// shutdown flag and drops the report channels, and the detached workers
/// finish (or bail) on their own. A stale `tab` id renders an empty
/// window until closed.
/// Run the windowed event loop for `tab` until the window closes.
///
/// [`run_windowed_with_js`] with the production page-script event
/// dispatcher ([`PageEventDispatch`](crate::script::PageEventDispatch)):
/// clicks run inline handlers and `addEventListener` listeners, and
/// fall back to link navigation unless the page prevented the default
/// action; keys go to the page.
pub fn run_windowed(browser: &mut Browser, tab: TabId) -> Result<(), WindowError> {
    run_windowed_with_js(browser, tab, Some(Box::new(PageEventDispatch::new())))
}

/// [`run_windowed`] with a page-script event dispatcher installed.
///
/// `Some(js)` routes clicks and keys through [`JsDispatch`] first,
/// falling back to default actions (link navigation) unless the page
/// prevented them; `None` keeps the pre-dispatcher behavior (clicks
/// fall back to link navigation, keys are ignored). The install point
/// for the bindings worker's dispatcher (and for tests of the dispatch
/// path).
///
/// The browser is split into disjoint borrows — the tab arena mutably for
/// the event loop, the profile cookie jar shared for the loader workers.
/// Document loads run on detached worker threads (never the event
/// thread); each worker owns the jar it swapped out of the profile, so
/// loads never block each other on the profile lock. Returns when the
/// window closes without waiting for in-flight loads: closing sets the
/// shutdown flag and drops the report channels, and the detached workers
/// finish (or bail) on their own. A stale `tab` id renders an empty
/// window until closed.
pub fn run_windowed_with_js(
    browser: &mut Browser,
    tab: TabId,
    js: Option<Box<dyn JsDispatch>>,
) -> Result<(), WindowError> {
    let event_loop = EventLoop::new().map_err(|e| WindowError::EventLoop(e.to_string()))?;
    // Disjoint borrows: the event loop mutates `tabs` while loader
    // workers briefly borrow the jar for the cookie swap, through the
    // cloned `Arc` handle below (the window owns no borrow of the
    // profile itself). `viewport` is `Copy`, so its borrow ends before
    // the mutable borrows begin.
    let viewport = browser.config().viewport;
    let user_agent = browser.config().user_agent.clone();
    let tabs: &mut TabArena = &mut browser.tabs;
    let jar = Arc::clone(&browser.profile.cookie_jar);
    let mut app = WindowApp {
        tabs,
        jar,
        tab,
        viewport,
        window: None,
        gpu: None,
        input: InputState::default(),
        modifiers: ModifiersState::default(),
        touch: TouchScrollState::new(),
        scrollbar_drag: None,
        autoscroll: None,
        wheel_latch: None,
        pending: None,
        stale: Vec::new(),
        handoff_tx: None,
        load_control: LoadControl::new(),
        js,
        user_agent,
        fonts: Arc::new(Mutex::new(FontCache::new())),
        scale: 1.0,
        needs_redraw: false,
        init_error: None,
    };
    app.start_initial_load();
    let result = event_loop.run_app(&mut app);
    // Never wait for loader threads on the way out: they are detached,
    // so this just stops listening for their reports.
    app.shutdown();
    if let Err(e) = result {
        return Err(WindowError::EventLoop(e.to_string()));
    }
    match app.init_error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Tests (no window needed; window creation is #[ignore]-gated)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::Webfont;
    use crate::tab::mark_all_dirty;
    use bowser_js::Value;
    use bowser_layout::IntrinsicDims;
    use bowser_layout::{BoxType, FixedFontMetrics, Rect};
    use bowser_paint::fonts::FontStyle;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn test_tab() -> Tab {
        Tab::new(Size { w: 800.0, h: 600.0 })
    }

    /// Translate-resolution context for direct `hit_test` calls in
    /// tests: matches `test_tab`'s viewport and the px fallback.
    fn test_tctx() -> TranslateCtx {
        TranslateCtx {
            root_font_size: 16.0,
            viewport_w: 800.0,
            viewport_h: 600.0,
        }
    }

    fn layout_box(rect: Rect, node: Option<NodeId>, children: Vec<LayoutBox>) -> LayoutBox {
        LayoutBox {
            rect,
            children,
            box_type: BoxType::Block,
            style: None,
            text: None,
            node,
            fragments: Vec::new(),
            z_index: None,
            collapsed_borders: None,
            phantom_w: false,
            phantom_h: false,
        }
    }

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect { x, y, w, h }
    }

    /// Tab whose layout tree has two sibling boxes; the second paints on
    /// top and contains a nested child.
    fn tab_with_boxes() -> (Tab, NodeId, NodeId) {
        let (doc, _) = Document::parse_html(b"<p>a</p><p>b</p>");
        let els = doc.get_elements_by_tag_name("p");
        let (first, second) = (
            els.item(0).expect("first p"),
            els.item(1).expect("second p"),
        );
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        tab.layout_root = Some(Rc::new(layout_box(
            rect(0.0, 0.0, 800.0, 600.0),
            None,
            vec![
                layout_box(rect(0.0, 0.0, 800.0, 100.0), Some(first), vec![]),
                layout_box(
                    rect(0.0, 50.0, 800.0, 100.0),
                    Some(second),
                    vec![layout_box(
                        rect(10.0, 60.0, 100.0, 20.0),
                        Some(first),
                        vec![],
                    )],
                ),
            ],
        )));
        (tab, first, second)
    }

    #[test]
    fn hit_test_prefers_deepest_and_topmost() {
        let (tab, first, second) = tab_with_boxes();
        let root = tab.layout_root.as_deref().unwrap();
        // (20, 70): inside second's child (node `first`) and second itself;
        // the deeper child wins.
        assert_eq!(
            hit_test(
                &tab.scroll_tree,
                root,
                20.0,
                70.0,
                (0.0, 0.0),
                (0.0, 0.0),
                test_tctx(),
                None
            ),
            Some(first)
        );
        // (400, 120): inside second only.
        assert_eq!(
            hit_test(
                &tab.scroll_tree,
                root,
                400.0,
                120.0,
                (0.0, 0.0),
                (0.0, 0.0),
                test_tctx(),
                None
            ),
            Some(second)
        );
        // (400, 20): inside first only.
        assert_eq!(
            hit_test(
                &tab.scroll_tree,
                root,
                400.0,
                20.0,
                (0.0, 0.0),
                (0.0, 0.0),
                test_tctx(),
                None
            ),
            Some(first)
        );
        // Outside everything.
        assert_eq!(
            hit_test(
                &tab.scroll_tree,
                root,
                900.0,
                900.0,
                (0.0, 0.0),
                (0.0, 0.0),
                test_tctx(),
                None
            ),
            None
        );
    }

    #[test]
    fn hover_updates_element_state_and_reports_change() {
        let (tab, first, second) = tab_with_boxes();
        let mut input = InputState::default();

        assert!(update_hover(&mut input, &tab, 400.0, 20.0));
        assert_eq!(input.hover, Some(first));
        assert!(input.element_states[&first].hover);
        // Same target again: no change reported.
        assert!(!update_hover(&mut input, &tab, 410.0, 25.0));
        // Move to the other box: old hover cleared, new set.
        assert!(update_hover(&mut input, &tab, 400.0, 120.0));
        assert!(!input.element_states[&first].hover);
        assert!(input.element_states[&second].hover);
        // Move off all boxes.
        assert!(update_hover(&mut input, &tab, 900.0, 900.0));
        assert_eq!(input.hover, None);
        assert!(!input.element_states[&second].hover);
    }

    #[test]
    fn press_and_release_on_same_node_is_a_click() {
        let (tab, first, _) = tab_with_boxes();
        let mut input = InputState::default();

        assert_eq!(press_at(&mut input, &tab, 400.0, 20.0), Some(first));
        assert_eq!(input.active, Some(first));
        assert_eq!(input.focused, Some(first));
        assert!(input.element_states[&first].active);
        assert!(input.element_states[&first].focus);

        assert_eq!(release_at(&mut input, &tab, 410.0, 25.0), Some(first));
        assert_eq!(input.active, None);
        assert!(!input.element_states[&first].active);
        // Focus survives the click.
        assert_eq!(input.focused, Some(first));
        assert!(input.element_states[&first].focus);
    }

    #[test]
    fn release_on_another_node_is_not_a_click() {
        let (tab, first, _) = tab_with_boxes();
        let mut input = InputState::default();

        assert_eq!(press_at(&mut input, &tab, 400.0, 20.0), Some(first));
        assert_eq!(release_at(&mut input, &tab, 400.0, 120.0), None);
        assert_eq!(input.active, None);
        // Focus follows the press target only.
        assert_eq!(input.focused, Some(first));
        assert!(!input.element_states[&first].active);
    }

    #[test]
    fn wheel_scrolls_without_invalidating() {
        let (mut tab, _, _) = tab_with_boxes();
        tab.dirty = DirtyFlags::default();
        // Fake the content bounds `refresh` would publish for a scrolled
        // document: the max scroll offset is (0, 200).
        tab.scroll().set_max_offset((0.0, 200.0));
        let root_ptr = tab.layout_root.as_deref().map(|r| r as *const _);

        assert_eq!(apply_wheel(&mut tab, 10.0, 10.0, 0.0, 120.0), vec![None]);
        assert_eq!(tab.scroll().offset(), (0.0, 120.0));
        // Clamped at zero ...
        assert_eq!(apply_wheel(&mut tab, 10.0, 10.0, -50.0, -200.0), vec![None]);
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
        // ... and at the content bound: no overscroll past the document.
        assert_eq!(apply_wheel(&mut tab, 10.0, 10.0, 0.0, 10_000.0), vec![None]);
        assert_eq!(tab.scroll().offset(), (0.0, 200.0));
        // Already at the bound: no movement reported.
        assert!(apply_wheel(&mut tab, 10.0, 10.0, 0.0, 1.0).is_empty());
        assert_eq!(tab.scroll().offset(), (0.0, 200.0));
        // Uniform-only: no pipeline stage invalidated, tree untouched.
        assert_eq!(tab.dirty, DirtyFlags::default());
        assert_eq!(tab.layout_root.as_deref().map(|r| r as *const _), root_ptr);
    }

    #[test]
    fn wheel_is_noop_before_first_layout() {
        // Fresh tab: no layout ran, so the max scroll is (0, 0) and the
        // wheel cannot move the offset.
        let mut tab = test_tab();
        tab.dirty = DirtyFlags::default();
        assert!(apply_wheel(&mut tab, 10.0, 10.0, 0.0, 120.0).is_empty());
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
    }

    #[test]
    fn wheel_start_container_prefers_hit_overlay_over_covered_scroller() {
        use bowser_css::parse_stylesheet;
        use bowser_layout::{FixedFontMetrics, layout};
        // A fixed modal painted over a scroll container: the geometric
        // routing (`innermost_at`) would start the wheel at the covered
        // container, but the DOM hit test sees the modal on top, whose
        // scroll container is the viewport.
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div>\
              <div id=\"modal\">m</div></body></html>",
        );
        let sc = doc.get_element_by_id("sc").expect("sc");
        let modal = doc.get_element_by_id("modal").expect("modal");
        let sheet = parse_stylesheet(
            "#sc { overflow-y: auto; height: 100px; width: 200px; } \
             #tall { height: 500px; } \
             #modal { position: fixed; top: 0; left: 0; width: 200px; height: 100px; }",
        );
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        tab.layout_root = Some(Rc::new(root));
        let tree = {
            let doc = tab.document_ref().expect("installed document");
            ScrollTree::build(
                &doc,
                tab.layout_root.as_deref().expect("layout root"),
                tab.viewport,
                tab.scroll(),
                tab.element_scroll(),
            )
        };
        tab.scroll_tree = tree;
        let isc = tab.scroll_tree.index_of(sc).expect("sc is a container");
        // Geometrically the point is inside the scroller...
        assert_eq!(tab.scroll_tree.innermost_at(10.0, 10.0), isc);
        // ...but the hit test finds the fixed modal painted on top, and
        // the wheel starts at the modal's scroll container: the viewport.
        assert_eq!(target_at(&tab, 10.0, 10.0), Some(modal));
        assert_eq!(wheel_start_container(&tab, 10.0, 10.0), 0);
    }

    /// Builds a tab whose layout has a scroll container `#sc`
    /// (100px-tall scrollport, 300px-tall content `#tall`), for the
    /// scrollport-clipping and `scrollend` tests.
    fn tab_with_scroller() -> (Tab, NodeId, NodeId) {
        use bowser_css::parse_stylesheet;
        use bowser_layout::{FixedFontMetrics, layout};
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
        );
        let sc = doc.get_element_by_id("sc").expect("sc");
        let tall = doc.get_element_by_id("tall").expect("tall");
        let sheet = parse_stylesheet(
            "#sc { overflow-y: auto; height: 100px; width: 200px; } \
             #tall { height: 300px; }",
        );
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        tab.layout_root = Some(Rc::new(root));
        let tree = {
            let doc = tab.document_ref().expect("installed document");
            ScrollTree::build(
                &doc,
                tab.layout_root.as_deref().expect("layout root"),
                tab.viewport,
                tab.scroll(),
                tab.element_scroll(),
            )
        };
        tab.scroll_tree = tree;
        (tab, sc, tall)
    }

    #[test]
    fn hit_test_clips_scrolled_out_content_to_scrollport() {
        let (tab, sc, tall) = tab_with_scroller();
        let isc = tab.scroll_tree.index_of(sc).expect("sc is a container");
        let port = tab.scroll_tree.container(isc).scrollport;
        // Scroll the container down 100px (max is 200): `#tall`'s box
        // still geometrically contains points below the scrollport, but
        // that content is painted clipped out.
        tab.scroll_tree.container(isc).state.set_offset(0.0, 100.0);
        let root = tab.layout_root.as_deref().unwrap();

        // Below the scrollport but inside `#tall`'s shifted box: the old
        // hit test found `#tall` here; the clipped hit test must not.
        let x = port.x + 10.0;
        let y_below = port.y + port.h + 10.0;
        let hit = hit_test(
            &tab.scroll_tree,
            root,
            x,
            y_below,
            (0.0, 0.0),
            (0.0, 0.0),
            test_tctx(),
            None,
        );
        assert_ne!(hit, Some(tall));
        assert_ne!(hit, Some(sc));
        // The same point through the viewport-coordinate entry points:
        // clicks target whatever is behind, and the wheel chain starts
        // at the viewport scroller instead of the covered container.
        let target = target_at(&tab, x, y_below);
        assert_ne!(target, Some(tall));
        assert_ne!(target, Some(sc));
        assert_eq!(wheel_start_container(&tab, x, y_below), 0);

        // Positive control: a point inside the scrollport still hits the
        // visible content, and the container itself is hittable on its
        // own box.
        let y_inside = port.y + 50.0;
        assert_eq!(
            hit_test(
                &tab.scroll_tree,
                root,
                x,
                y_inside,
                (0.0, 0.0),
                (0.0, 0.0),
                test_tctx(),
                None
            ),
            Some(tall)
        );
        assert_eq!(target_at(&tab, x, y_inside), Some(tall));
        assert_eq!(wheel_start_container(&tab, x, y_inside), isc);
    }

    // -- nested scroll transform fast path -------------------------------

    /// A tab with a nested scroller, fully painted: the display list is
    /// built and the scroll transform wrapped, so scroll ticks take the
    /// fast path. Returns the tab, `#sc`'s node, and its container index.
    fn tab_with_wrapped_scroller() -> (Tab, NodeId, usize) {
        let (mut tab, sc, _tall) = tab_with_scroller();
        let isc = tab.scroll_tree.index_of(sc).expect("sc is a container");
        let paint_root = tab
            .scroll_tree
            .bake_paint_tree(tab.layout_root.as_deref().expect("layout root"))
            .into_owned();
        tab.paint_root = Some(paint_root);
        tab.dirty.paint = true;
        refresh(&mut tab, &test_fonts());
        assert!(
            tab.scroll_tree.container(isc).transform_fast.get(),
            "#sc has plain in-flow content: eligible for the fast path"
        );
        assert!(
            tab.scroll_tree.container(isc).wrap_item.get().is_some(),
            "#sc content wrapped in a scroll transform"
        );
        (tab, sc, isc)
    }

    /// The scroll transform matrix wrapping container `isc`.
    fn scroll_transform_of(tab: &Tab, isc: usize) -> [f32; 6] {
        use bowser_paint::display_list::DisplayItem;
        let idx = tab
            .scroll_tree
            .container(isc)
            .wrap_item
            .get()
            .expect("container is wrapped");
        match &tab.display_list.as_ref().expect("display list").items[idx] {
            DisplayItem::PushTransform { matrix } => *matrix,
            other => panic!("expected PushTransform, got {other:?}"),
        }
    }

    #[test]
    fn nested_scroll_tick_patches_transform_without_rebuild() {
        let (mut tab, sc, isc) = tab_with_wrapped_scroller();
        let items_before = tab.display_list.as_ref().expect("list").items.len();
        // Scroll the nested container 40px down through the wheel path.
        let scrolled = apply_wheel(&mut tab, 10.0, 10.0, 0.0, 40.0);
        assert_eq!(scrolled, vec![Some(sc)]);
        // No full rebuild: paint stays clean; only the epilogue is stale.
        assert!(
            !tab.dirty.paint,
            "fast path must not rebuild the display list"
        );
        assert!(tab.scroll_tree.scrollbars_dirty());
        // The transform is exactly the scroll delta: no smear.
        assert_eq!(
            scroll_transform_of(&tab, isc),
            [1.0, 0.0, 0.0, 1.0, 0.0, -40.0]
        );
        // The epilogue re-push preserves the patched transform and the
        // item count (truncate + re-emit, no growth).
        refresh(&mut tab, &test_fonts());
        let list = tab.display_list.as_ref().expect("list");
        assert_eq!(list.items.len(), items_before);
        assert_eq!(
            scroll_transform_of(&tab, isc),
            [1.0, 0.0, 0.0, 1.0, 0.0, -40.0]
        );
        // A second tick patches from the absolute live offset.
        let scrolled = apply_wheel(&mut tab, 10.0, 10.0, 0.0, 30.0);
        assert_eq!(scrolled, vec![Some(sc)]);
        assert!(!tab.dirty.paint);
        assert_eq!(
            scroll_transform_of(&tab, isc),
            [1.0, 0.0, 0.0, 1.0, 0.0, -70.0]
        );
    }

    #[test]
    fn nested_scroll_fast_path_scrollbar_tracks_offset() {
        use bowser_paint::display_list::{DisplayItem, scrollbar_thumb_color};
        let (mut tab, _sc, isc) = tab_with_wrapped_scroller();
        let thumb_before = tab
            .scroll_tree
            .scrollbar_geom(isc)
            .vertical
            .expect("vertical bar")
            .thumb;
        apply_wheel(&mut tab, 10.0, 10.0, 0.0, 40.0);
        refresh(&mut tab, &test_fonts());
        let thumb_after = tab
            .scroll_tree
            .scrollbar_geom(isc)
            .vertical
            .expect("vertical bar")
            .thumb;
        assert!(
            thumb_after.y > thumb_before.y,
            "thumb follows the scroll offset"
        );
        // The re-pushed epilogue carries the live thumb rect.
        let list = tab.display_list.as_ref().expect("list");
        let thumbs: Vec<Rect> = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::FillRect { rect, color } if *color == scrollbar_thumb_color() => {
                    Some(*rect)
                }
                _ => None,
            })
            .collect();
        assert!(
            thumbs.contains(&thumb_after),
            "epilogue thumb matches the live geometry"
        );
    }

    #[test]
    fn nested_overscroll_spring_rides_fast_path() {
        use std::time::Duration;
        let (mut tab, sc, isc) = tab_with_wrapped_scroller();
        // Park at the bottom edge, then press past it: the offset stays
        // clamped and the residual stretches the rubber band. (A chain
        // that still consumes offset records no edge press — only full
        // clamps do.)
        tab.scroll_tree.container(isc).state.set_offset(0.0, 200.0);
        sync_nested_paint(&mut tab, &[Some(sc)]);
        assert!(!tab.dirty.paint);
        tab.scroll_tree.scroll_chain(isc, 0.0, 500.0);
        let (_, osy) = tab.scroll_tree.overscroll_shift(isc);
        assert!(osy > 0.0, "edge press stretched the rubber band");
        // Fold like the spring tick does: no rebuild, no scroll events,
        // and the matrix carries the rubber-band shift on top of the
        // offset translation.
        let moved = tab
            .render
            .scroll_tree
            .try_fast_fold(tab.display_list.as_mut().expect("list"))
            .expect("wrapped: fast fold succeeds");
        assert!(moved.is_empty(), "overscroll fires no scroll events");
        assert!(!tab.dirty.paint, "overscroll tick must not rebuild");
        let m = scroll_transform_of(&tab, isc);
        assert!(
            (m[5] - (-200.0 + osy)).abs() < 1e-4,
            "matrix carries offset + rubber-band shift, got {m:?}"
        );
        // Advance the spring: the matrix tracks the decay, still no
        // rebuild.
        let t0 = Instant::now();
        tab.scroll_tree.advance_overscroll(t0);
        let changed = tab
            .scroll_tree
            .advance_overscroll(t0 + Duration::from_millis(16));
        assert!(
            changed.iter().any(|n| n.is_some()),
            "spring moved the nested shift"
        );
        let moved = tab
            .render
            .scroll_tree
            .try_fast_fold(tab.display_list.as_mut().expect("list"))
            .expect("wrapped: fast fold succeeds");
        assert!(moved.is_empty(), "overscroll fires no scroll events");
        let m2 = scroll_transform_of(&tab, isc);
        assert!(
            m2[5] < m[5],
            "matrix tracks the settling spring: {m:?} -> {m2:?}"
        );
        assert!(!tab.dirty.paint);
    }

    #[test]
    fn nested_scroll_sticky_foldable_shift_stays_fast_path() {
        use bowser_css::parse_stylesheet;
        use bowser_layout::{FixedFontMetrics, layout};
        use bowser_paint::display_list::DisplayItem;
        // Wave 39: an in-band sticky box whose scroll-driven shift is
        // exactly foldable stays on the fast path — `sticky_shifts_at`
        // folds its extra shift on top of the scroll transform
        // (`sticky_constrained_at` is fold-aware) — instead of dirtying
        // layout for a re-bake. Only a genuine constraint transition
        // still re-bakes.
        let setup = |html: &[u8], css: &str| {
            let (doc, _) = Document::parse_html(html);
            let sc = doc.get_element_by_id("sc").expect("sc");
            let sheet = parse_stylesheet(css);
            let root = layout(
                &doc,
                &[&sheet],
                Size { w: 800.0, h: 600.0 },
                &FixedFontMetrics,
            );
            let mut tab = test_tab();
            tab.install_test_document(Rc::new(RefCell::new(doc)));
            tab.layout_root = Some(Rc::new(root));
            tab.rebuild_scroll_after_layout();
            let isc = tab.scroll_tree.index_of(sc).expect("sc container");
            assert!(tab.scroll_tree.has_sticky());
            // Paint once so the fast path is armed. Sticky is in-flow,
            // so the container stays eligible.
            tab.dirty.paint = true;
            refresh(&mut tab, &test_fonts());
            assert!(tab.scroll_tree.container(isc).transform_fast.get());
            assert!(tab.scroll_tree.container(isc).wrap_item.get().is_some());
            (tab, sc, isc)
        };
        // Foldable: the sticky box is stuck to the scrollport top at the
        // bake offset, so a small scroll's shift is exactly the scroll
        // delta — no layout re-bake, no repaint.
        let (mut tab, sc, isc) = setup(
            b"<html><body><div id=\"sc\"><div id=\"sticky\">s</div><div id=\"tall\">x</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; width: 200px; } \
             #sticky { position: sticky; top: 0; height: 50px; } \
             #tall { height: 500px; }",
        );
        assert!(
            !tab.scroll_tree.sticky_constrained_at(isc, 0.0, 10.0),
            "foldable in-band sticky must not trip the re-bake gate"
        );
        assert!(
            !tab.scroll_tree
                .container(isc)
                .sticky_wrap_items
                .borrow()
                .is_empty(),
            "the box gets a dedicated fold pair"
        );
        tab.scroll_tree.container(isc).state.set_offset(0.0, 10.0);
        sync_nested_paint(&mut tab, &[Some(sc)]);
        assert!(
            !tab.dirty.layout,
            "foldable sticky shift stays on the fast path"
        );
        assert!(!tab.dirty.paint, "fast fold patches, no repaint");
        // The scroll transform carries the offset; the dedicated sticky
        // pair folds the extra shift on top, so the box still renders at
        // the scrollport top.
        let m = scroll_transform_of(&tab, isc);
        assert!((m[5] - -10.0).abs() < 1e-4, "scroll translation, got {m:?}");
        let sticky_at = tab.scroll_tree.container(isc).sticky_wrap_items.borrow()[0].1;
        let sm = match &tab.display_list.as_ref().expect("list").items[sticky_at] {
            DisplayItem::PushTransform { matrix } => *matrix,
            other => panic!("expected PushTransform, got {other:?}"),
        };
        assert!(
            (sm[5] - 10.0).abs() < 1e-4,
            "sticky extra shift, got {sm:?}"
        );
        // Genuine constraint transition: the box is free at the bake
        // offset (natural at 40, clear of the containing-block clamp)
        // and the tick approaches the stick point — not exactly
        // foldable, so the tick re-bakes via layout.
        let (mut tab, sc, isc) = setup(
            b"<html><body><div id=\"sc\"><div id=\"pad\">x</div><div id=\"sticky\">s</div><div id=\"tall\">x</div></div></body></html>",
            "#sc { overflow-y: auto; height: 100px; width: 200px; } \
             #pad { height: 40px; } \
             #sticky { position: sticky; top: 0; height: 50px; } \
             #tall { height: 500px; }",
        );
        assert!(
            !tab.scroll_tree.sticky_constrained_at(isc, 0.0, 20.0),
            "free interval folds to zero shift"
        );
        assert!(
            tab.scroll_tree.sticky_constrained_at(isc, 0.0, 39.0),
            "approaching the stick transition must trip the re-bake gate"
        );
        tab.scroll_tree.container(isc).state.set_offset(0.0, 39.0);
        sync_nested_paint(&mut tab, &[Some(sc)]);
        assert!(
            tab.dirty.layout,
            "constraint transition falls back to a layout re-bake"
        );
    }

    #[test]
    fn nested_scroll_abspos_child_stays_fast_path() {
        use bowser_css::parse_stylesheet;
        use bowser_layout::{FixedFontMetrics, layout};
        // Wave 39: out-of-flow children no longer disqualify the
        // container. The `auto` z-index abspos child drains inside the
        // container's clip range (covered by the container's own wrap,
        // not even recorded); the viewport-anchored fixed child shifts
        // with no scroller (never recorded). A z-indexed abspos child
        // that drains out of range would get its own transform pair
        // (`OofWrap`). Either way the tick keeps the transform fast path
        // instead of the full rebuild.
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"abs\">a</div><div id=\"tall\">x</div><div id=\"fix\">f</div></div></body></html>",
        );
        let sc = doc.get_element_by_id("sc").expect("sc");
        let sheet = parse_stylesheet(
            "#sc { position: relative; overflow-y: auto; height: 100px; width: 200px; } \
             #abs { position: absolute; top: 0; left: 0; width: 50px; height: 50px; } \
             #tall { height: 500px; } \
             #fix { position: fixed; top: 300px; left: 5px; width: 50px; height: 20px; }",
        );
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        let isc = tab.scroll_tree.index_of(sc).expect("sc container");
        tab.dirty.paint = true;
        refresh(&mut tab, &test_fonts());
        assert!(
            tab.scroll_tree.container(isc).transform_fast.get(),
            "out-of-flow children keep the fast path"
        );
        assert!(tab.scroll_tree.container(isc).wrap_item.get().is_some());
        assert!(
            tab.scroll_tree.container(isc).oof_wraps.borrow().is_empty(),
            "in-range/viewport-anchored boxes need no extra wraps"
        );
        let scrolled = apply_wheel(&mut tab, 10.0, 10.0, 0.0, 40.0);
        assert_eq!(scrolled, vec![Some(sc)]);
        assert!(!tab.dirty.paint, "eligible container rides the fast path");
    }

    #[test]
    fn nested_scroll_rebuild_rewraps_and_goes_fast_again() {
        use bowser_css::parse_stylesheet;
        use bowser_layout::{FixedFontMetrics, layout};
        use bowser_paint::display_list::{Color, DisplayItem};
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
        );
        let sc = doc.get_element_by_id("sc").expect("sc");
        let sheet = parse_stylesheet(
            "#sc { overflow-y: auto; height: 100px; width: 200px; } \
             #tall { height: 400px; } \
             #new { height: 50px; background-color: red; }",
        );
        let do_layout = |doc: &Document| {
            layout(
                doc,
                &[&sheet],
                Size { w: 800.0, h: 600.0 },
                &FixedFontMetrics,
            )
        };
        let root = do_layout(&doc);
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        tab.dirty.paint = true;
        refresh(&mut tab, &test_fonts());
        let isc = tab.scroll_tree.index_of(sc).expect("sc");
        assert!(tab.scroll_tree.container(isc).wrap_item.get().is_some());
        // Fast path works before the mutation.
        let scrolled = apply_wheel(&mut tab, 10.0, 10.0, 0.0, 40.0);
        assert_eq!(scrolled, vec![Some(sc)]);
        assert!(!tab.dirty.paint);
        // A real DOM mutation inside the container: append a styled div,
        // then run the pipeline the way DOM dirt would (relayout, scroll
        // tree rebuild, repaint). The fallback re-bakes, the new content
        // is painted, and the wrap is re-established.
        tab.installed()
            .expect("installed")
            .with_document(|cell| {
                let mut doc = cell.borrow_mut();
                let sc = doc.get_element_by_id("sc").expect("sc");
                let new = doc.create_element("div");
                doc.set_attribute(new, "id", "new");
                doc.append_child(sc, new);
            })
            .expect("installed document is live");
        let root = tab
            .installed()
            .expect("installed")
            .with_document(|cell| do_layout(&cell.borrow()))
            .expect("installed document is live");
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        tab.dirty.paint = true;
        refresh(&mut tab, &test_fonts());
        let isc = tab.scroll_tree.index_of(sc).expect("sc");
        assert!(
            tab.scroll_tree.container(isc).wrap_item.get().is_some(),
            "re-wrapped after the DOM mutation"
        );
        let list = tab.display_list.as_ref().expect("display list");
        assert!(
            list.items.iter().any(|item| matches!(
                item,
                DisplayItem::FillRect { color, .. } if *color == Color::rgba8(255, 0, 0, 255)
            )),
            "mutated content is painted"
        );
        // And the fast path is live again on the new display list.
        let scrolled = apply_wheel(&mut tab, 10.0, 10.0, 0.0, 10.0);
        assert_eq!(scrolled, vec![Some(sc)]);
        assert!(
            !tab.dirty.paint,
            "fast path again after the mutation rebuild"
        );
    }

    #[test]
    fn nested_scroll_nested_containers_compose() {
        use bowser_css::parse_stylesheet;
        use bowser_layout::{FixedFontMetrics, layout};
        // Two nested scrollers, both with plain in-flow content: both
        // wrap, and their transforms compose (inner matrix * outer
        // matrix).
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"outer\"><div id=\"inner\"><div id=\"tall\">x</div></div><div id=\"tall2\">y</div></div></body></html>",
        );
        let outer = doc.get_element_by_id("outer").expect("outer");
        let inner = doc.get_element_by_id("inner").expect("inner");
        let sheet = parse_stylesheet(
            "#outer { overflow-y: auto; height: 200px; width: 300px; } \
             #inner { overflow-y: auto; height: 100px; width: 200px; } \
             #tall { height: 400px; } #tall2 { height: 300px; }",
        );
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        let iouter = tab.scroll_tree.index_of(outer).expect("outer");
        let iinner = tab.scroll_tree.index_of(inner).expect("inner");
        assert_ne!(iouter, iinner);
        tab.dirty.paint = true;
        refresh(&mut tab, &test_fonts());
        assert!(
            tab.scroll_tree.container(iouter).wrap_item.get().is_some(),
            "outer wrapped"
        );
        assert!(
            tab.scroll_tree.container(iinner).wrap_item.get().is_some(),
            "inner wrapped"
        );
        // Scroll the inner 50px via the wheel over it.
        let scrolled = apply_wheel(&mut tab, 10.0, 10.0, 0.0, 50.0);
        assert_eq!(scrolled, vec![Some(inner)]);
        assert!(!tab.dirty.paint);
        assert_eq!(scroll_transform_of(&tab, iinner)[5], -50.0);
        assert_eq!(
            scroll_transform_of(&tab, iouter),
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        );
        // Scroll the outer by wheeling over #tall2, below #inner.
        let scrolled = apply_wheel(&mut tab, 10.0, 150.0, 0.0, 30.0);
        assert_eq!(scrolled, vec![Some(outer)]);
        assert!(!tab.dirty.paint);
        assert_eq!(scroll_transform_of(&tab, iouter)[5], -30.0);
        assert_eq!(
            scroll_transform_of(&tab, iinner)[5],
            -50.0,
            "inner matrix untouched by the outer scroll"
        );
    }

    /// A tab with a nested scroller holding enough content for a full
    /// display-list rebuild to dominate a tick: the benchmark fixture.
    fn tab_with_big_wrapped_scroller() -> (Tab, NodeId, usize) {
        use bowser_css::parse_stylesheet;
        use bowser_layout::{FixedFontMetrics, layout};
        let mut html = String::from("<html><body><div id=\"sc\">");
        for i in 0..300 {
            html.push_str(&format!("<p>paragraph {i} with some text</p>"));
        }
        html.push_str("</div></body></html>");
        let (doc, _) = Document::parse_html(html.as_bytes());
        let sc = doc.get_element_by_id("sc").expect("sc");
        let sheet = parse_stylesheet(
            "#sc { overflow-y: auto; height: 100px; width: 200px; } p { height: 20px; }",
        );
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        let isc = tab.scroll_tree.index_of(sc).expect("sc container");
        tab.dirty.paint = true;
        refresh(&mut tab, &test_fonts());
        assert!(tab.scroll_tree.container(isc).wrap_item.get().is_some());
        (tab, sc, isc)
    }

    #[test]
    fn nested_scroll_tick_cost_benchmark() {
        // Wall-clock comparison on one fixture: fast-path ticks
        // (transform patch + epilogue re-push) vs legacy ticks
        // (paint-tree fold + full display-list rebuild).
        let (mut tab, sc, isc) = tab_with_big_wrapped_scroller();
        let fonts = test_fonts();
        const N: u32 = 200;
        // Fast path.
        let t0 = Instant::now();
        for i in 0..N {
            let y = (i % 100) as f32;
            tab.scroll_tree.container(isc).state.set_offset(0.0, y);
            sync_nested_paint(&mut tab, &[Some(sc)]);
            refresh(&mut tab, &fonts);
        }
        let fast = t0.elapsed();
        assert!(!tab.dirty.paint, "benchmark stayed on the fast path");
        // Legacy path: fold + rebuild per tick (re-wraps each time, so
        // the comparison includes the wrap cost, like production did).
        let t0 = Instant::now();
        for i in 0..N {
            let y = (i % 100) as f32;
            tab.scroll_tree.container(isc).state.set_offset(0.0, y);
            let paint = tab.render.paint_root.as_mut().expect("paint root");
            tab.render.scroll_tree.sync_paint_offsets(paint);
            tab.dirty.paint = true;
            refresh(&mut tab, &fonts);
        }
        let legacy = t0.elapsed();
        eprintln!("nested tick x{N}: fast={fast:?} legacy={legacy:?}");
        // Generous ratio bound instead of a strict ordering assertion: the
        // fast path must win by at least 4x so a pathological CI stall
        // (both sides slowed ~equally) cannot flip the comparison, while
        // the actual margin (~2700x here) still validates the fast path.
        assert!(
            fast * 4 < legacy,
            "fast path must beat the full rebuild by 4x: fast={fast:?} legacy={legacy:?}"
        );
    }

    /// Builds a tab whose scroll container `#sc` (100px-tall scrollport,
    /// 500px-tall content `#tall`) sits inside `#wrap`, which has
    /// `transform: translate(100px, 50px)`.
    fn tab_with_translated_scroller() -> (Tab, NodeId, NodeId) {
        use bowser_css::parse_stylesheet;
        use bowser_layout::{FixedFontMetrics, layout};
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"wrap\"><div id=\"sc\">\
              <div id=\"tall\">x</div></div></div></body></html>",
        );
        let sc = doc.get_element_by_id("sc").expect("sc");
        let tall = doc.get_element_by_id("tall").expect("tall");
        let sheet = parse_stylesheet(
            "#wrap { transform: translate(100px, 50px); } \
             #sc { overflow-y: auto; height: 100px; width: 200px; } \
             #tall { height: 500px; }",
        );
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        tab.layout_root = Some(Rc::new(root));
        let tree = {
            let doc = tab.document_ref().expect("installed document");
            ScrollTree::build(
                &doc,
                tab.layout_root.as_deref().expect("layout root"),
                tab.viewport,
                tab.scroll(),
                tab.element_scroll(),
            )
        };
        tab.scroll_tree = tree;
        (tab, sc, tall)
    }

    #[test]
    fn hit_test_follows_translate_around_nested_scroller() {
        let (tab, sc, tall) = tab_with_translated_scroller();
        let isc = tab.scroll_tree.index_of(sc).expect("sc is a container");
        let port = tab.scroll_tree.container(isc).scrollport;
        // Visual location of a point inside the translated scrollport.
        let (vx, vy) = (port.x + 100.0 + 10.0, port.y + 50.0 + 10.0);
        // The DOM hit test finds the translated content...
        assert_eq!(target_at(&tab, vx, vy), Some(tall));
        // ...and the wheel chain starts at the translated scroller.
        assert_eq!(wheel_start_container(&tab, vx, vy), isc);
        // The pre-transform location hits whatever is behind instead,
        // and the wheel chain starts at the viewport.
        let target = target_at(&tab, port.x + 10.0, port.y + 10.0);
        assert_ne!(target, Some(tall));
        assert_ne!(target, Some(sc));
        assert_eq!(wheel_start_container(&tab, port.x + 10.0, port.y + 10.0), 0);
    }

    #[test]
    fn wheel_scrolls_translated_nested_scroller_at_visual_location() {
        let (mut tab, sc, _) = tab_with_translated_scroller();
        let isc = tab.scroll_tree.index_of(sc).expect("sc is a container");
        let port = tab.scroll_tree.container(isc).scrollport;
        let (vx, vy) = (port.x + 100.0 + 10.0, port.y + 50.0 + 10.0);
        tab.dirty = DirtyFlags::default();
        // A wheel at the visual location scrolls the translated
        // container (max offset is 400: 500px of content in 100px).
        let moved = apply_wheel(&mut tab, vx, vy, 0.0, 120.0);
        assert_eq!(moved, vec![Some(sc)]);
        assert_eq!(tab.scroll_tree.container(isc).state.offset(), (0.0, 120.0));
    }

    #[test]
    fn scrollend_fires_on_quiet_tick_after_scroll() {
        let (tab, sc, _) = tab_with_scroller();
        let isc = tab.scroll_tree.index_of(sc).expect("sc is a container");
        let port = tab.scroll_tree.container(isc).scrollport;
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let stub = StubDispatch::default();
        let log = stub.log.clone();
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, Some(Box::new(stub)));

        // Scroll the nested container: `scroll` fires, `scrollend` does not.
        let x = port.x + 10.0;
        let y = port.y + 50.0;
        let scrolled = {
            let tab = app.tabs.get_mut(tab_id).expect("tab");
            apply_wheel(tab, x, y, 0.0, 50.0)
        };
        assert_eq!(scrolled, vec![Some(sc)]);
        app.fire_scroll_listeners_for(tab_id, &scrolled);
        assert_eq!(log.borrow().scrolls, vec![sc]);
        assert!(log.borrow().scrollends.is_empty());

        // End of the event-loop tick: the container moved this tick, so
        // no `scrollend` yet.
        app.pump_scripts();
        assert!(log.borrow().scrollends.is_empty());

        // A quiet tick: `scrollend` fires for the nested container...
        app.pump_scripts();
        assert_eq!(log.borrow().scrollends, vec![sc]);

        // ...and only once: further quiet ticks fire nothing.
        app.pump_scripts();
        assert_eq!(log.borrow().scrollends.len(), 1);
    }

    #[test]
    fn scrollend_does_not_fire_without_scroll() {
        // A tab that never scrolled: quiet ticks never produce `scrollend`.
        let (tab, _, _) = tab_with_scroller();
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let stub = StubDispatch::default();
        let log = stub.log.clone();
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, Some(Box::new(stub)));

        app.pump_scripts();
        app.pump_scripts();
        assert!(log.borrow().scrollends.is_empty());
        assert!(log.borrow().scrolls.is_empty());
    }

    #[test]
    fn scrollbar_thumb_drag_drives_container_offset() {
        // Full pointer-event path: press on the thumb starts a drag
        // (not a DOM press), cursor motion scrolls the container,
        // release ends the gesture without a click.
        let (tab, sc, _) = tab_with_scroller();
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, tab_id);

        // Viewport offset is zero: viewport coords == page coords.
        let (thumb_cx, thumb_cy, thumb_h, track_h) = {
            let tab = app.tabs.get(tab_id).expect("tab");
            let isc = tab.scroll_tree.index_of(sc).expect("sc container");
            let v = tab.scroll_tree.scrollbar_geom(isc).vertical.expect("bar");
            (
                v.thumb.x + v.thumb.w / 2.0,
                v.thumb.y + v.thumb.h / 2.0,
                v.thumb.h,
                v.track.h,
            )
        };
        // `#sc`: 100px scrollport, 300px content, max offset 200.
        assert!((thumb_h - 100.0 * 100.0 / 300.0).abs() < 0.01);
        assert!((track_h - 100.0).abs() < 0.01);

        // Press on the thumb: a drag starts, the DOM press path is skipped.
        app.on_cursor_moved(thumb_cx, thumb_cy);
        app.on_mouse_button(true, MouseButton::Left);
        assert!(app.scrollbar_drag.is_some(), "thumb press starts a drag");
        assert!(
            app.input.active.is_none(),
            "scrollbar press must not press DOM content"
        );

        // Drag down 40px: thumb travel is 100 - 33.33 px over a 200px
        // range, so the offset moves by 40 * 200 / 66.67 = 120.
        app.on_cursor_moved(thumb_cx, thumb_cy + 40.0);
        let offset = {
            let tab = app.tabs.get(tab_id).expect("tab");
            let isc = tab.scroll_tree.index_of(sc).expect("sc container");
            tab.scroll_tree.container(isc).state.offset()
        };
        assert!(
            (offset.1 - 120.0).abs() < 0.5,
            "drag scrolls the container, got {offset:?}"
        );

        // Release: the gesture ends, no click dispatches.
        app.on_mouse_button(false, MouseButton::Left);
        assert!(app.scrollbar_drag.is_none(), "release ends the drag");
        assert!(app.input.active.is_none(), "no click target armed");
    }

    #[test]
    fn autoscroll_delta_dead_zone_and_scaling() {
        let anchor = (100.0, 100.0);
        // At the anchor and inside the dead zone: no scroll.
        assert_eq!(autoscroll_delta(anchor, (100.0, 100.0)), (0.0, 0.0));
        assert_eq!(autoscroll_delta(anchor, (108.0, 100.0)), (0.0, 0.0));
        assert_eq!(autoscroll_delta(anchor, (104.0, 104.0)), (0.0, 0.0));
        // Past the dead zone: proportional to the excess distance,
        // pointing away from the anchor.
        let (dx, dy) = autoscroll_delta(anchor, (100.0, 118.0));
        assert_eq!(dx, 0.0);
        let want = (18.0 - AUTOSCROLL_DEADZONE) * AUTOSCROLL_RATE;
        assert!((dy - want).abs() < 1e-6, "got {dy}, want {want}");
        // Twice the excess distance: twice the delta.
        let (_, dy2) = autoscroll_delta(anchor, (100.0, 128.0));
        assert!((dy2 - 2.0 * want).abs() < 1e-6, "got {dy2}");
        // Up/left of the anchor scrolls negative.
        let (dx3, dy3) = autoscroll_delta(anchor, (70.0, 100.0));
        assert!(dx3 < 0.0 && dy3 == 0.0, "got ({dx3}, {dy3})");
        let (dx4, dy4) = autoscroll_delta(anchor, (100.0, 70.0));
        assert!(dx4 == 0.0 && dy4 < 0.0, "got ({dx4}, {dy4})");
    }

    #[test]
    fn middle_click_autoscroll_scrolls_anchored_container() {
        // Full gesture path: middle press anchors, motion beyond the
        // dead zone scrolls the container under the anchor through the
        // scroll chain, release ends the gesture.
        let (tab, sc, _) = tab_with_scroller();
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, tab_id);

        // Anchor at the center of `#sc`'s scrollport (100px tall,
        // 300px content, max offset 200).
        let (ax, ay) = {
            let tab = app.tabs.get(tab_id).expect("tab");
            let isc = tab.scroll_tree.index_of(sc).expect("sc container");
            let port = tab.scroll_tree.container(isc).scrollport;
            (port.x + port.w / 2.0, port.y + port.h / 2.0)
        };
        let offset_y = |app: &WindowApp<'_>| {
            let tab = app.tabs.get(tab_id).expect("tab");
            let isc = tab.scroll_tree.index_of(sc).expect("sc container");
            tab.scroll_tree.container(isc).state.offset().1
        };

        app.on_cursor_moved(ax, ay);
        app.on_mouse_button(true, MouseButton::Middle);
        assert!(app.autoscroll.is_some(), "middle press anchors autoscroll");
        // A middle press is not a DOM press: no active node, no click.
        assert!(app.input.active.is_none());

        // Motion inside the dead zone: no scroll.
        app.on_cursor_moved(ax, ay + AUTOSCROLL_DEADZONE);
        assert_eq!(offset_y(&app), 0.0);

        // Motion 60px below the anchor: the per-event delta is
        // (60 - deadzone) * rate along the vector.
        app.on_cursor_moved(ax, ay + 60.0);
        let want = (60.0 - AUTOSCROLL_DEADZONE) * AUTOSCROLL_RATE;
        let got = offset_y(&app);
        assert!(
            (got - want).abs() < 1e-3,
            "autoscroll scrolled {got}, want {want}"
        );

        // Release ends the gesture: further motion scrolls nothing.
        app.on_mouse_button(false, MouseButton::Middle);
        assert!(app.autoscroll.is_none(), "release ends autoscroll");
        app.on_cursor_moved(ax, ay + 120.0);
        let got = offset_y(&app);
        assert!(
            (got - want).abs() < 1e-3,
            "no scroll after release, got {got}"
        );
    }

    #[test]
    fn middle_click_autoscroll_ends_on_escape() {
        let (tab, sc, _) = tab_with_scroller();
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, tab_id);

        let (ax, ay) = {
            let tab = app.tabs.get(tab_id).expect("tab");
            let isc = tab.scroll_tree.index_of(sc).expect("sc container");
            let port = tab.scroll_tree.container(isc).scrollport;
            (port.x + port.w / 2.0, port.y + port.h / 2.0)
        };

        app.on_cursor_moved(ax, ay);
        app.on_mouse_button(true, MouseButton::Middle);
        assert!(app.autoscroll.is_some());

        app.on_key(&Key::Named(NamedKey::Escape), true);
        assert!(app.autoscroll.is_none(), "Escape ends autoscroll");

        // Escape with no active gesture keeps its normal path: no
        // gesture starts, nothing panics.
        app.on_key(&Key::Named(NamedKey::Escape), true);
        assert!(app.autoscroll.is_none());
    }

    /// Like [`tab_with_scroller`], but `#sc` is a mandatory
    /// scroll-snap container: snap areas at content y=0 and y=100
    /// (both `scroll-snap-align: start`), 100px scrollport, max y=200.
    fn tab_with_snap_scroller() -> (Tab, NodeId, NodeId) {
        use bowser_css::parse_stylesheet;
        use bowser_layout::{FixedFontMetrics, layout};
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div><div id=\"gap\">y</div></div></body></html>",
        );
        let sc = doc.get_element_by_id("sc").expect("sc");
        let a = doc.get_element_by_id("a").expect("a");
        let sheet = parse_stylesheet(
            "#sc { overflow-y: auto; height: 100px; width: 200px; \
               scroll-snap-type: y mandatory; } \
             #a { height: 100px; scroll-snap-align: start; } \
             #gap { height: 300px; }",
        );
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        tab.layout_root = Some(Rc::new(root));
        let tree = {
            let doc = tab.document_ref().expect("installed document");
            ScrollTree::build(
                &doc,
                tab.layout_root.as_deref().expect("layout root"),
                tab.viewport,
                tab.scroll(),
                tab.element_scroll(),
            )
        };
        tab.scroll_tree = tree;
        (tab, sc, a)
    }

    #[test]
    fn scrollbar_thumb_drag_does_not_snap_mid_gesture() {
        // Full pointer-event path on a mandatory snap container: drag
        // motions must not launch snap smooth-scrolls (they would
        // judder the content away from the cursor); the gesture
        // settles with one snap evaluation on release.
        let (tab, sc, _) = tab_with_snap_scroller();
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, tab_id);

        let (thumb_cx, thumb_cy) = {
            let tab = app.tabs.get(tab_id).expect("tab");
            let isc = tab.scroll_tree.index_of(sc).expect("sc container");
            let v = tab.scroll_tree.scrollbar_geom(isc).vertical.expect("bar");
            (v.thumb.x + v.thumb.w / 2.0, v.thumb.y + v.thumb.h / 2.0)
        };

        // Press on the thumb, drag down 40px: thumb travel is
        // 100 - 25 px over a 300px range, so the offset moves by
        // 40 * 300 / 75 = 160.
        app.on_cursor_moved(thumb_cx, thumb_cy);
        app.on_mouse_button(true, MouseButton::Left);
        assert!(app.scrollbar_drag.is_some(), "thumb press starts a drag");
        app.on_cursor_moved(thumb_cx, thumb_cy + 40.0);
        {
            let tab = app.tabs.get(tab_id).expect("tab");
            let isc = tab.scroll_tree.index_of(sc).expect("sc container");
            let offset = tab.scroll_tree.container(isc).state.offset();
            assert!(
                (offset.1 - 160.0).abs() < 0.5,
                "drag scrolls the container, got {offset:?}"
            );
            assert!(
                !tab.scroll_tree.has_smooth_in_flight(),
                "no snap smooth-scroll may fight the drag mid-gesture"
            );
        }

        // Release: the gesture ends and settles with one snap
        // evaluation — offset 160 snaps back to the y=0 area.
        app.on_mouse_button(false, MouseButton::Left);
        assert!(app.scrollbar_drag.is_none(), "release ends the drag");
        let tab = app.tabs.get(tab_id).expect("tab");
        assert!(
            tab.scroll_tree.has_smooth_in_flight(),
            "release settles with one snap evaluation"
        );
    }

    #[test]
    fn scrollbar_track_click_pages_container() {
        // Press on the track (not the thumb): the container pages one
        // scrollport toward the click, with no drag starting.
        let (tab, sc, _) = tab_with_scroller();
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, tab_id);

        let (cx, below_thumb, above_thumb) = {
            let tab = app.tabs.get(tab_id).expect("tab");
            let isc = tab.scroll_tree.index_of(sc).expect("sc container");
            let v = tab.scroll_tree.scrollbar_geom(isc).vertical.expect("bar");
            let cx = v.thumb.x + v.thumb.w / 2.0;
            // Thumb starts at the track top (offset 0).
            assert!((v.thumb.y - v.track.y).abs() < 0.01);
            (cx, v.thumb.y + v.thumb.h + 10.0, v.track.y + 10.0)
        };

        // Click below the thumb: page down one 100px scrollport.
        app.on_cursor_moved(cx, below_thumb);
        app.on_mouse_button(true, MouseButton::Left);
        assert!(app.scrollbar_drag.is_none(), "track click is not a drag");
        app.on_mouse_button(false, MouseButton::Left);
        let offset = {
            let tab = app.tabs.get(tab_id).expect("tab");
            let isc = tab.scroll_tree.index_of(sc).expect("sc container");
            tab.scroll_tree.container(isc).state.offset()
        };
        assert!(
            (offset.1 - 100.0).abs() < 0.01,
            "track click pages down, got {offset:?}"
        );

        // Click above the thumb: page back up to zero.
        app.on_cursor_moved(cx, above_thumb);
        app.on_mouse_button(true, MouseButton::Left);
        app.on_mouse_button(false, MouseButton::Left);
        let offset = {
            let tab = app.tabs.get(tab_id).expect("tab");
            let isc = tab.scroll_tree.index_of(sc).expect("sc container");
            tab.scroll_tree.container(isc).state.offset()
        };
        assert!(
            offset.1.abs() < 0.01,
            "track click pages up, got {offset:?}"
        );
    }

    #[test]
    fn pump_scripts_advances_smooth_scroll_to_completion() {
        let (tab, sc, _) = tab_with_scroller();
        let isc = tab.scroll_tree.index_of(sc).expect("sc container");
        // #sc's max is (0, 200): animate the full range.
        tab.scroll_tree.start_smooth_scroll(isc, 0.0, 200.0);
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let stub = StubDispatch::default();
        let log = stub.log.clone();
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, Some(Box::new(stub)));

        // The animation is in flight (or already moved): the pump keeps
        // the loop alive either way.
        app.pump_scripts();
        assert!(app.needs_redraw);

        // Drive pumps until the animation completes: the animation runs
        // on wall-clock time, so poll rather than assuming a fixed
        // number of ticks (scheduling delays under load make exact tick
        // counts meaningless).
        for _ in 0..100 {
            if !app
                .tabs
                .get(tab_id)
                .expect("tab")
                .scroll_tree
                .has_smooth_in_flight()
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
            app.pump_scripts();
        }
        let tab = app.tabs.get(tab_id).expect("tab");
        assert!(!tab.scroll_tree.has_smooth_in_flight());
        assert_eq!(tab.scroll_tree.container(isc).state.offset(), (0.0, 200.0));
        // `scroll` fired through the normal funnel while the animation
        // moved (once per moving tick, always for this container).
        assert!(!log.borrow().scrolls.is_empty());
        assert!(log.borrow().scrolls.iter().all(|&n| n == sc));

        // One quiet tick later `scrollend` fires for the container,
        // exactly once.
        app.pump_scripts();
        assert_eq!(log.borrow().scrollends, vec![sc]);
    }

    #[test]
    fn share_cookie_jar_reexport_retains_jar_in_js_context() {
        // w77-J: the `bowser_js::share_cookie_jar` re-export is the
        // app-side half of the W-E2 FontFace.load credentials wiring:
        // before it, `bowser_app` could not even name the function.
        // Retention is the observable: the context holds an `Arc`
        // clone, so the strong count rises, proving the JS context
        // received this exact jar (not a copy). NOTE: this proves the
        // share function, not the production install point — no
        // production install calls it yet (see the w77-J report:
        // `install_loaded_document` takes no jar, and `Profile` holds
        // a plain `Mutex`, not the `Arc` the share takes).
        let (doc, _) = Document::parse_html(b"<html><body><p>hi</p></body></html>");
        let doc = Rc::new(RefCell::new(doc));
        let url = bowser_url::parse("http://example.com/", None).expect("test url");
        let page_vm = crate::script::PageVm::new_for_document(doc, &url, "test", &[])
            .expect("install web apis");
        let ctx = page_vm.web_api_context().expect("web api ctx");
        let jar: Arc<Mutex<CookieJar>> = Arc::new(Mutex::new(CookieJar::new()));
        assert_eq!(Arc::strong_count(&jar), 1);
        bowser_js::share_cookie_jar(ctx, Arc::clone(&jar));
        assert!(
            Arc::strong_count(&jar) >= 2,
            "the JS context must retain the shared jar"
        );
        // Unknown contexts are ignored, never a panic.
        bowser_js::share_cookie_jar(u32::MAX, Arc::clone(&jar));
    }

    #[test]
    fn headless_drain_applies_pending_element_smooth_scroll_instantly() {
        use bowser_css::parse_stylesheet;
        use bowser_layout::{FixedFontMetrics, layout};

        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"outer\"><div id=\"tall\">x</div></div></body></html>",
        );
        let doc = Rc::new(RefCell::new(doc));
        let sheet = parse_stylesheet(
            "#outer { overflow-y: auto; height: 100px; } #tall { height: 300px; }",
        );
        let root = layout(
            &doc.borrow(),
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let url = bowser_url::parse("http://example.com/", None).expect("test url");
        let page_vm = crate::script::PageVm::new_for_document(doc.clone(), &url, "test", &[])
            .expect("install web apis");
        let mut tab = test_tab();
        // Wire the tab's scroll states into the page realm, like
        // navigation does at document install.
        if let Some(ctx) = page_vm.web_api_context() {
            bowser_js::share_scroll_state(ctx, Rc::clone(tab.scroll()));
            bowser_js::share_element_scroll_states(ctx, Rc::clone(tab.element_scroll()));
        }
        tab.install_document(doc, page_vm);
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();

        let outer = tab
            .installed()
            .expect("installed document")
            .with_document(|cell| cell.borrow().get_element_by_id("outer").expect("div#outer"))
            .expect("installed document is live");
        let isc = tab.scroll_tree.index_of(outer).expect("outer container");
        assert_eq!(
            tab.element_scroll()
                .state_for(outer)
                .expect("outer is a scroll container")
                .max_offset(),
            (0.0, 200.0)
        );

        // Queue a smooth scroll through the real binding: the request
        // sits pending — no animation ticks run headless.
        tab.page_vm_mut()
            .expect("page vm")
            .eval("document.getElementById('outer').scrollTo({top: 200, behavior: 'smooth'});")
            .expect("queue smooth scroll");
        assert_eq!(
            tab.scroll_tree.container(isc).state.offset(),
            (0.0, 0.0),
            "smooth request is pending, not applied"
        );

        // The headless drain degrades smooth→instant (documented
        // headless policy): the offset lands on the target, and no
        // animation is left in flight for advance_smooth.
        drain_pending_element_smooth_scroll_headless(&mut tab);
        assert_eq!(
            tab.scroll_tree.container(isc).state.offset(),
            (0.0, 200.0),
            "headless drain applies the smooth scroll target instantly"
        );
        assert!(!tab.scroll_tree.has_smooth_in_flight());
        // The queue is consumed: a second drain is a no-op.
        drain_pending_element_smooth_scroll_headless(&mut tab);
        assert_eq!(tab.scroll_tree.container(isc).state.offset(), (0.0, 200.0));
    }

    #[test]
    fn animations_in_flight_covers_background_tabs_and_touch() {
        // Regression: only the active tab's smooth animations kept the
        // event loop turning, so a background tab's smooth scroll
        // stalled mid-flight. The predicate must see every live tab —
        // plus touch fling momentum.
        let (bg_tab, sc, _) = tab_with_scroller();
        let isc = bg_tab.scroll_tree.index_of(sc).expect("sc container");
        bg_tab.scroll_tree.start_smooth_scroll(isc, 0.0, 200.0);
        let (fg_tab, _, _) = tab_with_scroller();
        let mut tabs = TabArena::default();
        let _bg_id = tabs.insert(bg_tab);
        let fg_id = tabs.insert(fg_tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        // Active tab is the foreground one, with no animation of its own.
        let mut app = test_app(&mut tabs, &jar, fg_id);
        assert!(
            !app.tabs
                .get(fg_id)
                .expect("fg")
                .scroll_tree
                .has_smooth_in_flight()
        );
        assert!(
            app.animations_in_flight(),
            "background smooth scroll keeps the loop turning"
        );

        // Touch fling momentum counts too, on the active tab.
        let t = Instant::now();
        app.touch.started(
            9,
            0.0,
            400.0,
            t,
            None,
            crate::scroll_touch::TouchHit::permissive(),
        );
        {
            let tab = app.tabs.get_mut(fg_id).expect("fg");
            app.touch.moved(
                &tab.scroll_tree,
                9,
                0.0,
                100.0,
                t + std::time::Duration::from_millis(50),
            );
        }
        app.touch
            .ended(9, 0.0, 100.0, t + std::time::Duration::from_millis(50));
        assert!(app.touch.touch_has_animation_in_flight());
        assert!(app.animations_in_flight());

        // Nothing in flight anywhere: the loop may idle.
        let mut tabs = TabArena::default();
        let fg_id = tabs.insert(test_tab());
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let app = test_app(&mut tabs, &jar, fg_id);
        assert!(!app.animations_in_flight());
    }

    #[test]
    fn pump_scripts_advances_overscroll_spring_and_keeps_loop_alive() {
        let (mut tab, sc, _) = tab_with_scroller();
        let isc = tab.scroll_tree.index_of(sc).expect("sc container");
        // Park at the max edge and bake clean; a press past the edge
        // then stretches the nested container's transient shift.
        tab.scroll_tree.container(isc).state.scroll_by(0.0, 200.0);
        let root = tab.layout_root.as_deref().expect("layout root");
        tab.paint_root = Some(tab.scroll_tree.bake_paint_tree(root).into_owned());
        tab.scroll_tree.scroll_chain(isc, 0.0, 500.0);
        assert!(tab.scroll_tree.has_nested_overscroll_in_flight());

        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, tab_id);
        assert!(
            app.animations_in_flight(),
            "overscroll spring keeps the loop turning"
        );

        // The pump folds the stretch into the paint tree and schedules
        // the redraw that renders it...
        app.pump_scripts();
        assert!(app.needs_redraw);
        // ...then the spring settles on wall-clock time and releases
        // the loop. Polled at ~16ms: the spring integrates with
        // semi-implicit Euler, which needs small steps (the 50ms
        // cadence of the smooth-scroll pump test above would make the
        // stiff spring diverge instead of settle).
        for _ in 0..400 {
            if !app
                .tabs
                .get(tab_id)
                .expect("tab")
                .scroll_tree
                .has_overscroll_in_flight()
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(16));
            app.pump_scripts();
        }
        let tab = app.tabs.get(tab_id).expect("tab");
        assert!(!tab.scroll_tree.has_overscroll_in_flight());
        assert!(
            tab.dirty.paint,
            "nested overscroll dirtied paint while settling"
        );
        assert!(!app.animations_in_flight());
    }

    #[test]
    fn viewport_overscroll_keeps_loop_alive_without_paint_dirt() {
        let (tab, _, _) = tab_with_scroller();
        // The document is not scrollable: the whole delta presses past
        // the viewport edge and stretches only its transient shift.
        tab.scroll_tree.scroll_chain(0, 0.0, 500.0);
        assert!(tab.scroll_tree.has_overscroll_in_flight());
        assert!(!tab.scroll_tree.has_nested_overscroll_in_flight());

        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, tab_id);
        assert!(
            app.animations_in_flight(),
            "viewport overscroll keeps the loop turning"
        );
        app.pump_scripts();
        // The viewport shift rides the per-frame GPU uniform: the pump
        // schedules the redraw but dirties no pipeline stage...
        assert!(app.needs_redraw);
        assert!(!app.tabs.get(tab_id).expect("tab").dirty.paint);
        // ...and the spring still settles on its own (~16ms pump
        // cadence for integrator stability, as above).
        for _ in 0..400 {
            if !app
                .tabs
                .get(tab_id)
                .expect("tab")
                .scroll_tree
                .has_overscroll_in_flight()
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(16));
            app.pump_scripts();
        }
        let tab = app.tabs.get(tab_id).expect("tab");
        assert!(!tab.scroll_tree.has_overscroll_in_flight());
        assert!(!app.animations_in_flight());
    }

    #[test]
    fn pump_scripts_smooth_scroll_interrupted_by_wheel() {
        // Interruption through the real input path: a wheel over the
        // animating container writes the offset directly, so the pump's
        // animation tick sees the foreign write and cancels instead of
        // advancing — the wheel's offset wins. (Mid-flight cancellation
        // is covered deterministically in `scroll.rs`; this proves the
        // pump path honors it.)
        let (tab, sc, _) = tab_with_scroller();
        let isc = tab.scroll_tree.index_of(sc).expect("sc container");
        tab.scroll_tree.start_smooth_scroll(isc, 0.0, 200.0);
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let stub = StubDispatch::default();
        let log = stub.log.clone();
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, Some(Box::new(stub)));
        let port = app
            .tabs
            .get(tab_id)
            .expect("tab")
            .scroll_tree
            .container(isc)
            .scrollport;

        // The wheel's own `scroll` dispatch is covered by the existing
        // wheel tests; here it is only the interrupting write.
        {
            let tab = app.tabs.get_mut(tab_id).expect("tab");
            apply_wheel(tab, port.x + 10.0, port.y + 10.0, 0.0, 30.0);
        }
        app.pump_scripts();
        let tab = app.tabs.get(tab_id).expect("tab");
        assert_eq!(tab.scroll_tree.container(isc).state.offset(), (0.0, 30.0));
        assert!(!tab.scroll_tree.has_smooth_in_flight());
        assert!(log.borrow().scrolls.is_empty());
    }

    /// `scroll::rebuild_after_layout` on a layout without scrollable
    /// overflow: the tree stays viewport-only, the document maximum is
    /// the layout extent minus the viewport, a stale offset is
    /// re-clamped, and the paint tree is baked for the display list.
    #[test]
    fn rebuild_after_layout_publishes_max_and_bakes_paint() {
        let (doc, _) = Document::parse_html(b"<html><body></body></html>");
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        tab.layout_root = Some(Rc::new(layout_box(
            rect(0.0, 0.0, 800.0, 600.0),
            None,
            vec![layout_box(rect(0.0, 0.0, 900.0, 1500.0), None, vec![])],
        )));
        // Stale overscrolled offset: the rebuild clamps it into the new
        // range instead of leaving it stranded.
        tab.scroll().set_max_offset((5000.0, 5000.0));
        tab.scroll().set_offset(4000.0, 4000.0);
        tab.rebuild_scroll_after_layout();
        assert_eq!(tab.scroll().max_offset(), (100.0, 900.0));
        assert_eq!(tab.scroll().offset(), (100.0, 900.0));
        assert!(tab.scroll_tree.is_viewport_only());
        assert!(tab.paint_root().is_some());

        // A shrunken tree unscrolls the tab.
        tab.layout_root = Some(Rc::new(layout_box(
            rect(0.0, 0.0, 800.0, 600.0),
            None,
            vec![],
        )));
        tab.rebuild_scroll_after_layout();
        assert_eq!(tab.scroll().max_offset(), (0.0, 0.0));
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
    }

    /// Tab with a document-backed page VM and a `scrollIntoView()`
    /// request recorded through page script — the same shape a
    /// timer/click handler produces during `pump_scripts`. The layout
    /// tree puts the target node at y=500 with scroll headroom to 1400.
    fn tab_with_scroll_into_view() -> (Tab, NodeId, u32) {
        let (doc, _) = Document::parse_html(b"<html><body><div id=\"d\">x</div></body></html>");
        let doc = Rc::new(RefCell::new(doc));
        let el = doc.borrow().get_element_by_id("d").expect("div#d");
        let url = bowser_url::parse("http://example.com/", None).expect("test url");
        let mut page_vm = crate::script::PageVm::new_for_document(doc.clone(), &url, "test", &[])
            .expect("install web apis");
        page_vm
            .eval("document.getElementById('d').scrollIntoView();")
            .expect("scrollIntoView");
        let ctx = page_vm.web_api_context().expect("web api ctx");
        let mut tab = test_tab();
        tab.install_document(doc, page_vm);
        tab.layout_root = Some(Rc::new(layout_box(
            rect(0.0, 0.0, 800.0, 2000.0),
            None,
            vec![layout_box(rect(0.0, 500.0, 800.0, 50.0), Some(el), vec![])],
        )));
        tab.scroll().set_max_offset((0.0, 1400.0));
        (tab, el, ctx)
    }

    #[test]
    fn scroll_into_view_waits_for_clean_layout() {
        let (mut tab, _, ctx) = tab_with_scroll_into_view();
        // Same-tick DOM mutation: layout has not re-run yet. Start
        // scrolled partway down so a degradation to scroll-to-top is
        // observable.
        tab.dirty.layout = true;
        tab.scroll().set_offset(0.0, 300.0);
        let before = tab.scroll().offset();
        // The drain must not consume the request ...
        assert!(drain_scroll_into_view(&mut tab).is_empty());
        // ... and must not move the offset: the pre-fix code resolved
        // against the stale tree, consumed the request, and degraded to
        // scroll-to-top — silently converting the most common
        // scrollIntoView() usage pattern into scroll-to-top.
        assert_eq!(tab.scroll().offset(), before);
        assert!(
            bowser_js::take_scroll_into_view_request(ctx).is_some(),
            "request must stay pending until layout runs"
        );
    }

    #[test]
    fn scroll_into_view_drains_on_clean_layout() {
        let (mut tab, _, ctx) = tab_with_scroll_into_view();
        tab.dirty.layout = false;
        assert!(!drain_scroll_into_view(&mut tab).is_empty());
        assert_eq!(tab.scroll().offset(), (0.0, 500.0));
        // Consumed: nothing left for the next tick.
        assert_eq!(bowser_js::take_scroll_into_view_request(ctx), None);
    }

    /// A script-driven nested scroll fires the element's `scroll`
    /// listener exactly once. The bindings mutate the shared offset but
    /// never invoke listeners (that double-dispatched every
    /// script-driven scroll: once synchronously from the scroll native,
    /// once from the host's per-tick offset diff); the host is the sole
    /// owner of `scroll` event delivery.
    #[test]
    fn script_driven_element_scroll_fires_listener_exactly_once() {
        use bowser_css::parse_stylesheet;
        use bowser_layout::{FixedFontMetrics, layout};

        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"outer\"><div id=\"tall\">x</div></div></body></html>",
        );
        let doc = Rc::new(RefCell::new(doc));
        let sheet = parse_stylesheet(
            "#outer { overflow-y: auto; height: 100px; } #tall { height: 300px; }",
        );
        let root = layout(
            &doc.borrow(),
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let url = bowser_url::parse("http://example.com/", None).expect("test url");
        let page_vm = crate::script::PageVm::new_for_document(doc.clone(), &url, "test", &[])
            .expect("install web apis");
        let mut tab = test_tab();
        // Wire the tab's scroll states into the page realm, like
        // navigation does at document install.
        if let Some(ctx) = page_vm.web_api_context() {
            bowser_js::share_scroll_state(ctx, Rc::clone(tab.scroll()));
            bowser_js::share_element_scroll_states(ctx, Rc::clone(tab.element_scroll()));
        }
        tab.install_document(doc, page_vm);
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();

        // Sole-ownership install: the document is reached only through
        // the tab's scoped handle — no strong alias retained past install.
        let outer = tab
            .installed()
            .expect("installed document")
            .with_document(|cell| cell.borrow().get_element_by_id("outer").expect("div#outer"))
            .expect("installed document is live");
        // Sanity: the layout made #outer a nested scroll container with
        // 200px of vertical headroom.
        assert_eq!(
            tab.element_scroll()
                .state_for(outer)
                .expect("outer is a scroll container")
                .max_offset(),
            (0.0, 200.0)
        );

        let read_scrolled = |tab: &mut Tab| -> i32 {
            match tab
                .page_vm_mut()
                .expect("page vm")
                .eval("scrolled")
                .expect("read scrolled")
            {
                Value::Int(n) => n,
                v => panic!("scrolled must be an int, got {v:?}"),
            }
        };
        tab.page_vm_mut()
            .expect("page vm")
            .eval("scrolled = 0; document.getElementById('outer').addEventListener('scroll', function() { scrolled = scrolled + 1; });")
            .expect("register listener");

        // Script-driven scroll: mutates the shared offset...
        tab.page_vm_mut()
            .expect("page vm")
            .eval("document.getElementById('outer').scrollTop = 50;")
            .expect("script scroll");
        assert_eq!(
            tab.element_scroll()
                .state_for(outer)
                .expect("outer state")
                .offset(),
            (0.0, 50.0)
        );
        assert_eq!(
            read_scrolled(&mut tab),
            0,
            "bindings must not dispatch scroll listeners synchronously"
        );

        // ...and the host tick dispatches it exactly once. This mirrors
        // `pump_scripts`: fold the offsets, then dispatch for the moved
        // containers the way `fire_scroll_listeners_for` does. (The
        // production pre-first-layout viewport fallback is unreachable
        // here: `paint_root` exists.)
        let script_scrolled = match tab.render.paint_root.as_mut() {
            Some(paint) => tab.render.scroll_tree.sync_paint_offsets(paint),
            None => Vec::new(),
        };
        assert_eq!(script_scrolled, vec![Some(outer)]);
        let mut dispatch = crate::script::PageEventDispatch::new();
        for target in script_scrolled {
            match target {
                Some(node) => dispatch.dispatch_scroll(&mut tab, node),
                None => panic!("only the nested container moved"),
            }
        }
        assert_eq!(
            read_scrolled(&mut tab),
            1,
            "host must dispatch the scroll listener exactly once"
        );

        // A second tick with no movement dispatches nothing.
        let script_scrolled = match tab.render.paint_root.as_mut() {
            Some(paint) => tab.render.scroll_tree.sync_paint_offsets(paint),
            None => Vec::new(),
        };
        assert!(script_scrolled.is_empty(), "no movement, no events");
        assert_eq!(read_scrolled(&mut tab), 1);
    }

    /// `refresh` publishes the fresh layout tree to the page's web APIs:
    /// before the first layout pass the CSSOM View geometry getters
    /// report 0 (no tree published yet); after `refresh` runs the layout
    /// pass they measure the laid-out tree.
    #[test]
    fn refresh_publishes_layout_tree_to_js_geometry_getters() {
        let mut tab = test_tab();
        let url = bowser_url::parse("http://example.com/", None).expect("test url");
        // Production install path: web APIs installed, scroll states
        // shared, everything dirty so the next `refresh` lays out.
        install_welcome_page(
            &mut tab,
            &url,
            "BowserTest/1.0",
            &test_fonts(),
            &test_cookie_jar(),
        );
        assert!(
            tab.page_vm().and_then(|vm| vm.web_api_context()).is_some(),
            "web APIs must install for the publish to have a target"
        );

        let offset_width = |tab: &mut Tab| -> i32 {
            match tab
                .page_vm_mut()
                .expect("page vm")
                .eval("document.documentElement.offsetWidth")
                .expect("eval offsetWidth")
            {
                Value::Int(n) => n,
                v => panic!("offsetWidth must be an int, got {v:?}"),
            }
        };

        // No layout has run yet: the getters report 0, the honest
        // no-layout answer.
        assert_eq!(offset_width(&mut tab), 0);

        refresh(&mut tab, &test_fonts());

        // The layout pass published the tree: `<html>` spans the 800px
        // viewport.
        assert_eq!(offset_width(&mut tab), 800);
        // A second layout pass re-publishes the fresh tree without
        // disturbing the reading.
        tab.dirty.layout = true;
        refresh(&mut tab, &test_fonts());
        assert_eq!(offset_width(&mut tab), 800);
    }

    /// When a tab has a web API context but no laid-out tree yet,
    /// `publish_layout_tree` clears the previously published tree: the
    /// CSSOM View geometry getters must report 0 (the no-layout answer),
    /// never a stale tree from a prior document/layout.
    #[test]
    fn publish_layout_tree_clears_stale_tree_without_layout_root() {
        let mut tab = test_tab();
        let url = bowser_url::parse("http://example.com/", None).expect("test url");
        install_welcome_page(
            &mut tab,
            &url,
            "BowserTest/1.0",
            &test_fonts(),
            &test_cookie_jar(),
        );
        assert!(
            tab.page_vm().and_then(|vm| vm.web_api_context()).is_some(),
            "web APIs must install for the publish to have a target"
        );

        let offset_width = |tab: &mut Tab| -> i32 {
            match tab
                .page_vm_mut()
                .expect("page vm")
                .eval("document.documentElement.offsetWidth")
                .expect("eval offsetWidth")
            {
                Value::Int(n) => n,
                v => panic!("offsetWidth must be an int, got {v:?}"),
            }
        };

        refresh(&mut tab, &test_fonts());
        assert_eq!(offset_width(&mut tab), 800);

        // The context exists but nothing has been laid out yet: drop the
        // layout root and publish, exercising the previously-early
        // return branch.
        tab.layout_root = None;
        publish_layout_tree(&tab);

        // The stale tree is gone: the getter reports 0 instead of the
        // dead tree's 800px.
        assert_eq!(offset_width(&mut tab), 0);

        // A subsequent layout pass re-publishes a live tree.
        tab.dirty.layout = true;
        tab.layout_root = None;
        refresh(&mut tab, &test_fonts());
        assert_eq!(offset_width(&mut tab), 800);
    }

    #[test]
    fn key_scroll_fallback_pans_viewport() {
        let mut tab = test_tab();
        tab.scroll().set_max_offset((500.0, 1000.0));

        assert_eq!(
            apply_key_scroll_with_shift(&mut tab, None, "ArrowDown", false),
            vec![None]
        );
        assert_eq!(tab.scroll().offset(), (0.0, 40.0));
        assert_eq!(
            apply_key_scroll_with_shift(&mut tab, None, "ArrowUp", false),
            vec![None]
        );
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
        assert_eq!(
            apply_key_scroll_with_shift(&mut tab, None, "ArrowRight", false),
            vec![None]
        );
        assert_eq!(tab.scroll().offset(), (40.0, 0.0));
        assert_eq!(
            apply_key_scroll_with_shift(&mut tab, None, "ArrowLeft", false),
            vec![None]
        );
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));

        // Page keys move by the viewport height (600).
        assert_eq!(
            apply_key_scroll_with_shift(&mut tab, None, "PageDown", false),
            vec![None]
        );
        assert_eq!(tab.scroll().offset(), (0.0, 600.0));
        assert_eq!(
            apply_key_scroll_with_shift(&mut tab, None, "PageUp", false),
            vec![None]
        );
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
        assert_eq!(
            apply_key_scroll_with_shift(&mut tab, None, " ", false),
            vec![None]
        );
        assert_eq!(tab.scroll().offset(), (0.0, 600.0));

        // Home/End jump to the edges.
        assert_eq!(
            apply_key_scroll_with_shift(&mut tab, None, "End", false),
            vec![None]
        );
        assert_eq!(tab.scroll().offset(), (0.0, 1000.0));
        assert_eq!(
            apply_key_scroll_with_shift(&mut tab, None, "Home", false),
            vec![None]
        );
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
        // Already at the top: Home reports no movement.
        assert!(apply_key_scroll_with_shift(&mut tab, None, "Home", false).is_empty());

        // Non-scrolling keys are ignored.
        assert!(apply_key_scroll_with_shift(&mut tab, None, "a", false).is_empty());
        assert!(apply_key_scroll_with_shift(&mut tab, None, "Enter", false).is_empty());
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
    }

    /// Keyboard scrolling targets the focused element's scroll container,
    /// not the viewport: arrows pan the nested scroller and invalidate
    /// paint (its offset is baked into the paint tree).
    #[test]
    fn key_scroll_targets_focused_container() {
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div><div id=\"pad\">y</div></body></html>",
        );
        let sheet = bowser_css::parse_stylesheet(
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 500px; } \
             #pad { height: 2000px; }",
        );
        let root = bowser_layout::layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let sc = doc.get_element_by_id("sc").expect("sc");
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        let isc = tab.scroll_tree.index_of(sc).expect("sc container");

        let scrolled = apply_key_scroll_with_shift(&mut tab, Some(sc), "ArrowDown", false);
        assert_eq!(scrolled, vec![Some(sc)]);
        assert_eq!(tab.scroll_tree.container(isc).state.offset(), (0.0, 40.0));
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
        assert!(tab.dirty.paint, "nested scroll invalidates paint");

        // Unfocused: the viewport pans instead, paint untouched.
        tab.dirty.paint = false;
        let scrolled = apply_key_scroll_with_shift(&mut tab, None, "ArrowDown", false);
        assert_eq!(scrolled, vec![None]);
        assert!(!tab.dirty.paint);
    }

    /// Home/End honor the `user_scroll` mask: a focused
    /// `overflow: hidden` container never consumes them (the old direct
    /// write scrolled it, unlike the arrows) — the delta chains past it
    /// to the viewport.
    #[test]
    fn key_home_end_respects_hidden_mask() {
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"h\"><div id=\"tall\">x</div></div><div id=\"pad\">y</div></body></html>",
        );
        let sheet = bowser_css::parse_stylesheet(
            "#h { overflow: hidden; height: 100px; } #tall { height: 500px; } \
             #pad { height: 2000px; }",
        );
        let root = bowser_layout::layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let h = doc.get_element_by_id("h").expect("h");
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        let ih = tab.scroll_tree.index_of(h).expect("h container");
        assert_eq!(tab.scroll_tree.container(ih).user_scroll, (false, false));
        let (_, doc_max_y) = tab.scroll().max_offset();
        assert!(doc_max_y > 0.0);

        // End with focus inside the hidden container: the container must
        // not move — the delta chains past it to the viewport.
        let scrolled = apply_key_scroll_with_shift(&mut tab, Some(h), "End", false);
        assert_eq!(scrolled, vec![None]);
        assert_eq!(tab.scroll_tree.container(ih).state.offset(), (0.0, 0.0));
        assert_eq!(tab.scroll().offset(), (0.0, doc_max_y));

        // Home chains back the same way.
        let scrolled = apply_key_scroll_with_shift(&mut tab, Some(h), "Home", false);
        assert_eq!(scrolled, vec![None]);
        assert_eq!(tab.scroll_tree.container(ih).state.offset(), (0.0, 0.0));
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
    }

    /// Sequential focus navigation order: positive `tabindex` first in
    /// ascending order (stable on ties), then zero/implicit focusables in
    /// document order.
    #[test]
    fn focus_order_positive_tabindex_first() {
        let (doc, _) = Document::parse_html(
            b"<button id=\"a\" tabindex=\"2\">a</button>\
              <button id=\"b\">b</button>\
              <button id=\"c\" tabindex=\"1\">c</button>\
              <div id=\"d\" tabindex=\"0\">d</div>\
              <button id=\"e\" tabindex=\"1\">e</button>",
        );
        let id = |name: &str| doc.get_element_by_id(name).expect(name);
        assert_eq!(
            focus_order(&doc),
            vec![id("c"), id("e"), id("a"), id("b"), id("d")]
        );
    }

    /// Sequential focus navigation skips negative `tabindex` (script-focus
    /// only), `disabled` form controls, `input type=hidden`, and anchors
    /// without `href`; an unparseable `tabindex` behaves as absent.
    #[test]
    fn focus_order_skips_unfocusable() {
        let (doc, _) = Document::parse_html(
            b"<button id=\"neg\" tabindex=\"-1\">x</button>\
              <button id=\"dis\" disabled>y</button>\
              <input id=\"hid\" type=\"hidden\">\
              <a id=\"nohref\">z</a>\
              <a id=\"href\" href=\"https://example.com/\">w</a>\
              <div id=\"bad\" tabindex=\"abc\">v</div>\
              <button id=\"weird\" tabindex=\"abc\">u</button>\
              <div id=\"plain\">t</div>\
              <button id=\"ok\">s</button>",
        );
        let id = |name: &str| doc.get_element_by_id(name).expect(name);
        assert_eq!(focus_order(&doc), vec![id("href"), id("weird"), id("ok")]);
    }

    /// Tab moves focus forward through the order, Shift+Tab backward,
    /// wrapping at both ends; the `:focus` pseudo-class state follows the
    /// focused node and clears on the old one.
    #[test]
    fn focus_nav_walks_order_and_wraps() {
        let (doc, _) =
            Document::parse_html(b"<button id=\"a\">a</button><button id=\"b\">b</button>");
        let id = |name: &str| doc.get_element_by_id(name).expect(name);
        let (a, b) = (id("a"), id("b"));
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        let mut input = InputState::default();

        // Nothing focused: Tab starts at the first focusable, Shift+Tab
        // at the last.
        apply_focus_nav(&mut input, &mut tab, false);
        assert_eq!(input.focused, Some(a));
        let mut input2 = InputState::default();
        apply_focus_nav(&mut input2, &mut tab, true);
        assert_eq!(input2.focused, Some(b));

        // Forward, then wrap past the end.
        apply_focus_nav(&mut input, &mut tab, false);
        assert_eq!(input.focused, Some(b));
        apply_focus_nav(&mut input, &mut tab, false);
        assert_eq!(input.focused, Some(a));
        // Backward wraps too, moving `:focus` with it.
        apply_focus_nav(&mut input, &mut tab, true);
        assert_eq!(input.focused, Some(b));
        assert!(state_for(&mut input, b).focus);
        assert!(!state_for(&mut input, a).focus);
    }

    /// Tab scrolls the newly focused element into view through the
    /// existing reveal path: focusing a below-the-fold button pans the
    /// viewport, reported innermost-first for `scroll` listeners.
    #[test]
    fn focus_nav_reveals_newly_focused_element() {
        let (doc, _) = Document::parse_html(
            b"<html><body><button id=\"top\">top</button>\
              <div id=\"pad\"></div><button id=\"low\">low</button>\
              </body></html>",
        );
        let sheet = bowser_css::parse_stylesheet("#pad { height: 2000px; }");
        let root = bowser_layout::layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let top = doc.get_element_by_id("top").expect("top");
        let low = doc.get_element_by_id("low").expect("low");
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        // Mimic the post-`refresh` state `scroll_to_node` expects: layout
        // is clean, so the stale-layout guard lets the reveal through.
        tab.dirty.layout = false;

        let mut input = InputState::default();
        // First Tab focuses `top`, already visible: nothing scrolls.
        let scrolled = apply_focus_nav(&mut input, &mut tab, false);
        assert_eq!(input.focused, Some(top));
        assert!(scrolled.is_empty());
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));

        // Second Tab focuses `low` below the fold: the viewport pans to
        // reveal it, reported as the viewport (`None`) for listeners.
        let scrolled = apply_focus_nav(&mut input, &mut tab, false);
        assert_eq!(input.focused, Some(low));
        assert_eq!(scrolled, vec![None]);
        assert!(tab.scroll().offset().1 > 0.0);
    }

    /// Home/End chain like the other scrolling keys: End on a nested
    /// container already at its edge moves only the viewport, and Home
    /// walks back down the same chain.
    #[test]
    fn key_end_chains_past_saturated_nested_container() {
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div><div id=\"pad\">y</div></body></html>",
        );
        let sheet = bowser_css::parse_stylesheet(
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 300px; } \
             #pad { height: 2000px; }",
        );
        let root = bowser_layout::layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let sc = doc.get_element_by_id("sc").expect("sc");
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        let isc = tab.scroll_tree.index_of(sc).expect("sc container");
        let (_, doc_max_y) = tab.scroll().max_offset();
        assert!(doc_max_y > 0.0);

        // End drives #sc to its edge, then chains the remainder to the
        // viewport — both move in one gesture.
        let scrolled = apply_key_scroll_with_shift(&mut tab, Some(sc), "End", false);
        assert_eq!(scrolled, vec![Some(sc), None]);
        assert_eq!(tab.scroll_tree.container(isc).state.offset(), (0.0, 200.0));
        assert_eq!(tab.scroll().offset(), (0.0, doc_max_y));

        // End again: everything is at its edge, nothing moves.
        assert!(apply_key_scroll_with_shift(&mut tab, Some(sc), "End", false).is_empty());

        // Home walks back down the same chain.
        let scrolled = apply_key_scroll_with_shift(&mut tab, Some(sc), "Home", false);
        assert_eq!(scrolled, vec![Some(sc), None]);
        assert_eq!(tab.scroll_tree.container(isc).state.offset(), (0.0, 0.0));
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
    }

    /// Script-driven element scrolls (`el.scrollTop = ...` mutating the
    /// shared cells through the bindings, bypassing wheel/keyboard) are
    /// folded on the next `pump_scripts`: the paint tree syncs without a
    /// relayout, paint is invalidated, and the element `scroll` listener
    /// fires — mirroring the wheel/keyboard path.
    #[test]
    fn pump_scripts_folds_script_driven_element_scroll() {
        fn find_box(
            root: &bowser_layout::LayoutBox,
            node: bowser_dom::NodeId,
        ) -> Option<&bowser_layout::LayoutBox> {
            if root.node == Some(node) {
                return Some(root);
            }
            root.children.iter().find_map(|c| find_box(c, node))
        }

        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
        );
        let sheet = bowser_css::parse_stylesheet(
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 500px; }",
        );
        let root = bowser_layout::layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let sc = doc.get_element_by_id("sc").expect("sc");
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        let isc = tab.scroll_tree.index_of(sc).expect("sc container");

        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let stub = StubDispatch::default();
        let log = stub.log.clone();
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, Some(Box::new(stub)));

        // Simulate the bindings' `el.scrollTop = 40` write: the shared
        // cell mutates behind the tab's back.
        app.tabs
            .get_mut(tab_id)
            .expect("tab")
            .element_scroll()
            .state_for(sc)
            .expect("element state")
            .set_offset(0.0, 40.0);
        app.pump_scripts();

        let tab = app.tabs.get(tab_id).expect("tab");
        assert_eq!(tab.scroll_tree.container(isc).state.offset(), (0.0, 40.0));
        // The paint tree synced without a relayout: the content moved up
        // by the new offset while the layout tree is untouched.
        let tall = tab
            .document_ref()
            .as_ref()
            .expect("document")
            .get_element_by_id("tall")
            .expect("tall");
        let paint = tab.paint_root.as_ref().expect("paint root");
        let layout = tab.layout_root.as_deref().expect("layout root");
        let py = find_box(paint, tall).expect("tall in paint").rect.y;
        let ly = find_box(layout, tall).expect("tall in layout").rect.y;
        assert_eq!(py, ly - 40.0);
        assert!(tab.dirty.paint, "script scroll invalidates paint");
        // ...and the element `scroll` listener fired exactly once.
        assert_eq!(log.borrow().scrolls, vec![sc]);
    }

    /// Fragment navigation into a nested scroll container folds the
    /// movement into the paint tree immediately, like the
    /// wheel/keyboard/`scrollIntoView()` paths: the baked paint offsets
    /// match the live offsets at once, paint is invalidated, the
    /// element's `scroll` listener fires exactly once, and the next
    /// `pump_scripts` tick finds no drift (so it fires no duplicate).
    #[test]
    fn fragment_navigation_into_nested_scroller_syncs_paint() {
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"frag\">x</div><div id=\"pad\">y</div></div></body></html>",
        );
        let sheet = bowser_css::parse_stylesheet(
            "#sc { overflow-y: auto; height: 100px; } \
             #frag { margin-top: 300px; } \
             #pad { height: 500px; }",
        );
        let root = bowser_layout::layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let sc = doc.get_element_by_id("sc").expect("sc");
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        // The fragment fast path compares against the current history
        // entry.
        tab.history
            .push(bowser_url::parse("https://example.com/page", None).unwrap());
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        let isc = tab.scroll_tree.index_of(sc).expect("sc container");

        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let stub = StubDispatch::default();
        let log = stub.log.clone();
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, Some(Box::new(stub)));

        // Same-document link: the fragment fast path, no worker load.
        app.navigate_link("#frag");
        assert!(
            app.pending.is_none(),
            "fragment click must not spawn a load"
        );

        let tab = app.tabs.get(tab_id).expect("tab");
        // The anchor sits below the fold inside the nested container:
        // the container revealed it.
        let (ox, oy) = tab.scroll_tree.container(isc).state.offset();
        assert_eq!(ox, 0.0);
        assert!(oy > 0.0, "nested container revealed the anchor");
        assert!(tab.dirty.paint, "nested fragment scroll invalidates paint");
        // The element's `scroll` listener fired exactly once.
        assert_eq!(log.borrow().scrolls, vec![sc]);
        // The paint tree is already in sync with the live offsets: a
        // re-sync moves nothing.
        let tab = app.tabs.get_mut(tab_id).expect("tab");
        let drift = {
            let paint = tab.render.paint_root.as_mut().expect("paint root");
            tab.render.scroll_tree.sync_paint_offsets(paint)
        };
        assert!(drift.is_empty(), "paint tree already matches live offsets");

        // The next script pump sees live == baked: no duplicate event.
        app.pump_scripts();
        assert_eq!(
            log.borrow().scrolls,
            vec![sc],
            "no second scroll event from the script pump"
        );
    }

    #[test]
    fn resize_invalidates_layout_and_paint() {
        let (mut tab, _, _) = tab_with_boxes();
        tab.dirty = DirtyFlags::default();

        apply_resize(&mut tab, 1024.0, 768.0);
        assert_eq!(
            tab.viewport,
            Size {
                w: 1024.0,
                h: 768.0
            }
        );
        assert!(tab.dirty.layout && tab.dirty.paint);
        assert!(!tab.dirty.style, "pure size change does not restyle");
    }

    /// Throwaway font cache for [`refresh`] tests (no window owns one).
    fn test_fonts() -> Arc<Mutex<FontCache>> {
        Arc::new(Mutex::new(FontCache::new()))
    }

    /// Throwaway profile cookie jar handle for install call sites
    /// (w77-L: `install_loaded_document` takes the shared `Arc`).
    fn test_cookie_jar() -> Arc<Mutex<CookieJar>> {
        Arc::new(Mutex::new(CookieJar::new()))
    }

    #[test]
    fn refresh_runs_dirty_stages_and_clears_flags() {
        let mut tab = test_tab();
        let (doc, _) = Document::parse_html(
            b"<html><head><title>t</title></head><body><p>hi</p></body></html>",
        );
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        mark_all_dirty(&mut tab);

        refresh(&mut tab, &test_fonts());
        assert!(tab.layout_root.is_some());
        assert!(tab.display_list.is_some());
        assert_eq!(tab.dirty, DirtyFlags::default());

        // Second refresh with no dirty flags is a no-op.
        let root_ptr = tab.layout_root.as_deref().map(|r| r as *const _);
        refresh(&mut tab, &test_fonts());
        assert_eq!(tab.layout_root.as_deref().map(|r| r as *const _), root_ptr);
    }

    /// Preorder geometry snapshot of a layout tree: (node, border-box
    /// rect, text) per box — the differential comparison unit for the
    /// partial-layout tests.
    fn geometry_snapshot(root: &LayoutBox) -> Vec<(Option<NodeId>, Rect, Option<String>)> {
        let mut out = Vec::new();
        let mut stack = vec![root];
        while let Some(box_) = stack.pop() {
            out.push((
                box_.node,
                box_.rect,
                box_.text.as_deref().map(str::to_owned),
            ));
            stack.extend(box_.children.iter().rev());
        }
        out
    }

    fn leaf_tab() -> (Tab, NodeId, NodeId) {
        use bowser_css::parse_stylesheet;
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"wrap\"><p id=\"leaf\">hello world</p>\
              <p id=\"sib\">other text</p></div></body></html>",
        );
        let leaf = doc.get_element_by_id("leaf").expect("leaf");
        let text = doc.first_child(leaf).expect("leaf text");
        let mut tab = test_tab();
        tab.author.push(parse_stylesheet(
            "body { margin: 0; } #wrap { padding: 10px; } p { margin: 0; }",
        ));
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        (tab, leaf, text)
    }

    fn set_text(tab: &Tab, text: NodeId, value: &str) {
        tab.installed()
            .expect("document")
            .with_document(|cell| cell.borrow_mut().set_text_content(text, value))
            .expect("tab owns document");
    }

    fn set_attr(tab: &Tab, el: NodeId, name: &str, value: &str) {
        tab.installed()
            .expect("document")
            .with_document(|cell| cell.borrow_mut().set_attribute(el, name, value))
            .expect("tab owns document");
    }

    fn element_by_id(tab: &Tab, id: &str) -> NodeId {
        tab.installed()
            .expect("document")
            .with_document(|cell| cell.borrow().get_element_by_id(id).expect("element"))
            .expect("tab owns document")
    }

    #[test]
    fn refresh_partial_splices_size_stable_text_edit() {
        let (mut tab, leaf, text) = leaf_tab();
        let fonts = test_fonts();
        mark_all_dirty(&mut tab);
        refresh(&mut tab, &fonts);
        assert_eq!(tab.layout_full_passes, 1);
        assert_eq!(tab.layout_partial_passes, 0);
        let leaf_before = geometry_snapshot(tab.layout_root.as_deref().expect("layout"))
            .into_iter()
            .find(|(n, _, _)| *n == Some(leaf))
            .expect("leaf geometry");

        // Size-stable text edit: same single line, same block height.
        set_text(&tab, text, "goodbye world");
        tab.sync_dirty_flags();
        assert_eq!(tab.dirty_subtrees, vec![text], "text dirt retained");
        assert!(!tab.dirty_full_fallback);
        refresh(&mut tab, &fonts);
        assert_eq!(
            tab.layout_partial_passes, 1,
            "size-stable leaf edit must take the partial path"
        );
        assert_eq!(tab.layout_full_passes, 1, "no full pass for the edit");
        assert!(tab.boxes_spliced >= 2, "p box plus its text box spliced");
        assert!(tab.dirty_subtrees.is_empty(), "report serviced");
        let partial_snapshot = geometry_snapshot(tab.layout_root.as_deref().expect("layout"));
        // The spliced subtree carries the new text at the old geometry.
        let leaf_after = partial_snapshot
            .iter()
            .find(|(n, _, _)| *n == Some(leaf))
            .expect("leaf geometry after splice");
        assert_eq!(leaf_after.1, leaf_before.1, "leaf box did not move");
        assert!(
            partial_snapshot
                .iter()
                .any(|(_, _, t)| t.as_deref() == Some("goodbye world")),
            "spliced tree carries the new text"
        );

        // Differential: a forced full layout of the same document must
        // produce exactly the same geometry, box by box.
        mark_all_dirty(&mut tab);
        refresh(&mut tab, &fonts);
        assert_eq!(tab.layout_full_passes, 2);
        let full_snapshot = geometry_snapshot(tab.layout_root.as_deref().expect("layout"));
        assert_eq!(partial_snapshot, full_snapshot);
    }

    #[test]
    fn refresh_partial_falls_back_on_size_change() {
        let (mut tab, leaf, text) = leaf_tab();
        let fonts = test_fonts();
        mark_all_dirty(&mut tab);
        refresh(&mut tab, &fonts);
        let height_before = geometry_snapshot(tab.layout_root.as_deref().expect("layout"))
            .into_iter()
            .find(|(n, _, _)| *n == Some(leaf))
            .expect("leaf geometry")
            .1
            .h;

        // A long replacement wraps to many lines: the leaf grows, so
        // the splice is refused and the full path re-places siblings.
        set_text(&tab, text, &"word ".repeat(120));
        tab.sync_dirty_flags();
        refresh(&mut tab, &fonts);
        assert_eq!(tab.layout_partial_passes, 0, "size change must not splice");
        assert_eq!(tab.layout_full_passes, 2, "fallback ran the full path");
        let height_after = geometry_snapshot(tab.layout_root.as_deref().expect("layout"))
            .into_iter()
            .find(|(n, _, _)| *n == Some(leaf))
            .expect("leaf geometry after fallback")
            .1
            .h;
        assert!(
            height_after > height_before + 1.0,
            "full layout grew the leaf: {height_before} -> {height_after}"
        );
    }

    #[test]
    fn refresh_partial_splices_paint_only_class_change() {
        use bowser_css::parse_stylesheet;
        let (mut tab, leaf, _text) = leaf_tab();
        // Paint-only rule: flipping this class cannot move any box.
        // (The fixture stylesheet has no sibling combinators or
        // `:has()`, so the subtree-locality precondition holds.)
        tab.author.push(parse_stylesheet(".red { color: red; }"));
        let fonts = test_fonts();
        mark_all_dirty(&mut tab);
        refresh(&mut tab, &fonts);
        assert_eq!(tab.layout_full_passes, 1);
        assert_eq!(tab.layout_partial_passes, 0);
        let before = geometry_snapshot(tab.layout_root.as_deref().expect("layout"));

        set_attr(&tab, leaf, "class", "red");
        tab.sync_dirty_flags();
        assert!(tab.dirty.style, "class write sets style dirt");
        assert_eq!(tab.dirty_subtrees, vec![leaf]);
        assert_eq!(
            tab.style_subtrees,
            vec![leaf],
            "class write records a localized restyle root"
        );
        refresh(&mut tab, &fonts);
        assert_eq!(
            tab.layout_partial_passes, 1,
            "paint-only class change must take the splice path"
        );
        assert_eq!(tab.layout_full_passes, 1, "no full pass for the change");
        assert!(tab.dirty_subtrees.is_empty(), "report serviced");
        assert!(!tab.dirty.style, "style flag serviced");

        // Differential: the spliced tree must match a forced full
        // layout of the same document exactly.
        let partial_snapshot = geometry_snapshot(tab.layout_root.as_deref().expect("layout"));
        assert_eq!(partial_snapshot, before, "paint-only change moved nothing");
        mark_all_dirty(&mut tab);
        refresh(&mut tab, &fonts);
        assert_eq!(tab.layout_full_passes, 2);
        assert_eq!(
            partial_snapshot,
            geometry_snapshot(tab.layout_root.as_deref().expect("layout"))
        );
    }

    #[test]
    fn refresh_partial_falls_back_on_display_changing_class() {
        use bowser_css::parse_stylesheet;
        let (mut tab, leaf, _text) = leaf_tab();
        tab.author
            .push(parse_stylesheet(".gone { display: none; }"));
        let fonts = test_fonts();
        mark_all_dirty(&mut tab);
        refresh(&mut tab, &fonts);
        assert_eq!(tab.layout_full_passes, 1);

        // `display` is box-affecting: the splice path must refuse even
        // though the restyle scope is localized.
        set_attr(&tab, leaf, "class", "gone");
        tab.sync_dirty_flags();
        assert!(tab.dirty.style);
        assert_eq!(tab.style_subtrees, vec![leaf]);
        refresh(&mut tab, &fonts);
        assert_eq!(
            tab.layout_partial_passes, 0,
            "display-changing class must not splice"
        );
        assert_eq!(tab.layout_full_passes, 2, "fallback ran the full path");
        assert!(
            geometry_snapshot(tab.layout_root.as_deref().expect("layout"))
                .iter()
                .all(|(n, _, _)| *n != Some(leaf)),
            "leaf box is gone after display:none"
        );
    }

    #[test]
    fn refresh_partial_falls_back_on_sibling_selector_stylesheet() {
        use bowser_css::parse_stylesheet;
        let (mut tab, leaf, _text) = leaf_tab();
        // A sibling combinator anywhere in the stylesheet set voids the
        // subtree-locality precondition: the class write below could
        // restyle `#sib` via `.red + p`, so the splice must refuse.
        tab.author.push(parse_stylesheet(
            ".red { color: red; } .red + p { color: blue; }",
        ));
        let fonts = test_fonts();
        mark_all_dirty(&mut tab);
        refresh(&mut tab, &fonts);
        assert_eq!(tab.layout_full_passes, 1);

        set_attr(&tab, leaf, "class", "red");
        tab.sync_dirty_flags();
        refresh(&mut tab, &fonts);
        assert_eq!(
            tab.layout_partial_passes, 0,
            "sibling-selector stylesheet must force the full path"
        );
        assert_eq!(tab.layout_full_passes, 2);
    }

    #[test]
    fn refresh_partial_restyles_subtree_on_inherited_font_size_change() {
        let (mut tab, leaf, _text) = leaf_tab();
        let wrap = element_by_id(&tab, "wrap");
        let fonts = test_fonts();
        mark_all_dirty(&mut tab);
        refresh(&mut tab, &fonts);
        assert_eq!(tab.layout_full_passes, 1);
        let leaf_height_before = geometry_snapshot(tab.layout_root.as_deref().expect("layout"))
            .into_iter()
            .find(|(n, _, _)| *n == Some(leaf))
            .expect("leaf geometry")
            .1
            .h;

        // Inherited-property change on the ancestor: the whole subtree
        // restyles. `font-size` is in the splice blacklist (it anchors
        // em-relative margins), so this takes the full path — the
        // requirement is that the subtree restyles correctly.
        set_attr(&tab, wrap, "style", "font-size: 30px");
        tab.sync_dirty_flags();
        assert!(tab.dirty.style);
        refresh(&mut tab, &fonts);
        let partial_snapshot = geometry_snapshot(tab.layout_root.as_deref().expect("layout"));
        let leaf_height_after = partial_snapshot
            .iter()
            .find(|(n, _, _)| *n == Some(leaf))
            .expect("leaf geometry after")
            .1
            .h;
        assert!(
            leaf_height_after > leaf_height_before + 1.0,
            "subtree restyled with the inherited font size: {leaf_height_before} -> {leaf_height_after}"
        );

        // Differential: whatever path was taken must match a forced
        // full layout exactly.
        mark_all_dirty(&mut tab);
        refresh(&mut tab, &fonts);
        assert_eq!(
            partial_snapshot,
            geometry_snapshot(tab.layout_root.as_deref().expect("layout"))
        );
    }

    #[test]
    fn refresh_partial_inside_absolute_ancestor_stays_correct() {
        // The dirtied <p> is in-flow inside an absolutely positioned
        // ancestor: the <p>'s own subtree contains no out-of-flow box
        // and its containing block is the ancestor's content box,
        // which the partial path derives from the previous tree — so
        // either path may run, but the result must equal a full
        // layout exactly. (The out-of-flow *root* case, where the
        // partial path must refuse, is pinned separately below.)
        use bowser_css::parse_stylesheet;
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"abs\"><p id=\"leaf\">hello world</p></div></body></html>",
        );
        let leaf = doc.get_element_by_id("leaf").expect("leaf");
        let text = doc.first_child(leaf).expect("leaf text");
        let mut tab = test_tab();
        tab.author.push(parse_stylesheet(
            "body { margin: 0; } p { margin: 0; } \
             #abs { position: absolute; top: 0; left: 0; width: 300px; }",
        ));
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        let fonts = test_fonts();
        mark_all_dirty(&mut tab);
        refresh(&mut tab, &fonts);

        set_text(&tab, text, "goodbye world");
        tab.sync_dirty_flags();
        refresh(&mut tab, &fonts);
        let snapshot = geometry_snapshot(tab.layout_root.as_deref().expect("layout"));
        let partials = tab.layout_partial_passes;
        mark_all_dirty(&mut tab);
        refresh(&mut tab, &fonts);
        assert_eq!(
            snapshot,
            geometry_snapshot(tab.layout_root.as_deref().expect("layout")),
            "whichever path ran, geometry matches the full layout (partials={partials})"
        );
    }

    #[test]
    fn refresh_partial_falls_back_for_out_of_flow_root() {
        use bowser_css::parse_stylesheet;
        let (doc, _) =
            Document::parse_html(b"<html><body><p id=\"leaf\">hello world</p></body></html>");
        let leaf = doc.get_element_by_id("leaf").expect("leaf");
        let text = doc.first_child(leaf).expect("leaf text");
        let mut tab = test_tab();
        tab.author.push(parse_stylesheet(
            "body { margin: 0; } \
             #leaf { position: absolute; top: 0; left: 0; width: 300px; margin: 0; }",
        ));
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        let fonts = test_fonts();
        mark_all_dirty(&mut tab);
        refresh(&mut tab, &fonts);
        assert_eq!(tab.layout_full_passes, 1);

        // The dirtied element itself is out of flow: its containing
        // block lies outside its own subtree, so the partial path must
        // refuse even though the edit is size-stable.
        set_text(&tab, text, "goodbye world");
        tab.sync_dirty_flags();
        refresh(&mut tab, &fonts);
        assert_eq!(tab.layout_partial_passes, 0);
        assert_eq!(tab.layout_full_passes, 2);
    }

    #[test]
    fn refresh_partial_single_root_only_for_multiple_dirt() {
        let (mut tab, _leaf, text) = leaf_tab();
        let fonts = test_fonts();
        mark_all_dirty(&mut tab);
        refresh(&mut tab, &fonts);

        // Two disjoint text edits before one refresh: coalesced to
        // two roots, and step 2 serves exactly one — full fallback.
        let sib_text = tab
            .installed()
            .expect("document")
            .with_document(|cell| {
                let doc = cell.borrow();
                let sib = doc.get_element_by_id("sib").expect("sib");
                doc.first_child(sib).expect("sib text")
            })
            .expect("tab owns document");
        set_text(&tab, text, "goodbye world");
        set_text(&tab, sib_text, "other words");
        tab.sync_dirty_flags();
        assert_eq!(tab.dirty_subtrees.len(), 2);
        refresh(&mut tab, &fonts);
        assert_eq!(tab.layout_partial_passes, 0);
        assert_eq!(tab.layout_full_passes, 2);
    }

    // -----------------------------------------------------------------
    // Differential corpus (wave 51 rework): every case runs the same
    // (document, mutation) through a partial-eligible `refresh` and a
    // forced-full `refresh` and asserts identical geometry snapshots.
    // A case may additionally pin *which* path must be taken.
    // -----------------------------------------------------------------

    type Snapshot = Vec<(Option<NodeId>, Rect, Option<String>)>;

    /// A tab with `html`/`css` installed and one full layout done.
    fn corpus_tab(html: &str, css: &str, fonts: &Mutex<FontCache>) -> Tab {
        use bowser_css::parse_stylesheet;
        let (doc, _) = Document::parse_html(html.as_bytes());
        let mut tab = test_tab();
        tab.author.push(parse_stylesheet(css));
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        mark_all_dirty(&mut tab);
        refresh(&mut tab, fonts);
        assert_eq!(tab.layout_full_passes, 1, "corpus setup is one full pass");
        tab
    }

    /// The text node inside the element with id `id`.
    fn corpus_text(tab: &Tab, id: &str) -> NodeId {
        tab.installed()
            .expect("document")
            .with_document(|cell| {
                let doc = cell.borrow();
                let el = doc.get_element_by_id(id).expect("corpus element");
                doc.first_child(el).expect("corpus text")
            })
            .expect("tab owns document")
    }

    fn snapshot_of(tab: &Tab) -> Snapshot {
        geometry_snapshot(tab.layout_root.as_deref().expect("layout"))
    }

    fn snap_width(snapshot: &Snapshot, tab: &Tab, id: &str) -> f32 {
        let node = tab
            .installed()
            .expect("document")
            .with_document(|cell| cell.borrow().get_element_by_id(id).expect("element"))
            .expect("tab owns document");
        snapshot
            .iter()
            .find(|(n, _, _)| *n == Some(node))
            .expect("element geometry")
            .1
            .w
    }

    fn snap_x(snapshot: &Snapshot, tab: &Tab, id: &str) -> f32 {
        let node = tab
            .installed()
            .expect("document")
            .with_document(|cell| cell.borrow().get_element_by_id(id).expect("element"))
            .expect("tab owns document");
        snapshot
            .iter()
            .find(|(n, _, _)| *n == Some(node))
            .expect("element geometry")
            .1
            .x
    }

    /// The inline-start x of the first text fragment under the element
    /// with id `id` — i.e. where its first line box begins. Used to
    /// observe float intrusion into a following sibling's line boxes.
    fn first_text_frag_x(snapshot: &Snapshot, tab: &Tab, id: &str) -> f32 {
        let node = tab
            .installed()
            .expect("document")
            .with_document(|cell| cell.borrow().get_element_by_id(id).expect("element"))
            .expect("tab owns document");
        let start = snapshot
            .iter()
            .position(|(n, _, _)| *n == Some(node))
            .expect("element in snapshot");
        snapshot[start..]
            .iter()
            .find(|(_, _, t)| t.is_some())
            .expect("text under element")
            .1
            .x
    }

    /// Run `act` (the mutation, including its `sync_dirty_flags` and
    /// any same-turn non-DOM dirt), then a partial-eligible refresh,
    /// then a forced-full refresh of the same document state. Asserts
    /// the two geometry snapshots are identical and returns
    /// `(initial, after, partial_passes_taken, full_passes_taken)` for
    /// the mutation refresh.
    fn differential(
        tab: &mut Tab,
        fonts: &Mutex<FontCache>,
        act: impl FnOnce(&mut Tab),
    ) -> (Snapshot, Snapshot, u64, u64) {
        let initial = snapshot_of(tab);
        act(tab);
        let (p0, f0) = (tab.layout_partial_passes, tab.layout_full_passes);
        refresh(tab, fonts);
        let after = snapshot_of(tab);
        let (partials, fulls) = (tab.layout_partial_passes - p0, tab.layout_full_passes - f0);
        mark_all_dirty(tab);
        refresh(tab, fonts);
        assert_eq!(
            after,
            snapshot_of(tab),
            "partial-eligible refresh geometry must equal forced-full geometry"
        );
        (initial, after, partials, fulls)
    }

    #[test]
    fn corpus_plain_size_stable_text_edit_takes_partial() {
        let fonts = test_fonts();
        let mut tab = corpus_tab(
            "<html><body><div id=\"wrap\"><p id=\"leaf\">hello world</p>\
             <p id=\"sib\">other text</p></div></body></html>",
            "body { margin: 0; } #wrap { padding: 10px; } p { margin: 0; }",
            &fonts,
        );
        let text = corpus_text(&tab, "leaf");
        let (_, _, partials, fulls) = differential(&mut tab, &fonts, |tab| {
            set_text(tab, text, "goodbye world");
            tab.sync_dirty_flags();
        });
        assert_eq!(partials, 1, "plain size-stable edit MUST splice");
        assert_eq!(fulls, 0);
    }

    #[test]
    fn corpus_float_shrink_to_fit_ancestor_takes_full() {
        // The F2 probe shape: the float's width is its content's
        // shrink-to-fit width; the subtree pass at the old width
        // cannot widen it.
        let fonts = test_fonts();
        let mut tab = corpus_tab(
            "<html><body><div id=\"fl\"><p id=\"leaf\">hi</p></div></body></html>",
            "body { margin: 0; } #fl { float: left; } \
             p { margin: 0; white-space: nowrap; }",
            &fonts,
        );
        let text = corpus_text(&tab, "leaf");
        let (initial, after, partials, fulls) = differential(&mut tab, &fonts, |tab| {
            set_text(tab, text, "hello world, this is much longer");
            tab.sync_dirty_flags();
        });
        assert_eq!(partials, 0, "float ancestor must refuse the splice");
        assert_eq!(fulls, 1);
        let (before, after_w) = (
            snap_width(&initial, &tab, "fl"),
            snap_width(&after, &tab, "fl"),
        );
        assert!(
            after_w > before + 50.0,
            "full layout widens the float: {before} -> {after_w}"
        );
    }

    #[test]
    fn corpus_flex_item_definite_width_auto_minimum_takes_full() {
        // N1: a flex item with a definite `width: 200px` but the
        // default `min-width: auto` (the automatic minimum) and
        // `flex-basis: auto`. The gate must refuse the partial path:
        // the item's used main size can be content-driven through the
        // automatic minimum / basis resolution, which the subtree
        // pass (replaying the old containing width) cannot reproduce.
        // Conservative by design: one full pass instead of a
        // possibly-stale splice.
        let fonts = test_fonts();
        let mut tab = corpus_tab(
            "<html><body><div id=\"fc\"><div id=\"item\"><div id=\"grow\">\
             aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa</div>\
             <div id=\"cand\"><p id=\"leaf\">hello</p></div></div></div></body></html>",
            "body { margin: 0; } p { margin: 0; } \
             #fc { display: flex; } #item { width: 200px; } \
             #grow { white-space: nowrap; }",
            &fonts,
        );
        let text = corpus_text(&tab, "grow");
        let (_, after, partials, fulls) = differential(&mut tab, &fonts, |tab| {
            set_text(tab, text, &"a".repeat(160));
            tab.sync_dirty_flags();
        });
        assert_eq!(
            partials, 0,
            "flex item with definite width but auto minimum/basis must refuse the splice"
        );
        assert_eq!(fulls, 1);
        // Forced-full geometry is correct (the differential already
        // asserted it equals a second full pass): the item keeps its
        // specified width.
        assert_eq!(snap_width(&after, &tab, "item"), 200.0);
    }

    #[test]
    fn corpus_grid_vw_track_and_calc_width() {
        // The F3 probe shape, end to end: 400px wrapper, 800px
        // viewport, `1fr 25vw 100px` tracks, and a `calc(10px + 5vw)`
        // width inside the subtree.
        let fonts = test_fonts();
        let mut tab = corpus_tab(
            "<html><body><div id=\"wrap\"><div id=\"grid\">\
             <div id=\"c1\">hello world</div><div id=\"c2\">b</div><div id=\"c3\">c</div>\
             </div></div></body></html>",
            "body { margin: 0; } #wrap { width: 400px; } \
             #grid { display: grid; grid-template-columns: 1fr 25vw 100px; } \
             #c1 { width: calc(10px + 5vw); } \
             #grid div { height: 10px; margin: 0; }",
            &fonts,
        );
        let text = corpus_text(&tab, "c1");
        let (_, after, _, _) = differential(&mut tab, &fonts, |tab| {
            set_text(tab, text, "goodbye world");
            tab.sync_dirty_flags();
        });
        assert_eq!(
            snap_width(&after, &tab, "c1"),
            50.0,
            "calc(10px + 5vw) at 800px viewport"
        );
        assert_eq!(snap_width(&after, &tab, "c2"), 200.0, "25vw track is 200px");
        assert_eq!(snap_width(&after, &tab, "c3"), 100.0);
    }

    #[test]
    fn corpus_abspos_auto_width_ancestor_takes_full() {
        let fonts = test_fonts();
        let mut tab = corpus_tab(
            "<html><body><div id=\"abs\"><p id=\"leaf\">hi</p></div></body></html>",
            "body { margin: 0; } \
             #abs { position: absolute; top: 0; left: 0; } \
             p { margin: 0; white-space: nowrap; }",
            &fonts,
        );
        let text = corpus_text(&tab, "leaf");
        let (initial, after, partials, fulls) = differential(&mut tab, &fonts, |tab| {
            set_text(tab, text, "hello world, this is much longer");
            tab.sync_dirty_flags();
        });
        assert_eq!(partials, 0, "abspos auto-width ancestor must refuse");
        assert_eq!(fulls, 1);
        let (before, after_w) = (
            snap_width(&initial, &tab, "abs"),
            snap_width(&after, &tab, "abs"),
        );
        assert!(
            after_w > before + 50.0,
            "abspos box widens: {before} -> {after_w}"
        );
    }

    #[test]
    fn corpus_inline_block_ancestor_takes_full() {
        let fonts = test_fonts();
        let mut tab = corpus_tab(
            "<html><body><div id=\"ib\"><p id=\"leaf\">hi</p></div></body></html>",
            "body { margin: 0; } #ib { display: inline-block; } \
             p { margin: 0; white-space: nowrap; }",
            &fonts,
        );
        let text = corpus_text(&tab, "leaf");
        let (initial, after, partials, fulls) = differential(&mut tab, &fonts, |tab| {
            set_text(tab, text, "hello world, this is much longer");
            tab.sync_dirty_flags();
        });
        assert_eq!(partials, 0, "inline-block ancestor must refuse");
        assert_eq!(fulls, 1);
        let (before, after_w) = (
            snap_width(&initial, &tab, "ib"),
            snap_width(&after, &tab, "ib"),
        );
        assert!(
            after_w > before + 50.0,
            "inline-block widens: {before} -> {after_w}"
        );
    }

    #[test]
    fn corpus_table_ancestor_geometry() {
        let fonts = test_fonts();
        let mut tab = corpus_tab(
            "<html><body><table id=\"t\"><tr><td id=\"cell\">hi</td></tr></table></body></html>",
            "body { margin: 0; } td { white-space: nowrap; }",
            &fonts,
        );
        let text = corpus_text(&tab, "cell");
        let (initial, after, partials, _) = differential(&mut tab, &fonts, |tab| {
            set_text(tab, text, "hello world, this is much longer");
            tab.sync_dirty_flags();
        });
        assert_eq!(
            partials, 0,
            "table content growth must not splice stale column widths"
        );
        let (before, after_w) = (
            snap_width(&initial, &tab, "t"),
            snap_width(&after, &tab, "t"),
        );
        assert!(
            after_w > before + 50.0,
            "table widens: {before} -> {after_w}"
        );
    }

    #[test]
    fn corpus_sticky_in_subtree_takes_full() {
        // Already gate-excluded (sticky inside the candidate subtree
        // is a partial blocker); pinned so the exclusion cannot regress.
        let fonts = test_fonts();
        let mut tab = corpus_tab(
            "<html><body><div id=\"st\">hello world</div><div id=\"tall\">x</div></body></html>",
            "body { margin: 0; } #st { position: sticky; top: 0; } #tall { height: 500px; }",
            &fonts,
        );
        let text = corpus_text(&tab, "st");
        let (_, _, partials, fulls) = differential(&mut tab, &fonts, |tab| {
            set_text(tab, text, "goodbye world");
            tab.sync_dirty_flags();
        });
        assert_eq!(partials, 0, "sticky in subtree must refuse");
        assert_eq!(fulls, 1);
    }

    #[test]
    fn corpus_size_changing_text_edit_takes_full() {
        let fonts = test_fonts();
        let mut tab = corpus_tab(
            "<html><body><div id=\"wrap\"><p id=\"leaf\">hello world</p></div></body></html>",
            "body { margin: 0; } #wrap { padding: 10px; } p { margin: 0; }",
            &fonts,
        );
        let text = corpus_text(&tab, "leaf");
        let (_, _, partials, fulls) = differential(&mut tab, &fonts, |tab| {
            set_text(tab, text, &"word ".repeat(120));
            tab.sync_dirty_flags();
        });
        assert_eq!(partials, 0, "size change must not splice");
        assert_eq!(fulls, 1);
    }

    #[test]
    fn corpus_text_edit_after_resize_same_turn_takes_full() {
        // The F1 trigger: a text edit's retained subtree and a resize
        // in the same turn — the resize wins, the full path runs at
        // the new viewport, and no stale-geometry splice happens.
        let fonts = test_fonts();
        let mut tab = corpus_tab(
            "<html><body><div id=\"wrap\"><p id=\"leaf\">hello world</p></div></body></html>",
            "body { margin: 0; } #wrap { padding: 10px; } p { margin: 0; }",
            &fonts,
        );
        let text = corpus_text(&tab, "leaf");
        let (initial, after, partials, fulls) = differential(&mut tab, &fonts, |tab| {
            set_text(tab, text, "goodbye world");
            tab.sync_dirty_flags();
            assert_eq!(tab.dirty_subtrees.len(), 1, "text dirt retained pre-resize");
            apply_resize(tab, 700.0, 500.0);
            assert!(tab.dirty_full_fallback, "resize forces full fallback");
            assert!(
                tab.dirty_subtrees.is_empty(),
                "resize drops retained subtrees"
            );
        });
        assert_eq!(partials, 0, "resize in the same turn must not splice");
        assert_eq!(fulls, 1);
        assert_eq!(tab.last_layout_viewport, Some(Size { w: 700.0, h: 500.0 }));
        let (before, after_w) = (
            snap_width(&initial, &tab, "leaf"),
            snap_width(&after, &tab, "leaf"),
        );
        assert!(
            (after_w - (before - 100.0)).abs() < 0.5,
            "leaf re-laid at the new 700px viewport: {before} -> {after_w}"
        );
    }

    #[test]
    fn corpus_image_loaded_dirt_plus_text_edit_same_turn_takes_full() {
        // The F1 image trigger: a late image's intrinsic size lands in
        // the same turn as a text edit — the image dirt forces full.
        let fonts = test_fonts();
        let mut tab = corpus_tab(
            "<html><body><div id=\"wrap\"><p id=\"leaf\">hello world</p>\
             <img id=\"im\" src=\"http://example.com/x.png\"></div></body></html>",
            "body { margin: 0; } #wrap { padding: 10px; } p { margin: 0; }",
            &fonts,
        );
        let text = corpus_text(&tab, "leaf");
        let (_, after, partials, fulls) = differential(&mut tab, &fonts, |tab| {
            set_text(tab, text, "goodbye world");
            tab.sync_dirty_flags();
            assert_eq!(tab.dirty_subtrees.len(), 1, "text dirt retained pre-image");
            let pending = tab.pending_late_images();
            assert_eq!(pending.len(), 1, "one late image fetch");
            let p = &pending[0];
            tab.note_image_loaded(
                p.node,
                p.generation,
                Some(IntrinsicDims::from_size(120.0, 80.0)),
            );
            assert!(
                tab.dirty_full_fallback,
                "image size dirt forces full fallback"
            );
            assert!(
                tab.dirty_subtrees.is_empty(),
                "image dirt drops retained subtrees"
            );
        });
        assert_eq!(partials, 0, "image dirt in the same turn must not splice");
        assert_eq!(fulls, 1);
        assert_eq!(
            snap_width(&after, &tab, "im"),
            120.0,
            "img laid out at its intrinsic width"
        );
    }

    #[test]
    fn corpus_break_after_class_flip_takes_full() {
        // H1 (CSS reviewer): flipping `break-after` to `column` on e1
        // pushes e2 into the second column while e1's own border box
        // is unchanged — the rect gate cannot see the change, so the
        // property must fail the splice closed via EXTERNAL_EFFECT.
        // (`column-fill: auto` with a definite height keeps e2 in the
        // first column initially; with the default `balance` the two
        // blocks would already be split across columns and the flip
        // would move nothing.)
        let fonts = test_fonts();
        let mut tab = corpus_tab(
            "<html><body><div id=\"mc\"><div id=\"e1\">a</div>\
             <div id=\"e2\">b</div></div></body></html>",
            "body { margin: 0; } \
             #mc { column-count: 2; column-fill: auto; height: 100px; width: 400px; } \
             #e1, #e2 { width: 100px; margin: 0; } \
             #e1.b { break-after: column; }",
            &fonts,
        );
        let e1 = element_by_id(&tab, "e1");
        let (initial, after, partials, fulls) = differential(&mut tab, &fonts, |tab| {
            set_attr(tab, e1, "class", "b");
            tab.sync_dirty_flags();
        });
        assert_eq!(partials, 0, "break-after change must refuse the splice");
        assert_eq!(fulls, 1);
        assert!(
            snap_x(&after, &tab, "e2") > snap_x(&initial, &tab, "e2") + 100.0,
            "full pass moved e2 to the second column"
        );
    }

    #[test]
    fn corpus_float_introduced_in_subtree_takes_full() {
        // H1 (JS reviewer), variant A: floating the inline span
        // promotes the dirty root to #p; #p's own style and border box
        // are unchanged (fixed height), so the rect gate and
        // `style_change_breaks_splice` — which diffs only #p's own
        // style — both pass. But the new float intrudes into #s's line
        // boxes, which only a full pass re-wraps; the whole-subtree
        // float check must fail the splice closed. (The span holds
        // pre-formatted lines so the float is tall enough to reach
        // #s: a one-line float would not overlap it.)
        let fonts = test_fonts();
        let mut tab = corpus_tab(
            "<html><body><div id=\"p\"><span id=\"d\">x\nx\nx\nx\nx\nx\nx\nx</span></div>\
             <div id=\"s\">lorem ipsum dolor sit amet consectetur adipiscing elit sed</div></body></html>",
            "body { margin: 0; } #p { height: 100px; } #d { white-space: pre; } #s { margin: 0; }",
            &fonts,
        );
        let d = element_by_id(&tab, "d");
        let (initial, after, partials, fulls) = differential(&mut tab, &fonts, |tab| {
            set_attr(tab, d, "style", "float: left");
            tab.sync_dirty_flags();
        });
        assert_eq!(
            partials, 0,
            "float introduced in the subtree must refuse the splice"
        );
        assert_eq!(fulls, 1);
        assert!(
            first_text_frag_x(&after, &tab, "s") > first_text_frag_x(&initial, &tab, "s") + 1.0,
            "full pass wrapped #s's first line around the float"
        );
    }

    #[test]
    fn corpus_float_grown_by_line_height_takes_full() {
        // H1 (JS reviewer), variant B: the float already exists, so the
        // old subtree carries it; growing #p's line-height grows the
        // float (inherited) while #p's own border box is unchanged
        // (fixed height). #s's wrapping goes stale unless the splice
        // fails closed on the old subtree's float.
        let fonts = test_fonts();
        let mut tab = corpus_tab(
            "<html><body><div id=\"p\"><span id=\"d\">x\nx\nx\nx</span></div>\
             <div id=\"s\">lorem ipsum dolor sit amet consectetur adipiscing elit sed</div></body></html>",
            "body { margin: 0; } #p { height: 100px; } \
             #d { float: left; white-space: pre; } #s { margin: 0; }",
            &fonts,
        );
        let p = element_by_id(&tab, "p");
        let (initial, after, partials, fulls) = differential(&mut tab, &fonts, |tab| {
            set_attr(tab, p, "style", "height: 100px; line-height: 40px");
            tab.sync_dirty_flags();
        });
        assert_eq!(
            partials, 0,
            "float in the previous subtree must refuse the splice"
        );
        assert_eq!(fulls, 1);
        assert!(
            first_text_frag_x(&after, &tab, "s") > first_text_frag_x(&initial, &tab, "s") + 1.0,
            "full pass re-wrapped #s around the grown float"
        );
    }

    #[test]
    fn failed_partial_attempt_restores_published_js_share() {
        // F6: `try_partial_layout` drops the JS bindings' published
        // share to make the previous tree uniquely owned; a later
        // failure (here the size gate) must re-publish the untouched
        // tree, so the tab — including script-visible geometry — is
        // exactly as the failed attempt found it.
        use bowser_js::Value;
        let (doc, _) =
            Document::parse_html(b"<html><body><p id=\"leaf\">hello world</p></body></html>");
        let doc = Rc::new(RefCell::new(doc));
        let leaf = doc.borrow().get_element_by_id("leaf").expect("leaf");
        let text = doc.borrow().first_child(leaf).expect("leaf text");
        let url = bowser_url::parse("http://example.com/", None).expect("test url");
        let page_vm = crate::script::PageVm::new_for_document(doc.clone(), &url, "test", &[])
            .expect("install web apis");
        let mut tab = test_tab();
        tab.install_document(doc, page_vm);
        let fonts = test_fonts();
        mark_all_dirty(&mut tab);
        refresh(&mut tab, &fonts);
        let leaf_width = |tab: &mut Tab| -> i32 {
            match tab
                .page_vm_mut()
                .expect("page vm")
                .eval("document.getElementById('leaf').offsetWidth")
                .expect("eval offsetWidth")
            {
                Value::Int(n) => n,
                v => panic!("offsetWidth must be an int, got {v:?}"),
            }
        };
        assert_eq!(leaf_width(&mut tab), 800, "full tree published");
        assert!(
            Rc::strong_count(tab.layout_root.as_ref().expect("layout rc")) > 1,
            "the JS share is the second strong owner"
        );

        // A size-changing edit: the partial attempt gets past the
        // ownership drop, then the size gate refuses the splice.
        set_text(&tab, text, &"word ".repeat(120));
        tab.sync_dirty_flags();
        assert_eq!(tab.dirty_subtrees.len(), 1);
        {
            // Disjoint split for the direct partial-layout call.
            let (installed, render) = tab.split_for_reentry();
            let installed = installed.as_ref().expect("installed");
            let ctx = installed.page_vm().web_api_context();
            assert!(
                !try_partial_layout(render, installed.document(), ctx, &fonts),
                "size gate refuses"
            );
        }

        // The published share is back: script still measures the old
        // tree, and the tab's counters/report are untouched.
        assert_eq!(
            leaf_width(&mut tab),
            800,
            "old tree re-published after failure"
        );
        assert_eq!(tab.layout_partial_passes, 0);
        assert_eq!(
            tab.dirty_subtrees.len(),
            1,
            "report not serviced by the failed attempt"
        );
    }

    /// Sticky inside a nested scroller sticks relative to the
    /// scroller's scrollport: `refresh` feeds the scroller's live
    /// offset into layout's sticky constraints, so scrolling the
    /// container 80px re-bakes the sticky box 80px below its in-flow
    /// spot (stuck at the scrollport top) instead of leaving it at
    /// the zero-offset position.
    #[test]
    fn refresh_bakes_nested_offset_into_sticky_constraint() {
        use bowser_css::parse_stylesheet;

        fn box_for_node(root: &LayoutBox, node: NodeId) -> &LayoutBox {
            fn walk(box_: &LayoutBox, node: NodeId) -> Option<&LayoutBox> {
                if box_.node == Some(node) {
                    return Some(box_);
                }
                box_.children.iter().find_map(|c| walk(c, node))
            }
            walk(root, node).expect("box for node")
        }

        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"sticky\">s</div>\
              <div id=\"tall\">x</div></div></body></html>",
        );
        let doc = Rc::new(RefCell::new(doc));
        let sc = doc.borrow().get_element_by_id("sc").expect("sc");
        let sticky = doc.borrow().get_element_by_id("sticky").expect("sticky");
        let mut tab = test_tab();
        tab.author.push(parse_stylesheet(
            "#sc { overflow-y: scroll; height: 200px; width: 300px; } \
             #sticky { position: sticky; top: 0; height: 50px; } \
             #tall { height: 500px; }",
        ));
        tab.install_test_document(doc);
        mark_all_dirty(&mut tab);
        let fonts = test_fonts();
        refresh(&mut tab, &fonts);
        let root = tab.layout_root.as_deref().expect("layout root");
        let scroller_y = box_for_node(root, sc).rect.y;
        // Unscrolled: the sticky sits at its in-flow position, the
        // scroller's content top.
        assert_eq!(box_for_node(root, sticky).rect.y, scroller_y);

        // Scroll the nested container 80px (max is 350) and relayout:
        // the sticky constraint re-bakes at the live offset.
        let i = tab.scroll_tree.index_of(sc).expect("sc is a container");
        tab.scroll_tree.container(i).state.set_offset(0.0, 80.0);
        assert_eq!(tab.scroll_tree.container(i).state.offset(), (0.0, 80.0));
        tab.dirty.layout = true;
        refresh(&mut tab, &fonts);
        let root = tab.layout_root.as_deref().expect("layout root");
        assert_eq!(
            box_for_node(root, sticky).rect.y,
            box_for_node(root, sc).rect.y + 80.0,
            "sticky constrains against the scroller's live offset"
        );
    }

    /// Pins the sticky-lag fix: a wheel scroll dirties layout when the
    /// page holds `position: sticky` boxes, so the next `refresh`
    /// re-bakes the sticky constraint at the live offset. The test never
    /// invalidates layout itself — without the dirty flag the stuck box
    /// would keep the offset it was baked for while the GPU uniform
    /// moved everything else.
    #[test]
    fn wheel_scroll_dirties_layout_when_sticky_present() {
        use bowser_css::parse_stylesheet;

        fn box_for_node(root: &LayoutBox, node: NodeId) -> &LayoutBox {
            fn walk(box_: &LayoutBox, node: NodeId) -> Option<&LayoutBox> {
                if box_.node == Some(node) {
                    return Some(box_);
                }
                box_.children.iter().find_map(|c| walk(c, node))
            }
            walk(root, node).expect("box for node")
        }

        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sticky\">s</div><div id=\"tall\">x</div></body></html>",
        );
        let doc = Rc::new(RefCell::new(doc));
        let sticky = doc.borrow().get_element_by_id("sticky").expect("sticky");
        let mut tab = test_tab();
        tab.author.push(parse_stylesheet(
            "#sticky { position: sticky; top: 0; height: 50px; } #tall { height: 2000px; }",
        ));
        tab.install_test_document(doc);
        mark_all_dirty(&mut tab);
        let fonts = test_fonts();
        refresh(&mut tab, &fonts);
        assert!(tab.scroll_tree.has_sticky());
        let y_before = box_for_node(tab.layout_root.as_deref().expect("layout"), sticky)
            .rect
            .y;
        assert_eq!(tab.dirty, DirtyFlags::default());

        // Scroll the viewport 120px with no manual layout invalidation:
        // the scroll path itself must schedule the sticky re-bake.
        assert_eq!(apply_wheel(&mut tab, 10.0, 10.0, 0.0, 120.0), vec![None]);
        assert_eq!(tab.scroll().offset(), (0.0, 120.0));
        assert!(
            tab.dirty.layout,
            "scroll must dirty layout when sticky boxes are present"
        );

        refresh(&mut tab, &fonts);
        // Stuck at the viewport top: page y == scroll offset + top (0).
        assert_eq!(
            box_for_node(tab.layout_root.as_deref().expect("layout"), sticky)
                .rect
                .y,
            120.0,
            "sticky box follows the scroll offset once layout re-bakes"
        );
        assert!(y_before < 120.0, "the box actually moved");
    }

    /// The other half of the tradeoff: a page with no sticky boxes
    /// keeps the no-layout fast path — a wheel scroll rides the GPU
    /// uniform and invalidates nothing.
    #[test]
    fn wheel_scroll_keeps_fast_path_without_sticky() {
        use bowser_css::parse_stylesheet;

        let (doc, _) = Document::parse_html(b"<html><body><div id=\"tall\">x</div></body></html>");
        let doc = Rc::new(RefCell::new(doc));
        let mut tab = test_tab();
        tab.author
            .push(parse_stylesheet("#tall { height: 2000px; }"));
        tab.install_test_document(doc);
        mark_all_dirty(&mut tab);
        let fonts = test_fonts();
        refresh(&mut tab, &fonts);
        assert!(!tab.scroll_tree.has_sticky());

        assert_eq!(apply_wheel(&mut tab, 10.0, 10.0, 0.0, 120.0), vec![None]);
        assert_eq!(tab.scroll().offset(), (0.0, 120.0));
        assert_eq!(
            tab.dirty,
            DirtyFlags::default(),
            "no sticky boxes: scroll must not invalidate any pipeline stage"
        );
    }

    /// The hysteresis half: a sticky box far from the viewport does not
    /// cost a layout per scroll tick — only offsets near its
    /// constraint band dirty layout.
    #[test]
    fn wheel_scroll_skips_layout_when_sticky_far_from_viewport() {
        use bowser_css::parse_stylesheet;

        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"pad\">x</div><div id=\"sticky\">s</div></body></html>",
        );
        let doc = Rc::new(RefCell::new(doc));
        let mut tab = test_tab();
        tab.author.push(parse_stylesheet(
            "#pad { height: 5000px; } #sticky { position: sticky; top: 0; height: 50px; }",
        ));
        tab.install_test_document(doc);
        mark_all_dirty(&mut tab);
        let fonts = test_fonts();
        refresh(&mut tab, &fonts);
        assert!(tab.scroll_tree.has_sticky());

        // 120px at the top of the page: the sticky box sits ~5000px
        // down, nowhere near its constraint band — the fast path
        // holds.
        assert_eq!(apply_wheel(&mut tab, 10.0, 10.0, 0.0, 120.0), vec![None]);
        assert_eq!(tab.scroll().offset(), (0.0, 120.0));
        assert_eq!(
            tab.dirty,
            DirtyFlags::default(),
            "distant sticky box: scroll must not invalidate any pipeline stage"
        );

        // Scroll into the box's constraint band: now layout must be
        // dirtied so the next refresh re-bakes the sticky constraint.
        assert_eq!(apply_wheel(&mut tab, 10.0, 10.0, 0.0, 10000.0), vec![None]);
        assert!(
            tab.dirty.layout,
            "scroll into the sticky band must dirty layout"
        );
    }

    #[test]
    fn refresh_with_no_document_clears_flags() {
        let mut tab = test_tab();
        mark_all_dirty(&mut tab);
        refresh(&mut tab, &test_fonts());
        assert_eq!(tab.dirty, DirtyFlags::default());
        assert!(tab.layout_root.is_none());
    }

    #[test]
    fn refresh_reuses_one_image_id_per_src() {
        use crate::loader::prime_image_cache;
        use bowser_paint::display_list::DisplayItem;
        prime_image_cache(
            "http://example.com/a.png",
            DecodedImage::from_rgba(2, 2, vec![9; 16]),
        );
        let mut tab = test_tab();
        tab.history
            .push(bowser_url::parse("http://example.com/page", None).unwrap());
        let (doc, _) = Document::parse_html(
            b"<html><body><img src=\"a.png\"><img src=\"a.png\"></body></html>",
        );
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        mark_all_dirty(&mut tab);
        refresh(&mut tab, &test_fonts());

        let list = tab.display_list.as_ref().expect("display list built");
        let ids: Vec<ImageId> = list
            .items
            .iter()
            .filter_map(|item| match item {
                DisplayItem::Image { image, .. } => Some(*image),
                _ => None,
            })
            .collect();
        assert_eq!(ids.len(), 2, "both imgs paint an image");
        assert_eq!(ids[0], ids[1], "one src mints one ImageId");
        assert!(list.image(ids[0]).is_some(), "decoded bytes are registered");
    }

    #[test]
    fn image_id_map_mints_one_id_per_src() {
        use crate::loader::prime_image_cache;
        prime_image_cache(
            "http://example.com/b.png",
            DecodedImage::from_rgba(2, 2, vec![7; 16]),
        );
        prime_image_cache(
            "http://example.com/c.png",
            DecodedImage::from_rgba(2, 2, vec![8; 16]),
        );
        let (doc, _) = Document::parse_html(b"<html></html>");
        let base = bowser_url::parse("http://example.com/page", None).unwrap();
        let mut map = ImageIdMap::new(&doc, Some(&base));

        let (id_b, _) = map.resolve("b.png").expect("cached image resolves");
        let (id_b_again, decoded) = map.resolve("b.png").expect("cached image resolves");
        assert_eq!(id_b, id_b_again, "one src mints one ImageId");
        assert_eq!(decoded.rgba, vec![7; 16]);
        let (id_c, _) = map.resolve("c.png").expect("cached image resolves");
        assert_ne!(id_b, id_c, "distinct srcs mint distinct ImageIds");
        assert!(
            map.resolve("missing.png").is_none(),
            "an uncached src misses"
        );
    }

    #[test]
    fn image_id_map_without_document_url_misses_everything() {
        let (doc, _) = Document::parse_html(b"<html></html>");
        let mut map = ImageIdMap::new(&doc, None);
        assert!(
            map.resolve("b.png").is_none(),
            "no document URL means no resolvable src"
        );
    }
    #[test]
    fn refresh_resolves_text_run_fonts_from_computed_style() {
        use bowser_paint::display_list::DisplayItem;
        let mut tab = test_tab();
        let (doc, _) = Document::parse_html(
            b"<html><body><p style=\"font-family: monospace\">a</p></body></html>",
        );
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        mark_all_dirty(&mut tab);
        let fonts = test_fonts();
        refresh(&mut tab, &fonts);

        let list = tab.display_list.as_ref().expect("display list built");
        let keys: Vec<FontKey> = list
            .items
            .iter()
            .filter_map(|item| match item {
                DisplayItem::TextRun { font_key, .. } => Some(*font_key),
                _ => None,
            })
            .collect();
        assert!(!keys.is_empty(), "expected at least one text run");
        // The window's resolver must agree with a direct resolve of the
        // computed (family, weight, style) triple. On fontless CI both
        // sides degrade to the FontKey(0) fallback and the equality still
        // holds.
        let expected = fonts
            .lock()
            .ok()
            .and_then(|cache| cache.resolve("monospace", 400, FontStyle::Normal))
            .unwrap_or(FontKey(0));
        assert!(
            keys.iter().all(|&k| k == expected),
            "text runs must use the resolved font, got {keys:?}"
        );
    }

    #[test]
    fn key_labels() {
        assert_eq!(
            key_label(&Key::Character("a".into())),
            Some("a".to_string())
        );
        assert_eq!(
            key_label(&Key::Named(NamedKey::Enter)),
            Some("Enter".to_string())
        );
        assert_eq!(key_label(&Key::Named(NamedKey::F5)), Some("F5".to_string()));
        assert_eq!(key_label(&Key::Dead(None)), None);
    }

    #[test]
    fn install_loaded_document_resets_caches_and_dirties() {
        // Pins the shared install behavior the window's async completion
        // funnels through (`navigation::complete_navigation`).
        let mut tab = test_tab();
        tab.scroll().set_offset(0.0, 500.0);
        let url = bowser_url::parse("https://example.com/", None).unwrap();
        let html = b"<html><head><title>hello</title></head><body></body></html>";
        let (document, _) = Document::parse_html(html);
        let loaded = LoadedDocument {
            url: url.clone(),
            trace: FetchTrace { hops: vec![url] },
            document,
            stylesheets: Vec::new(),
            image_sizes: HashMap::new(),
            script_sources: HashMap::new(),
            body: html.to_vec(),
            webfonts: Vec::new(),
        };
        crate::navigation::install_loaded_document(
            &mut tab,
            loaded,
            "TestBrowser/9.9",
            &test_fonts(),
            &test_cookie_jar(),
        );
        assert_eq!(tab.title, "hello");
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
        assert!(tab.layout_root.is_none());
        assert!(tab.display_list.is_none());
        assert!(tab.dirty.style && tab.dirty.layout && tab.dirty.paint);
        // No scripts: a fresh, empty realm is still installed.
        let report = tab.script_report().expect("page VM installed");
        assert_eq!(report.executed, 0);
        assert_eq!(report.skipped, 0);
        assert!(report.errors.is_empty());
    }

    #[test]
    fn push_navigation_installs_webfonts() {
        // w75f1: the push-history branch (`complete_navigation`)
        // must install webfonts exactly like the initial-load path:
        // a navigated document measures the webfont, not fallback.
        use crate::script::collect_script_refs;
        use crate::script_executor::prefetch_script_sources;
        use bowser_layout::FontMetrics;
        let ahem: std::sync::Arc<[u8]> = std::sync::Arc::from(
            include_bytes!("../../bowser_paint/tests/fixtures/Ahem.ttf").to_vec(),
        );
        let html = b"<html><head></head><body><p>X</p></body></html>";
        let url = bowser_url::parse("https://example.com/page", None).unwrap();
        let (document, _) = Document::parse_html(html);
        let script_sources =
            prefetch_script_sources(&collect_script_refs(&document), &url, |_| Ok(String::new()));
        let fonts = std::sync::Arc::new(Mutex::new(FontCache::new()));
        let before = {
            let guard = fonts.lock().expect("font cache lock poisoned");
            let ahem_id = guard.intern_family("Ahem");
            guard.advance("X", 50.0, ahem_id)
        };
        assert!(
            before < 40.0,
            "fallback advance must not be Ahem's 50, got {before}"
        );
        let mut tab = test_tab();
        crate::navigation::complete_navigation(
            &mut tab,
            LoadedDocument {
                url: url.clone(),
                trace: FetchTrace {
                    hops: vec![url.clone()],
                },
                document,
                stylesheets: Vec::new(),
                image_sizes: HashMap::new(),
                script_sources,
                body: html.to_vec(),
                webfonts: vec![Webfont {
                    family: "Ahem".to_string(),
                    weight: 400,
                    weight_range: None,
                    style: "normal".to_string(),
                    bytes: ahem,
                }],
            },
            true,
            "TestBrowser/9.9",
            &fonts,
            &test_cookie_jar(),
        );
        assert_eq!(tab.history.len(), 1, "push path must record the entry");
        let after = {
            let guard = fonts.lock().expect("font cache lock poisoned");
            let ahem_id = guard.intern_family("Ahem");
            guard.advance("X", 50.0, ahem_id)
        };
        assert!(
            (after - 50.0).abs() < 0.01,
            "push-path advance must be Ahem's 50, got {after}"
        );
    }

    #[test]
    fn js_loaded_face_via_shared_cache_changes_layout_measurement() {
        // w75f1: the window install path must share the window's
        // LIVE cache with the page realm, so a `FontFace.load()` in
        // JS installs into the same cache layout measures with —
        // not into a lazily created orphan cache.
        use crate::script::collect_script_refs;
        use crate::script_executor::prefetch_script_sources;
        use bowser_layout::FontMetrics;
        let ahem_bytes = include_bytes!("../../bowser_paint/tests/fixtures/Ahem.ttf").to_vec();
        let b64 = bowser_url::base64_encode(&ahem_bytes);
        let html = b"<html><head></head><body><p>X</p></body></html>";
        let url = bowser_url::parse("https://example.com/page", None).unwrap();
        let (document, _) = Document::parse_html(html);
        let script_sources =
            prefetch_script_sources(&collect_script_refs(&document), &url, |_| Ok(String::new()));
        let fonts = std::sync::Arc::new(Mutex::new(FontCache::new()));
        let mut tab = test_tab();
        crate::navigation::complete_navigation(
            &mut tab,
            LoadedDocument {
                url: url.clone(),
                trace: FetchTrace {
                    hops: vec![url.clone()],
                },
                document,
                stylesheets: Vec::new(),
                image_sizes: HashMap::new(),
                script_sources,
                body: html.to_vec(),
                webfonts: Vec::new(),
            },
            true,
            "TestBrowser/9.9",
            &fonts,
            &test_cookie_jar(),
        );
        let vm = tab.page_vm_mut().expect("page VM installed");
        let status = match vm
            .eval(&format!(
                "var __f = new FontFace('Ahem', 'url(data:font/ttf;base64,{b64})'); \
                 document.fonts.add(__f); __f.load(); __f.status;"
            ))
            .expect("FontFace.load eval")
        {
            Value::Str(s) => s.as_wtf8().to_utf8_lossy().into_owned(),
            other => panic!("FontFace status must be a string, got {other:?}"),
        };
        assert_eq!(status, "loaded", "JS face must load");
        let check = vm
            .eval("document.fonts.check('16px Ahem')")
            .expect("fonts.check eval");
        assert_eq!(
            check,
            Value::Bool(true),
            "shared cache must report the JS face"
        );
        let after = {
            let guard = fonts.lock().expect("font cache lock poisoned");
            let ahem_id = guard.intern_family("Ahem");
            guard.advance("X", 50.0, ahem_id)
        };
        assert!(
            (after - 50.0).abs() < 0.01,
            "window-cache advance must be Ahem's 50 after JS load, got {after}"
        );
    }

    #[test]
    fn pump_headless_reflows_after_late_js_font_install() {
        // w76: the headless pump's font-load drain must reflow a tab
        // whose JS installed a face AFTER the first layout. The text
        // box for `X` in Ahem measures 50 at 50px; before the install
        // the fallback advance is narrower (w75's bindings test pins
        // it under 40). The width is read off the published tree
        // directly — never through a geometry getter, which would
        // force layout through the Ensurer and mask a missing drain.
        use crate::script::collect_script_refs;
        use crate::script_executor::prefetch_script_sources;
        fn text_box_width(tab: &Tab, needle: &str) -> Option<f32> {
            fn walk(bx: &bowser_layout::LayoutBox, needle: &str) -> Option<f32> {
                if bx.box_type == bowser_layout::BoxType::Text && bx.text.as_deref() == Some(needle)
                {
                    return Some(bx.rect.w);
                }
                bx.children.iter().find_map(|c| walk(c, needle))
            }
            tab.render
                .layout_root
                .as_deref()
                .and_then(|root| walk(root, needle))
        }
        let ahem_bytes = include_bytes!("../../bowser_paint/tests/fixtures/Ahem.ttf").to_vec();
        let b64 = bowser_url::base64_encode(&ahem_bytes);
        let html =
            b"<html><head></head><body><p style=\"font-family:Ahem;font-size:50px\">X</p></body></html>";
        let url = bowser_url::parse("https://example.com/page", None).unwrap();
        let (document, _) = Document::parse_html(html);
        let script_sources =
            prefetch_script_sources(&collect_script_refs(&document), &url, |_| Ok(String::new()));
        let fonts = test_fonts();
        let mut tab = test_tab();
        crate::navigation::complete_navigation(
            &mut tab,
            LoadedDocument {
                url: url.clone(),
                trace: FetchTrace {
                    hops: vec![url.clone()],
                },
                document,
                stylesheets: Vec::new(),
                image_sizes: HashMap::new(),
                script_sources,
                body: html.to_vec(),
                webfonts: Vec::new(),
            },
            true,
            "TestBrowser/9.9",
            &fonts,
            &test_cookie_jar(),
        );
        refresh(&mut tab, &fonts);
        let before = text_box_width(&tab, "X").expect("text box after initial layout");
        assert!(
            before < 40.0,
            "fallback advance must be under 40 before the install, got {before}"
        );
        {
            let vm = tab.page_vm_mut().expect("page VM installed");
            vm.eval(&format!(
                "var __f = new FontFace('Ahem', 'url(data:font/ttf;base64,{b64})'); \
                 document.fonts.add(__f); __f.load();"
            ))
            .expect("FontFace.load eval");
        }
        let mut frames = 0u64;
        pump_headless_with_budget(
            &mut tab,
            &fonts,
            Duration::from_secs(5),
            HEADLESS_PUMP_TASK_BUDGET,
            &mut frames,
        );
        let after = text_box_width(&tab, "X").expect("text box after pump");
        assert!(
            (after - 50.0).abs() < 1.0,
            "pumped tab must reflow to Ahem's 50 advance, got {after} (before {before})"
        );
    }

    #[test]
    fn pump_scripts_windowed_reflows_after_late_js_font_install() {
        // w76fix M9, re-pinned w76fix2 M3: the windowed pump's
        // font-load REDRAIN — the font installed by a listener
        // dispatched AFTER the per-tab loop's ordinary font drain.
        // The old shape evaluated `FontFace.load()` BEFORE
        // `pump_scripts()`, so the ordinary drain caught the
        // install and deleting the redrain block would not fail
        // the test. This shape instead installs from a `scrollend`
        // listener, and pins the redrain with no additional pump:
        // pump 1 scrolls (a 0 ms timer fires during the script
        // turn, the fold reports the movement, the `scroll`
        // listener runs, `scrollend` does NOT fire on the moving
        // tick); a redraw then services the scroll dirt, so the
        // tab enters pump 2 clean; pump 2 is a quiet tick, so its
        // `scrollend` dispatch is the only thing that runs after
        // that iteration's font drain — and the `scrollend`
        // listener leaves no DOM dirt (a JS global plus a font
        // install), so `dirty.layout` after pump 2 can only come
        // from the redrain draining the listener's install. A
        // `scroll` listener would NOT discriminate: its own scroll
        // dirt sets `dirty.layout` regardless of the redrain. The
        // redrain block served here is the same one the `scroll`
        // path runs through (both dispatch before it), so this
        // pins the shared block, not a scrollend-only path.
        use crate::script::collect_script_refs;
        use crate::script_executor::prefetch_script_sources;
        fn text_box_width(tab: &Tab, needle: &str) -> Option<f32> {
            fn walk(bx: &bowser_layout::LayoutBox, needle: &str) -> Option<f32> {
                if bx.box_type == bowser_layout::BoxType::Text && bx.text.as_deref() == Some(needle)
                {
                    return Some(bx.rect.w);
                }
                bx.children.iter().find_map(|c| walk(c, needle))
            }
            tab.render
                .layout_root
                .as_deref()
                .and_then(|root| walk(root, needle))
        }
        let ahem_bytes = include_bytes!("../../bowser_paint/tests/fixtures/Ahem.ttf").to_vec();
        let b64 = bowser_url::base64_encode(&ahem_bytes);
        let html =
            b"<html><head></head><body><p style=\"font-family:Ahem;font-size:50px\">X</p><div style=\"height:2000px\"></div></body></html>";
        let url = bowser_url::parse("https://example.com/page", None).unwrap();
        let (document, _) = Document::parse_html(html);
        let script_sources =
            prefetch_script_sources(&collect_script_refs(&document), &url, |_| Ok(String::new()));
        let fonts = test_fonts();
        let mut tab = test_tab();
        crate::navigation::complete_navigation(
            &mut tab,
            LoadedDocument {
                url: url.clone(),
                trace: FetchTrace {
                    hops: vec![url.clone()],
                },
                document,
                stylesheets: Vec::new(),
                image_sizes: HashMap::new(),
                script_sources,
                body: html.to_vec(),
                webfonts: Vec::new(),
            },
            true,
            "TestBrowser/9.9",
            &fonts,
            &test_cookie_jar(),
        );
        refresh(&mut tab, &fonts);
        let before = text_box_width(&tab, "X").expect("text box after initial layout");
        assert!(
            before < 40.0,
            "fallback advance must be under 40 before the install, got {before}"
        );
        {
            let vm = tab.page_vm_mut().expect("page VM installed");
            vm.eval(&format!(
                "window.addEventListener('scroll', () => {{ __sawScroll = true; }}); \
                 window.addEventListener('scrollend', () => {{ \
                 __scrollendInstalled = true; \
                 var __f = new FontFace('Ahem', 'url(data:font/ttf;base64,{b64})'); \
                 document.fonts.add(__f); __f.load(); }}); \
                 setTimeout(() => {{ window.scrollTo(0, 100); }}, 0);"
            ))
            .expect("listener + timer registration eval");
        }
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = WindowApp {
            tabs: &mut tabs,
            jar: Arc::clone(&jar),
            tab: tab_id,
            viewport: (800, 600),
            window: None,
            gpu: None,
            input: InputState::default(),
            modifiers: ModifiersState::default(),
            touch: TouchScrollState::new(),
            scrollbar_drag: None,
            autoscroll: None,
            wheel_latch: None,
            pending: None,
            stale: Vec::new(),
            handoff_tx: None,
            load_control: LoadControl::new(),
            js: None,
            user_agent: "TestBrowser/9.9".to_string(),
            fonts: fonts.clone(),
            scale: 1.0,
            needs_redraw: false,
            init_error: None,
        };
        // Pump 1: the timer scrolls, the fold reports the movement,
        // the `scroll` listener runs, `scrollend` does not fire on
        // the moving tick. The redraw services the scroll dirt so
        // the tab enters pump 2 clean (text still fallback: no
        // font has been installed yet).
        app.pump_scripts();
        {
            let tab = app.tabs.get(tab_id).expect("tab");
            let saw_scroll = tab
                .page_vm()
                .and_then(|vm| vm.global("__sawScroll"))
                .is_some_and(|v| v == bowser_js::Value::Bool(true));
            assert!(saw_scroll, "pump 1 must fire the scroll listener");
            let installed = tab
                .page_vm()
                .and_then(|vm| vm.global("__scrollendInstalled"))
                .is_some_and(|v| v == bowser_js::Value::Bool(true));
            assert!(
                !installed,
                "scrollend must not fire on the moving tick (pump 1)"
            );
        }
        app.handle_redraw();
        {
            let tab = app.tabs.get(tab_id).expect("tab");
            assert!(
                !tab.dirty.layout,
                "redraw after pump 1 must service the scroll dirt"
            );
            let still_fallback = text_box_width(tab, "X").expect("text box after redraw 1");
            assert!(
                still_fallback < 40.0,
                "no font installed yet: width must stay fallback, got {still_fallback}"
            );
        }
        // Pump 2 (quiet tick): `scrollend` fires, its listener
        // installs the font AFTER this iteration's font drain.
        // The redrain — and only the redrain — marks the reflow:
        // assert it here, with no additional pump call, before any
        // redraw. (The redraw flag alone would not pin this:
        // firing `scrollend` sets it too.)
        app.pump_scripts();
        {
            let tab = app.tabs.get(tab_id).expect("tab");
            let installed = tab
                .page_vm()
                .and_then(|vm| vm.global("__scrollendInstalled"))
                .is_some_and(|v| v == bowser_js::Value::Bool(true));
            assert!(installed, "pump 2 must fire the scrollend listener");
            assert!(
                tab.dirty.layout,
                "windowed pump's redrain must mark the full reflow after the scrollend listener's install"
            );
        }
        assert!(app.needs_redraw, "windowed pump must request a redraw");
        app.handle_redraw();
        let tab = app.tabs.get(tab_id).expect("tab");
        let after = text_box_width(tab, "X").expect("text box after redraw");
        assert!(
            (after - 50.0).abs() < 1.0,
            "redrawn tab must reflow to Ahem's 50 advance, got {after} (before {before})"
        );
    }

    #[test]
    fn pump_headless_reflows_after_synthetic_network_font_completion() {
        // w76fix M9: the network-URL completion path through the
        // headless pump — no data URL, no pool worker. The face is
        // registered but never `load()`ed; a synthetic outcome (200,
        // same-origin, Ahem bytes) is injected via
        // `inject_font_load_completion`, and the pump loop must run
        // it through `complete_font_load`, drain the invalidation,
        // and reflow the text to Ahem's 50 advance.
        // NOTE (w76fix2 M3): this pins end-to-end completion, NOT
        // redrain-after-listener timing — the completion is
        // injected before the pump, so the iteration's ordinary
        // drain catches it. Redrain-after-listener timing is
        // pinned by `pump_scripts_windowed_reflows_after_late_js_font_install`,
        // which installs from a `scrollend` listener on a quiet
        // tick; this test is deliberately left in its
        // completion-before-pump shape so the network path keeps
        // its own coverage.
        use crate::script::collect_script_refs;
        use crate::script_executor::prefetch_script_sources;
        fn text_box_width(tab: &Tab, needle: &str) -> Option<f32> {
            fn walk(bx: &bowser_layout::LayoutBox, needle: &str) -> Option<f32> {
                if bx.box_type == bowser_layout::BoxType::Text && bx.text.as_deref() == Some(needle)
                {
                    return Some(bx.rect.w);
                }
                bx.children.iter().find_map(|c| walk(c, needle))
            }
            tab.render
                .layout_root
                .as_deref()
                .and_then(|root| walk(root, needle))
        }
        let ahem_bytes = include_bytes!("../../bowser_paint/tests/fixtures/Ahem.ttf").to_vec();
        let html =
            b"<html><head></head><body><p style=\"font-family:Ahem;font-size:50px\">X</p></body></html>";
        let url = bowser_url::parse("https://example.com/page", None).unwrap();
        let (document, _) = Document::parse_html(html);
        let script_sources =
            prefetch_script_sources(&collect_script_refs(&document), &url, |_| Ok(String::new()));
        let fonts = test_fonts();
        let mut tab = test_tab();
        crate::navigation::complete_navigation(
            &mut tab,
            LoadedDocument {
                url: url.clone(),
                trace: FetchTrace {
                    hops: vec![url.clone()],
                },
                document,
                stylesheets: Vec::new(),
                image_sizes: HashMap::new(),
                script_sources,
                body: html.to_vec(),
                webfonts: Vec::new(),
            },
            true,
            "TestBrowser/9.9",
            &fonts,
            &test_cookie_jar(),
        );
        refresh(&mut tab, &fonts);
        let before = text_box_width(&tab, "X").expect("text box after initial layout");
        assert!(
            before < 40.0,
            "fallback advance must be under 40 before the install, got {before}"
        );
        {
            let vm = tab.page_vm_mut().expect("page VM installed");
            vm.eval(
                "var __f = new FontFace('Ahem', 'url(https://example.com/Ahem.ttf)'); \
                 document.fonts.add(__f);",
            )
            .expect("face registration eval");
        }
        let ctx = tab
            .page_vm()
            .and_then(|p| p.web_api_context())
            .expect("web api context");
        let outcome: Result<bowser_net::fetch::Response, String> =
            Ok(bowser_net::fetch::Response {
                status: 200,
                headers: Vec::new(),
                body: ahem_bytes,
                url: bowser_url::parse("https://example.com/Ahem.ttf", None).unwrap(),
            });
        assert!(
            bowser_js::inject_font_load_completion(
                bowser_js::ContextId(ctx),
                0xC0FF_EE03,
                "Ahem",
                outcome
            ),
            "synthetic completion must find the registered face"
        );
        let mut frames = 0u64;
        pump_headless_with_budget(
            &mut tab,
            &fonts,
            Duration::from_secs(5),
            HEADLESS_PUMP_TASK_BUDGET,
            &mut frames,
        );
        let after = text_box_width(&tab, "X").expect("text box after pump");
        assert!(
            (after - 50.0).abs() < 1.0,
            "pumped tab must reflow to Ahem's 50 advance, got {after} (before {before})"
        );
    }

    #[test]
    fn install_loaded_document_runs_scripts_in_a_fresh_realm() {
        use crate::script::collect_script_refs;

        fn load_from(html: &[u8]) -> LoadedDocument {
            use crate::script_executor::prefetch_script_sources;

            let url = bowser_url::parse("https://example.com/", None).unwrap();
            let (document, _) = Document::parse_html(html);
            let script_sources =
                prefetch_script_sources(&collect_script_refs(&document), &url, |_| {
                    Ok(String::new())
                });
            LoadedDocument {
                url: url.clone(),
                trace: FetchTrace { hops: vec![url] },
                document,
                stylesheets: Vec::new(),
                image_sizes: HashMap::new(),
                script_sources,
                body: html.to_vec(),
                webfonts: Vec::new(),
            }
        }

        let mut tab = test_tab();
        // Cross-script state rides sloppy-mode globals on the shared
        // global object (the engine scopes each script's lexical `var`s
        // to that script's eval).
        crate::navigation::install_loaded_document(
            &mut tab,
            load_from(
                b"<html><head><script>onlyFirst = 1; marker = 'first';</script></head></html>",
            ),
            "TestBrowser/9.9",
            &test_fonts(),
            &test_cookie_jar(),
        );
        let report = tab.script_report().expect("page VM installed");
        assert_eq!(report.executed, 1);
        assert!(report.errors.is_empty());
        let marker = tab
            .page_vm_mut()
            .expect("page VM installed")
            .eval("marker === 'first'")
            .expect("marker");
        assert_eq!(marker, Value::Bool(true));

        // Reinstalling drops the old realm: its globals are gone and the
        // new document's scripts run in a fresh one.
        crate::navigation::install_loaded_document(
            &mut tab,
            load_from(b"<html><head><script>marker = 'second';</script></head></html>"),
            "TestBrowser/9.9",
            &test_fonts(),
            &test_cookie_jar(),
        );
        let vm = tab.page_vm_mut().expect("page VM installed");
        let marker = vm.eval("marker === 'second'").expect("marker");
        assert_eq!(marker, Value::Bool(true));
        assert!(
            vm.eval("onlyFirst").is_err(),
            "old realm's globals must not survive a full navigation"
        );
    }

    /// Pins the fix for the old "late-added `<img>`s lay out as 0x0
    /// boxes until a full reload" limitation: when a late image's load
    /// completes, the loader's completion hook records its intrinsics,
    /// the tab marks layout dirty only when the outcome differs from
    /// what layout assumed, and the next window refresh grows the box.
    #[test]
    fn late_image_load_invalidates_layout_and_sizes_the_box() {
        use crate::loader::{fetch_late_image, prime_image_cache};

        let url = bowser_url::parse("http://example.com/page", None).unwrap();
        let mut tab = test_tab();
        let (document, _) = Document::parse_html(b"<html><body></body></html>");
        tab.install_test_document(Rc::new(RefCell::new(document)));
        // Sole-ownership install: from here the document is reached
        // only through the tab's scoped `with_document` handle — no
        // strong alias is retained past install.
        tab.history.push(url.clone());

        // The script-side insertion: a late <img> the load-time image
        // step never saw. No intrinsics are recorded, so layout assumes
        // 0x0.
        let img = tab
            .installed()
            .expect("installed document")
            .with_document(|cell| {
                let mut doc = cell.borrow_mut();
                let img = doc.create_element("img");
                doc.set_attribute(img, "src", "/late.png");
                let body = doc.get_elements_by_tag_name("body").item(0).expect("body");
                doc.append_child(body, img);
                img
            })
            .expect("installed document is live");
        // As the script pump would: fold the DOM dirt into the render
        // flags, then refresh the pipeline.
        tab.sync_dirty_flags();
        assert!(tab.dirty.layout && tab.dirty.paint);
        let fonts = Arc::new(Mutex::new(FontCache::new()));
        refresh(&mut tab, &fonts);
        assert!(
            !tab.dirty.layout && !tab.dirty.paint,
            "refresh must have cleared the flags it serviced"
        );

        fn box_for(root: &LayoutBox, node: NodeId) -> Option<&LayoutBox> {
            if root.node == Some(node) {
                return Some(root);
            }
            root.children.iter().find_map(|child| box_for(child, node))
        }
        fn find_img(tab: &Tab, node: NodeId) -> Rect {
            box_for(tab.layout_root.as_deref().expect("layout"), node)
                .expect("img box")
                .rect
        }
        let r = find_img(&tab, img);
        assert_eq!(
            (r.w, r.h),
            (0.0, 0.0),
            "a late img lays out as a 0x0 box before its load completes"
        );

        // The driver scan picks up the late <img> (no bookkeeping yet)
        // and starts a generation-0 fetch for its src.
        let pending = tab.pending_late_images();
        assert_eq!(pending.len(), 1, "the late img must be picked up");
        assert_eq!(pending[0].node, img);
        assert_eq!(pending[0].generation, 0);
        assert!(pending[0].url.serialize().ends_with("/late.png"));

        // The image load completes: the cache is primed with a 4x2
        // decode (no network in the test), and the completion hook
        // reports its intrinsics through the same code path a worker
        // thread would use.
        prime_image_cache(
            &pending[0].url.serialize(),
            DecodedImage::from_rgba(4, 2, vec![255; 4 * 2 * 4]),
        );
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let intrinsics = fetch_late_image(&pending[0].url, &url, &jar, "TestBrowser/9.9");
        assert_eq!(intrinsics, Some(IntrinsicDims::from_size(4.0, 2.0)));

        // The completion lands on the tab: the assumption (0x0) changed,
        // so layout and paint are invalidated.
        tab.note_image_loaded(img, pending[0].generation, intrinsics);
        assert!(
            tab.dirty.layout && tab.dirty.paint,
            "a late load with new intrinsics must invalidate layout+paint"
        );

        // The window refresh pipeline picks it up: the relayout runs
        // with the recorded intrinsics and the box takes its real size.
        refresh(&mut tab, &fonts);
        let r = find_img(&tab, img);
        assert_eq!(
            (r.w, r.h),
            (4.0, 2.0),
            "the img box must take its intrinsic size after the late load"
        );
        assert!(
            !tab.dirty.layout && !tab.dirty.paint,
            "refresh must have cleared the flags it serviced"
        );

        // A duplicate delivery of the same completion is a no-op: no
        // redundant relayout is scheduled.
        tab.note_image_loaded(
            img,
            pending[0].generation,
            Some(IntrinsicDims::from_size(4.0, 2.0)),
        );
        assert!(
            !tab.dirty.layout && !tab.dirty.paint,
            "an unchanged completion must not invalidate anything"
        );

        // Failure path: a second late img whose load fails keeps the
        // assumed 0x0, so nothing is invalidated. (The failure is fed
        // directly: a real fetch would need the network, and the loader
        // reports `None` on any failure.)
        let failed = tab
            .installed()
            .expect("installed document")
            .with_document(|cell| {
                let mut doc = cell.borrow_mut();
                let img = doc.create_element("img");
                doc.set_attribute(img, "src", "/missing.png");
                let body = doc.get_elements_by_tag_name("body").item(0).expect("body");
                doc.append_child(body, img);
                img
            })
            .expect("installed document is live");
        tab.sync_dirty_flags();
        refresh(&mut tab, &fonts);
        let pending = tab.pending_late_images();
        assert_eq!(pending.len(), 1, "only the failed img needs a fetch");
        assert_eq!(pending[0].node, failed);
        tab.note_image_loaded(failed, pending[0].generation, None);
        assert!(
            !tab.dirty.layout && !tab.dirty.paint,
            "a failed late load at the assumed 0x0 must invalidate nothing"
        );
    }

    /// F6 regression: the driver trigger must refetch an already-loaded
    /// `<img>` whose `src` changes. The old "no intrinsics entry"
    /// trigger never refetched it, so the new URL's pixels never
    /// arrived.
    #[test]
    fn late_image_src_change_triggers_refetch() {
        use crate::loader::{fetch_late_image, prime_image_cache};

        let url = bowser_url::parse("http://example.com/page", None).unwrap();
        let mut tab = test_tab();
        let (document, _) = Document::parse_html(b"<html><body><img src=\"/a.png\"></body></html>");
        tab.install_test_document(Rc::new(RefCell::new(document)));
        // Sole-ownership install: from here the document is reached
        // only through the tab's scoped `with_document` handle — no
        // strong alias is retained past install.
        tab.history.push(url.clone());
        let img = tab
            .installed()
            .expect("installed document")
            .with_document(|cell| {
                cell.borrow()
                    .get_elements_by_tag_name("img")
                    .item(0)
                    .expect("img")
            })
            .expect("installed document is live");

        // First scan: the img is new, so the driver starts a
        // generation-0 fetch for /a.png.
        let pending = tab.pending_late_images();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].generation, 0);
        assert!(pending[0].url.serialize().ends_with("/a.png"));
        prime_image_cache(
            &pending[0].url.serialize(),
            DecodedImage::from_rgba(4, 2, vec![1; 4 * 2 * 4]),
        );
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let size = fetch_late_image(&pending[0].url, &url, &jar, "TestBrowser/9.9");
        tab.note_image_loaded(img, pending[0].generation, size);
        assert_eq!(
            tab.image_intrinsics[&img],
            IntrinsicDims::from_size(4.0, 2.0)
        );
        assert_eq!(tab.image_loads[&img].generation, 0);

        // No DOM change: the recorded URL matches, so nothing is
        // refetched.
        tab.dirty = DirtyFlags::default();
        assert!(
            tab.pending_late_images().is_empty(),
            "an unchanged src must not refetch"
        );

        // Script changes src to /b.png: the next scan sees the recorded
        // URL differ from the current src, bumps the generation, and
        // refetches.
        tab.installed()
            .expect("installed document")
            .with_document(|cell| cell.borrow_mut().set_attribute(img, "src", "/b.png"))
            .expect("installed document is live");
        let pending = tab.pending_late_images();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].node, img);
        assert_eq!(
            pending[0].generation, 1,
            "a src change must bump the generation"
        );
        assert!(pending[0].url.serialize().ends_with("/b.png"));
        prime_image_cache(
            &pending[0].url.serialize(),
            DecodedImage::from_rgba(8, 1, vec![2; 8 * 4]),
        );
        let size = fetch_late_image(&pending[0].url, &url, &jar, "TestBrowser/9.9");
        assert_eq!(size, Some(IntrinsicDims::from_size(8.0, 1.0)));
        tab.note_image_loaded(img, pending[0].generation, size);
        assert!(
            tab.dirty.layout && tab.dirty.paint,
            "a new size must invalidate layout+paint"
        );
        assert_eq!(
            tab.image_intrinsics[&img],
            IntrinsicDims::from_size(8.0, 1.0)
        );
        assert!(tab.image_loads[&img].url.ends_with("/b.png"));
    }

    /// F5 regression: a new completion whose decoded size equals the
    /// recorded one must still repaint. The pixels are new — the last
    /// paint showed a placeholder while the fetch was in flight — even
    /// though layout is unaffected.
    #[test]
    fn late_image_same_size_completion_repaints() {
        use crate::loader::{fetch_late_image, prime_image_cache};

        let url = bowser_url::parse("http://example.com/page", None).unwrap();
        let mut tab = test_tab();
        let (document, _) = Document::parse_html(b"<html><body><img src=\"/a.png\"></body></html>");
        tab.install_test_document(Rc::new(RefCell::new(document)));
        // Sole-ownership install: from here the document is reached
        // only through the tab's scoped `with_document` handle — no
        // strong alias is retained past install.
        tab.history.push(url.clone());
        let img = tab
            .installed()
            .expect("installed document")
            .with_document(|cell| {
                cell.borrow()
                    .get_elements_by_tag_name("img")
                    .item(0)
                    .expect("img")
            })
            .expect("installed document is live");

        let pending = tab.pending_late_images();
        prime_image_cache(
            &pending[0].url.serialize(),
            DecodedImage::from_rgba(4, 2, vec![1; 4 * 2 * 4]),
        );
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let size = fetch_late_image(&pending[0].url, &url, &jar, "TestBrowser/9.9");
        tab.note_image_loaded(img, pending[0].generation, size);
        assert_eq!(
            tab.image_intrinsics[&img],
            IntrinsicDims::from_size(4.0, 2.0)
        );

        // Script swaps src A→B (both decode 4x2). The driver records the
        // new URL and bumps the generation; then a paint pass runs while
        // B is still uncached, leaving a placeholder on screen.
        tab.installed()
            .expect("installed document")
            .with_document(|cell| cell.borrow_mut().set_attribute(img, "src", "/b.png"))
            .expect("installed document is live");
        let pending = tab.pending_late_images();
        assert_eq!(pending[0].generation, 1);
        tab.dirty = DirtyFlags::default(); // the placeholder paint pass

        // B's fetch completes at the already-recorded size: paint must
        // re-run for the new pixels, layout must not.
        prime_image_cache(
            &pending[0].url.serialize(),
            DecodedImage::from_rgba(4, 2, vec![2; 4 * 2 * 4]),
        );
        let size = fetch_late_image(&pending[0].url, &url, &jar, "TestBrowser/9.9");
        assert_eq!(size, Some(IntrinsicDims::from_size(4.0, 2.0)));
        tab.note_image_loaded(img, pending[0].generation, size);
        assert!(
            tab.dirty.paint,
            "new pixels must repaint even at an unchanged size"
        );
        assert!(
            !tab.dirty.layout,
            "an unchanged size must not invalidate layout"
        );
    }

    /// F7 regression: out-of-order completions must not install stale
    /// sizes. src goes A→B→A quickly; B's slow fetch lands last and is
    /// dropped instead of recording B's size.
    #[test]
    fn late_image_stale_completions_are_dropped() {
        let url = bowser_url::parse("http://example.com/page", None).unwrap();
        let mut tab = test_tab();
        let (document, _) = Document::parse_html(b"<html><body><img src=\"/a.png\"></body></html>");
        tab.install_test_document(Rc::new(RefCell::new(document)));
        // Sole-ownership install: from here the document is reached
        // only through the tab's scoped `with_document` handle — no
        // strong alias is retained past install.
        tab.history.push(url.clone());
        let img = tab
            .installed()
            .expect("installed document")
            .with_document(|cell| {
                cell.borrow()
                    .get_elements_by_tag_name("img")
                    .item(0)
                    .expect("img")
            })
            .expect("installed document is live");

        // Three fetches start in quick succession: A, then B, then A
        // again. Each bumps the generation and records its URL.
        let gen_a1 = tab.pending_late_images().pop().expect("fetch for /a.png");
        assert_eq!(gen_a1.generation, 0);
        tab.installed()
            .expect("installed document")
            .with_document(|cell| cell.borrow_mut().set_attribute(img, "src", "/b.png"))
            .expect("installed document is live");
        let gen_b = tab.pending_late_images().pop().expect("refetch for /b.png");
        assert_eq!(gen_b.generation, 1);
        assert!(gen_b.url.serialize().ends_with("/b.png"));
        tab.installed()
            .expect("installed document")
            .with_document(|cell| cell.borrow_mut().set_attribute(img, "src", "/a.png"))
            .expect("installed document is live");
        let gen_a2 = tab.pending_late_images().pop().expect("refetch for /a.png");
        assert_eq!(gen_a2.generation, 2);

        // The latest fetch (A, generation 2) completes first: new size,
        // so layout+paint.
        // (Completions are fed directly; the fetch layer is covered by
        // the loader's own tests.)
        tab.note_image_loaded(
            img,
            gen_a2.generation,
            Some(IntrinsicDims::from_size(4.0, 2.0)),
        );
        assert!(tab.dirty.layout && tab.dirty.paint);
        tab.dirty = DirtyFlags::default();

        // The first A's completion arrives late: stale (generation 0 <
        // current 2), so it is dropped even though the URL matches —
        // only the current generation ever applies.
        tab.note_image_loaded(
            img,
            gen_a1.generation,
            Some(IntrinsicDims::from_size(4.0, 2.0)),
        );
        assert!(
            !tab.dirty.layout && !tab.dirty.paint,
            "a stale same-URL completion must invalidate nothing"
        );

        // B's slow fetch lands last: stale (generation 1 < current 2),
        // so it is dropped — B's size is never installed.
        tab.note_image_loaded(
            img,
            gen_b.generation,
            Some(IntrinsicDims::from_size(8.0, 8.0)),
        );
        assert!(
            !tab.dirty.layout && !tab.dirty.paint,
            "a stale completion must invalidate nothing"
        );
        assert_eq!(
            tab.image_intrinsics[&img],
            IntrinsicDims::from_size(4.0, 2.0),
            "the stale size must not be recorded"
        );
        assert!(
            tab.image_loads[&img].url.ends_with("/a.png"),
            "the recorded URL stays at the latest fetch"
        );

        // A duplicate delivery of the applied completion is dropped too.
        tab.note_image_loaded(
            img,
            gen_a2.generation,
            Some(IntrinsicDims::from_size(4.0, 2.0)),
        );
        assert!(
            !tab.dirty.layout && !tab.dirty.paint,
            "a duplicate completion must invalidate nothing"
        );
    }

    /// F8: a failed load leaves the recorded size unchanged — the old
    /// doc claimed this while the code removed the entry and collapsed
    /// the box to 0x0.
    #[test]
    fn late_image_failed_load_keeps_recorded_size() {
        let url = bowser_url::parse("http://example.com/page", None).unwrap();
        let mut tab = test_tab();
        let (document, _) = Document::parse_html(b"<html><body><img src=\"/a.png\"></body></html>");
        tab.install_test_document(Rc::new(RefCell::new(document)));
        // Sole-ownership install: from here the document is reached
        // only through the tab's scoped `with_document` handle — no
        // strong alias is retained past install.
        tab.history.push(url.clone());
        let img = tab
            .installed()
            .expect("installed document")
            .with_document(|cell| {
                cell.borrow()
                    .get_elements_by_tag_name("img")
                    .item(0)
                    .expect("img")
            })
            .expect("installed document is live");

        // /a.png loads at 4x2.
        let pending = tab.pending_late_images();
        tab.note_image_loaded(
            img,
            pending[0].generation,
            Some(IntrinsicDims::from_size(4.0, 2.0)),
        );
        tab.dirty = DirtyFlags::default();

        // src changes to /b.png, whose fetch fails (fed as None, the way
        // fetch_late_image reports any failure).
        tab.installed()
            .expect("installed document")
            .with_document(|cell| cell.borrow_mut().set_attribute(img, "src", "/b.png"))
            .expect("installed document is live");
        let pending = tab.pending_late_images();
        assert_eq!(pending[0].generation, 1);
        tab.note_image_loaded(img, pending[0].generation, None);

        // Nothing is invalidated and the old size stays recorded: the
        // box keeps its 4x2 instead of collapsing to 0x0.
        assert!(
            !tab.dirty.layout && !tab.dirty.paint,
            "a failed load must invalidate nothing"
        );
        assert_eq!(
            tab.image_intrinsics[&img],
            IntrinsicDims::from_size(4.0, 2.0),
            "a failed load keeps the recorded size"
        );
        // The failure is recorded against its generation, so a later
        // retry starts from a distinguishable state...
        assert_eq!(tab.image_loads[&img].completed_generation, Some(1));
        // ...and a duplicate failure delivery is dropped.
        tab.note_image_loaded(img, pending[0].generation, None);
        assert!(!tab.dirty.layout && !tab.dirty.paint);
    }

    /// Load-time images are seeded, not refetched: an `<img>` with an
    /// intrinsics entry but no bookkeeping entry (installed by
    /// navigation, which the driver has not scanned yet) is recorded as
    /// a completed generation 0 on the first scan.
    #[test]
    fn pending_late_images_seeds_load_time_images_without_refetch() {
        let url = bowser_url::parse("http://example.com/page", None).unwrap();
        let mut tab = test_tab();
        let (document, _) = Document::parse_html(b"<html><body><img src=\"/a.png\"></body></html>");
        tab.install_test_document(Rc::new(RefCell::new(document)));
        // Sole-ownership install: from here the document is reached
        // only through the tab's scoped `with_document` handle — no
        // strong alias is retained past install.
        tab.history.push(url.clone());
        let img = tab
            .installed()
            .expect("installed document")
            .with_document(|cell| {
                cell.borrow()
                    .get_elements_by_tag_name("img")
                    .item(0)
                    .expect("img")
            })
            .expect("installed document is live");
        // As navigation would leave it: intrinsics installed, no
        // bookkeeping yet.
        tab.image_intrinsics
            .insert(img, IntrinsicDims::from_size(4.0, 2.0));

        assert!(
            tab.pending_late_images().is_empty(),
            "a load-time image must not be refetched"
        );
        let load = &tab.image_loads[&img];
        assert!(load.url.ends_with("/a.png"));
        assert_eq!(load.generation, 0);
        assert_eq!(load.completed_generation, Some(0));

        // ...but a later src change on it IS picked up.
        tab.installed()
            .expect("installed document")
            .with_document(|cell| cell.borrow_mut().set_attribute(img, "src", "/b.png"))
            .expect("installed document is live");
        let pending = tab.pending_late_images();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].generation, 1);
        assert!(pending[0].url.serialize().ends_with("/b.png"));
    }

    #[test]
    fn initial_load_rewrites_history_entry_to_final_url() {
        // The async initial load records the pre-redirect URL at tab
        // creation (`Browser::new_tab`) and installs the final document
        // without pushing a new entry: the recorded entry must be
        // rewritten to the final URL so history and the installed
        // document (and relative URL resolution against the entry) agree.
        let mut tab = test_tab();
        let requested = bowser_url::parse("http://example.com/", None).unwrap();
        tab.history.push(requested);
        let final_url = bowser_url::parse("http://example.com/landing", None).unwrap();
        let html = b"<html><head><title>landed</title></head><body></body></html>";
        let (document, _) = Document::parse_html(html);
        install_final_document(
            &mut tab,
            LoadedDocument {
                url: final_url.clone(),
                trace: FetchTrace {
                    hops: vec![final_url.clone()],
                },
                document,
                stylesheets: Vec::new(),
                image_sizes: HashMap::new(),
                script_sources: HashMap::new(),
                body: html.to_vec(),
                webfonts: Vec::new(),
            },
            bowser_net::fetch::USER_AGENT,
            &test_fonts(),
            &test_cookie_jar(),
        );
        assert_eq!(tab.title, "landed", "final document installed");
        assert_eq!(tab.history.len(), 1, "no new entry pushed");
        assert_eq!(
            tab.current_url(),
            Some(&final_url),
            "history entry rewritten to the final URL after redirects"
        );
    }

    #[test]
    fn initial_load_without_redirect_keeps_history_entry() {
        let mut tab = test_tab();
        let url = bowser_url::parse("http://example.com/", None).unwrap();
        tab.history.push(url.clone());
        let html = b"<html><head><title>t</title></head><body></body></html>";
        let (document, _) = Document::parse_html(html);
        install_final_document(
            &mut tab,
            LoadedDocument {
                url: url.clone(),
                trace: FetchTrace {
                    hops: vec![url.clone()],
                },
                document,
                stylesheets: Vec::new(),
                image_sizes: HashMap::new(),
                script_sources: HashMap::new(),
                body: html.to_vec(),
                webfonts: Vec::new(),
            },
            bowser_net::fetch::USER_AGENT,
            &test_fonts(),
            &test_cookie_jar(),
        );
        assert_eq!(tab.history.len(), 1, "no new entry pushed");
        assert_eq!(tab.current_url(), Some(&url), "entry untouched");
    }

    #[test]
    fn pump_scripts_fires_page_timers_on_the_active_tab() {
        use crate::script::{PageVm, collect_script_refs, resolve_scripts};

        let mut tabs = TabArena::default();
        let tab = tabs.insert(Tab::new(Size { w: 800.0, h: 600.0 }));
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, tab);

        // No page VM yet (no document installed): the pump is a no-op,
        // not a panic.
        app.pump_scripts();

        // A page whose script registers a 0 ms timer; the effect is
        // observed through a sloppy global (each eval's lexical scope is
        // its own, the global object is shared).
        let (document, _) = Document::parse_html(
            b"<html><head><script>setTimeout(() => { timerFired = true; }, 0);</script></head></html>",
        );
        let url = bowser_url::parse("https://example.com/", None).unwrap();
        let scripts = resolve_scripts(collect_script_refs(&document), &url, |_| Ok(String::new()));
        let mut page_vm = PageVm::new();
        page_vm.run(&scripts);
        assert!(page_vm.report().errors.is_empty());
        app.tabs
            .get_mut(tab)
            .unwrap()
            .install_document(Rc::new(RefCell::new(document)), page_vm);

        let timer_fired = |app: &mut WindowApp<'_>| {
            app.tabs
                .get_mut(tab)
                .unwrap()
                .page_vm_mut()
                .unwrap()
                .global("timerFired")
        };
        // The timer has not fired: nothing pumps the loop during `run`.
        assert_eq!(timer_fired(&mut app), None);

        // One window tick fires it.
        app.pump_scripts();
        assert_eq!(timer_fired(&mut app), Some(Value::Bool(true)));
    }

    #[test]
    fn pump_scripts_fires_page_timers_on_background_tabs() {
        use crate::script::{PageVm, collect_script_refs, resolve_scripts};

        let mut tabs = TabArena::default();
        let active = tabs.insert(Tab::new(Size { w: 800.0, h: 600.0 }));
        let background = tabs.insert(Tab::new(Size { w: 800.0, h: 600.0 }));
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, active);

        // The background tab's page registers a 0 ms timer.
        let (document, _) = Document::parse_html(
            b"<html><head><script>setTimeout(() => { bgTimerFired = true; }, 0);</script></head></html>",
        );
        let url = bowser_url::parse("https://example.com/", None).unwrap();
        let scripts = resolve_scripts(collect_script_refs(&document), &url, |_| Ok(String::new()));
        let mut page_vm = PageVm::new();
        page_vm.run(&scripts);
        assert!(page_vm.report().errors.is_empty());
        app.tabs
            .get_mut(background)
            .unwrap()
            .install_document(Rc::new(RefCell::new(document)), page_vm);

        // One window tick fires the background tab's timer even though
        // it is not the active tab.
        app.pump_scripts();

        let fired = app
            .tabs
            .get_mut(background)
            .unwrap()
            .page_vm_mut()
            .unwrap()
            .global("bgTimerFired");
        assert_eq!(fired, Some(Value::Bool(true)));
        // ...but its paint dirt must not schedule a repaint: only the
        // active tab's dirt does.
        assert!(
            !app.needs_redraw,
            "background tab timers must not request a window redraw"
        );
    }

    /// Minimal two-page HTTP server for traversal tests.
    fn serve_two_pages() -> (String, String) {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("test server addr");
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                std::thread::spawn(move || {
                    let mut stream = stream;
                    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 4096];
                    loop {
                        match stream.read(&mut chunk) {
                            Ok(0) => break,
                            Ok(n) => {
                                buf.extend_from_slice(&chunk[..n]);
                                if buf.windows(4).any(|w| w == b"\r\n\r\n") || buf.len() > 65_536 {
                                    break;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                    let head = String::from_utf8_lossy(&buf);
                    let path = head
                        .lines()
                        .next()
                        .and_then(|l| l.split_whitespace().nth(1))
                        .unwrap_or("/");
                    let title = if path.contains("/b") {
                        "Page B"
                    } else {
                        "Page A"
                    };
                    let body = format!(
                        "<html><head><title>{title}</title></head><body>{title}</body></html>"
                    );
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(head.as_bytes());
                    let _ = stream.write_all(body.as_bytes());
                });
            }
        });
        (format!("http://{addr}/a"), format!("http://{addr}/b"))
    }

    #[test]
    fn browser_back_key_uses_fragment_fast_path() {
        let mut tabs = TabArena::default();
        let tab = tabs.insert(test_tab());
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, tab);

        // Two same-document entries with a loaded document: traversal
        // scrolls synchronously, no load spawns.
        let url_a = bowser_url::parse("http://example.com/page#one", None).unwrap();
        let url_b = bowser_url::parse("http://example.com/page#two", None).unwrap();
        {
            let t = app.tabs.get_mut(tab).unwrap();
            t.history.push(url_a);
            t.history.push(url_b);
            let (doc, _) = Document::parse_html(
                b"<html><body><div id=\"one\">a</div><div id=\"two\">b</div></body></html>",
            );
            t.install_test_document(Rc::new(RefCell::new(doc)));
        }

        app.on_key(&Key::Named(NamedKey::BrowserBack), true);

        let t = app.tabs.get(tab).unwrap();
        assert_eq!(t.history.index(), 0, "fragment back moves the index");
        assert!(
            app.pending.is_none(),
            "same-document traversal must not spawn a load"
        );

        // And forward again.
        app.on_key(&Key::Named(NamedKey::BrowserForward), true);
        assert_eq!(app.tabs.get(tab).unwrap().history.index(), 1);
        assert!(app.pending.is_none());
    }

    /// Pump `poll_pending` until `condition` holds or the deadline fires.
    /// Non-`wpt-driver` twin of the gated helper further below.
    fn pump_until_settled(
        app: &mut WindowApp<'_>,
        deadline: Instant,
        condition: impl Fn(&WindowApp<'_>) -> bool,
    ) {
        while !condition(app) {
            app.poll_pending();
            assert!(
                Instant::now() < deadline,
                "timed out waiting for background load"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn browser_back_key_traverses_through_async_loader() {
        let (url_a, url_b) = serve_two_pages();
        let mut tabs = TabArena::default();
        let tab = tabs.insert(test_tab());
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, tab);

        // Initial load of A (entry recorded at tab creation, like
        // `Browser::new_tab`), then a push navigation to B.
        let parsed_a = bowser_url::parse(&url_a, None).unwrap();
        app.tabs
            .get_mut(tab)
            .unwrap()
            .history
            .push(parsed_a.clone());
        app.spawn_load(&parsed_a, false, None);
        let deadline = Instant::now() + Duration::from_secs(15);
        pump_until_settled(&mut app, deadline, |app| {
            app.pending.is_none() && app.tabs.get(tab).is_some_and(|t| t.title == "Page A")
        });
        let parsed_b = bowser_url::parse(&url_b, None).unwrap();
        app.spawn_load(&parsed_b, true, None);
        let deadline = Instant::now() + Duration::from_secs(15);
        pump_until_settled(&mut app, deadline, |app| {
            app.pending.is_none() && app.tabs.get(tab).is_some_and(|t| t.title == "Page B")
        });
        assert_eq!(app.tabs.get(tab).unwrap().history.index(), 1);

        // BrowserBack: a full traversal loads through the async worker —
        // never the blocking traversal, which would steal the profile
        // jar out from under the in-flight load's cookie handoff.
        app.on_key(&Key::Named(NamedKey::BrowserBack), true);
        assert!(
            app.pending.is_some(),
            "cross-document traversal must spawn an async load"
        );

        let deadline = Instant::now() + Duration::from_secs(15);
        pump_until_settled(&mut app, deadline, |app| {
            app.pending.is_none() && app.tabs.get(tab).is_some_and(|t| t.title == "Page A")
        });
        let t = app.tabs.get(tab).unwrap();
        assert_eq!(
            t.history.index(),
            0,
            "history index moved back on completion"
        );
        assert_eq!(t.history.len(), 2, "traversal must not push a new entry");
    }

    #[test]
    fn link_href_only_for_anchors() {
        let (doc, _) = Document::parse_html(b"<a href=\"/x\">x</a><span>no</span>");
        let a = doc.get_elements_by_tag_name("a").item(0).expect("anchor");
        let span = doc.get_elements_by_tag_name("span").item(0).expect("span");
        assert_eq!(link_href(&doc, a).as_deref(), Some("/x"));
        assert_eq!(link_href(&doc, span), None);
    }

    #[test]
    fn target_at_accounts_for_scroll() {
        let (tab, first, _) = tab_with_boxes();
        // Fake the scrollable bounds `refresh` would publish, then scroll.
        tab.scroll().set_max_offset((0.0, 500.0));
        tab.scroll().set_offset(0.0, 100.0);
        // Viewport (400, 20) maps to page (400, 120): inside `second`'s box.
        let target = target_at(&tab, 400.0, 20.0);
        assert_ne!(target, Some(first));
    }

    /// Manual smoke test: needs a display (run under xvfb or natively).
    /// Opens about:blank, renders the welcome page, exits on close.
    #[test]
    #[ignore = "needs a display; run manually"]
    fn windowed_smoke() {
        use crate::browser::{Browser, BrowserConfig};
        let mut browser = Browser::new(BrowserConfig::default());
        let tab = browser.new_tab("about:blank");
        run_windowed(&mut browser, tab).expect("windowed run");
    }

    // -- background load plumbing (no window needed) ---------------------

    use std::time::Instant;

    fn test_app<'a>(
        tabs: &'a mut TabArena,
        jar: &Arc<Mutex<CookieJar>>,
        tab: TabId,
    ) -> WindowApp<'a> {
        test_app_with_js(tabs, jar, tab, None)
    }

    fn test_app_with_js<'a>(
        tabs: &'a mut TabArena,
        jar: &Arc<Mutex<CookieJar>>,
        tab: TabId,
        js: Option<Box<dyn JsDispatch>>,
    ) -> WindowApp<'a> {
        WindowApp {
            tabs,
            jar: Arc::clone(jar),
            tab,
            viewport: (800, 600),
            window: None,
            gpu: None,
            input: InputState::default(),
            modifiers: ModifiersState::default(),
            touch: TouchScrollState::new(),
            scrollbar_drag: None,
            autoscroll: None,
            wheel_latch: None,
            pending: None,
            stale: Vec::new(),
            handoff_tx: None,
            load_control: LoadControl::new(),
            js,
            user_agent: "TestBrowser/9.9".to_string(),
            fonts: Arc::new(Mutex::new(FontCache::new())),
            scale: 1.0,
            needs_redraw: false,
            init_error: None,
        }
    }

    #[derive(Default)]
    struct CallLog {
        clicks: Vec<NodeId>,
        keys: Vec<String>,
        wheels: Vec<(NodeId, f64, f64, u32)>,
        scrolls: Vec<NodeId>,
        scrollends: Vec<NodeId>,
        snaps: Vec<(Option<NodeId>, String, SnapTargets)>,
        /// Unified dispatch order across scroll/snap/scrollend, for
        /// the snap-event ordering assertions.
        order: Vec<String>,
    }

    /// [`JsDispatch`] test double recording dispatched events.
    #[derive(Clone, Default)]
    struct StubDispatch {
        log: std::rc::Rc<std::cell::RefCell<CallLog>>,
        /// What `dispatch_*` report: whether the page prevented the
        /// default action.
        prevent: bool,
    }

    impl JsDispatch for StubDispatch {
        fn dispatch_click(&mut self, _tab: &mut Tab, target: NodeId) -> bool {
            self.log.borrow_mut().clicks.push(target);
            self.prevent
        }

        fn dispatch_key(&mut self, _tab: &mut Tab, key: &str) -> bool {
            self.log.borrow_mut().keys.push(key.to_owned());
            self.prevent
        }

        fn dispatch_wheel(
            &mut self,
            _tab: &mut Tab,
            target: NodeId,
            delta_x: f64,
            delta_y: f64,
            delta_mode: u32,
        ) -> bool {
            self.log
                .borrow_mut()
                .wheels
                .push((target, delta_x, delta_y, delta_mode));
            self.prevent
        }

        fn dispatch_scroll(&mut self, _tab: &mut Tab, target: NodeId) {
            let mut log = self.log.borrow_mut();
            log.scrolls.push(target);
            log.order.push("scroll".to_string());
        }

        fn dispatch_scrollend(&mut self, _tab: &mut Tab, target: NodeId) {
            let mut log = self.log.borrow_mut();
            log.scrollends.push(target);
            log.order.push("scrollend".to_string());
        }

        fn dispatch_snap_event(
            &mut self,
            _tab: &mut Tab,
            target: Option<NodeId>,
            event_type: &str,
            targets: SnapTargets,
        ) {
            let mut log = self.log.borrow_mut();
            log.snaps.push((target, event_type.to_string(), targets));
            log.order.push(event_type.to_string());
        }
    }

    /// A tab whose `#sc` is a mandatory snap container with two
    /// targets (`#a` at 0, `#b` at 100), installed the production way
    /// (build + carry from the placeholder tree) with the initial
    /// snap-event deliveries drained.
    fn tab_with_two_target_snap_scroller() -> (Tab, NodeId, NodeId) {
        use bowser_css::parse_stylesheet;
        use bowser_layout::{FixedFontMetrics, layout};
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"sc\"><div id=\"a\">x</div><div id=\"b\">y</div></div></body></html>",
        );
        let sc = doc.get_element_by_id("sc").expect("sc");
        let b = doc.get_element_by_id("b").expect("b");
        let sheet = parse_stylesheet(
            "#sc { overflow-y: auto; height: 100px; width: 200px; \
               scroll-snap-type: y mandatory; } \
             #a { height: 100px; scroll-snap-align: start; } \
             #b { height: 100px; scroll-snap-align: start; }",
        );
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        tab.layout_root = Some(Rc::new(root));
        let mut tree = {
            let doc = tab.document_ref().expect("installed document");
            ScrollTree::build(
                &doc,
                tab.layout_root.as_deref().expect("layout root"),
                tab.viewport,
                tab.scroll(),
                tab.element_scroll(),
            )
        };
        let old = ScrollTree::for_viewport(tab.scroll(), tab.viewport, (true, true));
        tree.carry_scroll_state(&old);
        // The initial layout-snap deliveries are not this test's subject.
        let _ = tree.take_snap_events();
        tab.scroll_tree = tree;
        (tab, sc, b)
    }

    #[test]
    fn snap_event_ordering_changing_scroll_change_scrollend() {
        // One wheel scroll on a snapped container, driven through the
        // production funnels: `scrollsnapchanging` precedes the
        // movement's `scroll`, and `scrollsnapchange` precedes
        // `scrollend` (CSS Scroll Snap 2 §5).
        let (tab, sc, b) = tab_with_two_target_snap_scroller();
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let stub = StubDispatch::default();
        let log = stub.log.clone();
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, Some(Box::new(stub)));

        {
            let tab = app.tabs.get(tab_id).expect("tab");
            let isc = tab.scroll_tree.index_of(sc).expect("sc container");
            tab.scroll_tree.scroll_chain(isc, 0.0, 60.0);
            tab.scroll_tree
                .advance_smooth(Instant::now() + Duration::from_secs(10));
        }
        app.fire_scroll_listeners_for(tab_id, &[Some(sc)]);
        app.pump_scripts();
        app.pump_scripts();
        assert_eq!(
            log.borrow().order,
            vec![
                "scrollsnapchanging".to_string(),
                "scroll".to_string(),
                "scrollsnapchange".to_string(),
                "scrollend".to_string(),
            ]
        );
        assert_eq!(
            log.borrow().snaps,
            vec![
                (
                    Some(sc),
                    "scrollsnapchanging".to_string(),
                    SnapTargets {
                        block: Some(b),
                        inline: None,
                    },
                ),
                (
                    Some(sc),
                    "scrollsnapchange".to_string(),
                    SnapTargets {
                        block: Some(b),
                        inline: None,
                    },
                ),
            ]
        );
    }

    #[test]
    fn snap_event_resnap_without_scroll_fires_changing_then_change() {
        // A relayout that removes the settled target produces the
        // layout-snap pair at the top of the next pump. The resnap
        // also moved the scroller (100 -> 0), and a layout-induced
        // offset change is a real scroll: the fold reports it right
        // after the snap pair (`scrollend` lands on a later quiet
        // tick, beyond this single pump).
        use bowser_css::parse_stylesheet;
        use bowser_layout::{FixedFontMetrics, layout};
        let (mut tab, sc, _b) = tab_with_two_target_snap_scroller();
        // Settle on #b first (offset 100 is its snap position): run
        // the scroll to completion so `scrollsnapchange` has reported
        // #b before the relayout.
        {
            let isc = tab.scroll_tree.index_of(sc).expect("sc container");
            tab.scroll_tree.scroll_chain(isc, 0.0, 100.0);
            tab.scroll_tree.mark_scrolled(&[Some(sc)]);
            let _ = tab.scroll_tree.end_tick();
            let _ = tab.scroll_tree.end_tick();
            let _ = tab.scroll_tree.take_snap_events();
        }
        // Relayout without #b: resnap falls back to #a at 0.
        let sheet = parse_stylesheet(
            "#sc { overflow-y: auto; height: 100px; width: 200px; \
               scroll-snap-type: y mandatory; } \
             #a { height: 100px; scroll-snap-align: start; }",
        );
        let root = {
            let doc = tab.document_ref().expect("installed document");
            layout(
                &doc,
                &[&sheet],
                Size { w: 800.0, h: 600.0 },
                &FixedFontMetrics,
            )
        };
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();

        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let stub = StubDispatch::default();
        let log = stub.log.clone();
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, Some(Box::new(stub)));
        app.pump_scripts();
        assert_eq!(
            log.borrow().order,
            vec![
                "scrollsnapchanging".to_string(),
                "scrollsnapchange".to_string(),
                "scroll".to_string(),
            ]
        );
    }

    /// A tab with a real page realm (so `window` scroll/scrollend
    /// listeners observe the viewport dispatches), laid out with `css`
    /// and rebuilt after layout, with the install-time snap deliveries
    /// drained. Viewport `scroll`/`scrollend` dispatch through the page
    /// rather than the `JsDispatch` bridge, so the test observes them
    /// with page listeners on `window.__events`; snap events go through
    /// the bridge's `StubDispatch` log.
    fn viewport_snap_tab(css: &str) -> Tab {
        use crate::script::PageVm;
        use bowser_css::parse_stylesheet;
        use bowser_layout::{FixedFontMetrics, layout};
        let (document, _) = Document::parse_html(
            b"<html><head></head><body><div id=\"sp\"></div><div id=\"a\">x</div>\
              <div id=\"b\">y</div><div id=\"tail\">z</div></body></html>",
        );
        let sheet = parse_stylesheet(css);
        let root = layout(
            &document,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let document = Rc::new(RefCell::new(document));
        let url = bowser_url::parse("https://example.com/", None).unwrap();
        let page_vm =
            PageVm::new_for_document(Rc::clone(&document), &url, "test-ua", &[]).expect("realm");
        let mut tab = test_tab();
        // Wire the tab's viewport scroll state into the page realm, like
        // navigation does at document install.
        if let Some(ctx) = page_vm.web_api_context() {
            bowser_js::share_scroll_state(ctx, Rc::clone(tab.scroll()));
        }
        tab.install_document(document, page_vm);
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        // The initial layout-snap deliveries are not these tests' subject.
        let _ = tab.scroll_tree.take_snap_events();
        tab.page_vm_mut()
            .unwrap()
            .eval(
                "window.__events = [];\
                 window.addEventListener('scroll', function() { window.__events.push('scroll'); });\
                 window.addEventListener('scrollend', function() { window.__events.push('scrollend'); });",
            )
            .unwrap();
        tab
    }

    /// Relayout `tab` with `css`, the way a style change would.
    fn viewport_relayout(tab: &mut Tab, css: &str) {
        use bowser_css::parse_stylesheet;
        use bowser_layout::{FixedFontMetrics, layout};
        let sheet = parse_stylesheet(css);
        let root = {
            let doc = tab.document_ref().expect("installed document");
            layout(
                &doc,
                &[&sheet],
                Size { w: 800.0, h: 600.0 },
                &FixedFontMetrics,
            )
        };
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
    }

    /// Read the page-side `__events` log for `tab_id`, comma-joined.
    fn page_events(app: &mut WindowApp<'_>, tab_id: TabId) -> String {
        let tab = app.tabs.get_mut(tab_id).expect("tab");
        match tab.page_vm_mut().unwrap().eval("window.__events.join(',')") {
            Ok(Value::Str(s)) => s.as_wtf8().to_utf8_lossy().into_owned(),
            other => panic!("unexpected events value: {other:?}"),
        }
    }

    #[test]
    fn viewport_resnap_move_fires_scroll_then_scrollend() {
        // Viewport analogue of
        // `snap_event_resnap_without_scroll_fires_changing_then_change`:
        // a relayout that removes the viewport's settled snap target
        // re-snaps the viewport (600 -> 0) and must fire
        // `scrollsnapchanging`, `scrollsnapchange`, `scroll`, then
        // `scrollend` — never a bare `scrollend` with no preceding
        // `scroll` for the move.
        let mut tab = viewport_snap_tab(
            "html { scroll-snap-type: y mandatory; } \
             #a { height: 600px; scroll-snap-align: start; } \
             #b { height: 600px; scroll-snap-align: start; } \
             #tail { height: 400px; }",
        );
        // Settle on #b at 600 and end the scroll, draining its events.
        tab.scroll_tree.scroll_chain(0, 0.0, 600.0);
        let moved = tab
            .scroll_tree
            .advance_smooth(Instant::now() + Duration::from_secs(10));
        tab.scroll_tree.mark_scrolled(&moved);
        tab.scroll_tree.mark_scrolled(&[None]);
        assert!(tab.scroll_tree.end_tick().is_empty());
        assert_eq!(tab.scroll_tree.end_tick(), vec![None]);
        let _ = tab.scroll_tree.take_snap_events();
        assert_eq!(tab.scroll().offset(), (0.0, 600.0));
        // Relayout without #b's snap area: resnap falls back to #a at 0.
        viewport_relayout(
            &mut tab,
            "html { scroll-snap-type: y mandatory; } \
             #a { height: 600px; scroll-snap-align: start; } \
             #b { height: 600px; } \
             #tail { height: 400px; }",
        );
        assert_eq!(
            tab.scroll().offset(),
            (0.0, 0.0),
            "resnap follows the surviving target"
        );

        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let stub = StubDispatch::default();
        let log = stub.log.clone();
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, Some(Box::new(stub)));

        app.pump_scripts();
        // Top deliveries: the layout-snap pair for the viewport.
        assert_eq!(
            log.borrow().order,
            vec![
                "scrollsnapchanging".to_string(),
                "scrollsnapchange".to_string(),
            ]
        );
        // The fold reported the resnap move: the page saw `scroll`,
        // and `scrollend` waits for a quiet tick.
        assert_eq!(page_events(&mut app, tab_id), "scroll");

        app.pump_scripts();
        assert_eq!(
            page_events(&mut app, tab_id),
            "scroll,scrollend",
            "scrollend follows its scroll, never bare"
        );
        assert_eq!(
            log.borrow().order.len(),
            2,
            "no duplicate snap events on the quiet tick"
        );
    }

    #[test]
    fn viewport_resnap_pair_unchanged_fires_scroll_without_duplicate_change() {
        // The resnap moves the viewport (0 -> 500) but the settled
        // target (#a) is unchanged: the pump must fire exactly `scroll`
        // then `scrollend`, with no `scrollsnapchange` at all.
        let mut tab = viewport_snap_tab(
            "html { scroll-snap-type: y mandatory; } \
             #sp { height: 0px; } \
             #a { height: 600px; scroll-snap-align: start; } \
             #tail { height: 400px; }",
        );
        // The spacer grows 500px: #a's snap position moves to 500 and
        // the resnap follows it without changing the pair.
        viewport_relayout(
            &mut tab,
            "html { scroll-snap-type: y mandatory; } \
             #sp { height: 500px; } \
             #a { height: 600px; scroll-snap-align: start; } \
             #tail { height: 400px; }",
        );
        assert_eq!(tab.scroll().offset(), (0.0, 500.0));

        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let stub = StubDispatch::default();
        let log = stub.log.clone();
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, Some(Box::new(stub)));

        app.pump_scripts();
        assert!(
            log.borrow().order.is_empty(),
            "pair unchanged: no snap events, got {:?}",
            log.borrow().order
        );
        assert_eq!(page_events(&mut app, tab_id), "scroll");

        app.pump_scripts();
        assert_eq!(page_events(&mut app, tab_id), "scroll,scrollend");
        assert!(
            log.borrow().order.is_empty(),
            "no duplicate scrollsnapchange on the quiet tick"
        );
    }

    #[test]
    fn viewport_install_time_seating_fires_no_scroll() {
        // Initial layout seats the viewport on a snap target away from
        // the origin (0 -> 500) via `scroll-initial-target`: an
        // install-time move is initial layout, not a scroll — the pump
        // must fire no `scroll` and no `scrollend`.
        let tab = viewport_snap_tab(
            "html { scroll-snap-type: y mandatory; } \
             #sp { height: 500px; } \
             #a { height: 600px; scroll-snap-align: start; scroll-initial-target: nearest; } \
             #tail { height: 400px; }",
        );
        assert_eq!(
            tab.scroll().offset(),
            (0.0, 500.0),
            "install seats the viewport on #a"
        );

        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let stub = StubDispatch::default();
        let log = stub.log.clone();
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, Some(Box::new(stub)));

        app.pump_scripts();
        app.pump_scripts();
        assert_eq!(
            page_events(&mut app, tab_id),
            "",
            "install-time seating fires no scroll"
        );
        assert!(
            log.borrow().order.is_empty(),
            "install-time snap deliveries were drained"
        );
    }

    /// Build a display list from the tab's paint root, with dummy
    /// font/image resolvers (tests use no real text shaping or
    /// images): the post-first-frame state the windowed pump sees, so
    /// the script fold takes the `try_fast_fold` branch instead of the
    /// `sync_paint_offsets` fallback.
    fn viewport_display_list(tab: &Tab) -> DisplayList {
        use bowser_paint::display_list::{FontKey, build_display_list};
        let doc = tab.document_ref().expect("installed document");
        let paint = tab.paint_root.as_ref().expect("paint root");
        build_display_list(
            paint,
            &doc,
            &mut |_| FontKey(0),
            &mut |_| None,
            Size { w: 800.0, h: 600.0 },
        )
    }

    #[test]
    fn pump_scripts_viewport_fold_with_display_list_fires_scroll() {
        // Post-first-frame production shape: a viewport-only page WITH
        // a display list. An instant script-driven viewport scroll must
        // fold through `try_fast_fold` (not the pre-first-layout
        // fallback) and fire `scroll`. The fold used to return
        // `Some([])` here — no `scroll`, no tick stamp, no `scrollend`,
        // and the stale baseline was never re-armed.
        let mut tab = viewport_snap_tab(
            "#sp { height: 0px; } #a { height: 600px; } #b { height: 600px; } #tail { height: 400px; }",
        );
        assert!(tab.scroll_tree.is_viewport_only(), "viewport-only page");
        tab.display_list = Some(viewport_display_list(&tab));

        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let stub = StubDispatch::default();
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, Some(Box::new(stub)));

        // Simulate the bindings' `window.scrollTo(0, 200)` write: the
        // shared cell mutates behind the tab's back.
        app.tabs
            .get(tab_id)
            .expect("tab")
            .scroll()
            .set_offset(0.0, 200.0);
        app.pump_scripts();
        assert_eq!(
            page_events(&mut app, tab_id),
            "scroll",
            "script viewport scroll folds and fires scroll"
        );

        // Quiet tick: `scrollend` follows its `scroll`, never bare.
        app.pump_scripts();
        assert_eq!(page_events(&mut app, tab_id), "scroll,scrollend");
    }

    #[test]
    fn wheel_viewport_scroll_then_pump_fold_reports_exactly_once() {
        // Direct-path pin: a wheel-style viewport scroll moves through
        // `sync_nested_paint` (which re-arms the fold baselines), so the
        // next pump fold must be a no-op — the page sees exactly one
        // `scroll`, never a second from the fold.
        let mut tab = viewport_snap_tab(
            "#sp { height: 0px; } #a { height: 600px; } #b { height: 600px; } #tail { height: 400px; }",
        );
        tab.display_list = Some(viewport_display_list(&tab));
        let scrolled = apply_wheel(&mut tab, 400.0, 300.0, 0.0, 200.0);
        assert_eq!(scrolled, vec![None]);
        assert_eq!(
            tab.render
                .scroll_tree
                .try_fast_fold(tab.display_list.as_mut().expect("display list")),
            Some(Vec::new()),
            "direct path re-armed the fold: no double report"
        );

        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let stub = StubDispatch::default();
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, Some(Box::new(stub)));
        // Fire the wheel's `scroll` through the direct-path dispatch.
        app.fire_scroll_listeners_for_direct(tab_id, &[None]);
        assert_eq!(page_events(&mut app, tab_id), "scroll");
        // The pump fold adds nothing: no second `scroll`.
        app.pump_scripts();
        assert_eq!(
            page_events(&mut app, tab_id),
            "scroll",
            "pump fold must not re-report the wheel scroll"
        );
        // Quiet tick ends it: exactly one `scroll` then `scrollend`.
        app.pump_scripts();
        assert_eq!(page_events(&mut app, tab_id), "scroll,scrollend");
    }

    #[test]
    fn click_with_dispatcher_prevented_skips_navigation() {
        // With a dispatcher installed, a click on a link goes to page
        // scripts; when the page prevents the default action, no
        // navigation load spawns.
        let (doc, _) = Document::parse_html(b"<a href=\"https://example.com/\">x</a>");
        let anchor = doc.get_elements_by_tag_name("a").item(0).expect("anchor");
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let stub = StubDispatch {
            prevent: true,
            ..Default::default()
        };
        let log = stub.log.clone();
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, Some(Box::new(stub)));

        app.on_click(anchor);

        assert_eq!(log.borrow().clicks, vec![anchor]);
        assert!(
            app.pending.is_none(),
            "prevented default: no navigation load may spawn"
        );
    }

    #[test]
    fn click_with_dispatcher_not_prevented_navigates() {
        // The dispatcher ran, but the page did not prevent the default
        // action: the click falls back to link navigation. The href
        // points at a refused localhost port so the spawned worker fails
        // fast without touching the network.
        let (doc, _) = Document::parse_html(b"<a href=\"http://127.0.0.1:9/\">x</a>");
        let anchor = doc.get_elements_by_tag_name("a").item(0).expect("anchor");
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let stub = StubDispatch::default();
        let log = stub.log.clone();
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, Some(Box::new(stub)));

        app.on_click(anchor);

        assert_eq!(log.borrow().clicks, vec![anchor]);
        assert!(
            app.pending.is_some(),
            "unprevented click must fall back to navigation"
        );
        drop(app);
    }

    #[test]
    fn click_without_dispatcher_navigates_links() {
        // The `run_windowed` default (no dispatcher): a click on a link
        // falls back to navigation. The href points at a refused
        // localhost port so the spawned worker fails fast without
        // touching the network.
        let (doc, _) = Document::parse_html(b"<a href=\"http://127.0.0.1:9/\">x</a>");
        let anchor = doc.get_elements_by_tag_name("a").item(0).expect("anchor");
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, tab_id);

        app.on_click(anchor);

        assert!(
            app.pending.is_some(),
            "without a dispatcher the click must spawn a navigation load"
        );
        drop(app);
    }

    #[test]
    fn key_with_dispatcher_reaches_page_scripts() {
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(test_tab());
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let stub = StubDispatch::default();
        let log = stub.log.clone();
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, Some(Box::new(stub)));

        app.on_key(&Key::Character("a".into()), true);
        assert_eq!(log.borrow().keys, vec!["a".to_string()]);

        // Key release is not dispatched.
        app.on_key(&Key::Character("a".into()), false);
        assert_eq!(log.borrow().keys.len(), 1);
    }

    #[test]
    fn wheel_with_dispatcher_prevented_skips_scroll() {
        // A `wheel` listener that prevents the default action cancels the
        // scroll: page scripts run before the scroll chain, and a
        // prevented gesture moves nothing and fires no `scroll` events.
        let (tab, _, second) = tab_with_boxes();
        // Fake the content bounds `refresh` would publish for a scrolled
        // document: the max scroll offset is (0, 200).
        tab.scroll().set_max_offset((0.0, 200.0));
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let stub = StubDispatch {
            prevent: true,
            ..Default::default()
        };
        let log = stub.log.clone();
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, Some(Box::new(stub)));
        // Cursor over `second`'s box: the wheel targets that node.
        app.input.cursor = (400.0, 120.0);

        // Three lines down, `deltaMode` 1: would scroll 120px unprevented.
        app.on_wheel(0.0, 3.0, 1, TouchPhase::Moved);

        assert_eq!(log.borrow().wheels, vec![(second, 0.0, 3.0, 1)]);
        assert!(
            log.borrow().scrolls.is_empty(),
            "prevented wheel moves nothing, so no scroll events fire"
        );
        let tab = tabs.get(tab_id).expect("tab");
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
    }

    #[test]
    fn wheel_with_dispatcher_not_prevented_scrolls() {
        // Without prevention the wheel keeps its old behavior: the event
        // reaches page scripts first, then the scroll chain runs — lines
        // at 40 CSS px each.
        let (tab, _, second) = tab_with_boxes();
        tab.scroll().set_max_offset((0.0, 200.0));
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let stub = StubDispatch::default();
        let log = stub.log.clone();
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, Some(Box::new(stub)));
        app.input.cursor = (400.0, 120.0);

        app.on_wheel(0.0, 3.0, 1, TouchPhase::Moved);

        assert_eq!(log.borrow().wheels, vec![(second, 0.0, 3.0, 1)]);
        let tab = tabs.get(tab_id).expect("tab");
        assert_eq!(tab.scroll().offset(), (0.0, 120.0));
    }

    #[test]
    fn wheel_pixel_deltas_report_delta_mode_zero() {
        // `PixelDelta`-style gestures dispatch with `deltaMode` 0 and
        // scroll the raw CSS px delta (no 40px line scaling).
        let (tab, _, second) = tab_with_boxes();
        tab.scroll().set_max_offset((0.0, 200.0));
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let stub = StubDispatch::default();
        let log = stub.log.clone();
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, Some(Box::new(stub)));
        app.input.cursor = (400.0, 120.0);

        app.on_wheel(0.0, 120.0, 0, TouchPhase::Moved);

        assert_eq!(log.borrow().wheels, vec![(second, 0.0, 120.0, 0)]);
        let tab = tabs.get(tab_id).expect("tab");
        assert_eq!(tab.scroll().offset(), (0.0, 120.0));
    }

    /// Tab with two nested vertical scrollers (`#outer` 200px, `#inner`
    /// 100px, both with overflowing content) plus a tall `#pad` so the
    /// document itself scrolls. Returns the tab, the `outer`/`inner`
    /// nodes, a non-container content node (`tall`), and the two
    /// container indices.
    fn tab_with_nested_scrollers() -> (Tab, NodeId, NodeId, NodeId, usize, usize) {
        use bowser_css::parse_stylesheet;
        use bowser_layout::{FixedFontMetrics, layout};
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"outer\"><div id=\"inner\"><div id=\"tall\">x</div></div><div id=\"tall2\">y</div></div><div id=\"pad\">z</div></body></html>",
        );
        let outer = doc.get_element_by_id("outer").expect("outer");
        let inner = doc.get_element_by_id("inner").expect("inner");
        let tall = doc.get_element_by_id("tall").expect("tall");
        let sheet = parse_stylesheet(
            "#outer { overflow-y: auto; height: 200px; width: 300px; } \
             #inner { overflow-y: auto; height: 100px; width: 200px; } \
             #tall { height: 400px; } #tall2 { height: 300px; } \
             #pad { height: 2000px; }",
        );
        let root = layout(
            &doc,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let mut tab = test_tab();
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        let iouter = tab.scroll_tree.index_of(outer).expect("outer container");
        let iinner = tab.scroll_tree.index_of(inner).expect("inner container");
        assert_ne!(iouter, iinner);
        (tab, outer, inner, tall, iouter, iinner)
    }

    /// One page is the target container's own scrollport height: a
    /// PageDown inside the 100px `#inner` moves 100px, not a 600px
    /// viewport page.
    #[test]
    fn key_page_down_uses_nested_scrollport_not_viewport() {
        let (mut tab, _outer, inner, _tall, _iouter, iinner) = tab_with_nested_scrollers();
        assert_eq!(
            tab.scroll_tree.container(iinner).scrollport.h,
            100.0,
            "fixture: inner scrollport is 100px"
        );
        let scrolled = apply_key_scroll_with_shift(&mut tab, Some(inner), "PageDown", false);
        assert_eq!(scrolled, vec![Some(inner)]);
        assert_eq!(
            tab.scroll_tree.container(iinner).state.offset(),
            (0.0, 100.0),
            "one page = the container's own scrollport, not the viewport"
        );
        assert_eq!(
            tab.scroll().offset(),
            (0.0, 0.0),
            "the inner consumed the whole page: no chaining"
        );
    }

    /// Shift+Space pages up by the container's page; plain Space pages
    /// down. The plain [`apply_key_scroll`] entry keeps the unshifted
    /// behavior for the synthetic testdriver path.
    #[test]
    fn shift_space_pages_up() {
        let (mut tab, _outer, inner, _tall, _iouter, iinner) = tab_with_nested_scrollers();
        let offset = |tab: &Tab| tab.scroll_tree.container(iinner).state.offset();
        // Plain Space pages down one container page.
        let scrolled = apply_key_scroll_with_shift(&mut tab, Some(inner), " ", false);
        assert_eq!(scrolled, vec![Some(inner)]);
        assert_eq!(offset(&tab), (0.0, 100.0));
        // Shift+Space pages back up.
        let scrolled = apply_key_scroll_with_shift(&mut tab, Some(inner), " ", true);
        assert_eq!(scrolled, vec![Some(inner)]);
        assert_eq!(offset(&tab), (0.0, 0.0));
        // The unshifted entry point is unchanged.
        apply_key_scroll_with_shift(&mut tab, Some(inner), " ", false);
        assert_eq!(offset(&tab), (0.0, 100.0));
    }

    /// `deltaMode: 2` (pages) scales by the latched container's own
    /// scrollport: one page unit over `#inner` scrolls 100px.
    #[test]
    fn wheel_page_delta_uses_nested_scrollport() {
        let (tab, _outer, _inner, _tall, _iouter, iinner) = tab_with_nested_scrollers();
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, None);
        // Cursor over #inner (10, 10).
        app.input.cursor = (10.0, 10.0);
        app.on_wheel(0.0, 1.0, 2, TouchPhase::Moved);
        let tab = tabs.get(tab_id).expect("tab");
        assert_eq!(
            tab.scroll_tree.container(iinner).state.offset(),
            (0.0, 100.0),
            "one page unit = the 100px nested scrollport, not the 600px viewport"
        );
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
    }

    /// ...while a page unit over the page itself is the viewport
    /// height.
    #[test]
    fn wheel_page_delta_uses_viewport_height_on_page() {
        let (tab, _outer, _inner, _tall, _iouter, _iinner) = tab_with_nested_scrollers();
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, None);
        // Cursor at (700, 590): outside #outer (300px wide), over the page.
        app.input.cursor = (700.0, 590.0);
        app.on_wheel(0.0, 1.0, 2, TouchPhase::Moved);
        let tab = tabs.get(tab_id).expect("tab");
        assert_eq!(
            tab.scroll().offset(),
            (0.0, 600.0),
            "one page unit on the viewport = the 600px viewport height"
        );
    }

    /// A wheel gesture latches the scroller resolved at its first
    /// event: later events in the same gesture keep that scroller even
    /// when the cursor moves across a scroller boundary.
    #[test]
    fn wheel_gesture_latches_to_start_scroller() {
        let (tab, _outer, _inner, _tall, iouter, iinner) = tab_with_nested_scrollers();
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, None);
        let t = Instant::now();
        // First event over #inner latches it and reports its page unit.
        let (start, page) = app
            .wheel_gesture_start(10.0, 10.0, TouchPhase::Moved, t)
            .expect("gesture");
        assert_eq!(start, iinner);
        assert_eq!(
            page, 100.0,
            "page unit is the latched container's scrollport"
        );
        // Cursor moves over #outer (below #inner), still within the
        // gesture: the latch keeps the start scroller.
        let (start, _) = app
            .wheel_gesture_start(
                10.0,
                150.0,
                TouchPhase::Moved,
                t + Duration::from_millis(100),
            )
            .expect("gesture");
        assert_eq!(
            start, iinner,
            "latched gesture keeps its scroller across cursor moves"
        );
        // A gap past the timeout ends the gesture: the next event
        // re-resolves from the cursor. (The previous event refreshed the
        // latch clock, so the gap is measured from it.)
        let (start, _) = app
            .wheel_gesture_start(
                10.0,
                150.0,
                TouchPhase::Moved,
                t + Duration::from_millis(700),
            )
            .expect("gesture");
        assert_eq!(
            start, iouter,
            "timed-out gesture re-resolves from the cursor"
        );
        // `Ended` closes the gesture even without a gap.
        app.wheel_gesture_start(
            10.0,
            150.0,
            TouchPhase::Moved,
            t + Duration::from_millis(750),
        );
        app.wheel_gesture_start(
            10.0,
            150.0,
            TouchPhase::Ended,
            t + Duration::from_millis(760),
        );
        assert!(app.wheel_latch.is_none(), "Ended clears the latch");
        // `Started` always begins a fresh gesture.
        let (start, _) = app
            .wheel_gesture_start(
                10.0,
                10.0,
                TouchPhase::Started,
                t + Duration::from_millis(770),
            )
            .expect("gesture");
        assert_eq!(start, iinner);
    }

    /// A cancelled OS wheel gesture clears the latch: the event
    /// scrolls nothing, and the next unrelated wheel flick within the
    /// timeout window latches fresh from the cursor instead of chaining
    /// from the dead scroller.
    #[test]
    fn wheel_gesture_cancelled_clears_latch() {
        let (tab, _outer, _inner, _tall, iouter, iinner) = tab_with_nested_scrollers();
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, None);
        let t = Instant::now();
        // First event over #inner latches it.
        let (start, _) = app
            .wheel_gesture_start(10.0, 10.0, TouchPhase::Moved, t)
            .expect("gesture");
        assert_eq!(start, iinner);
        // The OS cancels the gesture: the latch dies, the event scrolls
        // nothing.
        let out = app.wheel_gesture_start(
            10.0,
            10.0,
            TouchPhase::Cancelled,
            t + Duration::from_millis(16),
        );
        assert!(out.is_none(), "cancelled gesture scrolls nothing");
        assert!(app.wheel_latch.is_none(), "Cancelled clears the latch");
        // The next wheel flick — well within the old gesture's 500 ms
        // window — latches fresh from the cursor instead of chaining
        // from the dead scroller.
        let (start, _) = app
            .wheel_gesture_start(
                10.0,
                150.0,
                TouchPhase::Moved,
                t + Duration::from_millis(100),
            )
            .expect("gesture");
        assert_eq!(
            start, iouter,
            "flick after cancel re-latches from the cursor"
        );
    }

    /// An `Ended` with no live latch — no gesture, or an expired one —
    /// scrolls nothing instead of taking the fresh path and emitting
    /// one phantom delta.
    #[test]
    fn wheel_gesture_ended_without_live_latch_scrolls_nothing() {
        let (tab, _outer, _inner, _tall, _iouter, iinner) = tab_with_nested_scrollers();
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, None);
        let t = Instant::now();
        // No gesture ever started: a stray `Ended` scrolls nothing and
        // latches nothing.
        let out = app.wheel_gesture_start(10.0, 10.0, TouchPhase::Ended, t);
        assert!(out.is_none(), "stray Ended scrolls no phantom delta");
        assert!(app.wheel_latch.is_none(), "stray Ended latches nothing");
        // Same after the latch expired: `Ended` closes the dead gesture
        // without scrolling.
        let (start, _) = app
            .wheel_gesture_start(10.0, 10.0, TouchPhase::Moved, t)
            .expect("gesture");
        assert_eq!(start, iinner);
        let out = app.wheel_gesture_start(
            10.0,
            150.0,
            TouchPhase::Ended,
            t + Duration::from_millis(600),
        );
        assert!(out.is_none(), "Ended on an expired latch scrolls nothing");
        assert!(app.wheel_latch.is_none(), "Ended clears the dead latch");
        // End to end: a stray `Ended` through `on_wheel` moves nothing.
        app.input.cursor = (10.0, 10.0);
        app.on_wheel(0.0, 50.0, 0, TouchPhase::Ended);
        let tab = tabs.get(tab_id).expect("tab");
        assert_eq!(
            tab.scroll_tree.container(iinner).state.offset(),
            (0.0, 0.0),
            "stray Ended through on_wheel scrolls nothing"
        );
    }

    /// A latched wheel container that vanished mid-gesture ends the
    /// gesture: the event scrolls nothing and the latch clears, so the
    /// next event re-latches fresh instead of driving a recycled tree
    /// index.
    #[test]
    fn wheel_latch_vanished_container_ends_gesture() {
        let (tab, _outer, _inner, tall, _iouter, iinner) = tab_with_nested_scrollers();
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, None);
        let t = Instant::now();
        let (start, _) = app
            .wheel_gesture_start(10.0, 10.0, TouchPhase::Moved, t)
            .expect("gesture");
        assert_eq!(start, iinner);
        // Simulate a relayout dropping the container: re-point the latch
        // at a node that is not (and never was) a scroll container.
        app.wheel_latch.as_mut().expect("latch").node = Some(tall);
        let out =
            app.wheel_gesture_start(10.0, 10.0, TouchPhase::Moved, t + Duration::from_millis(16));
        assert!(out.is_none(), "vanished latch target scrolls nothing");
        assert!(
            app.wheel_latch.is_none(),
            "vanished latch target ends the gesture"
        );
    }

    /// A wheel gesture over the whole page scrolls through `on_wheel`
    /// with the latch held: a second event still targets the latched
    /// scroller after the cursor moves away.
    #[test]
    fn wheel_on_wheel_keeps_latched_scroller_across_cursor_move() {
        let (tab, _outer, _inner, _tall, _iouter, iinner) = tab_with_nested_scrollers();
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, None);
        // First wheel over #inner latches it.
        app.input.cursor = (10.0, 10.0);
        app.on_wheel(0.0, 50.0, 0, TouchPhase::Moved);
        // The cursor drifts over #outer, but the gesture is latched:
        // the second wheel still drives #inner.
        app.input.cursor = (10.0, 150.0);
        app.on_wheel(0.0, 50.0, 0, TouchPhase::Moved);
        let tab = tabs.get(tab_id).expect("tab");
        assert_eq!(
            tab.scroll_tree.container(iinner).state.offset(),
            (0.0, 100.0),
            "both wheels drove the latched #inner"
        );
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
    }

    /// `resolve_latched_node`: the viewport latch is container 0, a
    /// container node resolves to its index, anything else is gone.
    #[test]
    fn resolve_latched_node_maps_viewport_and_containers() {
        let (tab, _outer, inner, tall, _iouter, iinner) = tab_with_nested_scrollers();
        assert_eq!(resolve_latched_node(&tab.scroll_tree, None), Some(0));
        assert_eq!(
            resolve_latched_node(&tab.scroll_tree, Some(inner)),
            Some(iinner)
        );
        assert_eq!(resolve_latched_node(&tab.scroll_tree, Some(tall)), None);
    }

    /// Middle-click autoscroll latches the scroller under the press
    /// point for the gesture's lifetime: motion scrolls it even as the
    /// cursor moves.
    #[test]
    fn autoscroll_latches_scroller_at_press() {
        let (tab, _outer, inner, _tall, _iouter, iinner) = tab_with_nested_scrollers();
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, None);
        // Middle press over #inner latches its node.
        app.on_middle_button(true, 10.0, 10.0);
        assert_eq!(app.autoscroll.expect("gesture").node, Some(inner));
        // Motion beyond the dead zone scrolls the latched container.
        app.on_cursor_moved(10.0, 200.0);
        let latched_node = app.autoscroll.expect("gesture").node;
        // Release ends it.
        app.on_middle_button(false, 10.0, 200.0);
        assert!(app.autoscroll.is_none());
        // (Last use of `app` above, so the `tabs` borrow is free now.)
        let tab = tabs.get(tab_id).expect("tab");
        let scrolled_y = tab.scroll_tree.container(iinner).state.offset().1;
        assert!(
            scrolled_y > 0.0,
            "latched container scrolled, y={scrolled_y}"
        );
        assert_eq!(
            latched_node,
            Some(inner),
            "the gesture stays latched across motion"
        );
    }

    /// An autoscroll whose latched container vanished mid-gesture ends
    /// the gesture instead of scrolling a recycled tree index.
    #[test]
    fn autoscroll_vanished_container_ends_gesture() {
        let (tab, _outer, _inner, tall, _iouter, iinner) = tab_with_nested_scrollers();
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app_with_js(&mut tabs, &jar, tab_id, None);
        app.on_middle_button(true, 10.0, 10.0);
        assert!(app.autoscroll.is_some());
        // Simulate a relayout dropping the container: re-point the latch
        // at a node that is not a scroll container.
        app.autoscroll.as_mut().expect("gesture").node = Some(tall);
        app.on_cursor_moved(10.0, 200.0);
        assert!(
            app.autoscroll.is_none(),
            "vanished container ends the gesture"
        );
        let tab = tabs.get(tab_id).expect("tab");
        assert_eq!(
            tab.scroll_tree.container(iinner).state.offset(),
            (0.0, 0.0),
            "nothing scrolled"
        );
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
    }

    #[test]
    fn load_control_generation_and_shutdown() {
        let control = LoadControl::new();
        let g1 = control.next_generation();
        let g2 = control.next_generation();
        assert_ne!(g1, g2);
        assert!(control.is_live(g2));
        assert!(!control.is_live(g1), "superseded generation is not live");
        control
            .shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);
        assert!(!control.is_live(g2), "shutdown retires every generation");
    }

    #[test]
    fn stale_drain_forwards_jar_to_newest_handoff() {
        use bowser_net::cookies::RequestContext;

        let mut tabs = TabArena::default();
        let tab = tabs.insert(Tab::new(Size { w: 800.0, h: 600.0 }));
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, tab);

        // Simulate a superseded worker's jar carrying a cookie.
        let mut carried = CookieJar::new();
        let ctx = RequestContext {
            host: "example.com",
            path: "/",
            secure: false,
            http_api: true,
            top_host: Some("example.com"),
            top_secure: Some(false),
            top_level_safe_navigation: true,
        };
        carried.set_cookie("session=abc; Path=/", &ctx, 1_700_000_000);
        assert_eq!(carried.len(), 1);

        let (stale_tx, stale_rx) = mpsc::channel();
        let (handoff_tx, handoff_rx) = mpsc::channel();
        app.handoff_tx = Some(handoff_tx);
        app.stale.push(StaleLoad { rx: stale_rx });
        stale_tx
            .send(LoadReport {
                result: None,
                jar: carried,
                handoff_missed: false,
            })
            .unwrap();

        app.poll_pending();

        assert!(app.stale.is_empty(), "drained stale load stays queued");
        let forwarded = handoff_rx.try_recv().expect("jar forwarded to handoff");
        assert_eq!(forwarded.len(), 1, "cookies survive the supersede");
    }

    /// Pump `poll_pending` until `condition` holds or the deadline fires.
    /// Only used by the `wpt-driver`-gated async navigation tests below.
    #[cfg(feature = "wpt-driver")]
    fn pump_until(
        app: &mut WindowApp<'_>,
        deadline: Instant,
        condition: impl Fn(&WindowApp<'_>) -> bool,
    ) {
        while !condition(app) {
            app.poll_pending();
            assert!(
                Instant::now() < deadline,
                "timed out waiting for background load"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[cfg(feature = "wpt-driver")]
    #[test]
    fn superseding_load_installs_latest() {
        use crate::wpt_driver::FixtureServer;

        let root = std::env::temp_dir().join(format!("bowser-window-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("a.html"),
            "<html><head><title>A</title></head><body>a</body></html>",
        )
        .unwrap();
        std::fs::write(
            root.join("b.html"),
            "<html><head><title>B</title></head><body>b</body></html>",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let mut tabs = TabArena::default();
        let tab = tabs.insert(Tab::new(Size { w: 800.0, h: 600.0 }));
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, tab);
        let url_a = bowser_url::parse(&format!("{base}/a.html"), None).unwrap();
        let url_b = bowser_url::parse(&format!("{base}/b.html"), None).unwrap();

        app.spawn_load(&url_a, false, None);
        // Supersede before A can finish: only B may install.
        app.spawn_load(&url_b, true, None);

        let deadline = Instant::now() + Duration::from_secs(15);
        pump_until(&mut app, deadline, |app| {
            app.tabs.get(tab).is_some_and(|t| t.title == "B")
        });

        assert!(app.pending.is_none());
        assert!(app.stale.is_empty(), "stale worker was drained");
        // The profile jar is back in place and usable after the swap.
        let _guard = jar.lock().expect("jar lock usable after loads");
        drop(app);
        shutdown.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn close_does_not_wait_for_stalled_load() {
        // Listener that accepts and never responds: the worker stalls in
        // its socket read (60s timeout) instead of failing fast.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                let mut buf = [0u8; 4096];
                let _ = std::io::Read::read(&mut stream, &mut buf);
                std::thread::sleep(Duration::from_secs(300));
            }
        });

        let mut tabs = TabArena::default();
        let tab = tabs.insert(Tab::new(Size { w: 800.0, h: 600.0 }));
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, tab);
        let url = bowser_url::parse(&format!("http://127.0.0.1:{port}/stall.html"), None).unwrap();
        app.spawn_load(&url, false, None);
        // Let the worker reach the stalled read.
        std::thread::sleep(Duration::from_millis(500));

        let start = Instant::now();
        app.shutdown();
        app.poll_pending();
        let elapsed = start.elapsed();

        assert!(app.pending.is_none());
        assert!(
            elapsed < Duration::from_secs(5),
            "close blocked {elapsed:?} on a stalled load"
        );
        drop(app);
        // NOTE: the detached worker is still stuck in its read timeout;
        // it owns everything it touches, so dropping the app is safe and
        // the process exit reaps the thread.
    }

    /// Request context for the `example.com` cookie tests below.
    fn test_cookie_ctx() -> bowser_net::cookies::RequestContext<'static> {
        bowser_net::cookies::RequestContext {
            host: "example.com",
            path: "/",
            secure: false,
            http_api: true,
            top_host: Some("example.com"),
            top_secure: Some(false),
            top_level_safe_navigation: true,
        }
    }

    /// Build a jar holding one cookie for `example.com`.
    fn jar_with_cookie(cookie: &str) -> CookieJar {
        let mut jar = CookieJar::new();
        jar.set_cookie(cookie, &test_cookie_ctx(), 1_700_000_000);
        assert_eq!(jar.len(), 1);
        jar
    }

    fn jar_has_cookie(jar: &mut CookieJar, name: &str) -> bool {
        jar.cookie_header(&test_cookie_ctx(), 1_700_000_000)
            .is_some_and(|header| header.split("; ").any(|c| c.starts_with(name)))
    }

    #[test]
    fn shutdown_restores_newest_buffered_jar() {
        let mut tabs = TabArena::default();
        let tab = tabs.insert(Tab::new(Size { w: 800.0, h: 600.0 }));
        // The profile jar holds a real cookie before the load starts.
        let jar = Arc::new(Mutex::new(jar_with_cookie("session=real; Path=/")));
        let mut app = test_app(&mut tabs, &jar, tab);

        // Exactly what `spawn_load` does with no load in flight: swap the
        // profile jar out, leaving a fresh jar behind. The worker owns the
        // only copy of the real cookies from here on.
        let mut worker_jar = crate::navigation::take_profile_jar(&jar);
        assert!(jar.lock().unwrap().is_empty(), "swap left a fresh jar");
        // The load sets a cookie before the window closes.
        worker_jar.set_cookie("fresh=1; Path=/", &test_cookie_ctx(), 1_700_000_000);

        // The worker finished just before close: its report is buffered in
        // the channel, never polled — close beat the next event-loop turn.
        // `result: None` stands in for any finished report; shutdown only
        // looks at the jar.
        let (tx, rx) = mpsc::channel();
        app.pending = Some(PendingLoad {
            rx,
            push_history: false,
            traverse: None,
        });
        tx.send(LoadReport {
            result: None,
            jar: worker_jar,
            handoff_missed: false,
        })
        .unwrap();
        drop(tx);

        app.shutdown();

        let mut guard = jar.lock().expect("jar lock usable after shutdown");
        assert!(
            jar_has_cookie(&mut guard, "session="),
            "shutdown must restore the pre-existing cookie"
        );
        assert!(
            jar_has_cookie(&mut guard, "fresh="),
            "shutdown must keep the cookie the load set"
        );
    }

    #[test]
    fn shutdown_waits_for_inflight_load_jar() {
        let mut tabs = TabArena::default();
        let tab = tabs.insert(Tab::new(Size { w: 800.0, h: 600.0 }));
        // The profile jar holds a real cookie before the load starts.
        let jar = Arc::new(Mutex::new(jar_with_cookie("session=real; Path=/")));
        let mut app = test_app(&mut tabs, &jar, tab);

        // Exactly what `spawn_load` does with no load in flight: swap the
        // profile jar out, leaving a fresh jar behind. The worker owns the
        // only copy of the real cookies from here on.
        let mut worker_jar = crate::navigation::take_profile_jar(&jar);
        assert!(jar.lock().unwrap().is_empty(), "swap left a fresh jar");
        // The load sets a cookie while the window is closing.
        worker_jar.set_cookie("fresh=1; Path=/", &test_cookie_ctx(), 1_700_000_000);

        // The worker is still in flight when the window closes: nothing
        // is buffered yet, and it reports just after `shutdown` starts
        // draining — the way a real load completing around the close
        // moment would. A non-blocking drain would miss it and leave the
        // profile holding the empty swap jar.
        let (tx, rx) = mpsc::channel();
        app.pending = Some(PendingLoad {
            rx,
            push_history: false,
            traverse: None,
        });
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            tx.send(LoadReport {
                result: None,
                jar: worker_jar,
                handoff_missed: false,
            })
            .expect("shutdown still draining");
        });

        app.shutdown();

        let mut guard = jar.lock().expect("jar lock usable after shutdown");
        assert!(
            jar_has_cookie(&mut guard, "session="),
            "shutdown must restore the pre-existing cookie, not the empty swap jar"
        );
        assert!(
            jar_has_cookie(&mut guard, "fresh="),
            "shutdown must keep the cookie the in-flight load set"
        );
        assert!(app.pending.is_none());
    }

    #[test]
    fn shutdown_prefers_pending_jar_over_stale() {
        let mut tabs = TabArena::default();
        let tab = tabs.insert(Tab::new(Size { w: 800.0, h: 600.0 }));
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, tab);

        // A superseded worker's buffered report (older cookies)...
        let (stale_tx, stale_rx) = mpsc::channel();
        app.stale.push(StaleLoad { rx: stale_rx });
        stale_tx
            .send(LoadReport {
                result: None,
                jar: jar_with_cookie("old=1; Path=/"),
                handoff_missed: false,
            })
            .unwrap();
        drop(stale_tx);
        // ...and the pending worker's buffered report (newer cookies).
        let (tx, rx) = mpsc::channel();
        app.pending = Some(PendingLoad {
            rx,
            push_history: false,
            traverse: None,
        });
        tx.send(LoadReport {
            result: None,
            jar: jar_with_cookie("new=2; Path=/"),
            handoff_missed: false,
        })
        .unwrap();
        drop(tx);

        app.shutdown();

        let mut guard = jar.lock().expect("jar lock usable after shutdown");
        assert!(
            jar_has_cookie(&mut guard, "new="),
            "the pending (newest) jar must win the restore"
        );
        assert!(
            !jar_has_cookie(&mut guard, "old="),
            "the stale jar must not overwrite the newer one"
        );
        assert!(app.stale.is_empty() && app.pending.is_none());
    }

    #[test]
    fn shutdown_empty_pending_jar_does_not_clobber_real_jar() {
        let mut tabs = TabArena::default();
        let tab = tabs.insert(Tab::new(Size { w: 800.0, h: 600.0 }));
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, tab);

        // A superseded worker's buffered report carrying the real
        // cookies...
        let (stale_tx, stale_rx) = mpsc::channel();
        app.stale.push(StaleLoad { rx: stale_rx });
        stale_tx
            .send(LoadReport {
                result: None,
                jar: jar_with_cookie("session=real; Path=/"),
                handoff_missed: false,
            })
            .unwrap();
        drop(stale_tx);
        // ...and the pending worker's buffered report holding a fresh
        // default jar: its handoff timed out, so this jar is not in the
        // cookie lineage and must not win the restore.
        let (tx, rx) = mpsc::channel();
        app.pending = Some(PendingLoad {
            rx,
            push_history: false,
            traverse: None,
        });
        tx.send(LoadReport {
            result: None,
            jar: CookieJar::new(),
            // The scenario under test: its handoff timed out.
            handoff_missed: true,
        })
        .unwrap();
        drop(tx);

        app.shutdown();

        let mut guard = jar.lock().expect("jar lock usable after shutdown");
        assert!(
            jar_has_cookie(&mut guard, "session="),
            "the real jar must survive a handoff-missed pending report"
        );
        assert!(app.stale.is_empty() && app.pending.is_none());
    }

    /// Simulate the supersede race for the handoff-timeout tests below:
    /// load A swapped the profile jar out (it owns the only copy of the
    /// real cookies), load B superseded it and registered a handoff
    /// waiter whose wait timed out (the receiver is dropped, exactly as
    /// the real worker's `recv_timeout` leaves it). Returns the app, A's
    /// report sender (carrying the real jar), and B's report sender.
    fn supersede_handoff_setup<'a>(
        tabs: &'a mut TabArena,
        jar: &Arc<Mutex<CookieJar>>,
        tab: TabId,
    ) -> (
        WindowApp<'a>,
        CookieJar,
        mpsc::Sender<LoadReport>,
        mpsc::Sender<LoadReport>,
    ) {
        let mut app = test_app(tabs, jar, tab);
        let worker_jar = crate::navigation::take_profile_jar(jar);
        assert!(jar.lock().unwrap().is_empty(), "swap left a fresh jar");
        let (stale_tx, stale_rx) = mpsc::channel();
        app.stale.push(StaleLoad { rx: stale_rx });
        let (handoff_tx, handoff_rx) = mpsc::channel();
        app.handoff_tx = Some(handoff_tx);
        // B's handoff wait timed out: the waiter is gone, so the stale
        // drain's forward fails exactly like the real race.
        drop(handoff_rx);
        let (pending_tx, pending_rx) = mpsc::channel();
        app.pending = Some(PendingLoad {
            rx: pending_rx,
            push_history: true,
            traverse: None,
        });
        (app, worker_jar, stale_tx, pending_tx)
    }

    /// The supersede race, B-reports-first order: load A starts with an
    /// authenticated jar, load B supersedes it within [`HANDOFF_TIMEOUT`],
    /// B's handoff wait times out, B reports its empty jar first, and A
    /// finishes after B. The profile jar must not revert to empty —
    /// "empty jars never win" on the normal path, not just at shutdown.
    #[test]
    fn supersede_handoff_timeout_keeps_profile_cookies() {
        let mut tabs = TabArena::default();
        let tab = tabs.insert(Tab::new(Size { w: 800.0, h: 600.0 }));
        let jar = Arc::new(Mutex::new(jar_with_cookie("session=real; Path=/")));
        let (mut app, worker_jar, stale_tx, pending_tx) =
            supersede_handoff_setup(&mut tabs, &jar, tab);

        // B reports first, with the empty jar it started from after the
        // handoff timed out.
        pending_tx
            .send(LoadReport {
                result: None,
                jar: CookieJar::new(),
                handoff_missed: true,
            })
            .unwrap();
        app.poll_pending();

        assert!(
            jar.lock().unwrap().is_empty(),
            "the empty, lineage-less jar must not become the profile jar"
        );

        // A reports afterwards, carrying the real cookies.
        stale_tx
            .send(LoadReport {
                result: None,
                jar: worker_jar,
                handoff_missed: false,
            })
            .unwrap();
        app.poll_pending();

        let mut guard = jar.lock().expect("jar lock usable after loads");
        assert!(
            jar_has_cookie(&mut guard, "session="),
            "the superseded load's cookies must survive the handoff timeout"
        );
        assert!(app.stale.is_empty() && app.pending.is_none());
    }

    /// The supersede race, A-reports-first order: the stale drain cannot
    /// forward A's jar to B's dead handoff waiter, so it restores the jar
    /// to the profile directly; B's later empty report must not clobber
    /// it.
    #[test]
    fn supersede_handoff_timeout_stale_reports_first() {
        let mut tabs = TabArena::default();
        let tab = tabs.insert(Tab::new(Size { w: 800.0, h: 600.0 }));
        let jar = Arc::new(Mutex::new(jar_with_cookie("session=real; Path=/")));
        let (mut app, worker_jar, stale_tx, pending_tx) =
            supersede_handoff_setup(&mut tabs, &jar, tab);

        // A reports before B.
        stale_tx
            .send(LoadReport {
                result: None,
                jar: worker_jar,
                handoff_missed: false,
            })
            .unwrap();
        app.poll_pending();

        {
            let mut guard = jar.lock().expect("jar lock usable after loads");
            assert!(
                jar_has_cookie(&mut guard, "session="),
                "the unclaimed stale jar must be restored to the profile"
            );
        }

        // B reports afterwards with its empty handoff-missed jar.
        pending_tx
            .send(LoadReport {
                result: None,
                jar: CookieJar::new(),
                handoff_missed: true,
            })
            .unwrap();
        app.poll_pending();

        let mut guard = jar.lock().expect("jar lock usable after loads");
        assert!(
            jar_has_cookie(&mut guard, "session="),
            "B's empty jar must not clobber the restored real jar"
        );
        assert!(app.stale.is_empty() && app.pending.is_none());
    }

    #[test]
    fn shutdown_restores_jar_when_close_beats_worker_startup() {
        let mut tabs = TabArena::default();
        let tab = tabs.insert(Tab::new(Size { w: 800.0, h: 600.0 }));
        // The profile jar holds a real cookie before the load starts.
        let jar = Arc::new(Mutex::new(jar_with_cookie("session=real; Path=/")));
        let mut app = test_app(&mut tabs, &jar, tab);

        // Spawn a load, then close before the worker thread runs: the
        // worker observes the shutdown flag at startup and must still
        // hand the swapped-out jar back instead of dropping it with the
        // thread. The URL is never fetched — every worker path bails
        // before touching the network once shutdown is set (and nothing
        // listens on port 9 in the nano-race where it slips through).
        let url = bowser_url::parse("http://127.0.0.1:9/unused", None).unwrap();
        app.spawn_load(&url, false, None);
        app.shutdown();

        let mut guard = jar.lock().expect("jar lock usable after shutdown");
        assert!(
            jar_has_cookie(&mut guard, "session="),
            "shutdown during an in-flight load must restore the real jar"
        );
        assert!(app.pending.is_none());
    }

    // -- unified load path -------------------------------------------------
    //
    // The window worker's async loads and the blocking
    // `navigation::navigate` share one implementation of load completion
    // (`navigation::complete_navigation`) and the fragment fast path
    // (`navigation::navigate_fragment`). These tests pin that the two
    // paths produce identical observable tab state.
    //
    // Needs the `wpt-driver` feature (the `FixtureServer` test harness).

    #[cfg(feature = "wpt-driver")]
    fn serve_page(
        name: &str,
        html: &str,
    ) -> (String, std::path::PathBuf, crate::wpt_driver::Shutdown) {
        use crate::wpt_driver::FixtureServer;
        let root =
            std::env::temp_dir().join(format!("bowser-window-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("page.html"), html).unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();
        (format!("{base}/page.html"), root, shutdown)
    }

    #[cfg(feature = "wpt-driver")]
    #[test]
    fn async_load_matches_sync_navigation() {
        let (url, root, shutdown) = serve_page(
            "unified",
            "<html><head><title>Unified</title><style>p{color:red}</style></head>\
             <body><p>hi</p></body></html>",
        );

        // Synchronous path: blocking `navigation::navigate`.
        let mut sync_tab = Tab::new(Size { w: 800.0, h: 600.0 });
        let sync_jar = Mutex::new(CookieJar::new());
        crate::navigation::navigate(
            &mut sync_tab,
            &sync_jar,
            &url,
            bowser_net::fetch::USER_AGENT,
            &test_fonts(),
        )
        .expect("sync navigate");

        // Asynchronous path: the window worker's spawn_load + poll_pending.
        let mut tabs = TabArena::default();
        let tab = tabs.insert(Tab::new(Size { w: 800.0, h: 600.0 }));
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, tab);
        let parsed = bowser_url::parse(&url, None).unwrap();
        app.spawn_load(&parsed, true, None);
        let deadline = Instant::now() + Duration::from_secs(15);
        pump_until(&mut app, deadline, |app| {
            app.pending.is_none() && app.tabs.get(tab).is_some_and(|t| t.has_document())
        });

        let async_tab = app.tabs.get(tab).expect("tab");
        assert_eq!(async_tab.title, sync_tab.title);
        assert_eq!(async_tab.author.len(), sync_tab.author.len());
        assert_eq!(async_tab.scroll(), sync_tab.scroll());
        assert_eq!(async_tab.dirty, sync_tab.dirty);
        assert_eq!(
            async_tab.current_url().map(|u| u.serialize()),
            sync_tab.current_url().map(|u| u.serialize()),
            "async completion pushes the same history entry",
        );
        assert_eq!(async_tab.history.len(), sync_tab.history.len());
        drop(app);
        shutdown.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(feature = "wpt-driver")]
    #[test]
    fn link_fragment_navigation_does_not_refetch() {
        let (url, root, shutdown) = serve_page(
            "frag",
            "<html><head><title>Frag</title></head><body><p>hi</p></body></html>",
        );

        let mut tabs = TabArena::default();
        let tab = tabs.insert(Tab::new(Size { w: 800.0, h: 600.0 }));
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, tab);
        let parsed = bowser_url::parse(&url, None).unwrap();
        app.spawn_load(&parsed, true, None);
        let deadline = Instant::now() + Duration::from_secs(15);
        pump_until(&mut app, deadline, |app| {
            app.pending.is_none() && app.tabs.get(tab).is_some_and(|t| t.has_document())
        });
        assert_eq!(app.tabs.get(tab).unwrap().history.len(), 1);

        // Fragment click takes the shared same-document fast path: no
        // worker load is spawned, history commits the fragment URL, and
        // the document is kept.
        app.navigate_link(&format!("{url}#frag"));
        assert!(
            app.pending.is_none(),
            "fragment click must not spawn a load"
        );
        let tab = app.tabs.get(tab).unwrap();
        assert_eq!(tab.history.len(), 2);
        assert_eq!(tab.current_url().unwrap().fragment(), Some("frag"));
        assert_eq!(tab.title, "Frag", "same document kept");
        assert!(tab.has_document());
        drop(app);
        shutdown.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Clicked hrefs resolve against the document's effective `<base
    /// href>`, not the tab's current URL: `/dir/page.html` carrying
    /// `<base href="/assets/">` and a click on `next.html` must navigate
    /// to `/assets/next.html`.
    #[test]
    fn link_click_resolves_href_against_base_element() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                let mut buf = [0u8; 4096];
                let _ = std::io::Read::read(&mut stream, &mut buf);
                let head = std::str::from_utf8(&buf).unwrap_or("");
                let target = head
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("/");
                // The fetch client sends absolute-form targets; match on
                // the path suffix.
                let body = if target.ends_with("/assets/next.html") {
                    "<html><head><title>Next</title></head><body>next</body></html>"
                } else {
                    "<html><head><title>nope</title></head><body>nope</body></html>"
                };
                let status = if target.ends_with("/assets/next.html") {
                    "200 OK"
                } else {
                    "404 Not Found"
                };
                let head = format!(
                    "HTTP/1.1 {status}\r\n\
                     Content-Type: text/html; charset=utf-8\r\n\
                     Content-Length: {}\r\n\
                     Connection: close\r\n\r\n",
                    body.len()
                );
                let _ = std::io::Write::write_all(&mut stream, head.as_bytes());
                let _ = std::io::Write::write_all(&mut stream, body.as_bytes());
            }
        });

        let mut tab = Tab::new(Size { w: 800.0, h: 600.0 });
        let page_url = format!("http://127.0.0.1:{port}/dir/page.html");
        tab.history
            .push(bowser_url::parse(&page_url, None).unwrap());
        let (document, _) = Document::parse_html(
            b"<html><head><base href=\"/assets/\"><title>Page</title></head>\
              <body><a href=\"next.html\">next</a></body></html>",
        );
        tab.install_test_document(Rc::new(RefCell::new(document)));
        let mut tabs = TabArena::default();
        let tab_id = tabs.insert(tab);
        let jar = Arc::new(Mutex::new(CookieJar::new()));
        let mut app = test_app(&mut tabs, &jar, tab_id);

        // What the click path hands over: the anchor's raw href.
        app.navigate_link("next.html");

        let deadline = Instant::now() + Duration::from_secs(15);
        while app.pending.is_some() {
            app.poll_pending();
            assert!(
                Instant::now() < deadline,
                "link navigation did not finish in time"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        let tab = app.tabs.get(tab_id).unwrap();
        assert_eq!(
            tab.current_url().unwrap().path(),
            "/assets/next.html",
            "href must resolve against <base href>, not the document URL"
        );
        assert_eq!(tab.title, "Next");
    }

    #[test]
    fn load_tab_headless_builds_display_list() {
        let tab = load_tab_headless("about:blank", Size { w: 800.0, h: 600.0 })
            .expect("headless load of about:blank");
        assert_eq!(tab.title, "welcome to bowser");
        assert_eq!(tab.history.len(), 1);
        let list = tab.display_list.expect("display list built");
        assert!(!list.items.is_empty(), "welcome page paints something");
    }

    /// A [`LoadedDocument`] straight from HTML with inline scripts
    /// resolved (no network): the unit-test stand-in for a finished
    /// load, for tests that need a live page realm.
    fn loaded_page_from(html: &[u8]) -> LoadedDocument {
        use crate::script::collect_script_refs;
        use crate::script_executor::prefetch_script_sources;

        let url = bowser_url::parse("https://example.com/", None).unwrap();
        let (document, _) = Document::parse_html(html);
        let script_sources =
            prefetch_script_sources(&collect_script_refs(&document), &url, |_| Ok(String::new()));
        LoadedDocument {
            url: url.clone(),
            trace: FetchTrace { hops: vec![url] },
            document,
            stylesheets: Vec::new(),
            image_sizes: HashMap::new(),
            script_sources,
            body: html.to_vec(),
            webfonts: Vec::new(),
        }
    }

    /// `settle_reftest_wait` fires `TestRendered` when `<html>` carries
    /// `reftest-wait` (the scrollbar-gutter-dynamic-004 shape): the
    /// page's listener runs synchronously inside `dispatchEvent`, before
    /// the settle loop and rasterization.
    ///
    /// The listener is registered post-install: mid-parse
    /// `document.addEventListener` throws in the parse shim (the shim
    /// document wrapper has no `addEventListener`), so a page registering
    /// there never gets its listener adopted — a JS-engine gap outside
    /// this handshake.
    #[test]
    fn settle_reftest_wait_fires_test_rendered_before_rasterize() {
        let mut tab = test_tab();
        crate::navigation::install_loaded_document(
            &mut tab,
            loaded_page_from(b"<html class=\"reftest-wait\"><head></head><body></body></html>"),
            "TestBrowser/9.9",
            &test_fonts(),
            &test_cookie_jar(),
        );
        let fonts = Arc::new(Mutex::new(FontCache::new()));
        // Production realm: `document.addEventListener` lives on the
        // document wrapper's prototype (`element_proto`).
        tab.page_vm_mut()
            .unwrap()
            .eval(
                "window.__test_rendered_fired = false;\
                 document.addEventListener('TestRendered', function() {\
                   window.__test_rendered_fired = true;\
                 });",
            )
            .unwrap();

        let waiting = tab
            .page_vm_mut()
            .unwrap()
            .eval(
                "(' ' + document.documentElement.getAttribute('class') + ' ').indexOf(' reftest-wait ') !== -1",
            )
            .unwrap();
        assert_eq!(waiting, Value::Bool(true), "class present before settle");

        // Short deadline: nothing clears the class here, so a full
        // budget would just burn.
        settle_reftest_wait(
            &mut tab,
            &fonts,
            Instant::now() + Duration::from_millis(200),
        );

        // The listener ran synchronously during the handshake dispatch.
        let fired = tab
            .page_vm_mut()
            .unwrap()
            .eval("window.__test_rendered_fired")
            .unwrap();
        assert_eq!(
            fired,
            Value::Bool(true),
            "TestRendered listener ran during the handshake"
        );
    }

    /// `settle_reftest_wait` folds the synchronous `TestRendered`
    /// listener's dirt before the trailing refresh: a listener that
    /// mutates style (the scrollbar-gutter-dynamic-004 shape) leaves
    /// the tab's render flags set, so a refresh re-lays out instead of
    /// being a no-op. Registered post-install (see above for why
    /// mid-parse registration cannot be used here).
    #[test]
    fn settle_reftest_wait_folds_listener_style_dirt() {
        let mut tab = test_tab();
        crate::navigation::install_loaded_document(
            &mut tab,
            loaded_page_from(
                b"<html class=\"reftest-wait\"><head></head><body><p id=\"t\">x</p></body></html>",
            ),
            "TestBrowser/9.9",
            &test_fonts(),
            &test_cookie_jar(),
        );
        let fonts = Arc::new(Mutex::new(FontCache::new()));
        // Baseline layout, so the settle's refresh is a re-layout.
        refresh(&mut tab, &fonts);
        let passes_before = tab.layout_full_passes;
        assert!(passes_before > 0, "baseline layout ran");

        tab.page_vm_mut()
            .unwrap()
            .eval(
                "document.addEventListener('TestRendered', function() {\
                   document.getElementById('t').setAttribute('style', 'color: red');\
                 });",
            )
            .unwrap();

        // Short deadline: nothing clears the class here, so a full
        // budget would just burn.
        settle_reftest_wait(
            &mut tab,
            &fonts,
            Instant::now() + Duration::from_millis(200),
        );

        // The listener ran...
        let doc = tab.document_ref().expect("document");
        let t = doc.get_element_by_id("t").expect("t");
        let style = doc.get_attribute(t, "style").unwrap_or_default();
        assert!(
            style.contains("color"),
            "listener's style mutation landed, got: {style:?}"
        );
        // ...and its dirt was folded: a refresh re-laid out instead of
        // no-opping.
        assert!(
            tab.layout_full_passes > passes_before,
            "listener dirt triggered a re-layout"
        );
    }

    /// The settle loop waits for a deferred class removal: the listener
    /// schedules the removal on a timer (through the working
    /// `setAttribute` API), so the handshake must pump the event loop
    /// until the class is gone rather than dispatch once and rasterize.
    /// Registered post-install (see above for why mid-parse
    /// `document.addEventListener` cannot be used here).
    #[test]
    fn settle_reftest_wait_waits_for_deferred_class_removal() {
        let mut tab = test_tab();
        crate::navigation::install_loaded_document(
            &mut tab,
            loaded_page_from(b"<html class=\"reftest-wait\"><head></head><body></body></html>"),
            "TestBrowser/9.9",
            &test_fonts(),
            &test_cookie_jar(),
        );
        let fonts = Arc::new(Mutex::new(FontCache::new()));
        tab.page_vm_mut()
            .unwrap()
            .eval(
                "document.addEventListener('TestRendered', function() {\
                   setTimeout(function() {\
                     document.documentElement.setAttribute('class', '');\
                   }, 20);\
                 });",
            )
            .unwrap();

        settle_reftest_wait(&mut tab, &fonts, Instant::now() + Duration::from_secs(10));

        let waiting = tab
            .page_vm_mut()
            .unwrap()
            .eval(
                "(' ' + document.documentElement.getAttribute('class') + ' ').indexOf(' reftest-wait ') !== -1",
            )
            .unwrap();
        assert_eq!(
            waiting,
            Value::Bool(false),
            "deferred class removal observed by the settle loop"
        );
    }

    /// Without the `reftest-wait` class the handshake is a no-op: no
    /// `TestRendered` event is dispatched, so pages that never opted in
    /// see no behavior change.
    #[test]
    fn settle_reftest_wait_without_class_dispatches_nothing() {
        let mut tab = test_tab();
        crate::navigation::install_loaded_document(
            &mut tab,
            loaded_page_from(
                b"<html><head><script>\
                  window.__test_rendered_seen = false;\
                  document.addEventListener('TestRendered', function() {\
                    window.__test_rendered_seen = true;\
                  });\
                  </script></head><body></body></html>",
            ),
            "TestBrowser/9.9",
            &test_fonts(),
            &test_cookie_jar(),
        );
        let fonts = Arc::new(Mutex::new(FontCache::new()));

        settle_reftest_wait(&mut tab, &fonts, Instant::now() + Duration::from_secs(5));

        let seen = tab
            .page_vm_mut()
            .unwrap()
            .eval("window.__test_rendered_seen")
            .unwrap();
        assert_eq!(
            seen,
            Value::Bool(false),
            "no spurious TestRendered dispatch without the class"
        );
    }

    /// The settle loop is bounded by `deadline`: a page that never drops
    /// the class returns promptly instead of hanging the render.
    #[test]
    fn settle_reftest_wait_honors_deadline() {
        let mut tab = test_tab();
        crate::navigation::install_loaded_document(
            &mut tab,
            loaded_page_from(b"<html class=\"reftest-wait\"><head></head><body></body></html>"),
            "TestBrowser/9.9",
            &test_fonts(),
            &test_cookie_jar(),
        );
        let fonts = Arc::new(Mutex::new(FontCache::new()));

        // An already-past deadline: the pump loop body must not run at all.
        let start = Instant::now();
        settle_reftest_wait(&mut tab, &fonts, start);
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "past deadline returns without pumping"
        );

        let waiting = tab
            .page_vm_mut()
            .unwrap()
            .eval(
                "(' ' + document.documentElement.getAttribute('class') + ' ').indexOf(' reftest-wait ') !== -1",
            )
            .unwrap();
        assert_eq!(
            waiting,
            Value::Bool(true),
            "class untouched when the deadline already passed"
        );
    }

    /// The headless event-loop pump drives `requestAnimationFrame`: a
    /// double-rAF wait (the shape WPT's `waitForCompositorCommit` and
    /// reftest `takeScreenshot` use) settles instead of wedging on a
    /// promise that never resolves.
    #[test]
    fn headless_pump_drives_request_animation_frame() {
        use crate::script::PageVm;

        let (document, _) = Document::parse_html(b"<html><head></head><body></body></html>");
        let document = Rc::new(RefCell::new(document));
        let url = bowser_url::parse("https://example.com/", None).unwrap();
        let page_vm =
            PageVm::new_for_document(Rc::clone(&document), &url, "test-ua", &[]).expect("realm");
        let mut tab = test_tab();
        tab.install_document(document, page_vm);
        tab.page_vm_mut()
            .unwrap()
            .eval(
                "window.__raf_order = [];\
                 requestAnimationFrame(function() {\
                   window.__raf_order.push(1);\
                   requestAnimationFrame(function() { window.__raf_order.push(2); });\
                 });",
            )
            .unwrap();

        let mut frames_pumped = 0u64;
        let fonts = Arc::new(Mutex::new(FontCache::new()));
        pump_headless_with_budget(
            &mut tab,
            &fonts,
            Duration::from_millis(500),
            HEADLESS_PUMP_TASK_BUDGET,
            &mut frames_pumped,
        );

        let order = tab
            .page_vm_mut()
            .unwrap()
            .eval(
                "window.__raf_order.length === 2 && \
                 window.__raf_order[0] === 1 && window.__raf_order[1] === 2",
            )
            .unwrap();
        assert_eq!(
            order,
            Value::Bool(true),
            "both rAF callbacks fired, in registration order"
        );
    }

    /// The synthetic rAF clock is caller-owned so it stays monotonic
    /// across `pump_headless_with_budget` calls: the reftest settle loop
    /// slices its budget into 50 ms pumps, and a per-call frame counter
    /// would sawtooth the timestamp (16.7→83.3 ms, back to 16.7 ms),
    /// violating the frame-pump contract.
    #[test]
    fn headless_raf_timestamps_monotonic_across_pump_calls() {
        use crate::script::PageVm;

        let (document, _) = Document::parse_html(b"<html><head></head><body></body></html>");
        let document = Rc::new(RefCell::new(document));
        let url = bowser_url::parse("https://example.com/", None).unwrap();
        let page_vm =
            PageVm::new_for_document(Rc::clone(&document), &url, "test-ua", &[]).expect("realm");
        let mut tab = test_tab();
        tab.install_document(document, page_vm);
        tab.page_vm_mut()
            .unwrap()
            .eval(
                "window.__raf_ts = [];\
                 window.__raf_arm = function() {\
                   requestAnimationFrame(function(t) {\
                     window.__raf_ts.push(t);\
                     window.__raf_arm();\
                   });\
                 };\
                 window.__raf_arm();",
            )
            .unwrap();

        let mut frames_pumped = 0u64;
        let fonts = Arc::new(Mutex::new(FontCache::new()));
        pump_headless_with_budget(
            &mut tab,
            &fonts,
            Duration::from_millis(150),
            HEADLESS_PUMP_TASK_BUDGET,
            &mut frames_pumped,
        );
        let fonts = Arc::new(Mutex::new(FontCache::new()));
        pump_headless_with_budget(
            &mut tab,
            &fonts,
            Duration::from_millis(150),
            HEADLESS_PUMP_TASK_BUDGET,
            &mut frames_pumped,
        );

        let count = tab
            .page_vm_mut()
            .unwrap()
            .eval("window.__raf_ts.length > 2")
            .unwrap();
        assert_eq!(
            count,
            Value::Bool(true),
            "several frames ran across the two pumps"
        );
        let monotonic = tab
            .page_vm_mut()
            .unwrap()
            .eval("window.__raf_ts.every(function(v, i, a) { return i === 0 || a[i-1] <= v; })")
            .unwrap();
        assert_eq!(
            monotonic,
            Value::Bool(true),
            "rAF timestamps never go backwards across pump calls"
        );
    }

    /// Script-driven element scrolls in the headless pump run the same
    /// scroll-event lifecycle the windowed `pump_scripts` runs: the
    /// shared-cell write behind the tab's back (what the bindings'
    /// `n_scroll_top_set` does — no DOM dirt) folds into the paint tree,
    /// the element's `scroll` listener fires, and the next quiet
    /// iteration fires `scrollend`. Without this, WPT's
    /// `waitForScrollendEventNoTimeout` can never resolve headless.
    #[test]
    fn headless_pump_fires_scroll_then_scrollend_for_script_scroll() {
        use crate::script::PageVm;

        let (document, _) = Document::parse_html(
            b"<html><head></head><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
        );
        // Layout + scroll tree + baked paint tree, mirroring `refresh`'s
        // pipeline (the headless pump runs before the first refresh).
        let sheet = bowser_css::parse_stylesheet(
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 500px; }",
        );
        let root = bowser_layout::layout(
            &document,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let sc = document.get_element_by_id("sc").expect("sc");
        let tall = document.get_element_by_id("tall").expect("tall");
        let document = Rc::new(RefCell::new(document));
        let url = bowser_url::parse("https://example.com/", None).unwrap();
        let page_vm =
            PageVm::new_for_document(Rc::clone(&document), &url, "test-ua", &[]).expect("realm");
        let mut tab = test_tab();
        // The refresh inside the pump re-lays out from `tab.author`:
        // without the sheet the relayout loses the scroll container.
        tab.author.push(sheet);
        tab.install_document(document, page_vm);
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        let isc = tab.scroll_tree.index_of(sc).expect("sc container");

        tab.page_vm_mut()
            .unwrap()
            .eval(
                "window.__events = [];\
                 document.getElementById('sc').addEventListener('scroll', function() { window.__events.push('scroll'); });\
                 document.getElementById('sc').addEventListener('scrollend', function() { window.__events.push('scrollend'); });",
            )
            .unwrap();

        // Simulate the bindings' `el.scrollTop = 40` write: the shared
        // cell mutates behind the tab's back, with no DOM dirt.
        tab.element_scroll()
            .state_for(sc)
            .expect("element state")
            .set_offset(0.0, 40.0);

        let mut frames_pumped = 0u64;
        let fonts = Arc::new(Mutex::new(FontCache::new()));
        pump_headless_with_budget(
            &mut tab,
            &fonts,
            Duration::from_millis(300),
            HEADLESS_PUMP_TASK_BUDGET,
            &mut frames_pumped,
        );

        let events = tab.page_vm_mut().unwrap().eval("window.__events.join(',')");
        match events {
            Ok(Value::Str(s)) => assert_eq!(
                s.as_wtf8().to_utf8_lossy().as_ref(),
                "scroll,scrollend",
                "scroll fires on the moving iteration, scrollend on the next quiet one"
            ),
            other => panic!("unexpected events value: {other:?}"),
        }
        // The paint tree folded the scroll (pixels, not just events):
        // the content moved up by the new offset. (The `dirty.paint`
        // the fold sets is serviced by the pump's refresh, so assert
        // the folded tree instead of the flag.)
        assert_eq!(tab.scroll_tree.container(isc).state.offset(), (0.0, 40.0));
        fn find_box(
            root: &bowser_layout::LayoutBox,
            node: bowser_dom::NodeId,
        ) -> Option<&bowser_layout::LayoutBox> {
            if root.node == Some(node) {
                return Some(root);
            }
            root.children.iter().find_map(|c| find_box(c, node))
        }
        let paint = tab.paint_root.as_ref().expect("paint root");
        let layout = tab.layout_root.as_deref().expect("layout root");
        let py = find_box(paint, tall).expect("tall in paint").rect.y;
        let ly = find_box(layout, tall).expect("tall in layout").rect.y;
        assert_eq!(py, ly - 40.0, "paint tree folded the scroll offset");
    }

    /// A script task that both scrolls an element and dirties style must
    /// still fire the element's `scroll` listener exactly once: the
    /// headless pump folds script-driven scrolls BEFORE the layout/style
    /// refresh the dirt triggers (windowed `pump_scripts` parity — there
    /// the fold and the listener dispatch run before `handle_redraw`'s
    /// refresh). Every layout/style refresh re-bakes the paint tree and
    /// re-seeds the folded baselines at the live offsets, so folding
    /// after the refresh would see no drift and silently drop the
    /// `scroll` (and the `scrollend`) for
    /// `el.scrollTop = 40; el.classList.add('x')`.
    #[test]
    fn headless_pump_fires_scroll_when_scroll_and_style_dirt_share_a_task() {
        use crate::script::PageVm;

        let (document, _) = Document::parse_html(
            b"<html><head></head><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
        );
        let sheet = bowser_css::parse_stylesheet(
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 500px; }",
        );
        let root = bowser_layout::layout(
            &document,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let document = Rc::new(RefCell::new(document));
        let url = bowser_url::parse("https://example.com/", None).unwrap();
        let page_vm =
            PageVm::new_for_document(Rc::clone(&document), &url, "test-ua", &[]).expect("realm");
        let mut tab = test_tab();
        // The refresh inside the pump re-lays out from `tab.author`:
        // without the sheet the relayout loses the scroll container.
        tab.author.push(sheet);
        tab.install_document(document, page_vm);
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        // Share the tab's element scroll states with the page realm so
        // the real `scrollTop` binding writes the container's shared
        // cell (the production wiring from `navigation.rs`).
        let ctx = tab
            .page_vm_mut()
            .unwrap()
            .web_api_context()
            .expect("web api context");
        bowser_js::share_element_scroll_states(ctx, Rc::clone(tab.element_scroll()));

        tab.page_vm_mut()
            .unwrap()
            .eval(
                "window.__events = [];\
                 document.getElementById('sc').addEventListener('scroll', function() { window.__events.push('scroll'); });\
                 document.getElementById('sc').addEventListener('scrollend', function() { window.__events.push('scrollend'); });",
            )
            .unwrap();

        // One task: the `scrollTop` write mutates the shared cell with
        // no DOM dirt; the `classList.add` dirties style, so this same
        // pump iteration runs the layout/style refresh.
        tab.page_vm_mut()
            .unwrap()
            .eval(
                "var sc = document.getElementById('sc'); \
                 sc.scrollTop = 40; sc.classList.add('x');",
            )
            .unwrap();

        let mut frames_pumped = 0u64;
        let fonts = Arc::new(Mutex::new(FontCache::new()));
        pump_headless_with_budget(
            &mut tab,
            &fonts,
            Duration::from_millis(300),
            HEADLESS_PUMP_TASK_BUDGET,
            &mut frames_pumped,
        );

        let events = tab.page_vm_mut().unwrap().eval("window.__events.join(',')");
        match events {
            Ok(Value::Str(s)) => assert_eq!(
                s.as_wtf8().to_utf8_lossy().as_ref(),
                "scroll,scrollend",
                "exactly one scroll fires even though the same task dirtied style"
            ),
            other => panic!("unexpected events value: {other:?}"),
        }
    }

    /// A `scrollIntoView()` recorded by the bindings must resolve in the
    /// headless pump: before this fix the headless pump never drained the
    /// request, so it sat queued forever and the viewport never moved.
    /// The drain runs after the pump's `refresh` so the request resolves
    /// against the fresh tree, and its movement flows through the same
    /// `report_scroll_movement` lifecycle the script-scroll fold uses.
    #[test]
    fn headless_pump_drains_scroll_into_view() {
        use crate::script::PageVm;

        let (document, _) = Document::parse_html(
            b"<html><head></head><body><div style=\"height: 1500px;\"></div>\
              <div id=\"target\">x</div></body></html>",
        );
        let root = bowser_layout::layout(
            &document,
            &[],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let document = Rc::new(RefCell::new(document));
        let url = bowser_url::parse("https://example.com/", None).unwrap();
        let page_vm =
            PageVm::new_for_document(Rc::clone(&document), &url, "test-ua", &[]).expect("realm");
        let mut tab = test_tab();
        // Wire the tab's viewport scroll state into the page realm, like
        // navigation does at document install.
        if let Some(ctx) = page_vm.web_api_context() {
            bowser_js::share_scroll_state(ctx, Rc::clone(tab.scroll()));
        }
        tab.install_document(document, page_vm);
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        let max_y = tab.scroll().max_offset().1;
        assert!(max_y > 0.0, "document is scrollable");

        // Record a scrollIntoView through the real binding: the request
        // sits pending — resolving it is the headless pump's job.
        tab.page_vm_mut()
            .unwrap()
            .eval(
                "window.__events = [];\
                 window.addEventListener('scroll', function() { window.__events.push('scroll'); });\
                 window.addEventListener('scrollend', function() { window.__events.push('scrollend'); });\
                 document.getElementById('target').scrollIntoView();",
            )
            .unwrap();
        assert_eq!(
            tab.scroll().offset(),
            (0.0, 0.0),
            "the request is pending, not applied"
        );

        let mut frames_pumped = 0u64;
        let fonts = Arc::new(Mutex::new(FontCache::new()));
        pump_headless_with_budget(
            &mut tab,
            &fonts,
            Duration::from_millis(300),
            HEADLESS_PUMP_TASK_BUDGET,
            &mut frames_pumped,
        );

        // Default `block: "start"`: the target's top aligns with the
        // viewport top, clamped into the document's scroll range.
        assert_eq!(tab.scroll().offset(), (0.0, max_y));
        let events = tab.page_vm_mut().unwrap().eval("window.__events.join(',')");
        match events {
            Ok(Value::Str(s)) => assert_eq!(
                s.as_wtf8().to_utf8_lossy().as_ref(),
                "scroll,scrollend",
                "the drained scrollIntoView fires scroll, then scrollend on the next quiet iteration"
            ),
            other => panic!("unexpected events value: {other:?}"),
        }
    }

    /// A smooth-behavior `scrollIntoView()` must complete on the SAME pump
    /// iteration: the headless drain degrades smooth→instant, so the
    /// offset must land immediately with no animation left in flight. A
    /// previous version only completed the animation on the next
    /// iteration's scroll-event fold — one iteration of latency, and
    /// forever stuck if the pump budget expired first.
    #[test]
    fn headless_pump_drains_smooth_scroll_into_view_same_iteration() {
        use crate::script::PageVm;

        let (document, _) = Document::parse_html(
            b"<html><head></head><body><div style=\"height: 1500px;\"></div>\
              <div id=\"target\">x</div></body></html>",
        );
        let root = bowser_layout::layout(
            &document,
            &[],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let document = Rc::new(RefCell::new(document));
        let url = bowser_url::parse("https://example.com/", None).unwrap();
        let page_vm =
            PageVm::new_for_document(Rc::clone(&document), &url, "test-ua", &[]).expect("realm");
        let mut tab = test_tab();
        // Wire the tab's viewport scroll state into the page realm, like
        // navigation does at document install.
        if let Some(ctx) = page_vm.web_api_context() {
            bowser_js::share_scroll_state(ctx, Rc::clone(tab.scroll()));
        }
        tab.install_document(document, page_vm);
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        let max_y = tab.scroll().max_offset().1;
        assert!(max_y > 0.0, "document is scrollable");

        // Record a smooth scrollIntoView through the real binding.
        tab.page_vm_mut()
            .unwrap()
            .eval(
                "window.__events = [];\
                 window.addEventListener('scroll', function() { window.__events.push('scroll'); });\
                 document.getElementById('target').scrollIntoView({behavior: 'smooth'});",
            )
            .unwrap();
        assert_eq!(
            tab.scroll().offset(),
            (0.0, 0.0),
            "the request is pending, not applied"
        );

        // One iteration only: 10ms between iterations (HEADLESS_PUMP_INTERVAL),
        // so a 1ms budget runs exactly one. The scrollend assertion below
        // needs no second iteration — the scroll listener must fire this
        // iteration, from the completion's own movement report.
        let mut frames_pumped = 0u64;
        let fonts = Arc::new(Mutex::new(FontCache::new()));
        pump_headless_with_budget(
            &mut tab,
            &fonts,
            Duration::from_millis(1),
            HEADLESS_PUMP_TASK_BUDGET,
            &mut frames_pumped,
        );

        // Default `block: "start"`: the target's top aligns with the
        // viewport top, clamped into the document's scroll range — already
        // on this iteration, with no animation left in flight.
        assert_eq!(tab.scroll().offset(), (0.0, max_y));
        assert!(
            !tab.scroll_tree.has_smooth_in_flight(),
            "the smooth→instant degrade completes the animation on the same iteration"
        );
        let events = tab.page_vm_mut().unwrap().eval("window.__events.join(',')");
        match events {
            Ok(Value::Str(s)) => assert_eq!(
                s.as_wtf8().to_utf8_lossy().as_ref(),
                "scroll",
                "the completion reports its movement this iteration (scrollend needs a second quiet iteration)"
            ),
            other => panic!("unexpected events value: {other:?}"),
        }
    }

    /// A testdriver touch-drag action sequence must scroll the page in
    /// the headless pump and run the full scroll-event lifecycle. The
    /// executor used to dispatch only the DOM touch events, so the
    /// gesture moved nothing and `scrollend` never fired — WPT
    /// scroll-snap user-scroll tests hung on their `scrollend` wait
    /// (they failed fast before only because the action promise
    /// rejected instead). The executor now pans the anchored container
    /// as the gesture's default action, stamping the scrollend tick
    /// like a real finger drag.
    #[test]
    fn headless_pump_completes_testdriver_touch_drag() {
        use crate::script::PageVm;

        let (document, _) = Document::parse_html(
            b"<html><head></head><body><div id=\"sc\"><div id=\"a\">x</div>\
              <div id=\"b\">y</div><div id=\"c\">z</div></div></body></html>",
        );
        // 400px snap areas in a 400px-tall mandatory snapport: a drag
        // that lands between snap points must snap to the nearest.
        let sheet = bowser_css::parse_stylesheet(
            "#sc { overflow-y: auto; width: 400px; height: 400px; scroll-snap-type: y mandatory; } \
             #a { height: 400px; scroll-snap-align: start; } \
             #b { height: 400px; scroll-snap-align: start; } \
             #c { height: 400px; scroll-snap-align: start; }",
        );
        let root = bowser_layout::layout(
            &document,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let sc = document.get_element_by_id("sc").expect("sc");
        let document = Rc::new(RefCell::new(document));
        let url = bowser_url::parse("https://example.com/", None).unwrap();
        let page_vm =
            PageVm::new_for_document(Rc::clone(&document), &url, "test-ua", &[]).expect("realm");
        let mut tab = test_tab();
        tab.author.push(sheet);
        tab.install_document(document, page_vm);
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        let isc = tab.scroll_tree.index_of(sc).expect("sc container");
        let ctx = tab
            .page_vm_mut()
            .unwrap()
            .web_api_context()
            .expect("web api context");
        bowser_js::share_element_scroll_states(ctx, Rc::clone(tab.element_scroll()));

        // Record the lifecycle the WPT user-scroll tests wait on, arm
        // the testdriver bridge (the pump only drains armed requests),
        // and queue a touch drag: down at (200, 350), up at (200, 100)
        // — a 250px upward flick of the finger, scrolling the offset
        // down by 250 to a non-snap point.
        tab.page_vm_mut()
            .unwrap()
            .eval(
                "window.__events = [];\
                 var sc = document.getElementById('sc');\
                 sc.addEventListener('scroll', function() { window.__events.push('scroll'); });\
                 sc.addEventListener('scrollsnapchange', function() { window.__events.push('scrollsnapchange'); });\
                 sc.addEventListener('scrollend', function() { window.__events.push('scrollend'); });\
                 window.__td_result = 'pending';\
                 window.__bowser_testdriver.__complete = function(token, ok, err) {\
                   window.__td_result = ok ? 'ok' : 'err:' + err;\
                   return true;\
                 };\
                 window.__bowser_testdriver.action_sequence([\
                   { type: 'pointer', id: 'TestPointer', parameters: { pointerType: 'touch' }, actions: [\
                     { type: 'pointerMove', duration: 0, x: 200, y: 350 },\
                     { type: 'pointerDown', button: 0 },\
                     { type: 'pause', duration: 200 },\
                     { type: 'pointerMove', duration: 0, x: 200, y: 100 },\
                     { type: 'pointerUp', button: 0 }\
                   ] }\
                 ]);",
            )
            .unwrap();

        let mut frames_pumped = 0u64;
        let fonts = Arc::new(Mutex::new(FontCache::new()));
        pump_headless_with_budget(
            &mut tab,
            &fonts,
            Duration::from_millis(500),
            HEADLESS_PUMP_TASK_BUDGET,
            &mut frames_pumped,
        );

        // The action promise settled: the gesture completed instead of
        // hanging the drain.
        let td_result = tab.page_vm_mut().unwrap().eval("window.__td_result");
        match td_result {
            Ok(Value::Str(s)) => assert_eq!(
                s.as_wtf8().to_utf8_lossy().as_ref(),
                "ok",
                "the touch action sequence must complete"
            ),
            other => panic!("unexpected __td_result: {other:?}"),
        }
        // The drag scrolled to 250 and the mandatory backstop snapped
        // to the nearest snap point at 400.
        assert_eq!(
            tab.scroll_tree.container(isc).state.offset(),
            (0.0, 400.0),
            "touch drag pans the anchored container and the snap backstop settles it"
        );
        // The full lifecycle ran in spec order: the page's `scrollend`
        // wait (what the WPT tests hang on) resolves, after the
        // `scrollsnapchange` the drag's snap-target change produced.
        let events = tab.page_vm_mut().unwrap().eval("window.__events.join(',')");
        match events {
            Ok(Value::Str(s)) => {
                let events = s.as_wtf8().to_utf8_lossy();
                let change = events.find("scrollsnapchange");
                let end = events.find("scrollend");
                assert!(
                    events.contains("scroll"),
                    "the drag fires scroll, got: {events}"
                );
                assert!(
                    change.is_some() && end.is_some() && change < end,
                    "scrollsnapchange fires before scrollend, got: {events}"
                );
                assert!(
                    events.ends_with("scrollend"),
                    "scrollend is last, got: {events}"
                );
            }
            other => panic!("unexpected events value: {other:?}"),
        }
    }

    /// A smooth `window.scrollTo()` recorded by the bindings must apply
    /// in the headless pump: before this fix the headless pump only
    /// drained the *element* smooth-scroll variant, so a viewport smooth
    /// scroll sat queued forever — the offset never moved and no `scroll`
    /// events fired. The headless drain degrades smooth→instant (the
    /// documented headless policy: headless runs have no animation
    /// ticks); the pump's scroll-event fold then reports the movement.
    #[test]
    fn headless_pump_applies_pending_viewport_smooth_scroll() {
        use crate::script::PageVm;

        let (document, _) = Document::parse_html(
            b"<html><head></head><body><div style=\"height: 2000px;\">x</div></body></html>",
        );
        let root = bowser_layout::layout(
            &document,
            &[],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let document = Rc::new(RefCell::new(document));
        let url = bowser_url::parse("https://example.com/", None).unwrap();
        let page_vm =
            PageVm::new_for_document(Rc::clone(&document), &url, "test-ua", &[]).expect("realm");
        let mut tab = test_tab();
        // Wire the tab's viewport scroll state into the page realm, like
        // navigation does at document install.
        if let Some(ctx) = page_vm.web_api_context() {
            bowser_js::share_scroll_state(ctx, Rc::clone(tab.scroll()));
        }
        tab.install_document(document, page_vm);
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();

        // Queue a smooth viewport scroll through the real binding: the
        // request sits pending — no animation ticks run headless.
        tab.page_vm_mut()
            .unwrap()
            .eval(
                "window.__events = [];\
                 window.addEventListener('scroll', function() { window.__events.push('scroll'); });\
                 window.addEventListener('scrollend', function() { window.__events.push('scrollend'); });\
                 window.scrollTo({top: 500, behavior: 'smooth'});",
            )
            .unwrap();
        assert_eq!(
            tab.scroll().offset(),
            (0.0, 0.0),
            "the smooth request is pending, not applied"
        );

        let mut frames_pumped = 0u64;
        let fonts = Arc::new(Mutex::new(FontCache::new()));
        pump_headless_with_budget(
            &mut tab,
            &fonts,
            Duration::from_millis(300),
            HEADLESS_PUMP_TASK_BUDGET,
            &mut frames_pumped,
        );

        // Smooth→instant degrade: the offset lands on the target with no
        // animation left in flight for `advance_smooth`.
        assert_eq!(tab.scroll().offset(), (0.0, 500.0));
        assert!(!tab.scroll_tree.has_smooth_in_flight());
        let events = tab.page_vm_mut().unwrap().eval("window.__events.join(',')");
        match events {
            Ok(Value::Str(s)) => assert_eq!(
                s.as_wtf8().to_utf8_lossy().as_ref(),
                "scroll,scrollend",
                "the drained smooth scroll fires scroll, then scrollend on the next quiet iteration"
            ),
            other => panic!("unexpected events value: {other:?}"),
        }
    }

    /// A script scroll of a snap container that does not land on a snap
    /// point must settle on the snap point headless: the scroll-snap
    /// backstop launches a smooth scroll toward it, and the headless
    /// pump has no animation ticks, so it completes those instantly
    /// (smooth→instant degrade, the documented headless policy).
    /// Without the completion the offset would freeze at the written
    /// value forever — and the stuck in-flight animation would suppress
    /// every later backstop and resnap for the container.
    #[test]
    fn headless_pump_completes_snap_backstop_scroll() {
        use crate::script::PageVm;

        let (document, _) = Document::parse_html(
            b"<html><head></head><body><div id=\"sc\"><div id=\"pad\">p</div><div id=\"a\">x</div><div id=\"tail\">y</div></div></body></html>",
        );
        // `#a` starts at content y = 50, so its `scroll-snap-align:
        // start` makes 50 the only snap position. It is taller than the
        // 100px snapport, so no offset fully covers it — a scroll to 40
        // must snap forward to 50 under `mandatory`.
        let sheet = bowser_css::parse_stylesheet(
            "#sc { overflow-y: auto; height: 100px; scroll-snap-type: y mandatory; } \
             #pad { height: 50px; } \
             #a { height: 150px; scroll-snap-align: start; } \
             #tail { height: 200px; }",
        );
        let root = bowser_layout::layout(
            &document,
            &[&sheet],
            Size { w: 800.0, h: 600.0 },
            &FixedFontMetrics,
        );
        let sc = document.get_element_by_id("sc").expect("sc");
        let document = Rc::new(RefCell::new(document));
        let url = bowser_url::parse("https://example.com/", None).unwrap();
        let page_vm =
            PageVm::new_for_document(Rc::clone(&document), &url, "test-ua", &[]).expect("realm");
        let mut tab = test_tab();
        tab.author.push(sheet);
        tab.install_document(document, page_vm);
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        let isc = tab.scroll_tree.index_of(sc).expect("sc container");
        let ctx = tab
            .page_vm_mut()
            .unwrap()
            .web_api_context()
            .expect("web api context");
        bowser_js::share_element_scroll_states(ctx, Rc::clone(tab.element_scroll()));

        // Script scroll to a non-snap offset: the backstop must carry it
        // to the snap point at 50 instead of freezing at 40.
        tab.page_vm_mut()
            .unwrap()
            .eval("document.getElementById('sc').scrollTop = 40;")
            .unwrap();

        let mut frames_pumped = 0u64;
        let fonts = Arc::new(Mutex::new(FontCache::new()));
        pump_headless_with_budget(
            &mut tab,
            &fonts,
            Duration::from_millis(300),
            HEADLESS_PUMP_TASK_BUDGET,
            &mut frames_pumped,
        );

        assert_eq!(
            tab.scroll_tree.container(isc).state.offset(),
            (0.0, 50.0),
            "snap backstop completed: offset rests on the snap point, not the written 40"
        );
        // The completion cleared the in-flight animation: a second script
        // scroll still gets its backstop (a stuck `Some` would suppress
        // it and leave the offset at 200). Under `mandatory` with a
        // single snap point, 200 snaps back to 50.
        tab.page_vm_mut()
            .unwrap()
            .eval("document.getElementById('sc').scrollTop = 200;")
            .unwrap();
        let mut frames_pumped = 0u64;
        pump_headless_with_budget(
            &mut tab,
            &fonts,
            Duration::from_millis(300),
            HEADLESS_PUMP_TASK_BUDGET,
            &mut frames_pumped,
        );
        assert_eq!(
            tab.scroll_tree.container(isc).state.offset(),
            (0.0, 50.0),
            "later backstops still run: no stuck in-flight smooth scroll"
        );
    }

    /// The headless event-loop pump drives timers to completion: a
    /// `setTimeout` callback scheduled during initial script execution
    /// must run (and its DOM mutation land) even though headless has no
    /// windowed `about_to_wait` pump.
    #[test]
    fn headless_pump_runs_deferred_timers() {
        let mut tab = test_tab();
        crate::navigation::install_loaded_document(
            &mut tab,
            loaded_page_from(
                b"<html><head><script>\
                  setTimeout(() => { document.body.setAttribute('data-timer', 'fired'); }, 0);\
                  </script></head><body></body></html>",
            ),
            "TestBrowser/9.9",
            &test_fonts(),
            &test_cookie_jar(),
        );
        let report = tab.script_report().expect("page VM installed");
        assert_eq!(report.executed, 1);
        assert!(report.errors.is_empty());

        let body = tab
            .document_ref()
            .expect("document")
            .get_elements_by_tag_name("body")
            .item(0)
            .expect("body");
        let before = tab
            .document_ref()
            .expect("document")
            .get_attribute(body, "data-timer")
            .map(str::to_owned);
        assert_eq!(before, None, "timer must not have fired at install time");

        let fonts = Arc::new(Mutex::new(FontCache::new()));
        pump_headless(&mut tab, &fonts);

        let after = tab
            .document_ref()
            .expect("document")
            .get_attribute(body, "data-timer")
            .map(str::to_owned);
        assert_eq!(
            after.as_deref(),
            Some("fired"),
            "deferred timer ran during the headless pump"
        );
    }

    /// The headless pump waits past the fixed [`HEADLESS_PUMP_BUDGET`]
    /// for a harness-armed page: an async test completing at ~3s (past
    /// the old 2s budget) still has its completion observed instead of
    /// being cut off into `SubtestsUnknown`.
    #[test]
    fn headless_pump_waits_for_harness_completion_past_fixed_budget() {
        let flag = crate::script::HARNESS_COMPLETION_FLAG;
        let html = format!(
            "<html><head><script>\
             window['{flag}'] = false;\
             setTimeout(() => {{ window['{flag}'] = true; }}, 3000);\
             </script></head><body></body></html>"
        );
        let mut tab = test_tab();
        crate::navigation::install_loaded_document(
            &mut tab,
            loaded_page_from(html.as_bytes()),
            "TestBrowser/9.9",
            &test_fonts(),
            &test_cookie_jar(),
        );
        assert!(
            tab.page_vm().is_some_and(|vm| vm.harness_reporter_armed()),
            "reporter flag armed at install"
        );

        let start = Instant::now();
        let fonts = Arc::new(Mutex::new(FontCache::new()));
        pump_headless(&mut tab, &fonts);
        let elapsed = start.elapsed();

        assert!(
            tab.page_vm().is_some_and(|vm| vm.harness_completed()),
            "completion flag observed"
        );
        assert!(
            elapsed >= Duration::from_millis(2500),
            "pump waited well past the old 2s budget for the ~3s completion, took {elapsed:?}"
        );
        assert!(
            elapsed < HEADLESS_HARNESS_WAIT_BUDGET,
            "pump returned on completion, not on the wait budget: {elapsed:?}"
        );
    }

    /// A harness-armed page that never completes still terminates: the
    /// pump is bounded by its deadline, not by the completion flag.
    #[test]
    fn headless_pump_terminates_when_harness_never_completes() {
        let flag = crate::script::HARNESS_COMPLETION_FLAG;
        let html = format!(
            "<html><head><script>window['{flag}'] = false;</script></head>\
             <body></body></html>"
        );
        let mut tab = test_tab();
        crate::navigation::install_loaded_document(
            &mut tab,
            loaded_page_from(html.as_bytes()),
            "TestBrowser/9.9",
            &test_fonts(),
            &test_cookie_jar(),
        );
        assert!(
            tab.page_vm().is_some_and(|vm| vm.harness_reporter_armed()),
            "reporter flag armed at install"
        );

        let start = Instant::now();
        let mut frames_pumped = 0u64;
        let fonts = Arc::new(Mutex::new(FontCache::new()));
        pump_headless_with_budget(
            &mut tab,
            &fonts,
            Duration::from_millis(300),
            1_000,
            &mut frames_pumped,
        );
        let elapsed = start.elapsed();

        assert!(!tab.page_vm().is_some_and(|vm| vm.harness_completed()));
        assert!(
            elapsed >= Duration::from_millis(300),
            "pump waited out the budget: {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_secs(10),
            "pump terminated at its budget instead of hanging: {elapsed:?}"
        );
    }

    /// A page that keeps scheduling due work (`setInterval` with a 0 ms
    /// delay) cannot spin the headless pump forever: the per-iteration
    /// task budget bounds each pump, and the deadline bounds the loop.
    #[test]
    fn headless_pump_terminates_on_spinning_interval() {
        let mut tab = test_tab();
        crate::navigation::install_loaded_document(
            &mut tab,
            loaded_page_from(
                b"<html><head><script>setInterval(() => {}, 0);</script></head>\
                  <body></body></html>",
            ),
            "TestBrowser/9.9",
            &test_fonts(),
            &test_cookie_jar(),
        );

        let start = Instant::now();
        let mut frames_pumped = 0u64;
        let fonts = Arc::new(Mutex::new(FontCache::new()));
        pump_headless_with_budget(
            &mut tab,
            &fonts,
            Duration::from_millis(300),
            1_000,
            &mut frames_pumped,
        );
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(10),
            "spinning page did not hang the pump: {elapsed:?}"
        );
    }

    /// Closing a tab tears down its page realm: pending timers are
    /// dropped with the VM (timers ride the engine's per-context queue —
    /// there are no timer threads to join — and `PageVm`'s drop cancels
    /// them), and the shared DOM must have no surviving strong refs, so
    /// nothing leaks past close.
    #[test]
    fn closing_tab_frees_document_and_drops_pending_timers() {
        let mut tab = test_tab();
        crate::navigation::install_loaded_document(
            &mut tab,
            loaded_page_from(
                b"<html><head><script>\
                  setTimeout(() => { document.body.setAttribute('data-timer', 'fired'); }, 60000);\
                  </script></head><body></body></html>",
            ),
            "TestBrowser/9.9",
            &test_fonts(),
            &test_cookie_jar(),
        );
        assert!(tab.page_vm().is_some(), "realm installed");
        // The only observable `Weak` to the installed document: upgrade
        // the web API registry's own `Weak` for this context, downgrade
        // the transient strong handle, and let the `Rc` die — what
        // remains is a pure observer, never a second owner. (There is no
        // other way to name the allocation: the scoped `with_document`
        // handle cannot escape its closure.)
        let weak = {
            let ctx = tab
                .installed()
                .expect("document")
                .page_vm()
                .web_api_context()
                .expect("web api context");
            let live = tab
                .installed_mut()
                .expect("document")
                .page_vm_mut()
                .vm_mut()
                .web_apis_document(ctx)
                .expect("registry serves the installed document");
            Rc::downgrade(&live)
        };
        drop(tab);
        assert!(
            weak.upgrade().is_none(),
            "dropping the tab must free the shared DOM: the realm and \
             its pending timers hold no strong refs past close"
        );
    }

    #[test]
    fn sync_reflow_geometry_getter_sees_same_turn_style_mutation() {
        // The LayoutEnsurer contract: a geometry getter inside a page
        // re-entry forces a synchronous style/layout pass, so it
        // observes a style mutation made by the same script turn. This
        // is what unblocks dynamic static->fixed and nested sticky.
        let (doc, _) = Document::parse_html(
            b"<html><head><title>t</title></head><body><div id=box style='width:100px;height:50px'>x</div></body></html>",
        );
        let doc = Rc::new(RefCell::new(doc));
        let url = bowser_url::parse("http://example.com/", None).expect("test url");
        let page_vm = crate::script::PageVm::new_for_document(doc.clone(), &url, "test", &[])
            .expect("install web apis");
        let mut tab = test_tab();
        if let Some(ctx) = page_vm.web_api_context() {
            bowser_js::share_scroll_state(ctx, Rc::clone(tab.scroll()));
            bowser_js::share_element_scroll_states(ctx, Rc::clone(tab.element_scroll()));
        }
        tab.install_document(doc, page_vm);
        let fonts = test_fonts();
        mark_all_dirty(&mut tab);
        refresh(&mut tab, &fonts);

        // Mutate the width and read it back through offsetWidth in one
        // script turn, inside the re-entered page (slot installed).
        let width = {
            let (_guard, page_vm) = tab.reenter_page(&fonts).expect("reenter page");
            page_vm
                .eval("var b=document.getElementById('box'); b.style.width='300px'; b.offsetWidth")
                .expect("eval")
        };
        let width = match width {
            bowser_js::Value::Int(i) => i as f64,
            bowser_js::Value::Float(f) => f,
            _ => panic!("expected a number from offsetWidth"),
        };
        assert!(
            (width - 300.0).abs() < 1.0,
            "geometry getter saw the same-turn mutation via sync reflow, got {width}"
        );
    }
}
