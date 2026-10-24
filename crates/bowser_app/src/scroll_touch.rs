//! Touch scrolling: drag gestures plus fling momentum (inertia).
//!
//! The drag half of touch input. DOM `TouchEvent` dispatch to page
//! scripts is a sibling's job (the bindings worker); this module is the
//! scroll pipeline only: a finger drag scrolls the anchored container,
//! and a fast release starts an inertia animation that decays to rest.
//!
//! Everything drives through [`TouchScroller::chain_scroll_touch`] —
//! [`ScrollTree::scroll_chain_touch`] in production — so touch drags
//! and flings inherit the existing per-axis clamping, `user_scroll`
//! mask, overscroll containment, and outward scroll chaining without
//! reaching into [`ScrollTree`](crate::scroll::ScrollTree) internals.
//! The touch chain additionally re-gates the residual through each
//! container's `touch-action` mask as it ascends (Pointer Events 3
//! §5.2); wheel and keyboard keep the ungated
//! [`ScrollTree::scroll_chain`].
//! In particular, a drag that moves a container with an in-flight
//! smooth scroll cancels that animation automatically: the smooth
//! engine's interruption rule drops any animation whose container
//! moved externally since its last tick.
//!
//! The single-active-gesture rule: one finger owns the gesture; extra
//! fingers are ignored (no pinch-zoom here).

use std::time::{Duration, Instant};

use bowser_dom::NodeId;

use crate::scroll::ScrollTree;
use crate::scroll::TouchAction;

/// How far back from release the drag samples count toward the
/// release-velocity estimate.
const VELOCITY_WINDOW: Duration = Duration::from_millis(100);
/// Release speed below this never starts inertia (CSS px/s): slow
/// drags and taps stop dead instead of drifting.
const INERTIA_CUTOFF_PX_PER_S: f32 = 60.0;
/// Inertia decay: `v *= exp(-dt / INERTIA_TAU)` each tick. A 2000 px/s
/// fling travels ~300 px and stops in about half a second.
const INERTIA_TAU: f32 = 0.15;
/// Hard cap on release speed (CSS px/s): a pathological sample burst —
/// two samples microseconds apart — must not fling the page for
/// seconds.
const MAX_RELEASE_SPEED: f32 = 8000.0;
/// Cap on samples kept per gesture: a drag held for minutes without
/// release must not grow memory unboundedly. The oldest non-start
/// samples are dropped once full.
const MAX_SAMPLES: usize = 256;

/// The scroll-chain sink a touch gesture drives.
///
/// [`ScrollTree`] is the production implementation; tests stub this to
/// record deltas and simulate edge clamping. Three methods on purpose:
/// the touch pipeline must not reach past the touch chain entry point
/// below, the anchor lookup, and the container's `touch-action` mask.
pub trait TouchScroller {
    /// Touch-driven scroll chain (Pointer Events 3 §5.2): scroll the
    /// container at `start` by `(dx, dy)` CSS px, chaining the
    /// unconsumed remainder outward with the residual re-gated through
    /// each container's `touch-action` mask as it ascends — every
    /// scroller the remainder reaches implements its own portion.
    /// Touch drags and flings drive through this; wheel and keyboard
    /// use [`ScrollTree::scroll_chain`] directly since `touch-action`
    /// does not apply to them. Returns the moved containers
    /// innermost-first (`None` = viewport), for `scroll` dispatch.
    fn chain_scroll_touch(&self, start: usize, dx: f32, dy: f32) -> Vec<Option<NodeId>>;
    /// Resolve a latched scroller identity to the current tree index:
    /// `None` (the viewport anchor) is always index 0; `Some(node)`
    /// looks the node's container up in the live tree. Returns `None`
    /// when the container is gone — removed, or a rebuild that dropped
    /// it — so the caller ends the gesture instead of chaining from a
    /// recycled index.
    fn resolve_anchor(&self, node: Option<NodeId>) -> Option<usize>;
    /// The container at `start`'s `touch-action` pan permissions (see
    /// [`TouchAction`]): drag and fling deltas are gated through this
    /// before chaining, so a `touch-action: pan-x` scroller never pans
    /// vertically from touch input.
    fn touch_action(&self, start: usize) -> TouchAction;
}

impl TouchScroller for ScrollTree {
    fn chain_scroll_touch(&self, start: usize, dx: f32, dy: f32) -> Vec<Option<NodeId>> {
        self.scroll_chain_touch(start, dx, dy).nodes()
    }
    fn resolve_anchor(&self, node: Option<NodeId>) -> Option<usize> {
        match node {
            // The viewport scroller is always container 0; `chain_scroll`
            // already tolerates an empty tree, so no emptiness guard is
            // needed here.
            None => Some(0),
            Some(n) => self.index_of(n),
        }
    }
    fn touch_action(&self, start: usize) -> TouchAction {
        self.container(start).touch_action
    }
}

/// One in-flight fling: the scroll-offset velocity decays exponentially
/// each tick until it drops below the cutoff.
#[derive(Debug, Clone, Copy)]
struct Inertia {
    /// Scroll-offset velocity (CSS px/s): already inverted, so a
    /// downward finger flick is a negative `vy`.
    vx: f32,
    vy: f32,
    /// Scroller the gesture anchored at, by DOM node (`None` =
    /// viewport): every tick re-resolves this against the live tree
    /// (see [`TouchScroller::resolve_anchor`]), so a relayout between
    /// ticks — which re-keys every container index — cannot send the
    /// fling through a recycled index. A container that vanished
    /// mid-fling ends the inertia instead of scrolling something else
    /// (the same rule the scrollbar drag follows).
    node: Option<NodeId>,
    /// The gesture's latched §5.2 pre-gate (see
    /// [`TouchGesture::pregate`]): the fling inherits the touch
    /// target's `touch-action` context, so momentum on a
    /// `touch-action: none` element decays without moving anything.
    pregate: TouchAction,
    /// Wall-clock time of the last tick, for `dt`.
    last: Instant,
}

/// The Pointer Events 3 §5.2 context of a touch start, latched by
/// [`TouchScrollState::started`]: the `touch-action` mask intersected
/// for the touch target. Bundled so `started`'s arg list stays under
/// the too-many-arguments lint.
#[derive(Debug, Clone, Copy)]
pub struct TouchHit {
    /// The §5.2 pre-gate: the touched element's `touch-action`
    /// intersected with its non-scrolling ancestors' masks up to but
    /// excluding the anchored container (see
    /// [`crate::scroll::touch_action_pregate`]). The spec evaluates
    /// `touch-action` when the gesture begins, so the mask is fixed for
    /// the gesture's lifetime while the anchored container's own mask
    /// stays live (re-read per move/tick).
    pub pregate: TouchAction,
}

impl TouchHit {
    /// No hit information: the permissive pre-gate. Used when there is
    /// no DOM node under the finger, and by tests that do not exercise
    /// `touch-action`.
    pub fn permissive() -> Self {
        Self {
            pregate: TouchAction::all(),
        }
    }
}

/// One active drag gesture: the finger currently scrolling the page.
#[derive(Debug)]
pub struct TouchGesture {
    /// winit touch id of the owning finger.
    pub id: u64,
    /// Position (CSS px) where the finger went down.
    pub start_pos: (f32, f32),
    /// Wall-clock time of the touch start. Populated per the gesture
    /// shape (the DOM-event sibling reads it for `TouchEvent.timeStamp`);
    /// the scroll pipeline itself estimates release velocity from
    /// `samples`, so this stays unread here.
    #[allow(dead_code)]
    pub start_time: Instant,
    /// Time-ordered `(position, time)` samples, starting with the touch
    /// start: the release-velocity estimate reads the last ~100 ms of
    /// these.
    pub samples: Vec<((f32, f32), Instant)>,
    /// The anchored container's DOM node (`None` = viewport), captured
    /// at touch start and carried into the fling's [`Inertia`] at
    /// release, so the inertia ticks can re-resolve it by node after a
    /// relayout re-keys the tree indices.
    pub node: Option<NodeId>,
    /// The §5.2 context latched at touch start (see [`TouchHit`]).
    pub hit: TouchHit,
}

/// Per-window touch-scroll state: at most one active gesture and at
/// most one in-flight inertia animation.
#[derive(Debug, Default)]
pub struct TouchScrollState {
    gesture: Option<TouchGesture>,
    inertia: Option<Inertia>,
}

/// Release velocity in scroll-offset space (CSS px/s), estimated from
/// the samples inside the last [`VELOCITY_WINDOW`] before `now`.
///
/// The sign is inverted relative to finger motion: dragging down moves
/// content down with the finger, which decreases the scroll offset, so
/// a downward flick yields a negative `vy`. Returns `(0, 0)` when fewer
/// than two in-window samples span positive time (tap, hold-release).
fn release_velocity(samples: &[((f32, f32), Instant)], now: Instant) -> (f32, f32) {
    let floor = now.checked_sub(VELOCITY_WINDOW).unwrap_or(now);
    let mut first: Option<((f32, f32), Instant)> = None;
    let mut last: Option<((f32, f32), Instant)> = None;
    for &(pos, t) in samples {
        if t >= floor {
            if first.is_none() {
                first = Some((pos, t));
            }
            last = Some((pos, t));
        }
    }
    let (Some((p0, t0)), Some((p1, t1))) = (first, last) else {
        return (0.0, 0.0);
    };
    let dt = t1.saturating_duration_since(t0).as_secs_f32();
    if dt <= 0.0 {
        return (0.0, 0.0);
    }
    (-(p1.0 - p0.0) / dt, -(p1.1 - p0.1) / dt)
}

impl TouchScrollState {
    /// Empty state: no gesture, no inertia.
    pub fn new() -> Self {
        Self::default()
    }

    /// Finger down at `(x, y)` CSS px. Anchors the gesture at the hit
    /// node's innermost scroll container, remembering its DOM node
    /// (`node`, `None` for the viewport): every drag delta re-resolves
    /// that node against the live tree ([`TouchScroller::resolve_anchor`]),
    /// so a relayout that re-keys the tree indices mid-drag cannot send
    /// the chain through a recycled container, and a fling started at
    /// release keeps the same guarantee tick by tick.
    ///
    /// A finger already down owns the gesture: extra fingers are
    /// ignored (no pinch). A new touch cancels in-flight inertia — a
    /// finger catching the page stops its momentum — but does not
    /// disturb an in-flight smooth scroll on its own: the smooth
    /// engine's interruption rule cancels it as soon as a drag delta
    /// moves its container.
    ///
    /// `hit` is the Pointer Events 3 §5.2 context latched at touch start
    /// (see [`TouchHit`]): the pre-gate mask intersected for the touch
    /// target. It is carried into the fling's [`Inertia`] at release.
    pub fn started(
        &mut self,
        id: u64,
        x: f32,
        y: f32,
        now: Instant,
        node: Option<NodeId>,
        hit: TouchHit,
    ) {
        if self.gesture.is_some() {
            return;
        }
        self.inertia = None;
        self.gesture = Some(TouchGesture {
            id,
            start_pos: (x, y),
            start_time: now,
            samples: vec![((x, y), now)],
            node,
            hit,
        });
    }

    /// Finger moved to `(x, y)` CSS px: feed the inverted drag delta
    /// into the scroll chain from the anchored container and return the
    /// moved containers innermost-first (`None` = viewport), mirroring
    /// what the wheel path reports for `scroll` dispatch.
    ///
    /// The delta is inverted — finger down means the offset decreases —
    /// and zero deltas (touch jitter, duplicate events) skip the chain
    /// entirely. Moves from a finger that does not own the gesture are
    /// ignored.
    ///
    /// The delta is gated through the anchored container's
    /// `touch-action` before chaining — and through the gesture's
    /// latched §5.2 pre-gate on top of that: a `touch-action: none`
    /// touched element (or non-scrolling ancestor) fences the drag off
    /// even when the anchored container itself would pan. A
    /// `touch-action: pan-x` scroller drops the vertical component,
    /// `none` drops the whole drag. [`TouchScroller::chain_scroll_touch`]
    /// then re-gates the residual through every outer container's mask
    /// as it ascends (Pointer Events 3 §5.2), so a chained remainder
    /// cannot pan an outer scroller the author fenced off.
    ///
    /// The anchor re-resolves by DOM node on every move (see
    /// [`TouchScroller::resolve_anchor`]): a relayout between move
    /// events rebuilds the tree and re-keys container indices, so
    /// chaining from the press-time index could scroll a recycled
    /// container. A container that vanished mid-drag ends the gesture
    /// instead (the same rule the fling ticks apply).
    pub fn moved(
        &mut self,
        target: &impl TouchScroller,
        id: u64,
        x: f32,
        y: f32,
        now: Instant,
    ) -> Vec<Option<NodeId>> {
        let (node, pregate, dx, dy) = {
            let Some(g) = self.gesture.as_mut() else {
                return Vec::new();
            };
            if g.id != id {
                return Vec::new();
            }
            let (lx, ly) = g.samples.last().map(|&(p, _)| p).unwrap_or(g.start_pos);
            if g.samples.len() >= MAX_SAMPLES {
                // Keep the tail: drop the oldest non-start sample.
                g.samples.remove(1);
            }
            g.samples.push(((x, y), now));
            let (dx, dy) = (lx - x, ly - y);
            if dx == 0.0 && dy == 0.0 {
                return Vec::new();
            }
            (g.node, g.hit.pregate, dx, dy)
        };
        let Some(start) = target.resolve_anchor(node) else {
            // The anchored container is gone (removed, or a rebuild
            // dropped it): end the gesture instead of chaining from a
            // recycled tree index.
            self.gesture = None;
            return Vec::new();
        };
        // `touch-action` gates the drag at the anchor: disallowed axes
        // never reach the chain. The gesture's latched §5.2 pre-gate
        // applies on top: the touched element (or a non-scrolling
        // ancestor) may fence the gesture off even when the anchored
        // container itself would pan. The touch chain re-gates each
        // outer container's mask as the residual ascends (idempotent
        // here, since the anchor already passed through its own mask),
        // so a fully gated drag skips the chain entirely.
        let (dx, dy) = target.touch_action(start).gate(dx, dy);
        let (dx, dy) = pregate.gate(dx, dy);
        if dx == 0.0 && dy == 0.0 {
            return Vec::new();
        }
        target.chain_scroll_touch(start, dx, dy)
    }

    /// Finger lifted at `(x, y)` CSS px: ends the gesture, starting
    /// inertia when the release velocity clears the cutoff. A release
    /// from the non-owning finger is ignored.
    pub fn ended(&mut self, id: u64, x: f32, y: f32, now: Instant) {
        if self.gesture.as_ref().is_some_and(|g| g.id != id) {
            return;
        }
        let Some(g) = self.gesture.take() else {
            return;
        };
        let mut samples = g.samples;
        samples.push(((x, y), now));
        let (vx, vy) = release_velocity(&samples, now);
        let speed = vx.hypot(vy);
        if speed >= INERTIA_CUTOFF_PX_PER_S {
            // Clamp pathological bursts (two samples microseconds apart)
            // without changing ordinary fling speeds.
            let scale = (MAX_RELEASE_SPEED / speed).min(1.0);
            self.inertia = Some(Inertia {
                vx: vx * scale,
                vy: vy * scale,
                node: g.node,
                pregate: g.hit.pregate,
                last: now,
            });
        }
    }

    /// Gesture cancelled (system interruption): the gesture ends with
    /// no inertia, whatever the drag velocity was.
    pub fn cancelled(&mut self, id: u64) {
        if self.gesture.as_ref().is_some_and(|g| g.id == id) {
            self.gesture = None;
        }
    }

    /// Advance in-flight inertia to wall-clock `now`, applying the
    /// exponentially decayed displacement through the scroll chain from
    /// the anchored container.
    ///
    /// The per-tick delta is gated through the anchored container's
    /// `touch-action` like the drag deltas are (see
    /// [`TouchScrollState::moved`]) — anchor mask plus the gesture's
    /// latched §5.2 pre-gate: a fling on a `touch-action: none` touched
    /// element decays without moving anything. The touch chain
    /// re-gates the residual through each outer container's mask as it
    /// ascends, so chained momentum honors outer `touch-action` too.
    ///
    /// The anchor re-resolves by DOM node every tick (see
    /// [`TouchScroller::resolve_anchor`]): a relayout between ticks
    /// rebuilds the tree and re-keys container indices, so chaining
    /// from the press-time index could scroll a recycled container. A
    /// container that vanished mid-fling ends the inertia instead.
    ///
    /// Edge behavior is the documented choice: the inertia delta rides
    /// the touch chain ([`ScrollTree::scroll_chain_touch`]), so when the
    /// anchored container clamps on an axis the remainder chains
    /// outward to its scroller ancestors — momentum transfers to the
    /// outer scroller instead of dying at the edge (and honors
    /// `overscroll-behavior` containment exactly like wheel input,
    /// while `touch-action` re-gating keeps each outer scroller's pan
    /// permissions authoritative).
    /// Even when nothing moves (every container at its edge), the
    /// velocity keeps decaying, so the animation always terminates at
    /// the cutoff and the event loop never stays hot.
    ///
    /// Returns the moved containers innermost-first (`None` =
    /// viewport), for the same moved-container list the smooth engine
    /// reports into (so `scroll`/`scrollend` fire).
    pub fn advance_inertia(
        &mut self,
        target: &impl TouchScroller,
        now: Instant,
    ) -> Vec<Option<NodeId>> {
        let Some(anim) = self.inertia.as_mut() else {
            return Vec::new();
        };
        let dt = now.saturating_duration_since(anim.last).as_secs_f32();
        anim.last = now;
        if dt <= 0.0 {
            // Same-tick re-entry: no time passed, no movement, no decay.
            return Vec::new();
        }
        let decay = (-dt / INERTIA_TAU).exp();
        anim.vx *= decay;
        anim.vy *= decay;
        if anim.vx.hypot(anim.vy) < INERTIA_CUTOFF_PX_PER_S {
            self.inertia = None;
            return Vec::new();
        }
        let Some(start) = target.resolve_anchor(anim.node) else {
            // The anchored container is gone (removed, or a rebuild
            // dropped it): end the fling instead of chaining from a
            // recycled tree index.
            self.inertia = None;
            return Vec::new();
        };
        // `touch-action` gates the fling at the anchor: disallowed axes
        // never reach the chain. The gesture's latched §5.2 pre-gate
        // applies on top, like the drag deltas in
        // [`TouchScrollState::moved`]. The velocity keeps decaying
        // below, so the animation always terminates even when fully
        // gated.
        let (dx, dy) = target.touch_action(start).gate(anim.vx * dt, anim.vy * dt);
        let (dx, dy) = anim.pregate.gate(dx, dy);
        if dx == 0.0 && dy == 0.0 {
            return Vec::new();
        }
        target.chain_scroll_touch(start, dx, dy)
    }

    /// Whether an inertia animation is in flight. The window worker
    /// folds this into the smooth-scroll conjunction that keeps the
    /// event loop turning until every animation completes; the cutoff
    /// in [`TouchScrollState::advance_inertia`] guarantees the loop
    /// goes idle again.
    pub fn touch_has_animation_in_flight(&self) -> bool {
        self.inertia.is_some()
    }

    /// Whether a gesture is currently active (test hook).
    #[cfg(test)]
    fn has_gesture(&self) -> bool {
        self.gesture.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bowser_dom::Document;
    use std::cell::RefCell;
    use std::collections::HashMap;

    /// Stub scroll sink: records the deltas it was asked to chain,
    /// replays a canned moved-container list, and resolves latched
    /// anchors through a configurable node -> index map (standing in
    /// for the tree across a simulated relayout). The `touch-action`
    /// mask is `all()` (the `auto` initial) unless a test narrows it —
    /// globally via [`FakeScroller::with_touch_action`], or per
    /// container via [`FakeScroller::with_chain_actions`].
    struct FakeScroller {
        calls: RefCell<Vec<(usize, f32, f32)>>,
        moved: Vec<Option<NodeId>>,
        anchors: RefCell<HashMap<Option<NodeId>, usize>>,
        action: TouchAction,
        /// Per-container masks for multi-container chain tests:
        /// `chain[i]` is the mask at index `i` (index 0 = viewport).
        /// When set, [`TouchScroller::chain_scroll_touch`] walks the
        /// anchor's linear parent chain (`start` down to 0), re-gating
        /// the residual through each container's mask and recording one
        /// call per container — the inner is treated as exhausted, so
        /// the full gated residual ascends. When unset, the chain is a
        /// single anchor call, as before.
        chain: Option<Vec<TouchAction>>,
    }

    impl FakeScroller {
        fn new(moved: Vec<Option<NodeId>>) -> Self {
            Self {
                calls: RefCell::new(Vec::new()),
                moved,
                anchors: RefCell::new(HashMap::new()),
                action: TouchAction::all(),
                chain: None,
            }
        }

        /// Map a latched anchor node to its current tree index, like a
        /// fresh [`ScrollTree`] would after layout.
        fn with_anchor(self, node: Option<NodeId>, index: usize) -> Self {
            self.anchors.borrow_mut().insert(node, index);
            self
        }

        /// The `touch-action` mask every container reports.
        fn with_touch_action(self, action: TouchAction) -> Self {
            Self { action, ..self }
        }

        /// Per-container `touch-action` masks for a linear ancestor
        /// chain (`chain[0]` = viewport). Enables the multi-container
        /// chain simulation in
        /// [`TouchScroller::chain_scroll_touch`].
        fn with_chain_actions(self, chain: Vec<TouchAction>) -> Self {
            Self {
                chain: Some(chain),
                ..self
            }
        }

        /// The `touch-action` mask the container at `index` reports.
        fn action_at(&self, index: usize) -> TouchAction {
            self.chain
                .as_ref()
                .and_then(|c| c.get(index).copied())
                .unwrap_or(self.action)
        }

        fn deltas(&self) -> Vec<(usize, f32, f32)> {
            self.calls.borrow().clone()
        }
    }

    impl TouchScroller for FakeScroller {
        fn chain_scroll_touch(&self, start: usize, dx: f32, dy: f32) -> Vec<Option<NodeId>> {
            let Some(chain) = &self.chain else {
                // Single-container mode: the anchor call, as before.
                self.calls.borrow_mut().push((start, dx, dy));
                return self.moved.clone();
            };
            // Multi-container simulation: the inner is exhausted, so the
            // gated residual ascends the linear parent chain, re-gated
            // through each container's mask like the production touch
            // chain.
            let mut idx = start.min(chain.len().saturating_sub(1));
            let (mut rx, mut ry) = (dx, dy);
            loop {
                let (gx, gy) = self.action_at(idx).gate(rx, ry);
                self.calls.borrow_mut().push((idx, gx, gy));
                rx = gx;
                ry = gy;
                if idx == 0 {
                    break;
                }
                idx -= 1;
            }
            self.moved.clone()
        }

        fn resolve_anchor(&self, node: Option<NodeId>) -> Option<usize> {
            // Mirror the production rule: the viewport anchor is always
            // container 0; element anchors resolve through the map.
            match node {
                None => Some(0),
                Some(_) => self.anchors.borrow().get(&node).copied(),
            }
        }

        fn touch_action(&self, start: usize) -> TouchAction {
            self.action_at(start)
        }
    }

    fn t0() -> Instant {
        Instant::now()
    }

    fn ms(t: Instant, ms: u64) -> Instant {
        t + Duration::from_millis(ms)
    }

    /// Two distinct element nodes for anchor-identity tests.
    fn test_nodes() -> (NodeId, NodeId) {
        let (doc, _) = Document::parse_html(b"<div id=\"a\"></div><div id=\"b\"></div>");
        (
            doc.get_element_by_id("a").expect("a"),
            doc.get_element_by_id("b").expect("b"),
        )
    }

    #[test]
    fn drag_state_machine_drives_chain_with_inverted_deltas() {
        let (a, _) = test_nodes();
        let mut s = TouchScrollState::new();
        // The gesture anchors by node: it resolves to the press-time
        // container index 3 on every move.
        let fake = FakeScroller::new(vec![None]).with_anchor(Some(a), 3);
        let t = t0();
        s.started(7, 100.0, 200.0, t, Some(a), TouchHit::permissive());
        assert!(s.has_gesture());
        // Finger moves down 50 px: the scroll offset decreases by 50.
        let moved = s.moved(&fake, 7, 100.0, 250.0, ms(t, 16));
        assert_eq!(moved, vec![None]);
        assert_eq!(fake.deltas(), vec![(3, 0.0, -50.0)]);
        // Finger moves up-left 10 px: the offset increases by 10.
        s.moved(&fake, 7, 90.0, 240.0, ms(t, 32));
        assert_eq!(fake.deltas()[1], (3, 10.0, 10.0));
        // Release below the fling cutoff: no inertia.
        s.ended(7, 90.0, 240.0, ms(t, 500));
        assert!(!s.has_gesture());
        assert!(!s.touch_has_animation_in_flight());
    }

    #[test]
    fn zero_delta_moves_skip_the_chain() {
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None]);
        let t = t0();
        s.started(1, 0.0, 0.0, t, None, TouchHit::permissive());
        let moved = s.moved(&fake, 1, 0.0, 0.0, ms(t, 16));
        assert!(moved.is_empty());
        assert!(fake.deltas().is_empty());
    }

    #[test]
    fn extra_fingers_are_ignored_no_pinch() {
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None]);
        let t = t0();
        s.started(1, 0.0, 0.0, t, None, TouchHit::permissive());
        // Second finger down: ignored, does not steal or reset the gesture.
        s.started(2, 500.0, 500.0, ms(t, 10), None, TouchHit::permissive());
        assert!(s.has_gesture());
        // Its moves and release are ignored too.
        assert!(s.moved(&fake, 2, 500.0, 600.0, ms(t, 20)).is_empty());
        assert!(fake.deltas().is_empty());
        s.ended(2, 500.0, 600.0, ms(t, 30));
        assert!(s.has_gesture());
        // The owning finger still drives and ends the gesture.
        s.ended(1, 0.0, 0.0, ms(t, 40));
        assert!(!s.has_gesture());
    }

    #[test]
    fn cancelled_gesture_never_flings() {
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None]);
        let t = t0();
        s.started(1, 0.0, 0.0, t, None, TouchHit::permissive());
        // Fast downward flick...
        s.moved(&fake, 1, 0.0, 300.0, ms(t, 50));
        // ...then a system cancel: no inertia despite the velocity.
        s.cancelled(1);
        assert!(!s.has_gesture());
        assert!(!s.touch_has_animation_in_flight());
        let moved = s.advance_inertia(&fake, ms(t, 60));
        assert!(moved.is_empty());
        assert_eq!(fake.deltas().len(), 1, "only the drag delta");
    }

    #[test]
    fn fast_release_starts_inertia_in_fling_direction() {
        let (a, _) = test_nodes();
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None]).with_anchor(Some(a), 2);
        let t = t0();
        s.started(1, 0.0, 400.0, t, Some(a), TouchHit::permissive());
        // Flick upward: 300 px in 50 ms => 6000 px/s finger velocity.
        s.moved(&fake, 1, 0.0, 100.0, ms(t, 50));
        s.ended(1, 0.0, 100.0, ms(t, 50));
        assert!(s.touch_has_animation_in_flight());
        // One tick 16 ms later: the chain sees the decayed fling delta,
        // still upward (offset increasing => positive dy).
        let before = fake.deltas().len();
        let moved = s.advance_inertia(&fake, ms(t, 66));
        assert_eq!(moved, vec![None]);
        let (start, dx, dy) = fake.deltas()[before];
        assert_eq!(start, 2, "inertia chains from the anchored container");
        assert_eq!(dx, 0.0);
        assert!(dy > 0.0, "upward flick keeps increasing the offset");
        // Roughly v*dt with v = 6000*exp(-0.016/0.15) px/s.
        let expected = 6000.0 * (-0.016 / INERTIA_TAU).exp() * 0.016;
        assert!((dy - expected).abs() < 1.0, "dy={dy} expected~{expected}");
    }

    #[test]
    fn slow_release_starts_no_inertia() {
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None]);
        let t = t0();
        s.started(1, 0.0, 0.0, t, None, TouchHit::permissive());
        // 5 px over a full second: well under the cutoff.
        s.moved(&fake, 1, 0.0, 5.0, ms(t, 1000));
        s.ended(1, 0.0, 5.0, ms(t, 1000));
        assert!(!s.touch_has_animation_in_flight());
    }

    #[test]
    fn hold_then_release_does_not_fling() {
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None]);
        let t = t0();
        s.started(1, 0.0, 0.0, t, None, TouchHit::permissive());
        // Fast drag, then the finger rests 300 ms before lifting: the
        // last 100 ms of samples show no motion.
        s.moved(&fake, 1, 0.0, 300.0, ms(t, 50));
        s.ended(1, 0.0, 300.0, ms(t, 400));
        assert!(!s.touch_has_animation_in_flight());
    }

    #[test]
    fn inertia_decays_below_cutoff_and_stops() {
        let (a, _) = test_nodes();
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None]).with_anchor(Some(a), 0);
        let t = t0();
        s.started(1, 0.0, 400.0, t, Some(a), TouchHit::permissive());
        s.moved(&fake, 1, 0.0, 100.0, ms(t, 50));
        s.ended(1, 0.0, 100.0, ms(t, 50));
        assert!(s.touch_has_animation_in_flight());
        // Tick until it dies: exponential decay always crosses the
        // cutoff, so this terminates (the loop would spin forever
        // otherwise — and so would the event loop).
        let mut ticks = 0;
        let mut now = ms(t, 50);
        while s.touch_has_animation_in_flight() {
            now = ms(now, 16);
            s.advance_inertia(&fake, now);
            ticks += 1;
            assert!(ticks < 10_000, "inertia never decayed out");
        }
        // No movement reported on the terminal tick.
        let moved = s.advance_inertia(&fake, ms(now, 16));
        assert!(moved.is_empty());
    }

    #[test]
    fn inertia_keeps_chaining_from_anchor_past_edges() {
        // The stub simulates "anchor clamped, outer consumes": whatever
        // the inertia tick hands to the chain is what the outer scroller
        // would receive. The assertion is on our side of the seam: every
        // tick chains the full decayed delta from the anchored container,
        // letting `scroll_chain` propagate the remainder outward.
        let (a, _) = test_nodes();
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None]).with_anchor(Some(a), 5);
        let t = t0();
        s.started(1, 0.0, 400.0, t, Some(a), TouchHit::permissive());
        s.moved(&fake, 1, 0.0, 100.0, ms(t, 50));
        s.ended(1, 0.0, 100.0, ms(t, 50));
        let base = fake.deltas().len();
        let moved = s.advance_inertia(&fake, ms(t, 66));
        assert_eq!(moved, vec![None]);
        for &(start, _, _) in &fake.deltas()[base..] {
            assert_eq!(start, 5, "every tick re-chains from the anchor");
        }
    }

    #[test]
    fn inertia_reresolves_anchor_by_node_after_relayout() {
        // The fling anchors by DOM node, not tree index: when a relayout
        // re-keys the containers, the ticks follow the node to its new
        // index instead of chaining from the stale press-time one.
        let (a, _) = test_nodes();
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None]).with_anchor(Some(a), 2);
        let t = t0();
        s.started(1, 0.0, 400.0, t, Some(a), TouchHit::permissive());
        s.moved(&fake, 1, 0.0, 100.0, ms(t, 50));
        s.ended(1, 0.0, 100.0, ms(t, 50));
        assert!(s.touch_has_animation_in_flight());
        // First tick: the press-time index.
        s.advance_inertia(&fake, ms(t, 66));
        assert_eq!(fake.deltas().last().expect("tick chained").0, 2);
        // Relayout re-keys the tree: node `a` now lives at index 7.
        fake.anchors.borrow_mut().insert(Some(a), 7);
        let base = fake.deltas().len();
        let moved = s.advance_inertia(&fake, ms(t, 82));
        assert_eq!(moved, vec![None]);
        assert_eq!(
            fake.deltas()[base].0,
            7,
            "tick follows the node, not the stale press-time index"
        );
        assert!(
            s.touch_has_animation_in_flight(),
            "re-resolution keeps the fling alive"
        );
    }

    #[test]
    fn inertia_ends_when_anchor_container_vanishes() {
        // A relayout that drops the anchored container ends the fling
        // instead of chaining from a recycled tree index.
        let (a, _) = test_nodes();
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None]).with_anchor(Some(a), 2);
        let t = t0();
        s.started(1, 0.0, 400.0, t, Some(a), TouchHit::permissive());
        s.moved(&fake, 1, 0.0, 100.0, ms(t, 50));
        s.ended(1, 0.0, 100.0, ms(t, 50));
        assert!(s.touch_has_animation_in_flight());
        // The container is removed: the anchor no longer resolves.
        fake.anchors.borrow_mut().remove(&Some(a));
        let base = fake.deltas().len();
        let moved = s.advance_inertia(&fake, ms(t, 66));
        assert!(moved.is_empty());
        assert!(
            !s.touch_has_animation_in_flight(),
            "vanished anchor ends the fling"
        );
        assert_eq!(
            fake.deltas().len(),
            base,
            "no chain traffic after the anchor vanished"
        );
    }

    #[test]
    fn drag_reresolves_anchor_by_node_mid_drag() {
        // The drag anchors by DOM node, not tree index: when a DOM
        // mutation mid-drag rebuilds the tree and re-keys the
        // containers, the next move follows the node to its new index
        // instead of chaining from the stale press-time one.
        let (a, _) = test_nodes();
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None]).with_anchor(Some(a), 2);
        let t = t0();
        s.started(1, 0.0, 400.0, t, Some(a), TouchHit::permissive());
        s.moved(&fake, 1, 0.0, 350.0, ms(t, 16));
        assert_eq!(fake.deltas().last().expect("move chained").0, 2);
        // The DOM mutates mid-drag: node `a` now lives at index 7.
        fake.anchors.borrow_mut().insert(Some(a), 7);
        let base = fake.deltas().len();
        s.moved(&fake, 1, 0.0, 300.0, ms(t, 32));
        assert_eq!(fake.deltas().len(), base + 1, "move chained");
        assert_eq!(
            fake.deltas()[base].0,
            7,
            "move follows the node, not the stale press-time index"
        );
        assert!(s.has_gesture(), "re-resolution keeps the drag alive");
    }

    #[test]
    fn drag_ends_when_anchor_container_vanishes_mid_drag() {
        // A mid-drag rebuild that drops the anchored container ends the
        // gesture instead of chaining from a recycled tree index.
        let (a, _) = test_nodes();
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None]).with_anchor(Some(a), 2);
        let t = t0();
        s.started(1, 0.0, 400.0, t, Some(a), TouchHit::permissive());
        s.moved(&fake, 1, 0.0, 350.0, ms(t, 16));
        let base = fake.deltas().len();
        // The container is removed: the anchor no longer resolves.
        fake.anchors.borrow_mut().remove(&Some(a));
        let moved = s.moved(&fake, 1, 0.0, 300.0, ms(t, 32));
        assert!(moved.is_empty());
        assert!(!s.has_gesture(), "vanished anchor ends the drag");
        assert_eq!(
            fake.deltas().len(),
            base,
            "no chain traffic after the anchor vanished"
        );
    }

    #[test]
    fn new_touchstart_cancels_in_flight_inertia() {
        let (a, b) = test_nodes();
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None])
            .with_anchor(Some(a), 0)
            .with_anchor(Some(b), 1);
        let t = t0();
        s.started(1, 0.0, 400.0, t, Some(a), TouchHit::permissive());
        s.moved(&fake, 1, 0.0, 100.0, ms(t, 50));
        s.ended(1, 0.0, 100.0, ms(t, 50));
        assert!(s.touch_has_animation_in_flight());
        // Finger back down: momentum dies immediately, the new gesture
        // anchors fresh.
        s.started(2, 10.0, 10.0, ms(t, 200), Some(b), TouchHit::permissive());
        assert!(!s.touch_has_animation_in_flight());
        assert!(s.has_gesture());
        let base = fake.deltas().len();
        let moved = s.advance_inertia(&fake, ms(t, 300));
        assert!(moved.is_empty());
        assert_eq!(fake.deltas().len(), base, "no chain traffic after cancel");
        // The new gesture drives from its own anchor.
        s.moved(&fake, 2, 10.0, 30.0, ms(t, 216));
        assert_eq!(fake.deltas()[base], (1, 0.0, -20.0));
    }

    #[test]
    fn tap_without_move_never_flings() {
        let mut s = TouchScrollState::new();
        let t = t0();
        s.started(1, 50.0, 50.0, t, None, TouchHit::permissive());
        s.ended(1, 50.0, 50.0, ms(t, 80));
        assert!(!s.touch_has_animation_in_flight());
    }

    /// `touch-action: pan-x` on the scroller: the vertical drag
    /// component never reaches the chain, the horizontal one does.
    #[test]
    fn touch_action_pan_x_gates_vertical_drag() {
        let pan_x = TouchAction {
            x_neg: true,
            x_pos: true,
            y_neg: false,
            y_pos: false,
        };
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None]).with_touch_action(pan_x);
        let t = t0();
        s.started(1, 0.0, 0.0, t, None, TouchHit::permissive());
        // Straight down: dy < 0, gated away entirely.
        let moved = s.moved(&fake, 1, 0.0, 100.0, ms(t, 16));
        assert!(moved.is_empty());
        assert!(fake.deltas().is_empty(), "gated drag skips the chain");
        // Diagonal: only the horizontal component survives.
        s.moved(&fake, 1, 40.0, 150.0, ms(t, 32));
        assert_eq!(fake.deltas(), vec![(0, -40.0, 0.0)]);
    }

    /// `touch-action: none`: drags and flings move nothing.
    #[test]
    fn touch_action_none_blocks_drag_and_fling() {
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None]).with_touch_action(TouchAction::default());
        let t = t0();
        s.started(1, 0.0, 400.0, t, None, TouchHit::permissive());
        // Fast flick in both axes...
        s.moved(&fake, 1, 100.0, 100.0, ms(t, 50));
        assert!(fake.deltas().is_empty(), "gated drag skips the chain");
        s.ended(1, 100.0, 100.0, ms(t, 50));
        // ...starts inertia (the release was fast)...
        assert!(s.touch_has_animation_in_flight());
        // ...but every tick is gated to zero and chains nothing.
        let moved = s.advance_inertia(&fake, ms(t, 66));
        assert!(moved.is_empty());
        assert!(fake.deltas().is_empty(), "gated fling ticks skip the chain");
    }

    /// Pointer Events 3 §5.2 pre-gate: the latched mask of the touched
    /// element and its non-scrolling ancestors gates the drag on top of
    /// the anchored container's own mask — a `touch-action: none`
    /// element inside a permissive scroller fences the gesture off.
    #[test]
    fn pregate_none_fences_drag_despite_permissive_anchor() {
        let (a, _) = test_nodes();
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None]).with_anchor(Some(a), 0);
        let t = t0();
        s.started(
            1,
            0.0,
            400.0,
            t,
            Some(a),
            TouchHit {
                pregate: TouchAction::default(),
            },
        );
        s.moved(&fake, 1, 0.0, 100.0, ms(t, 16));
        assert!(fake.deltas().is_empty(), "pre-gated drag skips the chain");
    }

    /// The pre-gate composes with the anchor mask per axis: `pan-x` on
    /// the touched element keeps the horizontal drag component while the
    /// permissive anchor would have let both through.
    #[test]
    fn pregate_pan_x_keeps_horizontal_drag_component() {
        let pan_x = TouchAction {
            x_neg: true,
            x_pos: true,
            y_neg: false,
            y_pos: false,
        };
        let (a, _) = test_nodes();
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None]).with_anchor(Some(a), 0);
        let t = t0();
        s.started(1, 0.0, 400.0, t, Some(a), TouchHit { pregate: pan_x });
        // Vertical drag: the pre-gate drops it before the chain.
        s.moved(&fake, 1, 0.0, 500.0, ms(t, 16));
        assert!(fake.deltas().is_empty());
        // Horizontal drag: survives the pre-gate and chains.
        s.moved(&fake, 1, 100.0, 500.0, ms(t, 32));
        assert_eq!(fake.deltas(), vec![(0, -100.0, 0.0)]);
    }

    /// The fling inherits the gesture's pre-gate: momentum from a
    /// `touch-action: none` touch target decays without chaining.
    #[test]
    fn pregate_none_fences_fling() {
        let (a, _) = test_nodes();
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None]).with_anchor(Some(a), 0);
        let t = t0();
        s.started(
            1,
            0.0,
            400.0,
            t,
            Some(a),
            TouchHit {
                pregate: TouchAction::default(),
            },
        );
        // Fast flick: starts inertia...
        s.moved(&fake, 1, 100.0, 100.0, ms(t, 50));
        s.ended(1, 100.0, 100.0, ms(t, 50));
        assert!(s.touch_has_animation_in_flight());
        // ...but every tick is pre-gated to zero and chains nothing.
        let moved = s.advance_inertia(&fake, ms(t, 66));
        assert!(moved.is_empty());
        assert!(
            fake.deltas().is_empty(),
            "pre-gated fling ticks skip the chain"
        );
    }

    /// Directional keywords gate per direction, not just per axis:
    /// `pan-up` lets content pan up (`dy > 0`) and blocks the reverse.
    #[test]
    fn touch_action_directional_keyword_gates_one_direction() {
        let pan_up = TouchAction {
            x_neg: false,
            x_pos: false,
            y_neg: false,
            y_pos: true,
        };
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None]).with_touch_action(pan_up);
        let t = t0();
        s.started(1, 0.0, 200.0, t, None, TouchHit::permissive());
        // Finger down: dy < 0, gated away.
        assert!(s.moved(&fake, 1, 0.0, 250.0, ms(t, 16)).is_empty());
        assert!(fake.deltas().is_empty());
        // Finger up: dy > 0, chains.
        s.moved(&fake, 1, 0.0, 150.0, ms(t, 32));
        assert_eq!(fake.deltas(), vec![(0, 0.0, 100.0)]);
    }

    /// A fling's per-tick delta is gated like the drag deltas: with
    /// `pan-x` only the horizontal velocity component reaches the chain.
    #[test]
    fn touch_action_gates_fling_ticks() {
        let pan_x = TouchAction {
            x_neg: true,
            x_pos: true,
            y_neg: false,
            y_pos: false,
        };
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None]).with_touch_action(pan_x);
        let t = t0();
        s.started(1, 200.0, 400.0, t, None, TouchHit::permissive());
        // Diagonal flick up-left: vx > 0 and vy > 0 in offset space.
        s.moved(&fake, 1, 100.0, 300.0, ms(t, 50));
        s.ended(1, 100.0, 300.0, ms(t, 50));
        assert!(s.touch_has_animation_in_flight());
        let base = fake.deltas().len();
        let moved = s.advance_inertia(&fake, ms(t, 66));
        assert_eq!(moved, vec![None]);
        let (start, dx, dy) = fake.deltas()[base];
        assert_eq!(start, 0);
        assert!(dx > 0.0, "horizontal fling component survives");
        assert_eq!(dy, 0.0, "vertical fling component is gated away");
    }

    /// Outer `touch-action: pan-x`, inner `auto`: a vertical drag pans
    /// the inner scroller, but the chained remainder is gated at the
    /// outer container and never pans it vertically. A horizontal drag
    /// still reaches the outer scroller (Pointer Events 3 §5.2).
    #[test]
    fn touch_action_gates_chained_residual_per_container() {
        let pan_x = TouchAction {
            x_neg: true,
            x_pos: true,
            y_neg: false,
            y_pos: false,
        };
        let (a, _) = test_nodes();
        let mut s = TouchScrollState::new();
        // Index 1 = inner (`auto`), index 0 = outer (`pan-x`).
        let fake = FakeScroller::new(vec![None])
            .with_anchor(Some(a), 1)
            .with_chain_actions(vec![pan_x, TouchAction::all()]);
        let t = t0();
        s.started(1, 0.0, 0.0, t, Some(a), TouchHit::permissive());
        // Vertical drag: finger down 100 px => offset dy = -100.
        s.moved(&fake, 1, 0.0, 100.0, ms(t, 16));
        let deltas = fake.deltas();
        let inner = deltas
            .iter()
            .find(|&&(idx, _, _)| idx == 1)
            .expect("inner chained");
        assert_eq!(*inner, (1, 0.0, -100.0), "inner pans vertically");
        let deltas = fake.deltas();
        let outer = deltas
            .iter()
            .find(|&&(idx, _, _)| idx == 0)
            .expect("outer reached by the chain");
        assert_eq!(
            *outer,
            (0, 0.0, 0.0),
            "outer's pan-x drops the vertical remainder"
        );
        // Horizontal drag: the outer scroller pans freely on x.
        let base = fake.deltas().len();
        s.moved(&fake, 1, 60.0, 100.0, ms(t, 32));
        let deltas = fake.deltas();
        let outer = deltas
            .iter()
            .skip(base)
            .find(|&&(idx, _, _)| idx == 0)
            .expect("outer reached by the chain");
        assert_eq!(*outer, (0, -60.0, 0.0), "outer pans horizontally");
    }

    /// Outer `touch-action: none`: the inner scroller pans, but the
    /// chained remainder dies at the outer container on every axis.
    #[test]
    fn touch_action_none_outer_drops_chained_residual() {
        let (a, _) = test_nodes();
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None])
            .with_anchor(Some(a), 1)
            .with_chain_actions(vec![TouchAction::default(), TouchAction::all()]);
        let t = t0();
        s.started(1, 0.0, 0.0, t, Some(a), TouchHit::permissive());
        // Diagonal drag: finger down-left => offset (+40, -100).
        s.moved(&fake, 1, 40.0, 100.0, ms(t, 16));
        let deltas = fake.deltas();
        let inner = deltas
            .iter()
            .find(|&&(idx, _, _)| idx == 1)
            .expect("inner chained");
        assert_eq!(*inner, (1, -40.0, -100.0), "inner pans on both axes");
        let deltas = fake.deltas();
        let outer = deltas
            .iter()
            .find(|&&(idx, _, _)| idx == 0)
            .expect("outer reached by the chain");
        assert_eq!(*outer, (0, 0.0, 0.0), "none drops the whole remainder");
    }

    /// Fling momentum chains through the same per-container gating: the
    /// inner scroller flings vertically, but an outer `pan-x` scroller
    /// still drops the vertical remainder.
    #[test]
    fn touch_action_gates_chained_fling_residual_per_container() {
        let pan_x = TouchAction {
            x_neg: true,
            x_pos: true,
            y_neg: false,
            y_pos: false,
        };
        let (a, _) = test_nodes();
        let mut s = TouchScrollState::new();
        let fake = FakeScroller::new(vec![None])
            .with_anchor(Some(a), 1)
            .with_chain_actions(vec![pan_x, TouchAction::all()]);
        let t = t0();
        s.started(1, 0.0, 400.0, t, Some(a), TouchHit::permissive());
        // Flick upward: 300 px in 50 ms.
        s.moved(&fake, 1, 0.0, 100.0, ms(t, 50));
        s.ended(1, 0.0, 100.0, ms(t, 50));
        assert!(s.touch_has_animation_in_flight());
        let base = fake.deltas().len();
        s.advance_inertia(&fake, ms(t, 66));
        let deltas = fake.deltas();
        let inner = deltas[base..]
            .iter()
            .find(|&&(idx, _, _)| idx == 1)
            .expect("inner chained");
        assert!(inner.2 > 0.0, "inner flings vertically, got {inner:?}");
        let deltas = fake.deltas();
        let outer = deltas[base..]
            .iter()
            .find(|&&(idx, _, _)| idx == 0)
            .expect("outer reached by the chain");
        assert_eq!(outer.2, 0.0, "outer's pan-x drops the vertical remainder");
    }
}
