//! DOM event plumbing: the [`Event`] object and the dispatch contract with
//! the JS runtime.
//!
//! `bowser_dom` is deliberately JS-agnostic: it stores no event listeners.
//! The listener table lives on the JS side (`bowser_js`), keyed by
//! `(NodeId, event_type, capture)`, holding GC roots for the callback
//! functions so a registered listener keeps its closure alive.
//!
//! Dispatch contract for the JS runtime:
//!
//! 1. Build the propagation path with
//!    [`Document::propagation_path`](crate::Document::propagation_path):
//!    the target first, then each ancestor up to the document root.
//! 2. Capture phase: walk the path in *reverse* (root down to the target's
//!    parent), invoking each node's listeners registered with
//!    `capture = true` for `event.event_type`.
//! 3. Target phase: invoke the target's own listeners for both `capture`
//!    values, in registration order.
//! 4. Bubble phase: if `event.bubbles` is true, walk the path *forward*
//!    from the target's parent up to the root, invoking listeners with
//!    `capture = false`.
//! 5. If a listener called [`Event::prevent_default`] and
//!    [`Event::cancelable`] is true, skip the event's default action
//!    (navigation, checkbox toggling, ...).
//!
//! The path is computed once per dispatch; listeners added or removed
//! during dispatch do not change the node set being visited.

use crate::NodeId;

/// A DOM event object handed to JS listeners during dispatch.
///
/// Mirrors the web `Event` interface's core fields. Constructed by the JS
/// runtime (e.g. from `new Event(type, { bubbles, cancelable })` or a
/// trusted UI event source) and passed through the dispatch walk described
/// in the module docs.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    /// The event type, e.g. `"click"`.
    pub event_type: String,
    /// Whether the event has a bubble phase.
    pub bubbles: bool,
    /// Whether [`Event::prevent_default`] suppresses the default action.
    pub cancelable: bool,
    /// Set by [`Event::prevent_default`]; the JS runtime reads it after
    /// dispatch to decide whether to run the default action.
    pub default_prevented: bool,
    /// Keyboard-event `key` value (UI Events `key`), e.g. `"a"` or
    /// `"Tab"`. `None` for non-keyboard events: the JS wrapper only
    /// defines the `key` property when this is `Some` (populated by the
    /// testdriver automation bridge's key dispatch; see
    /// `bowser_app::testdriver`).
    pub key: Option<String>,
    /// Mouse/pointer state (coordinates, buttons, modifiers, and the
    /// pointer-specific fields) for `MouseEvent`/`PointerEvent`
    /// instances. `None` for events that are neither: the JS wrapper
    /// still reads the accessors as the default state (see
    /// [`PointerData::default`]) so `new MouseEvent("x").clientX` is 0.
    /// Populated by the `MouseEvent`/`PointerEvent` constructors and by
    /// host pointer/mouse dispatches (see
    /// `bowser_js::bindings::Vm::dispatch_pointer_mouse`).
    pub pointer: Option<PointerData>,
    /// Snap target pair for `SnapEvent` (`scrollsnapchange` /
    /// `scrollsnapchanging`, CSS Scroll Snap 2 §5): `(block, inline)`
    /// snap target node ids, each `None` when that axis is unsnapped.
    /// `None` for events that are not snap events: the JS wrapper only
    /// surfaces the `snapTargetBlock`/`snapTargetInline` accessors on
    /// `SnapEvent.prototype`, which read this through the stored event.
    pub snap_targets: Option<(Option<NodeId>, Option<NodeId>)>,
}

impl Event {
    /// Build an event of the given type.
    pub fn new(event_type: impl Into<String>, bubbles: bool, cancelable: bool) -> Self {
        Event {
            event_type: event_type.into(),
            bubbles,
            cancelable,
            default_prevented: false,
            key: None,
            pointer: None,
            snap_targets: None,
        }
    }

    /// Attach a keyboard `key` value to this event.
    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }

    /// Attach mouse/pointer state to this event.
    pub fn with_pointer(mut self, pointer: PointerData) -> Self {
        self.pointer = Some(pointer);
        self
    }

    /// Attach a snap target pair (logical `(block, inline)` axes) to
    /// this event, making it a `SnapEvent` payload carrier.
    pub fn with_snap_targets(mut self, block: Option<NodeId>, inline: Option<NodeId>) -> Self {
        self.snap_targets = Some((block, inline));
        self
    }

    /// Flag the event as canceled. Per DOM `preventDefault()`, this is a
    /// no-op when the event is not [`cancelable`](Event::cancelable): the
    /// flag is left unset, so `defaultPrevented` stays false.
    pub fn prevent_default(&mut self) {
        if self.cancelable {
            self.default_prevented = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Mouse / pointer events (UI Events, Pointer Events Level 3)
// ---------------------------------------------------------------------------

/// The kind of pointing device behind a [`PointerData`]: the web
/// `PointerEvent.pointerType`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PointerType {
    /// A mouse (the default device).
    #[default]
    Mouse,
    /// A pen / stylus.
    Pen,
    /// A direct-contact (touch) pointer.
    Touch,
}

impl PointerType {
    /// The `PointerEvent.pointerType` string.
    pub fn as_str(self) -> &'static str {
        match self {
            PointerType::Mouse => "mouse",
            PointerType::Pen => "pen",
            PointerType::Touch => "touch",
        }
    }

    /// Parse a `PointerEvent.pointerType` string; unknown values fall
    /// back to [`PointerType::Mouse`] (the WebDriver/pointer-actions
    /// default device).
    pub fn parse(s: &str) -> PointerType {
        match s {
            "pen" => PointerType::Pen,
            "touch" => PointerType::Touch,
            _ => PointerType::Mouse,
        }
    }
}

/// The mouse/pointer state one `MouseEvent`/`PointerEvent` carries:
/// the `MouseEvent` coordinate/button/modifier fields plus the
/// `PointerEvent` identity/contact fields.
///
/// All coordinates are CSS pixels; `client_*` is viewport-relative,
/// `screen_*` screen-relative, `page_*` document-relative, and
/// `offset_*` relative to the target's padding edge (the host
/// dispatcher computes them; the struct just carries them). `button` is
/// the button that changed state for this event (0 = primary, 1 =
/// middle, 2 = secondary, -1 = none, e.g. a plain `mousemove`);
/// `buttons` is the bitmask of buttons currently pressed (bit 0 =
/// primary, bit 1 = secondary, bit 2 = middle, ...). `pressure` is the
/// normalized contact pressure in `[0, 1]` (0.5 while a button is held
/// for devices without pressure support); `width`/`height` are the
/// contact geometry in CSS pixels (1x1 for a mouse). The
/// [`Default`] impl is the spec's zero state, which is what the JS
/// accessors read when an event carries no [`PointerData`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PointerData {
    /// Viewport-relative coordinates.
    pub client_x: f64,
    pub client_y: f64,
    /// Screen-relative coordinates.
    pub screen_x: f64,
    pub screen_y: f64,
    /// Document-relative coordinates.
    pub page_x: f64,
    pub page_y: f64,
    /// Target-padding-edge-relative coordinates.
    pub offset_x: f64,
    pub offset_y: f64,
    /// The button whose state changed (see the struct docs).
    pub button: i32,
    /// Bitmask of buttons currently pressed.
    pub buttons: u32,
    /// Movement since the previous pointer event (`movementX`/`Y`).
    pub movement_x: f64,
    pub movement_y: f64,
    /// Modifier keys held during the event.
    pub alt_key: bool,
    pub ctrl_key: bool,
    pub meta_key: bool,
    pub shift_key: bool,
    /// The pointer's identifier, stable for one contact/device stream
    /// (`PointerEvent.pointerId`).
    pub pointer_id: i32,
    /// The pointing device kind (`PointerEvent.pointerType`).
    pub pointer_type: PointerType,
    /// Whether this is the primary pointer of its type.
    pub is_primary: bool,
    /// Contact geometry in CSS pixels (`PointerEvent.width`/`height`).
    pub width: f64,
    pub height: f64,
    /// Normalized contact pressure in `[0, 1]`.
    pub pressure: f64,
    /// Pen tilt in degrees (`PointerEvent.tiltX`/`tiltY`).
    pub tilt_x: f64,
    pub tilt_y: f64,
    /// Pen barrel rotation in degrees (`PointerEvent.twist`).
    pub twist: f64,
}

impl PointerData {
    /// Pointer data for a mouse pointer at `(client_x, client_y)` with
    /// no buttons pressed: the common host-dispatch case, with the
    /// spec defaults for the rest (primary, 1x1 contact, pointer id of
    /// the mouse stream).
    pub fn mouse_at(client_x: f64, client_y: f64) -> Self {
        PointerData {
            client_x,
            client_y,
            screen_x: client_x,
            screen_y: client_y,
            page_x: client_x,
            page_y: client_y,
            button: -1,
            is_primary: true,
            width: 1.0,
            height: 1.0,
            ..PointerData::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Touch events (Touch Events, Level 2)
// ---------------------------------------------------------------------------

/// The gesture phase a host touch dispatcher reports, selecting the
/// `TouchEvent` type: [`TouchPhase::Start`] -> `"touchstart"`,
/// [`TouchPhase::Move`] -> `"touchmove"`, [`TouchPhase::End`] ->
/// `"touchend"`. `touchcancel` is not dispatched by the gesture
/// recognizer yet (v1 gap).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TouchPhase {
    /// A new contact point appeared: `"touchstart"`.
    Start,
    /// An existing contact point moved: `"touchmove"`.
    Move,
    /// A contact point lifted: `"touchend"`.
    End,
}

impl TouchPhase {
    /// The DOM event type this phase dispatches.
    pub fn event_type(self) -> &'static str {
        match self {
            TouchPhase::Start => "touchstart",
            TouchPhase::Move => "touchmove",
            TouchPhase::End => "touchend",
        }
    }
}

/// One contact point on the touch surface: a finger or stylus.
///
/// Mirrors the web `Touch` interface. All coordinates are CSS pixels:
/// `client_x`/`client_y` relative to the viewport, `screen_x`/`screen_y`
/// relative to the screen, `page_x`/`page_y` relative to the document.
#[derive(Clone, Debug, PartialEq)]
pub struct Touch {
    /// The contact's identifier, stable across the gesture.
    pub identifier: i32,
    /// The element the touch started on (per the Touch Events spec the
    /// target never changes for the gesture's lifetime).
    pub target: Option<NodeId>,
    /// Viewport-relative coordinates.
    pub client_x: f64,
    pub client_y: f64,
    /// Screen-relative coordinates.
    pub screen_x: f64,
    pub screen_y: f64,
    /// Document-relative coordinates.
    pub page_x: f64,
    pub page_y: f64,
}

impl Touch {
    /// Build a touch with the given identifier, target, and coordinates.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        identifier: i32,
        target: Option<NodeId>,
        client_x: f64,
        client_y: f64,
        screen_x: f64,
        screen_y: f64,
        page_x: f64,
        page_y: f64,
    ) -> Self {
        Touch {
            identifier,
            target,
            client_x,
            client_y,
            screen_x,
            screen_y,
            page_x,
            page_y,
        }
    }
}

/// An ordered list of [`Touch`] points: the `touches`, `targetTouches`,
/// and `changedTouches` of a [`TouchEvent`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TouchList {
    touches: Vec<Touch>,
}

impl TouchList {
    /// An empty list.
    pub fn new() -> Self {
        TouchList {
            touches: Vec::new(),
        }
    }

    /// Build a list from an iterator of touches.
    pub fn from_touches<I: IntoIterator<Item = Touch>>(iter: I) -> Self {
        TouchList {
            touches: iter.into_iter().collect(),
        }
    }

    /// Append a touch.
    pub fn push(&mut self, touch: Touch) {
        self.touches.push(touch);
    }

    /// The number of touches in the list.
    pub fn len(&self) -> usize {
        self.touches.len()
    }

    /// Whether the list holds no touches.
    pub fn is_empty(&self) -> bool {
        self.touches.is_empty()
    }

    /// The touch at `index`, or `None` when out of range (mirrors the
    /// web `TouchList.item()` returning `null`).
    pub fn item(&self, index: usize) -> Option<&Touch> {
        self.touches.get(index)
    }

    /// Iterate over the touches in order.
    pub fn iter(&self) -> impl Iterator<Item = &Touch> {
        self.touches.iter()
    }
}

/// A touch event: the [`Event`] base plus the three touch lists and the
/// modifier-key state.
///
/// Mirrors the web `TouchEvent` interface. Per the Touch Events spec the
/// event bubbles and is cancelable; `preventDefault()` on `touchstart`
/// cancels the default gesture (scroll/pan), which is how the embedder
/// knows to suppress its own panning.
#[derive(Clone, Debug, PartialEq)]
pub struct TouchEvent {
    /// The base event: type from [`TouchPhase::event_type`], `bubbles`
    /// and `cancelable` both true.
    pub event: Event,
    /// Every contact point currently on the surface.
    pub touches: TouchList,
    /// The subset of `touches` whose target is the event's target.
    pub target_touches: TouchList,
    /// The contact points that changed in this event.
    pub changed_touches: TouchList,
    /// Modifier keys held during the event.
    pub alt_key: bool,
    pub ctrl_key: bool,
    pub meta_key: bool,
    pub shift_key: bool,
}

impl TouchEvent {
    /// Build a touch event for `phase` with the given lists and modifier
    /// state.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        phase: TouchPhase,
        touches: TouchList,
        target_touches: TouchList,
        changed_touches: TouchList,
        alt_key: bool,
        ctrl_key: bool,
        meta_key: bool,
        shift_key: bool,
    ) -> Self {
        TouchEvent {
            event: Event::new(phase.event_type(), true, true),
            touches,
            target_touches,
            changed_touches,
            alt_key,
            ctrl_key,
            meta_key,
            shift_key,
        }
    }
}
