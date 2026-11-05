//! testdriver.js Action API: `test_driver.bless`, `test_driver.click`,
//! `test_driver.send_keys` (phase 1) and `test_driver.action_sequence`
//! (phase 2, the WebDriver Actions executor — see the executor section
//! below).
//!
//! ## Protocol design
//!
//! Upstream `resources/testdriver.js` is built around a vendor extension
//! point: the page-visible `window.test_driver` API delegates every
//! automation command to `window.test_driver_internal`, which the
//! browser vendor overrides (upstream's `testdriver-vendor.js`) with
//! `in_automation = true` plus real implementations. Without an
//! override, `click()`/`send_keys()` fall back to waiting forever for a
//! physical user — every automated call hangs.
//!
//! Bowser implements exactly that vendor contract, in two halves:
//!
//! * **Page-JS half** — [`VENDOR_SHIM_JS`], appended by the WPT fixture
//!   server to every served `/resources/testdriver.js` response (see
//!   [`maybe_append_vendor_shim`]). Appending (rather than a separate
//!   file) guarantees the shim runs immediately after upstream defines
//!   `test_driver_internal` and before any test script runs. The shim
//!   sets `in_automation = true` and routes `click`/`send_keys`/`bless`
//!   to the native bridge, returning a promise per call.
//! * **Native half** — `window.__bowser_testdriver`, installed by
//!   `bowser_js` on every page global. Its `click(element)` /
//!   `send_keys(element, keys)` / `bless()` natives translate the
//!   element argument to a [`NodeId`](bowser_dom::NodeId) and queue a
//!   [`bowser_js::TestDriverRequest`]. They never dispatch events
//!   themselves: dispatch re-enters the page VM, so the embedder drains
//!   the queue between event-loop tasks
//!   ([`drain_testdriver_requests`], hooked into the headless pump and
//!   the windowed `pump_scripts`)
//!   and completes each promise through the shim-installed
//!   `__complete(token, ok, err)` callback.
//!
//! Why not an HTTP `/testdriver` route on the fixture server, or
//! WebDriver `/session` endpoints? `BowserDriver` is a one-shot
//! subprocess driver: the browser child owns the live page, and the
//! fixture server (driver process) cannot reach into it. The only
//! channel that can act on the live page is in-process — the native
//! bridge above. WebDriver session endpoints are explicitly out of
//! scope for this phase.
//!
//! Automation gating: `window.__bowser_testdriver` is installed on
//! every page global, but a queued request executes only while the
//! bridge is armed — the vendor shim installs `__complete` on the
//! bridge exactly while its own calls are in flight, and
//! [`drain_testdriver_requests`] drops anything queued otherwise. A
//! page that never ran the shim cannot self-bless through direct
//! bridge calls: no dispatch, no user-activation grant. (A page that
//! deliberately replicates the shim's protocol is indistinguishable
//! from the shim; the gate stops the trivial forgery, not a page
//! reimplementing testdriver-vendor.js.)
//!
//! ## What each action does
//!
//! * `click(element)`: upstream's page-JS checks run first (`inView`
//!   via `document.elementsFromPoint`, the click-interception check —
//!   both need the `bowser_js` geometry natives). The driver then
//!   scrolls the element into view (authoritative: the page-JS
//!   `scrollIntoView` records a request the headless pump never
//!   drains) and dispatches a trusted click through
//!   [`PageEventDispatch::dispatch_click`](crate::script::PageEventDispatch::dispatch_click)
//!   — the same primitive real mouse input uses — then grants
//!   transient user activation.
//! * `send_keys(element, keys)`: scroll into view, then one
//!   `keydown`+`keyup` pair per key through
//!   [`PageEventDispatch::dispatch_key_at`](crate::script::PageEventDispatch::dispatch_key_at),
//!   carrying the UI-Events `key` value. WebDriver special-key
//!   codepoints (`\u{E004}` = Tab, …) are mapped by
//!   [`split_testdriver_keys`].
//! * `bless(intent, action)`: the shim overrides upstream's public
//!   `bless` (whose button-click mechanism needs a synchronous layout
//!   pass the headless pump does not run) to grant transient user
//!   activation directly, then run `action()` under it.
//!
//! ## Phase-1 gaps (for phase 2)
//!
//! * No obscured-element *re*-check after the driver scrolls; the
//!   vendor clicks the element itself and ignores the `coords`
//!   argument (it clicks the in-view center, which is the element).
//! * `elementsFromPoint` is a paint-order approximation (no full
//!   stacking contexts, transforms, clip paths, or `position: fixed`
//!   frames; `visibility: hidden` / `pointer-events: none` not
//!   filtered).
//! * `getClientRects()` returns one border-box rect; fragment-level
//!   rects for multi-line inlines are missing.
//! * Key dispatch has no `keypress`, no `repeat`, no `code`/
//!   `location`, and no default action (typing into a text field
//!   does not change its value). Modifier-key *state* exists only in
//!   the phase-2 executor below (held modifiers reach subsequent
//!   key/pointer/wheel events there); `send_keys` still dispatches
//!   bare pairs.
//! * `test_driver.bless` grants activation without the upstream
//!   button-click dance; cross-context `bless` is ignored.
//! * Transient activation never expires and nothing consumes it yet.
//! * Elements created after the last layout pass see stale geometry in
//!   upstream's `inView` check (the headless pump does not relayout
//!   between tasks): `click()` on them rejects with "element click
//!   intercepted"; `send_keys()` still works because it never rejects
//!   on that check.
//!
//! ## Phase 2: `action_sequence`
//!
//! The Actions executor lives below (see the "Actions executor"
//! section docs): tick-synchronized sources, per-source pointer
//! state and a global modifier set persisted across calls, synthesized
//! pointer/mouse/key/touch/wheel events, and an executor-local focus
//! model. Its own gaps are listed with its section docs.

use bowser_dom::{DialogFocusEvent, EscapeTarget, NodeId};

use crate::navigation;
use crate::script::PageEventDispatch;
use crate::scroll::RevealAlign;
use crate::tab::Tab;
use crate::window::JsDispatch;

/// Path (decoded, query stripped) whose responses gain the vendor shim.
/// Matches upstream WPT's canonical testdriver.js location.
/// Path suffix the fixture server appends the vendor shim to. Only used
/// when the `wpt-driver` feature enables the fixture server.
#[allow(dead_code)]
const TESTDRIVER_JS_PATH: &str = "/resources/testdriver.js";

/// Bowser's `testdriver-vendor.js` equivalent, appended to the served
/// `/resources/testdriver.js` (see [`maybe_append_vendor_shim`]).
///
/// Requires `window.__bowser_testdriver` (the native bridge); without
/// it the shim returns immediately and upstream's non-automation
/// fallbacks stay in place, so the shim is inert in foreign contexts.
///
/// Arming: the shim installs `__complete` on the bridge synchronously
/// before invoking a bridge native and removes it when its last
/// in-flight call settles. The embedder drains a queued request only
/// while `__complete` is present, so a page that never ran the shim
/// cannot self-bless through direct `__bowser_testdriver` calls: the
/// drain drops those requests without dispatching or granting user
/// activation (see [`drain_testdriver_requests`]). On load the shim
/// also calls the bridge's `__arm()` when it exists (forward
/// compatibility with a native-side armed flag; a no-op today).
///
/// Only used when the `wpt-driver` feature enables the fixture server.
#[allow(dead_code)]
pub const VENDOR_SHIM_JS: &str = r#";(function(){
var tdi=window.test_driver_internal;
if(!tdi)return;
var pending=new Map();
var inFlight=0;
function bridge(){return window.__bowser_testdriver;}
(function(){
var b=bridge();
if(b&&typeof b.__arm==="function"){try{b.__arm();}catch(e){}}
})();
function uninstallComplete(){
var b=bridge();
if(b){try{delete b.__complete;}catch(e){}}
}
function complete(token,ok,err){
var p=pending.get(token);
if(p===undefined)return false;
pending.delete(token);
if(--inFlight===0)uninstallComplete();
if(ok){p.resolve();}else{p.reject(new Error(String(err)));}
return true;
}
function callNative(name,args){
return new Promise(function(resolve,reject){
var b=bridge();
if(!b){reject(new Error("testdriver bridge not installed"));return;}
inFlight++;
b.__complete=complete;
var token;
try{token=b[name].apply(b,args);}
catch(e){
if(--inFlight===0)uninstallComplete();
reject(e instanceof Error?e:new Error(String(e)));
return;
}
if(typeof token!=="number"||!(token>0)){
if(--inFlight===0)uninstallComplete();
reject(new Error("testdriver bridge returned no token"));
return;
}
pending.set(token,{resolve:resolve,reject:reject});
});
}
tdi.in_automation=true;
tdi.click=function(element,coords){return callNative("click",[element]);};
tdi.send_keys=function(element,keys){return callNative("send_keys",[element,keys]);};
tdi.bless=function(intent){return callNative("bless",[]);};
tdi.action_sequence=function(actions){return callNative("action_sequence",[actions]);};
if(window.test_driver){
window.test_driver.bless=function(intent,action,context){
return tdi.bless(intent).then(function(){
if(typeof action==="function")return action();
return null;
});
};
}
})();"#;

/// Append [`VENDOR_SHIM_JS`] to a response body when `decoded_path` is
/// the testdriver.js URL path (query already stripped by the caller);
/// any other path passes through untouched. The fixture server calls
/// this on both its plain and harness-prefix static routes.
///
/// The path is normalized ([`normalize_url_path`]) before comparing:
/// the server canonicalizes `/resources/./testdriver.js` (and friends)
/// to the same file, so the shim decision must see through those
/// spellings too — otherwise the page gets upstream bytes without the
/// shim, `in_automation` stays false, and upstream `click()` waits
/// forever for a physical user (hang to timeout).
///
/// Only used when the `wpt-driver` feature enables the fixture server.
#[allow(dead_code)]
pub fn maybe_append_vendor_shim(decoded_path: &str, mut bytes: Vec<u8>) -> Vec<u8> {
    if normalize_url_path(decoded_path) == TESTDRIVER_JS_PATH {
        bytes.extend_from_slice(VENDOR_SHIM_JS.as_bytes());
    }
    bytes
}

/// Normalize a percent-decoded URL path for the shim comparison:
/// collapse duplicate slashes and resolve `.`/`..` segments lexically.
/// Mirrors what the fixture server's path canonicalization does when it
/// resolves the file, so every URL spelling that serves testdriver.js
/// also gains the vendor shim. A trailing slash is preserved:
/// `/resources/testdriver.js/` does not serve the file (and must not
/// gain the shim).
fn normalize_url_path(path: &str) -> String {
    let mut segs: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                segs.pop();
            }
            s => segs.push(s),
        }
    }
    let mut out = String::with_capacity(path.len());
    out.push('/');
    out.push_str(&segs.join("/"));
    if path.ends_with('/') && !out.ends_with('/') {
        out.push('/');
    }
    out
}

// ---------------------------------------------------------------------------
// WebDriver special-key codepoints -> UI-Events `key` values
// ---------------------------------------------------------------------------
//
// The normalized key values from the WebDriver spec's keyboard-actions
// table (https://w3c.github.io/webdriver/#keyboard-actions). Unlisted
// codepoints pass through unchanged ("otherwise it is key").

/// Map one WebDriver private-use key codepoint to its normalized
/// UI-Events `key` value. `None` for ordinary characters (they map to
/// themselves).
pub fn webdriver_key_value(c: char) -> Option<&'static str> {
    Some(match c {
        '\u{E000}' => "Unidentified",
        '\u{E001}' => "Cancel",
        '\u{E002}' => "Help",
        '\u{E003}' => "Backspace",
        '\u{E004}' => "Tab",
        '\u{E005}' => "Clear",
        '\u{E006}' => "Return",
        '\u{E007}' => "Enter",
        '\u{E008}' => "Shift",
        '\u{E009}' => "Control",
        '\u{E00A}' => "Alt",
        '\u{E00B}' => "Pause",
        '\u{E00C}' => "Escape",
        '\u{E00D}' => " ",
        '\u{E00E}' => "PageUp",
        '\u{E00F}' => "PageDown",
        '\u{E010}' => "End",
        '\u{E011}' => "Home",
        '\u{E012}' => "ArrowLeft",
        '\u{E013}' => "ArrowUp",
        '\u{E014}' => "ArrowRight",
        '\u{E015}' => "ArrowDown",
        '\u{E016}' => "Insert",
        '\u{E017}' => "Delete",
        '\u{E018}' => ";",
        '\u{E019}' => "=",
        '\u{E01A}' => "0",
        '\u{E01B}' => "1",
        '\u{E01C}' => "2",
        '\u{E01D}' => "3",
        '\u{E01E}' => "4",
        '\u{E01F}' => "5",
        '\u{E020}' => "6",
        '\u{E021}' => "7",
        '\u{E022}' => "8",
        '\u{E023}' => "9",
        '\u{E024}' => "*",
        '\u{E025}' => "+",
        '\u{E026}' => ",",
        '\u{E027}' => "-",
        '\u{E028}' => ".",
        '\u{E029}' => "/",
        '\u{E031}' => "F1",
        '\u{E032}' => "F2",
        '\u{E033}' => "F3",
        '\u{E034}' => "F4",
        '\u{E035}' => "F5",
        '\u{E036}' => "F6",
        '\u{E037}' => "F7",
        '\u{E038}' => "F8",
        '\u{E039}' => "F9",
        '\u{E03A}' => "F10",
        '\u{E03B}' => "F11",
        '\u{E03C}' => "F12",
        '\u{E03D}' => "Meta",
        '\u{E040}' => "ZenkakuHankaku",
        '\u{E050}' => "Shift",
        '\u{E051}' => "Control",
        '\u{E052}' => "Alt",
        '\u{E053}' => "Meta",
        '\u{E054}' => "PageUp",
        '\u{E055}' => "PageDown",
        '\u{E056}' => "End",
        '\u{E057}' => "Home",
        '\u{E058}' => "ArrowLeft",
        '\u{E059}' => "ArrowUp",
        '\u{E05A}' => "ArrowRight",
        '\u{E05B}' => "ArrowDown",
        '\u{E05C}' => "Insert",
        '\u{E05D}' => "Delete",
        _ => return None,
    })
}

/// Split a `test_driver.send_keys` string into per-key UI-Events `key`
/// values, mapping WebDriver special-key codepoints through
/// [`webdriver_key_value`]; ordinary characters map to themselves.
pub fn split_testdriver_keys(keys: &str) -> Vec<String> {
    keys.chars()
        .map(|c| {
            webdriver_key_value(c)
                .map(str::to_owned)
                .unwrap_or_else(|| c.to_string())
        })
        .collect()
}

// ---------------------------------------------------------------------------
// WebDriver Actions executor (phase 2: `test_driver.action_sequence`)
// ---------------------------------------------------------------------------
//
// The vendor shim forwards `test_driver.action_sequence(actions)` to
// the `__bowser_testdriver.action_sequence` native, which deserializes
// the payload into a [`bowser_js::TestDriverActionSequence`] (element
// origins resolved to `NodeId`s natively, since they cannot survive a
// JSON round trip). The executor below ticks it: every tick advances
// every source one action (WebDriver §17 "Actions": sources shorter
// than the longest idle with implicit pauses), dispatching through
// [`PageEventDispatch`] and
// [`Vm::dispatch_pointer_mouse`](bowser_js::Vm::dispatch_pointer_mouse).
//
// ## State
//
// One [`ExecutorState`] per page (not per request): pointer positions,
// pressed buttons, the per-button press targets (for the
// click-on-release rule), the global modifier set, and the
// executor-local focused node all persist across `action_sequence`
// calls — a `keyDown` in one call still holds the modifier for the
// next. [`drain_testdriver_requests`] hands the state back and forth
// through a thread-local keyed by the page's web-API context, so state
// never leaks across pages but survives separate drains of one page.
//
// ## What each source does
//
// * pointer (mouse first): `pointerMove` resolves its origin (viewport
//   = absolute, pointer = relative to the pointer's current position,
//   element = element in-view center + offset, via
//   [`PageEventDispatch::element_client_center`]); on a target change
//   the `pointerout`/`pointerleave`/`mouseout`/`mouseleave` and
//   `pointerover`/`pointerenter`/`mouseover`/`mouseenter` boundary
//   chains fire, then paired `pointermove`/`mousemove` (a cancelled
//   `pointermove` suppresses the compat `mousemove`). `pointerDown`
//   fires `pointerdown`+`mousedown` and moves focus — a cancelled
//   `pointerdown`/`touchstart` suppresses the compat `mousedown`,
//   a cancelled `mousedown` suppresses the focus move, and only
//   focusable targets take focus at all; `pointerUp` fires
//   `pointerup`+`mouseup`, then a synthesized `click` (primary
//   button) or `auxclick` (middle/back/forward) at the nearest
//   common inclusive ancestor of the press and release targets,
//   or `contextmenu` when the released button is the secondary one
//   and a matching press landed on the release target (a cancelled
//   `pointerup`/`touchend` suppresses that whole compat chain).
//   `pointerCancel` fires `pointercancel`. A `"touch"` pointer also routes each phase
//   through `Vm::dispatch_touch` (`touchstart`/`touchmove`/
//   `touchend`); a `"pen"` rides the mouse path with `PointerType::Pen`.
// * key: separate `keyDown`/`keyUp` dispatches (the phase-1 paired
//   halves, split), targeting the executor-focused node (falling back
//   to `<body>`/the document element). The modifier state machine
//   holds `Shift`/`Control`/`Alt`/`Meta` across actions and sets the
//   flags on every subsequent event (fully on pointer events through
//   `PointerData`; on key events the flags ride the dispatched init
//   object, so inline handlers see them). `Tab` moves focus through
//   the sequential focus order.
// * wheel: a `scroll` action dispatches a `wheel` event at the resolved
//   point, then performs the default scroll (the scroll-tree chain,
//   unless prevented).
// * none/pause: durations are clamped to 0 (headless policy — the
//   executor runs between event-loop tasks and never sleeps).
//
// ## Deliberate gaps (documented, not silent)
//
// * Key events carry the modifier flags only as init-object fields:
//   inline handlers see `event.shiftKey`, but `addEventListener`
//   listeners get a wrapped `Event` whose stored shape carries `key`
//   only (no `KeyboardEvent` wrapper exists in the bindings yet).
//   Pointer events carry the full `PointerData` set, including
//   modifiers, to all listeners.
// * `wheel` events have no modifier channel in `Vm::dispatch_wheel`:
//   the held flags are not visible on `WheelEvent` objects.
// * Inline `on*` handlers do not run for the pointer/mouse events:
//   they dispatch through `Vm::dispatch_pointer_mouse`, which only
//   fires `addEventListener` listeners (the phase-1 click path runs
//   inline handlers manually; the executor does not).
// * `pointerenter`/`pointerleave`/`mouseenter`/`mouseleave` fire
//   without `relatedTarget` (no such channel on `PointerData`).
// * No `keypress`, no text-editing default actions, no pointer
//   capture, no activation expiry — same list as the task's deferred
//   set.

use std::cell::RefCell;
use std::collections::HashMap;

use bowser_dom::{Document, NodeKind, PointerData, PointerType, Touch, TouchPhase};
use bowser_js::{
    ActionItem, ActionItemKind, ActionOrigin, ActionSource, ActionSourceKind, PointerMouseKind,
    TestDriverActionSequence,
};

use crate::script::KeyModifiers;

/// WebDriver button index -> `PointerData.buttons` bit: bit 0 =
/// primary, bit 1 = secondary, bit 2 = middle (UI Events), so
/// WebDriver's (left=0, middle=1, right=2, back=3, forward=4) indices
/// map 0->1, 1->4, 2->2, 3->8, 4->16. Higher indices fall back to
/// `1 << button`, saturating to 0 beyond 31.
fn button_bit(button: u32) -> u32 {
    match button {
        0 => 1,
        1 => 4,
        2 => 2,
        3 => 8,
        4 => 16,
        b => 1u32.checked_shl(b).unwrap_or(0),
    }
}

/// Normalize an action-sequence key `value` to a UI-Events `key` name:
/// a single WebDriver special-key codepoint (e.g. `"\u{E008}"`)
/// maps through the phase-1 [`webdriver_key_value`] table, a single
/// ordinary char maps to itself, and anything longer (already a key
/// name like `"Shift"`) passes through unchanged.
fn normalize_action_key(raw: &str) -> String {
    let mut chars = raw.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => webdriver_key_value(c)
            .map(str::to_owned)
            .unwrap_or_else(|| c.to_string()),
        _ => raw.to_string(),
    }
}

/// Resolve a `pointerMove`/`scroll` action's target point in client
/// coordinates from its origin: viewport-absolute, relative to the
/// pointer's `current` position, or relative to an element's in-view
/// `element_center`. An element origin without a laid-out box is an
/// executor error (WebDriver "move target out of bounds").
///
/// Non-finite coordinates (NaN, ±∞ — in the action's own `x`/`y`,
/// in the pointer's current position, in the element centre, or
/// produced by the addition) are rejected: propagating them into
/// hit testing would silently miss every box (all comparisons
/// against NaN are false) and dispatch at a meaningless point, so
/// the action errors instead of dispatching.
fn resolve_action_point(
    origin: &ActionOrigin,
    x: f64,
    y: f64,
    current: (f64, f64),
    element_center: Option<(f64, f64)>,
) -> Result<(f64, f64), String> {
    if !x.is_finite() || !y.is_finite() {
        return Err("action_sequence: non-finite action coordinate".to_string());
    }
    let point = match origin {
        ActionOrigin::Viewport => (x, y),
        ActionOrigin::Pointer => (current.0 + x, current.1 + y),
        ActionOrigin::Element(_) => element_center
            .map(|(cx, cy)| (cx + x, cy + y))
            .ok_or_else(|| "action_sequence: element origin has no layout box".to_string())?,
    };
    if !point.0.is_finite() || !point.1.is_finite() {
        return Err("action_sequence: non-finite resolved point".to_string());
    }
    Ok(point)
}

/// Whether `node` is still in `doc`'s tree: its propagation path
/// reaches the document root. A node detached after being focused
/// (removed from its parent, never appended) tops out at itself
/// instead, so the executor's stored focus must not trust it.
fn node_in_document(doc: &Document, node: NodeId) -> bool {
    doc.propagation_path(node).last() == Some(&doc.root())
}

/// Whether `node` can take focus from a click (the focus-lite
/// model's click-focus rule): naturally focusable elements, plus
/// any element with `tabindex >= -1` (`-1` is click/programmatic
/// focus only; sequential Tab order uses the stricter
/// [`td_is_sequentially_focusable`]). A plain `<div>` without
/// `tabindex` does NOT take focus — the previously focused element
/// keeps it. Disabled form controls and `type=hidden` inputs never
/// take focus.
fn td_is_focusable(doc: &Document, node: NodeId) -> bool {
    let NodeKind::Element(data) = doc.kind(node) else {
        return false;
    };
    let tag = &data.tag_name;
    if matches!(tag.as_str(), "button" | "input" | "select" | "textarea")
        && doc.has_attribute(node, "disabled")
    {
        return false;
    }
    if tag.as_str() == "input"
        && matches!(doc.get_attribute(node, "type"), Some(t) if t.eq_ignore_ascii_case("hidden"))
    {
        return false;
    }
    if let Some(index) = td_tab_index(doc, node) {
        return index >= -1;
    }
    match tag.as_str() {
        "a" | "area" => doc.has_attribute(node, "href"),
        "button" | "select" | "textarea" | "input" => true,
        _ => false,
    }
}

/// The nearest common inclusive ancestor of `a` and `b` (the
/// click/auxclick target when press and release land on different
/// nodes): the first element on `a`'s inclusive ancestor chain that
/// is also on `b`'s. `None` when the two share no element ancestor
/// (detached or disjoint subtrees).
fn nearest_common_inclusive_ancestor(doc: &Document, a: NodeId, b: NodeId) -> Option<NodeId> {
    let a_path = element_path(doc, a);
    let b_path = element_path(doc, b);
    a_path.into_iter().find(|n| b_path.contains(n))
}

/// The global modifier-key state, held across ticks and across
/// `action_sequence` calls until released.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ModifierState {
    shift: bool,
    ctrl: bool,
    alt: bool,
    meta: bool,
}

impl ModifierState {
    /// Track one key press/release: returns true when `key` is a
    /// modifier whose flag was set (`down`) or cleared (`!down`).
    fn set_key(&mut self, key: &str, down: bool) -> bool {
        let slot = match key {
            "Shift" => &mut self.shift,
            "Control" => &mut self.ctrl,
            "Alt" => &mut self.alt,
            "Meta" => &mut self.meta,
            _ => return false,
        };
        *slot = down;
        true
    }

    /// Stamp the flags onto a pointer event's `PointerData`.
    fn apply_to_pointer(&self, data: &mut PointerData) {
        data.shift_key = self.shift;
        data.ctrl_key = self.ctrl;
        data.alt_key = self.alt;
        data.meta_key = self.meta;
    }

    /// The same flags as a key-dispatch [`KeyModifiers`].
    fn as_key_modifiers(&self) -> KeyModifiers {
        KeyModifiers {
            shift: self.shift,
            ctrl: self.ctrl,
            alt: self.alt,
            meta: self.meta,
        }
    }
}

/// Per-pointer-source state: position, pressed buttons, the
/// per-button press targets (the click-on-release rule compares the
/// release target against these), and the current hover target for
/// the boundary chains. For `"touch"` sources `touch_target` is the
/// touch's original target (the spec fixes it for the gesture's
/// lifetime); the hover target still tracks the point for the
/// `pointerover`/`pointerout` chains. `touch_pan` is the scroll
/// container an unprevented `touchstart` anchored the gesture's
/// default pan at (`None` = no pan: not a touch contact, or the page
/// prevented the gesture); cleared on release/cancel.
#[derive(Debug, Clone)]
struct PointerSourceState {
    x: f64,
    y: f64,
    buttons: u32,
    down_targets: HashMap<u32, NodeId>,
    hover_target: Option<NodeId>,
    pointer_type: PointerType,
    pointer_id: i32,
    touch_target: Option<NodeId>,
    touch_pan: Option<usize>,
    /// Light-dismiss pointerdown records (HTML §6.12.2,
    /// §4.11.5; w85 hide-all-until), mirroring the windowed
    /// input path: the press-time popover ancestor and
    /// dialog ancestor, each `Some(ancestor)` when a press
    /// was recorded (`None` ancestor = press outside all
    /// of that kind) and `None` when no press was recorded.
    /// The release dismisses only on an equal ancestor;
    /// cleared on every pointerup / pointercancel.
    light_dismiss_popover_press: Option<Option<NodeId>>,
    light_dismiss_dialog_press: Option<Option<NodeId>>,
    /// Executor tick of this pointer's last stored action (see
    /// [`ExecutorState::pointer_use_seq`]): the deterministic
    /// "most recently active pointer" key for wheel actions with
    /// `origin: "pointer"`.
    last_used: u64,
}

impl Default for PointerSourceState {
    fn default() -> Self {
        PointerSourceState {
            x: 0.0,
            y: 0.0,
            buttons: 0,
            down_targets: HashMap::new(),
            hover_target: None,
            pointer_type: PointerType::Mouse,
            pointer_id: 1,
            touch_target: None,
            touch_pan: None,
            light_dismiss_popover_press: None,
            light_dismiss_dialog_press: None,
            last_used: 0,
        }
    }
}

/// The executor's page-scoped state.
#[derive(Debug)]
struct ExecutorState {
    pointers: HashMap<String, PointerSourceState>,
    modifiers: ModifierState,
    focused: Option<NodeId>,
    /// Dialog focus stack for this page (w82 fix, B2): the
    /// testdriver mirror of the window's tab-keyed stack, but
    /// per page — executor state is already keyed by page
    /// context, so no tab key is needed here. Entries are
    /// `(dialog, previously-focused element at show time,
    /// was-modal)`, replayed from the document's dialog journal
    /// by [`TestDriver::sync_modal_focus`]. Persisted across
    /// drains through the thread-local hand-off, like the rest
    /// of this state.
    modal_focus_stack: Vec<(NodeId, Option<NodeId>, bool)>,
    next_pointer_id: i32,
    /// Monotonic counter stamped onto a pointer state every time
    /// it is stored back after an action: defines the "most
    /// recently active pointer" without relying on `HashMap`
    /// iteration order (which is nondeterministic).
    pointer_use_seq: u64,
}

impl Default for ExecutorState {
    fn default() -> Self {
        ExecutorState {
            pointers: HashMap::new(),
            modifiers: ModifierState::default(),
            focused: None,
            modal_focus_stack: Vec::new(),
            // Pointer ids start at 1, like the mouse stream's
            // `PointerData::mouse_at` convention of a nonzero id.
            next_pointer_id: 1,
            pointer_use_seq: 0,
        }
    }
}

impl ExecutorState {
    /// The current position of the pointer a wheel action with
    /// `origin: "pointer"` resolves against: the most recently
    /// active pointer (highest `last_used` stamp; ties — pointers
    /// minted but never stored back — break toward the smallest
    /// source key, so the choice is fully deterministic). `None`
    /// when no pointer source exists yet; the caller then falls
    /// back to the viewport origin handling used elsewhere.
    fn wheel_pointer_position(&self) -> Option<(f64, f64)> {
        let mut best: Option<(&String, &PointerSourceState)> = None;
        for (key, st) in &self.pointers {
            let wins = match best {
                None => true,
                Some((bk, bst)) => {
                    st.last_used > bst.last_used || (st.last_used == bst.last_used && key < bk)
                }
            };
            if wins {
                best = Some((key, st));
            }
        }
        best.map(|(_, st)| (st.x, st.y))
    }
}

impl TestDriver {
    /// The executor-state key for a pointer source: its WebDriver id,
    /// or a single fallback bucket when the payload omits the id.
    fn pointer_key(source: &ActionSource) -> String {
        if source.id.is_empty() {
            "pointer".to_string()
        } else {
            source.id.clone()
        }
    }

    /// The persistent state for one pointer source, minting it on
    /// first use (cloned out: dispatches re-enter the page VM, so
    /// the state is snapshotted, dispatched against, and stored back
    /// rather than held borrowed across a dispatch).
    fn pointer_snapshot(&mut self, source: &ActionSource) -> PointerSourceState {
        let key = Self::pointer_key(source);
        if let Some(st) = self.actions.pointers.get(&key) {
            return st.clone();
        }
        let pointer_type = PointerType::parse(source.pointer_type.as_deref().unwrap_or("mouse"));
        let pointer_id = self.actions.next_pointer_id;
        self.actions.next_pointer_id += 1;
        let st = PointerSourceState {
            pointer_type,
            pointer_id,
            ..PointerSourceState::default()
        };
        self.actions.pointers.insert(key, st.clone());
        st
    }

    /// Store a snapshotted pointer state back, stamping it as the
    /// most recently active pointer (the deterministic target of
    /// wheel actions with `origin: "pointer"`).
    fn store_pointer(&mut self, source: &ActionSource, mut st: PointerSourceState) {
        self.actions.pointer_use_seq += 1;
        st.last_used = self.actions.pointer_use_seq;
        self.actions.pointers.insert(Self::pointer_key(source), st);
    }

    /// The key-event target: the executor-focused node, falling back
    /// to `<body>`, the document element, and the document root, in
    /// that order.
    fn key_target(tab: &Tab, focused: Option<NodeId>) -> Option<NodeId> {
        if focused.is_some() {
            return focused;
        }
        let doc = tab.document_ref()?;
        let body = doc.get_elements_by_tag_name("body").into_iter().next();
        body.or_else(|| doc.document_element())
            .or_else(|| Some(doc.root()))
    }

    /// The synthesized-event target when the hit test misses every
    /// box (the point lies outside the viewport content): the body /
    /// document element / root, mirroring [`key_target`].
    fn fallback_target(tab: &Tab) -> Option<NodeId> {
        let doc = tab.document_ref()?;
        let body = doc.get_elements_by_tag_name("body").into_iter().next();
        body.or_else(|| doc.document_element())
            .or_else(|| Some(doc.root()))
    }

    /// Build the [`PointerData`] for a pointer event at the source
    /// state's current position: button state, movement delta, held
    /// modifiers, and the pointer's identity. `offset_x`/`offset_y`
    /// are filled by the caller (which knows the target's rect).
    fn pointer_data(
        &self,
        tab: &Tab,
        st: &PointerSourceState,
        button: i32,
        movement: (f64, f64),
    ) -> PointerData {
        let (sx, sy) = tab.scroll().offset();
        let mut data = PointerData::mouse_at(st.x, st.y);
        data.page_x = st.x + sx as f64;
        data.page_y = st.y + sy as f64;
        data.button = button;
        data.buttons = st.buttons;
        data.movement_x = movement.0;
        data.movement_y = movement.1;
        self.actions.modifiers.apply_to_pointer(&mut data);
        data.pointer_id = st.pointer_id;
        data.pointer_type = st.pointer_type;
        // Devices without pressure support report 0.5 while a button
        // is held (the `PointerEvent.pressure` default for contact).
        data.pressure = if st.buttons != 0 { 0.5 } else { 0.0 };
        data
    }

    /// [`pointer_data`](Self::pointer_data) plus target-relative
    /// `offset_x`/`offset_y` from the target's client rect.
    fn pointer_data_at(
        &mut self,
        tab: &Tab,
        st: &PointerSourceState,
        target: NodeId,
        button: i32,
        movement: (f64, f64),
    ) -> PointerData {
        let mut data = self.pointer_data(tab, st, button, movement);
        if let Some(rect) = self.dispatch.element_client_rect(tab, target) {
            data.offset_x = st.x - rect.x as f64;
            data.offset_y = st.y - rect.y as f64;
        }
        data
    }

    /// One contact point for `Vm::dispatch_touch`: the touch's
    /// identifier and original target, carrying the source's current
    /// position (page coords include the viewport offset).
    fn touch_point(tab: &Tab, st: &PointerSourceState, target: NodeId) -> Touch {
        let (sx, sy) = tab.scroll().offset();
        Touch::new(
            st.pointer_id,
            Some(target),
            st.x,
            st.y,
            st.x,
            st.y,
            st.x + sx as f64,
            st.y + sy as f64,
        )
    }

    /// Anchor container for a touch-drag default pan: the hit node's
    /// innermost scroll container (an overlay intercepting the hit
    /// intercepts the gesture too), falling back to the geometric
    /// innermost container when the document is gone — the same rule
    /// [`PageEventDispatch::dispatch_wheel_at_point`](crate::script::PageEventDispatch::dispatch_wheel_at_point)
    /// uses for wheel scrolls.
    fn touch_pan_anchor(tab: &Tab, x: f32, y: f32, target: NodeId) -> usize {
        match tab.document_ref() {
            Some(doc) => tab.scroll_tree.container_for_node(target, &doc),
            None => tab.scroll_tree.innermost_at(x, y),
        }
    }

    /// Fire one pointer/mouse event at an explicit target: a small
    /// wrapper over the dispatch layer that keeps the executor's
    /// call sites one line. Returns whether a non-passive listener
    /// prevented the default — callers gate the compatibility
    /// mouse events and the focus move on it (a cancelled
    /// `pointerdown` suppresses the compat `mousedown`; a cancelled
    /// compat `mousedown` suppresses the focus move that normally
    /// follows it).
    #[allow(clippy::too_many_arguments)]
    fn fire_pointer(
        &mut self,
        tab: &mut Tab,
        st: &PointerSourceState,
        target: NodeId,
        event_type: &str,
        kind: PointerMouseKind,
        button: i32,
        movement: (f64, f64),
    ) -> bool {
        let data = self.pointer_data_at(tab, st, target, button, movement);
        self.dispatch
            .dispatch_pointer_mouse_at(tab, target, event_type, kind, data)
    }

    /// Move the focus like the real input path does (see the
    /// focus-lite model in the module docs): `blur`/`focusout` on the
    /// old node, then `focus`/`focusin` on the new one. Fires nothing
    /// when the focus does not change.
    ///
    /// GAP (documented, not silent): `document.activeElement` is not
    /// updated here — the bindings expose no `activeElement`
    /// accessor or setter at all (no script-visible path exists to
    /// update cheaply), so page script cannot observe executor
    /// focus through `activeElement` yet; the `focus`/`blur` event
    /// stream above is the observable channel.
    fn set_focus(&mut self, tab: &mut Tab, new: Option<NodeId>) {
        let old = self.actions.focused;
        if old == new {
            return;
        }
        if let Some(old) = old {
            self.dispatch.dispatch_focus_event(tab, old, "blur");
            self.dispatch.dispatch_focus_event(tab, old, "focusout");
        }
        self.actions.focused = new;
        // Mirror for the show-time snapshot (see the window's
        // `mirror_focus`): a script showModal after this focus
        // move must snapshot `new`, not a stale value.
        if let Some(installed) = tab.installed_mut() {
            installed.document().borrow_mut().set_host_focus(new);
        }
        if let Some(n) = new {
            self.dispatch.dispatch_focus_event(tab, n, "focus");
            self.dispatch.dispatch_focus_event(tab, n, "focusin");
        }
    }

    /// `Tab` (and Shift+`Tab`) move focus one step through the
    /// sequential focus order, wrapping at the ends, and reveal the
    /// newly focused element (`Nearest`/`Nearest`, instant — the same
    /// path the windowed `apply_focus_nav` uses). Shift is read from
    /// the held modifier set, like the windowed path reads the
    /// modifier state. With a top-layer scope open (a modal dialog,
    /// or a popover above it — see
    /// [`crate::window::top_layer_focus_scope`]), Tab traps inside
    /// that scope, mirroring the windowed path; a scope with no
    /// focusables falls back to the scope itself.
    fn move_focus_tab(&mut self, tab: &mut Tab, backward: bool) {
        let order = match tab.document_ref() {
            Some(doc) => match crate::window::top_layer_focus_scope(&doc) {
                Some(scope) => {
                    let mut candidates = vec![scope];
                    candidates.extend(doc.descendants(scope));
                    let order = sequential_focus_order_over(&doc, candidates);
                    if order.is_empty() { vec![scope] } else { order }
                }
                None => sequential_focus_order(&doc),
            },
            None => return,
        };
        if order.is_empty() {
            return;
        }
        let position = self
            .actions
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
            // Focus outside the order (nothing focused, or the focused
            // node is not in the order): Tab starts at the first
            // focusable, Shift+Tab at the last, like browsers.
            None => order[if backward { order.len() - 1 } else { 0 }],
        };
        if Some(next) == self.actions.focused {
            return;
        }
        self.set_focus(tab, Some(next));
        // Reveal like `apply_focus_nav`: skip on dirty layout (the
        // scroll would resolve against a stale tree).
        if !tab.dirty.layout {
            navigation::scroll_to_node(
                tab,
                next,
                RevealAlign::Nearest,
                RevealAlign::Nearest,
                bowser_js::ScrollBehavior::Auto,
            );
        }
    }

    /// Dialog focus sync for the headless/WPT path (w82 fix,
    /// B2): the executor mirror of
    /// `WindowApp::sync_modal_focus`, over executor focus
    /// (`self.actions.focused`) instead of window focus. It
    /// drains the document's dialog journal and replays it in
    /// order (show+close and close+re-show inside one drain
    /// interval, show-time snapshots from the bindings), then
    /// diffs the live state as a fallback for journal-less
    /// paths (`open`-attribute removal, element removal).
    /// Modal closes restore unconditionally (validity-gated);
    /// non-modal closes restore only when focus sits inside
    /// the dialog or nowhere — the same rules as the windowed
    /// path. Non-modal shows record an entry but never move
    /// focus (spec: `show()` does not move focus). Focus moves
    /// go through [`Self::set_focus`], so the page observes
    /// `blur`/`focusout`/`focus`/`focusin` for them.
    /// Called at the end of every action sequence and at the
    /// end of every testdriver drain (which covers script-driven
    /// shows/closes that ran during a request's completion
    /// pump). NOT called from `execute_escape`: that arm is
    /// dismissal-owned and still clears focus inside the
    /// dismissed dialog; the sequence/drain sync that follows
    /// restores it from the stack — so executor Escape restores
    /// *after* its `close` event, like the JS `close()` path,
    /// not before it like windowed Escape. That residual
    /// ordering difference is documented here, not hidden.
    fn sync_modal_focus(&mut self, tab: &mut Tab) {
        let events: Vec<DialogFocusEvent> = match tab.installed_mut() {
            Some(installed) => installed.document().borrow_mut().take_dialog_focus_events(),
            None => Vec::new(),
        };
        // Same staleness rule as the windowed sync (see
        // there): a show after a close replayed in this batch
        // uses the post-restore focus, not its snapshot.
        let mut saw_replayed_close = false;
        for event in events {
            match event {
                DialogFocusEvent::Shown { node, modal } => {
                    if self
                        .actions
                        .modal_focus_stack
                        .iter()
                        .any(|(stacked, _, _)| *stacked == node)
                    {
                        continue;
                    }
                    let recorded: Option<Option<NodeId>> = match tab.installed_mut() {
                        Some(installed) => installed
                            .document()
                            .borrow_mut()
                            .take_dialog_show_previous(node),
                        None => None,
                    };
                    let cur = self.actions.focused;
                    let inside_stacked = match cur {
                        Some(focused) => {
                            let doc_in = |tab: &Tab| -> bool {
                                match tab.document_ref() {
                                    Some(doc) => self.actions.modal_focus_stack.iter().any(
                                        |(stacked, _, _)| {
                                            focused == *stacked
                                                || doc.descendants(*stacked).contains(&focused)
                                        },
                                    ),
                                    None => false,
                                }
                            };
                            doc_in(tab)
                        }
                        None => false,
                    };
                    let (previous, move_in) = if saw_replayed_close {
                        (cur, modal)
                    } else {
                        match recorded {
                            Some(rec) => {
                                if cur == rec || inside_stacked {
                                    (if inside_stacked { cur } else { rec }, modal)
                                } else {
                                    (rec, false)
                                }
                            }
                            None => (cur, modal),
                        }
                    };
                    let move_in = move_in
                        && !match tab.document_ref() {
                            Some(doc) => {
                                crate::window::popover_scope_blocks_dialog_move(&doc, node)
                            }
                            None => false,
                        };
                    self.actions.modal_focus_stack.push((node, previous, modal));
                    if move_in {
                        let target = match tab.document_ref() {
                            Some(doc) => crate::window::dialog_initial_focus(&doc, node),
                            None => node,
                        };
                        self.set_focus(tab, Some(target));
                    }
                }
                DialogFocusEvent::Closed { node } => {
                    let popped = self
                        .actions
                        .modal_focus_stack
                        .iter()
                        .position(|(stacked, _, _)| *stacked == node)
                        .map(|pos| self.actions.modal_focus_stack.remove(pos));
                    if let Some((_, previous, was_modal)) = popped {
                        saw_replayed_close = true;
                        self.restore_focus_after_close(tab, node, previous, was_modal);
                    }
                }
            }
        }
        // Diff fallback (journal-less paths): modal entries pop
        // when no longer modal-open; non-modal entries pop when
        // no longer open or no longer attached.
        let current_modal: Vec<NodeId> = match tab.document_ref() {
            Some(doc) => doc
                .top_layer()
                .iter()
                .copied()
                .filter(|&node| doc.is_dialog_open(node) && doc.is_dialog_modal(node))
                .collect(),
            None => Vec::new(),
        };
        let snapshot = self.actions.modal_focus_stack.clone();
        for (dialog, previous, was_modal) in snapshot.iter().rev() {
            let still_open = match tab.document_ref() {
                Some(doc) => {
                    if *was_modal {
                        current_modal.contains(dialog)
                    } else {
                        doc.is_dialog_open(*dialog)
                            && doc.propagation_path(*dialog).last() == Some(&doc.root())
                    }
                }
                None => false,
            };
            if still_open {
                continue;
            }
            self.actions
                .modal_focus_stack
                .retain(|(stacked, _, _)| stacked != dialog);
            self.restore_focus_after_close(tab, *dialog, *previous, *was_modal);
        }
        // Modal adoption (journal-less shows: markup-open,
        // parse shim): record current focus and move in.
        for dialog in current_modal {
            if self
                .actions
                .modal_focus_stack
                .iter()
                .any(|(stacked, _, _)| *stacked == dialog)
            {
                continue;
            }
            let previous = self.actions.focused;
            self.actions
                .modal_focus_stack
                .push((dialog, previous, true));
            let blocked = match tab.document_ref() {
                Some(doc) => crate::window::popover_scope_blocks_dialog_move(&doc, dialog),
                None => false,
            };
            if blocked {
                continue;
            }
            let target = match tab.document_ref() {
                Some(doc) => crate::window::dialog_initial_focus(&doc, dialog),
                None => dialog,
            };
            self.set_focus(tab, Some(target));
        }
    }

    /// Restore executor focus after `dialog` closed (w82 fix):
    /// modal restores unconditionally, non-modal only when
    /// focus sits inside the dialog or nowhere; an invalid
    /// previous (detached, or inside a closed dialog) clears
    /// focus only when it sits inside or nowhere, and never
    /// clobbers focus living elsewhere.
    fn restore_focus_after_close(
        &mut self,
        tab: &mut Tab,
        dialog: NodeId,
        previous: Option<NodeId>,
        was_modal: bool,
    ) {
        // w85fix BLOCKER A, second wipe path: the sequence
        // -end sync runs after the dialog arm, so a `close`
        // listener that re-showed a nested popover has
        // already run by the time this restore replays the
        // close journal entry. Focus inside that live
        // popover is legitimate and must survive the
        // restore too — without this, the dialog-arm guard
        // is undone here (the previous focus inside the
        // closed dialog reads invalid, so the restore
        // clears it). The windowed path needs no twin:
        // its restore runs before `dispatch_close`, so a
        // listener re-show cannot precede it.
        if self.focus_inside_live_popover(tab, dialog) {
            return;
        }
        let (inside, valid_previous) = match tab.document_ref() {
            Some(doc) => {
                let inside = match self.actions.focused {
                    None => true,
                    Some(focused) => {
                        focused == dialog || doc.descendants(dialog).contains(&focused)
                    }
                };
                let valid = match previous {
                    Some(prev) => {
                        let attached =
                            prev == doc.root() || doc.ancestors(prev).contains(&doc.root());
                        let mut chain = vec![prev];
                        chain.extend(doc.ancestors(prev));
                        attached
                            && !chain.iter().any(|&node| {
                                doc.is_dialog_element(node) && !doc.is_dialog_open(node)
                            })
                    }
                    None => false,
                };
                (inside, valid)
            }
            None => (false, false),
        };
        match previous.filter(|_| valid_previous) {
            Some(target) if was_modal || inside => {
                self.set_focus(tab, Some(target));
            }
            _ => {
                if inside {
                    self.set_focus(tab, None);
                }
            }
        }
    }

    /// Run one action sequence: tick-synchronized execution across
    /// every source. Sources shorter than the longest idle on the
    /// extra ticks (WebDriver pads them with implicit pauses). The
    /// whole sequence is a trusted gesture: transient user activation
    /// is granted on completion, like phase 1's click/keys/bless.
    fn execute_action_sequence(
        &mut self,
        tab: &mut Tab,
        sequence: TestDriverActionSequence,
    ) -> Result<(), String> {
        let ticks = sequence
            .sources
            .iter()
            .map(|s| s.actions.len())
            .max()
            .unwrap_or(0);
        for tick in 0..ticks {
            for source in &sequence.sources {
                if let Some(item) = source.actions.get(tick) {
                    self.execute_action_item(tab, source, item)?;
                }
                // No action at this tick: the implicit pause.
            }
        }
        Self::grant_activation(tab);
        // Dialog focus sync (w82 fix B2): script-driven
        // shows/closes during the sequence (a click handler
        // calling showModal, an Escape close) land here.
        self.sync_modal_focus(tab);
        Ok(())
    }

    /// Run one action item of one source at one tick. Items whose
    /// kind does not belong to their source's kind (a `keyDown` on a
    /// pointer source, a `scroll` on a key source) are ignored —
    /// WebDriver would reject the payload, but the bridge already
    /// validated the shape; the executor fails closed on the
    /// individual item instead.
    fn execute_action_item(
        &mut self,
        tab: &mut Tab,
        source: &ActionSource,
        item: &ActionItem,
    ) -> Result<(), String> {
        let kind_ok = match source.kind {
            ActionSourceKind::None => matches!(item.kind, ActionItemKind::Pause),
            ActionSourceKind::Key => matches!(
                item.kind,
                ActionItemKind::KeyDown | ActionItemKind::KeyUp | ActionItemKind::Pause
            ),
            ActionSourceKind::Pointer => matches!(
                item.kind,
                ActionItemKind::PointerDown
                    | ActionItemKind::PointerUp
                    | ActionItemKind::PointerMove
                    | ActionItemKind::PointerCancel
                    | ActionItemKind::Pause
            ),
            ActionSourceKind::Wheel => {
                matches!(item.kind, ActionItemKind::Scroll | ActionItemKind::Pause)
            }
        };
        if !kind_ok {
            return Ok(());
        }
        match item.kind {
            // Headless policy: the executor runs between event-loop
            // tasks and never sleeps, so a pause is a pure no-op tick
            // — its duration is clamped to 0.
            ActionItemKind::Pause => Ok(()),
            ActionItemKind::KeyDown => self.execute_key_item(tab, item, true),
            ActionItemKind::KeyUp => self.execute_key_item(tab, item, false),
            ActionItemKind::PointerMove => self.execute_pointer_move(tab, source, item),
            ActionItemKind::PointerDown => {
                self.execute_pointer_down(tab, source, item);
                Ok(())
            }
            ActionItemKind::PointerUp => {
                self.execute_pointer_up(tab, source, item);
                Ok(())
            }
            ActionItemKind::PointerCancel => {
                self.execute_pointer_cancel(tab, source);
                Ok(())
            }
            ActionItemKind::Scroll => self.execute_scroll_item(tab, source, item),
        }
    }

    /// One key action: dispatch the `keydown`/`keyup` at the
    /// executor-focused node, tracking the modifier state machine.
    /// `Tab` on an un-prevented `keydown` moves focus through the
    /// sequential focus order (Shift held: backward).
    fn execute_key_item(
        &mut self,
        tab: &mut Tab,
        item: &ActionItem,
        down: bool,
    ) -> Result<(), String> {
        let raw = item.value.as_deref().unwrap_or("");
        if raw.is_empty() {
            return Ok(());
        }
        let key = normalize_action_key(raw);
        // Modifier state first: a modifier's own `keydown` reports it
        // held, its `keyup` reports it released (browser behavior).
        self.actions.modifiers.set_key(&key, down);
        // The stored focus is trusted only while the node is still
        // in the document: a node detached since it took focus (a
        // handler removed it, a navigation rebuilt the tree) resets
        // focus to the body fallback instead of dispatching keys
        // into a detached subtree.
        if let Some(focused) = self.actions.focused {
            let attached = tab
                .document_ref()
                .map(|doc| node_in_document(&doc, focused))
                .unwrap_or(false);
            if !attached {
                self.actions.focused = None;
            }
        }
        let target = Self::key_target(tab, self.actions.focused);
        let Some(target) = target else {
            return Ok(());
        };
        let mods = self.actions.modifiers.as_key_modifiers();
        let prevented = if down {
            self.dispatch.dispatch_key_down_at(tab, target, &key, mods)
        } else {
            self.dispatch.dispatch_key_up_at(tab, target, &key, mods)
        };
        // Default actions, mirroring the windowed key path: an
        // un-prevented Escape dismisses the topmost modal dialog /
        // auto popover, and an un-prevented Tab moves focus through
        // the top-layer focus scope when one is open (see
        // [`Self::move_focus_tab`]).
        if down && !prevented {
            if key == "Escape" {
                self.execute_escape(tab);
            } else if key == "Tab" {
                self.move_focus_tab(tab, self.actions.modifiers.shift);
            }
        }
        Ok(())
    }

    /// Escape default action for the executor, mirroring
    /// `WindowApp::handle_escape`: dismiss the top-layer member
    /// itself (a modal dialog via cancel-then-close, an `auto`
    /// popover via hide), with the `cancel` / `close` / `toggle`
    /// hooks routed through the [`JsDispatch`] trait — production
    /// delivery runs through the `script.rs` implementation
    /// (wired, not deferred), exactly as in the windowed path.
    /// Dialog closes and popover hides demote through the
    /// deferred helpers in [`crate::window`] (w82 fixpaths B3),
    /// so a stored exit duration renders the transition.
    /// Returns true when a dismissible target existed, even when
    /// a canceled `cancel` kept the dialog open. Executor focus
    /// inside a dismissed target is cleared, like the windowed
    /// path's `clear_focus_inside`.
    fn execute_escape(&mut self, tab: &mut Tab) -> bool {
        let target = match tab.document_ref() {
            Some(doc) => doc.topmost_escape_target(),
            None => None,
        };
        let Some(target) = target else {
            return false;
        };
        match target {
            EscapeTarget::ModalDialog(dialog) | EscapeTarget::NonModalDialog(dialog) => {
                let canceled = JsDispatch::dispatch_cancel(&mut self.dispatch, tab, dialog);
                if canceled {
                    return true;
                }
                // Dual snapshot after `cancel`, before the
                // close (see
                // [`crate::window::fire_dual_popover_toggle_after_close`]).
                let was_dual = tab
                    .document_ref()
                    .map(|doc| crate::window::is_live_dual_popover(&doc, dialog))
                    .unwrap_or(false);
                let closed = match tab.installed_mut() {
                    Some(installed) => crate::window::deferred_close_dialog(
                        &mut installed.document().borrow_mut(),
                        dialog,
                    ),
                    None => false,
                };
                if !closed {
                    return true;
                }
                JsDispatch::dispatch_close(&mut self.dispatch, tab, dialog);
                crate::window::fire_dual_popover_toggle_after_close(
                    &mut self.dispatch,
                    tab,
                    dialog,
                    was_dual,
                );
                self.clear_focus_inside(tab, dialog);
                true
            }
            EscapeTarget::AutoPopover(pop) => {
                let hidden = match tab.installed_mut() {
                    Some(installed) => {
                        let mut doc = installed.document().borrow_mut();
                        let was_open = doc.is_popover_open(pop);
                        was_open && crate::window::deferred_hide_popover(&mut doc, pop)
                    }
                    None => false,
                };
                if !hidden {
                    return true;
                }
                JsDispatch::dispatch_toggle(&mut self.dispatch, tab, pop);
                self.clear_focus_inside(tab, pop);
                true
            }
        }
    }

    /// Clear executor focus when it sits on `closed` or inside it
    /// (dialog/popover just dismissed): focus must not linger on
    /// hidden content. Focus elsewhere is untouched.
    fn clear_focus_inside(&mut self, tab: &mut Tab, closed: NodeId) {
        let inside = match (self.actions.focused, tab.document_ref()) {
            (Some(focused), Some(doc)) => {
                focused == closed || doc.descendants(closed).contains(&focused)
            }
            _ => false,
        };
        if inside {
            self.set_focus(tab, None);
        }
    }

    /// Whether executor focus currently sits inside a
    /// *live* top-layer popover other than `closed` itself
    /// (w85fix BLOCKER A guard, shared by the dialog-arm
    /// clear and by [`Self::restore_focus_after_close`]).
    /// A popover nested inside the dialog, re-shown by the
    /// dialog's `close` listener with focus moved into it,
    /// is a DOM descendant of the dialog, so a plain
    /// inside-dialog clear (or a close restore, which runs
    /// at sequence end, after the listener) would wipe
    /// that legitimate focus. The check walks the focused
    /// node's ancestor-or-self chain for a live top-layer
    /// member carrying a `popover` attribute: live
    /// membership (not `is_popover_open`, which counts
    /// retentions) is what makes the focus legitimate, and
    /// excluding `closed` itself keeps a dual dialog's own
    /// close clearing focus on the dialog.
    fn focus_inside_live_popover(&self, tab: &Tab, closed: NodeId) -> bool {
        match (self.actions.focused, tab.document_ref()) {
            (Some(focused), Some(doc)) => {
                let mut current = Some(focused);
                while let Some(node) = current {
                    if node != closed
                        && doc.top_layer().contains(&node)
                        && doc.popover_state(node).is_some()
                    {
                        return true;
                    }
                    current = doc.parent(node);
                }
                false
            }
            _ => false,
        }
    }

    /// Dialog-close focus clear with the w85fix BLOCKER A
    /// guard: like [`Self::clear_focus_inside`], except it
    /// never wipes focus sitting inside a *live* top-layer
    /// popover other than `closed` itself (see
    /// [`Self::focus_inside_live_popover`]). This guard was
    /// chosen over moving the clear before `dispatch_close`
    /// (the windowed ordering): moving it would still leave
    /// the testdriver's standing divergence — the close
    /// listener would observe cleared focus instead of the
    /// windowed path's restored focus — while the guard
    /// fixes the wipe without touching that divergence.
    fn clear_focus_inside_dialog_guarded(&mut self, tab: &mut Tab, closed: NodeId) {
        if !self.focus_inside_live_popover(tab, closed) {
            self.clear_focus_inside(tab, closed);
        }
    }

    /// Resolve the target point of a `pointerMove` action in client
    /// coordinates: element origins use the layout tree's in-view
    /// center (the drain runs after layout, headless and windowed).
    fn resolve_move_point(
        &mut self,
        tab: &Tab,
        source: &ActionSource,
        item: &ActionItem,
    ) -> Result<(f64, f64), String> {
        let st = self.pointer_snapshot(source);
        let center = match item.origin {
            ActionOrigin::Element(node) => self.dispatch.element_client_center(tab, node),
            _ => None,
        };
        let (x, y) = (item.x.unwrap_or(0.0), item.y.unwrap_or(0.0));
        resolve_action_point(&item.origin, x, y, (st.x, st.y), center)
    }

    /// One `pointerMove` action: move the pointer, firing the boundary
    /// chains when the hover target changes and then paired
    /// `pointermove`/`mousemove` (plus `touchmove` for a touch source
    /// whose contact is down).
    fn execute_pointer_move(
        &mut self,
        tab: &mut Tab,
        source: &ActionSource,
        item: &ActionItem,
    ) -> Result<(), String> {
        let (nx, ny) = self.resolve_move_point(tab, source, item)?;
        let mut st = self.pointer_snapshot(source);
        let movement = (nx - st.x, ny - st.y);
        st.x = nx;
        st.y = ny;
        let hit = self.dispatch.hit_test_at_point(tab, nx as f32, ny as f32);
        let new_target = hit.or_else(|| Self::fallback_target(tab));
        if new_target != st.hover_target {
            self.fire_boundary_events(tab, &st, st.hover_target, new_target, movement);
            st.hover_target = new_target;
        }
        if let Some(t) = new_target {
            let move_prevented = self.fire_pointer(
                tab,
                &st,
                t,
                "pointermove",
                PointerMouseKind::Pointer,
                -1,
                movement,
            );
            // A cancelled `pointermove` suppresses the compatibility
            // `mousemove`, like the down/up compat chains below.
            if !move_prevented {
                self.fire_pointer(
                    tab,
                    &st,
                    t,
                    "mousemove",
                    PointerMouseKind::Mouse,
                    -1,
                    movement,
                );
            }
        }
        if st.pointer_type == PointerType::Touch
            && let Some(touch_target) = st.touch_target
        {
            let touch = Self::touch_point(tab, &st, touch_target);
            self.dispatch
                .dispatch_touch_event(tab, TouchPhase::Move, &[touch]);
        }
        // Touch-drag default pan: the drag delta, inverted from finger
        // motion like the windowed `on_touch` drag arm (finger down
        // means the offset decreases), scrolls the container the
        // unprevented `touchstart` anchored at. Contact-down only; a
        // prevented touchstart never anchored (`touch_pan` is `None`).
        // Zero deltas skip the chain, mirroring `TouchScrollState`.
        if st.pointer_type == PointerType::Touch
            && st.buttons != 0
            && let Some(container) = st.touch_pan
        {
            let (dx, dy) = (-movement.0 as f32, -movement.1 as f32);
            crate::window::touch_drag_pan(&mut self.dispatch, tab, container, dx, dy);
        }
        self.store_pointer(source, st);
        Ok(())
    }

    /// One `pointerDown` action: `pointerdown` + `touchstart` (touch
    /// sources) + `mousedown`, in that order, then focus moves to the
    /// press target. The pressed button's bit is set before the
    /// events, so they report it held. A primary-button press
    /// swallowed by modal inertness ([`crate::window::modal_press_blocked`])
    /// dispatches nothing and records no press target, so its
    /// release synthesizes no click — mirroring the windowed path —
    /// but it still records its light-dismiss ancestors (w85
    /// exemption: inertness swallows activation, not
    /// light-dismiss recording), so the release can dismiss a
    /// `closedby=any` dialog or an auto popover above the modal.
    /// A primary-button press otherwise records its light-dismiss
    /// ancestors (HTML §6.12.2, §4.11.5) before any event fires;
    /// the release dismisses only on matching ancestors (see
    /// [`Self::execute_pointer_up`]).
    fn execute_pointer_down(&mut self, tab: &mut Tab, source: &ActionSource, item: &ActionItem) {
        let mut st = self.pointer_snapshot(source);
        let button = item.button.unwrap_or(0);
        let mut target = self
            .dispatch
            .hit_test_at_point(tab, st.x as f32, st.y as f32);
        if target.is_none() {
            target = Self::fallback_target(tab);
        }
        st.buttons |= button_bit(button);
        if button == 0 {
            let blocked = match tab.document_ref() {
                Some(doc) => crate::window::modal_press_blocked(&doc, target),
                None => false,
            };
            // Blocked or not, the light-dismiss ancestors are
            // recorded (w85 exemption above); only event
            // dispatch and click synthesis are swallowed by
            // the block.
            st.light_dismiss_popover_press = tab
                .document_ref()
                .map(|doc| crate::window::light_dismiss_popover_press_ancestor(&doc, target));
            st.light_dismiss_dialog_press = tab.document_ref().map(|_| {
                crate::window::dialog_light_dismiss_press_ancestor(
                    tab,
                    target,
                    st.x as f32,
                    st.y as f32,
                )
            });
            if blocked {
                self.store_pointer(source, st);
                return;
            }
        }
        if let Some(t) = target {
            st.down_targets.insert(button, t);
        }
        if st.pointer_type == PointerType::Touch {
            st.touch_target = target;
        }
        if let Some(t) = target {
            let down_prevented = self.fire_pointer(
                tab,
                &st,
                t,
                "pointerdown",
                PointerMouseKind::Pointer,
                button as i32,
                (0.0, 0.0),
            );
            let mut touch_prevented = false;
            if st.pointer_type == PointerType::Touch {
                let touch = Self::touch_point(tab, &st, t);
                touch_prevented =
                    self.dispatch
                        .dispatch_touch_event(tab, TouchPhase::Start, &[touch]);
                // Touch-drag default pan: an unprevented `touchstart`
                // anchors the gesture at the hit container; later
                // `pointerMove`s with the contact down pan from this
                // anchor (see `execute_pointer_move`). A prevented
                // touchstart means the page took the gesture — no pan.
                if !touch_prevented {
                    st.touch_pan = Some(Self::touch_pan_anchor(tab, st.x as f32, st.y as f32, t));
                }
            }
            // Compatibility chain (Pointer Events § compatibility
            // mapping with the legacy mouse events): a cancelled
            // `pointerdown` — or a cancelled `touchstart`, whose
            // preventDefault likewise suppresses the compat mouse
            // events — suppresses the compat `mousedown` entirely.
            if !down_prevented && !touch_prevented {
                let mouse_prevented = self.fire_pointer(
                    tab,
                    &st,
                    t,
                    "mousedown",
                    PointerMouseKind::Mouse,
                    button as i32,
                    (0.0, 0.0),
                );
                // Focus-lite: a press moves focus (the windowed input
                // path focuses the hit node on press too) — but only
                // when the compat `mousedown` was not cancelled, and
                // only onto a focusable node (a plain `<div>` without
                // `tabindex` never takes focus; the previously
                // focused element keeps it).
                if !mouse_prevented
                    && self.actions.focused != Some(t)
                    && tab
                        .document_ref()
                        .map(|doc| td_is_focusable(&doc, t))
                        .unwrap_or(false)
                {
                    self.set_focus(tab, Some(t));
                }
            }
        }
        self.store_pointer(source, st);
    }

    /// One `pointerUp` action: `pointerup` + `touchend` (touch
    /// sources) + `mouseup`, then the synthesized post-click: a
    /// `click` (primary button) or `auxclick` (middle/back/forward)
    /// at the nearest common inclusive ancestor of the press and
    /// release targets, or a `contextmenu` when the released button
    /// is the secondary one and its press landed on this target.
    /// The released button's bit is cleared before the events, so
    /// they report it released (UI Events `buttons`).
    fn execute_pointer_up(&mut self, tab: &mut Tab, source: &ActionSource, item: &ActionItem) {
        let mut st = self.pointer_snapshot(source);
        let button = item.button.unwrap_or(0);
        // The light-dismiss records are taken here so every release
        // clears them, whether or not the release dismisses.
        let light_dismiss_popover_press = st.light_dismiss_popover_press.take();
        let light_dismiss_dialog_press = st.light_dismiss_dialog_press.take();
        let hit = self
            .dispatch
            .hit_test_at_point(tab, st.x as f32, st.y as f32);
        // A touch contact's target is fixed at the touchstart point
        // (Touch Events): the release events go there, not to the
        // point's current hit.
        let mut target = if st.pointer_type == PointerType::Touch {
            st.touch_target.or(hit)
        } else {
            hit
        };
        if target.is_none() {
            target = Self::fallback_target(tab);
        }
        st.buttons &= !button_bit(button);
        let down_target = st.down_targets.remove(&button);
        if let Some(t) = target {
            let up_prevented = self.fire_pointer(
                tab,
                &st,
                t,
                "pointerup",
                PointerMouseKind::Pointer,
                button as i32,
                (0.0, 0.0),
            );
            let mut touch_prevented = false;
            if st.pointer_type == PointerType::Touch {
                let touch = Self::touch_point(tab, &st, t);
                touch_prevented =
                    self.dispatch
                        .dispatch_touch_event(tab, TouchPhase::End, &[touch]);
            }
            // A cancelled `pointerup` (or `touchend`) suppresses the
            // whole compatibility chain below — `mouseup` and the
            // synthesized `click`/`auxclick`/`contextmenu` — exactly
            // as a cancelled `pointerdown` suppresses `mousedown`.
            // Light-dismiss is part of that chain: a cancelled
            // `pointerup` dismisses nothing either.
            if !up_prevented && !touch_prevented {
                self.fire_pointer(
                    tab,
                    &st,
                    t,
                    "mouseup",
                    PointerMouseKind::Mouse,
                    button as i32,
                    (0.0, 0.0),
                );
                // Light-dismiss on release (HTML §6.12.2,
                // §4.11.5; w85 hide-all-until), before the
                // synthesized click: popovers first, then
                // dialogs. A kind dismisses only when the
                // release resolves to the same ancestor the
                // press recorded — a press outside dragged
                // inside the popover, or onto its invoker,
                // before release dismisses nothing, and the
                // click below then runs normally.
                if button == 0 {
                    // Deferred popover focus clears, mirroring
                    // the windowed `light_dismiss_for_release`
                    // (w85fix): collected here, applied only
                    // after the dialog arm below, so a dialog
                    // `cancel`/`close` listener observes the
                    // pre-dismissal focus instead of focus
                    // already cleared out of the popover.
                    let mut deferred_focus_clears: Vec<NodeId> = Vec::new();
                    if let Some(press_ancestor) = light_dismiss_popover_press {
                        let release_ancestor = tab.document_ref().map(|doc| {
                            crate::window::light_dismiss_popover_press_ancestor(&doc, Some(t))
                        });
                        if release_ancestor == Some(press_ancestor) {
                            let victims = tab
                                .document_ref()
                                .map(|doc| doc.light_dismiss_popovers_until(press_ancestor))
                                .unwrap_or_default();
                            for pop in victims {
                                // w82 fixpaths M4: gate on
                                // was-open, like the windowed
                                // Escape arm — `hide_popover`
                                // alone reports `Ok` for an
                                // already-hidden popover, which
                                // used to fire a spurious
                                // `toggle`. Deferred hide (B3)
                                // renders the exit transition
                                // when one was stored.
                                let hidden = match tab.installed_mut() {
                                    Some(installed) => {
                                        let mut doc = installed.document().borrow_mut();
                                        let was_open = doc.is_popover_open(pop);
                                        was_open
                                            && crate::window::deferred_hide_popover(&mut doc, pop)
                                    }
                                    None => false,
                                };
                                if hidden {
                                    JsDispatch::dispatch_toggle(&mut self.dispatch, tab, pop);
                                    // Deferred, not inline: the
                                    // dialog arm's `cancel`/`close`
                                    // listener must still observe
                                    // the pre-dismissal focus.
                                    deferred_focus_clears.push(pop);
                                }
                            }
                        }
                    }
                    if let Some(press_ancestor) = light_dismiss_dialog_press {
                        let release_ancestor = tab.document_ref().map(|_| {
                            crate::window::dialog_light_dismiss_press_ancestor(
                                tab,
                                Some(t),
                                st.x as f32,
                                st.y as f32,
                            )
                        });
                        if release_ancestor == Some(press_ancestor) {
                            let candidate = tab.document_ref().and_then(|doc| {
                                let effective = crate::window::dialog_effective_hit(
                                    tab,
                                    Some(t),
                                    st.x as f32,
                                    st.y as f32,
                                );
                                doc.light_dismiss_dialog_candidate(effective)
                            });
                            if let Some(dialog) = candidate {
                                let canceled =
                                    JsDispatch::dispatch_cancel(&mut self.dispatch, tab, dialog);
                                if !canceled {
                                    // Dual snapshot after
                                    // `cancel`, before the close
                                    // (see the helper docs).
                                    let was_dual = tab
                                        .document_ref()
                                        .map(|doc| {
                                            crate::window::is_live_dual_popover(&doc, dialog)
                                        })
                                        .unwrap_or(false);
                                    let closed = match tab.installed_mut() {
                                        Some(installed) => crate::window::deferred_close_dialog(
                                            &mut installed.document().borrow_mut(),
                                            dialog,
                                        ),
                                        None => false,
                                    };
                                    if closed {
                                        JsDispatch::dispatch_close(&mut self.dispatch, tab, dialog);
                                        crate::window::fire_dual_popover_toggle_after_close(
                                            &mut self.dispatch,
                                            tab,
                                            dialog,
                                            was_dual,
                                        );
                                        // Focus divergence, accepted
                                        // as the testdriver's
                                        // standing divergence (cf.
                                        // the Escape note at
                                        // [`Self::sync_modal_focus`]
                                        // and the submit note in
                                        // [`Self::execute_click`]):
                                        // this arm only clears
                                        // focus inside the dialog,
                                        // and the restore happens
                                        // later at the
                                        // sequence/drain
                                        // `sync_modal_focus`, so
                                        // a `close` listener here
                                        // observes cleared focus,
                                        // where the windowed path
                                        // restores focus before
                                        // firing `close` and its
                                        // listener observes the
                                        // restored focus. Do not
                                        // "fix" this arm in
                                        // isolation: the restore
                                        // belongs to the sync,
                                        // which is what replays
                                        // the dialog journal in
                                        // order.
                                        // w85fix BLOCKER A: guarded, not
                                        // plain — a `close` listener
                                        // that re-showed a popover
                                        // nested in this dialog and
                                        // moved focus into it must
                                        // keep that focus (see
                                        // `clear_focus_inside_dialog_guarded`).
                                        self.clear_focus_inside_dialog_guarded(tab, dialog);
                                    }
                                }
                            }
                        }
                    }
                    // Dialog arm done: now clear focus inside
                    // the dismissed popovers. A dialog restore
                    // that already moved focus out of a popover
                    // makes its clear a no-op, and the clear
                    // runs even when the dialog's `cancel` was
                    // prevented (the dialog arm then ran
                    // nothing above, but the popover still
                    // hid). The re-show check is live
                    // top-layer membership, as in the windowed
                    // path: a dialog listener that re-showed a
                    // dismissed popover keeps its focus, while
                    // an exit-retained victim — still
                    // `is_popover_open`, but not live — clears,
                    // since no later clear runs at retention
                    // expiry.
                    for pop in deferred_focus_clears {
                        let re_shown = tab
                            .document_ref()
                            .map(|doc| doc.top_layer().contains(&pop))
                            .unwrap_or(false);
                        if !re_shown {
                            self.clear_focus_inside(tab, pop);
                        }
                    }
                }
                match button {
                    // Secondary button: `contextmenu`, and only when
                    // a matching `pointerDown` landed on this same
                    // target — a bare `pointerUp` (no recorded press)
                    // never produces one.
                    2 => {
                        if down_target == Some(t) {
                            self.fire_pointer(
                                tab,
                                &st,
                                t,
                                "contextmenu",
                                PointerMouseKind::Mouse,
                                2,
                                (0.0, 0.0),
                            );
                        }
                    }
                    // Primary -> `click`; middle/back/forward ->
                    // `auxclick` only (never `click`). Both target
                    // the nearest common inclusive ancestor of the
                    // press and release targets, so a press on one
                    // child and a release on its sibling clicks
                    // their parent.
                    0 | 1 | 3 | 4 => {
                        if let Some(down) = down_target {
                            let click_target = tab
                                .document_ref()
                                .and_then(|doc| nearest_common_inclusive_ancestor(&doc, down, t))
                                .or(if down == t { Some(t) } else { None });
                            if let Some(ct) = click_target {
                                let click_prevented = self.fire_pointer(
                                    tab,
                                    &st,
                                    ct,
                                    if button == 0 { "click" } else { "auxclick" },
                                    PointerMouseKind::Mouse,
                                    button as i32,
                                    (0.0, 0.0),
                                );
                                // Primary-click default action, gated
                                // on the click not being prevented:
                                // popover-target activation, with the
                                // invoker resolved from the click
                                // target's ancestors (the target is
                                // the deepest hit box) and the
                                // `toggle` hook on a visibility
                                // change — the same default action
                                // the windowed and phase-1 click
                                // paths run. `auxclick` activates
                                // nothing.
                                if button == 0 && !click_prevented {
                                    // Full form, like `execute_click`
                                    // and the windowed click path: a
                                    // show branch auto-closes other
                                    // `auto` popovers, and every
                                    // victim fires its `toggle`
                                    // before the primary target's.
                                    // The primary-only form used to
                                    // drop those victim toggles
                                    // (w82 final-review fix).
                                    let (changed, auto_closed) = match tab.installed_mut() {
                                        Some(installed) => {
                                            let mut doc = installed.document().borrow_mut();
                                            match crate::window::resolve_popover_invoker(&doc, ct) {
                                                Some(invoker) => {
                                                    crate::window::activate_popover_target_full(
                                                        &mut doc, invoker, ct,
                                                    )
                                                }
                                                None => (None, Vec::new()),
                                            }
                                        }
                                        None => (None, Vec::new()),
                                    };
                                    tab.sync_dirty_flags();
                                    for victim in auto_closed {
                                        JsDispatch::dispatch_toggle(
                                            &mut self.dispatch,
                                            tab,
                                            victim,
                                        );
                                    }
                                    if let Some(target) = changed {
                                        JsDispatch::dispatch_toggle(
                                            &mut self.dispatch,
                                            tab,
                                            target,
                                        );
                                    }
                                    // Submit-button activation
                                    // (w85fix): the same shared
                                    // helper the windowed
                                    // `on_click` and
                                    // `execute_click` paths run,
                                    // after popover activation
                                    // and gated on the unprevented
                                    // click — a real pointer
                                    // sequence on a
                                    // `method="dialog"` submit
                                    // used to synthesize the
                                    // click and run popover
                                    // activation, then stop, so
                                    // it fired no `submit` and
                                    // closed nothing. A handled
                                    // submit is terminal: no
                                    // other default action
                                    // follows it in this arm.
                                    let (_handled, closed_dialog, was_dual) =
                                        crate::window::submit_activation_close(
                                            &mut self.dispatch,
                                            tab,
                                            ct,
                                        );
                                    if let Some(dialog) = closed_dialog {
                                        JsDispatch::dispatch_close(
                                            &mut self.dispatch,
                                            tab,
                                            dialog,
                                        );
                                        crate::window::fire_dual_popover_toggle_after_close(
                                            &mut self.dispatch,
                                            tab,
                                            dialog,
                                            was_dual,
                                        );
                                        self.clear_focus_inside(tab, dialog);
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        if st.pointer_type == PointerType::Touch {
            st.touch_target = None;
            // The gesture ended: drop the pan anchor. (No fling
            // inertia: the headless pump has no animation ticks to run
            // it on, and the executor's synchronous ticks leave the
            // release-velocity estimate at zero anyway.)
            st.touch_pan = None;
        }
        self.store_pointer(source, st);
    }

    /// One `pointerCancel` action: `pointercancel` at the pointer's
    /// current target (the touch's original target for touch), then
    /// the whole contact state is dropped. There is no
    /// `lostpointercapture` (pointer capture is deferred), and no
    /// `touchcancel` (the touch dispatcher has no cancel phase yet).
    fn execute_pointer_cancel(&mut self, tab: &mut Tab, source: &ActionSource) {
        let mut st = self.pointer_snapshot(source);
        let mut target = if st.pointer_type == PointerType::Touch {
            st.touch_target.or(st.hover_target)
        } else {
            st.hover_target.or_else(|| {
                self.dispatch
                    .hit_test_at_point(tab, st.x as f32, st.y as f32)
            })
        };
        if target.is_none() {
            target = Self::fallback_target(tab);
        }
        if let Some(t) = target {
            self.fire_pointer(
                tab,
                &st,
                t,
                "pointercancel",
                PointerMouseKind::Pointer,
                -1,
                (0.0, 0.0),
            );
        }
        st.buttons = 0;
        st.down_targets.clear();
        st.touch_target = None;
        st.touch_pan = None;
        st.light_dismiss_popover_press = None;
        st.light_dismiss_dialog_press = None;
        self.store_pointer(source, st);
    }

    /// One `scroll` action: resolve the point (viewport = absolute,
    /// element = in-view center + offset, pointer = the first
    /// pointer source's position + offset), dispatch the `wheel`
    /// event, and scroll by the deltas unless prevented (see
    /// [`PageEventDispatch::dispatch_wheel_at_point`]).
    fn execute_scroll_item(
        &mut self,
        tab: &mut Tab,
        _source: &ActionSource,
        item: &ActionItem,
    ) -> Result<(), String> {
        let (x, y) = (item.x.unwrap_or(0.0), item.y.unwrap_or(0.0));
        // `origin: "pointer"` resolves against the most recently
        // active pointer, tracked explicitly by store order — never
        // `HashMap` iteration order, which is nondeterministic with
        // two or more pointers. With no pointer yet, fall back to
        // the viewport origin handling (point (0, 0) + offset).
        let current = self.actions.wheel_pointer_position().unwrap_or((0.0, 0.0));
        let center = match item.origin {
            ActionOrigin::Element(node) => self.dispatch.element_client_center(tab, node),
            _ => None,
        };
        let (px, py) = resolve_action_point(&item.origin, x, y, current, center)?;
        let (dx, dy) = (item.delta_x.unwrap_or(0.0), item.delta_y.unwrap_or(0.0));
        self.dispatch
            .dispatch_wheel_at_point(tab, px as f32, py as f32, dx, dy);
        Ok(())
    }

    /// The `*over`/`*enter` boundary chains when the hover target
    /// changes from `old` to `new`, plus the matching leave chains
    /// off the old one. Order (one defensible UI-Events reading):
    /// `pointerout`, the `pointerleave` chain (target upward, stopping
    /// below the common ancestor), `mouseout`, the `mouseleave` chain,
    /// then `pointerover`, the `pointerenter` chain (ancestor-down to
    /// the target), `mouseover`, the `mouseenter` chain. The chains
    /// run over element nodes only; `bubbles`/`cancelable` come from
    /// the usual type table inside `dispatch_pointer_mouse`.
    fn fire_boundary_events(
        &mut self,
        tab: &mut Tab,
        st: &PointerSourceState,
        old: Option<NodeId>,
        new: Option<NodeId>,
        movement: (f64, f64),
    ) {
        let (old_path, new_path) = match tab.document_ref() {
            Some(doc) => (
                old.map(|n| element_path(&doc, n)).unwrap_or_default(),
                new.map(|n| element_path(&doc, n)).unwrap_or_default(),
            ),
            None => (Vec::new(), Vec::new()),
        };
        // The first common ancestor of the two paths (target-upward):
        // no leave/enter event fires for it or above it.
        let common = old_path.iter().find(|n| new_path.contains(n)).copied();
        let take_until_common = |path: &[NodeId]| {
            path.iter()
                .take_while(|n| Some(**n) != common)
                .copied()
                .collect::<Vec<_>>()
        };
        if let Some(o) = old {
            self.fire_pointer(
                tab,
                st,
                o,
                "pointerout",
                PointerMouseKind::Pointer,
                -1,
                movement,
            );
            for n in take_until_common(&old_path) {
                self.fire_pointer(
                    tab,
                    st,
                    n,
                    "pointerleave",
                    PointerMouseKind::Pointer,
                    -1,
                    movement,
                );
            }
            self.fire_pointer(
                tab,
                st,
                o,
                "mouseout",
                PointerMouseKind::Mouse,
                -1,
                movement,
            );
            for n in take_until_common(&old_path) {
                self.fire_pointer(
                    tab,
                    st,
                    n,
                    "mouseleave",
                    PointerMouseKind::Mouse,
                    -1,
                    movement,
                );
            }
        }
        if let Some(nw) = new {
            self.fire_pointer(
                tab,
                st,
                nw,
                "pointerover",
                PointerMouseKind::Pointer,
                -1,
                movement,
            );
            // Enter chain: ancestor-down, so reverse the target-upward slice.
            let enter: Vec<NodeId> = take_until_common(&new_path);
            for n in enter.iter().rev() {
                self.fire_pointer(
                    tab,
                    st,
                    *n,
                    "pointerenter",
                    PointerMouseKind::Pointer,
                    -1,
                    movement,
                );
            }
            self.fire_pointer(
                tab,
                st,
                nw,
                "mouseover",
                PointerMouseKind::Mouse,
                -1,
                movement,
            );
            for n in enter.iter().rev() {
                self.fire_pointer(
                    tab,
                    st,
                    *n,
                    "mouseenter",
                    PointerMouseKind::Mouse,
                    -1,
                    movement,
                );
            }
        }
    }
}

/// The element-only propagation path (target first, then ancestors),
/// for the executor's `*enter`/`*leave` chains: boundary events never
/// fire at the document node itself.
fn element_path(doc: &Document, node: NodeId) -> Vec<NodeId> {
    doc.propagation_path(node)
        .into_iter()
        .filter(|n| matches!(doc.kind(*n), NodeKind::Element(_)))
        .collect()
}

/// Parsed `tabindex` attribute: `None` when absent or unparseable
/// (unparseable behaves as absent per HTML). Mirrors the windowed
/// input path's focus navigation (`window::tab_index`), which lives
/// in a module this executor cannot reach.
fn td_tab_index(doc: &Document, node: NodeId) -> Option<i32> {
    doc.get_attribute(node, "tabindex")?.trim().parse().ok()
}

/// Whether `node` participates in sequential focus navigation (the
/// WHATWG HTML "focusable" subset that Tab reaches). Mirrors
/// `window::is_sequentially_focusable` — same subset, kept in sync
/// by the shared comment, not shared code.
fn td_is_sequentially_focusable(doc: &Document, node: NodeId) -> bool {
    let NodeKind::Element(data) = doc.kind(node) else {
        return false;
    };
    let tag = &data.tag_name;
    if tag == "input"
        && matches!(doc.get_attribute(node, "type"), Some(t) if t.eq_ignore_ascii_case("hidden"))
    {
        return false;
    }
    if let Some(index) = td_tab_index(doc, node) {
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
/// the zero/`tabindex`-less focusables in document order. Mirrors
/// `window::focus_order`.
fn sequential_focus_order(doc: &Document) -> Vec<NodeId> {
    sequential_focus_order_over(doc, doc.descendants(doc.root()))
}

/// [`sequential_focus_order`] over an explicit candidate list,
/// already in document order: the top-layer Tab trap passes the
/// scope element followed by its descendants, so Tab cycles inside
/// the scope instead of escaping to the page behind it.
fn sequential_focus_order_over(doc: &Document, candidates: Vec<NodeId>) -> Vec<NodeId> {
    let mut positive: Vec<(i32, usize, NodeId)> = Vec::new();
    let mut zero: Vec<NodeId> = Vec::new();
    for (i, node) in candidates.iter().enumerate() {
        if !td_is_sequentially_focusable(doc, *node) {
            continue;
        }
        match td_tab_index(doc, *node) {
            Some(index) if index > 0 => positive.push((index, i, *node)),
            _ => zero.push(*node),
        }
    }
    positive.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    positive
        .into_iter()
        .map(|(_, _, node)| node)
        .chain(zero)
        .collect()
}

// ---------------------------------------------------------------------------
// Embedder-side drain
// ---------------------------------------------------------------------------

/// Executes queued [`bowser_js::TestDriverRequest`]s against the live
/// page. Owns the [`PageEventDispatch`] the actions dispatch through.
pub struct TestDriver {
    dispatch: PageEventDispatch,
    /// WebDriver Actions executor state, persistent across
    /// `action_sequence` calls (and, through
    /// [`drain_testdriver_requests`]' thread-local hand-off, across
    /// drains of one page).
    actions: ExecutorState,
}

impl TestDriver {
    /// A fresh executor.
    pub fn new() -> Self {
        TestDriver {
            dispatch: PageEventDispatch::new(),
            actions: ExecutorState::default(),
        }
    }

    /// Run one queued request: scroll into view, dispatch, grant user
    /// activation, then complete the shim's promise.
    fn execute(&mut self, tab: &mut Tab, req: bowser_js::TestDriverRequest) {
        let result = match req.action {
            bowser_js::TestDriverAction::Click { node } => self.execute_click(tab, node),
            bowser_js::TestDriverAction::SendKeys { node, keys } => {
                self.execute_send_keys(tab, node, &keys)
            }
            bowser_js::TestDriverAction::Bless => self.execute_bless(tab),
            bowser_js::TestDriverAction::ActionSequence { sequence } => {
                self.execute_action_sequence(tab, sequence)
            }
        };
        self.complete(tab, req.token, result);
    }

    /// Scroll the element into view (upstream's `block: "end"`,
    /// `inline: "nearest"`), skipping the scroll on dirty layout like
    /// the `scrollIntoView` drain does: resolving against a stale tree
    /// would mis-scroll.
    fn scroll_into_view(&mut self, tab: &mut Tab, node: NodeId) {
        if tab.dirty.layout {
            return;
        }
        navigation::scroll_to_node(
            tab,
            node,
            RevealAlign::End,
            RevealAlign::Nearest,
            bowser_js::ScrollBehavior::Auto,
        );
    }

    /// Grant transient user activation on the page, like a real trusted
    /// gesture would.
    fn grant_activation(tab: &mut Tab) {
        if let Some(installed) = tab.installed_mut() {
            installed.page_vm_mut().grant_transient_activation();
        }
    }

    fn execute_click(&mut self, tab: &mut Tab, node: NodeId) -> Result<(), String> {
        self.scroll_into_view(tab, node);
        // The same primitive real mouse input uses: inline `onclick`
        // plus a trusted bubbling, cancelable `click` dispatch. Its
        // return value is the default-action gate: a prevented click
        // runs no default action, popover-target activation included
        // (see [`crate::window::activate_popover_target`]).
        let prevented = self.dispatch.dispatch_click(tab, node);
        if !prevented {
            let (changed, auto_closed) = match tab.installed_mut() {
                Some(installed) => {
                    let mut doc = installed.document().borrow_mut();
                    match crate::window::resolve_popover_invoker(&doc, node) {
                        Some(invoker) => {
                            crate::window::activate_popover_target_full(&mut doc, invoker, node)
                        }
                        None => (None, Vec::new()),
                    }
                }
                None => (None, Vec::new()),
            };
            tab.sync_dirty_flags();
            for victim in auto_closed {
                JsDispatch::dispatch_toggle(&mut self.dispatch, tab, victim);
            }
            if let Some(target) = changed {
                JsDispatch::dispatch_toggle(&mut self.dispatch, tab, target);
            }
            // Submit-button activation (w85, shared helper
            // with the windowed `on_click` branch — see
            // [`crate::window::submit_activation_close`]):
            // validation vacuous, then `submit` at the form,
            // then the dialog close, minus the window's modal
            // focus stack: the executor clears focus inside
            // the closed dialog instead (see
            // [`Self::execute_escape`]). A handled submit
            // never falls through to anything else; a
            // non-dialog method fires `submit` and stops
            // (navigation deferred).
            let (_handled, closed_dialog, was_dual) = crate::window::submit_activation_close(
                &mut self.dispatch,
                tab,
                node,
            );
            if let Some(dialog) = closed_dialog {
                JsDispatch::dispatch_close(&mut self.dispatch, tab, dialog);
                crate::window::fire_dual_popover_toggle_after_close(
                    &mut self.dispatch,
                    tab,
                    dialog,
                    was_dual,
                );
                self.clear_focus_inside(tab, dialog);
            }
        }
        Self::grant_activation(tab);
        Ok(())
    }

    fn execute_send_keys(&mut self, tab: &mut Tab, node: NodeId, keys: &str) -> Result<(), String> {
        self.scroll_into_view(tab, node);
        for key in split_testdriver_keys(keys) {
            // Split the pair so the Escape default action can be
            // gated on the `keydown` alone, like the windowed key
            // path (a prevented `keyup` must not suppress it, and a
            // prevented `keydown` must). Other keys behave exactly
            // as `dispatch_key_at` did: one down plus one up.
            let prevented_down =
                self.dispatch
                    .dispatch_key_down_at(tab, node, &key, Default::default());
            self.dispatch
                .dispatch_key_up_at(tab, node, &key, Default::default());
            if key == "Escape" && !prevented_down {
                self.execute_escape(tab);
            }
        }
        Self::grant_activation(tab);
        Ok(())
    }

    fn execute_bless(&mut self, tab: &mut Tab) -> Result<(), String> {
        // Upstream's button-click mechanism is bypassed by the shim (see
        // the module docs): the vendor grants the activation directly.
        Self::grant_activation(tab);
        Ok(())
    }

    /// Settle the shim's promise for `token`: call the shim-installed
    /// `__complete(token, ok, err)`. Defensive: when the shim never ran
    /// (the bridge natives were called directly), `__complete` is
    /// absent and the eval reports `false` instead of throwing.
    fn complete(&mut self, tab: &mut Tab, token: u64, result: Result<(), String>) {
        let Some(installed) = tab.installed_mut() else {
            return;
        };
        // Disjoint field borrow: the DOM handle and the realm, with no
        // `Rc` clone — the tab keeps the sole strong owner.
        let (doc, page_vm) = installed.document_and_vm_mut();
        // Reentrancy gate (see `InstalledDocument::assert_no_live_borrow`):
        // the completion eval re-enters the page VM, whose natives
        // re-borrow the document through the registry `Weak` — so no
        // `Ref`/`RefMut` may be alive across it.
        crate::tab::InstalledDocument::assert_no_live_borrow(doc);
        let (ok, err) = match result {
            Ok(()) => ("true", String::new()),
            Err(e) => ("false", e),
        };
        let err = crate::script::js_string_escape(&err);
        let source = format!(
            "(function(){{var b=window.__bowser_testdriver;\
             return (b&&typeof b.__complete===\"function\")\
             ?b.__complete({token},{ok},\"{err}\"):false;}})()"
        );
        if let Err(e) = page_vm.eval(&source) {
            // No node to attribute to; the completion is driver
            // bookkeeping, so record it node-less rather than dropping.
            page_vm.record_error(crate::script::ScriptError::task_error(e));
        }
        page_vm.pump();
        tab.sync_dirty_flags();
    }
}

impl Default for TestDriver {
    fn default() -> Self {
        Self::new()
    }
}

/// Drain every pending `__bowser_testdriver` request queued by page
/// script and execute it against the live tab. Called between
/// event-loop tasks (the headless pump and the windowed
/// `pump_scripts`); each dispatch pumps and folds DOM dirt itself,
/// like any other UI-event dispatch. A no-op when the page queued
/// nothing.
///
/// Automation gating: a request executes only while the bridge is
/// armed (see [`testdriver_bridge_armed`]). Requests queued by direct
/// page-script calls to `window.__bowser_testdriver` — without the
/// vendor shim — are dropped without dispatching and without granting
/// user activation; the refusal is recorded, not silent.
pub fn drain_testdriver_requests(tab: &mut Tab) {
    let requests = tab
        .page_vm()
        .and_then(|p| p.web_api_context())
        .map(bowser_js::take_testdriver_requests)
        .unwrap_or_default();
    if requests.is_empty() {
        return;
    }
    if !testdriver_bridge_armed(tab) {
        record_unarmed_testdriver_drop(tab, requests.len());
        return;
    }
    let mut driver = TestDriver::new();
    // Actions executor state persists across drains of one page: a
    // sequence that holds Shift in one drain and releases it in the
    // next must keep its modifier state (see the executor docs). The
    // hand-off is keyed by the page's web-API context, which is minted
    // fresh per document, so state can never leak across pages.
    let ctx = tab.page_vm().and_then(|p| p.web_api_context());
    if let Some(ctx) = ctx {
        driver.actions = take_persisted_action_state(ctx);
    }
    for req in requests {
        driver.execute(tab, req);
    }
    // Dialog focus sync (w82 fix B2): covers script-driven
    // shows/closes from any request in this drain, including
    // shows that ran during a completion pump. Idempotent
    // with the per-sequence sync above.
    driver.sync_modal_focus(tab);
    if let Some(ctx) = ctx {
        store_persisted_action_state(ctx, std::mem::take(&mut driver.actions));
    }
}

thread_local! {
    /// The [`ExecutorState`]s parked between drains, keyed by the
    /// web-API context each belongs to. A map (not a single slot):
    /// pages interleave on the event-loop thread, and draining one
    /// page must not discard another page's parked state — a
    /// single slot dropped page A's held modifiers the moment page
    /// B drained.
    static PERSISTED_ACTION_STATE: RefCell<HashMap<u32, ExecutorState>> =
        RefCell::new(HashMap::new());
}

/// Take the parked executor state for page context `ctx` (a fresh
/// state when none is parked for it). Other contexts' parked
/// states are untouched.
fn take_persisted_action_state(ctx: u32) -> ExecutorState {
    PERSISTED_ACTION_STATE.with(|slot| slot.borrow_mut().remove(&ctx).unwrap_or_default())
}

/// Park `state` for the next drain of page context `ctx`.
fn store_persisted_action_state(ctx: u32, state: ExecutorState) {
    PERSISTED_ACTION_STATE.with(|slot| {
        slot.borrow_mut().insert(ctx, state);
    });
}

/// Whether the testdriver bridge is currently armed: the vendor shim
/// has a bridge call in flight, observed as the shim-installed
/// `__complete` callback on `window.__bowser_testdriver`. The shim
/// installs `__complete` synchronously before invoking a bridge native
/// and removes it when its last in-flight call settles, so presence
/// means "this page's vendor shim mediated the queued requests". A
/// page that never ran the shim fails this check, and its direct
/// bridge calls are dropped by [`drain_testdriver_requests`] —
/// unarmed pages cannot self-bless.
///
/// Deliberate limit, documented not silent: a page that replicates the
/// shim's protocol (installing its own `__complete`) is
/// indistinguishable from the shim — but at that point it IS running
/// the vendor contract, which upstream WPT treats as trusted vendor
/// code. The gate stops the trivial forgery (a bare
/// `__bowser_testdriver.click(el)` from page script), not a page
/// deliberately reimplementing testdriver-vendor.js.
fn testdriver_bridge_armed(tab: &mut Tab) -> bool {
    let Some(installed) = tab.installed_mut() else {
        return false;
    };
    // Disjoint field borrow: the DOM handle and the realm, with no
    // `Rc` clone — the tab keeps the sole strong owner.
    let (doc, page_vm) = installed.document_and_vm_mut();
    // Reentrancy gate (see `InstalledDocument::assert_no_live_borrow`):
    // the check eval re-enters the page VM, whose natives re-borrow
    // the document through the registry `Weak` — so no `Ref`/`RefMut`
    // may be alive across it.
    crate::tab::InstalledDocument::assert_no_live_borrow(doc);
    matches!(
        page_vm.eval(
            "(function(){var b=window.__bowser_testdriver;\
             return b&&typeof b.__complete===\"function\";})()"
        ),
        Ok(bowser_js::Value::Bool(true))
    )
}

/// Record one diagnostic for a batch of refused unarmed bridge
/// requests: node-less, like the completion bookkeeping in
/// [`TestDriver::complete`].
fn record_unarmed_testdriver_drop(tab: &mut Tab, count: usize) {
    let Some(installed) = tab.installed_mut() else {
        return;
    };
    let (doc, page_vm) = installed.document_and_vm_mut();
    crate::tab::InstalledDocument::assert_no_live_borrow(doc);
    page_vm.record_error(crate::script::ScriptError {
        node: None,
        url: None,
        kind: crate::script::ScriptErrorKind::Eval(format!(
            "dropped {count} unarmed __bowser_testdriver request(s): \
             the testdriver vendor shim never ran on this page"
        )),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_key_codepoints_map_to_ui_events_key_names() {
        // The example from the task plus the full private-use range.
        assert_eq!(webdriver_key_value('\u{E004}'), Some("Tab"));
        assert_eq!(webdriver_key_value('\u{E00C}'), Some("Escape"));
        assert_eq!(webdriver_key_value('\u{E007}'), Some("Enter"));
        assert_eq!(webdriver_key_value('\u{E006}'), Some("Return"));
        assert_eq!(webdriver_key_value('\u{E00D}'), Some(" "));
        assert_eq!(webdriver_key_value('\u{E000}'), Some("Unidentified"));
        assert_eq!(webdriver_key_value('\u{E03C}'), Some("F12"));
        assert_eq!(webdriver_key_value('\u{E05D}'), Some("Delete"));
        // Ordinary characters are not special keys.
        assert_eq!(webdriver_key_value('a'), None);
        assert_eq!(webdriver_key_value(' '), None);
        // Unlisted private-use codepoints are not special keys either.
        assert_eq!(webdriver_key_value('\u{E02A}'), None);
    }

    #[test]
    fn split_testdriver_keys_maps_specials_and_passes_through_chars() {
        assert_eq!(
            split_testdriver_keys("a\u{E004}b"),
            vec!["a".to_string(), "Tab".to_string(), "b".to_string()]
        );
        assert_eq!(
            split_testdriver_keys("\u{E00C}"),
            vec!["Escape".to_string()]
        );
        assert!(split_testdriver_keys("").is_empty());
    }

    #[test]
    fn vendor_shim_only_appends_to_testdriver_js() {
        let body = b"upstream();".to_vec();
        let out = maybe_append_vendor_shim("/resources/testdriver.js", body.clone());
        assert!(out.len() > body.len());
        assert!(out.starts_with(&body));
        assert!(out.ends_with(VENDOR_SHIM_JS.as_bytes()));
        // Non-canonical spellings of the same file gain the shim too:
        // the fixture server canonicalizes these to the same file, and
        // serving them without the shim would hang upstream automation
        // (in_automation stays false).
        for path in [
            "/resources/./testdriver.js",
            "/resources//testdriver.js",
            "//resources/testdriver.js",
            "/./resources/testdriver.js",
            "/resources/../resources/testdriver.js",
        ] {
            let out = maybe_append_vendor_shim(path, body.clone());
            assert!(
                out.ends_with(VENDOR_SHIM_JS.as_bytes()),
                "path {path} must gain the shim"
            );
        }
        // Query strings are stripped by the caller; other paths pass
        // through byte-identical. A trailing slash does not serve the
        // file, so it must not gain the shim either.
        for path in [
            "/resources/testdriver-vendor.js",
            "/resources/testharness.js",
            "/resources/testdriver.js/",
            "/testdriver.js",
        ] {
            assert_eq!(
                maybe_append_vendor_shim(path, body.clone()),
                body,
                "path {path} must not gain the shim"
            );
        }
    }

    #[test]
    fn vendor_shim_overrides_the_vendor_contract_points() {
        // The shim must set in_automation and override exactly the
        // actions the bridge implements (phase 1's click/send_keys/
        // bless plus phase 2's action_sequence). The bridge is
        // resolved lazily (it appears only after the post-parse
        // production install, while the shim runs mid-parse), and
        // promises complete through the shim-installed __complete.
        for needle in [
            "in_automation=true",
            "tdi.click=function",
            "tdi.send_keys=function",
            "tdi.bless=function",
            "tdi.action_sequence=function",
            "test_driver.bless=function",
            "function bridge()",
            "b.__complete=complete",
            "window.__bowser_testdriver",
            // Arming: the shim opts into a native __arm() when present
            // and keeps __complete installed exactly while its own
            // calls are in flight (the embedder's armed signal).
            "__arm",
            "inFlight",
            "uninstallComplete",
            // Token-0 contract: a 0/missing token rejects instead of
            // parking the promise in `pending` forever.
            "no token",
        ] {
            assert!(VENDOR_SHIM_JS.contains(needle), "shim missing {needle:?}");
        }
    }

    #[test]
    fn shared_js_string_escaper_covers_u2028_u2029() {
        // The shared escaper (crate::script::js_string_escape) is what
        // TestDriver::complete() splices into its eval: U+2028/U+2029
        // must come out escaped because bowser_js's lexer rejects them
        // raw inside string literals (LexError::UnterminatedString),
        // which would otherwise leave the shim's promise unsettled.
        use crate::script::js_string_escape;
        assert_eq!(js_string_escape("a\"b\\c"), "a\\\"b\\\\c");
        assert_eq!(js_string_escape("x\u{0}y"), "x\\u0000y");
        assert_eq!(js_string_escape("\t"), "\\t");
        assert_eq!(js_string_escape("\u{2028}"), "\\u2028");
        assert_eq!(js_string_escape("a\u{2029}b"), "a\\u2029b");
    }

    #[test]
    fn drain_refuses_unarmed_direct_bridge_calls() {
        use std::cell::RefCell;
        use std::rc::Rc;

        // A page with web APIs but no vendor shim: the bridge is
        // installed on the global, but no shim call is in flight, so a
        // direct `__bowser_testdriver` call must be refused — no trusted
        // dispatch, no user-activation grant, queue consumed.
        let doc = Rc::new(RefCell::new(
            bowser_dom::Document::parse_html(
                b"<html><body><button id=\"t\">go</button></body></html>",
            )
            .0,
        ));
        let base = bowser_url::parse("http://example.com/page.html", None).expect("test url");
        let page =
            crate::script::PageVm::new_for_document(doc.clone(), &base, "TestBrowser/9.9", &[])
                .expect("install web apis");
        let mut tab = crate::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
        tab.install_document(doc, page);

        let vm = tab.page_vm_mut().expect("page vm");
        vm.eval("bridgePresent = typeof window.__bowser_testdriver === 'object';")
            .expect("probe bridge");
        let present = vm.eval("bridgePresent").expect("read bridgePresent");
        assert!(
            matches!(present, bowser_js::Value::Bool(true)),
            "bridge is installed on every page global, got: {present:?}"
        );
        // Direct call, bypassing the shim: queues a request with no
        // shim-installed `__complete`.
        vm.eval(
            "clicked = 0; \
             document.getElementById('t').addEventListener('click', function() { clicked = 1; }); \
             window.__bowser_testdriver.click(document.getElementById('t'));",
        )
        .expect("direct bridge call");

        drain_testdriver_requests(&mut tab);

        let vm = tab.page_vm_mut().expect("page vm");
        let clicked = vm.eval("clicked").expect("read clicked");
        assert!(
            matches!(clicked, bowser_js::Value::Int(0)),
            "unarmed direct click must not dispatch, got: {clicked:?}"
        );
        // The refused requests are consumed, not left to accumulate.
        let rest = tab
            .page_vm()
            .and_then(|p| p.web_api_context())
            .map(bowser_js::take_testdriver_requests)
            .unwrap_or_default();
        assert!(rest.is_empty(), "refused requests must be consumed");
        // And the refusal is recorded, not silent.
        let errors = tab.page_vm_mut().expect("page vm").report().errors.len();
        assert!(errors > 0, "unarmed bridge use must record a diagnostic");
    }

    // -----------------------------------------------------------------
    // Actions executor (phase 2)
    // -----------------------------------------------------------------

    /// One layout box for the executor fixtures.
    fn actions_box(
        rect: (f32, f32, f32, f32),
        node: Option<NodeId>,
        children: Vec<bowser_layout::LayoutBox>,
    ) -> bowser_layout::LayoutBox {
        bowser_layout::LayoutBox {
            rect: bowser_layout::Rect {
                x: rect.0,
                y: rect.1,
                w: rect.2,
                h: rect.3,
            },
            children,
            box_type: bowser_layout::BoxType::Block,
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

    /// A tab with web APIs installed and a hand-built layout tree:
    /// `#a` covers (0,0)-(100,50), `#b` covers (100,0)-(200,50) in an
    /// 800x600 viewport. The page's scripts never ran; tests register
    /// listeners by eval.
    fn actions_tab(html: &str) -> (Tab, NodeId, NodeId) {
        use std::cell::RefCell;
        use std::rc::Rc;
        let doc = Rc::new(RefCell::new(
            bowser_dom::Document::parse_html(html.as_bytes()).0,
        ));
        let (a, b) = {
            let d = doc.borrow();
            (
                d.get_element_by_id("a").expect("a element"),
                d.get_element_by_id("b").expect("b element"),
            )
        };
        let base = bowser_url::parse("http://example.com/page.html", None).expect("test url");
        let page =
            crate::script::PageVm::new_for_document(doc.clone(), &base, "TestBrowser/9.9", &[])
                .expect("install web apis");
        let mut tab = crate::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
        tab.install_document(doc, page);
        tab.layout_root = Some(Rc::new(actions_box(
            (0.0, 0.0, 800.0, 600.0),
            None,
            vec![
                actions_box((0.0, 0.0, 100.0, 50.0), Some(a), vec![]),
                actions_box((100.0, 0.0, 100.0, 50.0), Some(b), vec![]),
            ],
        )));
        (tab, a, b)
    }

    fn eval(tab: &mut Tab, source: &str) {
        tab.page_vm_mut()
            .expect("page vm")
            .eval(source)
            .expect("eval");
    }

    /// Evaluate a JS boolean expression against the page.
    fn js_true(tab: &mut Tab, expr: &str) -> bool {
        matches!(
            tab.page_vm_mut().expect("page vm").eval(expr),
            Ok(bowser_js::Value::Bool(true))
        )
    }

    fn bare_item(kind: ActionItemKind) -> ActionItem {
        ActionItem {
            kind,
            value: None,
            button: None,
            x: None,
            y: None,
            delta_x: None,
            delta_y: None,
            duration: None,
            origin: ActionOrigin::Viewport,
        }
    }

    fn move_item(x: f64, y: f64, origin: ActionOrigin) -> ActionItem {
        let mut item = bare_item(ActionItemKind::PointerMove);
        item.x = Some(x);
        item.y = Some(y);
        item.origin = origin;
        item
    }

    fn button_item(kind: ActionItemKind, button: u32) -> ActionItem {
        let mut item = bare_item(kind);
        item.button = Some(button);
        item
    }

    fn key_item(kind: ActionItemKind, value: &str) -> ActionItem {
        let mut item = bare_item(kind);
        item.value = Some(value.to_string());
        item
    }

    fn pointer_source(id: &str, actions: Vec<ActionItem>) -> ActionSource {
        ActionSource {
            kind: ActionSourceKind::Pointer,
            id: id.to_string(),
            pointer_type: Some("mouse".to_string()),
            actions,
        }
    }

    fn key_source(actions: Vec<ActionItem>) -> ActionSource {
        ActionSource {
            kind: ActionSourceKind::Key,
            id: "keyboard".to_string(),
            pointer_type: None,
            actions,
        }
    }

    fn sequence_request(sources: Vec<ActionSource>) -> bowser_js::TestDriverRequest {
        bowser_js::TestDriverRequest {
            token: 1,
            action: bowser_js::TestDriverAction::ActionSequence {
                sequence: TestDriverActionSequence { sources },
            },
        }
    }

    /// Register a recorder on `#a` and `#b` for the given event types,
    /// pushing `"<type>@<id>"` into the page-global `log` array.
    fn record_pointer_events(tab: &mut Tab, types: &[&str]) {
        let mut src = String::from("log = [];");
        for id in ["a", "b"] {
            for ty in types {
                src.push_str(&format!(
                    "document.getElementById('{id}').addEventListener('{ty}', \
                     function(e) {{ log.push('{ty}@{id}'); }});"
                ));
            }
        }
        eval(tab, &src);
    }

    #[test]
    fn button_bit_maps_webdriver_indices_to_ui_events_bits() {
        assert_eq!(button_bit(0), 1);
        assert_eq!(button_bit(1), 4);
        assert_eq!(button_bit(2), 2);
        assert_eq!(button_bit(3), 8);
        assert_eq!(button_bit(4), 16);
    }

    #[test]
    fn normalize_action_key_maps_codepoints_and_names() {
        assert_eq!(normalize_action_key("\u{E008}"), "Shift");
        assert_eq!(normalize_action_key("\u{E004}"), "Tab");
        assert_eq!(normalize_action_key("a"), "a");
        assert_eq!(normalize_action_key("Shift"), "Shift");
        assert_eq!(normalize_action_key("ArrowLeft"), "ArrowLeft");
    }

    #[test]
    fn resolve_action_point_covers_all_origins() {
        assert_eq!(
            resolve_action_point(&ActionOrigin::Viewport, 3.0, 4.0, (10.0, 10.0), None),
            Ok((3.0, 4.0))
        );
        assert_eq!(
            resolve_action_point(&ActionOrigin::Pointer, 3.0, 4.0, (10.0, 10.0), None),
            Ok((13.0, 14.0))
        );
        let node = {
            let doc = bowser_dom::Document::parse_html(b"<p id='x'>t</p>").0;
            doc.get_element_by_id("x").expect("x")
        };
        assert_eq!(
            resolve_action_point(
                &ActionOrigin::Element(node),
                3.0,
                4.0,
                (0.0, 0.0),
                Some((100.0, 50.0))
            ),
            Ok((103.0, 54.0))
        );
        // An element origin with no laid-out box is an error.
        assert!(
            resolve_action_point(&ActionOrigin::Element(node), 0.0, 0.0, (0.0, 0.0), None).is_err()
        );
    }

    #[test]
    fn actions_click_fires_on_up_same_target() {
        let (mut tab, _, _) =
            actions_tab("<html><body><div id=\"a\">A</div><div id=\"b\">B</div></body></html>");
        record_pointer_events(
            &mut tab,
            &["pointerdown", "mousedown", "pointerup", "mouseup", "click"],
        );
        let mut driver = TestDriver::new();
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(50.0, 25.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            js_true(
                &mut tab,
                "log.join(',') === 'pointerdown@a,mousedown@a,pointerup@a,mouseup@a,click@a'"
            ),
            "unexpected event log"
        );
    }

    #[test]
    fn actions_drag_to_other_element_fires_no_click() {
        let (mut tab, _, _) =
            actions_tab("<html><body><div id=\"a\">A</div><div id=\"b\">B</div></body></html>");
        record_pointer_events(
            &mut tab,
            &["pointerdown", "mousedown", "pointerup", "mouseup", "click"],
        );
        let mut driver = TestDriver::new();
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(50.0, 25.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    move_item(150.0, 25.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        // Release on #b after a press on #a: no click anywhere.
        assert!(
            js_true(
                &mut tab,
                "log.join(',') === 'pointerdown@a,mousedown@a,pointerup@b,mouseup@b'"
            ),
            "unexpected event log"
        );
    }

    #[test]
    fn actions_element_and_pointer_origin_math() {
        let (mut tab, _, b) =
            actions_tab("<html><body><div id=\"a\">A</div><div id=\"b\">B</div></body></html>");
        eval(
            &mut tab,
            "log = []; document.getElementById('b').addEventListener('pointermove', \
             function(e) { log.push(e.clientX + ',' + e.clientY); });",
        );
        let mut driver = TestDriver::new();
        // Element origin: #b's in-view center is (150, 25); + (5, 5).
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![move_item(5.0, 5.0, ActionOrigin::Element(b))],
            )]),
        );
        // Pointer origin: relative to the current position (155, 30).
        // State persists across action_sequence calls on one driver.
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![move_item(10.0, 0.0, ActionOrigin::Pointer)],
            )]),
        );
        assert!(
            js_true(&mut tab, "log.join(',') === '155,30,165,30'"),
            "unexpected move log"
        );
    }

    #[test]
    fn actions_modifier_state_reaches_pointer_events() {
        let (mut tab, _, _) =
            actions_tab("<html><body><div id=\"a\">A</div><div id=\"b\">B</div></body></html>");
        eval(
            &mut tab,
            "log = []; \
             document.addEventListener('keydown', function(e) { log.push('key:' + e.key); }); \
             document.getElementById('a').addEventListener('mousedown', \
               function(e) { log.push('mdown:shift=' + e.shiftKey); }); \
             document.getElementById('a').addEventListener('click', \
               function(e) { log.push('click@a'); });",
        );
        let mut driver = TestDriver::new();
        // Park the pointer over #a first (state persists across calls).
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![move_item(50.0, 25.0, ActionOrigin::Viewport)],
            )]),
        );
        // Tick-synchronized chord: Shift goes down on tick 0; the
        // pointer presses on tick 1 while 'a' is also keyed down.
        driver.execute(
            &mut tab,
            sequence_request(vec![
                key_source(vec![
                    key_item(ActionItemKind::KeyDown, "\u{E008}"),
                    key_item(ActionItemKind::KeyDown, "a"),
                    key_item(ActionItemKind::KeyUp, "a"),
                    key_item(ActionItemKind::KeyUp, "\u{E008}"),
                ]),
                pointer_source(
                    "mouse",
                    vec![
                        bare_item(ActionItemKind::Pause),
                        button_item(ActionItemKind::PointerDown, 0),
                        button_item(ActionItemKind::PointerUp, 0),
                        bare_item(ActionItemKind::Pause),
                    ],
                ),
            ]),
        );
        // After the chord, Shift is released: a fresh press sees it up.
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            js_true(
                &mut tab,
                "log.join(',') === \
                 'key:Shift,key:a,mdown:shift=true,click@a,mdown:shift=false,click@a'"
            ),
            "unexpected chord log"
        );
    }

    #[test]
    fn actions_secondary_button_fires_contextmenu_not_click() {
        let (mut tab, _, _) =
            actions_tab("<html><body><div id=\"a\">A</div><div id=\"b\">B</div></body></html>");
        record_pointer_events(&mut tab, &["contextmenu", "click"]);
        let mut driver = TestDriver::new();
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(50.0, 25.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 2),
                    button_item(ActionItemKind::PointerUp, 2),
                ],
            )]),
        );
        assert!(
            js_true(&mut tab, "log.join(',') === 'contextmenu@a'"),
            "unexpected contextmenu log"
        );
    }

    #[test]
    fn actions_touch_pointer_routes_touch_events() {
        let (mut tab, _, _) =
            actions_tab("<html><body><div id=\"a\">A</div><div id=\"b\">B</div></body></html>");
        eval(
            &mut tab,
            "log = []; \
             ['touchstart','touchend','pointerdown','pointerup'].forEach(function(t) { \
               document.getElementById('a').addEventListener(t, \
                 function(e) { log.push(t + '@a:' + (e.pointerType || '')); }); \
             });",
        );
        let mut driver = TestDriver::new();
        let mut touch = pointer_source(
            "touch",
            vec![
                move_item(50.0, 25.0, ActionOrigin::Viewport),
                button_item(ActionItemKind::PointerDown, 0),
                button_item(ActionItemKind::PointerUp, 0),
            ],
        );
        touch.pointer_type = Some("touch".to_string());
        driver.execute(&mut tab, sequence_request(vec![touch]));
        // pointerdown (PointerEvent, pointerType "touch"), then
        // touchstart, then on release pointerup + touchend. The touch
        // events carry no pointerType (TouchEvent), so "" there.
        assert!(
            js_true(
                &mut tab,
                "log.join(',') === \
                 'pointerdown@a:touch,touchstart@a:,pointerup@a:touch,touchend@a:'"
            ),
            "unexpected touch log"
        );
    }

    #[test]
    fn actions_wheel_scrolls_hit_container_by_default() {
        let (mut tab, _, _) =
            actions_tab("<html><body><div id=\"a\">A</div><div id=\"b\">B</div></body></html>");
        // Publish a scrollable viewport: max offset (0, 500).
        tab.scroll().set_max_offset((0.0, 500.0));
        eval(
            &mut tab,
            "log = []; document.addEventListener('wheel', \
             function(e) { log.push('wheel:' + e.deltaY); });",
        );
        let mut driver = TestDriver::new();
        let mut scroll = bare_item(ActionItemKind::Scroll);
        scroll.x = Some(50.0);
        scroll.y = Some(25.0);
        scroll.delta_x = Some(0.0);
        scroll.delta_y = Some(120.0);
        driver.execute(
            &mut tab,
            sequence_request(vec![ActionSource {
                kind: ActionSourceKind::Wheel,
                id: "wheel".to_string(),
                pointer_type: None,
                actions: vec![scroll],
            }]),
        );
        // The wheel event fired, and the default scroll moved the
        // viewport offset by the delta.
        assert!(js_true(&mut tab, "log.join(',') === 'wheel:120'"));
        assert_eq!(tab.scroll().offset(), (0.0, 120.0));
    }

    #[test]
    fn actions_tab_moves_focus_through_focusables() {
        let (mut tab, _, _) = actions_tab(
            "<html><body><button id=\"a\">A</button><button id=\"b\">B</button></body></html>",
        );
        eval(
            &mut tab,
            "log = []; \
             document.getElementById('a').addEventListener('focus', \
               function() { log.push('focus@a'); }); \
             document.getElementById('b').addEventListener('focus', \
               function() { log.push('focus@b'); }); \
             document.getElementById('b').addEventListener('keydown', \
               function(e) { log.push('key@b:' + e.key); });",
        );
        let mut driver = TestDriver::new();
        let tab_press = || {
            sequence_request(vec![key_source(vec![
                key_item(ActionItemKind::KeyDown, "\u{E004}"),
                key_item(ActionItemKind::KeyUp, "\u{E004}"),
            ])])
        };
        driver.execute(&mut tab, tab_press());
        driver.execute(&mut tab, tab_press());
        // Focus now sits on #b: a plain key targets it.
        driver.execute(
            &mut tab,
            sequence_request(vec![key_source(vec![
                key_item(ActionItemKind::KeyDown, "x"),
                key_item(ActionItemKind::KeyUp, "x"),
            ])]),
        );
        assert!(
            js_true(&mut tab, "log.join(',') === 'focus@a,focus@b,key@b:x'"),
            "unexpected focus log"
        );
    }

    // -----------------------------------------------------------------
    // Wave-50 fixes: cancellation (H4), click targeting (M1),
    // focus model (M2), wheel origin (M5), state parking (M9),
    // non-finite coordinates (M6)
    // -----------------------------------------------------------------

    /// A tab with a parent `#p` (0,0)-(200,50) containing `#a`
    /// (0,0)-(100,50) and `#b` (100,0)-(200,50): the M1 fixture.
    fn actions_tab_nested() -> (Tab, NodeId, NodeId, NodeId) {
        use std::cell::RefCell;
        use std::rc::Rc;
        let doc = Rc::new(RefCell::new(
            bowser_dom::Document::parse_html(
                b"<html><body><div id=\"p\"><div id=\"a\">A</div><div id=\"b\">B</div></div></body></html>",
            )
            .0,
        ));
        let (p, a, b) = {
            let d = doc.borrow();
            (
                d.get_element_by_id("p").expect("p element"),
                d.get_element_by_id("a").expect("a element"),
                d.get_element_by_id("b").expect("b element"),
            )
        };
        let base = bowser_url::parse("http://example.com/page.html", None).expect("test url");
        let page =
            crate::script::PageVm::new_for_document(doc.clone(), &base, "TestBrowser/9.9", &[])
                .expect("install web apis");
        let mut tab = crate::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
        tab.install_document(doc, page);
        tab.layout_root = Some(Rc::new(actions_box(
            (0.0, 0.0, 800.0, 600.0),
            None,
            vec![actions_box(
                (0.0, 0.0, 200.0, 50.0),
                Some(p),
                vec![
                    actions_box((0.0, 0.0, 100.0, 50.0), Some(a), vec![]),
                    actions_box((100.0, 0.0, 100.0, 50.0), Some(b), vec![]),
                ],
            )],
        )));
        (tab, p, a, b)
    }

    fn press_release(button: u32) -> Vec<ActionItem> {
        vec![
            button_item(ActionItemKind::PointerDown, button),
            button_item(ActionItemKind::PointerUp, button),
        ]
    }

    #[test]
    fn h4_pointerdown_prevented_suppresses_mousedown_and_focus() {
        let (mut tab, _, _) = actions_tab(
            "<html><body><button id=\"a\">A</button><button id=\"b\">B</button></body></html>",
        );
        eval(
            &mut tab,
            "log = []; \
             document.getElementById('a').addEventListener('pointerdown', \
               function(e) { log.push('pointerdown@a'); e.preventDefault(); }); \
             document.getElementById('a').addEventListener('mousedown', \
               function() { log.push('mousedown@a'); }); \
             document.getElementById('a').addEventListener('focus', \
               function() { log.push('focus@a'); });",
        );
        let mut driver = TestDriver::new();
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(50.0, 25.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            js_true(&mut tab, "log.join(',') === 'pointerdown@a'"),
            "cancelled pointerdown must suppress compat mousedown and the focus move"
        );
        assert_eq!(driver.actions.focused, None, "focus must not move");
    }

    #[test]
    fn h4_cancelled_mousedown_suppresses_focus_move() {
        let (mut tab, _, _) = actions_tab(
            "<html><body><button id=\"a\">A</button><button id=\"b\">B</button></body></html>",
        );
        eval(
            &mut tab,
            "log = []; \
             document.getElementById('a').addEventListener('pointerdown', \
               function() { log.push('pointerdown@a'); }); \
             document.getElementById('a').addEventListener('mousedown', \
               function(e) { log.push('mousedown@a'); e.preventDefault(); }); \
             document.getElementById('a').addEventListener('focus', \
               function() { log.push('focus@a'); });",
        );
        let mut driver = TestDriver::new();
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(50.0, 25.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            js_true(&mut tab, "log.join(',') === 'pointerdown@a,mousedown@a'"),
            "mousedown fired but focus must stay put when it is cancelled"
        );
        assert_eq!(driver.actions.focused, None, "focus must not move");
    }

    #[test]
    fn m1_click_targets_nearest_common_inclusive_ancestor() {
        let (mut tab, _, _, _) = actions_tab_nested();
        eval(
            &mut tab,
            "log = []; \
             ['a','b','p'].forEach(function(id) { \
               document.getElementById(id).addEventListener('click', \
                 function() { log.push('click@' + id); }); \
             });",
        );
        let mut driver = TestDriver::new();
        // Press on child #a, drag to sibling #b, release: the click
        // lands on their common ancestor #p, not on either child.
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(50.0, 25.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    move_item(150.0, 25.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            js_true(&mut tab, "log.join(',') === 'click@p'"),
            "click must target the common ancestor #p"
        );
    }

    #[test]
    fn m1_middle_button_fires_auxclick_only() {
        let (mut tab, _, _) =
            actions_tab("<html><body><div id=\"a\">A</div><div id=\"b\">B</div></body></html>");
        record_pointer_events(&mut tab, &["click", "auxclick"]);
        let mut driver = TestDriver::new();
        let mut actions = vec![move_item(50.0, 25.0, ActionOrigin::Viewport)];
        actions.extend(press_release(1));
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source("mouse", actions)]),
        );
        assert!(
            js_true(&mut tab, "log.join(',') === 'auxclick@a'"),
            "middle button must fire auxclick and no click"
        );
    }

    #[test]
    fn m1_bare_right_pointerup_fires_no_contextmenu() {
        let (mut tab, _, _) =
            actions_tab("<html><body><div id=\"a\">A</div><div id=\"b\">B</div></body></html>");
        record_pointer_events(&mut tab, &["contextmenu"]);
        let mut driver = TestDriver::new();
        // A pointerUp for the secondary button with no preceding
        // pointerDown: no contextmenu may be synthesized.
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(50.0, 25.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerUp, 2),
                ],
            )]),
        );
        assert!(
            js_true(&mut tab, "log.length === 0"),
            "bare right-button release must not fire contextmenu"
        );
    }

    #[test]
    fn m2_click_on_plain_div_does_not_take_focus() {
        let (mut tab, _, _) = actions_tab(
            "<html><body><button id=\"a\">A</button><div id=\"b\">B</div></body></html>",
        );
        eval(
            &mut tab,
            "log = []; \
             document.getElementById('a').addEventListener('keydown', \
               function(e) { log.push('key@a:' + e.key); }); \
             document.getElementById('b').addEventListener('keydown', \
               function(e) { log.push('key@b:' + e.key); }); \
             document.getElementById('b').addEventListener('focus', \
               function() { log.push('focus@b'); });",
        );
        let mut driver = TestDriver::new();
        // Click the button: it takes focus.
        let mut click_a = vec![move_item(50.0, 25.0, ActionOrigin::Viewport)];
        click_a.extend(press_release(0));
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source("mouse", click_a)]),
        );
        // Click the plain div (no tabindex): focus must stay on the
        // button, and later keys still go there.
        let mut click_b = vec![move_item(150.0, 25.0, ActionOrigin::Viewport)];
        click_b.extend(press_release(0));
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source("mouse", click_b)]),
        );
        driver.execute(
            &mut tab,
            sequence_request(vec![key_source(vec![
                key_item(ActionItemKind::KeyDown, "x"),
                key_item(ActionItemKind::KeyUp, "x"),
            ])]),
        );
        assert!(
            js_true(&mut tab, "log.join(',') === 'key@a:x'"),
            "plain div must not take focus; keys stay with the button"
        );
    }

    #[test]
    fn m2_detached_focused_node_falls_back_for_keys() {
        let (mut tab, a, _) = actions_tab(
            "<html><body><button id=\"a\">A</button><button id=\"b\">B</button></body></html>",
        );
        eval(
            &mut tab,
            "log = []; \
             document.getElementById('a').addEventListener('keydown', \
               function(e) { log.push('key@a:' + e.key); }); \
             document.addEventListener('keydown', \
               function(e) { log.push('key@doc:' + e.key); });",
        );
        let mut driver = TestDriver::new();
        let mut click_a = vec![move_item(50.0, 25.0, ActionOrigin::Viewport)];
        click_a.extend(press_release(0));
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source("mouse", click_a)]),
        );
        assert_eq!(driver.actions.focused, Some(a));
        // Detach the focused node (through page script, the way a
        // handler would), then send a key: dispatch must fall back
        // safely (body/document), not into the detached subtree —
        // the document listener still sees the bubbled event, the
        // detached node's own listener does not run as the target
        // path (it is no longer in the document).
        eval(
            &mut tab,
            "var elA = document.getElementById('a'); elA.parentNode.removeChild(elA);",
        );
        driver.execute(
            &mut tab,
            sequence_request(vec![key_source(vec![
                key_item(ActionItemKind::KeyDown, "y"),
                key_item(ActionItemKind::KeyUp, "y"),
            ])]),
        );
        assert_eq!(
            driver.actions.focused, None,
            "detached focus must reset to the fallback"
        );
        assert!(
            js_true(&mut tab, "log.join(',') === 'key@doc:y'"),
            "key must fall back to the document path, got a detached dispatch"
        );
    }

    #[test]
    fn m2_focusable_rules() {
        let doc = bowser_dom::Document::parse_html(
            b"<html><body><div id=\"plain\">x</div><div id=\"neg\" tabindex=\"-1\">x</div>\
               <div id=\"zero\" tabindex=\"0\">x</div><button id=\"btn\">x</button>\
               <button id=\"dis\" disabled>x</button><a id=\"lnk\" href=\"/x\">x</a>\
               <a id=\"nolnk\">x</a></body></html>",
        )
        .0;
        let f = |id: &str| td_is_focusable(&doc, doc.get_element_by_id(id).expect("el"));
        assert!(!f("plain"), "plain div is not focusable");
        assert!(f("neg"), "tabindex=-1 is click-focusable");
        assert!(f("zero"), "tabindex=0 is focusable");
        assert!(f("btn"), "button is naturally focusable");
        assert!(!f("dis"), "disabled button is not focusable");
        assert!(f("lnk"), "a[href] is focusable");
        assert!(!f("nolnk"), "a without href is not focusable");
    }

    #[test]
    fn m5_wheel_pointer_origin_uses_most_recently_active_pointer() {
        let (mut tab, _, _) =
            actions_tab("<html><body><div id=\"a\">A</div><div id=\"b\">B</div></body></html>");
        eval(
            &mut tab,
            "log = []; \
             document.getElementById('a').addEventListener('wheel', \
               function() { log.push('wheel@a'); }); \
             document.getElementById('b').addEventListener('wheel', \
               function() { log.push('wheel@b'); });",
        );
        let wheel_seq = || {
            let mut item = bare_item(ActionItemKind::Scroll);
            item.origin = ActionOrigin::Pointer;
            item.x = Some(0.0);
            item.y = Some(0.0);
            item.delta_x = Some(0.0);
            item.delta_y = Some(10.0);
            sequence_request(vec![ActionSource {
                kind: ActionSourceKind::Wheel,
                id: "wheel".to_string(),
                pointer_type: None,
                actions: vec![item],
            }])
        };
        let mut driver = TestDriver::new();
        // Two pointers: mouse over #a first, second over #b last —
        // the wheel resolves against the most recently active one.
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![move_item(50.0, 25.0, ActionOrigin::Viewport)],
            )]),
        );
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "second",
                vec![move_item(150.0, 25.0, ActionOrigin::Viewport)],
            )]),
        );
        driver.execute(&mut tab, wheel_seq());
        // Reactivate the mouse pointer: the wheel follows it back.
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![move_item(50.0, 25.0, ActionOrigin::Viewport)],
            )]),
        );
        driver.execute(&mut tab, wheel_seq());
        assert!(
            js_true(&mut tab, "log.join(',') === 'wheel@b,wheel@a'"),
            "wheel pointer origin must be the deterministic most-recent pointer"
        );
    }

    #[test]
    fn m9_parked_state_survives_other_context_drain() {
        let mut state_a = ExecutorState::default();
        state_a.modifiers.shift = true;
        let mut state_b = ExecutorState::default();
        state_b.modifiers.ctrl = true;
        store_persisted_action_state(101, state_a);
        store_persisted_action_state(202, state_b);
        // Draining page A (ctx 101) must not discard page B's state.
        let taken_a = take_persisted_action_state(101);
        assert!(taken_a.modifiers.shift, "ctx 101 state returned");
        let taken_b = take_persisted_action_state(202);
        assert!(
            taken_b.modifiers.ctrl,
            "ctx 202 state survived ctx 101's drain"
        );
        // Both slots are now empty: a fresh take yields defaults.
        assert!(!take_persisted_action_state(101).modifiers.shift);
        assert!(!take_persisted_action_state(202).modifiers.ctrl);
    }

    #[test]
    fn m6_non_finite_coordinates_are_rejected() {
        assert!(
            resolve_action_point(&ActionOrigin::Viewport, f64::NAN, 0.0, (0.0, 0.0), None).is_err()
        );
        assert!(
            resolve_action_point(&ActionOrigin::Pointer, f64::INFINITY, 0.0, (0.0, 0.0), None)
                .is_err()
        );
        assert!(
            resolve_action_point(&ActionOrigin::Pointer, 1.0, 0.0, (f64::NAN, 0.0), None).is_err(),
            "a poisoned current position must not propagate"
        );
    }

    #[test]
    fn m6_nan_move_action_dispatches_nothing_and_keeps_position() {
        let (mut tab, _, _) =
            actions_tab("<html><body><div id=\"a\">A</div><div id=\"b\">B</div></body></html>");
        eval(
            &mut tab,
            "log = []; document.getElementById('a').addEventListener('pointermove', \
             function(e) { log.push(e.clientX + ',' + e.clientY); }); \
             document.getElementById('b').addEventListener('pointermove', \
             function(e) { log.push(e.clientX + ',' + e.clientY); });",
        );
        let mut driver = TestDriver::new();
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![move_item(50.0, 25.0, ActionOrigin::Viewport)],
            )]),
        );
        // The NaN move errors inside the sequence: nothing is
        // dispatched and the pointer position is not poisoned.
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![move_item(f64::NAN, 0.0, ActionOrigin::Viewport)],
            )]),
        );
        // A pointer-relative move still resolves from (50, 25).
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![move_item(10.0, 0.0, ActionOrigin::Pointer)],
            )]),
        );
        assert!(
            js_true(&mut tab, "log.join(',') === '50,25,60,25'"),
            "NaN move must dispatch nothing and leave the position intact"
        );
    }

    #[test]
    fn transient_activation_grant_consume_cycle() {
        // The `bless` primitive: grant -> visible -> consumed once.
        let mut page = crate::script::PageVm::new();
        assert!(!page.has_transient_activation());
        // Consuming without a grant is a no-op false.
        assert!(!page.consume_transient_activation());
        page.grant_transient_activation();
        assert!(page.has_transient_activation());
        assert!(page.consume_transient_activation());
        assert!(!page.has_transient_activation());
        assert!(!page.consume_transient_activation());
    }

    /// The testdriver Actions executor applies the keydown default
    /// action: `ArrowDown` pans the focused element's scroll container
    /// (the windowed key path's `apply_key_scroll`) and fires `scroll`
    /// for it — before this fix the key event dispatched and nothing
    /// moved.
    #[test]
    fn actions_arrow_key_scrolls_focused_scroller() {
        use std::cell::RefCell;
        use std::rc::Rc;
        let doc = Rc::new(RefCell::new(
            bowser_dom::Document::parse_html(
                b"<html><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
            )
            .0,
        ));
        let sheet = bowser_css::parse_stylesheet(
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 500px; }",
        );
        let root = bowser_layout::layout(
            &doc.borrow(),
            &[&sheet],
            bowser_layout::Size { w: 800.0, h: 600.0 },
            &bowser_layout::FixedFontMetrics,
        );
        let sc = doc.borrow().get_element_by_id("sc").expect("sc");
        let base = bowser_url::parse("http://example.com/page.html", None).expect("test url");
        let page =
            crate::script::PageVm::new_for_document(doc.clone(), &base, "TestBrowser/9.9", &[])
                .expect("install web apis");
        let mut tab = crate::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
        tab.install_document(doc, page);
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        let isc = tab.scroll_tree.index_of(sc).expect("sc container");

        eval(
            &mut tab,
            "window.__events = []; document.getElementById('sc').addEventListener('scroll', \
             function() { window.__events.push('scroll'); });",
        );

        let mut driver = TestDriver::new();
        driver.actions.focused = Some(sc);
        let down = key_item(ActionItemKind::KeyDown, "ArrowDown");
        driver
            .execute_key_item(&mut tab, &down, true)
            .expect("arrow key down executes");
        assert_eq!(
            tab.scroll_tree.container(isc).state.offset(),
            (0.0, 40.0),
            "ArrowDown pans the focused scroller by one line step"
        );
        assert_eq!(
            tab.scroll().offset(),
            (0.0, 0.0),
            "the focused container consumed the delta; the viewport did not move"
        );
        assert!(
            js_true(&mut tab, "window.__events.join(',') === 'scroll'"),
            "the key scroll fires scroll on the moved container"
        );

        // Release is not a scroll: the offset stays put and no second
        // `scroll` fires.
        let up = key_item(ActionItemKind::KeyUp, "ArrowDown");
        driver
            .execute_key_item(&mut tab, &up, false)
            .expect("arrow key up executes");
        assert_eq!(tab.scroll_tree.container(isc).state.offset(), (0.0, 40.0));
        assert!(
            js_true(&mut tab, "window.__events.join(',') === 'scroll'"),
            "keyup fires no further scroll"
        );
    }

    // -----------------------------------------------------------------
    // popovertarget activation (w81-G): through execute_click, the
    // same path test_driver.click() takes.
    // -----------------------------------------------------------------

    /// A tab with web APIs and no layout tree: `execute_click`
    /// skips scrolling when `dirty.layout` is set, so no layout
    /// boxes are needed for these click tests.
    fn popover_click_tab(html: &str) -> (Tab, NodeId, NodeId) {
        use std::cell::RefCell;
        use std::rc::Rc;
        let doc = Rc::new(RefCell::new(
            bowser_dom::Document::parse_html(html.as_bytes()).0,
        ));
        let (btn, pop) = {
            let d = doc.borrow();
            (
                d.get_element_by_id("btn").expect("invoker"),
                d.get_element_by_id("pop").expect("popover target"),
            )
        };
        let base = bowser_url::parse("http://example.com/page.html", None).expect("test url");
        let page =
            crate::script::PageVm::new_for_document(doc.clone(), &base, "TestBrowser/9.9", &[])
                .expect("install web apis");
        let mut tab = crate::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
        tab.install_document(doc, page);
        tab.dirty.layout = true;
        (tab, btn, pop)
    }

    fn popover_open(tab: &Tab, pop: NodeId) -> bool {
        tab.document_ref()
            .map(|doc| doc.is_popover_open(pop))
            .unwrap_or(false)
    }

    #[test]
    fn testdriver_click_activates_popover_target_toggle_show_hide() {
        let (mut tab, btn, pop) = popover_click_tab(
            "<html><body><button id=\"btn\" popovertarget=\"pop\">t</button>\
             <div id=\"pop\" popover>hi</div></body></html>",
        );
        let mut driver = TestDriver::new();
        driver.execute_click(&mut tab, btn).expect("click");
        assert!(popover_open(&tab, pop), "toggle click must open");
        driver.execute_click(&mut tab, btn).expect("click");
        assert!(!popover_open(&tab, pop), "toggle click must close");

        // Show / hide actions, including their no-op directions.
        let (mut tab, btn, pop) = popover_click_tab(
            "<html><body><button id=\"btn\" popovertarget=\"pop\" \
             popovertargetaction=\"show\">s</button>\
             <button id=\"hide\" popovertarget=\"pop\" popovertargetaction=\"hide\">h</button>\
             <div id=\"pop\" popover>hi</div></body></html>",
        );
        let hide = tab
            .document_ref()
            .and_then(|doc| doc.get_element_by_id("hide"))
            .expect("hide button");
        let mut driver = TestDriver::new();
        driver.execute_click(&mut tab, btn).expect("click");
        assert!(popover_open(&tab, pop), "show click must open");
        driver.execute_click(&mut tab, btn).expect("click");
        assert!(popover_open(&tab, pop), "show on open is a no-op");
        driver.execute_click(&mut tab, hide).expect("click");
        assert!(!popover_open(&tab, pop), "hide click must close");
    }

    #[test]
    fn testdriver_click_popover_target_noops_and_prevented_click() {
        // Missing id and non-popover target: no-op.
        let (mut tab, btn, pop) = popover_click_tab(
            "<html><body><button id=\"btn\" popovertarget=\"nope\">t</button>\
             <div id=\"pop\">not a popover</div></body></html>",
        );
        let mut driver = TestDriver::new();
        driver.execute_click(&mut tab, btn).expect("click");
        assert!(!popover_open(&tab, pop), "missing id must no-op");

        // A preventDefault()ed click runs no default action, so the
        // popover stays closed even though the listener fired.
        let (mut tab, btn, pop) = popover_click_tab(
            "<html><body><button id=\"btn\" popovertarget=\"pop\">t</button>\
             <div id=\"pop\" popover>hi</div></body></html>",
        );
        eval(
            &mut tab,
            "window.__fired = 0; \
             document.getElementById('btn').addEventListener('click', function(e) { \
               window.__fired = 1; e.preventDefault(); });",
        );
        let mut driver = TestDriver::new();
        driver.execute_click(&mut tab, btn).expect("click");
        assert!(
            js_true(&mut tab, "window.__fired === 1"),
            "listener must still fire on a prevented click"
        );
        assert!(
            !popover_open(&tab, pop),
            "preventDefault()ed click must not activate the popover"
        );
    }

    // -----------------------------------------------------------------
    // Action-sequence / send_keys window-path parity (w81fix-B1)
    // -----------------------------------------------------------------

    /// A tab with web APIs and layout boxes for popover action
    /// sequences: `#btn` (0,0,100,50) containing `#span`
    /// (10,10,80,30), `#pop` (200,200,100,100). Mirrors
    /// `actions_tab`'s hand-built tree, with popover ids instead
    /// of `#a`/`#b`.
    fn popover_seq_tab(html: &str) -> (Tab, NodeId, NodeId) {
        use std::cell::RefCell;
        use std::rc::Rc;
        let doc = Rc::new(RefCell::new(
            bowser_dom::Document::parse_html(html.as_bytes()).0,
        ));
        let (btn, span, pop) = {
            let d = doc.borrow();
            (
                d.get_element_by_id("btn").expect("btn"),
                d.get_element_by_id("span").expect("span"),
                d.get_element_by_id("pop").expect("pop"),
            )
        };
        let base = bowser_url::parse("http://example.com/page.html", None).expect("test url");
        let page =
            crate::script::PageVm::new_for_document(doc.clone(), &base, "TestBrowser/9.9", &[])
                .expect("install web apis");
        let mut tab = crate::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
        tab.install_document(doc, page);
        tab.layout_root = Some(Rc::new(actions_box(
            (0.0, 0.0, 800.0, 600.0),
            None,
            vec![
                actions_box(
                    (0.0, 0.0, 100.0, 50.0),
                    Some(btn),
                    vec![actions_box((10.0, 10.0, 80.0, 30.0), Some(span), vec![])],
                ),
                actions_box((200.0, 200.0, 100.0, 100.0), Some(pop), vec![]),
            ],
        )));
        (tab, btn, pop)
    }

    fn open_popover_td(tab: &mut Tab, pop: NodeId) {
        tab.installed_mut()
            .expect("installed")
            .document()
            .borrow_mut()
            .show_popover(pop)
            .expect("show");
    }

    fn popover_seq_html() -> &'static str {
        "<html><body><button id=\"btn\" popovertarget=\"pop\">\
         <span id=\"span\">t</span></button>\
         <div id=\"pop\" popover>hi</div></body></html>"
    }

    #[test]
    fn actions_click_activates_popover_target_via_nested_span() {
        let (mut tab, _, pop) = popover_seq_tab(popover_seq_html());
        let click = || {
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(50.0, 25.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )])
        };
        let mut driver = TestDriver::new();
        driver.execute(&mut tab, click());
        assert!(
            popover_open(&tab, pop),
            "action-sequence click on the nested span must activate via the button"
        );
        driver.execute(&mut tab, click());
        assert!(
            !popover_open(&tab, pop),
            "second action-sequence click must toggle closed"
        );
    }

    #[test]
    fn actions_pointerup_invoker_fires_victim_toggles_before_primary() {
        // w82 final-review fix: the pointer-up invoker path
        // must use `activate_popover_target_full`, like
        // `execute_click` and the windowed click path. A naive
        // victim test would not prove it: pressing the invoker
        // outside the victim light-dismisses the victim on
        // release, before the click, so its toggle fires even
        // with the primary-only form. Shield the victim
        // instead: a `manual` popover above it makes
        // `light_dismiss_candidate` return `None` (the victim
        // is not the top-layer member), while a show still
        // auto-closes it — so only the victim-toggles path
        // fires the victim's `toggle`, before the primary's.
        use std::cell::RefCell;
        use std::rc::Rc;
        let doc = Rc::new(RefCell::new(
            bowser_dom::Document::parse_html(
                b"<html><body><button id=\"btn\" popovertarget=\"pb\">\
                 <span id=\"span\">t</span></button>\
                 <div id=\"pa\" popover>victim</div>\
                 <div id=\"pm\" popover=\"manual\">shield</div>\
                 <div id=\"pb\" popover>target</div></body></html>",
            )
            .0,
        ));
        let (btn, span, pa, pm, pb) = {
            let d = doc.borrow();
            (
                d.get_element_by_id("btn").expect("btn"),
                d.get_element_by_id("span").expect("span"),
                d.get_element_by_id("pa").expect("pa"),
                d.get_element_by_id("pm").expect("pm"),
                d.get_element_by_id("pb").expect("pb"),
            )
        };
        let base = bowser_url::parse("http://example.com/page.html", None).expect("test url");
        let page =
            crate::script::PageVm::new_for_document(doc.clone(), &base, "TestBrowser/9.9", &[])
                .expect("install web apis");
        let mut tab = crate::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
        tab.install_document(doc, page);
        tab.layout_root = Some(Rc::new(actions_box(
            (0.0, 0.0, 800.0, 600.0),
            None,
            vec![
                actions_box(
                    (0.0, 0.0, 100.0, 50.0),
                    Some(btn),
                    vec![actions_box((10.0, 10.0, 80.0, 30.0), Some(span), vec![])],
                ),
                actions_box((200.0, 200.0, 100.0, 100.0), Some(pa), vec![]),
                actions_box((300.0, 200.0, 100.0, 100.0), Some(pm), vec![]),
                actions_box((400.0, 200.0, 100.0, 100.0), Some(pb), vec![]),
            ],
        )));
        open_popover_td(&mut tab, pa);
        open_popover_td(&mut tab, pm);
        eval(
            &mut tab,
            "window.__toggles = []; \
             document.getElementById('pa').addEventListener('toggle', function() { window.__toggles.push('a'); }); \
             document.getElementById('pm').addEventListener('toggle', function() { window.__toggles.push('m'); }); \
             document.getElementById('pb').addEventListener('toggle', function() { window.__toggles.push('b'); });",
        );
        let mut driver = TestDriver::new();
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(50.0, 25.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            !popover_open(&tab, pa),
            "showing pb must auto-close the victim pa"
        );
        assert!(popover_open(&tab, pb), "pb must open");
        assert!(popover_open(&tab, pm), "manual shield pm must stay open");
        assert!(
            js_true(&mut tab, "window.__toggles.join(',') === 'a,b'"),
            "victim toggle must fire before the primary toggle, and the shield must not toggle"
        );
    }

    #[test]
    fn actions_pointerup_light_dismiss_and_drag_into_popover() {
        // Same-spot outside release dismisses.
        let (mut tab, _, pop) = popover_seq_tab(popover_seq_html());
        open_popover_td(&mut tab, pop);
        let mut driver = TestDriver::new();
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(700.0, 500.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            !popover_open(&tab, pop),
            "outside press + same-spot release must light-dismiss"
        );

        // Press outside, drag into the popover, release inside:
        // no dismiss.
        let (mut tab, _, pop) = popover_seq_tab(popover_seq_html());
        open_popover_td(&mut tab, pop);
        let mut driver = TestDriver::new();
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(700.0, 500.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    move_item(250.0, 250.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            popover_open(&tab, pop),
            "release inside the popover must not light-dismiss"
        );
    }

    #[test]
    fn actions_light_dismiss_with_stored_duration_retains_and_fires_once() {
        // w82 fixpaths B3: a testdriver light-dismiss with a
        // show-time stored exit duration retains the popover
        // for its transition (rendering) while the toggle
        // fires once, at dismissal time.
        let (mut tab, _, pop) = popover_seq_tab(popover_seq_html());
        open_popover_td(&mut tab, pop);
        tab.installed_mut()
            .expect("installed")
            .document()
            .borrow_mut()
            .store_exit_duration(pop, 60.0);
        eval(
            &mut tab,
            "window.__toggles = 0; \
             document.getElementById('pop').addEventListener('toggle', function() { window.__toggles += 1; });",
        );
        let mut driver = TestDriver::new();
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(700.0, 500.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            tab.document_ref().expect("document").is_exit_retained(pop),
            "stored duration must retain the light-dismissed popover"
        );
        assert!(
            js_true(&mut tab, "window.__toggles === 1"),
            "light-dismiss must fire exactly one toggle"
        );
    }

    #[test]
    fn actions_light_dismiss_stale_pending_fires_no_toggle() {
        // w82 fixpaths M4: a pointerdown record whose popover
        // a pointerdown listener already closed must not fire
        // a spurious toggle on release — the release dismisses
        // nothing when the popover is not open there.
        let (mut tab, _, pop) = popover_seq_tab(popover_seq_html());
        open_popover_td(&mut tab, pop);
        eval(
            &mut tab,
            "window.__toggles = 0; \
             document.getElementById('pop').addEventListener('toggle', function() { window.__toggles += 1; }); \
             document.body.addEventListener('pointerdown', function() { document.getElementById('pop').hidePopover(); });",
        );
        let mut driver = TestDriver::new();
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(700.0, 500.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            !popover_open(&tab, pop),
            "pointerdown listener must have closed the popover"
        );
        assert!(
            js_true(&mut tab, "window.__toggles === 1"),
            "only the listener's own hide may toggle, never a stale light-dismiss"
        );
    }

    #[test]
    fn actions_escape_dismisses_auto_popover() {
        let (mut tab, _, pop) = popover_click_tab(popover_seq_html());
        open_popover_td(&mut tab, pop);
        let mut driver = TestDriver::new();
        driver.execute(
            &mut tab,
            sequence_request(vec![key_source(vec![
                key_item(ActionItemKind::KeyDown, "\u{E00C}"),
                key_item(ActionItemKind::KeyUp, "\u{E00C}"),
            ])]),
        );
        assert!(
            !popover_open(&tab, pop),
            "action-sequence Escape must dismiss the auto popover"
        );
    }

    #[test]
    fn send_keys_escape_dismisses_auto_popover() {
        let (mut tab, btn, pop) = popover_click_tab(popover_seq_html());
        open_popover_td(&mut tab, pop);
        let mut driver = TestDriver::new();
        driver
            .execute_send_keys(&mut tab, btn, "\u{E00C}")
            .expect("send_keys");
        assert!(
            !popover_open(&tab, pop),
            "send_keys Escape must dismiss the auto popover"
        );
    }

    #[test]
    fn send_keys_escape_delivers_trusted_cancel_close_and_toggle() {
        // w83: the testdriver Escape path routes through the same
        // trusted host dispatches as the windowed path, so the
        // dialog's `cancel`/`close` and the popover's `toggle`
        // events all arrive with `isTrusted === true` — and a
        // light-dismissed popover's `toggle` is trusted too (the
        // same hook, exercised below through the pointer path).
        let (mut tab, out, dlg, _a) = dialog_tab();
        eval(
            &mut tab,
            "document.getElementById('dlg').addEventListener('cancel', function(e) { cancelTrusted = e.isTrusted; }); document.getElementById('dlg').addEventListener('close', function(e) { closeTrusted = e.isTrusted; }); document.getElementById('dlg').showModal();",
        );
        let mut driver = TestDriver::new();
        driver
            .execute_send_keys(&mut tab, out, "\u{E00C}")
            .expect("send_keys");
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "Escape must close the modal dialog"
        );
        assert!(
            js_true(&mut tab, "cancelTrusted === true"),
            "Escape cancel must be trusted"
        );
        assert!(
            js_true(&mut tab, "closeTrusted === true"),
            "Escape close must be trusted"
        );

        // Popover Escape.
        let (mut tab, btn, pop) = popover_click_tab(popover_seq_html());
        open_popover_td(&mut tab, pop);
        eval(
            &mut tab,
            "document.getElementById('pop').addEventListener('toggle', function(e) { toggleTrusted = e.isTrusted; toggleOld = e.oldState; toggleNew = e.newState; });",
        );
        let mut driver = TestDriver::new();
        driver
            .execute_send_keys(&mut tab, btn, "\u{E00C}")
            .expect("send_keys");
        assert!(
            !popover_open(&tab, pop),
            "Escape must dismiss the auto popover"
        );
        assert!(
            js_true(&mut tab, "toggleTrusted === true"),
            "Escape toggle must be trusted"
        );
        assert!(
            js_true(&mut tab, "toggleOld === 'open' && toggleNew === 'closed'"),
            "Escape toggle must read open -> closed"
        );

        // Popover light-dismiss (pointer press/release outside).
        let (mut tab, _, pop) = popover_seq_tab(popover_seq_html());
        open_popover_td(&mut tab, pop);
        eval(
            &mut tab,
            "document.getElementById('pop').addEventListener('toggle', function(e) { ldToggleTrusted = e.isTrusted; });",
        );
        let mut driver = TestDriver::new();
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(700.0, 500.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            !popover_open(&tab, pop),
            "light-dismiss must hide the auto popover"
        );
        assert!(
            js_true(&mut tab, "ldToggleTrusted === true"),
            "light-dismiss toggle must be trusted"
        );
    }

    #[test]
    fn send_keys_escape_passive_cancel_listener_does_not_gate_close() {
        // S5, testdriver path: a passive `cancel` listener's
        // `preventDefault()` is a no-op, so Escape must still close
        // the modal dialog (the non-passive case gates it — see the
        // windowed Escape tests — but passive must not).
        let (mut tab, out, dlg, _a) = dialog_tab();
        eval(
            &mut tab,
            "document.getElementById('dlg').addEventListener('cancel', function(e) { passiveRan = true; e.preventDefault(); }, {passive: true}); document.getElementById('dlg').showModal();",
        );
        let mut driver = TestDriver::new();
        driver
            .execute_send_keys(&mut tab, out, "\u{E00C}")
            .expect("send_keys");
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "Escape must close the dialog despite a passive cancel preventDefault"
        );
        assert!(
            js_true(&mut tab, "passiveRan === true"),
            "passive cancel listener still ran on Escape"
        );
    }

    #[test]
    fn actions_tab_traps_in_popover_above_modal_dialog() {
        // Same shape as popover_click_tab (web APIs, dirty layout
        // so no geometry is needed), but with the trap fixture's
        // ids instead of btn/pop.
        let mut tab = {
            use std::cell::RefCell;
            use std::rc::Rc;
            let doc = Rc::new(RefCell::new(
                bowser_dom::Document::parse_html(
                    b"<html><body><button id=\"out\">o</button>\
                     <dialog id=\"dlg\"><button id=\"dinner\">d</button></dialog>\
                     <div id=\"pop\" popover><button id=\"p1\">1</button>\
                     <button id=\"p2\">2</button></div></body></html>",
                )
                .0,
            ));
            let base = bowser_url::parse("http://example.com/page.html", None).expect("test url");
            let page =
                crate::script::PageVm::new_for_document(doc.clone(), &base, "TestBrowser/9.9", &[])
                    .expect("install web apis");
            let mut tab = crate::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
            tab.install_document(doc, page);
            tab.dirty.layout = true;
            tab
        };
        let (out, dlg, pop, p1, p2) = {
            let doc = tab.document_ref().expect("doc");
            (
                doc.get_element_by_id("out").expect("out"),
                doc.get_element_by_id("dlg").expect("dlg"),
                doc.get_element_by_id("pop").expect("pop"),
                doc.get_element_by_id("p1").expect("p1"),
                doc.get_element_by_id("p2").expect("p2"),
            )
        };
        tab.installed_mut()
            .expect("installed")
            .document()
            .borrow_mut()
            .show_dialog_modal(dlg)
            .expect("showModal");
        open_popover_td(&mut tab, pop);
        let mut driver = TestDriver::new();
        driver.actions.focused = Some(out);
        let tab_key = || {
            sequence_request(vec![key_source(vec![
                key_item(ActionItemKind::KeyDown, "\u{E004}"),
                key_item(ActionItemKind::KeyUp, "\u{E004}"),
            ])])
        };
        driver.execute(&mut tab, tab_key());
        assert_eq!(driver.actions.focused, Some(p1));
        driver.execute(&mut tab, tab_key());
        assert_eq!(driver.actions.focused, Some(p2));
        driver.execute(&mut tab, tab_key());
        assert_eq!(
            driver.actions.focused,
            Some(p1),
            "Tab must wrap inside the popover, never reaching the dialog"
        );
    }
    /// Tab with an outside button, a modal dialog containing
    /// one button, and layout boxes for both (B2 tests drive
    /// real JS through the bindings, so the dialog methods run
    /// their production path, including the show-time
    /// snapshot).
    fn dialog_tab() -> (Tab, NodeId, NodeId, NodeId) {
        use std::cell::RefCell;
        use std::rc::Rc;
        let doc = Rc::new(RefCell::new(
            bowser_dom::Document::parse_html(
                b"<html><body><button id=\"out\">o</button>\
                  <dialog id=\"dlg\"><button id=\"a\">a</button></dialog></body></html>",
            )
            .0,
        ));
        let (out, dlg, a) = {
            let d = doc.borrow();
            (
                d.get_element_by_id("out").expect("out"),
                d.get_element_by_id("dlg").expect("dlg"),
                d.get_element_by_id("a").expect("a"),
            )
        };
        let base = bowser_url::parse("http://example.com/page.html", None).expect("test url");
        let page =
            crate::script::PageVm::new_for_document(doc.clone(), &base, "TestBrowser/9.9", &[])
                .expect("install web apis");
        let mut tab = crate::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
        tab.install_document(doc, page);
        tab.layout_root = Some(Rc::new(actions_box(
            (0.0, 0.0, 800.0, 600.0),
            None,
            vec![
                actions_box((0.0, 0.0, 100.0, 50.0), Some(out), vec![]),
                actions_box((100.0, 0.0, 100.0, 50.0), Some(a), vec![]),
            ],
        )));
        (tab, out, dlg, a)
    }

    #[test]
    fn testdriver_showmodal_moves_focus_and_js_close_restores() {
        // B2, through real JS: showModal moves executor focus
        // in, close() restores it at the next sync. The
        // bindings snapshot the mirrored focus at show time
        // (set here via `set_focus`, the executor's mirror
        // site), so this also pins the bindings integration.
        let (mut tab, out, _dlg, a) = dialog_tab();
        let mut driver = TestDriver::new();
        driver.set_focus(&mut tab, Some(out));
        eval(&mut tab, "document.getElementById('dlg').showModal();");
        driver.sync_modal_focus(&mut tab);
        assert_eq!(
            driver.actions.focused,
            Some(a),
            "JS showModal must move testdriver focus in"
        );
        assert_eq!(driver.actions.modal_focus_stack.len(), 1);
        eval(&mut tab, "document.getElementById('dlg').close();");
        driver.sync_modal_focus(&mut tab);
        assert_eq!(
            driver.actions.focused,
            Some(out),
            "JS close() must restore testdriver focus at the sync"
        );
        assert!(driver.actions.modal_focus_stack.is_empty());
    }

    #[test]
    fn testdriver_same_interval_open_close_and_close_reopen() {
        // B2 + M3 through real JS: same-interval sequences
        // replay from the journal instead of vanishing.
        let (mut tab, out, _dlg, a) = dialog_tab();
        let mut driver = TestDriver::new();
        driver.set_focus(&mut tab, Some(out));
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); d.showModal(); d.close();",
        );
        driver.sync_modal_focus(&mut tab);
        assert_eq!(
            driver.actions.focused,
            Some(out),
            "open+close in one interval must end at the previous focus"
        );
        assert!(driver.actions.modal_focus_stack.is_empty());
        eval(&mut tab, "document.getElementById('dlg').showModal();");
        driver.sync_modal_focus(&mut tab);
        assert_eq!(driver.actions.focused, Some(a));
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); d.close(); d.showModal();",
        );
        driver.sync_modal_focus(&mut tab);
        assert_eq!(
            driver.actions.focused,
            Some(a),
            "close+reopen in one interval must move focus in again"
        );
        assert_eq!(driver.actions.modal_focus_stack.len(), 1);
        assert_eq!(
            driver.actions.modal_focus_stack[0],
            (_dlg, Some(out), true),
            "re-show must not keep a stale previous"
        );
    }
    #[test]
    fn w84_request_close_modal_cancel_close_prevented_and_non_modal() {
        // C3, through real JS: `requestClose(rv)` fires a trusted
        // cancel then a trusted close and applies `rv`; a
        // prevented cancel keeps the dialog open with `rv`
        // untouched; an omitted argument keeps the stored value;
        // a closed dialog is a no-op with no events; a
        // non-dialog receiver throws NotSupportedError; and a
        // non-modal dialog (show(), not showModal()) fires
        // cancel and closes too (WHATWG HTML §4.11.4: show()
        // sets a close watcher, requestClose() enables it).
        let (mut tab, out, dlg, _a) = dialog_tab();
        let mut driver = TestDriver::new();
        driver.set_focus(&mut tab, Some(out));
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); \
             window.__log = []; window.__prevent = false; \
             d.addEventListener('cancel', function(e) { \
               __log.push('cancel:' + e.isTrusted + ':' + e.cancelable); \
               if (__prevent) e.preventDefault(); }); \
             d.addEventListener('close', function(e) { \
               __log.push('close:' + e.isTrusted + ':' + e.cancelable); }); \
             d.showModal(); d.requestClose('ok');",
        );
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "unprevented requestClose must close the modal dialog"
        );
        assert!(
            js_true(
                &mut tab,
                "__log.join(';') === 'cancel:true:true;close:true:false'"
            ),
            "cancel (cancelable, trusted) must precede close (non-cancelable, trusted)"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'ok'"),
            "requestClose('ok') must apply its argument"
        );
        // Prevented cancel: stays open, returnValue untouched,
        // no close event.
        eval(
            &mut tab,
            "d.showModal(); __prevent = true; d.requestClose('nope');",
        );
        assert!(
            tab.document_ref().expect("document").is_dialog_open(dlg),
            "prevented requestClose must keep the dialog open"
        );
        assert!(
            js_true(
                &mut tab,
                "__log.length === 3 && __log[2] === 'cancel:true:true'"
            ),
            "prevented requestClose fires cancel only"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'ok'"),
            "prevented requestClose must leave returnValue untouched"
        );
        // Unprevented, omitted argument: closes, stored value kept.
        eval(&mut tab, "__prevent = false; d.requestClose();");
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "second requestClose must close"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'ok'"),
            "omitted requestClose argument must keep the stored value"
        );
        // Closed dialog: no-op, no events.
        eval(&mut tab, "d.requestClose('late');");
        assert!(
            js_true(&mut tab, "__log.length === 5"),
            "requestClose on a closed dialog must fire nothing"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'ok'"),
            "closed-dialog requestClose must not apply its argument"
        );
        // Non-dialog receiver throws a `TypeError` naming
        // `NotSupportedError` (engine idiom: no `DOMException`
        // type) — asserted as error identity (`instanceof` +
        // `name` + message), not a bare substring of the
        // stringified error.
        eval(
            &mut tab,
            "window.__nd = false; try { document.querySelector('button').requestClose(); } \
             catch (e) { window.__nd = (e instanceof TypeError) && e.name === 'TypeError' && \
               String(e.message).indexOf('NotSupportedError') !== -1; }",
        );
        assert!(
            js_true(&mut tab, "__nd === true"),
            "requestClose on a non-dialog must throw a TypeError naming NotSupportedError"
        );
        // Non-modal dialog: cancel then close, value applied.
        eval(&mut tab, "d.show(); d.requestClose('nm');");
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "requestClose must close a non-modal dialog too"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'nm'"),
            "non-modal requestClose must apply its argument"
        );
        assert!(
            js_true(
                &mut tab,
                "__log.slice(-2).join(';') === 'cancel:true:true;close:true:false'"
            ),
            "non-modal requestClose must fire cancel then close"
        );
        // returnValue matrix (w84fix): `null` stringifies to
        // `"null"`, `""` clears the stored value (DOMString
        // conversion, like `close()`).
        eval(&mut tab, "d.showModal(); d.requestClose(null);");
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "requestClose(null) must close"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'null'"),
            "requestClose(null) must apply the string \"null\""
        );
        eval(&mut tab, "d.showModal(); d.requestClose('');");
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "requestClose('') must close"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === ''"),
            "requestClose('') must clear the stored value"
        );
        // A `{passive:true}` cancel listener cannot prevent the
        // close: its `preventDefault` is a no-op, like Escape's.
        eval(
            &mut tab,
            "d.addEventListener('cancel', function(e) { e.preventDefault(); }, {passive:true}); \
             d.showModal(); d.requestClose('pv');",
        );
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "a passive cancel listener must not prevent requestClose"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'pv'"),
            "passive-cancel requestClose must apply its argument"
        );
        // A bare-`open` dialog (markup / attribute open, never
        // shown via `show()` / `showModal()`) fires cancel then
        // close: under the current spec the dialog setup steps
        // (including close-watcher establishment) run when the
        // `open` attribute is added for a connected dialog, so
        // w84's stale "bare open has no close watcher" note is
        // corrected, not carried forward.
        eval(
            &mut tab,
            "d.setAttribute('open', ''); d.requestClose('bare');",
        );
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "bare-open requestClose must close"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'bare'"),
            "bare-open requestClose must apply its argument"
        );
        assert!(
            js_true(
                &mut tab,
                "__log.slice(-2).join(';') === 'cancel:true:true;close:true:false'"
            ),
            "bare-open requestClose must fire cancel then close"
        );
    }

    #[test]
    fn w84_request_close_cancel_listener_reentrancy_two_cases() {
        // w84fix (dialog #2): the cancel listener's own close
        // splits two ways, and the comment on
        // `n_request_close_dialog` used to conflate them.
        // (a) cancel closes without reopening: exactly one
        // `close` event, with the listener's value, and the
        // requestClose argument is discarded. (b) cancel closes
        // AND re-opens: a second `close` event fires, with the
        // requestClose argument applied — the open state is not
        // snapshotted at cancel time.
        let (mut tab, _out, dlg, _a) = dialog_tab();
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); window.__closes = 0; window.__mode = 'close'; \
             d.addEventListener('cancel', function() { \
               if (__mode === 'close') { d.close('from-cancel'); } \
               else if (__mode === 'reopen') { d.close('tmp'); d.showModal(); } }); \
             d.addEventListener('close', function() { __closes += 1; }); \
             d.showModal(); d.requestClose('arg');",
        );
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "cancel-close case must end closed"
        );
        assert!(
            js_true(&mut tab, "__closes === 1"),
            "cancel-close must fire exactly one close event"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'from-cancel'"),
            "cancel-close must keep the listener's value, discarding the requestClose argument"
        );
        eval(
            &mut tab,
            "__mode = 'reopen'; d.showModal(); d.requestClose('arg2');",
        );
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "cancel close-then-reopen case must end closed"
        );
        assert!(
            js_true(&mut tab, "__closes === 3"),
            "cancel close-then-reopen must fire a second close event (listener close plus requestClose close)"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'arg2'"),
            "cancel close-then-reopen must apply the requestClose argument on the second close"
        );
    }

    #[test]
    fn w84_request_close_disconnected_dialog_is_noop() {
        // w84fix (whole-diff #2): per the spec's request-to-close
        // step 2, a dialog removed while open (`open` survives
        // removal) answers `requestClose` silently — no `cancel`,
        // no `close`, argument untouched.
        let (mut tab, _out, dlg, _a) = dialog_tab();
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); window.__log = []; \
             d.addEventListener('cancel', function() { __log.push('cancel'); }); \
             d.addEventListener('close', function() { __log.push('close'); }); \
             d.showModal(); document.body.removeChild(d); d.requestClose('late');",
        );
        assert!(
            js_true(&mut tab, "__log.length === 0"),
            "requestClose on a disconnected dialog must fire nothing"
        );
        assert!(
            tab.document_ref().expect("document").is_dialog_open(dlg),
            "disconnected requestClose must leave the open attribute in place"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === ''"),
            "disconnected requestClose must not apply its argument"
        );
    }

    #[test]
    fn w84_request_close_inline_oncancel_is_not_served() {
        // w84fix (dialog #3), KNOWN DIVERGENCE pinned, not
        // hidden: JS `requestClose` dispatches `cancel` through
        // the JS-native dispatch core, which serves only
        // `addEventListener` listeners — an inline `oncancel`
        // attribute (`return false`) does NOT gate the close,
        // so the dialog closes. The windowed Escape path serves
        // the same attribute through the host dispatch
        // (`dispatch_cancel` in `script.rs`), so Escape on the
        // same dialog keeps it open. Routing the JS cancel
        // through the host dispatch would need a
        // bindings-to-window channel that does not exist; until
        // it does, this test pins the split.
        let (mut tab, _out, dlg, _a) = dialog_tab();
        let mut driver = TestDriver::new();
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); \
             d.setAttribute('oncancel', 'return false'); d.showModal(); d.requestClose('x');",
        );
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "JS requestClose must close despite inline oncancel returning false (pinned limitation)"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'x'"),
            "JS requestClose must apply its argument despite inline oncancel"
        );
        eval(&mut tab, "d.showModal();");
        assert!(
            driver.execute_escape(&mut tab),
            "Escape must find the modal dialog dismissible"
        );
        assert!(
            tab.document_ref().expect("document").is_dialog_open(dlg),
            "Escape must keep the dialog open when inline oncancel returns false"
        );
    }

    #[test]
    fn w84_reclose_inside_close_listener_is_noop_and_reshow_sees_closed() {
        // C2: a `close` listener that calls `close()` again
        // must not fire a second `close` event, and must observe
        // the dialog already closed; re-showing afterwards works.
        let (mut tab, _out, dlg, _a) = dialog_tab();
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); \
             window.__closes = 0; window.__openInClose = null; \
             d.addEventListener('close', function() { \
               __closes += 1; __openInClose = d.open; d.close(); }); \
             d.showModal(); d.close();",
        );
        assert!(
            js_true(&mut tab, "__closes === 1 && __openInClose === false"),
            "re-close inside a close listener must be a no-op and see closed state"
        );
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "dialog must stay closed after the listener re-close"
        );
        eval(&mut tab, "d.showModal();");
        assert!(
            tab.document_ref().expect("document").is_dialog_open(dlg),
            "re-show after a listener re-close must reopen"
        );
        eval(&mut tab, "d.close();");
        assert!(
            js_true(&mut tab, "__closes === 2"),
            "a second real close must fire close again"
        );
    }

    #[test]
    fn w84_request_close_restores_focus_at_sync_and_prevented_keeps_it() {
        // Focus parity with JS close(): an unprevented
        // requestClose restores the previous focus at the next
        // sync; a prevented one leaves focus inside the dialog.
        let (mut tab, out, _dlg, a) = dialog_tab();
        let mut driver = TestDriver::new();
        driver.set_focus(&mut tab, Some(out));
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); \
             d.addEventListener('cancel', function(e) { if (window.__prevent) e.preventDefault(); }); \
             window.__prevent = false; d.showModal();",
        );
        driver.sync_modal_focus(&mut tab);
        assert_eq!(driver.actions.focused, Some(a));
        eval(&mut tab, "d.requestClose();");
        // w84fix pin, mirroring the window-level
        // `js_close_listener_observes_pre_restore_focus_then_sync_restores`:
        // the `close` listener ran synchronously inside the
        // eval above, but focus has not moved yet — the
        // executor mirror still points inside the dialog, i.e.
        // pre-restore focus is what a close listener observes.
        assert_eq!(
            driver.actions.focused,
            Some(a),
            "focus must still be inside the dialog after the requestClose eval and before the sync"
        );
        driver.sync_modal_focus(&mut tab);
        assert_eq!(
            driver.actions.focused,
            Some(out),
            "unprevented requestClose must restore previous focus at the sync"
        );
        assert!(driver.actions.modal_focus_stack.is_empty());
        eval(
            &mut tab,
            "d.showModal(); __prevent = true; d.requestClose();",
        );
        // w84fix: no sync ran between the second `showModal`
        // and this prevented `requestClose`, so that show's
        // move-in is still pending — pre-sync focus is unchanged
        // (still the pre-show focus, `out`), and the prevented
        // close itself moves nothing. The sync below replays the
        // show and lands focus inside the dialog.
        assert_eq!(
            driver.actions.focused,
            Some(out),
            "prevented requestClose must move no focus before the sync"
        );
        driver.sync_modal_focus(&mut tab);
        assert_eq!(
            driver.actions.focused,
            Some(a),
            "prevented requestClose must leave focus inside the dialog"
        );
        assert_eq!(driver.actions.modal_focus_stack.len(), 1);
    }

    #[test]
    fn w84_remove_attribute_open_fires_no_close_event() {
        // C8 (JS level): removing `open` from a modal dialog
        // closes it state-wise but fires no `close` event.
        let (mut tab, _out, dlg, _a) = dialog_tab();
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); window.__closes = 0; \
             d.addEventListener('close', function() { __closes += 1; }); \
             d.showModal(); d.returnValue = 'kept'; d.removeAttribute('open');",
        );
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "removeAttribute('open') must close the dialog state-wise"
        );
        assert!(
            js_true(&mut tab, "__closes === 0"),
            "removeAttribute('open') must fire no close event"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'kept'"),
            "removeAttribute('open') must leave returnValue untouched"
        );
    }

    /// A tab with a `<dialog>` containing a `<form
    /// method="dialog">` with a valued submit button (`#sub`)
    /// and an unvalued one (`#plain`). No layout boxes: the
    /// click path only needs the document, like
    /// `popover_click_tab`.
    fn dialog_form_tab() -> (Tab, NodeId, NodeId, NodeId) {
        use std::cell::RefCell;
        use std::rc::Rc;
        let doc = Rc::new(RefCell::new(
            bowser_dom::Document::parse_html(
                b"<html><body><button id=\"out\">o</button>\
                  <dialog id=\"dlg\"><form method=\"dialog\">\
                  <button id=\"sub\" type=\"submit\" value=\"confirm\">ok</button>\
                  <button id=\"plain\" type=\"submit\">p</button>\
                  </form></dialog></body></html>",
            )
            .0,
        ));
        let (dlg, sub, plain) = {
            let d = doc.borrow();
            (
                d.get_element_by_id("dlg").expect("dlg"),
                d.get_element_by_id("sub").expect("sub"),
                d.get_element_by_id("plain").expect("plain"),
            )
        };
        let base = bowser_url::parse("http://example.com/page.html", None).expect("test url");
        let page =
            crate::script::PageVm::new_for_document(doc.clone(), &base, "TestBrowser/9.9", &[])
                .expect("install web apis");
        let mut tab = crate::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
        tab.install_document(doc, page);
        tab.dirty.layout = true;
        (tab, dlg, sub, plain)
    }

    #[test]
    fn w84_dialog_form_method_dialog_submit_closes_with_submitter_value() {
        // C7, expected per WHATWG HTML §4.11.4 / form
        // submission: submitting a form with method="dialog"
        // closes the dialog directly — no `cancel` event, one
        // `close` event, `returnValue` = the submitter's value
        // (`""` when the submitter has none). Enabled by w85:
        // the click now runs a real submit-activation path
        // (constraint validation, `submit` event, formmethod
        // override), not a click hook that closes dialogs.
        let (mut tab, dlg, sub, plain) = dialog_form_tab();
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); window.__ev = []; \
             d.addEventListener('cancel', function() { __ev.push('cancel'); }); \
             d.addEventListener('close', function() { __ev.push('close'); }); \
             d.showModal();",
        );
        let mut driver = TestDriver::new();
        driver.execute_click(&mut tab, sub).expect("click");
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "method=dialog submit must close the dialog"
        );
        assert!(
            js_true(&mut tab, "__ev.join(',') === 'close'"),
            "method=dialog submit must fire close only, no cancel"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'confirm'"),
            "returnValue must be the valued submitter's value"
        );
        eval(&mut tab, "d.showModal();");
        driver.execute_click(&mut tab, plain).expect("click");
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "unvalued submitter must close the dialog too"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === ''"),
            "returnValue must be empty for an unvalued submitter"
        );
    }

    /// A tab from arbitrary HTML, for the w85 submit-activation
    /// tests (each builds its own dialog/form shape). Same
    /// install pattern as [`dialog_form_tab`]: no layout boxes,
    /// the click path only needs the document.
    fn tab_from_html(html: &[u8]) -> Tab {
        use std::cell::RefCell;
        use std::rc::Rc;
        let doc = Rc::new(RefCell::new(bowser_dom::Document::parse_html(html).0));
        let base = bowser_url::parse("http://example.com/page.html", None).expect("test url");
        let page =
            crate::script::PageVm::new_for_document(doc.clone(), &base, "TestBrowser/9.9", &[])
                .expect("install web apis");
        let mut tab = crate::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
        tab.install_document(doc, page);
        tab.dirty.layout = true;
        tab
    }

    /// The element with `id` in `tab`'s document.
    fn el(tab: &Tab, id: &str) -> NodeId {
        tab.document_ref()
            .expect("document")
            .get_element_by_id(id)
            .unwrap_or_else(|| panic!("element #{id}"))
    }

    #[test]
    fn w85_dialog_formmethod_overrides_form_method_both_directions() {
        // Effective method = submitter `formmethod` when present,
        // else the form's `method` (ASCII case-insensitive). A
        // `formmethod="dialog"` on a `method="get"` form closes;
        // a `formmethod="post"` on a `method="dialog"` form does
        // not close — it fires `submit` and stops (navigation
        // deferred in w85), leaving the dialog open.
        let mut tab = tab_from_html(
            b"<html><body><dialog id=\"dlg1\"><form id=\"f1\" method=\"get\">\
              <button id=\"b1\" type=\"submit\" formmethod=\"dialog\" value=\"ok1\">ok</button>\
              </form></dialog>\
              <dialog id=\"dlg2\"><form id=\"f2\" method=\"dialog\">\
              <button id=\"b2\" type=\"submit\" formmethod=\"post\" value=\"x\">go</button>\
              </form></dialog></body></html>",
        );
        eval(
            &mut tab,
            "var d1 = document.getElementById('dlg1'); var d2 = document.getElementById('dlg2'); \
             window.__ev1 = []; window.__ev2 = []; window.__sub2 = 0; \
             d1.addEventListener('cancel', function() { window.__ev1.push('cancel'); }); \
             d1.addEventListener('close', function() { window.__ev1.push('close'); }); \
             d2.addEventListener('close', function() { window.__ev2.push('close'); }); \
             document.getElementById('f2').addEventListener('submit', function() { window.__sub2++; }); \
             d1.showModal();",
        );
        let mut driver = TestDriver::new();
        let b1 = el(&tab, "b1");
        driver.execute_click(&mut tab, b1).expect("click");
        assert!(
            !tab.document_ref()
                .expect("document")
                .is_dialog_open(el(&tab, "dlg1")),
            "formmethod=dialog must close even when the form method is get"
        );
        assert!(
            js_true(&mut tab, "window.__ev1.join(',') === 'close'"),
            "formmethod=dialog must fire close only, no cancel"
        );
        assert!(
            js_true(&mut tab, "d1.returnValue === 'ok1'"),
            "returnValue must be the overriding submitter's value"
        );
        eval(&mut tab, "d2.showModal();");
        let b2 = el(&tab, "b2");
        driver.execute_click(&mut tab, b2).expect("click");
        assert!(
            tab.document_ref()
                .expect("document")
                .is_dialog_open(el(&tab, "dlg2")),
            "formmethod=post must keep a method=dialog form's dialog open"
        );
        assert!(
            js_true(&mut tab, "window.__sub2 === 1"),
            "the non-dialog submit must still fire a submit event"
        );
        assert!(
            js_true(&mut tab, "window.__ev2.length === 0"),
            "the non-dialog submit must fire no close event"
        );
    }

    #[test]
    fn w85_dialog_input_submit_value_and_no_value() {
        // `<input type="submit">` is a submitter too. Pinned rule
        // (see `resolve_submit_activation`): returnValue is the
        // `value` attribute, or `""` when absent — for inputs as
        // for buttons. Browsers give a valueless input submit a
        // localized `"Submit"` default value; Bowser has no such
        // default anywhere today, so `""` is the pinned Bowser
        // behavior and the browser default is deferred, not
        // silently approximated.
        let mut tab = tab_from_html(
            b"<html><body><dialog id=\"dlg\"><form method=\"dialog\">\
              <input id=\"go\" type=\"submit\" value=\"go\">\
              <input id=\"bare\" type=\"submit\">\
              </form></dialog></body></html>",
        );
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); d.showModal();",
        );
        let mut driver = TestDriver::new();
        let go = el(&tab, "go");
        driver.execute_click(&mut tab, go).expect("click");
        assert!(
            !tab.document_ref()
                .expect("document")
                .is_dialog_open(el(&tab, "dlg")),
            "valued input submit must close the dialog"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'go'"),
            "returnValue must be the input submitter's value"
        );
        eval(&mut tab, "d.showModal();");
        let bare = el(&tab, "bare");
        driver.execute_click(&mut tab, bare).expect("click");
        assert!(
            !tab.document_ref()
                .expect("document")
                .is_dialog_open(el(&tab, "dlg")),
            "valueless input submit must close the dialog too"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === ''"),
            "returnValue must be empty for a valueless input submitter (pinned)"
        );
    }

    #[test]
    fn w85_dialog_form_no_ancestor_dialog_closes_nothing() {
        // A `method="dialog"` form with no dialog ancestor has
        // nothing to close: the submit event still fires, the
        // unrelated open dialog stays open with no `close`
        // event, and the form is not submitted anywhere (no
        // navigation in this wave). The testdriver click path
        // performs no navigation at all, so the pin here is
        // submit-fired plus nothing-closed; the windowed path
        // returns before link navigation by construction.
        let mut tab = tab_from_html(
            b"<html><body><dialog id=\"dlg\">hi</dialog>\
              <form id=\"f\" method=\"dialog\">\
              <button id=\"sub\" type=\"submit\" value=\"v\">ok</button>\
              </form></body></html>",
        );
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); window.__ev = []; window.__sub = 0; \
             d.addEventListener('close', function() { window.__ev.push('close'); }); \
             document.getElementById('f').addEventListener('submit', function() { window.__sub++; }); \
             d.showModal();",
        );
        let mut driver = TestDriver::new();
        let sub = el(&tab, "sub");
        driver.execute_click(&mut tab, sub).expect("click");
        assert!(
            js_true(&mut tab, "window.__sub === 1"),
            "the dialog-less submit must still fire a submit event"
        );
        assert!(
            tab.document_ref()
                .expect("document")
                .is_dialog_open(el(&tab, "dlg")),
            "an unrelated open dialog must stay open: the form has no dialog ancestor"
        );
        assert!(
            js_true(&mut tab, "window.__ev.length === 0"),
            "no close event may fire without a dialog ancestor"
        );
        assert!(
            tab.document_ref()
                .expect("document")
                .get_element_by_id("f")
                .is_some(),
            "the form must survive: no navigation replaced the document"
        );
    }

    #[test]
    fn w85_dialog_prevented_submit_keeps_dialog_open() {
        // A prevented `submit` event keeps the dialog open —
        // both via a listener calling `preventDefault()` and via
        // an inline `onsubmit` returning false. The submit event
        // must still have fired in the listener case.
        let mut tab = tab_from_html(
            b"<html><body><dialog id=\"dlg1\"><form id=\"f1\" method=\"dialog\">\
              <button id=\"b1\" type=\"submit\" value=\"v1\">ok</button>\
              </form></dialog>\
              <dialog id=\"dlg2\"><form id=\"f2\" method=\"dialog\" onsubmit=\"return false;\">\
              <button id=\"b2\" type=\"submit\" value=\"v2\">ok</button>\
              </form></dialog></body></html>",
        );
        eval(
            &mut tab,
            "var d1 = document.getElementById('dlg1'); var d2 = document.getElementById('dlg2'); \
             window.__ev1 = []; window.__ev2 = []; window.__sub1 = 0; \
             d1.addEventListener('close', function() { window.__ev1.push('close'); }); \
             d2.addEventListener('close', function() { window.__ev2.push('close'); }); \
             document.getElementById('f1').addEventListener('submit', function(e) { window.__sub1++; e.preventDefault(); }); \
             d1.showModal(); d2.showModal();",
        );
        let mut driver = TestDriver::new();
        let b1 = el(&tab, "b1");
        driver.execute_click(&mut tab, b1).expect("click");
        assert!(
            js_true(&mut tab, "window.__sub1 === 1"),
            "the prevented submit must still fire a submit event"
        );
        assert!(
            tab.document_ref()
                .expect("document")
                .is_dialog_open(el(&tab, "dlg1")),
            "a preventDefaulted submit must keep the dialog open"
        );
        assert!(
            js_true(&mut tab, "window.__ev1.length === 0"),
            "a prevented submit must fire no close event"
        );
        let b2 = el(&tab, "b2");
        driver.execute_click(&mut tab, b2).expect("click");
        assert!(
            tab.document_ref()
                .expect("document")
                .is_dialog_open(el(&tab, "dlg2")),
            "an onsubmit returning false must keep the dialog open"
        );
        assert!(
            js_true(&mut tab, "window.__ev2.length === 0"),
            "an inline-prevented submit must fire no close event"
        );
    }

    #[test]
    fn w85_dialog_non_submitter_and_disabled_do_not_submit() {
        // Neither a `type="button"` button nor a `disabled`
        // submit button activates submission: no `submit` event,
        // no close, dialog stays open.
        let mut tab = tab_from_html(
            b"<html><body><dialog id=\"dlg\"><form id=\"f\" method=\"dialog\">\
              <button id=\"plain\" type=\"button\" value=\"x\">no</button>\
              <button id=\"dis\" type=\"submit\" disabled value=\"y\">no</button>\
              </form></dialog></body></html>",
        );
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); window.__ev = []; window.__sub = 0; \
             d.addEventListener('close', function() { window.__ev.push('close'); }); \
             document.getElementById('f').addEventListener('submit', function() { window.__sub++; }); \
             d.showModal();",
        );
        let mut driver = TestDriver::new();
        let plain = el(&tab, "plain");
        driver.execute_click(&mut tab, plain).expect("click");
        let dis = el(&tab, "dis");
        driver.execute_click(&mut tab, dis).expect("click");
        assert!(
            tab.document_ref()
                .expect("document")
                .is_dialog_open(el(&tab, "dlg")),
            "non-submitter and disabled clicks must leave the dialog open"
        );
        assert!(
            js_true(&mut tab, "window.__sub === 0"),
            "neither click may fire a submit event"
        );
        assert!(
            js_true(&mut tab, "window.__ev.length === 0"),
            "neither click may fire a close event"
        );
    }

    #[test]
    fn w85_dialog_click_on_button_child_submits() {
        // Hit-testing targets the deepest child box, so a click
        // on a `<span>` inside the submit button must still
        // resolve the button as the submitter.
        let mut tab = tab_from_html(
            b"<html><body><dialog id=\"dlg\"><form method=\"dialog\">\
              <button id=\"sub\" type=\"submit\" value=\"inner\"><span id=\"inner\">ok</span></button>\
              </form></dialog></body></html>",
        );
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); d.showModal();",
        );
        let mut driver = TestDriver::new();
        let inner = el(&tab, "inner");
        driver.execute_click(&mut tab, inner).expect("click");
        assert!(
            !tab.document_ref()
                .expect("document")
                .is_dialog_open(el(&tab, "dlg")),
            "a click on the submit button's child must submit the form"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'inner'"),
            "returnValue must be the button's value even when a child was clicked"
        );
    }

    #[test]
    fn w85_dialog_form_attribute_owner_closes_form_dialog() {
        // The form owner comes from the `form` attribute when
        // present, even when the submitter sits outside the
        // dialog: the dialog closed is the FORM's nearest
        // ancestor dialog, not the submitter's.
        let mut tab = tab_from_html(
            b"<html><body><dialog id=\"dlg\"><form id=\"f\" method=\"dialog\"></form></dialog>\
              <button id=\"ext\" type=\"submit\" form=\"f\" value=\"ext\">ok</button></body></html>",
        );
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); d.showModal();",
        );
        let mut driver = TestDriver::new();
        let ext = el(&tab, "ext");
        driver.execute_click(&mut tab, ext).expect("click");
        assert!(
            !tab.document_ref()
                .expect("document")
                .is_dialog_open(el(&tab, "dlg")),
            "a form-attribute submitter outside the dialog must close the form's dialog"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'ext'"),
            "returnValue must be the outside submitter's value"
        );
    }

    /// A tab with layout boxes for a pointer-sequence submit
    /// test: `#sub` (0,0,100,50) and `#plain` (100,0,100,50)
    /// inside a `method="dialog"` form in `#dlg`. Same
    /// document shape as [`dialog_form_tab`], plus boxes so
    /// pointer hit-testing can find the buttons.
    fn dialog_form_seq_tab() -> (Tab, NodeId, NodeId, NodeId) {
        use std::cell::RefCell;
        use std::rc::Rc;
        let doc = Rc::new(RefCell::new(
            bowser_dom::Document::parse_html(
                b"<html><body><button id=\"out\">o</button>\
                  <dialog id=\"dlg\"><form method=\"dialog\">\
                  <button id=\"sub\" type=\"submit\" value=\"confirm\">ok</button>\
                  <button id=\"plain\" type=\"submit\">p</button>\
                  </form></dialog></body></html>",
            )
            .0,
        ));
        let (dlg, sub, plain) = {
            let d = doc.borrow();
            (
                d.get_element_by_id("dlg").expect("dlg"),
                d.get_element_by_id("sub").expect("sub"),
                d.get_element_by_id("plain").expect("plain"),
            )
        };
        let base = bowser_url::parse("http://example.com/page.html", None).expect("test url");
        let page =
            crate::script::PageVm::new_for_document(doc.clone(), &base, "TestBrowser/9.9", &[])
                .expect("install web apis");
        let mut tab = crate::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
        tab.install_document(doc, page);
        tab.layout_root = Some(Rc::new(actions_box(
            (0.0, 0.0, 800.0, 600.0),
            None,
            vec![
                actions_box((0.0, 0.0, 100.0, 50.0), Some(sub), vec![]),
                actions_box((100.0, 0.0, 100.0, 50.0), Some(plain), vec![]),
            ],
        )));
        (tab, dlg, sub, plain)
    }

    #[test]
    fn w85fix_pointer_sequence_submit_closes_with_submitter_value() {
        // Mirror of
        // `w84_dialog_form_method_dialog_submit_closes_with_submitter_value`,
        // through a real pointer down/up sequence instead of
        // `execute_click`: the pointer-up arm used to
        // synthesize the click and run popover activation,
        // then stop, so a `method="dialog"` submit fired no
        // `submit` event and closed nothing.
        let (mut tab, dlg, _sub, _plain) = dialog_form_seq_tab();
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); window.__ev = []; window.__sub = 0; \
             d.addEventListener('cancel', function() { __ev.push('cancel'); }); \
             d.addEventListener('close', function() { __ev.push('close'); }); \
             document.querySelector('form').addEventListener('submit', function() { window.__sub++; }); \
             d.showModal();",
        );
        let click_at = |x: f64, y: f64| {
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(x, y, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )])
        };
        let mut driver = TestDriver::new();
        driver.execute(&mut tab, click_at(50.0, 25.0));
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "pointer-sequence method=dialog submit must close the dialog"
        );
        assert!(
            js_true(&mut tab, "__ev.join(',') === 'close'"),
            "pointer-sequence submit must fire close only, no cancel"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'confirm'"),
            "returnValue must be the valued submitter's value"
        );
        assert!(
            js_true(&mut tab, "window.__sub === 1"),
            "pointer-sequence submit must fire exactly one submit event"
        );
        eval(&mut tab, "d.showModal();");
        driver.execute(&mut tab, click_at(150.0, 25.0));
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "unvalued pointer-sequence submit must close the dialog too"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === ''"),
            "returnValue must be empty for an unvalued submitter"
        );
    }

    /// A tab with layout boxes for a two-popover pointer
    /// test: `#btnA` (0,0,100,50) targets `#popA`,
    /// `#btnB` (0,60,100,50) targets `#popB`, `#popA`
    /// (200,200,100,100), `#popB` (350,200,100,100).
    /// `#popB` is nested inside `#popA`, so showing both
    /// autos does not auto-close the first.
    fn two_popover_seq_tab() -> (Tab, NodeId, NodeId, NodeId) {
        use std::cell::RefCell;
        use std::rc::Rc;
        let doc = Rc::new(RefCell::new(
            bowser_dom::Document::parse_html(
                b"<html><body><button id=\"btnA\" popovertarget=\"popA\">a</button>\
                  <button id=\"btnB\" popovertarget=\"popB\">b</button>\
                  <div id=\"popA\" popover>hi<div id=\"popB\" popover>hi</div></div></body></html>",
            )
            .0,
        ));
        let (btn_a, btn_b, pop_a, pop_b) = {
            let d = doc.borrow();
            (
                d.get_element_by_id("btnA").expect("btnA"),
                d.get_element_by_id("btnB").expect("btnB"),
                d.get_element_by_id("popA").expect("popA"),
                d.get_element_by_id("popB").expect("popB"),
            )
        };
        let base = bowser_url::parse("http://example.com/page.html", None).expect("test url");
        let page =
            crate::script::PageVm::new_for_document(doc.clone(), &base, "TestBrowser/9.9", &[])
                .expect("install web apis");
        let mut tab = crate::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
        tab.install_document(doc, page);
        tab.layout_root = Some(Rc::new(actions_box(
            (0.0, 0.0, 800.0, 600.0),
            None,
            vec![
                actions_box((0.0, 0.0, 100.0, 50.0), Some(btn_a), vec![]),
                actions_box((0.0, 60.0, 100.0, 50.0), Some(btn_b), vec![]),
                actions_box((200.0, 200.0, 100.0, 100.0), Some(pop_a), vec![]),
                actions_box((350.0, 200.0, 100.0, 100.0), Some(pop_b), vec![]),
            ],
        )));
        (tab, btn_a, pop_a, pop_b)
    }

    #[test]
    fn w85fix_pointer_lower_invoker_closes_both_exactly_once() {
        // Pointer-sequence twin of the windowed lower-invoker
        // test: a click on the lower auto's invoker
        // light-dismisses only the auto above it, then
        // toggles the lower auto — both closed, one toggle
        // each, upper first.
        let (mut tab, _btn_a, pop_a, pop_b) = two_popover_seq_tab();
        open_popover_td(&mut tab, pop_a);
        open_popover_td(&mut tab, pop_b);
        eval(
            &mut tab,
            "window.__toggles = []; \
             document.getElementById('popA').addEventListener('toggle', function() { window.__toggles.push('a'); }); \
             document.getElementById('popB').addEventListener('toggle', function() { window.__toggles.push('b'); });",
        );
        let mut driver = TestDriver::new();
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(50.0, 25.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            !popover_open(&tab, pop_a),
            "lower auto must end closed after its own invoker was clicked"
        );
        assert!(
            !popover_open(&tab, pop_b),
            "upper auto must end closed"
        );
        assert!(
            js_true(&mut tab, "window.__toggles.join(',') === 'b,a'"),
            "exactly one toggle per popover, light-dismiss victim before activation"
        );

        // Press on the lower invoker, release outside:
        // ancestors mismatch, so nothing dismisses.
        let (mut tab, _, pop_a, pop_b) = two_popover_seq_tab();
        open_popover_td(&mut tab, pop_a);
        open_popover_td(&mut tab, pop_b);
        eval(
            &mut tab,
            "window.__toggles = []; \
             document.getElementById('popA').addEventListener('toggle', function() { window.__toggles.push('a'); }); \
             document.getElementById('popB').addEventListener('toggle', function() { window.__toggles.push('b'); });",
        );
        let mut driver = TestDriver::new();
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(50.0, 25.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    move_item(700.0, 500.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            popover_open(&tab, pop_a),
            "press-on-invoker/release-outside must not dismiss the lower auto"
        );
        assert!(
            popover_open(&tab, pop_b),
            "press-on-invoker/release-outside must not dismiss the upper auto"
        );
        assert!(
            js_true(&mut tab, "window.__toggles.length === 0"),
            "no toggle may fire on a mismatched press/release pair"
        );
    }

    #[test]
    fn w85fix_submit_listener_mutates_value_and_method_mid_submit() {
        // A `submit` listener mutating the submitter must
        // change the close: the live value (not the
        // pre-event snapshot) becomes `returnValue`, and a
        // flipped effective method decides whether the
        // dialog closes at all, in both directions.
        let mut tab = tab_from_html(
            b"<html><body><dialog id=\"dlg\"><form id=\"f\" method=\"dialog\">\
              <button id=\"sub\" type=\"submit\" value=\"orig\">ok</button>\
              </form></dialog></body></html>",
        );
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); \
             document.getElementById('f').addEventListener('submit', function() { \
               document.getElementById('sub').setAttribute('value', 'mutated'); }); \
             d.showModal();",
        );
        let mut driver = TestDriver::new();
        let click_node = el(&tab, "sub");
        driver.execute_click(&mut tab, click_node).expect("click");
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(el(&tab, "dlg")),
            "value-mutated submit must still close"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'mutated'"),
            "returnValue must be the listener-mutated value, not the pre-event one"
        );

        // Dialog -> non-dialog flip: the dialog stays open.
        let mut tab = tab_from_html(
            b"<html><body><dialog id=\"dlg\"><form id=\"f\" method=\"dialog\">\
              <button id=\"sub\" type=\"submit\" formmethod=\"dialog\" value=\"x\">ok</button>\
              </form></dialog></body></html>",
        );
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); window.__sub = 0; window.__close = 0; \
             d.addEventListener('close', function() { window.__close++; }); \
             document.getElementById('f').addEventListener('submit', function() { \
               window.__sub++; \
               document.getElementById('sub').setAttribute('formmethod', 'post'); }); \
             d.showModal();",
        );
        let mut driver = TestDriver::new();
        let click_node = el(&tab, "sub");
        driver.execute_click(&mut tab, click_node).expect("click");
        assert!(
            tab.document_ref().expect("document").is_dialog_open(el(&tab, "dlg")),
            "a listener flipping formmethod away from dialog must keep the dialog open"
        );
        assert!(
            js_true(&mut tab, "window.__sub === 1 && window.__close === 0"),
            "the flipped submit must fire submit and no close"
        );

        // Non-dialog -> dialog flip: the dialog closes with
        // the mutated value.
        let mut tab = tab_from_html(
            b"<html><body><dialog id=\"dlg\"><form id=\"f\" method=\"post\">\
              <button id=\"sub\" type=\"submit\" value=\"before\">ok</button>\
              </form></dialog></body></html>",
        );
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); \
             document.getElementById('f').addEventListener('submit', function() { \
               var s = document.getElementById('sub'); \
               s.setAttribute('formmethod', 'dialog'); s.setAttribute('value', 'flipped'); }); \
             d.showModal();",
        );
        let mut driver = TestDriver::new();
        let click_node = el(&tab, "sub");
        driver.execute_click(&mut tab, click_node).expect("click");
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(el(&tab, "dlg")),
            "a listener flipping formmethod to dialog must close the dialog"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'flipped'"),
            "returnValue must be the listener-mutated value on a flipped-to-dialog submit"
        );
    }

    #[test]
    fn w85fix_dialog_fieldset_disabled_and_uppercase_click() {
        // End-to-end twin of the resolve-level test: a
        // fieldset-disabled submitter fires no `submit` and
        // closes nothing; its first-legend sibling closes;
        // uppercase `formmethod` / `method` close.
        let mut tab = tab_from_html(
            b"<html><body><dialog id=\"dlg\"><form id=\"f\" method=\"dialog\">\
              <fieldset disabled><button id=\"dis\" type=\"submit\" value=\"x\">x</button></fieldset>\
              </form></dialog></body></html>",
        );
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); window.__sub = 0; window.__close = 0; \
             d.addEventListener('close', function() { window.__close++; }); \
             document.getElementById('f').addEventListener('submit', function() { window.__sub++; }); \
             d.showModal();",
        );
        let mut driver = TestDriver::new();
        let click_node = el(&tab, "dis");
        driver.execute_click(&mut tab, click_node).expect("click");
        assert!(
            tab.document_ref().expect("document").is_dialog_open(el(&tab, "dlg")),
            "a fieldset-disabled submitter must leave the dialog open"
        );
        assert!(
            js_true(&mut tab, "window.__sub === 0 && window.__close === 0"),
            "a fieldset-disabled submitter must fire neither submit nor close"
        );

        let mut tab = tab_from_html(
            b"<html><body><dialog id=\"dlg\"><form method=\"dialog\">\
              <fieldset disabled><legend><button id=\"leg\" type=\"submit\" value=\"leg\">x</button></legend></fieldset>\
              </form></dialog></body></html>",
        );
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); d.showModal();",
        );
        let mut driver = TestDriver::new();
        let click_node = el(&tab, "leg");
        driver.execute_click(&mut tab, click_node).expect("click");
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(el(&tab, "dlg")),
            "a first-legend submitter escapes the fieldset's disabled"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'leg'"),
            "returnValue must be the first-legend submitter's value"
        );

        let mut tab = tab_from_html(
            b"<html><body><dialog id=\"dlg1\"><form method=\"get\">\
              <button id=\"b1\" type=\"submit\" formmethod=\"DIALOG\" value=\"u\">u</button></form></dialog>\
              <dialog id=\"dlg2\"><form method=\"DIALOG\">\
              <button id=\"b2\" type=\"submit\" value=\"v\">v</button></form></dialog></body></html>",
        );
        eval(
            &mut tab,
            "var d1 = document.getElementById('dlg1'); var d2 = document.getElementById('dlg2'); \
             d1.showModal();",
        );
        let mut driver = TestDriver::new();
        let click_node = el(&tab, "b1");
        driver.execute_click(&mut tab, click_node).expect("click");
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(el(&tab, "dlg1")),
            "formmethod=DIALOG (uppercase) must close"
        );
        eval(&mut tab, "d2.showModal();");
        let click_node = el(&tab, "b2");
        driver.execute_click(&mut tab, click_node).expect("click");
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(el(&tab, "dlg2")),
            "method=DIALOG (uppercase) must close"
        );
    }

    #[test]
    fn w85fix_request_close_closedby_none_closes() {
        // `requestClose()` never consults `computed_closed_by`:
        // a `closedby="none"` dialog — which blocks Escape
        // and light-dismiss — still closes through its close
        // watcher, modal and non-modal alike.
        let mut tab = tab_from_html(
            b"<html><body><dialog id=\"dlg\" closedby=\"none\"><button id=\"a\">a</button></dialog></body></html>",
        );
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); window.__log = []; \
             d.addEventListener('cancel', function(e) { __log.push('cancel:' + e.isTrusted + ':' + e.cancelable); }); \
             d.addEventListener('close', function(e) { __log.push('close:' + e.isTrusted + ':' + e.cancelable); }); \
             d.showModal(); d.requestClose('rv');",
        );
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(el(&tab, "dlg")),
            "requestClose must close a closedby=none modal dialog"
        );
        assert!(
            js_true(&mut tab, "__log.join(';') === 'cancel:true:true;close:true:false'"),
            "closedby=none requestClose must fire cancel then close"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'rv'"),
            "closedby=none requestClose must apply its argument"
        );
        eval(&mut tab, "d.show(); d.requestClose('nm');");
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(el(&tab, "dlg")),
            "requestClose must close a closedby=none non-modal dialog too"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'nm'"),
            "closedby=none non-modal requestClose must apply its argument"
        );
    }

    /// A tab with layout boxes for a dual light-dismiss
    /// test: `#dlg` at (200,200,100,100) in an 800x600
    /// viewport, so a pointer at (700,500) hits nothing and
    /// falls back to the body.
    fn dual_seq_tab(html: &str) -> (Tab, NodeId) {
        use std::cell::RefCell;
        use std::rc::Rc;
        let doc = Rc::new(RefCell::new(
            bowser_dom::Document::parse_html(html.as_bytes()).0,
        ));
        let dlg = doc
            .borrow()
            .get_element_by_id("dlg")
            .expect("dlg");
        let base = bowser_url::parse("http://example.com/page.html", None).expect("test url");
        let page =
            crate::script::PageVm::new_for_document(doc.clone(), &base, "TestBrowser/9.9", &[])
                .expect("install web apis");
        let mut tab = crate::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
        tab.install_document(doc, page);
        tab.layout_root = Some(Rc::new(actions_box(
            (0.0, 0.0, 800.0, 600.0),
            None,
            vec![actions_box((200.0, 200.0, 100.0, 100.0), Some(dlg), vec![])],
        )));
        (tab, dlg)
    }

    #[test]
    fn w85fix_dual_escape_fires_toggle_after_close() {
        // JS-level twin of the windowed Escape dual test,
        // with event order: a manual dual closed by Escape
        // fires cancel, then close, then its popover toggle
        // (`open` -> `closed`) — the toggle belongs to the
        // close, after it. Manual state reaches the dialog
        // arm directly (an auto dual on top would be
        // Escape's popover target instead).
        let mut tab = tab_from_html(
            b"<html><body><dialog id=\"dlg\" popover=\"manual\" closedby=\"closerequest\">\
              <button id=\"a\">a</button></dialog></body></html>",
        );
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); window.__log = []; \
             d.addEventListener('cancel', function() { __log.push('cancel'); }); \
             d.addEventListener('close', function() { __log.push('close'); }); \
             d.addEventListener('toggle', function() { __log.push('toggle'); }); \
             d.show(); d.showPopover(); window.__log = [];",
        );
        let mut driver = TestDriver::new();
        driver.execute(
            &mut tab,
            sequence_request(vec![key_source(vec![
                key_item(ActionItemKind::KeyDown, "Escape"),
                key_item(ActionItemKind::KeyUp, "Escape"),
            ])]),
        );
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(el(&tab, "dlg")),
            "Escape must close the dual dialog"
        );
        assert!(
            !tab.document_ref()
                .expect("document")
                .is_popover_open(el(&tab, "dlg")),
            "Escape must end the dual's popover membership"
        );
        assert!(
            js_true(&mut tab, "window.__log.join(',') === 'cancel,close,toggle'"),
            "Escape-closed dual must fire cancel, close, then exactly one toggle, in that order"
        );

        // Modal dual excluded (C10): Escape closes it with
        // cancel and close only, no toggle.
        let mut tab = tab_from_html(
            b"<html><body><dialog id=\"dlg\" popover=\"auto\"><button id=\"a\">a</button></dialog></body></html>",
        );
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); window.__log = []; \
             d.addEventListener('cancel', function() { __log.push('cancel'); }); \
             d.addEventListener('close', function() { __log.push('close'); }); \
             d.addEventListener('toggle', function() { __log.push('toggle'); }); \
             d.showModal(); window.__log = [];",
        );
        let mut driver = TestDriver::new();
        driver.execute(
            &mut tab,
            sequence_request(vec![key_source(vec![
                key_item(ActionItemKind::KeyDown, "Escape"),
                key_item(ActionItemKind::KeyUp, "Escape"),
            ])]),
        );
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(el(&tab, "dlg")),
            "Escape must close the modal dual dialog"
        );
        assert!(
            js_true(&mut tab, "window.__log.join(',') === 'cancel,close'"),
            "a modal dual's Escape close must fire no popover toggle"
        );
    }

    #[test]
    fn w85fix_dual_light_dismiss_fires_toggle_after_close() {
        // Manual dual, `closedby=any`, outside pointer
        // sequence: cancel, close, then the dual toggle.
        let (mut tab, dlg) = dual_seq_tab(
            "<html><body><dialog id=\"dlg\" popover=\"manual\" closedby=\"any\">\
             <button id=\"a\">a</button></dialog></body></html>",
        );
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); window.__log = []; \
             d.addEventListener('cancel', function() { __log.push('cancel'); }); \
             d.addEventListener('close', function() { __log.push('close'); }); \
             d.addEventListener('toggle', function() { __log.push('toggle'); }); \
             d.show(); d.showPopover(); window.__log = [];",
        );
        let mut driver = TestDriver::new();
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(700.0, 500.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "light-dismiss must close the dual dialog"
        );
        assert!(
            !tab.document_ref().expect("document").is_popover_open(dlg),
            "light-dismiss must end the dual's popover membership"
        );
        assert!(
            js_true(&mut tab, "window.__log.join(',') === 'cancel,close,toggle'"),
            "light-dismiss-closed dual must fire cancel, close, then exactly one toggle, in that order"
        );

        // Auto dual, outside pointer sequence: the popover
        // arm hides it first (one toggle), so the dialog
        // arm's snapshot is no longer live and must not
        // fire a second toggle for the same close.
        let (mut tab, dlg) = dual_seq_tab(
            "<html><body><dialog id=\"dlg\" popover=\"auto\" closedby=\"any\">\
             <button id=\"a\">a</button></dialog></body></html>",
        );
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); window.__log = []; \
             d.addEventListener('cancel', function() { __log.push('cancel'); }); \
             d.addEventListener('close', function() { __log.push('close'); }); \
             d.addEventListener('toggle', function() { __log.push('toggle'); }); \
             d.show(); d.showPopover(); window.__log = [];",
        );
        let mut driver = TestDriver::new();
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(700.0, 500.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "light-dismiss must close the auto dual dialog"
        );
        assert!(
            js_true(&mut tab, "window.__log.join(',') === 'toggle,cancel,close'"),
            "an auto dual light-dismissed outside must toggle exactly once (the popover hide first), with no second close toggle"
        );
    }

    #[test]
    fn w85fix_td_light_dismiss_defers_popover_focus_clear_past_dialog_close() {
        // Testdriver twin of the windowed deferral: one
        // outside gesture dismisses both an auto popover
        // and a `closedby=any` dialog. The popover's focus
        // clear (observable as the popover's `blur`, fired
        // by the executor's `set_focus(None)`) must land
        // AFTER the dialog's `cancel` and `close`, so a
        // dialog listener observes pre-dismissal focus —
        // the old inline clear blurred before `cancel`.
        // The popover and dialog are siblings, so the
        // dialog arm's own post-close clear (inside the
        // dialog only) cannot mask the popover clear.
        let html = "<html><body><button id=\"btn\" popovertarget=\"pop\">\
             <span id=\"span\">t</span></button>\
             <div id=\"pop\" popover>hi</div>\
             <dialog id=\"dlg\" closedby=\"any\"><button id=\"a\">a</button></dialog></body></html>";
        let (mut tab, _, pop) = popover_seq_tab(html);
        let dlg = el(&tab, "dlg");
        // Dialog first: `show_dialog` hides open autos, so
        // showing it after the popover would hide the
        // popover before the gesture (windowed tests use
        // the same dialog-then-popover order).
        tab.installed_mut()
            .expect("installed")
            .document()
            .borrow_mut()
            .show_dialog(dlg)
            .expect("show dialog");
        open_popover_td(&mut tab, pop);
        eval(
            &mut tab,
            "window.__log = []; \
             document.getElementById('pop').addEventListener('blur', function() { __log.push('blur'); }); \
             var d = document.getElementById('dlg'); \
             d.addEventListener('cancel', function() { __log.push('cancel'); }); \
             d.addEventListener('close', function() { __log.push('close'); });",
        );
        let mut driver = TestDriver::new();
        driver.actions.focused = Some(pop);
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(700.0, 500.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            !popover_open(&tab, pop),
            "outside gesture must light-dismiss the popover"
        );
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "outside gesture must light-dismiss the closedby=any dialog"
        );
        assert!(
            js_true(&mut tab, "window.__log.join(',') === 'cancel,close,blur'"),
            "popover blur must fire after the dialog cancel and close, not before cancel"
        );
        assert_eq!(
            driver.actions.focused, None,
            "deferred popover focus clear must still run after the dialog arm"
        );

        // Re-show half: a dialog `close` listener that
        // re-shows the dismissed popover keeps its focus —
        // the deferred clear's live-membership check must
        // skip a live-again popover, so no blur fires.
        let (mut tab, _, pop) = popover_seq_tab(html);
        let dlg = el(&tab, "dlg");
        // Dialog first: `show_dialog` hides open autos, so
        // showing it after the popover would hide the
        // popover before the gesture (windowed tests use
        // the same dialog-then-popover order).
        tab.installed_mut()
            .expect("installed")
            .document()
            .borrow_mut()
            .show_dialog(dlg)
            .expect("show dialog");
        open_popover_td(&mut tab, pop);
        eval(
            &mut tab,
            "window.__log = []; \
             document.getElementById('pop').addEventListener('blur', function() { __log.push('blur'); }); \
             var d = document.getElementById('dlg'); \
             d.addEventListener('cancel', function() { __log.push('cancel'); }); \
             d.addEventListener('close', function() { __log.push('close'); \
               document.getElementById('pop').showPopover(); });",
        );
        let mut driver = TestDriver::new();
        driver.actions.focused = Some(pop);
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(700.0, 500.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            popover_open(&tab, pop),
            "close listener must have re-shown the popover"
        );
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "outside gesture must light-dismiss the closedby=any dialog"
        );
        assert!(
            js_true(&mut tab, "window.__log.join(',') === 'cancel,close'"),
            "no blur must fire for a re-shown popover"
        );
        assert_eq!(
            driver.actions.focused,
            Some(pop),
            "deferred clear must not wipe focus inside a re-shown popover"
        );
    }

    #[test]
    fn w85fix_td_nested_popover_in_dialog_close_listener_reshow_keeps_focus() {
        // w85fix BLOCKER A: a popover nested inside the
        // dialog, re-shown by the dialog's `close`
        // listener with focus on its child, must keep
        // that focus. Without the guard, the dialog arm's
        // post-close clear wipes it — the popover child is
        // a DOM descendant of the dialog, so the plain
        // `clear_focus_inside(dialog)` matches it — even
        // though the deferred loop's re-show check would
        // have skipped the popover's own clear. The
        // listener's `focus()` call is in the script for
        // intent, but the pin is the executor focus on the
        // popover child at clear time: it must survive
        // with no blur.
        let html = "<html><body><button id=\"btn\" popovertarget=\"pop\">\
             <span id=\"span\">t</span></button>\
             <dialog id=\"dlg\" closedby=\"any\"><button id=\"a\">a</button>\
             <div id=\"pop\" popover><button id=\"pc\">pc</button></div></dialog></body></html>";
        let (mut tab, _, pop) = popover_seq_tab(html);
        let dlg = el(&tab, "dlg");
        let pc = el(&tab, "pc");
        tab.installed_mut()
            .expect("installed")
            .document()
            .borrow_mut()
            .show_dialog(dlg)
            .expect("show dialog");
        open_popover_td(&mut tab, pop);
        eval(
            &mut tab,
            "window.__log = []; \
             document.getElementById('pc').addEventListener('blur', function() { __log.push('blur'); }); \
             var d = document.getElementById('dlg'); \
             d.addEventListener('cancel', function() { __log.push('cancel'); }); \
             d.addEventListener('close', function() { __log.push('close'); \
               document.getElementById('pop').showPopover(); \
               document.getElementById('pc').focus(); });",
        );
        let mut driver = TestDriver::new();
        driver.actions.focused = Some(pc);
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(700.0, 500.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            popover_open(&tab, pop),
            "close listener must have re-shown the nested popover"
        );
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "outside gesture must light-dismiss the closedby=any dialog"
        );
        assert!(
            js_true(&mut tab, "window.__log.join(',') === 'cancel,close'"),
            "no blur must fire for focus inside a re-shown nested popover"
        );
        assert_eq!(
            driver.actions.focused,
            Some(pc),
            "dialog-arm clear must not wipe focus inside a live popover nested in the dialog"
        );
    }

    #[test]
    fn w85fix_td_nested_popover_close_listener_focus_move_limit() {
        // w85fix S3, testdriver half — stated as a limit,
        // not as moved-in coverage: focus starts on a
        // dialog child, and the `close` listener re-shows
        // the nested popover and calls `focus()` on its
        // child. But `HTMLElement.focus()` is a no-op in
        // the bindings (`n_element_focus` returns without
        // touching executor focus), so no JS listener can
        // move executor focus mid-close in testdriver:
        // the dialog child's focus is correctly cleared
        // (it must not linger on the closed dialog), and
        // the re-shown popover stays open. The guard's
        // moved-in condition — focus inside a live
        // popover at clear time — is pinned by the
        // sibling test above, where focus is on the
        // popover child when the dialog arm clears.
        let html = "<html><body><button id=\"btn\" popovertarget=\"pop\">\
             <span id=\"span\">t</span></button>\
             <dialog id=\"dlg\" closedby=\"any\"><button id=\"a\">a</button>\
             <div id=\"pop\" popover><button id=\"pc\">pc</button></div></dialog></body></html>";
        let (mut tab, _, pop) = popover_seq_tab(html);
        let dlg = el(&tab, "dlg");
        let a = el(&tab, "a");
        tab.installed_mut()
            .expect("installed")
            .document()
            .borrow_mut()
            .show_dialog(dlg)
            .expect("show dialog");
        open_popover_td(&mut tab, pop);
        eval(
            &mut tab,
            "window.__log = []; \
             document.getElementById('a').addEventListener('blur', function() { __log.push('blur'); }); \
             var d = document.getElementById('dlg'); \
             d.addEventListener('cancel', function() { __log.push('cancel'); }); \
             d.addEventListener('close', function() { __log.push('close'); \
               document.getElementById('pop').showPopover(); \
               document.getElementById('pc').focus(); });",
        );
        let mut driver = TestDriver::new();
        driver.actions.focused = Some(a);
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(700.0, 500.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            popover_open(&tab, pop),
            "close listener must have re-shown the nested popover"
        );
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "outside gesture must light-dismiss the closedby=any dialog"
        );
        assert!(
            js_true(&mut tab, "window.__log.join(',') === 'cancel,close,blur'"),
            "the dialog child's blur must fire after cancel and close, since JS focus() cannot move executor focus into the popover mid-close"
        );
        assert_eq!(
            driver.actions.focused, None,
            "focus on a dialog child must not linger on the closed dialog"
        );
    }

    #[test]
    fn w85fix_td_light_dismiss_deferred_clear_retained_and_prevented_cancel() {
        // w85fix S2: the halves the deferred loop's comment
        // claims but the plain deferral test does not pin.
        // Retained half: a victim with a stored exit
        // duration is exit-retained at deferred-clear time
        // (not live), so the clear must still run.
        // Retained-then-reshown half: a retained victim
        // the `close` listener re-shows is live again, so
        // the clear must skip it. Prevented-cancel half:
        // the dialog arm runs nothing when `cancel` is
        // prevented, but the popover still hid, so the
        // deferred clear must still run.
        let html = "<html><body><button id=\"btn\" popovertarget=\"pop\">\
             <span id=\"span\">t</span></button>\
             <div id=\"pop\" popover>hi</div>\
             <dialog id=\"dlg\" closedby=\"any\"><button id=\"a\">a</button></dialog></body></html>";
        let gesture = || {
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(700.0, 500.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )])
        };
        // Retained half.
        let (mut tab, _, pop) = popover_seq_tab(html);
        let dlg = el(&tab, "dlg");
        tab.installed_mut()
            .expect("installed")
            .document()
            .borrow_mut()
            .show_dialog(dlg)
            .expect("show dialog");
        open_popover_td(&mut tab, pop);
        tab.installed_mut()
            .expect("installed")
            .document()
            .borrow_mut()
            .store_exit_duration(pop, 60.0);
        eval(
            &mut tab,
            "window.__log = []; \
             document.getElementById('pop').addEventListener('blur', function() { __log.push('blur'); }); \
             var d = document.getElementById('dlg'); \
             d.addEventListener('cancel', function() { __log.push('cancel'); }); \
             d.addEventListener('close', function() { __log.push('close'); });",
        );
        let mut driver = TestDriver::new();
        driver.actions.focused = Some(pop);
        driver.execute(&mut tab, gesture());
        let (retained, live, dialog_open) = {
            let doc = tab.document_ref().expect("document");
            (
                doc.is_exit_retained(pop),
                doc.top_layer().contains(&pop),
                doc.is_dialog_open(dlg),
            )
        };
        assert!(
            retained,
            "stored duration must retain the dismissed popover"
        );
        assert!(
            !live,
            "a retained victim is not a live top-layer member"
        );
        assert!(
            !dialog_open,
            "outside gesture must light-dismiss the closedby=any dialog"
        );
        assert!(
            js_true(&mut tab, "window.__log.join(',') === 'cancel,close,blur'"),
            "focus inside a retained victim must end cleared, after cancel and close"
        );
        assert_eq!(
            driver.actions.focused, None,
            "deferred clear must run for a retained victim"
        );

        // Retained-then-reshown half.
        let (mut tab, _, pop) = popover_seq_tab(html);
        let dlg = el(&tab, "dlg");
        tab.installed_mut()
            .expect("installed")
            .document()
            .borrow_mut()
            .show_dialog(dlg)
            .expect("show dialog");
        open_popover_td(&mut tab, pop);
        tab.installed_mut()
            .expect("installed")
            .document()
            .borrow_mut()
            .store_exit_duration(pop, 60.0);
        eval(
            &mut tab,
            "window.__log = []; \
             document.getElementById('pop').addEventListener('blur', function() { __log.push('blur'); }); \
             var d = document.getElementById('dlg'); \
             d.addEventListener('cancel', function() { __log.push('cancel'); }); \
             d.addEventListener('close', function() { __log.push('close'); \
               document.getElementById('pop').showPopover(); });",
        );
        let mut driver = TestDriver::new();
        driver.actions.focused = Some(pop);
        driver.execute(&mut tab, gesture());
        let (live, retained) = {
            let doc = tab.document_ref().expect("document");
            (doc.top_layer().contains(&pop), doc.is_exit_retained(pop))
        };
        assert!(
            live,
            "close listener must have re-shown the retained popover to live membership"
        );
        assert!(
            !retained,
            "re-show must cancel the exit retention"
        );
        assert!(
            js_true(&mut tab, "window.__log.join(',') === 'cancel,close'"),
            "no blur must fire for a retained-then-reshown popover"
        );
        assert_eq!(
            driver.actions.focused,
            Some(pop),
            "deferred clear must not wipe focus inside a retained-then-reshown popover"
        );

        // Prevented-cancel half.
        let (mut tab, _, pop) = popover_seq_tab(html);
        let dlg = el(&tab, "dlg");
        tab.installed_mut()
            .expect("installed")
            .document()
            .borrow_mut()
            .show_dialog(dlg)
            .expect("show dialog");
        open_popover_td(&mut tab, pop);
        eval(
            &mut tab,
            "window.__log = []; \
             document.getElementById('pop').addEventListener('blur', function() { __log.push('blur'); }); \
             var d = document.getElementById('dlg'); \
             d.addEventListener('cancel', function(e) { __log.push('cancel'); e.preventDefault(); }); \
             d.addEventListener('close', function() { __log.push('close'); });",
        );
        let mut driver = TestDriver::new();
        driver.actions.focused = Some(pop);
        driver.execute(&mut tab, gesture());
        assert!(
            !popover_open(&tab, pop),
            "popover must light-dismiss even when the dialog cancel is prevented"
        );
        assert!(
            tab.document_ref().expect("document").is_dialog_open(dlg),
            "prevented cancel must keep the dialog open"
        );
        assert!(
            js_true(&mut tab, "window.__log.join(',') === 'cancel,blur'"),
            "deferred clear must run after a prevented cancel, with no close"
        );
        assert_eq!(
            driver.actions.focused, None,
            "deferred popover focus clear must run even when cancel is prevented"
        );
    }

    #[test]
    fn w85fix_td_light_dismiss_deferred_clear_child_blur_order() {
        // w85fix S4: the blur-order pin with a focusable
        // child inside the popover, not the plain
        // non-focusable popover div itself. Focus sits on
        // the child, so the deferred clear's `blur` fires
        // on the child and must still land after the
        // dialog's `cancel` and `close`.
        let html = "<html><body><button id=\"btn\" popovertarget=\"pop\">\
             <span id=\"span\">t</span></button>\
             <div id=\"pop\" popover><button id=\"pc\">pc</button></div>\
             <dialog id=\"dlg\" closedby=\"any\"><button id=\"a\">a</button></dialog></body></html>";
        let (mut tab, _, pop) = popover_seq_tab(html);
        let dlg = el(&tab, "dlg");
        let pc = el(&tab, "pc");
        tab.installed_mut()
            .expect("installed")
            .document()
            .borrow_mut()
            .show_dialog(dlg)
            .expect("show dialog");
        open_popover_td(&mut tab, pop);
        eval(
            &mut tab,
            "window.__log = []; \
             document.getElementById('pc').addEventListener('blur', function() { __log.push('blur'); }); \
             var d = document.getElementById('dlg'); \
             d.addEventListener('cancel', function() { __log.push('cancel'); }); \
             d.addEventListener('close', function() { __log.push('close'); });",
        );
        let mut driver = TestDriver::new();
        driver.actions.focused = Some(pc);
        driver.execute(
            &mut tab,
            sequence_request(vec![pointer_source(
                "mouse",
                vec![
                    move_item(700.0, 500.0, ActionOrigin::Viewport),
                    button_item(ActionItemKind::PointerDown, 0),
                    button_item(ActionItemKind::PointerUp, 0),
                ],
            )]),
        );
        assert!(
            !popover_open(&tab, pop),
            "outside gesture must light-dismiss the popover"
        );
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "outside gesture must light-dismiss the closedby=any dialog"
        );
        assert!(
            js_true(&mut tab, "window.__log.join(',') === 'cancel,close,blur'"),
            "the popover child's blur must fire after the dialog cancel and close, not before cancel"
        );
        assert_eq!(
            driver.actions.focused, None,
            "deferred popover focus clear must still run for a focusable child"
        );
    }

    #[test]
    fn w85fix_dual_submit_fires_toggle_after_close() {
        // Auto dual, `method="dialog"` submit: submit,
        // close, then the dual toggle.
        let mut tab = tab_from_html(
            b"<html><body><dialog id=\"dlg\" popover=\"auto\"><form method=\"dialog\">\
              <button id=\"sub\" type=\"submit\" value=\"go\">ok</button>\
              </form></dialog></body></html>",
        );
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); window.__log = []; \
             d.addEventListener('close', function() { __log.push('close'); }); \
             d.addEventListener('toggle', function() { __log.push('toggle'); }); \
             document.querySelector('form').addEventListener('submit', function() { __log.push('submit'); }); \
             d.show(); d.showPopover(); window.__log = [];",
        );
        let mut driver = TestDriver::new();
        let click_node = el(&tab, "sub");
        driver.execute_click(&mut tab, click_node).expect("click");
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(el(&tab, "dlg")),
            "submit must close the dual dialog"
        );
        assert!(
            !tab.document_ref()
                .expect("document")
                .is_popover_open(el(&tab, "dlg")),
            "submit must end the dual's popover membership"
        );
        assert!(
            js_true(&mut tab, "window.__log.join(',') === 'submit,close,toggle'"),
            "submit-closed dual must fire submit, close, then exactly one toggle, in that order"
        );
        assert!(
            js_true(&mut tab, "d.returnValue === 'go'"),
            "returnValue must still be the submitter's value"
        );

        // Close-listener re-show suppresses the dual
        // toggle: the re-show's own toggle is the only one
        // (count 1), and the popover ends open again while
        // the dialog stays closed.
        let mut tab = tab_from_html(
            b"<html><body><dialog id=\"dlg\" popover=\"auto\"><form method=\"dialog\">\
              <button id=\"sub\" type=\"submit\" value=\"go\">ok</button>\
              </form></dialog></body></html>",
        );
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); window.__log = []; \
             d.addEventListener('close', function() { __log.push('close'); d.showPopover(); }); \
             d.addEventListener('toggle', function() { __log.push('toggle'); }); \
             document.querySelector('form').addEventListener('submit', function() { __log.push('submit'); }); \
             d.show(); d.showPopover(); window.__log = [];",
        );
        let mut driver = TestDriver::new();
        let click_node = el(&tab, "sub");
        driver.execute_click(&mut tab, click_node).expect("click");
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(el(&tab, "dlg")),
            "re-show case: submit must still close the dialog"
        );
        assert!(
            tab.document_ref()
                .expect("document")
                .is_popover_open(el(&tab, "dlg")),
            "re-show case: the close listener's showPopover must leave the popover open"
        );
        assert!(
            js_true(&mut tab, "window.__log.join(',') === 'submit,close,toggle'"),
            "re-show case: exactly one toggle (the re-show's) must fire, never a second dual toggle"
        );
    }

    #[test]
    fn w85fix_dual_attribute_removal_fires_no_events() {
        // Boundary pin, not an app close path: removing
        // `open` from a dual closes it state-wise but fires
        // no events at all (the w84 silent-removal rule for
        // `close`), so it fires no popover toggle either —
        // membership ends by demotion, not by an event.
        // This is why the toggle rule is scoped to real
        // closes (Escape, light-dismiss, submit, `close()` /
        // `requestClose()`), not to every state change.
        let mut tab = tab_from_html(
            b"<html><body><dialog id=\"dlg\" popover=\"auto\"><button id=\"a\">a</button></dialog></body></html>",
        );
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); window.__log = []; \
             d.addEventListener('cancel', function() { __log.push('cancel'); }); \
             d.addEventListener('close', function() { __log.push('close'); }); \
             d.addEventListener('toggle', function() { __log.push('toggle'); }); \
             d.show(); d.showPopover(); window.__log = []; d.removeAttribute('open');",
        );
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(el(&tab, "dlg")),
            "removeAttribute('open') must close the dual state-wise"
        );
        assert!(
            !tab.document_ref()
                .expect("document")
                .is_popover_open(el(&tab, "dlg")),
            "removeAttribute('open') must end the dual's popover membership"
        );
        assert!(
            js_true(&mut tab, "window.__log.length === 0"),
            "removeAttribute('open') must fire neither close nor toggle"
        );
    }

    /// A tab with a `<dialog popover="auto">` (`#dlg`) and a
    /// plain auto popover (`#pop`), for the C10 capacity
    /// decision. No layout boxes needed.
    fn dialog_popover_tab() -> (Tab, NodeId, NodeId) {
        use std::cell::RefCell;
        use std::rc::Rc;
        let doc = Rc::new(RefCell::new(
            bowser_dom::Document::parse_html(
                b"<html><body><dialog id=\"dlg\" popover=\"auto\">hi</dialog>\
                  <div id=\"pop\" popover=\"auto\">hi</div></body></html>",
            )
            .0,
        ));
        let (dlg, pop) = {
            let d = doc.borrow();
            (
                d.get_element_by_id("dlg").expect("dlg"),
                d.get_element_by_id("pop").expect("pop"),
            )
        };
        let base = bowser_url::parse("http://example.com/page.html", None).expect("test url");
        let page =
            crate::script::PageVm::new_for_document(doc.clone(), &base, "TestBrowser/9.9", &[])
                .expect("install web apis");
        let mut tab = crate::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
        tab.install_document(doc, page);
        tab.dirty.layout = true;
        (tab, dlg, pop)
    }

    #[test]
    fn w84_dual_dialog_popover_modal_survives_auto_popover_show() {
        // C10, fixed w84fix (this was a decision test recording
        // the corrupt behaviour — it must not stay the
        // contract): a `<dialog popover="auto">` shown via
        // `showModal` is NOT in the spec's showing auto popover
        // list (HTML §6.12: that list holds only elements
        // opened in popover mode `auto`, and a modal dialog was
        // never opened in popover mode), so showing a second
        // auto popover leaves it untouched — no `toggle`, no
        // `close`, still open, still `:modal`, still in the top
        // layer. Recovery paths included: the never-stranded
        // dialog closes normally and re-shows via `showModal`
        // afterwards. A dialog shown via `showPopover` (not
        // modal) is still a legitimate victim; that direction
        // is pinned at the DOM level.
        let (mut tab, dlg, pop) = dialog_popover_tab();
        eval(
            &mut tab,
            "var d = document.getElementById('dlg'); window.__ev = []; \
             d.addEventListener('close', function() { __ev.push('close'); }); \
             d.addEventListener('toggle', function(e) { __ev.push('toggle:' + e.oldState + '->' + e.newState); }); \
             d.showModal();",
        );
        assert!(
            tab.document_ref().expect("document").is_dialog_open(dlg),
            "showModal must open the dual dialog"
        );
        assert!(
            js_true(&mut tab, "d.matches(':modal')"),
            "dual dialog must match :modal after showModal"
        );
        eval(&mut tab, "document.getElementById('pop').showPopover();");
        assert!(
            tab.document_ref().expect("document").is_popover_open(pop),
            "the second auto popover must be open"
        );
        assert!(
            js_true(&mut tab, "__ev.length === 0"),
            "a modal dialog must receive no toggle and no close when another auto popover shows"
        );
        assert!(
            tab.document_ref().expect("document").is_dialog_open(dlg),
            "a modal dialog must stay open when another auto popover shows"
        );
        assert!(
            tab.document_ref().expect("document").is_in_top_layer(dlg),
            "a modal dialog must stay in the top layer when another auto popover shows"
        );
        assert!(
            js_true(&mut tab, "d.matches(':modal')"),
            "a modal dialog must keep matching :modal when another auto popover shows"
        );
        // Recovery: never stranded, so close() works and a
        // fresh showModal() reopens it.
        eval(&mut tab, "d.close();");
        assert!(
            !tab.document_ref().expect("document").is_dialog_open(dlg),
            "close() must close the surviving dialog"
        );
        assert!(
            js_true(&mut tab, "__ev.join(',') === 'close'"),
            "close() must fire exactly one close event"
        );
        assert!(
            !js_true(&mut tab, "d.matches(':modal')"),
            "the closed dialog must no longer match :modal"
        );
        eval(&mut tab, "d.showModal();");
        assert!(
            tab.document_ref().expect("document").is_dialog_open(dlg),
            "showModal() must reopen the dialog after close()"
        );
        assert!(
            js_true(&mut tab, "d.matches(':modal')"),
            "the reopened dialog must match :modal again"
        );
    }

    #[test]
    fn testdriver_popover_scope_blocks_dialog_move_in() {
        // w82 fix regression (see
        // `actions_tab_traps_in_popover_above_modal_dialog`):
        // a dialog shown under an open popover records its
        // entry but must not move focus into the dialog below
        // the popover's trap.
        let (mut tab, out, dlg, _a) = {
            use std::cell::RefCell;
            use std::rc::Rc;
            let doc = Rc::new(RefCell::new(
                bowser_dom::Document::parse_html(
                    b"<html><body><button id=\"out\">o</button>\
                      <dialog id=\"dlg\"><button id=\"a\">a</button></dialog>\
                      <div id=\"pop\" popover><button id=\"p1\">1</button></div></body></html>",
                )
                .0,
            ));
            let (out, dlg, pop) = {
                let d = doc.borrow();
                (
                    d.get_element_by_id("out").expect("out"),
                    d.get_element_by_id("dlg").expect("dlg"),
                    d.get_element_by_id("pop").expect("pop"),
                )
            };
            let base = bowser_url::parse("http://example.com/page.html", None).expect("test url");
            let page =
                crate::script::PageVm::new_for_document(doc.clone(), &base, "TestBrowser/9.9", &[])
                    .expect("install web apis");
            let mut tab = crate::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
            tab.install_document(doc, page);
            (tab, out, dlg, pop)
        };
        let pop = tab
            .document_ref()
            .expect("doc")
            .get_element_by_id("pop")
            .expect("pop");
        tab.installed_mut()
            .expect("installed")
            .document()
            .borrow_mut()
            .show_dialog_modal(dlg)
            .expect("showModal");
        open_popover_td(&mut tab, pop);
        let mut driver = TestDriver::new();
        driver.set_focus(&mut tab, Some(out));
        driver.sync_modal_focus(&mut tab);
        assert_eq!(
            driver.actions.focused,
            Some(out),
            "popover scope above the dialog must block the move-in"
        );
        assert_eq!(driver.actions.modal_focus_stack.len(), 1);
    }
}
