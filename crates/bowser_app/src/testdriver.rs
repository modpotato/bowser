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

use bowser_dom::NodeId;

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
    /// modifier state.
    fn move_focus_tab(&mut self, tab: &mut Tab, backward: bool) {
        let order = match tab.document_ref() {
            Some(doc) => sequential_focus_order(&doc),
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
        if down && key == "Tab" && !prevented {
            self.move_focus_tab(tab, self.actions.modifiers.shift);
        }
        Ok(())
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
    /// events, so they report it held.
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
                                self.fire_pointer(
                                    tab,
                                    &st,
                                    ct,
                                    if button == 0 { "click" } else { "auxclick" },
                                    PointerMouseKind::Mouse,
                                    button as i32,
                                    (0.0, 0.0),
                                );
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
    let mut positive: Vec<(i32, usize, NodeId)> = Vec::new();
    let mut zero: Vec<NodeId> = Vec::new();
    for (i, node) in doc.descendants(doc.root()).iter().enumerate() {
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
        // plus a trusted bubbling, cancelable `click` dispatch.
        self.dispatch.dispatch_click(tab, node);
        Self::grant_activation(tab);
        Ok(())
    }

    fn execute_send_keys(&mut self, tab: &mut Tab, node: NodeId, keys: &str) -> Result<(), String> {
        self.scroll_into_view(tab, node);
        for key in split_testdriver_keys(keys) {
            self.dispatch.dispatch_key_at(tab, node, &key);
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
}
