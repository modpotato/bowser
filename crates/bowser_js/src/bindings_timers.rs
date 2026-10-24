//! Timer bindings: `setInterval`/`clearInterval` and
//! `requestAnimationFrame`/`cancelAnimationFrame`, plus the page-context GC
//! threshold.
//!
//! `setTimeout`/`clearTimeout`/`queueMicrotask` live in
//! [`bindings`](crate::bindings); this module owns the repeating-timer and
//! animation-frame surface. Intervals ride the page [`EventLoop`]'s timer
//! heap ([`EventLoop::set_interval`]/[`EventLoop::clear_timeout`]); rAF is
//! driven by the embedder through [`pump_animation_frame`].
//!
//! # Status: complete implementation; production install pending a pump hook
//!
//! The natives below are complete, unit-tested implementations — not
//! stubs — and [`install_timers`] is called from
//! [`install_web_apis`](crate::bindings::install_web_apis) (the full,
//! test-driven install). The *production* page install
//! ([`install_page_web_apis`](crate::bindings::install_page_web_apis))
//! deliberately skips them for now: firing interval/rAF callbacks from
//! the event-loop pump needs a `vm.rs` pump hook that does not exist
//! yet (see `ReentrantInvoker` in [`bindings`](crate::bindings)), and
//! installing `setInterval` over the VM's own working intrinsic would
//! silently break page timers. Until that hook lands, the entry points
//! below are production-dead code (`#[allow(dead_code)]`) exercised by
//! the unit tests. The "Firing" and "Host frame-pump contract" sections
//! describe the contracts the pump worker will follow.
//!
//! # Callback storage and GC
//!
//! Pending callback functions are stored as properties of a holder object
//! kept on the page global under a hidden key. The engine roots the page
//! global for the page's lifetime (the `Vm` registers its interpreter,
//! which owns the global, as a trace root), so pending callbacks survive
//! collections through the ordinary object graph. The per-context host
//! state ([`TimersHost`]) therefore holds no `Value`s at all: no custom
//! `Trace` impl, no pinned raw roots, and teardown is a plain map removal.
//!
//! # Firing
//!
//! When an interval's deadline passes, the [`EventLoop`] hands the
//! embedder `Task::FireTimer { callback, .. }` carrying the opaque callback
//! id this module registered. The embedder's runner should offer the task
//! to `BindingRunner::try_run_task` first and then to
//! [`try_run_timer_task`]; a `false` from both means the id belongs to the
//! VM's own function table. (Callback-id namespaces are independent per
//! module; a miss here is always safe.)
//!
//! # Wiring
//!
//! [`install_timers`] is already called from
//! [`install_web_apis`](crate::bindings::install_web_apis), which sets the
//! page heap's GC threshold at the same place (see
//! [`DEFAULT_GC_THRESHOLD`]). The production page install
//! ([`install_page_web_apis`](crate::bindings::install_page_web_apis))
//! skips timers until the pump hook exists (see "Status" above); nothing
//! in this module needs to change when it does.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use crate::bindings::{ApiError, CTX_KEY, ContextId, JsInvoker};
use crate::event_loop::{EventLoop, Task, TimerId, TimerIdMap};
use crate::gc::{AllocError, Gc, Heap, object_kind};
use crate::value::{NativeError, NativeFn, Object, ObjectKind, Property, Value};
use crate::wtf8::Wtf8;

/// Allocation count that triggers an automatic GC on the next allocation
/// once installed; see [`Heap::set_gc_threshold`].
///
/// Rationale: a page's timer/frame callbacks keep JS functions alive across
/// tasks and DOM churn allocates steadily, so without a threshold the heap
/// grows without bound until someone calls `collect()` explicitly (the
/// default is `usize::MAX`, i.e. disabled). 4096 live allocations between
/// collections bounds garbage while keeping pauses rare; tune with
/// profiling per AGENTS.md.
///
/// The call itself must live in `install_web_apis` (it owns `&mut Heap`
/// at page-context construction), placed before the first allocation:
///
/// ```ignore
/// heap.set_gc_threshold(crate::bindings_timers::DEFAULT_GC_THRESHOLD);
/// ```
pub const DEFAULT_GC_THRESHOLD: usize = 4096;

// Compile-time sanity: the threshold must actually enable collection.
const _: () = assert!(DEFAULT_GC_THRESHOLD > 0 && DEFAULT_GC_THRESHOLD < usize::MAX);

/// Hidden global property holding the callback-holder object installed by
/// [`install_timers`].
const HOLDER_KEY: &str = "\u{0}bowser:timers";

/// Hidden global property naming the frame-pump native installed by
/// [`install_timers`]; see [`pump_animation_frame_in_vm`].
const PUMP_RAF_KEY: &str = "\u{0}bowser:pumpRaf";

/// Per-page timer state owned by this module, keyed by [`ContextId`].
///
/// `bindings.rs` keeps its own `setTimeout` table inside `WebApis`; this
/// module cannot touch it, so interval and rAF bookkeeping lives here. See
/// the module docs for why the JS functions themselves are NOT stored
/// here (they hang off the global's holder object instead).
struct TimersHost {
    event_loop: Rc<EventLoop>,
    invoker: Rc<RefCell<dyn JsInvoker>>,
    /// Cached global handle, for resolving the holder at fire/pump time.
    /// Sound because the engine roots the page global for the page's
    /// lifetime (see module docs).
    global: Gc<Object>,
    next_callback: u32,
    /// Live interval timer id -> callback id, so `clearInterval` drops the
    /// function as well as the timer. SipHash-free: ids are dense `u32`s,
    /// see [`TimerIdMap`].
    intervals: TimerIdMap<u32>,
    next_raf: u32,
    /// Pending rAF handles in registration order.
    raf_pending: VecDeque<u32>,
}

impl TimersHost {
    fn next_callback(&mut self) -> u32 {
        let id = self.next_callback;
        self.next_callback = id.wrapping_add(1).max(1);
        id
    }

    fn next_raf(&mut self) -> u32 {
        let id = self.next_raf;
        self.next_raf = id.wrapping_add(1).max(1);
        id
    }
}

thread_local! {
    // SipHash-free host table: context ids are dense `u32`s, see [`TimerIdMap`].
    static HOSTS: RefCell<TimerIdMap<TimersHost>> = RefCell::new(TimerIdMap::default());
}

/// Install the timer globals on the page's global object: `setInterval`,
/// `clearInterval`, `requestAnimationFrame`, `cancelAnimationFrame`.
///
/// Mirrors [`install_web_apis`](crate::bindings::install_web_apis):
/// `heap`/`global` are the page's engine heap and global object, `ctx` is
/// the [`ContextId`] the installation registers under, `event_loop` is the
/// page's [`EventLoop`] (intervals ride its timer heap), and `invoker`
/// invokes the JS callback functions when timers fire or frames pump.
///
/// Registration used by the wiring pass inside
/// [`install_web_apis`](crate::bindings::install_web_apis), applied after
/// the [`ContextId`] is minted and before `WebApis::new` moves
/// `event_loop`/`invoker`:
///
/// ```ignore
/// crate::bindings_timers::install_timers(
///     heap,
///     global,
///     ContextId(id),
///     event_loop.clone(),
///     invoker.clone(),
/// )?;
/// ```
///
/// Reinstalling for an already-installed `ctx` replaces the host state but
/// does NOT cancel the previous install's timers on the [`EventLoop`]
/// (that is [`remove_timers_context`]'s job): install once per
/// freshly-minted [`ContextId`]. A reinstall would leave the old intervals
/// generating `FireTimer` tasks against the new holder, where their
/// callback ids miss (or, on id reuse, hit the wrong callback).
// Production install of these globals awaits the pump hook (see the
// module docs); reachable today from the full install in `bindings`.
#[allow(dead_code)]
pub fn install_timers(
    heap: &mut Heap,
    global: Gc<Object>,
    ctx: ContextId,
    event_loop: Rc<EventLoop>,
    invoker: Rc<RefCell<dyn JsInvoker>>,
) -> Result<(), ApiError> {
    // Holder for pending callback functions, reachable from the global so
    // the collector keeps them alive (see module docs).
    let holder = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    def(heap, global, HOLDER_KEY, Value::Object(holder))?;

    HOSTS.with(|hosts| {
        hosts.borrow_mut().insert(
            ctx.0,
            TimersHost {
                event_loop,
                invoker,
                global,
                next_callback: 1,
                intervals: TimerIdMap::default(),
                next_raf: 1,
                raf_pending: VecDeque::new(),
            },
        );
    });

    def_native(heap, global, "setInterval", n_set_interval)?;
    def_native(heap, global, "clearInterval", n_clear_interval)?;
    def_native(
        heap,
        global,
        "requestAnimationFrame",
        n_request_animation_frame,
    )?;
    def_native(
        heap,
        global,
        "cancelAnimationFrame",
        n_cancel_animation_frame,
    )?;
    // The out-of-VM frame driver (see `pump_animation_frame_in_vm`):
    // installed here, next to the surface it pumps, so the drain and
    // its entry stub can never disagree on the key.
    def_native(heap, global, PUMP_RAF_KEY, n_pump_raf)?;
    Ok(())
}

/// Tear down a context's timer state; call alongside
/// [`remove_context`](crate::bindings::remove_context) when the tab closes.
///
/// Live intervals are cancelled on the shared [`EventLoop`], pending rAF
/// callbacks are dropped, and the host state is forgotten. The
/// callback-holder object dies with the page's global. Safe to call for an
/// unknown id.
// Reached from `bindings::remove_context` (production page teardown
// included); safe for contexts that never installed timers.
#[allow(dead_code)]
pub fn remove_timers_context(ctx: ContextId) {
    let host = HOSTS.with(|hosts| hosts.borrow_mut().remove(&ctx.0));
    if let Some(host) = host {
        for timer_id in host.intervals.keys() {
            host.event_loop.clear_timeout(*timer_id);
        }
    }
}

/// Run one event-loop task if its callback id belongs to this module's
/// interval table. Returns `false` for foreign ids; see the module docs
/// for the runner-chaining contract.
///
/// Interval entries stay registered across fires (they repeat); only
/// `clearInterval` removes them. A fire for an already-cleared interval
/// misses the holder and returns `false`.
// Reached from `BindingRunner::try_run_task`; the production pump
// hook will drive it once the `vm.rs` hook exists.
#[allow(dead_code)]
pub fn try_run_timer_task(heap: &mut Heap, ctx: ContextId, task: &Task) -> bool {
    let Task::FireTimer { callback, .. } = *task else {
        return false;
    };
    // Copy the global handle and invoker out under the host borrow, then
    // release it before invoking: a callback may re-enter timer
    // registration, which borrows the host table again.
    let found = host(ctx.0, |h| (h.global, h.invoker.clone())).ok();
    let Some((global, invoker)) = found else {
        return false;
    };
    let Ok(holder) = holder_of(heap, global) else {
        return false;
    };
    let mut key = [0u8; 11];
    let Some(prop) = heap.get_property(holder, Wtf8::from_str(id_key(b'i', callback, &mut key)))
    else {
        return false;
    };
    // DOM report-and-continue: a throwing interval must not kill the loop.
    let _ = invoker
        .borrow()
        .call(heap, prop.value, Value::Undefined, &[]);
    true
}

/// Drive one animation frame for `ctx`: invoke every callback registered
/// by `requestAnimationFrame` since the previous pump, in registration
/// order, each called with `timestamp_ms` as its single argument
/// (a `DOMHighResTimeStamp`).
///
/// # Host frame-pump contract
///
/// The embedder owns the frame clock; this module only queues and fires.
/// The embedder MUST:
///
/// * call this once per presented frame — on vsync in a windowed browser,
///   once per simulated frame in headless/WPT runs — with a monotonically
///   non-decreasing `timestamp_ms` (epoch is the embedder's choice, but it
///   must be consistent across pumps);
/// * keep pumping the event loop (`run_until_idle`, plus the microtask
///   drain after every macrotask) around the pump: rAF callbacks may
///   schedule timers/microtasks, and timers scheduled from rAF must not
///   starve;
/// * NOT route rAF through the timer heap: rAF callbacks never fire from
///   `run_until_idle` alone. A tab with no pump simply never runs them,
///   which matches browsers' "no frames when occluded" behavior.
///
/// Snapshot semantics (per spec): the pending queue is drained before the
/// first callback runs, so callbacks registered *during* the pump run on
/// the NEXT pump, not the current one. `cancelAnimationFrame` drops a
/// still-pending callback; each callback fires at most once per
/// registration (one-shot).
///
/// Returns `false` when `ctx` has no timer installation (never installed
/// or already torn down); the embedder should stop pumping such contexts.
// Reached from the `pump_animation_frame` wrapper in `bindings`, and
// from the hidden `PUMP_RAF_KEY` native (see
// `pump_animation_frame_in_vm`): embedders without a live interpreter
// must go through the latter — callback invocation needs the engine's
// `ACTIVE` interpreter, which a direct call from outside the VM cannot
// provide.
#[allow(dead_code)]
pub fn pump_animation_frame(heap: &mut Heap, ctx: ContextId, timestamp_ms: f64) -> bool {
    let Some((global, invoker, pending)) = HOSTS.with(|hosts| {
        hosts.borrow_mut().get_mut(&ctx.0).map(|h| {
            // Move the queue out instead of collecting a snapshot `Vec`:
            // no allocation, and callbacks registered during the pump land
            // in the fresh queue, so snapshot semantics are preserved.
            let pending = std::mem::take(&mut h.raf_pending);
            (h.global, h.invoker.clone(), pending)
        })
    }) else {
        return false;
    };
    if pending.is_empty() {
        return true;
    }
    let Ok(holder) = holder_of(heap, global) else {
        return true;
    };
    let timestamp = Value::Float(timestamp_ms);
    let mut key = [0u8; 11];
    for handle in pending {
        let prop_key = id_key(b'r', handle, &mut key);
        let func = heap
            .get_property(holder, Wtf8::from_str(prop_key))
            .map(|p| p.value);
        // One-shot: the entry is gone even if the callback re-registers
        // (that registration lands in the next pump's queue).
        heap.delete_property(holder, Wtf8::from_str(prop_key));
        if let Some(func) = func {
            // DOM report-and-continue, mirroring `BindingRunner`.
            let _ = invoker
                .borrow()
                .call(heap, func, Value::Undefined, &[timestamp]);
        }
    }
    true
}

/// Hidden frame-pump native behind [`PUMP_RAF_KEY`]: drains one animation
/// frame exactly like [`pump_animation_frame`].
///
/// Exists because [`pump_animation_frame`] cannot run from outside the
/// VM: callback invocation goes through the [`JsInvoker`], whose
/// re-entry needs the engine's `ACTIVE` interpreter (see
/// [`crate::vm::reenter_call`]) — a direct pump-side invoke fails closed
/// with `TypeError: cannot invoke JavaScript function outside the VM`,
/// which the drain swallows, so callbacks silently never run. The
/// out-of-VM driver ([`pump_animation_frame_in_vm`]) evals a stub calling
/// this instead: the stub enters through the engine's `call_native`,
/// which sets `ACTIVE`, and the invoker can re-enter. Mirrors the
/// `\0bowser:fireTask` pattern in [`bindings`](crate::bindings).
///
/// Returns the [`pump_animation_frame`] verdict (`false` when `ctx` has
/// no timer installation). Callback throws are report-and-continue,
/// same as the direct drain.
fn n_pump_raf(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let ctx = ctx_of(heap, this)?;
    let timestamp_ms = match args.first().copied().unwrap_or(Value::Undefined) {
        Value::Int(i) => i as f64,
        Value::Float(f) => f,
        _ => 0.0,
    };
    Ok(Value::Bool(pump_animation_frame(
        heap,
        ContextId(ctx),
        timestamp_ms,
    )))
}

/// Drive one animation frame for the page realm from *outside* the VM.
///
/// Embedders without a live interpreter (the headless pump, or a future
/// vsync callback on another thread) must use this instead of
/// [`pump_animation_frame`]: it evals a stub calling the hidden
/// [`PUMP_RAF_KEY`] native on the VM's global, so callback invocation
/// happens with the engine's `ACTIVE` interpreter set and the
/// [`JsInvoker`] can re-enter the VM. A direct [`pump_animation_frame`]
/// call from outside the VM fails closed — every callback errors with
/// "cannot invoke JavaScript function outside the VM" and is swallowed,
/// so rAF callbacks silently never fire while the pump reports success.
///
/// `timestamp_ms` follows the [`pump_animation_frame`] frame-pump
/// contract (monotonically non-decreasing, consistent epoch across
/// pumps). One eval per frame is the v1 cost, mirroring `eval_fire` in
/// [`bindings`](crate::bindings).
///
/// Returns `false` when the realm's global has no timer installation
/// (never installed or already torn down) or the stub eval failed (the
/// page overwrote the hidden key); the embedder should stop pumping such
/// realms.
pub fn pump_animation_frame_in_vm(vm: &mut crate::vm::Vm, timestamp_ms: f64) -> bool {
    // The timestamp rides in the source; `f64`'s `Display` is
    // round-trip-exact, so the native sees the same value.
    let source = format!("globalThis[\"\\0bowser:pumpRaf\"]({timestamp_ms});");
    match vm.eval(&source) {
        Ok(Value::Bool(alive)) => alive,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Native-function plumbing (mirrors `bindings.rs`)
// ---------------------------------------------------------------------------

fn type_error(msg: impl Into<String>) -> NativeError {
    NativeError::TypeError(msg.into())
}

fn def(heap: &mut Heap, obj: Gc<Object>, name: &str, val: Value) -> Result<(), AllocError> {
    heap.define_property_str(obj, name, Property::data(val))
}

fn native_val(heap: &mut Heap, f: NativeFn) -> Result<Value, AllocError> {
    Ok(Value::Object(heap.alloc_object(Object::new(
        ObjectKind::NativeFunction(f),
    ))?))
}

fn def_native(heap: &mut Heap, obj: Gc<Object>, name: &str, f: NativeFn) -> Result<(), AllocError> {
    let v = native_val(heap, f)?;
    def(heap, obj, name, v)
}

/// The [`ContextId`] behind the receiver's hidden id property.
fn ctx_of(heap: &Heap, this: Value) -> Result<u32, NativeError> {
    let Value::Object(obj) = this else {
        return Err(type_error("timer function called with invalid receiver"));
    };
    match heap.get_property(obj, Wtf8::from_str(CTX_KEY)) {
        Some(p) => match p.value {
            Value::Int(id) => Ok(id as u32),
            _ => Err(type_error("timer function called with invalid receiver")),
        },
        None => Err(type_error("timer function called with invalid receiver")),
    }
}

/// The callback-holder object installed on the global by [`install_timers`].
fn holder_of(heap: &Heap, global: Gc<Object>) -> Result<Gc<Object>, NativeError> {
    match heap.get_property(global, Wtf8::from_str(HOLDER_KEY)) {
        Some(p) => match p.value {
            Value::Object(o) => Ok(o),
            _ => Err(type_error("timer context is gone")),
        },
        None => Err(type_error("timer context is gone")),
    }
}

/// Run `f` against the calling context's [`TimersHost`].
fn host<R>(ctx: u32, f: impl FnOnce(&mut TimersHost) -> R) -> Result<R, NativeError> {
    HOSTS.with(|hosts| {
        hosts
            .borrow_mut()
            .get_mut(&ctx)
            .map(f)
            .ok_or_else(|| type_error("timer context is gone"))
    })
}

/// Holder property key for a callback id: an `'i'` (interval) or `'r'`
/// (requestAnimationFrame) prefix followed by the decimal id, formatted into
/// `buf` without allocating. `try_run_timer_task` runs once per interval
/// fire and `pump_animation_frame` once per pending callback, where the
/// previous `format!` showed up as a per-fire heap allocation.
fn id_key(prefix: u8, id: u32, buf: &mut [u8; 11]) -> &str {
    debug_assert!(prefix == b'i' || prefix == b'r');
    buf[0] = prefix;
    // Decimal digits, least significant first, then reversed into place.
    // `u32::MAX` is 10 digits; the buffer holds the prefix plus those.
    let mut digits = [0u8; 10];
    let mut n = 0;
    let mut x = id;
    loop {
        digits[n] = b'0' + (x % 10) as u8;
        n += 1;
        x /= 10;
        if x == 0 {
            break;
        }
    }
    for (i, d) in digits[..n].iter().rev().enumerate() {
        buf[1 + i] = *d;
    }
    std::str::from_utf8(&buf[..1 + n]).expect("id_key writes ASCII only")
}

fn require_callable(who: &str, v: Value) -> Result<(), NativeError> {
    match v {
        Value::Object(o) => match object_kind(o) {
            ObjectKind::Function(_) | ObjectKind::NativeFunction(_) => Ok(()),
            _ => Err(type_error(format!("{who}: argument is not callable"))),
        },
        _ => Err(type_error(format!("{who}: argument is not callable"))),
    }
}

/// Delay coercion mirroring `bindings.rs`'s `arg_to_delay`: `undefined`
/// -> 0, NaN -> 0, negatives -> 0, non-numbers are a `TypeError`.
/// (Nested-timer 4ms clamping is not implemented in v1.)
fn to_delay_ms(who: &str, v: Value) -> Result<u64, NativeError> {
    match v {
        Value::Undefined => Ok(0),
        Value::Int(i) => Ok(i.max(0) as u64),
        Value::Float(f) => Ok(if f.is_nan() { 0 } else { f.max(0.0) as u64 }),
        Value::Bool(b) => Ok(u64::from(b)),
        _ => Err(type_error(format!("{who}: delay must be a number"))),
    }
}

/// Timer/frame handle argument: `Int`/`Float` -> `u32`; anything else ->
/// `None` (callers treat it as a no-op, mirroring `clearTimeout`).
fn to_handle(v: Value) -> Option<u32> {
    match v {
        Value::Int(i) => Some(i as u32),
        Value::Float(f) => Some(f as u32),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Timer and animation-frame natives
// ---------------------------------------------------------------------------

fn n_set_interval(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let ctx = ctx_of(heap, this)?;
    let cb = args.first().copied().unwrap_or(Value::Undefined);
    require_callable("setInterval", cb)?;
    let delay_ms = to_delay_ms(
        "setInterval",
        args.get(1).copied().unwrap_or(Value::Undefined),
    )?;

    // One host-table lookup: stash the function on the holder before
    // arming the timer, so a collection between here and the first fire
    // still sees it (the holder is reachable from the global).
    let timer_id = host(ctx, |h| -> Result<TimerId, NativeError> {
        let callback = h.next_callback();
        let holder = holder_of(heap, h.global)?;
        let mut key = [0u8; 11];
        heap.define_property_str(holder, id_key(b'i', callback, &mut key), Property::data(cb))
            .map_err(|_| NativeError::OutOfMemory)?;
        let timer_id = h.event_loop.set_interval(callback, delay_ms);
        h.intervals.insert(timer_id, callback);
        Ok(timer_id)
    })??;
    Ok(Value::Float(timer_id as f64))
}

fn n_clear_interval(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let ctx = ctx_of(heap, this)?;
    let Some(id) = args.first().copied().and_then(to_handle) else {
        return Ok(Value::Undefined);
    };
    let (event_loop, global, callback) = host(ctx, |h| {
        (h.event_loop.clone(), h.global, h.intervals.remove(&id))
    })?;
    event_loop.clear_timeout(id);
    // Drop the function so a cleared interval releases it for GC. (A
    // `clearTimeout` aimed at an interval id still stops the timer via the
    // shared heap; only `clearInterval` also releases the function. The
    // timer list itself is shared, per spec.)
    if let Some(callback) = callback {
        let holder = holder_of(heap, global)?;
        let mut key = [0u8; 11];
        heap.delete_property(holder, Wtf8::from_str(id_key(b'i', callback, &mut key)));
    }
    Ok(Value::Undefined)
}

fn n_request_animation_frame(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let ctx = ctx_of(heap, this)?;
    let cb = args.first().copied().unwrap_or(Value::Undefined);
    require_callable("requestAnimationFrame", cb)?;

    // One host-table lookup: stash the function on the holder and queue
    // the handle together.
    let handle = host(ctx, |h| -> Result<u32, NativeError> {
        let handle = h.next_raf();
        let holder = holder_of(heap, h.global)?;
        let mut key = [0u8; 11];
        heap.define_property_str(holder, id_key(b'r', handle, &mut key), Property::data(cb))
            .map_err(|_| NativeError::OutOfMemory)?;
        h.raf_pending.push_back(handle);
        Ok(handle)
    })??;
    Ok(Value::Float(handle as f64))
}

fn n_cancel_animation_frame(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let ctx = ctx_of(heap, this)?;
    let Some(handle) = args.first().copied().and_then(to_handle) else {
        return Ok(Value::Undefined);
    };
    let (global, was_pending) = host(ctx, |h| {
        let before = h.raf_pending.len();
        h.raf_pending.retain(|&x| x != handle);
        (h.global, h.raf_pending.len() != before)
    })?;
    if was_pending {
        let holder = holder_of(heap, global)?;
        let mut key = [0u8; 11];
        heap.delete_property(holder, Wtf8::from_str(id_key(b'r', handle, &mut key)));
    }
    Ok(Value::Undefined)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::InvokeError;
    use crate::event_loop::{Clock, FakeClock, TaskRunner};
    use crate::gc::HandleScope;

    fn probe_a(_heap: &mut Heap, _this: Value, _args: &[Value]) -> Result<Value, NativeError> {
        Ok(Value::Undefined)
    }

    fn probe_b(_heap: &mut Heap, _this: Value, _args: &[Value]) -> Result<Value, NativeError> {
        Ok(Value::Undefined)
    }

    /// Runs inside `call` so a test can register/cancel timers from
    /// within a callback.
    type OnCall = Option<Box<dyn FnMut(&mut Heap)>>;

    /// Records invocations as (function, args). `on_call` runs inside
    /// `call` so a test can register/cancel timers from within a callback.
    #[derive(Default)]
    struct FakeInvoker {
        calls: RefCell<Vec<(Value, Vec<Value>)>>,
        on_call: RefCell<OnCall>,
    }

    impl FakeInvoker {
        fn set_on_call(&mut self, f: OnCall) {
            *self.on_call.borrow_mut() = f;
        }
    }

    impl JsInvoker for FakeInvoker {
        fn call(
            &self,
            heap: &mut Heap,
            func: Value,
            _this: Value,
            args: &[Value],
        ) -> Result<Value, InvokeError> {
            self.calls.borrow_mut().push((func, args.to_vec()));
            if let Some(f) = self.on_call.borrow_mut().as_mut() {
                f(heap);
            }
            Ok(Value::Undefined)
        }
    }

    struct Fx {
        heap: Heap,
        global: Gc<Object>,
        ctx: ContextId,
        event_loop: Rc<EventLoop>,
        clock: Rc<FakeClock>,
        invoker: Rc<RefCell<FakeInvoker>>,
    }

    fn setup() -> Fx {
        let mut heap = Heap::new();
        let global = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let ctx = ContextId(7);
        // Mirror `install_web_apis`: the hidden context id on the global.
        heap.define_property_str(global, CTX_KEY, Property::data(Value::Int(ctx.0 as i32)))
            .unwrap();
        let clock = Rc::new(FakeClock::new());
        let event_loop = Rc::new(EventLoop::with_clock(clock.clone() as Rc<dyn Clock>));
        let invoker = Rc::new(RefCell::new(FakeInvoker::default()));
        let invoker_obj: Rc<RefCell<dyn JsInvoker>> = invoker.clone();
        install_timers(&mut heap, global, ctx, event_loop.clone(), invoker_obj).unwrap();
        Fx {
            heap,
            global,
            ctx,
            event_loop,
            clock,
            invoker,
        }
    }

    fn global_val(fx: &Fx) -> Value {
        Value::Object(fx.global)
    }

    fn callable(fx: &mut Fx, f: NativeFn) -> Value {
        Value::Object(
            fx.heap
                .alloc_object(Object::new(ObjectKind::NativeFunction(f)))
                .unwrap(),
        )
    }

    /// The embedder's runner: offer `FireTimer` to this module's table.
    struct Runner<'a> {
        heap: &'a mut Heap,
        ctx: ContextId,
    }

    impl TaskRunner for Runner<'_> {
        fn run_task(&mut self, task: Task) {
            try_run_timer_task(self.heap, self.ctx, &task);
        }
    }

    fn run_loop(fx: &mut Fx) {
        let el = fx.event_loop.clone();
        let mut runner = Runner {
            heap: &mut fx.heap,
            ctx: fx.ctx,
        };
        el.run_until_idle(&mut runner);
    }

    fn fired(fx: &Fx) -> Vec<Value> {
        fx.invoker
            .borrow()
            .calls
            .borrow()
            .iter()
            .map(|(f, _)| *f)
            .collect()
    }

    #[test]
    fn install_defines_timer_globals() {
        let fx = setup();
        for name in [
            "setInterval",
            "clearInterval",
            "requestAnimationFrame",
            "cancelAnimationFrame",
        ] {
            let v = fx
                .heap
                .get_property(fx.global, Wtf8::from_str(name))
                .unwrap_or_else(|| panic!("global is missing {name}"))
                .value;
            assert!(
                matches!(v, Value::Object(o) if matches!(object_kind(o), ObjectKind::NativeFunction(_))),
                "{name} is not a native function"
            );
        }
    }

    #[test]
    fn set_interval_fires_repeatedly_in_deadline_order() {
        let mut fx = setup();
        let g = global_val(&fx);
        let a = callable(&mut fx, probe_a);
        let b = callable(&mut fx, probe_b);
        let ta = n_set_interval(&mut fx.heap, g, &[a, Value::Int(10)]).unwrap();
        let tb = n_set_interval(&mut fx.heap, g, &[b, Value::Int(20)]).unwrap();
        assert_ne!(ta, tb);

        fx.clock.advance(45);
        run_loop(&mut fx);

        // a @10, b+a @20, a @30, b+a @40. At shared deadlines the *freshly
        // rescheduled* entry sorts after the older one: when `a` fires at
        // t=10 it is re-armed with a new sequence number, so `b` (armed
        // once at t=0) wins the t=20 and t=40 ties.
        assert_eq!(fired(&fx), vec![a, b, a, a, b, a]);
        assert!(fx.event_loop.has_pending_work());
    }

    #[test]
    fn clear_interval_cancels_future_fires() {
        let mut fx = setup();
        let g = global_val(&fx);
        let a = callable(&mut fx, probe_a);
        let ta = n_set_interval(&mut fx.heap, g, &[a, Value::Int(10)]).unwrap();

        fx.clock.advance(15);
        run_loop(&mut fx);
        assert_eq!(fired(&fx), vec![a]);

        assert_eq!(
            n_clear_interval(&mut fx.heap, g, &[ta]).unwrap(),
            Value::Undefined
        );
        fx.clock.advance(1000);
        run_loop(&mut fx);
        assert_eq!(fired(&fx), vec![a]);
        assert!(!fx.event_loop.has_pending_work());
    }

    #[test]
    fn clear_interval_unknown_or_invalid_id_is_noop() {
        let mut fx = setup();
        let g = global_val(&fx);
        assert_eq!(
            n_clear_interval(&mut fx.heap, g, &[Value::Int(4242)]).unwrap(),
            Value::Undefined
        );
        assert_eq!(
            n_clear_interval(&mut fx.heap, g, &[Value::Undefined]).unwrap(),
            Value::Undefined
        );
        assert_eq!(
            n_clear_interval(&mut fx.heap, g, &[]).unwrap(),
            Value::Undefined
        );
    }

    #[test]
    fn set_interval_rejects_non_callable() {
        let mut fx = setup();
        let g = global_val(&fx);
        let err = n_set_interval(&mut fx.heap, g, &[Value::Int(1), Value::Int(10)]).unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));
        let err = n_set_interval(&mut fx.heap, g, &[]).unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));
    }

    #[test]
    fn delay_coercion_mirrors_bindings() {
        assert_eq!(to_delay_ms("setInterval", Value::Undefined).unwrap(), 0);
        assert_eq!(to_delay_ms("setInterval", Value::Int(-3)).unwrap(), 0);
        assert_eq!(
            to_delay_ms("setInterval", Value::Float(f64::NAN)).unwrap(),
            0
        );
        assert_eq!(to_delay_ms("setInterval", Value::Float(10.9)).unwrap(), 10);
        assert_eq!(to_delay_ms("setInterval", Value::Bool(true)).unwrap(), 1);
        assert!(to_delay_ms("setInterval", Value::Null).is_err());
        assert!(to_delay_ms("setInterval", Value::Int(250)).unwrap() == 250);
    }

    #[test]
    fn request_animation_frame_pump_contract() {
        let mut fx = setup();
        let g = global_val(&fx);
        let a = callable(&mut fx, probe_a);
        let h = n_request_animation_frame(&mut fx.heap, g, &[a]).unwrap();
        assert_eq!(h, Value::Float(1.0));

        // The wrapper in `bindings` (`pump_animation_frame`)
        // delegates to the module implementation.
        assert!(crate::bindings::pump_animation_frame(
            &mut fx.heap,
            fx.ctx,
            16.7
        ));
        {
            let calls = fx.invoker.borrow();
            let calls = calls.calls.borrow();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].0, a);
            assert_eq!(calls[0].1, vec![Value::Float(16.7)]);
        }

        // One-shot: the second pump fires nothing.
        assert!(pump_animation_frame(&mut fx.heap, fx.ctx, 33.3));
        assert_eq!(fired(&fx).len(), 1);
    }

    #[test]
    fn raf_registered_during_pump_waits_for_next_frame() {
        let mut fx = setup();
        let g = global_val(&fx);
        let a = callable(&mut fx, probe_a);
        let b = callable(&mut fx, probe_b);
        // When `a` runs, register `b`: snapshot semantics put it on the
        // next frame, not the current one.
        fx.invoker
            .borrow_mut()
            .set_on_call(Some(Box::new(move |heap: &mut Heap| {
                let _ = n_request_animation_frame(heap, g, &[b]);
            })));
        n_request_animation_frame(&mut fx.heap, g, &[a]).unwrap();

        assert!(pump_animation_frame(&mut fx.heap, fx.ctx, 16.7));
        assert_eq!(fired(&fx), vec![a]);

        fx.invoker.borrow_mut().set_on_call(None);
        assert!(pump_animation_frame(&mut fx.heap, fx.ctx, 33.3));
        assert_eq!(fired(&fx), vec![a, b]);
    }

    #[test]
    fn cancel_animation_frame_drops_pending_callback() {
        let mut fx = setup();
        let g = global_val(&fx);
        let a = callable(&mut fx, probe_a);
        let b = callable(&mut fx, probe_b);
        let ha = n_request_animation_frame(&mut fx.heap, g, &[a]).unwrap();
        n_request_animation_frame(&mut fx.heap, g, &[b]).unwrap();

        assert_eq!(
            n_cancel_animation_frame(&mut fx.heap, g, &[ha]).unwrap(),
            Value::Undefined
        );
        // Unknown and invalid handles are no-ops.
        n_cancel_animation_frame(&mut fx.heap, g, &[Value::Int(999)]).unwrap();
        n_cancel_animation_frame(&mut fx.heap, g, &[Value::Undefined]).unwrap();

        assert!(pump_animation_frame(&mut fx.heap, fx.ctx, 16.7));
        assert_eq!(fired(&fx), vec![b]);
    }

    #[test]
    fn pump_unknown_context_returns_false() {
        let mut fx = setup();
        assert!(!pump_animation_frame(&mut fx.heap, ContextId(999), 0.0));
    }

    #[test]
    fn try_run_timer_task_ignores_foreign_tasks() {
        let mut fx = setup();
        // A callback id this module never issued: miss, no call.
        assert!(!try_run_timer_task(
            &mut fx.heap,
            fx.ctx,
            &Task::FireTimer {
                timer_id: 1,
                callback: 4242,
            }
        ));
        // Non-timer tasks are never ours.
        assert!(!try_run_timer_task(
            &mut fx.heap,
            fx.ctx,
            &Task::PromiseJob { job: 1 }
        ));
        assert!(fired(&fx).is_empty());
    }

    #[test]
    fn pending_interval_callback_survives_gc() {
        let mut fx = setup();
        let g = global_val(&fx);
        let a = callable(&mut fx, probe_a);
        n_set_interval(&mut fx.heap, g, &[a, Value::Int(10)]).unwrap();

        // Root the global the way the engine does (the `Vm` registers its
        // interpreter, which owns the global, as a trace root): pending
        // callbacks hang off the global's holder object.
        let heap_ptr: *mut Heap = std::ptr::addr_of_mut!(fx.heap);
        let scope = unsafe {
            // SAFETY: `heap_ptr` is a live heap that outlives the scope;
            // single-threaded test, no concurrent collection. The pointer
            // uses `addr_of_mut!` (not a `&mut` reborrow) per
            // `HandleScope::new`'s contract, so later `&mut fx.heap` uses
            // do not invalidate it.
            HandleScope::new(heap_ptr)
        };
        let _rooted = scope.add(fx.global);
        fx.heap.collect();
        drop(scope);

        fx.clock.advance(15);
        run_loop(&mut fx);
        assert_eq!(fired(&fx), vec![a]);
    }

    #[test]
    fn remove_context_cancels_intervals_and_stops_pump() {
        let mut fx = setup();
        let g = global_val(&fx);
        let a = callable(&mut fx, probe_a);
        n_set_interval(&mut fx.heap, g, &[a, Value::Int(10)]).unwrap();
        n_request_animation_frame(&mut fx.heap, g, &[a]).unwrap();

        remove_timers_context(fx.ctx);
        assert!(!pump_animation_frame(&mut fx.heap, fx.ctx, 16.7));

        fx.clock.advance(50);
        run_loop(&mut fx);
        assert!(fired(&fx).is_empty());
        assert!(!fx.event_loop.has_pending_work());

        remove_timers_context(fx.ctx); // idempotent
        assert!(!pump_animation_frame(&mut fx.heap, fx.ctx, 16.7));
    }

    /// The out-of-VM frame driver fires callbacks registered from JS.
    ///
    /// Regression test for the headless-pump wiring: [`pump_animation_frame`]
    /// invokes callbacks through the [`JsInvoker`], whose re-entry needs the
    /// engine's `ACTIVE` interpreter — a direct call from outside the VM
    /// fails closed and the drain swallows it, so JS-registered callbacks
    /// silently never ran. [`pump_animation_frame_in_vm`] evals a stub
    /// calling the hidden pump native instead, entering through the
    /// engine's `call_native`, and the callback runs.
    #[test]
    fn in_vm_driver_fires_js_registered_raf() {
        use crate::vm::Vm;
        use bowser_dom::Document;

        let mut vm = Vm::new();
        let (doc, _) = Document::parse_html(b"<html><head></head><body></body></html>");
        let doc = Rc::new(RefCell::new(doc));
        let base = bowser_url::parse("http://example.com/", None).unwrap();
        let ctx = ContextId(
            vm.install_page_web_apis(doc.clone(), base, "BowserTest/1.0")
                .expect("production install"),
        );
        vm.eval("globalThis.__fired = [];").expect("eval");
        vm.eval("requestAnimationFrame(function(t) { globalThis.__fired.push(t); });")
            .expect("eval");

        assert!(pump_animation_frame_in_vm(&mut vm, 16.7));
        let fired = vm
            .eval("globalThis.__fired.length === 1 && globalThis.__fired[0] === 16.7;")
            .expect("eval");
        assert_eq!(fired, Value::Bool(true), "JS-registered rAF callback ran");

        // The direct drain from outside the VM is the failure mode the
        // driver exists to route around: the queue drains and it reports
        // success, but the callback never runs (the invoker cannot reach
        // the interpreter). Pin it so a future change here is deliberate.
        vm.eval("requestAnimationFrame(function(t) { globalThis.__fired.push(t); });")
            .expect("eval");
        assert!(crate::bindings::pump_animation_frame(
            vm.heap_mut(),
            ctx,
            33.3
        ));
        let still_one = vm.eval("globalThis.__fired.length === 1;").expect("eval");
        assert_eq!(
            still_one,
            Value::Bool(true),
            "direct out-of-VM pump does not run the callback"
        );

        // A torn-down context stays a clean `false` through the driver
        // too: the host-table entry is gone, so the hidden native
        // reports `false` instead of firing.
        remove_timers_context(ctx);
        assert!(!pump_animation_frame_in_vm(&mut vm, 0.0));
    }
}
