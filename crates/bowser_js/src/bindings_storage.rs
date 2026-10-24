//! Storage, history, navigation, and fetch-`Response` extensions.
//!
//! Installed by [`install_storage`] after
//! [`crate::bindings::install_web_apis`]. Adds:
//!
//! * `localStorage` / `sessionStorage`: in-memory key-value stores with the
//!   standard `getItem` / `setItem` / `removeItem` / `clear` / `key` /
//!   `length` surface. `localStorage` is origin-scoped and shared process
//!   wide (across tabs of the same origin); `sessionStorage` is scoped to
//!   the tab's [`ContextId`]. `key()` follows insertion order, like the
//!   spec.
//! * `history`: session history (`pushState` / `replaceState` / `back` /
//!   `forward` / `go` / `length` / `state`). Navigation updates
//!   `location.href` and the tab's base URL for coherence, but no document
//!   is (re)loaded: wiring to the real tab history is future work.
//!   `pushState` / `replaceState` store the state value serialized as JSON
//!   (a v1 stand-in for structured clone) and enforce the same-origin
//!   check on the URL argument. A top-level `undefined`/function/symbol
//!   stores no JSON and reads back as `null`.
//! * `location.assign(url)` (validates against the tab base URL, pushes a
//!   stub-history entry) and `location.reload()` (stub no-op).
//! * `json()`, `arrayBuffer()`, and `headers` on fetch `Response` objects.
//!
//! # How the `Response` methods get there
//!
//! `Response` objects are built inside `bindings.rs` (`build_response_object`),
//! which this module must not edit, and the response bodies live in a
//! private table there. So `install_storage` wraps the global `fetch`: the
//! wrapper delegates to the original native, then registers an enricher as
//! the *first* fulfillment reaction. The wrapper returns the same promise
//! (v1 `then` semantics), so user callbacks run after the enricher and
//! observe a `Response` that already carries `json()` / `arrayBuffer()` /
//! `headers`. The enricher snapshots the body through the existing
//! *synchronous* v1 `text()` into a hidden property; `json()` parses that
//! snapshot with the small std-only parser below, and `arrayBuffer()`
//! returns the snapshot bytes as a plain `Array` of numbers (there are no
//! typed arrays in v1 yet).
//!
//! `headers` is a real `Headers` wrapper with case-insensitive `get` /
//! `has`, populated from the fetched response's header fields (read out
//! of the fetch owner's response table through the hidden response
//! handle).
//!
//! # Divergences from the web platform, v1
//!
//! * Storage is in-memory only; nothing is persisted to disk.
//! * `json()` / `arrayBuffer()` are synchronous (like the existing v1
//!   `text()`), instead of returning promises.
//! * `history.state` round-trips through JSON instead of structured clone:
//!   `undefined`/functions/symbols become `null` in arrays and are dropped
//!   from objects, non-finite numbers become `null`, and cyclic values
//!   throw `TypeError`.
//! * `location.reload()` does nothing.
//! * `history.pushState` / `replaceState` throw `TypeError`, not the spec's
//!   `SecurityError`, for cross-origin URLs.
//!
//! [`install_storage`] is called by
//! [`crate::bindings::install_web_apis`]; [`remove_storage_context`] is
//! the matching teardown entry point for the runtime worker.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use bowser_url::Url;

use crate::bindings::{ApiError, ContextId, WebApis, web_apis, with_temp_root};
use crate::gc::{AllocError, Gc, Heap, object_kind};
use crate::value::{BigInt, NativeError, NativeFn, Object, ObjectKind, Property, Value};
use crate::wtf8::Wtf8;

// ---------------------------------------------------------------------------
// Hidden property keys
// ---------------------------------------------------------------------------

/// Context id, stored on every host wrapper object (same key as bindings).
const CTX_KEY: &str = "\u{0}bowser:ctx";
/// Response handle, stored on fetch `Response` wrappers (same key as bindings).
const RESPONSE_KEY: &str = "\u{0}bowser:response";
/// install_storage ran on this global already.
const INSTALLED_KEY: &str = "\u{0}bowser:storage:installed";
/// Storage kind on storage wrappers: [`KIND_LOCAL`] / [`KIND_SESSION`].
const KIND_KEY: &str = "\u{0}bowser:storage:kind";
/// Body-text snapshot on enriched `Response` objects.
const BODY_KEY: &str = "\u{0}bowser:storage:body";
/// Header-table handle on `Headers` wrappers.
const HEADERS_KEY: &str = "\u{0}bowser:storage:headers";
/// Location object, stashed on the history object.
const LOCATION_KEY: &str = "\u{0}bowser:storage:location";
/// History object, stashed on the location object.
const HISTORY_KEY: &str = "\u{0}bowser:storage:history";

const KIND_LOCAL: i32 = 0;
const KIND_SESSION: i32 = 1;

/// Bound on stub-history entries per context: no unbounded growth.
const MAX_HISTORY: usize = 50;

// ---------------------------------------------------------------------------
// Per-thread storage state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct HistEntry {
    url: String,
    /// `history.state`, serialized to JSON at push/replace time. `None`
    /// when the value serializes to nothing (a top-level `undefined`,
    /// function, or symbol, like `JSON.stringify`); it reads back as
    /// `null`. The store keeps no `Value` handles, so there is nothing
    /// for the GC to trace.
    state_json: Option<String>,
}

#[derive(Default)]
struct StubHistory {
    entries: Vec<HistEntry>,
    index: usize,
}

/// All state is plain data (`String`s, fn pointers, integer handles): no
/// `Value` handles are retained, so there is nothing for the GC to trace.
#[derive(Default)]
struct StorageState {
    /// `localStorage`: origin -> ordered entries. Process-wide.
    local: HashMap<String, Vec<(String, String)>>,
    /// `sessionStorage`: context id -> ordered entries. Per-tab.
    session: HashMap<u32, Vec<(String, String)>>,
    /// Stub session history per context.
    history: HashMap<u32, StubHistory>,
    /// Original global `fetch` native per context (wrapped at install).
    orig_fetch: HashMap<u32, NativeFn>,
    /// `Headers` snapshots: (context, handle) -> entries.
    resp_headers: HashMap<(u32, u32), Vec<(String, String)>>,
    next_header_handle: u32,
}

thread_local! {
    static STATE: RefCell<StorageState> = RefCell::new(StorageState::default());
}

/// Drop per-context storage state (call when the tab closes).
// Awaiting app page-VM integration: called from `bindings::remove_context`
// once the app instantiates a page VM.
#[allow(dead_code)]
pub fn remove_storage_context(ctx: ContextId) {
    STATE.with(|s| {
        let mut st = s.borrow_mut();
        st.session.remove(&ctx.0);
        st.history.remove(&ctx.0);
        st.orig_fetch.remove(&ctx.0);
        st.resp_headers.retain(|(c, _), _| *c != ctx.0);
    });
}

// ---------------------------------------------------------------------------
// Installation
// ---------------------------------------------------------------------------

/// Install the storage/history/`Response`-extension surface on the engine's
/// global object. `ctx` is the [`ContextId`] returned by
/// [`crate::bindings::install_web_apis`]; the global must already carry the
/// web API surface (in particular `location` and `fetch`).
///
/// Idempotent: a second call for the same global is a no-op.
// Awaiting app page-VM integration: called from `bindings::install_web_apis`
// once the app instantiates a page VM.
#[allow(dead_code)]
pub fn install_storage(
    heap: &mut Heap,
    global: Gc<Object>,
    ctx: ContextId,
) -> Result<(), ApiError> {
    if heap
        .get_property(global, Wtf8::from_str(INSTALLED_KEY))
        .is_some()
    {
        return Ok(());
    }
    let ctx_id = ctx.0;
    raw_def(heap, global, INSTALLED_KEY, Value::Bool(true))?;

    // location.assign / location.reload. The location object has no context
    // id of its own, so stamp one for the natives below.
    if let Some(Value::Object(location)) = heap
        .get_property(global, Wtf8::from_str("location"))
        .map(|p| p.value)
    {
        raw_def(heap, location, CTX_KEY, Value::Int(ctx_id as i32))?;
        def_native_install(heap, location, "assign", n_location_assign)?;
        def_native_install(heap, location, "reload", n_location_reload)?;

        // Stub history. The history object keeps the location object in a
        // hidden property (both are heap objects, so the GC traces the
        // link); the location object keeps the history object for assign().
        // `history` is unreachable from any root until the final `raw_def`
        // onto `global`, and every `def_native_install` allocates a fresh
        // function object (a possible collection) in between, so hold it
        // as a temporary root across the build.
        let history = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
        with_temp_root(heap, Value::Object(history), |heap| {
            raw_def(heap, history, CTX_KEY, Value::Int(ctx_id as i32))?;
            raw_def(heap, history, LOCATION_KEY, Value::Object(location))?;
            def_native_install(heap, history, "back", n_history_back)?;
            def_native_install(heap, history, "forward", n_history_forward)?;
            def_native_install(heap, history, "go", n_history_go)?;
            def_native_install(heap, history, "pushState", n_history_push_state)?;
            def_native_install(heap, history, "replaceState", n_history_replace_state)?;
            raw_def(heap, history, "length", Value::Int(1))?;
            raw_def(heap, history, "state", Value::Null)?;
            raw_def(heap, location, HISTORY_KEY, Value::Object(history))?;
            raw_def(heap, global, "history", Value::Object(history))?;
            Ok::<(), ApiError>(())
        })?;
    }

    install_storage_object(heap, global, ctx_id, "localStorage", KIND_LOCAL)?;
    install_storage_object(heap, global, ctx_id, "sessionStorage", KIND_SESSION)?;

    // Wrap the global fetch installed by install_web_apis so every
    // Response the page observes carries json()/arrayBuffer()/headers.
    // (If fetch is absent — e.g. install order changed — skip quietly.)
    if let Some(orig) = fetch_native_of(heap, global) {
        STATE.with(|s| {
            s.borrow_mut().orig_fetch.insert(ctx_id, orig);
        });
        def_native_install(heap, global, "fetch", n_fetch_wrapped)?;
    }

    Ok(())
}

fn install_storage_object(
    heap: &mut Heap,
    global: Gc<Object>,
    ctx: u32,
    name: &str,
    kind: i32,
) -> Result<(), ApiError> {
    // `obj` is unreachable from any root until the final `raw_def` onto
    // `global`, and the `def_native_install` calls in between allocate
    // fresh function objects (a possible collection each), so hold it as
    // a temporary root across the build.
    let obj = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    with_temp_root(heap, Value::Object(obj), |heap| {
        raw_def(heap, obj, CTX_KEY, Value::Int(ctx as i32))?;
        raw_def(heap, obj, KIND_KEY, Value::Int(kind))?;
        def_native_install(heap, obj, "getItem", n_storage_get_item)?;
        def_native_install(heap, obj, "setItem", n_storage_set_item)?;
        def_native_install(heap, obj, "removeItem", n_storage_remove_item)?;
        def_native_install(heap, obj, "clear", n_storage_clear)?;
        def_native_install(heap, obj, "key", n_storage_key)?;
        raw_def(heap, obj, "length", Value::Int(0))?;
        raw_def(heap, global, name, Value::Object(obj))?;
        Ok::<(), ApiError>(())
    })
}

fn raw_def(heap: &mut Heap, obj: Gc<Object>, name: &str, val: Value) -> Result<(), AllocError> {
    heap.define_property_str(obj, name, Property::data(val))
}

fn native_object(heap: &mut Heap, f: NativeFn) -> Result<Value, AllocError> {
    Ok(Value::Object(heap.alloc_object(Object::new(
        ObjectKind::NativeFunction(f),
    ))?))
}

/// Define a native method in one `&mut Heap` borrow (avoids nesting
/// `native_object(heap, ..)` inside `raw_def(heap, ..)`).
fn def_native_install(
    heap: &mut Heap,
    obj: Gc<Object>,
    name: &str,
    f: NativeFn,
) -> Result<(), ApiError> {
    let v = native_object(heap, f)?;
    raw_def(heap, obj, name, v).map_err(ApiError::from)
}

/// The current global `fetch`, when it is a native function.
fn fetch_native_of(heap: &Heap, global: Gc<Object>) -> Option<NativeFn> {
    let Value::Object(o) = heap.get_property(global, Wtf8::from_str("fetch"))?.value else {
        return None;
    };
    match object_kind(o) {
        ObjectKind::NativeFunction(f) => Some(f),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Native-function plumbing
// ---------------------------------------------------------------------------

fn err_type(msg: impl Into<String>) -> NativeError {
    NativeError::TypeError(msg.into())
}

fn oom() -> NativeError {
    NativeError::OutOfMemory
}

/// The context id behind the receiver's hidden key.
fn ctx_of(heap: &Heap, this: Value) -> Result<u32, NativeError> {
    let Value::Object(o) = this else {
        return Err(err_type("web api called with invalid receiver"));
    };
    match heap.get_property(o, Wtf8::from_str(CTX_KEY)) {
        Some(p) => match p.value {
            Value::Int(id) => Ok(id as u32),
            _ => Err(err_type("web api called with invalid receiver")),
        },
        None => Err(err_type("web api called with invalid receiver")),
    }
}

fn apis_of(ctx: u32) -> Result<Rc<RefCell<WebApis>>, NativeError> {
    web_apis(ContextId(ctx)).ok_or_else(|| err_type("web api context is gone"))
}

/// Define-or-overwrite for use inside natives.
fn def_data(heap: &mut Heap, obj: Gc<Object>, name: &str, val: Value) -> Result<(), NativeError> {
    raw_def(heap, obj, name, val).map_err(|_| oom())
}

fn def_native_here(
    heap: &mut Heap,
    obj: Gc<Object>,
    name: &str,
    f: NativeFn,
) -> Result<(), NativeError> {
    let v = native_object(heap, f).map_err(|_| oom())?;
    def_data(heap, obj, name, v)
}

/// JS `ToString` for the primitive cases the bindings need (mirrors the
/// conversion in bindings.rs; objects are rejected).
fn js_to_string(_heap: &Heap, v: Value) -> Result<String, NativeError> {
    match v {
        // DOMString boundary: lossy.
        Value::Str(s) => Ok(s.as_wtf8().to_string()),
        Value::Int(i) => Ok(i.to_string()),
        Value::Float(f) => Ok(float_to_string(f)),
        Value::Bool(b) => Ok(b.to_string()),
        Value::Null => Ok("null".to_string()),
        Value::Undefined => Ok("undefined".to_string()),
        _ => Err(err_type("argument cannot be converted to string")),
    }
}

fn float_to_string(f: f64) -> String {
    if f.is_nan() {
        return "NaN".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        };
    }
    format!("{f}")
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

/// Origin key for `localStorage`: `scheme://host[:port]`; opaque origins
/// (no host) get one store per scheme.
fn origin_of(ctx: u32) -> Result<String, NativeError> {
    let url = apis_of(ctx)?.borrow().base_url.clone();
    Ok(url_origin(&url))
}

fn url_origin(url: &Url) -> String {
    match url.host() {
        Some(host) => match url.port() {
            Some(p) => format!("{}://{host}:{p}", url.scheme()),
            None => format!("{}://{host}", url.scheme()),
        },
        None => format!("{}:", url.scheme()),
    }
}

fn storage_target(heap: &Heap, this: Value) -> Result<(u32, i32, Gc<Object>), NativeError> {
    let Value::Object(o) = this else {
        return Err(err_type("storage called with invalid receiver"));
    };
    let ctx = ctx_of(heap, this)?;
    let kind = match heap.get_property(o, Wtf8::from_str(KIND_KEY)) {
        Some(p) => match p.value {
            Value::Int(k) if k == KIND_LOCAL || k == KIND_SESSION => k,
            _ => return Err(err_type("storage called with invalid receiver")),
        },
        None => return Err(err_type("storage called with invalid receiver")),
    };
    Ok((ctx, kind, o))
}

fn load_entries(ctx: u32, kind: i32) -> Result<Vec<(String, String)>, NativeError> {
    if kind == KIND_LOCAL {
        let origin = origin_of(ctx)?;
        Ok(STATE.with(|s| s.borrow().local.get(&origin).cloned().unwrap_or_default()))
    } else {
        Ok(STATE.with(|s| s.borrow().session.get(&ctx).cloned().unwrap_or_default()))
    }
}

fn save_entries(ctx: u32, kind: i32, entries: Vec<(String, String)>) -> Result<(), NativeError> {
    if kind == KIND_LOCAL {
        let origin = origin_of(ctx)?;
        STATE.with(|s| {
            let mut st = s.borrow_mut();
            if entries.is_empty() {
                st.local.remove(&origin);
            } else {
                st.local.insert(origin, entries);
            }
        });
    } else {
        STATE.with(|s| {
            let mut st = s.borrow_mut();
            if entries.is_empty() {
                st.session.remove(&ctx);
            } else {
                st.session.insert(ctx, entries);
            }
        });
    }
    Ok(())
}

/// Keep the `length` data property in sync (it is a plain property: v1 has
/// no getters). Synced on every call so cross-tab mutations stay visible.
fn sync_length(heap: &mut Heap, obj: Gc<Object>, len: usize) -> Result<(), NativeError> {
    def_data(
        heap,
        obj,
        "length",
        Value::Int(i32::try_from(len).unwrap_or(i32::MAX)),
    )
}

fn n_storage_get_item(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (ctx, kind, obj) = storage_target(heap, this)?;
    let key = js_to_string(heap, args.first().copied().unwrap_or(Value::Undefined))?;
    let entries = load_entries(ctx, kind)?;
    sync_length(heap, obj, entries.len())?;
    match entries.iter().find(|(k, _)| *k == key) {
        Some((_, v)) => Ok(Value::Str(heap.alloc_str(v).map_err(|_| oom())?)),
        None => Ok(Value::Null),
    }
}

fn n_storage_set_item(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (ctx, kind, obj) = storage_target(heap, this)?;
    let key = js_to_string(heap, args.first().copied().unwrap_or(Value::Undefined))?;
    let val = js_to_string(heap, args.get(1).copied().unwrap_or(Value::Undefined))?;
    let mut entries = load_entries(ctx, kind)?;
    match entries.iter_mut().find(|(k, _)| *k == key) {
        Some(slot) => slot.1 = val,
        None => entries.push((key, val)),
    }
    let len = entries.len();
    save_entries(ctx, kind, entries)?;
    sync_length(heap, obj, len)?;
    Ok(Value::Undefined)
}

fn n_storage_remove_item(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let (ctx, kind, obj) = storage_target(heap, this)?;
    let key = js_to_string(heap, args.first().copied().unwrap_or(Value::Undefined))?;
    let mut entries = load_entries(ctx, kind)?;
    entries.retain(|(k, _)| *k != key);
    let len = entries.len();
    save_entries(ctx, kind, entries)?;
    sync_length(heap, obj, len)?;
    Ok(Value::Undefined)
}

fn n_storage_clear(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let (ctx, kind, obj) = storage_target(heap, this)?;
    save_entries(ctx, kind, Vec::new())?;
    sync_length(heap, obj, 0)?;
    Ok(Value::Undefined)
}

/// WebIDL `unsigned long` conversion, lenient v1 (out-of-range just misses).
fn to_index(_heap: &Heap, v: Value) -> usize {
    let n: u64 = match v {
        Value::Undefined | Value::Null => 0,
        Value::Bool(b) => u64::from(b),
        Value::Int(i) => u64::from(i as u32),
        Value::Float(f) => {
            if f.is_nan() {
                0
            } else {
                (f as i64) as u64
            }
        }
        Value::Str(s) => s
            .as_wtf8()
            .as_utf8()
            .and_then(|t| t.parse::<f64>().ok())
            .map(|f| if f.is_nan() { 0 } else { (f as i64) as u64 })
            .unwrap_or(0),
        _ => 0,
    };
    n as usize
}

fn n_storage_key(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (ctx, kind, obj) = storage_target(heap, this)?;
    let idx = to_index(heap, args.first().copied().unwrap_or(Value::Undefined));
    let entries = load_entries(ctx, kind)?;
    sync_length(heap, obj, entries.len())?;
    match entries.get(idx) {
        Some((k, _)) => Ok(Value::Str(heap.alloc_str(k).map_err(|_| oom())?)),
        None => Ok(Value::Null),
    }
}

// ---------------------------------------------------------------------------
// Stub history + location navigation
// ---------------------------------------------------------------------------

fn history_target(heap: &Heap, this: Value) -> Result<(u32, Gc<Object>), NativeError> {
    let Value::Object(o) = this else {
        return Err(err_type("history called with invalid receiver"));
    };
    let ctx = ctx_of(heap, this)?;
    Ok((ctx, o))
}

/// Seed the stub history from the tab's base URL on first use.
fn ensure_history(ctx: u32) {
    let seeded = STATE.with(|s| s.borrow().history.contains_key(&ctx));
    if seeded {
        return;
    }
    let href = apis_of(ctx)
        .map(|a| a.borrow().base_url.serialize())
        .unwrap_or_default();
    STATE.with(|s| {
        s.borrow_mut().history.insert(
            ctx,
            StubHistory {
                entries: vec![HistEntry {
                    url: href,
                    state_json: None,
                }],
                index: 0,
            },
        );
    });
}

fn history_index_len(ctx: u32) -> (usize, usize) {
    STATE.with(|s| {
        s.borrow()
            .history
            .get(&ctx)
            .map(|h| (h.index, h.entries.len()))
            .unwrap_or((0, 1))
    })
}

/// Move the stub history to `index` and reflect it in `location.href`,
/// the tab's base URL, the history object's `length`, and `history.state`
/// (restored from the entry's serialized JSON).
fn history_go_to(
    heap: &mut Heap,
    ctx: u32,
    history_obj: Gc<Object>,
    index: usize,
) -> Result<(), NativeError> {
    let (href, state_json) = STATE.with(|s| {
        let mut st = s.borrow_mut();
        let h = st.history.get_mut(&ctx).expect("history seeded");
        h.index = index.min(h.entries.len().saturating_sub(1));
        let e = &h.entries[h.index];
        (e.url.clone(), e.state_json.clone())
    });
    // Entries were serialized from parsed URLs, so this only fails when the
    // history was seeded from a dead context; then there is nothing to
    // reflect into.
    let Ok(url) = bowser_url::parse(&href, None) else {
        return Ok(());
    };
    reflect_url(heap, ctx, history_obj, &url)?;
    let restored = state_from_json(heap, &state_json)?;
    def_data(heap, history_obj, "state", restored)?;
    Ok(())
}

/// Push a new entry, dropping forward entries (like a real navigation).
/// A `None` URL keeps the current entry's URL (`pushState` without a URL
/// argument); `state_json` is the entry's serialized `history.state`.
fn history_push(ctx: u32, url: Option<&Url>, state_json: Option<String>) {
    ensure_history(ctx);
    STATE.with(|s| {
        let mut st = s.borrow_mut();
        let h = st.history.get_mut(&ctx).expect("history seeded");
        h.entries.truncate(h.index + 1);
        let href = url
            .map(Url::serialize)
            .unwrap_or_else(|| h.entries[h.index].url.clone());
        h.entries.push(HistEntry {
            url: href,
            state_json,
        });
        if h.entries.len() > MAX_HISTORY {
            h.entries.remove(0);
        }
        h.index = h.entries.len() - 1;
    });
}

fn reflect_url(
    heap: &mut Heap,
    ctx: u32,
    history_obj: Gc<Object>,
    url: &Url,
) -> Result<(), NativeError> {
    let href = url.serialize();
    let location = match heap.get_property(history_obj, Wtf8::from_str(LOCATION_KEY)) {
        Some(p) => match p.value {
            Value::Object(o) => o,
            _ => return Err(err_type("history: location object is gone")),
        },
        None => return Err(err_type("history: location object is gone")),
    };
    let s = heap.alloc_str(&href).map_err(|_| oom())?;
    def_data(heap, location, "href", Value::Str(s))?;
    if let Ok(apis) = apis_of(ctx) {
        apis.borrow_mut().base_url = url.clone();
    }
    let (_, len) = history_index_len(ctx);
    def_data(
        heap,
        history_obj,
        "length",
        Value::Int(i32::try_from(len).unwrap_or(i32::MAX)),
    )?;
    Ok(())
}

fn n_history_back(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let (ctx, obj) = history_target(heap, this)?;
    ensure_history(ctx);
    let (index, _) = history_index_len(ctx);
    history_go_to(heap, ctx, obj, index.saturating_sub(1))?;
    Ok(Value::Undefined)
}

fn n_history_forward(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let (ctx, obj) = history_target(heap, this)?;
    ensure_history(ctx);
    let (index, len) = history_index_len(ctx);
    history_go_to(
        heap,
        ctx,
        obj,
        index.saturating_add(1).min(len.saturating_sub(1)),
    )?;
    Ok(Value::Undefined)
}

fn to_integer(_heap: &Heap, v: Value) -> i64 {
    match v {
        Value::Undefined | Value::Null => 0,
        Value::Bool(b) => i64::from(b),
        Value::Int(i) => i64::from(i),
        // `as` saturates; NaN becomes 0. Truncation toward zero matches
        // WebIDL's integer conversion for in-range values.
        Value::Float(f) => f as i64,
        Value::Str(s) => s
            .as_wtf8()
            .as_utf8()
            .and_then(|t| t.parse::<f64>().ok())
            .map(|f| f as i64)
            .unwrap_or(0),
        _ => 0,
    }
}

fn n_history_go(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (ctx, obj) = history_target(heap, this)?;
    ensure_history(ctx);
    let delta = to_integer(heap, args.first().copied().unwrap_or(Value::Undefined));
    let (index, len) = history_index_len(ctx);
    let target = (index as i64 + delta).clamp(0, len.saturating_sub(1) as i64) as usize;
    history_go_to(heap, ctx, obj, target)?;
    Ok(Value::Undefined)
}

// ---------------------------------------------------------------------------
// history.pushState / replaceState
// ---------------------------------------------------------------------------

fn n_history_push_state(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    history_set_state(heap, this, args, true, "pushState")
}

fn n_history_replace_state(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    history_set_state(heap, this, args, false, "replaceState")
}

/// Shared `pushState` / `replaceState` implementation. `args` are
/// `(state, title, url)`; the title is historically unused and ignored.
fn history_set_state(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
    push: bool,
    name: &str,
) -> Result<Value, NativeError> {
    let (ctx, history_obj) = history_target(heap, this)?;
    ensure_history(ctx);
    let state = args.first().copied().unwrap_or(Value::Undefined);
    let url = match args.get(2).copied().unwrap_or(Value::Undefined) {
        Value::Undefined | Value::Null => None,
        v => {
            let url_str = js_to_string(heap, v)?;
            let base = apis_of(ctx)?.borrow().base_url.clone();
            let url = bowser_url::parse(&url_str, Some(&base))
                .map_err(|e| err_type(format!("history.{name}: invalid URL: {e}")))?;
            // Same-origin check (the spec throws SecurityError here; v1
            // reports TypeError, see the module docs).
            if url_origin(&url) != url_origin(&base) {
                return Err(err_type(format!(
                    "history.{name}: URL is not same-origin with the document"
                )));
            }
            Some(url)
        }
    };
    // Serialize before mutating: cyclic state throws (the spec's
    // structured-clone DataCloneError), and a failed push must not add
    // an entry.
    let state_json = state_to_json(heap, state)?;
    if push {
        history_push(ctx, url.as_ref(), state_json.clone());
    } else {
        history_replace(ctx, url.as_ref(), state_json.clone());
    }
    if let Some(u) = &url {
        // pushState/replaceState update the URL without navigating.
        reflect_url(heap, ctx, history_obj, u)?;
    } else {
        // The entry count may have changed; keep `length` in sync.
        let (_, len) = history_index_len(ctx);
        def_data(
            heap,
            history_obj,
            "length",
            Value::Int(i32::try_from(len).unwrap_or(i32::MAX)),
        )?;
    }
    let restored = state_from_json(heap, &state_json)?;
    def_data(heap, history_obj, "state", restored)?;
    Ok(Value::Undefined)
}

/// Replace the current entry's URL and/or state (`replaceState`).
fn history_replace(ctx: u32, url: Option<&Url>, state_json: Option<String>) {
    ensure_history(ctx);
    STATE.with(|s| {
        let mut st = s.borrow_mut();
        let h = st.history.get_mut(&ctx).expect("history seeded");
        let idx = h.index.min(h.entries.len().saturating_sub(1));
        let e = &mut h.entries[idx];
        if let Some(u) = url {
            e.url = u.serialize();
        }
        e.state_json = state_json;
    });
}

/// Restore `history.state` from a stored entry: the JSON round-trip the
/// module docs promise. `None` (a top-level `undefined`, function, or
/// symbol) reads back as `null`.
fn state_from_json(heap: &mut Heap, json: &Option<String>) -> Result<Value, NativeError> {
    match json {
        None => Ok(Value::Null),
        Some(s) => {
            let parsed =
                parse_json(s).map_err(|e| err_type(format!("history state is corrupt: {e}")))?;
            json_to_value(heap, &parsed).map_err(|_| oom())
        }
    }
}

// ---------------------------------------------------------------------------
// history.state JSON serialization (structured-clone stand-in)
// ---------------------------------------------------------------------------

/// Cap on serialized array length: no unbounded allocation from a hostile
/// `length` property.
const STATE_MAX_ARRAY_LEN: usize = 1 << 20;

/// Serialize a `history.state` value to JSON (the v1 stand-in for
/// structured clone). Returns `None` for values `JSON.stringify` maps to
/// no string at the top level (`undefined`, functions, symbols). Nested,
/// those become `null` in arrays and are dropped from objects, per the
/// module docs; non-finite numbers become `null`; cyclic object graphs
/// are a `TypeError` (the spec's structured-clone `DataCloneError`).
fn state_to_json(heap: &Heap, v: Value) -> Result<Option<String>, NativeError> {
    let mut seen: Vec<*const Object> = Vec::new();
    state_value_json(heap, v, &mut seen, 0)
}

fn state_value_json(
    heap: &Heap,
    v: Value,
    seen: &mut Vec<*const Object>,
    depth: usize,
) -> Result<Option<String>, NativeError> {
    if depth > JSON_MAX_DEPTH {
        return Err(err_type("history state is too deeply nested"));
    }
    match v {
        Value::Null => Ok(Some("null".to_string())),
        Value::Bool(b) => Ok(Some(b.to_string())),
        Value::Int(i) => Ok(Some(i.to_string())),
        Value::Float(f) => Ok(Some(if f.is_finite() {
            float_to_string(f)
        } else {
            "null".to_string()
        })),
        Value::Str(s) => Ok(Some(json_escape_str(s.as_wtf8()))),
        Value::BigInt(b) => Ok(Some(bigint_to_decimal(heap.get(b)))),
        Value::Undefined | Value::Symbol(_) => Ok(None),
        Value::Object(o) => state_object_json(heap, o, seen, depth),
    }
}

fn state_object_json(
    heap: &Heap,
    o: Gc<Object>,
    seen: &mut Vec<*const Object>,
    depth: usize,
) -> Result<Option<String>, NativeError> {
    // Functions have no JSON representation (like `JSON.stringify`).
    let is_array = match object_kind(o) {
        ObjectKind::Array => true,
        ObjectKind::Ordinary | ObjectKind::RegExp | ObjectKind::Date { .. } => false,
        ObjectKind::TypedArrayView { .. }
        | ObjectKind::ArrayBuffer
        | ObjectKind::DataView { .. } => false,
        ObjectKind::Function(_) | ObjectKind::NativeFunction(_) => return Ok(None),
    };
    let ptr = o.as_ptr();
    if seen.contains(&ptr) {
        return Err(err_type("history state is cyclic"));
    }
    seen.push(ptr);
    let out = if is_array {
        state_array_json(heap, o, seen, depth)
    } else {
        state_plain_object_json(heap, o, seen, depth)
    };
    seen.pop();
    out.map(Some)
}

fn state_array_json(
    heap: &Heap,
    o: Gc<Object>,
    seen: &mut Vec<*const Object>,
    depth: usize,
) -> Result<String, NativeError> {
    let len = match heap
        .get_property(o, Wtf8::from_str("length"))
        .map(|p| p.value)
    {
        Some(v) => to_index(heap, v),
        None => 0,
    };
    if len > STATE_MAX_ARRAY_LEN {
        return Err(err_type("history state array is too large"));
    }
    let mut out = String::from("[");
    for i in 0..len {
        if i > 0 {
            out.push(',');
        }
        let v = heap
            .get_property(o, Wtf8::from_str(&i.to_string()))
            .map(|p| p.value)
            .unwrap_or(Value::Undefined);
        match state_value_json(heap, v, seen, depth + 1)? {
            Some(s) => out.push_str(&s),
            // Holes and unserializable values become null, like JSON.
            None => out.push_str("null"),
        }
    }
    out.push(']');
    Ok(out)
}

fn state_plain_object_json(
    heap: &Heap,
    o: Gc<Object>,
    seen: &mut Vec<*const Object>,
    depth: usize,
) -> Result<String, NativeError> {
    let mut out = String::from("{");
    let mut first = true;
    for (name_handle, prop) in o.properties.iter() {
        if !prop.enumerable {
            continue;
        }
        let name = name_handle.as_wtf8();
        // Host-internal hidden keys never leak into serialized state.
        if name.starts_with(Wtf8::from_str("\u{0}")) {
            continue;
        }
        // Unserializable values (undefined/functions/symbols) are dropped
        // from objects, like `JSON.stringify`.
        let Some(s) = state_value_json(heap, prop.value, seen, depth + 1)? else {
            continue;
        };
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(&json_escape_str(name));
        out.push(':');
        out.push_str(&s);
    }
    out.push('}');
    Ok(out)
}

/// Quote a string as a JSON string literal.
/// Escape a JS string as a JSON string literal. Operates on UTF-16 code
/// units: lone surrogates are emitted as `\uXXXX` escapes (valid JSON that
/// `JSON.parse` reads back as the same lone surrogate); astral pairs are
/// re-combined and emitted raw, as before.
fn json_escape_str(s: &Wtf8) -> String {
    let units = s.to_units();
    let mut out = String::with_capacity(s.as_bytes().len() + 2);
    out.push('"');
    let mut i = 0;
    while i < units.len() {
        let u = units[i];
        if (0xd800..0xdc00).contains(&u)
            && i + 1 < units.len()
            && (0xdc00..0xe000).contains(&units[i + 1])
        {
            let c = char::from_u32(
                0x1_0000 + ((u as u32 - 0xd800) << 10) + (units[i + 1] as u32 - 0xdc00),
            )
            .expect("astral pair decodes");
            out.push(c);
            i += 2;
            continue;
        }
        match u {
            0x22 => out.push_str("\\\""),
            0x5c => out.push_str("\\\\"),
            0x0a => out.push_str("\\n"),
            0x0d => out.push_str("\\r"),
            0x09 => out.push_str("\\t"),
            0x08 => out.push_str("\\b"),
            0x0c => out.push_str("\\f"),
            u if u < 0x20 || (0xd800..0xe000).contains(&u) => {
                out.push_str(&format!("\\u{u:04x}"));
            }
            u => out.push(char::from_u32(u as u32).expect("bmp unit decodes")),
        }
        i += 1;
    }
    out.push('"');
    out
}

/// Render a [`BigInt`] (sign plus base-2^32 digits, least significant
/// first) in decimal, as a JSON number literal. Precision beyond `f64` is
/// lost when the state is read back; see the module docs.
fn bigint_to_decimal(b: &BigInt) -> String {
    let mut digits = b.digits.clone();
    while digits.len() > 1 && digits.last() == Some(&0) {
        digits.pop();
    }
    if digits.iter().all(|&d| d == 0) {
        return "0".to_string();
    }
    // Repeated divmod by 1e9; groups come out least significant first.
    let mut groups: Vec<u32> = Vec::new();
    while !digits.is_empty() {
        let mut rem: u64 = 0;
        for d in digits.iter_mut().rev() {
            let cur = (rem << 32) | u64::from(*d);
            *d = (cur / 1_000_000_000) as u32;
            rem = cur % 1_000_000_000;
        }
        groups.push(rem as u32);
        while digits.last() == Some(&0) {
            digits.pop();
        }
    }
    let mut out = String::new();
    if b.negative {
        out.push('-');
    }
    let mut it = groups.iter().rev();
    match it.next() {
        Some(first) => out.push_str(&first.to_string()),
        None => out.push('0'),
    }
    for g in it {
        out.push_str(&format!("{g:09}"));
    }
    out
}

fn n_location_assign(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let ctx = ctx_of(heap, this)?;
    let url_str = js_to_string(heap, args.first().copied().unwrap_or(Value::Undefined))?;
    let base = apis_of(ctx)?.borrow().base_url.clone();
    let url = bowser_url::parse(&url_str, Some(&base))
        .map_err(|e| err_type(format!("location.assign: invalid URL: {e}")))?;
    history_push(ctx, Some(&url), None);
    let Value::Object(lo) = this else {
        return Err(err_type("location.assign: invalid receiver"));
    };
    let history_obj = match heap.get_property(lo, Wtf8::from_str(HISTORY_KEY)) {
        Some(p) => match p.value {
            Value::Object(o) => o,
            _ => return Err(err_type("location.assign: history object is gone")),
        },
        None => return Err(err_type("location.assign: history object is gone")),
    };
    reflect_url(heap, ctx, history_obj, &url)?;
    Ok(Value::Undefined)
}

/// Stub: v1 never (re)loads a document.
fn n_location_reload(
    _heap: &mut Heap,
    _this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    Ok(Value::Undefined)
}

// ---------------------------------------------------------------------------
// fetch Response enrichment
// ---------------------------------------------------------------------------

/// Wrapped global `fetch`: delegates to the original native, then
/// registers the [`n_enrich_response`] reaction first so user callbacks
/// observe `Response` objects already carrying `json()` / `arrayBuffer()` /
/// `headers`. Returns the same promise (v1 `then` semantics).
fn n_fetch_wrapped(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let ctx = ctx_of(heap, this)?;
    let orig = STATE
        .with(|s| s.borrow().orig_fetch.get(&ctx).copied())
        .ok_or_else(|| err_type("fetch: original fetch is not installed"))?;
    let promise = orig(heap, this, args)?;
    if let Value::Object(p) = promise {
        let then_obj = match heap.get_property(p, Wtf8::from_str("then")) {
            Some(pr) => match pr.value {
                Value::Object(t) => Some(t),
                _ => None,
            },
            None => None,
        };
        if let Some(t) = then_obj
            && let ObjectKind::NativeFunction(then_fn) = object_kind(t)
        {
            let enrich = native_object(heap, n_enrich_response).map_err(|_| oom())?;
            // The promise is pending here: outcomes only surface through
            // poll_fetch_completions on the JS thread, which has not run.
            let _ = then_fn(heap, promise, &[enrich]);
        }
    }
    Ok(promise)
}

/// First fulfillment reaction on every wrapped `fetch()`: installs the
/// extended `Response` surface on the settled value.
fn n_enrich_response(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let Value::Object(r) = args.first().copied().unwrap_or(Value::Undefined) else {
        return Ok(Value::Undefined);
    };
    if !is_response(heap, r) {
        return Ok(Value::Undefined);
    }
    // Snapshot the body through the existing synchronous v1 text(); the
    // headers come from the fetch owner's response table via
    // `WebApis::response_headers`.
    if let Some(body) = response_text_of(heap, r) {
        let s = heap.alloc_str(&body).map_err(|_| oom())?;
        def_data(heap, r, BODY_KEY, Value::Str(s))?;
    }
    def_native_here(heap, r, "json", n_resp_json)?;
    def_native_here(heap, r, "arrayBuffer", n_resp_array_buffer)?;
    // Real headers from the fetch owner's response table, looked up by
    // the hidden response handle.
    let ctx = ctx_of(heap, Value::Object(r)).unwrap_or(0);
    let headers: Vec<(String, String)> = match heap
        .get_property(r, Wtf8::from_str(RESPONSE_KEY))
        .map(|p| p.value)
    {
        Some(Value::Int(h)) => web_apis(ContextId(ctx))
            .and_then(|apis| {
                apis.borrow()
                    .response_headers(h as u32)
                    .map(|hs| hs.to_vec())
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    };
    let headers_obj = new_headers_object(heap, ctx, headers)?;
    def_data(heap, r, "headers", headers_obj)?;
    Ok(Value::Undefined)
}

fn is_response(heap: &Heap, r: Gc<Object>) -> bool {
    matches!(
        heap.get_property(r, Wtf8::from_str(RESPONSE_KEY))
            .map(|p| p.value),
        Some(Value::Int(_))
    )
}

/// Call the wrapper's own `text()` method. Returns `None` when it is
/// missing or misbehaving (the page may have clobbered it).
fn response_text_of(heap: &mut Heap, r: Gc<Object>) -> Option<String> {
    let text_val = heap.get_property(r, Wtf8::from_str("text"))?.value;
    let Value::Object(t) = text_val else {
        return None;
    };
    let ObjectKind::NativeFunction(f) = object_kind(t) else {
        return None;
    };
    match f(heap, Value::Object(r), &[]) {
        Ok(Value::Str(s)) => Some(s.as_wtf8().to_string()),
        _ => None,
    }
}

/// The body snapshot taken by the enricher.
fn response_body(heap: &Heap, this: Value, method: &str) -> Result<String, NativeError> {
    let Value::Object(r) = this else {
        return Err(err_type(format!("{method}: called with invalid receiver")));
    };
    if !is_response(heap, r) {
        return Err(err_type(format!("{method}: called with invalid receiver")));
    }
    match heap
        .get_property(r, Wtf8::from_str(BODY_KEY))
        .map(|p| p.value)
    {
        Some(Value::Str(s)) => Ok(s.as_wtf8().to_string()),
        _ => Err(err_type(format!(
            "{method}: response body is not available"
        ))),
    }
}

/// `Response.json()`: parse the body snapshot. Synchronous in v1, like the
/// existing `text()` (the spec returns a promise).
fn n_resp_json(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let body = response_body(heap, this, "json")?;
    let parsed = parse_json(&body).map_err(NativeError::SyntaxError)?;
    json_to_value(heap, &parsed).map_err(|_| oom())
}

/// `Response.arrayBuffer()`: v1 has no typed arrays, so the bytes come
/// back as a plain `Array` of numbers (documented divergence).
fn n_resp_array_buffer(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let body = response_body(heap, this, "arrayBuffer")?;
    let arr = heap
        .alloc_object(Object::new(ObjectKind::Array))
        .map_err(|_| oom())?;
    for (i, b) in body.as_bytes().iter().enumerate() {
        raw_def(heap, arr, &i.to_string(), Value::Int(i32::from(*b))).map_err(|_| oom())?;
    }
    raw_def(
        heap,
        arr,
        "length",
        Value::Int(i32::try_from(body.len()).unwrap_or(i32::MAX)),
    )
    .map_err(|_| oom())?;
    Ok(Value::Object(arr))
}

// ---------------------------------------------------------------------------
// Headers
// ---------------------------------------------------------------------------

/// Build a `Headers` wrapper over `headers`. Name lookup is ASCII
/// case-insensitive, per the Fetch spec.
fn new_headers_object(
    heap: &mut Heap,
    ctx: u32,
    headers: Vec<(String, String)>,
) -> Result<Value, NativeError> {
    let handle = STATE.with(|s| {
        let mut st = s.borrow_mut();
        let h = st.next_header_handle;
        st.next_header_handle = h.wrapping_add(1);
        st.resp_headers.insert((ctx, h), headers);
        h
    });
    let obj = heap
        .alloc_object(Object::new(ObjectKind::Ordinary))
        .map_err(|_| oom())?;
    def_data(heap, obj, CTX_KEY, Value::Int(ctx as i32))?;
    def_data(heap, obj, HEADERS_KEY, Value::Int(handle as i32))?;
    def_native_here(heap, obj, "get", n_headers_get)?;
    def_native_here(heap, obj, "has", n_headers_has)?;
    Ok(Value::Object(obj))
}

fn headers_target(heap: &Heap, this: Value, method: &str) -> Result<(u32, u32), NativeError> {
    let Value::Object(o) = this else {
        return Err(err_type(format!("Headers.{method}: invalid receiver")));
    };
    let ctx = ctx_of(heap, this)?;
    let handle = match heap.get_property(o, Wtf8::from_str(HEADERS_KEY)) {
        Some(p) => match p.value {
            Value::Int(h) => h as u32,
            _ => return Err(err_type(format!("Headers.{method}: invalid receiver"))),
        },
        None => return Err(err_type(format!("Headers.{method}: invalid receiver"))),
    };
    Ok((ctx, handle))
}

fn n_headers_get(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (ctx, handle) = headers_target(heap, this, "get")?;
    let name = js_to_string(heap, args.first().copied().unwrap_or(Value::Undefined))?;
    let found = STATE.with(|s| {
        s.borrow().resp_headers.get(&(ctx, handle)).and_then(|hs| {
            hs.iter()
                .find(|(n, _)| n.eq_ignore_ascii_case(&name))
                .map(|(_, v)| v.clone())
        })
    });
    match found {
        Some(v) => Ok(Value::Str(heap.alloc_str(&v).map_err(|_| oom())?)),
        None => Ok(Value::Null),
    }
}

fn n_headers_has(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (ctx, handle) = headers_target(heap, this, "has")?;
    let name = js_to_string(heap, args.first().copied().unwrap_or(Value::Undefined))?;
    let found = STATE.with(|s| {
        s.borrow()
            .resp_headers
            .get(&(ctx, handle))
            .is_some_and(|hs| hs.iter().any(|(n, _)| n.eq_ignore_ascii_case(&name)))
    });
    Ok(Value::Bool(found))
}

// ---------------------------------------------------------------------------
// Minimal JSON parser (std only)
// ---------------------------------------------------------------------------

/// Parsed JSON number: integers that fit `i32` stay integers.
enum JsonNum {
    Int(i32),
    Float(f64),
}

enum JsonVal {
    Null,
    Bool(bool),
    Num(JsonNum),
    Str(String),
    Arr(Vec<JsonVal>),
    Obj(Vec<(String, JsonVal)>),
}

const JSON_MAX_DEPTH: usize = 128;

struct JsonParser<'a> {
    b: &'a [u8],
    pos: usize,
}

/// Parse one complete JSON document; trailing characters are an error.
fn parse_json(text: &str) -> Result<JsonVal, String> {
    let mut p = JsonParser {
        b: text.as_bytes(),
        pos: 0,
    };
    p.ws();
    let v = p.value(0)?;
    p.ws();
    if p.pos != p.b.len() {
        return Err(format!(
            "unexpected trailing characters at offset {}",
            p.pos
        ));
    }
    Ok(v)
}

impl<'a> JsonParser<'a> {
    fn err(&self, msg: &str) -> String {
        format!("{msg} at offset {}", self.pos)
    }

    fn ws(&mut self) {
        while self.pos < self.b.len() && matches!(self.b[self.pos], b' ' | b'\t' | b'\n' | b'\r') {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }

    fn value(&mut self, depth: usize) -> Result<JsonVal, String> {
        if depth > JSON_MAX_DEPTH {
            return Err(self.err("too deeply nested"));
        }
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(JsonVal::Str(self.string()?)),
            Some(b't') => self.literal("true", JsonVal::Bool(true)),
            Some(b'f') => self.literal("false", JsonVal::Bool(false)),
            Some(b'n') => self.literal("null", JsonVal::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            _ => Err(self.err("unexpected character")),
        }
    }

    fn literal(&mut self, word: &str, v: JsonVal) -> Result<JsonVal, String> {
        if self.b[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(v)
        } else {
            Err(self.err("invalid literal"))
        }
    }

    fn object(&mut self, depth: usize) -> Result<JsonVal, String> {
        self.pos += 1; // consume '{'
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonVal::Obj(items));
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return Err(self.err("expected string key"));
            }
            let k = self.string()?;
            self.ws();
            if self.peek() != Some(b':') {
                return Err(self.err("expected ':'"));
            }
            self.pos += 1;
            self.ws();
            let v = self.value(depth + 1)?;
            items.push((k, v));
            self.ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonVal::Obj(items));
                }
                _ => return Err(self.err("expected ',' or '}'")),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<JsonVal, String> {
        self.pos += 1; // consume '['
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonVal::Arr(items));
        }
        loop {
            self.ws();
            items.push(self.value(depth + 1)?);
            self.ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonVal::Arr(items));
                }
                _ => return Err(self.err("expected ',' or ']'")),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.pos += 1; // consume opening '"'
        let mut out = String::new();
        loop {
            let c = self.peek().ok_or_else(|| self.err("unterminated string"))?;
            match c {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    let e = self.peek().ok_or_else(|| self.err("unterminated escape"))?;
                    self.pos += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\x08'),
                        b'f' => out.push('\x0C'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if self.b.get(self.pos..self.pos + 2) != Some(b"\\u".as_slice()) {
                                    return Err(self.err("lone high surrogate"));
                                }
                                self.pos += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(self.err("invalid low surrogate"));
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&hi) {
                                return Err(self.err("lone low surrogate"));
                            } else {
                                hi
                            };
                            out.push(
                                char::from_u32(cp).ok_or_else(|| self.err("invalid code point"))?,
                            );
                        }
                        _ => return Err(self.err("invalid escape")),
                    }
                }
                0x00..=0x1F => return Err(self.err("unescaped control character")),
                _ => {
                    // The input is a valid &str; decode one char at pos.
                    let rest = std::str::from_utf8(&self.b[self.pos..])
                        .map_err(|_| self.err("bad utf8"))?;
                    let ch = rest
                        .chars()
                        .next()
                        .ok_or_else(|| self.err("unterminated string"))?;
                    let start = self.pos;
                    self.pos = start + ch.len_utf8();
                    out.push(ch);
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        if self.pos + 4 > self.b.len() {
            return Err(self.err("truncated \\u escape"));
        }
        let mut v = 0u32;
        for &d in &self.b[self.pos..self.pos + 4] {
            v = v * 16
                + match d {
                    b'0'..=b'9' => u32::from(d - b'0'),
                    b'a'..=b'f' => u32::from(d - b'a') + 10,
                    b'A'..=b'F' => u32::from(d - b'A') + 10,
                    _ => return Err(self.err("invalid hex digit")),
                };
        }
        self.pos += 4;
        Ok(v)
    }

    fn number(&mut self) -> Result<JsonVal, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(d) if d.is_ascii_digit() => {
                while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.pos += 1;
                }
            }
            _ => return Err(self.err("invalid number")),
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.pos += 1;
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(self.err("invalid number"));
            }
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(self.err("invalid number"));
            }
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        let text = std::str::from_utf8(&self.b[start..self.pos])
            .map_err(|_| self.err("invalid number"))?;
        if !is_float && let Ok(i) = text.parse::<i32>() {
            return Ok(JsonVal::Num(JsonNum::Int(i)));
        }
        text.parse::<f64>()
            .map(|f| JsonVal::Num(JsonNum::Float(f)))
            .map_err(|_| self.err("invalid number"))
    }
}

/// Convert a parsed JSON value into engine values. Objects become ordinary
/// objects (duplicate keys: last wins, per spec); arrays become `Array`
/// objects with indexed properties plus `length`, mirroring the
/// `querySelectorAll` wrapper shape.
fn json_to_value(heap: &mut Heap, v: &JsonVal) -> Result<Value, AllocError> {
    match v {
        JsonVal::Null => Ok(Value::Null),
        JsonVal::Bool(b) => Ok(Value::Bool(*b)),
        JsonVal::Num(JsonNum::Int(i)) => Ok(Value::Int(*i)),
        JsonVal::Num(JsonNum::Float(f)) => Ok(Value::Float(*f)),
        JsonVal::Str(s) => Ok(Value::Str(heap.alloc_str(s)?)),
        JsonVal::Arr(items) => {
            let arr = heap.alloc_object(Object::new(ObjectKind::Array))?;
            for (i, item) in items.iter().enumerate() {
                let w = json_to_value(heap, item)?;
                raw_def(heap, arr, &i.to_string(), w)?;
            }
            raw_def(
                heap,
                arr,
                "length",
                Value::Int(i32::try_from(items.len()).unwrap_or(i32::MAX)),
            )?;
            Ok(Value::Object(arr))
        }
        JsonVal::Obj(entries) => {
            let obj = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
            for (k, item) in entries {
                let w = json_to_value(heap, item)?;
                raw_def(heap, obj, k, w)?;
            }
            Ok(Value::Object(obj))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::{
        BindingRunner, InvokeError, JsInvoker, install_web_apis, poll_fetch_completions,
    };
    use crate::event_loop::{Clock, EventLoop, FakeClock};
    use bowser_dom::Document;

    #[derive(Default)]
    struct RecInvoker {
        calls: RefCell<Vec<(Value, Vec<Value>)>>,
    }

    impl JsInvoker for RecInvoker {
        fn call(
            &self,
            heap: &mut Heap,
            func: Value,
            this: Value,
            args: &[Value],
        ) -> Result<Value, InvokeError> {
            self.calls.borrow_mut().push((func, args.to_vec()));
            // Actually execute native reactions (notably the Response
            // enricher) so the test observes their effects; other
            // callbacks are merely recorded.
            if let Value::Object(o) = func
                && let ObjectKind::NativeFunction(f) = object_kind(o)
            {
                f(heap, this, args)
                    .map_err(|e| InvokeError(format!("native reaction failed: {e:?}")))?;
            }
            Ok(Value::Undefined)
        }
    }

    struct Fx {
        heap: Heap,
        global: Gc<Object>,
        ctx: ContextId,
        apis: Rc<RefCell<WebApis>>,
        event_loop: Rc<EventLoop>,
        invoker: Rc<RefCell<RecInvoker>>,
        // The fixture plays the embedder's role: it owns the sole
        // strong document handle while the registry keeps only a
        // `Weak` (see `WebApis::document`). Without this the weak
        // handle would dangle the moment `setup` returns.
        #[allow(dead_code)]
        doc: Rc<RefCell<Document>>,
    }

    fn setup(base_url: &str) -> Fx {
        let mut heap = Heap::new();
        let global = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let (doc, _) = Document::parse_html(b"<html><body></body></html>");
        let doc = Rc::new(RefCell::new(doc));
        let clock = Rc::new(FakeClock::new());
        let event_loop = Rc::new(EventLoop::with_clock(clock as Rc<dyn Clock>));
        let invoker = Rc::new(RefCell::new(RecInvoker::default()));
        let invoker_obj: Rc<RefCell<dyn JsInvoker>> = invoker.clone();
        let base = bowser_url::parse(base_url, None).unwrap();
        let ctx = install_web_apis(
            &mut heap,
            global,
            doc.clone(),
            base,
            event_loop.clone(),
            invoker_obj,
            "Test/1.0",
        )
        .unwrap();
        install_storage(&mut heap, global, ctx).unwrap();
        let apis = web_apis(ctx).expect("context registered");
        Fx {
            heap,
            global,
            ctx,
            apis,
            event_loop,
            invoker,
            doc,
        }
    }

    fn obj_of(v: Value) -> Gc<Object> {
        let Value::Object(o) = v else {
            panic!("not an object: {v:?}");
        };
        o
    }

    fn prop(fx: &Fx, obj: Value, name: &str) -> Value {
        fx.heap
            .get_property(obj_of(obj), Wtf8::from_str(name))
            .unwrap()
            .value
    }

    fn global_prop(fx: &Fx, name: &str) -> Value {
        prop(fx, Value::Object(fx.global), name)
    }

    fn s(fx: &mut Fx, text: &str) -> Value {
        Value::Str(fx.heap.alloc_str(text).unwrap())
    }

    fn str_of(_fx: &Fx, v: Value) -> String {
        match v {
            Value::Str(g) => g.as_wtf8().to_string(),
            _ => panic!("not a string: {v:?}"),
        }
    }

    /// Call a native method by name on a wrapper object.
    fn call_method(
        fx: &mut Fx,
        obj: Value,
        name: &str,
        args: &[Value],
    ) -> Result<Value, NativeError> {
        let f = prop(fx, obj, name);
        let Value::Object(o) = f else {
            panic!("{name} is not an object");
        };
        let ObjectKind::NativeFunction(nf) = object_kind(o) else {
            panic!("{name} is not a native function");
        };
        nf(&mut fx.heap, obj, args)
    }

    /// Build the argument array before the call so the `&mut Fx` borrows
    /// in the arguments never overlap the call's own borrow.
    macro_rules! call {
        ($fx:ident, $obj:expr, $name:expr, [$($a:expr),* $(,)?]) => {{
            let args = [$($a),*];
            call_method(&mut $fx, $obj, $name, &args)
        }};
    }

    fn run_loop(fx: &mut Fx) {
        let mut runner = BindingRunner::new(&mut fx.heap, fx.apis.clone());
        fx.event_loop.run_until_idle(&mut runner);
    }

    // -- storage ----------------------------------------------------------

    #[test]
    fn storage_roundtrip_within_session() {
        let mut fx = setup("http://example.com/");
        let ls = global_prop(&fx, "localStorage");
        call!(fx, ls, "setItem", [s(&mut fx, "k"), s(&mut fx, "v")]).unwrap();
        let got = call!(fx, ls, "getItem", [s(&mut fx, "k")]).unwrap();
        assert_eq!(str_of(&fx, got), "v");
        let missing = call!(fx, ls, "getItem", [s(&mut fx, "nope")]).unwrap();
        assert_eq!(missing, Value::Null);
    }

    #[test]
    fn storage_origin_isolation() {
        let mut a1 = setup("http://a.com/");
        let mut a2 = setup("http://a.com/");
        let mut b = setup("http://b.com/");
        // localStorage is origin-scoped: shared across tabs of one origin.
        let la1 = global_prop(&a1, "localStorage");
        call!(a1, la1, "setItem", [s(&mut a1, "k"), s(&mut a1, "v")]).unwrap();
        let la2 = global_prop(&a2, "localStorage");
        let got = call!(a2, la2, "getItem", [s(&mut a2, "k")]).unwrap();
        assert_eq!(str_of(&a2, got), "v");
        // A different origin sees nothing.
        let lb = global_prop(&b, "localStorage");
        let missing = call!(b, lb, "getItem", [s(&mut b, "k")]).unwrap();
        assert_eq!(missing, Value::Null);
        // sessionStorage is per-tab even for the same origin.
        let sa1 = global_prop(&a1, "sessionStorage");
        call!(a1, sa1, "setItem", [s(&mut a1, "k"), s(&mut a1, "v")]).unwrap();
        let sa2 = global_prop(&a2, "sessionStorage");
        let missing2 = call!(a2, sa2, "getItem", [s(&mut a2, "k")]).unwrap();
        assert_eq!(missing2, Value::Null);
        let back = call!(a1, sa1, "getItem", [s(&mut a1, "k")]).unwrap();
        assert_eq!(str_of(&a1, back), "v");
    }

    #[test]
    fn storage_remove_clear_key_length() {
        let mut fx = setup("http://example.com/");
        let ls = global_prop(&fx, "localStorage");
        assert_eq!(prop(&fx, ls, "length"), Value::Int(0));
        call!(fx, ls, "setItem", [s(&mut fx, "b"), s(&mut fx, "2")]).unwrap();
        call!(fx, ls, "setItem", [s(&mut fx, "a"), s(&mut fx, "1")]).unwrap();
        assert_eq!(prop(&fx, ls, "length"), Value::Int(2));
        // key() follows insertion order.
        let k0 = call!(fx, ls, "key", [Value::Int(0)]).unwrap();
        assert_eq!(str_of(&fx, k0), "b");
        let k1 = call!(fx, ls, "key", [Value::Int(1)]).unwrap();
        assert_eq!(str_of(&fx, k1), "a");
        assert_eq!(call!(fx, ls, "key", [Value::Int(2)]).unwrap(), Value::Null);
        assert_eq!(call!(fx, ls, "key", [Value::Int(-1)]).unwrap(), Value::Null);
        // Overwriting keeps the original position.
        call!(fx, ls, "setItem", [s(&mut fx, "b"), s(&mut fx, "3")]).unwrap();
        assert_eq!(prop(&fx, ls, "length"), Value::Int(2));
        let k0b = call!(fx, ls, "key", [Value::Int(0)]).unwrap();
        assert_eq!(str_of(&fx, k0b), "b");
        let v = call!(fx, ls, "getItem", [s(&mut fx, "b")]).unwrap();
        assert_eq!(str_of(&fx, v), "3");
        // removeItem drops the entry and compacts the order.
        call!(fx, ls, "removeItem", [s(&mut fx, "b")]).unwrap();
        assert_eq!(prop(&fx, ls, "length"), Value::Int(1));
        let k0c = call!(fx, ls, "key", [Value::Int(0)]).unwrap();
        assert_eq!(str_of(&fx, k0c), "a");
        // clear() empties everything.
        call!(fx, ls, "clear", []).unwrap();
        assert_eq!(prop(&fx, ls, "length"), Value::Int(0));
        assert_eq!(call!(fx, ls, "key", [Value::Int(0)]).unwrap(), Value::Null);
    }

    #[test]
    fn remove_storage_context_drops_session() {
        let mut fx = setup("http://example.com/");
        let ss = global_prop(&fx, "sessionStorage");
        call!(fx, ss, "setItem", [s(&mut fx, "k"), s(&mut fx, "v")]).unwrap();
        remove_storage_context(fx.ctx);
        let gone = call!(fx, ss, "getItem", [s(&mut fx, "k")]).unwrap();
        assert_eq!(gone, Value::Null);
    }

    #[test]
    fn install_storage_is_idempotent() {
        let mut fx = setup("http://example.com/");
        let ls1 = global_prop(&fx, "localStorage");
        install_storage(&mut fx.heap, fx.global, fx.ctx).unwrap();
        let ls2 = global_prop(&fx, "localStorage");
        assert_eq!(ls1, ls2);
    }

    // -- history / location -------------------------------------------------

    #[test]
    fn history_stub_initial_state() {
        let fx = setup("http://example.com/");
        let hist = global_prop(&fx, "history");
        assert_eq!(prop(&fx, hist, "length"), Value::Int(1));
        assert_eq!(prop(&fx, hist, "state"), Value::Null);
    }

    #[test]
    fn location_assign_validates_and_drives_stub_history() {
        let mut fx = setup("http://example.com/");
        let loc = global_prop(&fx, "location");
        // Invalid URLs are rejected.
        let bad = s(&mut fx, "http://exa mple.com/");
        assert!(matches!(
            call!(fx, loc, "assign", [bad]),
            Err(NativeError::TypeError(_))
        ));
        // A relative URL resolves against the tab's base URL.
        let rel = s(&mut fx, "page2.html");
        call!(fx, loc, "assign", [rel]).unwrap();
        assert_eq!(
            str_of(&fx, prop(&fx, loc, "href")),
            "http://example.com/page2.html"
        );
        let hist = global_prop(&fx, "history");
        assert_eq!(prop(&fx, hist, "length"), Value::Int(2));
        // back()/forward()/go() move through the stub entries.
        call!(fx, hist, "back", []).unwrap();
        assert_eq!(str_of(&fx, prop(&fx, loc, "href")), "http://example.com/");
        call!(fx, hist, "forward", []).unwrap();
        assert_eq!(
            str_of(&fx, prop(&fx, loc, "href")),
            "http://example.com/page2.html"
        );
        call!(fx, hist, "go", [Value::Int(-1)]).unwrap();
        assert_eq!(str_of(&fx, prop(&fx, loc, "href")), "http://example.com/");
        // back() at the first entry is a no-op.
        call!(fx, hist, "back", []).unwrap();
        assert_eq!(str_of(&fx, prop(&fx, loc, "href")), "http://example.com/");
        assert_eq!(prop(&fx, hist, "length"), Value::Int(2));
        // reload() is a stub no-op returning undefined.
        assert_eq!(call!(fx, loc, "reload", []).unwrap(), Value::Undefined);
        assert_eq!(str_of(&fx, prop(&fx, loc, "href")), "http://example.com/");
    }

    #[test]
    fn location_assign_updates_base_url_for_relative_resolution() {
        let mut fx = setup("http://example.com/dir/");
        let loc = global_prop(&fx, "location");
        let abs = s(&mut fx, "http://other.com/x");
        call!(fx, loc, "assign", [abs]).unwrap();
        // The next relative assign resolves against the new base.
        let rel = s(&mut fx, "y");
        call!(fx, loc, "assign", [rel]).unwrap();
        assert_eq!(str_of(&fx, prop(&fx, loc, "href")), "http://other.com/y");
    }

    // -- history.pushState / replaceState -----------------------------------

    /// Ordinary object with the given data properties (test helper).
    fn test_obj(fx: &mut Fx, props: &[(&str, Value)]) -> Value {
        let o = fx
            .heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        for (k, v) in props {
            raw_def(&mut fx.heap, o, k, *v).unwrap();
        }
        Value::Object(o)
    }

    /// Array object with indexed elements plus `length` (test helper).
    fn test_arr(fx: &mut Fx, items: &[Value]) -> Value {
        let o = fx
            .heap
            .alloc_object(Object::new(ObjectKind::Array))
            .unwrap();
        for (i, v) in items.iter().enumerate() {
            raw_def(&mut fx.heap, o, &i.to_string(), *v).unwrap();
        }
        raw_def(
            &mut fx.heap,
            o,
            "length",
            Value::Int(i32::try_from(items.len()).unwrap()),
        )
        .unwrap();
        Value::Object(o)
    }

    fn dummy_native(_heap: &mut Heap, _this: Value, _args: &[Value]) -> Result<Value, NativeError> {
        Ok(Value::Undefined)
    }

    #[test]
    fn history_push_state_sets_state_and_url() {
        let mut fx = setup("http://example.com/");
        let hist = global_prop(&fx, "history");
        let loc = global_prop(&fx, "location");
        let bx = s(&mut fx, "x");
        let state = test_obj(&mut fx, &[("a", Value::Int(1)), ("b", bx)]);
        call!(
            fx,
            hist,
            "pushState",
            [state, s(&mut fx, ""), s(&mut fx, "/p2")]
        )
        .unwrap();
        assert_eq!(str_of(&fx, prop(&fx, loc, "href")), "http://example.com/p2");
        assert_eq!(prop(&fx, hist, "length"), Value::Int(2));
        let st = prop(&fx, hist, "state");
        assert_eq!(prop(&fx, st, "a"), Value::Int(1));
        assert_eq!(str_of(&fx, prop(&fx, st, "b")), "x");
        // back() restores the seeded entry: URL and null state.
        call!(fx, hist, "back", []).unwrap();
        assert_eq!(str_of(&fx, prop(&fx, loc, "href")), "http://example.com/");
        assert_eq!(prop(&fx, hist, "state"), Value::Null);
        // forward() restores the pushed entry's state.
        call!(fx, hist, "forward", []).unwrap();
        assert_eq!(str_of(&fx, prop(&fx, loc, "href")), "http://example.com/p2");
        let st2 = prop(&fx, hist, "state");
        assert_eq!(prop(&fx, st2, "a"), Value::Int(1));
    }

    #[test]
    fn history_push_state_url_rules() {
        let mut fx = setup("http://example.com/dir/");
        let hist = global_prop(&fx, "history");
        let loc = global_prop(&fx, "location");
        // No URL argument: new entry, same URL.
        call!(fx, hist, "pushState", [Value::Null, s(&mut fx, "")]).unwrap();
        assert_eq!(prop(&fx, hist, "length"), Value::Int(2));
        assert_eq!(
            str_of(&fx, prop(&fx, loc, "href")),
            "http://example.com/dir/"
        );
        // Explicit null URL also keeps the URL.
        call!(
            fx,
            hist,
            "pushState",
            [Value::Null, s(&mut fx, ""), Value::Null]
        )
        .unwrap();
        assert_eq!(prop(&fx, hist, "length"), Value::Int(3));
        // Cross-origin URLs are rejected and push nothing.
        let evil = s(&mut fx, "http://evil.com/");
        assert!(matches!(
            call!(fx, hist, "pushState", [Value::Null, s(&mut fx, ""), evil]),
            Err(NativeError::TypeError(_))
        ));
        assert_eq!(prop(&fx, hist, "length"), Value::Int(3));
        // Unparseable URLs are rejected too.
        let bad = s(&mut fx, "http://exa mple.com/");
        assert!(matches!(
            call!(fx, hist, "pushState", [Value::Null, s(&mut fx, ""), bad]),
            Err(NativeError::TypeError(_))
        ));
        assert_eq!(prop(&fx, hist, "length"), Value::Int(3));
        // replaceState enforces the same rule.
        let evil2 = s(&mut fx, "http://evil.com/");
        assert!(matches!(
            call!(
                fx,
                hist,
                "replaceState",
                [Value::Null, s(&mut fx, ""), evil2]
            ),
            Err(NativeError::TypeError(_))
        ));
        assert_eq!(prop(&fx, hist, "length"), Value::Int(3));
    }

    #[test]
    fn history_replace_state_replaces_current_entry() {
        let mut fx = setup("http://example.com/");
        let hist = global_prop(&fx, "history");
        let loc = global_prop(&fx, "location");
        call!(fx, loc, "assign", [s(&mut fx, "/a")]).unwrap();
        assert_eq!(prop(&fx, hist, "length"), Value::Int(2));
        let state = test_obj(&mut fx, &[("n", Value::Int(7))]);
        call!(
            fx,
            hist,
            "replaceState",
            [state, s(&mut fx, ""), s(&mut fx, "/b")]
        )
        .unwrap();
        // Same entry count, new URL and state.
        assert_eq!(prop(&fx, hist, "length"), Value::Int(2));
        assert_eq!(str_of(&fx, prop(&fx, loc, "href")), "http://example.com/b");
        assert_eq!(prop(&fx, prop(&fx, hist, "state"), "n"), Value::Int(7));
        // replaceState without a URL keeps the URL but swaps the state.
        call!(fx, hist, "replaceState", [Value::Int(42), s(&mut fx, "")]).unwrap();
        assert_eq!(str_of(&fx, prop(&fx, loc, "href")), "http://example.com/b");
        assert_eq!(prop(&fx, hist, "state"), Value::Int(42));
        assert_eq!(prop(&fx, hist, "length"), Value::Int(2));
    }

    #[test]
    fn history_state_json_edge_cases() {
        let mut fx = setup("http://example.com/");
        let hist = global_prop(&fx, "history");
        // Top-level undefined reads back as null (documented v1 divergence).
        call!(fx, hist, "pushState", [Value::Undefined, s(&mut fx, "")]).unwrap();
        assert_eq!(prop(&fx, hist, "state"), Value::Null);
        // undefined/function become null in arrays and are dropped from
        // objects; non-finite numbers become null; strings are escaped.
        let f = Value::Object(
            fx.heap
                .alloc_object(Object::new(ObjectKind::NativeFunction(dummy_native)))
                .unwrap(),
        );
        let arr = test_arr(&mut fx, &[Value::Undefined, f, Value::Int(1)]);
        let sval = s(&mut fx, "a\"b\\c");
        let state = test_obj(
            &mut fx,
            &[
                ("arr", arr),
                ("gone", Value::Undefined),
                ("inf", Value::Float(f64::INFINITY)),
                ("nan", Value::Float(f64::NAN)),
                ("s", sval),
            ],
        );
        call!(fx, hist, "pushState", [state, s(&mut fx, "")]).unwrap();
        let st = prop(&fx, hist, "state");
        let got_arr = prop(&fx, st, "arr");
        assert_eq!(prop(&fx, got_arr, "length"), Value::Int(3));
        assert_eq!(prop(&fx, got_arr, "0"), Value::Null);
        assert_eq!(prop(&fx, got_arr, "1"), Value::Null);
        assert_eq!(prop(&fx, got_arr, "2"), Value::Int(1));
        assert!(
            fx.heap
                .get_property(obj_of(st), Wtf8::from_str("gone"))
                .is_none()
        );
        assert_eq!(prop(&fx, st, "inf"), Value::Null);
        assert_eq!(prop(&fx, st, "nan"), Value::Null);
        assert_eq!(str_of(&fx, prop(&fx, st, "s")), "a\"b\\c");
        // Cyclic state throws and pushes no entry.
        let before = prop(&fx, hist, "length");
        let cyc = test_obj(&mut fx, &[]);
        raw_def(&mut fx.heap, obj_of(cyc), "self", cyc).unwrap();
        assert!(matches!(
            call!(fx, hist, "pushState", [cyc, s(&mut fx, "")]),
            Err(NativeError::TypeError(_))
        ));
        assert_eq!(prop(&fx, hist, "length"), before);
    }

    #[test]
    fn history_entries_are_capped() {
        let mut fx = setup("http://example.com/");
        let hist = global_prop(&fx, "history");
        for i in 0..60 {
            let url = format!("/{i}");
            call!(
                fx,
                hist,
                "pushState",
                [Value::Null, s(&mut fx, ""), s(&mut fx, &url)]
            )
            .unwrap();
        }
        assert_eq!(prop(&fx, hist, "length"), Value::Int(50));
        // The oldest surviving entry is /10: the seed plus /0../9 fell off.
        call!(fx, hist, "go", [Value::Int(-100)]).unwrap();
        let loc = global_prop(&fx, "location");
        assert_eq!(str_of(&fx, prop(&fx, loc, "href")), "http://example.com/10");
    }

    // -- Response.json / arrayBuffer ----------------------------------------

    /// A Response-shaped wrapper: the enricher only needs the hidden keys.
    fn response_shaped(fx: &mut Fx) -> Value {
        let obj = fx
            .heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let ctx_id = fx.ctx.0;
        raw_def(&mut fx.heap, obj, CTX_KEY, Value::Int(ctx_id as i32)).unwrap();
        raw_def(&mut fx.heap, obj, RESPONSE_KEY, Value::Int(7)).unwrap();
        Value::Object(obj)
    }

    /// Simulate the enricher's body snapshot, then attach one method.
    fn response_with_body(fx: &mut Fx, body: &str, method: &str, f: NativeFn) -> Value {
        let r = response_shaped(fx);
        let o = obj_of(r);
        let body_val = s(fx, body);
        raw_def(&mut fx.heap, o, BODY_KEY, body_val).unwrap();
        def_native_here(&mut fx.heap, o, method, f).unwrap();
        r
    }

    #[test]
    fn response_json_parses_values() {
        let mut fx = setup("http://example.com/");
        let r = response_with_body(
            &mut fx,
            "{\"a\":1,\"b\":[true,null,\"x\"],\"f\":1.5,\"n\":null,\"e\":1e3}",
            "json",
            n_resp_json,
        );
        let v = call!(fx, r, "json", []).unwrap();
        let Value::Object(o) = v else {
            panic!("json() did not return an object");
        };
        assert_eq!(prop(&fx, Value::Object(o), "a"), Value::Int(1));
        assert_eq!(prop(&fx, Value::Object(o), "f"), Value::Float(1.5));
        assert_eq!(prop(&fx, Value::Object(o), "n"), Value::Null);
        assert_eq!(prop(&fx, Value::Object(o), "e"), Value::Float(1000.0));
        let b = prop(&fx, Value::Object(o), "b");
        let Value::Object(bo) = b else {
            panic!("expected array");
        };
        assert_eq!(prop(&fx, Value::Object(bo), "length"), Value::Int(3));
        assert_eq!(prop(&fx, Value::Object(bo), "0"), Value::Bool(true));
        assert_eq!(prop(&fx, Value::Object(bo), "1"), Value::Null);
        assert_eq!(str_of(&fx, prop(&fx, Value::Object(bo), "2")), "x");
    }

    #[test]
    fn response_json_rejects_invalid() {
        let mut fx = setup("http://example.com/");
        for bad in ["{oops", "[1,]", "{\"a\":01}", "{\"a\":1} trailing", ""] {
            let r = response_with_body(&mut fx, bad, "json", n_resp_json);
            assert!(
                matches!(call!(fx, r, "json", []), Err(NativeError::SyntaxError(_))),
                "body {bad:?} should be a SyntaxError"
            );
        }
    }

    #[test]
    fn response_json_wrong_receiver() {
        let mut fx = setup("http://example.com/");
        let plain = fx
            .heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let r = response_with_body(&mut fx, "1", "json", n_resp_json);
        // json() called on a non-Response object is a TypeError.
        let f = prop(&fx, r, "json");
        let Value::Object(o) = f else {
            panic!("no json fn");
        };
        let ObjectKind::NativeFunction(nf) = object_kind(o) else {
            panic!("not native");
        };
        assert!(matches!(
            nf(&mut fx.heap, Value::Object(plain), &[]),
            Err(NativeError::TypeError(_))
        ));
    }

    #[test]
    fn response_array_buffer_returns_bytes() {
        let mut fx = setup("http://example.com/");
        let r = response_with_body(&mut fx, "AB", "arrayBuffer", n_resp_array_buffer);
        let v = call!(fx, r, "arrayBuffer", []).unwrap();
        let Value::Object(ao) = v else {
            panic!("arrayBuffer() did not return an object");
        };
        assert_eq!(prop(&fx, Value::Object(ao), "length"), Value::Int(2));
        assert_eq!(prop(&fx, Value::Object(ao), "0"), Value::Int(65));
        assert_eq!(prop(&fx, Value::Object(ao), "1"), Value::Int(66));
    }

    // -- Headers ------------------------------------------------------------

    #[test]
    fn headers_get_is_case_insensitive() {
        let mut fx = setup("http://example.com/");
        let h = new_headers_object(
            &mut fx.heap,
            fx.ctx.0,
            vec![
                ("Content-Type".to_string(), "application/json".to_string()),
                ("X-Custom".to_string(), "1".to_string()),
            ],
        )
        .unwrap();
        for name in [
            "content-type",
            "Content-Type",
            "CONTENT-TYPE",
            "cOnTeNt-TyPe",
        ] {
            let got = call!(fx, h, "get", [s(&mut fx, name)]).unwrap();
            assert_eq!(str_of(&fx, got), "application/json", "lookup {name}");
        }
        assert_eq!(
            call!(fx, h, "has", [s(&mut fx, "x-custom")]).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            call!(fx, h, "get", [s(&mut fx, "missing")]).unwrap(),
            Value::Null
        );
        assert_eq!(
            call!(fx, h, "has", [s(&mut fx, "missing")]).unwrap(),
            Value::Bool(false)
        );
    }

    // -- enricher -----------------------------------------------------------

    /// `text()` stand-in that serves a body stashed on a test-only key.
    fn fake_text(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
        let Value::Object(o) = this else {
            return Err(err_type("bad receiver"));
        };
        match heap
            .get_property(o, Wtf8::from_str("\u{0}test:body"))
            .map(|p| p.value)
        {
            Some(v @ Value::Str(_)) => Ok(v),
            _ => Err(err_type("no test body")),
        }
    }

    #[test]
    fn enricher_attaches_response_methods() {
        let mut fx = setup("http://example.com/");
        let r = response_shaped(&mut fx);
        let o = obj_of(r);
        let text_fn = native_object(&mut fx.heap, fake_text).unwrap();
        raw_def(&mut fx.heap, o, "text", text_fn).unwrap();
        let test_body = s(&mut fx, "[1,2]");
        raw_def(&mut fx.heap, o, "\u{0}test:body", test_body).unwrap();
        n_enrich_response(&mut fx.heap, Value::Undefined, &[r]).unwrap();
        // json() works off the snapshot taken through text().
        let v = call!(fx, r, "json", []).unwrap();
        let Value::Object(ao) = v else {
            panic!("expected array");
        };
        assert_eq!(prop(&fx, Value::Object(ao), "length"), Value::Int(2));
        assert_eq!(prop(&fx, Value::Object(ao), "0"), Value::Int(1));
        // arrayBuffer() exposes the snapshot bytes.
        let ab = call!(fx, r, "arrayBuffer", []).unwrap();
        let Value::Object(bo) = ab else {
            panic!("expected array");
        };
        assert_eq!(prop(&fx, Value::Object(bo), "length"), Value::Int(5));
        // headers is present (the fixture's handle is not registered in
        // the response table, so no headers resolve here; the end-to-end
        // test below covers the real-headers path).
        let h = prop(&fx, r, "headers");
        assert!(matches!(h, Value::Object(_)));
        assert_eq!(
            call!(fx, h, "get", [s(&mut fx, "content-type")]).unwrap(),
            Value::Null
        );
        // Non-Response objects are ignored.
        let plain = fx
            .heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        n_enrich_response(&mut fx.heap, Value::Undefined, &[Value::Object(plain)]).unwrap();
        assert!(
            fx.heap
                .get_property(plain, Wtf8::from_str("json"))
                .is_none()
        );
    }

    // -- JSON parser ----------------------------------------------------------

    #[test]
    fn json_parser_accepts_and_rejects() {
        assert!(parse_json("{\"a\":1}").is_ok());
        assert!(parse_json("[1,{\"b\":null},\"x\\u0041\",-2.5e-2]").is_ok());
        assert!(parse_json("  null  ").is_ok());
        for bad in [
            "",
            "{oops",
            "{\"a\":}",
            "{\"a\":01}",
            "[1,]",
            "{\"a\":1} trailing",
            "[1 2]",
            "\"\\uD83D\"",
            "\"bad \\x escape\"",
            "[",
            "{\"a\":1,}",
        ] {
            assert!(parse_json(bad).is_err(), "should reject {bad:?}");
        }
        // Depth limit: no stack overflow on hostile input.
        let deep = "[".repeat(200) + &"]".repeat(200);
        assert!(parse_json(&deep).is_err());
        // Surrogate pairs decode.
        match parse_json("\"\\uD83D\\uDE00\"").unwrap() {
            JsonVal::Str(st) => assert_eq!(st, "😀"),
            _ => panic!("expected string"),
        }
        // Integer vs float discrimination.
        match parse_json("42").unwrap() {
            JsonVal::Num(JsonNum::Int(42)) => {}
            _ => panic!("expected int"),
        }
        match parse_json("4.0").unwrap() {
            JsonVal::Num(JsonNum::Float(f)) => assert_eq!(f, 4.0),
            _ => panic!("expected float"),
        }
    }

    // -- end-to-end fetch wrapping --------------------------------------------

    #[test]
    fn fetch_wrapper_enriches_response_end_to_end() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        // Minimal loopback HTTP server serving one fixed JSON response.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let body = "{\"ok\":true}";
            let resp = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(resp.as_bytes());
        });

        let mut fx = setup("http://example.com/");
        let url = s(&mut fx, &format!("http://127.0.0.1:{port}/data"));
        // The installed global fetch is the wrapper.
        let global = Value::Object(fx.global);
        let promise = call!(fx, global, "fetch", [url]).unwrap();
        fn probe(_h: &mut Heap, _t: Value, _a: &[Value]) -> Result<Value, NativeError> {
            Ok(Value::Undefined)
        }
        let cb = Value::Object(
            fx.heap
                .alloc_object(Object::new(ObjectKind::NativeFunction(probe)))
                .unwrap(),
        );
        call!(fx, promise, "then", [cb]).unwrap();

        // Wait for the worker thread, then run the completion microtask.
        // The user callback fires only after the fetch settles.
        let mut done = false;
        for _ in 0..400 {
            poll_fetch_completions(&fx.apis);
            run_loop(&mut fx);
            if fx
                .invoker
                .borrow()
                .calls
                .borrow()
                .iter()
                .any(|(f, _)| *f == cb)
            {
                done = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(done, "fetch did not complete");

        let resp_val = {
            let calls = fx.invoker.borrow();
            let guard = calls.calls.borrow();
            let (_, args) = guard
                .iter()
                .find(|(f, _)| *f == cb)
                .expect("user callback ran");
            args[0]
        };
        // The enricher ran first: json()/arrayBuffer()/headers are present.
        let parsed = call!(fx, resp_val, "json", []).unwrap();
        let Value::Object(po) = parsed else {
            panic!("json() did not return an object");
        };
        assert_eq!(prop(&fx, Value::Object(po), "ok"), Value::Bool(true));
        assert!(matches!(prop(&fx, resp_val, "headers"), Value::Object(_)));
        // Real response headers flow through the enricher: the server
        // sent `content-type: application/json`.
        let headers = prop(&fx, resp_val, "headers");
        let ct = call!(fx, headers, "get", [s(&mut fx, "content-type")]).unwrap();
        assert_eq!(str_of(&fx, ct), "application/json");
        let ab = call!(fx, resp_val, "arrayBuffer", []).unwrap();
        let Value::Object(ao) = ab else {
            panic!("expected array");
        };
        assert_eq!(prop(&fx, Value::Object(ao), "length"), Value::Int(11));
        // The pre-existing surface is untouched.
        assert_eq!(prop(&fx, resp_val, "status"), Value::Int(200));
        assert_eq!(prop(&fx, resp_val, "ok"), Value::Bool(true));
    }
}
