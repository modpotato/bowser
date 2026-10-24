//! `URL` / `URLSearchParams`, `atob` / `btoa`, and `structuredClone`.
//!
//! Installs the URL/encoding globals on the engine's global object via
//! [`install_url`]. All state lives in ordinary heap objects behind hidden
//! `"\u{0}bowser:…"` properties, so no host-side tables or context ids are
//! needed: a URL wrapper keeps working as long as its heap does.
//!
//! # Conformance notes (v1 simplifications)
//!
//! * **No accessor properties.** [`Property`](crate::value::Property) is
//!   data-only, so the `URL` component getters (`href`, `protocol`, `host`,
//!   `hostname`, `port`, `pathname`, `search`, `hash`, `origin`) are
//!   non-writable, non-enumerable data properties, rewritten on every
//!   mutation (the same trick `Map.prototype.size` uses). The setters are
//!   exposed as prototype methods instead: `setHref`, `setProtocol`,
//!   `setHost`, `setHostname`, `setPort`, `setPathname`, `setSearch`,
//!   `setHash`. Plain assignment (`url.pathname = "/x"`) does **not**
//!   re-parse: it is silently ignored in sloppy mode and throws a
//!   `TypeError` in strict mode (the property is non-writable). When the
//!   VM learns accessors these become real getters/setters.
//! * **`new.target` is invisible without a VM.** `URL()` and `new URL()`
//!   behave identically (the spec throws for the call form). Instance
//!   prototypes chain to `URL.prototype` / `URLSearchParams.prototype`
//!   only when constructed through the VM, via
//!   [`active_new_target`](crate::vm::active_new_target) (the RegExp
//!   worker's precedent).
//! * **Document-base fallback.** `new URL(relative)` with no explicit base
//!   resolves against the document base URL baked into the `URL`
//!   constructor at install time (the spec's "API base URL"). Without a
//!   VM the fallback is unavailable and relative input throws.
//! * **`searchParams` is live-linked.** Each `URL` owns one
//!   `URLSearchParams` (identity-stable); mutating it updates the URL's
//!   query and vice versa, per spec. A standalone
//!   `new URLSearchParams(...)` has no link.
//! * **No iterator objects.** `keys()` / `values()` / `entries()` return
//!   fresh `Array`s (the `builtin_collections` precedent); `for`-`of` over
//!   them needs the VM's iterator protocol. `forEach` invokes callbacks
//!   through the shared [`call_function`](crate::builtin_object_array::call_function)
//!   choke point.
//! * **Constructor input forms.** `URLSearchParams` accepts a query
//!   string, a sequence of pairs (array-likes only, same restriction as
//!   `new Map(pairs)`), or a record object. There is no iterable protocol
//!   to consume yet.
//! * **`InvalidCharacterError` is a `TypeError`.** There is no
//!   `DOMException` hierarchy yet; `atob`/`btoa` failures surface as
//!   `TypeError` with the spec error name in the message.
//! * **`structuredClone` clones plain data.** Primitives (immutable
//!   strings/bigints are shared), ordinary objects, and arrays are
//!   deep-cloned with cycles preserved. Functions, symbols, regexps, and
//!   other exotic kinds throw `TypeError` ("could not be cloned"; the
//!   spec's `DataCloneError` needs `DOMException`). Host wrapper objects
//!   (DOM nodes, …) clone as plain objects, hidden properties included.
//!   The `transfer` option is ignored. A 1000-level nesting cap guards the
//!   native stack (`RangeError` past it).

use std::collections::HashMap;

use bowser_url::{Url, base64_decode, base64_encode, parse};

use crate::bindings::with_temp_root;
use crate::builtin_object_array::call_function;
use crate::gc::{AllocError, Gc, Heap};
use crate::value::{NativeError, NativeFn, Object, ObjectKind, Property, Value};
use crate::wtf8::Wtf8;

// ---------------------------------------------------------------------------
// Hidden property keys
// ---------------------------------------------------------------------------

/// Serialized href, on every `URL` instance.
const URL_KEY: &str = "\u{0}bowser:url";
/// Document base URL (serialized), baked onto the `URL` constructor by
/// [`install_url`] for the one-argument relative-URL form.
const URL_BASE_KEY: &str = "\u{0}bowser:url_base";
/// Query string without the leading `?`, on every `URLSearchParams`.
const PARAMS_KEY: &str = "\u{0}bowser:params";
/// Back-link to the owning `URL` object, or `undefined` for standalone
/// `URLSearchParams`. Enables the live `searchParams` link both ways.
const URL_LINK_KEY: &str = "\u{0}bowser:url_link";

// ---------------------------------------------------------------------------
// Installation
// ---------------------------------------------------------------------------

/// Install the URL/encoding globals: `URL`, `URLSearchParams`, `atob`,
/// `btoa`, and `structuredClone`.
///
/// `base_url` is the document base URL, used when `new URL(input)` is
/// called with a relative `input` and no explicit base (the spec's "API
/// base URL").
///
/// Called by [`install_web_apis`](crate::bindings::install_web_apis).
// Awaiting app page-VM integration: called from `bindings::install_web_apis`
// once the app instantiates a page VM.
#[allow(dead_code)]
pub fn install_url(
    heap: &mut Heap,
    global: Gc<Object>,
    base_url: &Url,
) -> Result<(), crate::bindings::ApiError> {
    // -- URL -----------------------------------------------------------
    // `url_ctor`/`url_proto` are unreachable from any root until they are
    // linked onto `global`; the `alloc_str`, per-method `def_native`
    // function objects, and fresh `alloc_object` calls in between can all
    // collect, so hold each as a temporary root across its build.
    let url_ctor = heap.alloc_object(Object::new(ObjectKind::NativeFunction(url_constructor)))?;
    // `URL` is a constructor (WebIDL `IsConstructor`): `is_constructor`
    // consults the install-time registry for `Reflect.construct` and
    // species paths.
    heap.mark_native_constructor(url_constructor);
    with_temp_root(heap, Value::Object(url_ctor), |heap| {
        let base_str = str_val(heap, &base_url.serialize())?;
        def(heap, url_ctor, URL_BASE_KEY, base_str)?;
        let url_proto = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
        with_temp_root(heap, Value::Object(url_proto), |heap| {
            for (name, f) in [
                ("toString", url_to_string as NativeFn),
                ("toJSON", url_to_string as NativeFn),
                ("setHref", url_set_href as NativeFn),
                ("setProtocol", url_set_protocol as NativeFn),
                ("setHost", url_set_host as NativeFn),
                ("setHostname", url_set_hostname as NativeFn),
                ("setPort", url_set_port as NativeFn),
                ("setPathname", url_set_pathname as NativeFn),
                ("setSearch", url_set_search as NativeFn),
                ("setHash", url_set_hash as NativeFn),
            ] {
                def_native(heap, url_proto, name, f)?;
            }
            def(heap, url_proto, "constructor", Value::Object(url_ctor))?;
            Ok::<(), crate::bindings::ApiError>(())
        })?;
        def(heap, url_ctor, "prototype", Value::Object(url_proto))?;
        def_native(heap, url_ctor, "canParse", url_can_parse)?;
        def(heap, global, "URL", Value::Object(url_ctor))?;
        Ok::<(), crate::bindings::ApiError>(())
    })?;

    // -- URLSearchParams -------------------------------------------------
    let usp_ctor =
        heap.alloc_object(Object::new(ObjectKind::NativeFunction(params_constructor)))?;
    // `URLSearchParams` is a constructor (WebIDL `IsConstructor`); see
    // the `URL` note above.
    heap.mark_native_constructor(params_constructor);
    with_temp_root(heap, Value::Object(usp_ctor), |heap| {
        let usp_proto = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
        with_temp_root(heap, Value::Object(usp_proto), |heap| {
            for (name, f) in [
                ("append", params_append as NativeFn),
                ("delete", params_delete as NativeFn),
                ("get", params_get as NativeFn),
                ("getAll", params_get_all as NativeFn),
                ("has", params_has as NativeFn),
                ("set", params_set as NativeFn),
                ("sort", params_sort as NativeFn),
                ("toString", params_to_string as NativeFn),
                ("forEach", params_for_each as NativeFn),
                ("keys", params_keys as NativeFn),
                ("values", params_values as NativeFn),
                ("entries", params_entries as NativeFn),
            ] {
                def_native(heap, usp_proto, name, f)?;
            }
            def(heap, usp_proto, "constructor", Value::Object(usp_ctor))?;
            Ok::<(), crate::bindings::ApiError>(())
        })?;
        def(heap, usp_ctor, "prototype", Value::Object(usp_proto))?;
        def(heap, global, "URLSearchParams", Value::Object(usp_ctor))?;
        Ok::<(), crate::bindings::ApiError>(())
    })?;

    // -- encoding globals ------------------------------------------------
    def_native(heap, global, "atob", global_atob)?;
    def_native(heap, global, "btoa", global_btoa)?;
    def_native(heap, global, "structuredClone", global_structured_clone)?;
    Ok(())
}

fn def(heap: &mut Heap, obj: Gc<Object>, name: &str, val: Value) -> Result<(), AllocError> {
    heap.define_property_str(obj, name, Property::data(val))
}

fn def_native(heap: &mut Heap, obj: Gc<Object>, name: &str, f: NativeFn) -> Result<(), AllocError> {
    let v = Value::Object(heap.alloc_object(Object::new(ObjectKind::NativeFunction(f)))?);
    def(heap, obj, name, v)
}

fn str_val(heap: &mut Heap, s: &str) -> Result<Value, AllocError> {
    heap.alloc_str(s).map(Value::Str)
}

fn type_err(msg: impl Into<String>) -> NativeError {
    NativeError::TypeError(msg.into())
}

/// JS `ToString` for the primitive cases the URL bindings need. Objects,
/// symbols, and bigints are rejected (a real `toString` call is the VM's
/// job), matching the `bindings.rs` convention.
fn to_js_string(_heap: &Heap, v: Value) -> Result<String, NativeError> {
    match v {
        // URL inputs are `&str` (url crate): lossy at the boundary.
        Value::Str(s) => Ok(s.as_wtf8().to_string()),
        Value::Int(i) => Ok(i.to_string()),
        Value::Float(f) => Ok(js_num_to_string(f)),
        Value::Bool(b) => Ok(b.to_string()),
        Value::Null => Ok("null".to_string()),
        Value::Undefined => Ok("undefined".to_string()),
        _ => Err(type_err("value cannot be converted to a string")),
    }
}

// v1 number formatting; the exact `Number::toString` (exponents, `-0`) is
// the builtins worker's domain.
fn js_num_to_string(f: f64) -> String {
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

/// A synced, non-writable, non-enumerable data property: the v1 stand-in
/// for an accessor (the `Map.prototype.size` precedent).
fn set_synced(heap: &mut Heap, obj: Gc<Object>, name: &str, val: &str) -> Result<(), NativeError> {
    let v = str_val(heap, val)?;
    heap.define_property_str(
        obj,
        name,
        Property {
            value: v,
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// URL
// ---------------------------------------------------------------------------

fn url_constructor(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let input = match args.first() {
        Some(v) => to_js_string(heap, *v)?,
        None => return Err(type_err("URL: at least 1 argument required")),
    };
    let base: Option<Url> = match args.get(1) {
        Some(v) if !v.is_undefined() && !v.is_null() => {
            let s = to_js_string(heap, *v)?;
            Some(parse(&s, None).map_err(|e| type_err(format!("URL: invalid base URL: {e}")))?)
        }
        // One-argument form: resolve against the document base baked into
        // the constructor (the spec's "API base URL").
        _ => install_base(heap),
    };
    let url =
        parse(&input, base.as_ref()).map_err(|e| type_err(format!("URL: invalid URL: {e}")))?;
    make_url(heap, &url)
}

/// The document base URL baked onto the `URL` constructor by
/// [`install_url`], reached through `new.target` when constructing via
/// the VM. `None` on direct native calls (no VM on the stack).
fn install_base(heap: &Heap) -> Option<Url> {
    let Value::Object(ctor) = crate::vm::active_new_target()? else {
        return None;
    };
    let p = heap.get_property(ctor, Wtf8::from_str(URL_BASE_KEY))?;
    let Value::Str(s) = p.value else { return None };
    // Ill-formed input cannot reach the url crate: it fails to parse.
    parse(s.as_wtf8().as_utf8().unwrap_or(""), None).ok()
}

/// Build a `URL` instance: hidden serialized href, the linked
/// `searchParams` object, and synced component properties. Chains to
/// `URL.prototype` when `new.target` is visible (the RegExp precedent).
fn make_url(heap: &mut Heap, url: &Url) -> Result<Value, NativeError> {
    let obj = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    if let Some(Value::Object(ctor)) = crate::vm::active_new_target()
        && let Some(p) = heap.get_property(ctor, Wtf8::from_str("prototype"))
        && matches!(p.value, Value::Object(_))
    {
        heap.set_prototype(obj, p.value);
    }
    let params = make_params(heap, url.query().unwrap_or_default(), Value::Object(obj))?;
    heap.define_property_str(
        obj,
        "searchParams",
        Property {
            value: Value::Object(params),
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )?;
    sync_url(heap, obj, url)?;
    Ok(Value::Object(obj))
}

/// Rewrite the hidden href and every synced component property from
/// `url`, and refresh the linked `searchParams`.
fn sync_url(heap: &mut Heap, obj: Gc<Object>, url: &Url) -> Result<(), NativeError> {
    let href = url.serialize();
    let href_val = str_val(heap, &href)?;
    def(heap, obj, URL_KEY, href_val)?;
    set_synced(heap, obj, "href", &href)?;
    set_synced(heap, obj, "protocol", &format!("{}:", url.scheme()))?;
    let host = url.host().unwrap_or_default();
    let mut hostport = host.to_string();
    if let Some(p) = url.port() {
        hostport.push(':');
        hostport.push_str(&p.to_string());
    }
    set_synced(heap, obj, "host", &hostport)?;
    set_synced(heap, obj, "hostname", host)?;
    set_synced(
        heap,
        obj,
        "port",
        &url.port().map(|p| p.to_string()).unwrap_or_default(),
    )?;
    set_synced(heap, obj, "pathname", url.path())?;
    set_synced(
        heap,
        obj,
        "search",
        &url.query().map(|q| format!("?{q}")).unwrap_or_default(),
    )?;
    set_synced(
        heap,
        obj,
        "hash",
        &url.fragment().map(|f| format!("#{f}")).unwrap_or_default(),
    )?;
    set_synced(heap, obj, "origin", &origin_string(url))?;
    if let Some(p) = heap.get_property(obj, Wtf8::from_str("searchParams"))
        && let Value::Object(po) = p.value
    {
        set_params_query(heap, po, url.query().unwrap_or_default())?;
    }
    Ok(())
}

/// Serialized origin for a `URL` object: `bowser_url` is the shared
/// home for this (both this crate and `bowser_app` depend on it), so
/// the default-port table lives there, not in a local copy.
fn origin_string(url: &Url) -> String {
    url.origin_serialized()
}

/// The [`Url`] behind a `URL` receiver, plus the receiver itself.
fn this_url(heap: &Heap, this: Value, op: &str) -> Result<(Gc<Object>, Url), NativeError> {
    let Value::Object(o) = this else {
        return Err(type_err(format!("{op}: invalid receiver")));
    };
    let href = match heap.get_property(o, Wtf8::from_str(URL_KEY)) {
        Some(p) => match p.value {
            // Only successful serializations are stored, so this is
            // well-formed; the gate is defensive.
            Value::Str(s) => s.as_wtf8().as_utf8().unwrap_or("").to_owned(),
            _ => return Err(type_err(format!("{op}: invalid receiver"))),
        },
        None => return Err(type_err(format!("{op}: invalid receiver"))),
    };
    // Only ever stores successful serializations, so this cannot fail;
    // the error is a defensive internal invariant.
    let url =
        parse(&href, None).map_err(|_| type_err(format!("{op}: internal URL state is corrupt")))?;
    Ok((o, url))
}

fn url_to_string(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let (_, url) = this_url(heap, this, "toString")?;
    Ok(str_val(heap, &url.serialize())?)
}

fn url_can_parse(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let input = match args.first() {
        Some(v) => to_js_string(heap, *v)?,
        None => return Err(type_err("URL.canParse: at least 1 argument required")),
    };
    let base: Option<Url> = match args.get(1) {
        Some(v) if !v.is_undefined() && !v.is_null() => {
            let s = to_js_string(heap, *v)?;
            match parse(&s, None) {
                Ok(b) => Some(b),
                Err(_) => return Ok(Value::Bool(false)),
            }
        }
        _ => install_base(heap),
    };
    Ok(Value::Bool(parse(&input, base.as_ref()).is_ok()))
}

fn url_set_href(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, mut url) = this_url(heap, this, "setHref")?;
    let v = to_js_string(heap, args.first().copied().unwrap_or(Value::Undefined))?;
    // The spec's `href` setter throws on parse failure.
    url.set_href(&v)
        .map_err(|e| type_err(format!("setHref: invalid URL: {e}")))?;
    sync_url(heap, obj, &url)?;
    Ok(Value::Undefined)
}

fn url_set_protocol(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, mut url) = this_url(heap, this, "setProtocol")?;
    let v = to_js_string(heap, args.first().copied().unwrap_or(Value::Undefined))?;
    // The spec's `protocol` setter terminates silently when the new scheme
    // cannot apply; `bowser_url` documents the same contract.
    if url.set_protocol(&v).is_ok() {
        sync_url(heap, obj, &url)?;
    }
    Ok(Value::Undefined)
}

fn url_set_host(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, mut url) = this_url(heap, this, "setHost")?;
    let v = to_js_string(heap, args.first().copied().unwrap_or(Value::Undefined))?;
    url.set_host(&v);
    sync_url(heap, obj, &url)?;
    Ok(Value::Undefined)
}

fn url_set_hostname(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, mut url) = this_url(heap, this, "setHostname")?;
    let v = to_js_string(heap, args.first().copied().unwrap_or(Value::Undefined))?;
    url.set_hostname(&v);
    sync_url(heap, obj, &url)?;
    Ok(Value::Undefined)
}

fn url_set_port(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, mut url) = this_url(heap, this, "setPort")?;
    let v = to_js_string(heap, args.first().copied().unwrap_or(Value::Undefined))?;
    url.set_port(&v);
    sync_url(heap, obj, &url)?;
    Ok(Value::Undefined)
}

fn url_set_pathname(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, mut url) = this_url(heap, this, "setPathname")?;
    let v = to_js_string(heap, args.first().copied().unwrap_or(Value::Undefined))?;
    url.set_pathname(&v);
    sync_url(heap, obj, &url)?;
    Ok(Value::Undefined)
}

fn url_set_search(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, mut url) = this_url(heap, this, "setSearch")?;
    let v = to_js_string(heap, args.first().copied().unwrap_or(Value::Undefined))?;
    url.set_search(&v);
    sync_url(heap, obj, &url)?;
    Ok(Value::Undefined)
}

fn url_set_hash(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, mut url) = this_url(heap, this, "setHash")?;
    let v = to_js_string(heap, args.first().copied().unwrap_or(Value::Undefined))?;
    url.set_hash(&v);
    sync_url(heap, obj, &url)?;
    Ok(Value::Undefined)
}

// ---------------------------------------------------------------------------
// URLSearchParams
// ---------------------------------------------------------------------------

/// application/x-www-form-urlencoded parsing: split on `&` (empty
/// sequences skipped), split each on the first `=`, `+` decodes to space,
/// then percent-decode and UTF-8 decode lossy.
fn parse_pairs(query: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for seq in query.split('&') {
        if seq.is_empty() {
            continue;
        }
        let (n, v) = match seq.find('=') {
            Some(i) => (&seq[..i], &seq[i + 1..]),
            None => (seq, ""),
        };
        out.push((urldecode(n), urldecode(v)));
    }
    out
}

fn urldecode(s: &str) -> String {
    let plus = s.replace('+', " ");
    let bytes = bowser_url::percent_decode(&plus);
    String::from_utf8_lossy(&bytes).into_owned()
}

/// application/x-www-form-urlencoded serialization: alphanumerics and
/// `* - . _` survive, space becomes `+`, everything else is uppercase
/// `%XX` over the UTF-8 bytes.
fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for &b in s.as_bytes() {
        match b {
            b'0'..=b'9' | b'A'..=b'Z' | b'a'..=b'z' | b'*' | b'-' | b'.' | b'_' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => {
                out.push('%');
                out.push_str(&format!("{b:02X}"));
            }
        }
    }
    out
}

fn serialize_pairs(pairs: &[(String, String)]) -> String {
    pairs
        .iter()
        .map(|(k, v)| format!("{}={}", urlencode(k), urlencode(v)))
        .collect::<Vec<_>>()
        .join("&")
}

/// Build a `URLSearchParams` holding `query`, optionally back-linked to
/// its owning `URL` object.
fn make_params(heap: &mut Heap, query: &str, url_link: Value) -> Result<Gc<Object>, NativeError> {
    let obj = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    if let Some(Value::Object(ctor)) = crate::vm::active_new_target()
        && let Some(p) = heap.get_property(ctor, Wtf8::from_str("prototype"))
        && matches!(p.value, Value::Object(_))
    {
        heap.set_prototype(obj, p.value);
    }
    let query_val = str_val(heap, query)?;
    def(heap, obj, PARAMS_KEY, query_val)?;
    def(heap, obj, URL_LINK_KEY, url_link)?;
    set_size(heap, obj, parse_pairs(query).len())?;
    Ok(obj)
}

/// Overwrite the query and synced `size` without touching the URL link.
fn set_params_query(heap: &mut Heap, obj: Gc<Object>, query: &str) -> Result<(), NativeError> {
    let query_val = str_val(heap, query)?;
    def(heap, obj, PARAMS_KEY, query_val)?;
    set_size(heap, obj, parse_pairs(query).len())?;
    Ok(())
}

fn set_size(heap: &mut Heap, obj: Gc<Object>, n: usize) -> Result<(), NativeError> {
    heap.define_property_str(
        obj,
        "size",
        Property {
            value: Value::Int(n.try_into().unwrap_or(i32::MAX)),
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )?;
    Ok(())
}

/// After a mutating params method: re-serialize, refresh `size`, and push
/// the new query into the linked `URL` (the live `searchParams` link).
fn params_updated(
    heap: &mut Heap,
    obj: Gc<Object>,
    pairs: &[(String, String)],
) -> Result<(), NativeError> {
    let query = serialize_pairs(pairs);
    set_params_query(heap, obj, &query)?;
    if let Some(p) = heap.get_property(obj, Wtf8::from_str(URL_LINK_KEY))
        && let Value::Object(uo) = p.value
        && let Some(hp) = heap.get_property(uo, Wtf8::from_str(URL_KEY))
        && let Value::Str(hs) = hp.value
    {
        // `href` feeds the url crate (`&str`): gate ill-formed input.
        let href = hs.as_wtf8().as_utf8().unwrap_or("").to_owned();
        if let Ok(mut url) = parse(&href, None) {
            url.set_search(&query);
            // Re-syncs the URL's components and the params' own query
            // (same value); never recurses back here.
            sync_url(heap, uo, &url)?;
        }
    }
    Ok(())
}

/// The query string behind a `URLSearchParams` receiver.
fn this_params(heap: &Heap, this: Value, op: &str) -> Result<(Gc<Object>, String), NativeError> {
    let Value::Object(o) = this else {
        return Err(type_err(format!("{op}: invalid receiver")));
    };
    match heap.get_property(o, Wtf8::from_str(PARAMS_KEY)) {
        Some(p) => match p.value {
            // Query strings feed the url crate: lossy at the boundary.
            Value::Str(s) => Ok((o, s.as_wtf8().to_string())),
            _ => Err(type_err(format!("{op}: invalid receiver"))),
        },
        None => Err(type_err(format!("{op}: invalid receiver"))),
    }
}

fn params_constructor(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let init = args.first().copied().unwrap_or(Value::Undefined);
    let query = params_init(heap, init)?;
    Ok(Value::Object(make_params(heap, &query, Value::Undefined)?))
}

/// Normalize constructor `init`: query string (leading `?` stripped), a
/// copy of another `URLSearchParams`, a sequence of pairs (array-likes
/// only), or a record of own enumerable string-keyed properties.
fn params_init(heap: &Heap, init: Value) -> Result<String, NativeError> {
    match init {
        Value::Undefined | Value::Null => Ok(String::new()),
        Value::Object(o) => {
            // Copy from another URLSearchParams (avoids re-encoding).
            if let Some(p) = heap.get_property(o, Wtf8::from_str(PARAMS_KEY))
                && let Value::Str(s) = p.value
            {
                return Ok(s.as_wtf8().to_string());
            }
            if heap.get_property(o, Wtf8::from_str("length")).is_some() {
                return init_sequence(heap, o);
            }
            init_record(heap, o)
        }
        _ => {
            let s = to_js_string(heap, init)?;
            let s = s.strip_prefix('?').unwrap_or(&s);
            Ok(serialize_pairs(&parse_pairs(s)))
        }
    }
}

/// Sequence form: an array-like of array-like `[name, value]` pairs.
fn init_sequence(heap: &Heap, o: Gc<Object>) -> Result<String, NativeError> {
    let n = match heap.get_property(o, Wtf8::from_str("length")) {
        Some(p) => match p.value {
            Value::Int(i) => i.max(0) as usize,
            Value::Float(f) if f.is_finite() && f >= 0.0 => f as usize,
            _ => return Err(type_err("URLSearchParams: sequence length is not a number")),
        },
        None => return Err(type_err("URLSearchParams: sequence length is not a number")),
    };
    let mut pairs = Vec::with_capacity(n);
    for i in 0..n {
        let entry = match heap.get_property(o, Wtf8::from_str(&i.to_string())) {
            Some(p) => p.value,
            None => {
                return Err(type_err(format!(
                    "URLSearchParams: sequence entry {i} is missing"
                )));
            }
        };
        let Value::Object(eo) = entry else {
            return Err(type_err(format!(
                "URLSearchParams: sequence entry {i} is not a pair"
            )));
        };
        let name = match heap.get_property(eo, Wtf8::from_str("0")) {
            Some(p) => to_js_string(heap, p.value)?,
            None => {
                return Err(type_err(format!(
                    "URLSearchParams: sequence entry {i} has no name"
                )));
            }
        };
        let value = match heap.get_property(eo, Wtf8::from_str("1")) {
            Some(p) => to_js_string(heap, p.value)?,
            None => {
                return Err(type_err(format!(
                    "URLSearchParams: sequence entry {i} has no value"
                )));
            }
        };
        pairs.push((name, value));
    }
    Ok(serialize_pairs(&pairs))
}

/// Record form: own enumerable string-keyed properties, skipping hidden
/// `"\u{0}…"` keys.
fn init_record(heap: &Heap, o: Gc<Object>) -> Result<String, NativeError> {
    let mut pairs = Vec::new();
    for (k, prop) in heap.get(o).properties.clone() {
        let name = k.as_wtf8();
        if name.starts_with(Wtf8::from_str("\0")) || !prop.enumerable {
            continue;
        }
        // Query strings feed the url crate (`&str`): lossy at the boundary.
        pairs.push((name.to_string(), to_js_string(heap, prop.value)?));
    }
    Ok(serialize_pairs(&pairs))
}

fn required_string(
    heap: &Heap,
    args: &[Value],
    idx: usize,
    op: &str,
) -> Result<String, NativeError> {
    match args.get(idx) {
        Some(v) => to_js_string(heap, *v),
        None => Err(type_err(format!("{op}: argument {idx} is required"))),
    }
}

fn params_append(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, query) = this_params(heap, this, "append")?;
    let name = required_string(heap, args, 0, "append")?;
    let value = required_string(heap, args, 1, "append")?;
    let mut pairs = parse_pairs(&query);
    pairs.push((name, value));
    params_updated(heap, obj, &pairs)?;
    Ok(Value::Undefined)
}

fn params_delete(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, query) = this_params(heap, this, "delete")?;
    let name = required_string(heap, args, 0, "delete")?;
    let value = match args.get(1) {
        Some(v) if !v.is_undefined() => Some(to_js_string(heap, *v)?),
        _ => None,
    };
    let mut pairs = parse_pairs(&query);
    pairs.retain(|(k, v)| !(k == &name && value.as_ref().is_none_or(|x| x == v)));
    params_updated(heap, obj, &pairs)?;
    Ok(Value::Undefined)
}

fn params_get(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (_, query) = this_params(heap, this, "get")?;
    let name = required_string(heap, args, 0, "get")?;
    for (k, v) in parse_pairs(&query) {
        if k == name {
            return Ok(str_val(heap, &v)?);
        }
    }
    Ok(Value::Null)
}

fn params_get_all(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (_, query) = this_params(heap, this, "getAll")?;
    let name = required_string(heap, args, 0, "getAll")?;
    let mut items = Vec::new();
    for (k, v) in parse_pairs(&query) {
        if k == name {
            items.push(str_val(heap, &v)?);
        }
    }
    str_array(heap, &items)
}

fn params_has(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (_, query) = this_params(heap, this, "has")?;
    let name = required_string(heap, args, 0, "has")?;
    let value = match args.get(1) {
        Some(v) if !v.is_undefined() => Some(to_js_string(heap, *v)?),
        _ => None,
    };
    for (k, v) in parse_pairs(&query) {
        if k == name && value.as_ref().is_none_or(|x| x == &v) {
            return Ok(Value::Bool(true));
        }
    }
    Ok(Value::Bool(false))
}

fn params_set(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, query) = this_params(heap, this, "set")?;
    let name = required_string(heap, args, 0, "set")?;
    let value = required_string(heap, args, 1, "set")?;
    let mut pairs = parse_pairs(&query);
    pairs.retain(|(k, _)| k != &name);
    pairs.push((name, value));
    params_updated(heap, obj, &pairs)?;
    Ok(Value::Undefined)
}

fn params_sort(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let (obj, query) = this_params(heap, this, "sort")?;
    let mut pairs = parse_pairs(&query);
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    params_updated(heap, obj, &pairs)?;
    Ok(Value::Undefined)
}

fn params_to_string(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let (_, query) = this_params(heap, this, "toString")?;
    Ok(str_val(heap, &query)?)
}

fn params_for_each(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, query) = this_params(heap, this, "forEach")?;
    let cb = match args.first() {
        Some(v) => *v,
        None => return Err(type_err("forEach: 1 argument required")),
    };
    let this_arg = args.get(1).copied().unwrap_or(Value::Undefined);
    for (k, v) in parse_pairs(&query) {
        let kv = str_val(heap, &k)?;
        let vv = str_val(heap, &v)?;
        call_function(heap, cb, this_arg, &[vv, kv, Value::Object(obj)])?;
    }
    Ok(Value::Undefined)
}

/// An `Array`-kind object of the given items (null prototype, indexed
/// properties plus `length`), the v1 stand-in for iterator results.
fn str_array(heap: &mut Heap, items: &[Value]) -> Result<Value, NativeError> {
    let arr = heap.alloc_object(Object::new(ObjectKind::Array))?;
    for (i, v) in items.iter().enumerate() {
        def(heap, arr, &i.to_string(), *v)?;
    }
    def(heap, arr, "length", Value::Int(items.len() as i32))?;
    Ok(Value::Object(arr))
}

fn params_keys(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let (_, query) = this_params(heap, this, "keys")?;
    let mut items = Vec::new();
    for (k, _) in parse_pairs(&query) {
        items.push(str_val(heap, &k)?);
    }
    str_array(heap, &items)
}

fn params_values(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let (_, query) = this_params(heap, this, "values")?;
    let mut items = Vec::new();
    for (_, v) in parse_pairs(&query) {
        items.push(str_val(heap, &v)?);
    }
    str_array(heap, &items)
}

fn params_entries(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let (_, query) = this_params(heap, this, "entries")?;
    let mut items = Vec::new();
    for (k, v) in parse_pairs(&query) {
        let pair = [str_val(heap, &k)?, str_val(heap, &v)?];
        items.push(str_array(heap, &pair)?);
    }
    str_array(heap, &items)
}

// ---------------------------------------------------------------------------
// atob / btoa use the shared `bowser_url::{base64_decode, base64_encode}`
// (single implementation also used for `data:` URL decoding).

fn global_atob(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let s = to_js_string(heap, args.first().copied().unwrap_or(Value::Undefined))?;
    let bytes = base64_decode(&s)
        .map_err(|_| type_err("atob: InvalidCharacterError: the string is not valid base64"))?;
    // Binary string: each byte becomes the code point U+0000..U+00FF.
    let out: String = bytes.into_iter().map(|b| b as char).collect();
    Ok(str_val(heap, &out)?)
}

fn global_btoa(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let s = to_js_string(heap, args.first().copied().unwrap_or(Value::Undefined))?;
    if s.chars().any(|c| c as u32 > 0xFF) {
        return Err(type_err(
            "btoa: InvalidCharacterError: the string contains characters outside Latin-1",
        ));
    }
    let bytes: Vec<u8> = s.chars().map(|c| c as u8).collect();
    Ok(str_val(heap, &base64_encode(&bytes))?)
}

// ---------------------------------------------------------------------------
// structuredClone
// ---------------------------------------------------------------------------

/// Native-stack guard for pathological nesting (the spec has no limit;
///
/// real engines cap out too).
const CLONE_DEPTH_LIMIT: u32 = 1000;

fn global_structured_clone(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    // The `transfer` / options argument is accepted and ignored (v1).
    let v = args.first().copied().unwrap_or(Value::Undefined);
    let mut seen: HashMap<Gc<Object>, Gc<Object>> = HashMap::new();
    clone_value(heap, v, &mut seen, 0)
}

fn clone_value(
    heap: &mut Heap,
    v: Value,
    seen: &mut HashMap<Gc<Object>, Gc<Object>>,
    depth: u32,
) -> Result<Value, NativeError> {
    if depth > CLONE_DEPTH_LIMIT {
        return Err(NativeError::RangeError(
            "structuredClone: value is too deeply nested".into(),
        ));
    }
    match v {
        Value::Undefined | Value::Null | Value::Bool(_) | Value::Int(_) | Value::Float(_) => Ok(v),
        // Immutable payloads: sharing is observationally identical.
        Value::Str(_) | Value::BigInt(_) => Ok(v),
        Value::Symbol(_) => Err(type_err("structuredClone: Symbol could not be cloned")),
        Value::Object(o) => {
            if let Some(&c) = seen.get(&o) {
                return Ok(Value::Object(c));
            }
            let (kind, prototype, props) = {
                let obj = heap.get(o);
                (obj.kind, obj.prototype, obj.properties.clone())
            };
            match kind {
                ObjectKind::Ordinary | ObjectKind::Array => {
                    let c = heap.alloc_object(Object::new(kind))?;
                    heap.set_prototype(c, prototype);
                    // Register before recursing so cycles terminate.
                    seen.insert(o, c);
                    for (k, prop) in props {
                        let cv = clone_value(heap, prop.value, seen, depth + 1)?;
                        heap.define_property(c, k.as_wtf8(), Property { value: cv, ..prop })?;
                    }
                    Ok(Value::Object(c))
                }
                ObjectKind::ArrayBuffer => {
                    // Deep-copy the byte storage: copying the slice
                    // into a fresh `Box<[u8]>` is a genuine independent
                    // copy, so mutating the clone never affects the
                    // original. `byteLength` comes along
                    // as an ordinary cloned property with the correct
                    // value (it is non-writable/non-configurable at
                    // construction, so script cannot have tampered with
                    // it).
                    let src = heap.get(o);
                    let bytes = src.bytes().map(Box::from);
                    if bytes.is_none() {
                        // Detached buffer: `bytes == None` is the detached
                        // state, and StructuredSerialize throws
                        // DataCloneError for it (surfaced as TypeError;
                        // there is no DOMException yet).
                        return Err(type_err(
                            "structuredClone: cannot clone a detached ArrayBuffer",
                        ));
                    }
                    let is_shared = src.ab_shared();
                    let max_byte_length = src.ab_max_byte_length();
                    let mut new_obj = Object::new(ObjectKind::ArrayBuffer);
                    new_obj.set_bytes(bytes);
                    // HTML StructuredSerialize: SerializeSharedArrayBuffer
                    // records only [[ArrayBufferData]] and
                    // [[ArrayBufferByteLength]], and deserialization
                    // allocates a fresh *non-shared* ArrayBuffer. Set the
                    // flag explicitly rather than relying on `Object::new`'s
                    // default, so the un-sharing step is deliberate.
                    new_obj.set_ab_shared(false);
                    // Resizable-ness rides in the struct, not a property:
                    // without this the clone of a resizable buffer would
                    // silently come back fixed-size.
                    new_obj.set_ab_max_byte_length(max_byte_length);
                    let c = heap.alloc_object(new_obj)?;
                    // Deserializing a SharedArrayBuffer allocates with
                    // %ArrayBuffer% as newTarget (AllocateArrayBuffer), so
                    // the clone gets %ArrayBuffer.prototype% — never the
                    // source's SharedArrayBuffer.prototype. Non-shared
                    // sources keep their own prototype (subclass lineage
                    // rides in it). When the intrinsic is not installed
                    // (native-level tests), fall back to the source proto.
                    let clone_proto = if is_shared {
                        heap.arraybuffer_prototype
                            .map(Value::Object)
                            .unwrap_or(prototype)
                    } else {
                        prototype
                    };
                    heap.set_prototype(c, clone_proto);
                    // Register before recursing so cycles terminate.
                    seen.insert(o, c);
                    for (k, prop) in props {
                        let cv = clone_value(heap, prop.value, seen, depth + 1)?;
                        heap.define_property(c, k.as_wtf8(), Property { value: cv, ..prop })?;
                    }
                    Ok(Value::Object(c))
                }
                ObjectKind::TypedArrayView {
                    buffer,
                    byte_offset,
                    len_elems,
                    etype,
                } => {
                    // Clone the underlying buffer through `seen`, so views
                    // sharing one buffer stay sharing one cloned buffer.
                    // The tag needs the cloned buffer at construction, but
                    // cloning it first would let a buffer->view->buffer
                    // cycle clone this view twice; allocate with the
                    // original as a placeholder, register in `seen`, then
                    // patch the tag (write barrier included) once the real
                    // buffer is known. Length-tracking rides in the struct,
                    // not the tag: the patch below only rewrites the tag, so
                    // copy it at construction, or a length-tracking view
                    // would silently come back fixed-length.
                    let length_tracking = heap.get(o).view_length_tracking();
                    let mut view_obj = Object::new(ObjectKind::TypedArrayView {
                        buffer: o,
                        byte_offset,
                        len_elems,
                        etype,
                    });
                    view_obj.set_view_length_tracking(length_tracking);
                    let c = heap.alloc_object(view_obj)?;
                    heap.set_prototype(c, prototype);
                    seen.insert(o, c);
                    let new_buffer =
                        match clone_value(heap, Value::Object(buffer), seen, depth + 1)? {
                            Value::Object(b) => b,
                            _ => {
                                return Err(type_err(
                                    "structuredClone: cloned buffer was not an object",
                                ));
                            }
                        };
                    heap.set_object_kind(
                        c,
                        ObjectKind::TypedArrayView {
                            buffer: new_buffer,
                            byte_offset,
                            len_elems,
                            etype,
                        },
                    );
                    for (k, prop) in props {
                        let cv = clone_value(heap, prop.value, seen, depth + 1)?;
                        heap.define_property(c, k.as_wtf8(), Property { value: cv, ..prop })?;
                    }
                    Ok(Value::Object(c))
                }
                ObjectKind::DataView {
                    buffer,
                    byte_offset,
                    byte_len,
                } => {
                    // Same placeholder-then-patch discipline as typed
                    // arrays; the own `buffer` property is cloned below
                    // (it resolves to the cloned buffer through `seen`).
                    // Length-tracking is copied at construction for the
                    // same reason as in the typed-array arm above.
                    let length_tracking = heap.get(o).view_length_tracking();
                    let mut dv_obj = Object::new(ObjectKind::DataView {
                        buffer: o,
                        byte_offset,
                        byte_len,
                    });
                    dv_obj.set_view_length_tracking(length_tracking);
                    let c = heap.alloc_object(dv_obj)?;
                    heap.set_prototype(c, prototype);
                    seen.insert(o, c);
                    let new_buffer =
                        match clone_value(heap, Value::Object(buffer), seen, depth + 1)? {
                            Value::Object(b) => b,
                            _ => {
                                return Err(type_err(
                                    "structuredClone: cloned buffer was not an object",
                                ));
                            }
                        };
                    heap.set_object_kind(
                        c,
                        ObjectKind::DataView {
                            buffer: new_buffer,
                            byte_offset,
                            byte_len,
                        },
                    );
                    for (k, prop) in props {
                        let cv = clone_value(heap, prop.value, seen, depth + 1)?;
                        heap.define_property(c, k.as_wtf8(), Property { value: cv, ..prop })?;
                    }
                    Ok(Value::Object(c))
                }
                ObjectKind::Date { value_bits } => {
                    // The time value rides in the tag; a fresh object with
                    // the same bits is a complete clone.
                    let c = heap.alloc_object(Object::new(ObjectKind::Date { value_bits }))?;
                    heap.set_prototype(c, prototype);
                    seen.insert(o, c);
                    for (k, prop) in props {
                        let cv = clone_value(heap, prop.value, seen, depth + 1)?;
                        heap.define_property(c, k.as_wtf8(), Property { value: cv, ..prop })?;
                    }
                    Ok(Value::Object(c))
                }
                // Functions, regexps, and any future exotic kind: the
                // spec's DataCloneError, surfaced as TypeError (no
                // DOMException yet).
                _ => Err(type_err("structuredClone: this value could not be cloned")),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn setup() -> (Heap, Gc<Object>) {
        let mut heap = Heap::new();
        let global = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let base = parse("https://example.org/dir/page", None).unwrap();
        install_url(&mut heap, global, &base).unwrap();
        (heap, global)
    }

    fn native_fn(heap: &Heap, obj: Gc<Object>, name: &str) -> NativeFn {
        match heap.get_property(obj, Wtf8::from_str(name)).unwrap().value {
            Value::Object(o) => match heap.get(o).kind {
                ObjectKind::NativeFunction(f) => f,
                _ => panic!("{name} is not a native function"),
            },
            v => panic!("{name} missing, got {v:?}"),
        }
    }

    fn global_fn(heap: &Heap, global: Gc<Object>, name: &str) -> NativeFn {
        native_fn(heap, global, name)
    }

    /// The constructor's `prototype` object (methods live there).
    fn proto(heap: &Heap, global: Gc<Object>, ctor: &str) -> Gc<Object> {
        let c = match heap
            .get_property(global, Wtf8::from_str(ctor))
            .unwrap()
            .value
        {
            Value::Object(o) => o,
            v => panic!("{ctor} missing: {v:?}"),
        };
        match heap
            .get_property(c, Wtf8::from_str("prototype"))
            .unwrap()
            .value
        {
            Value::Object(o) => o,
            v => panic!("{ctor}.prototype missing: {v:?}"),
        }
    }

    fn arg(heap: &mut Heap, s: &str) -> Value {
        Value::Str(heap.alloc_str(s).unwrap())
    }

    fn js_str(_heap: &Heap, v: Value) -> String {
        match v {
            // Only successful serializations are stored, so this is
            // well-formed; the gate is defensive.
            Value::Str(s) => s.as_wtf8().as_utf8().unwrap_or("").to_owned(),
            _ => panic!("not a string: {v:?}"),
        }
    }

    fn prop(heap: &Heap, o: Gc<Object>, name: &str) -> Value {
        heap.get_property(o, Wtf8::from_str(name)).unwrap().value
    }

    fn prop_str(heap: &Heap, o: Gc<Object>, name: &str) -> String {
        js_str(heap, prop(heap, o, name))
    }

    fn new_url(heap: &mut Heap, global: Gc<Object>, args: &[Value]) -> Gc<Object> {
        let ctor = global_fn(heap, global, "URL");
        match ctor(heap, Value::Undefined, args).unwrap() {
            Value::Object(o) => o,
            v => panic!("URL() did not return an object: {v:?}"),
        }
    }

    fn new_params(heap: &mut Heap, global: Gc<Object>, args: &[Value]) -> Gc<Object> {
        let ctor = global_fn(heap, global, "URLSearchParams");
        match ctor(heap, Value::Undefined, args).unwrap() {
            Value::Object(o) => o,
            v => panic!("URLSearchParams() did not return an object: {v:?}"),
        }
    }

    fn call_method(
        heap: &mut Heap,
        proto_obj: Gc<Object>,
        name: &str,
        this: Value,
        args: &[Value],
    ) -> Value {
        native_fn(heap, proto_obj, name)(heap, this, args).unwrap()
    }

    fn arr_items(heap: &Heap, arr: Value) -> Vec<Value> {
        let Value::Object(o) = arr else {
            panic!("not an array: {arr:?}")
        };
        let len = match prop(heap, o, "length") {
            Value::Int(i) => i as usize,
            v => panic!("bad length: {v:?}"),
        };
        (0..len).map(|i| prop(heap, o, &i.to_string())).collect()
    }

    fn arr_strs(heap: &Heap, arr: Value) -> Vec<String> {
        arr_items(heap, arr)
            .into_iter()
            .map(|v| js_str(heap, v))
            .collect()
    }

    // -- URL ------------------------------------------------------------

    #[test]
    fn ctors_marked_as_native_constructors() {
        // `URL` and `URLSearchParams` are genuinely constructible, so
        // the install must register both in the install-time
        // constructor registry `is_constructor` consults for
        // `Reflect.construct` and species paths. (A native that is not
        // installed as a constructor, e.g. `URL.canParse`, is *not*
        // marked: it stays a `TypeError` under `Reflect.construct`.)
        let (heap, _global) = setup();
        assert!(heap.is_native_constructor(url_constructor));
        assert!(heap.is_native_constructor(params_constructor));
        // A callable-but-not-constructible native stays unmarked.
        assert!(!heap.is_native_constructor(url_can_parse));
    }

    #[test]
    fn url_installs_globals() {
        let (heap, global) = setup();
        for name in ["URL", "URLSearchParams", "atob", "btoa", "structuredClone"] {
            assert!(
                matches!(
                    heap.get_property(global, Wtf8::from_str(name))
                        .unwrap()
                        .value,
                    Value::Object(_)
                ),
                "{name} not installed"
            );
        }
        // Prototype methods are present.
        let p = proto(&heap, global, "URL");
        for name in [
            "toString",
            "toJSON",
            "setHref",
            "setProtocol",
            "setHost",
            "setHostname",
            "setPort",
            "setPathname",
            "setSearch",
            "setHash",
        ] {
            native_fn(&heap, p, name);
        }
        let sp = proto(&heap, global, "URLSearchParams");
        for name in [
            "append", "delete", "get", "getAll", "has", "set", "sort", "toString", "forEach",
            "keys", "values", "entries",
        ] {
            native_fn(&heap, sp, name);
        }
    }

    #[test]
    fn url_parse_serialize_round_trip() {
        let (mut heap, global) = setup();
        let input = arg(&mut heap, "https://user:pass@example.com:8080/a/b?x=1#frag");
        let u = new_url(&mut heap, global, &[input]);
        assert_eq!(
            prop_str(&heap, u, "href"),
            "https://user:pass@example.com:8080/a/b?x=1#frag"
        );
        assert_eq!(prop_str(&heap, u, "protocol"), "https:");
        assert_eq!(prop_str(&heap, u, "host"), "example.com:8080");
        assert_eq!(prop_str(&heap, u, "hostname"), "example.com");
        assert_eq!(prop_str(&heap, u, "port"), "8080");
        assert_eq!(prop_str(&heap, u, "pathname"), "/a/b");
        assert_eq!(prop_str(&heap, u, "search"), "?x=1");
        assert_eq!(prop_str(&heap, u, "hash"), "#frag");
        assert_eq!(prop_str(&heap, u, "origin"), "https://example.com:8080");
        let p = proto(&heap, global, "URL");
        let s = call_method(&mut heap, p, "toString", Value::Object(u), &[]);
        assert_eq!(
            js_str(&heap, s),
            "https://user:pass@example.com:8080/a/b?x=1#frag"
        );
        let j = call_method(&mut heap, p, "toJSON", Value::Object(u), &[]);
        assert_eq!(
            js_str(&heap, j),
            "https://user:pass@example.com:8080/a/b?x=1#frag"
        );
    }

    #[test]
    fn url_default_port_and_opaque_origin() {
        let (mut heap, global) = setup();
        let input = arg(&mut heap, "https://example.com/a");
        let u = new_url(&mut heap, global, &[input]);
        assert_eq!(prop_str(&heap, u, "port"), "");
        assert_eq!(prop_str(&heap, u, "host"), "example.com");
        assert_eq!(prop_str(&heap, u, "origin"), "https://example.com");

        let mailto = arg(&mut heap, "mailto:foo@bar.example");
        let m = new_url(&mut heap, global, &[mailto]);
        assert_eq!(prop_str(&heap, m, "protocol"), "mailto:");
        assert_eq!(prop_str(&heap, m, "pathname"), "foo@bar.example");
        assert_eq!(prop_str(&heap, m, "host"), "");
        assert_eq!(prop_str(&heap, m, "origin"), "null");
    }

    #[test]
    fn url_base_resolution_and_can_parse() {
        let (mut heap, global) = setup();
        let rel = arg(&mut heap, "../other?q=1#frag");
        let base = arg(&mut heap, "https://example.org/dir/page");
        let u = new_url(&mut heap, global, &[rel, base]);
        assert_eq!(
            prop_str(&heap, u, "href"),
            "https://example.org/other?q=1#frag"
        );

        // canParse is a static on the constructor.
        let ctor = match heap
            .get_property(global, Wtf8::from_str("URL"))
            .unwrap()
            .value
        {
            Value::Object(o) => o,
            _ => panic!(),
        };
        let cp = native_fn(&heap, ctor, "canParse");
        let ok_arg = arg(&mut heap, "https://example.com/");
        assert_eq!(
            cp(&mut heap, Value::Undefined, &[ok_arg]).unwrap(),
            Value::Bool(true)
        );
        let bad_arg = arg(&mut heap, "http://");
        assert_eq!(
            cp(&mut heap, Value::Undefined, &[bad_arg]).unwrap(),
            Value::Bool(false)
        );
        let rel_arg = arg(&mut heap, "path/only");
        let base_arg = arg(&mut heap, "https://example.com/dir/");
        assert_eq!(
            cp(&mut heap, Value::Undefined, &[rel_arg, base_arg]).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn url_constructor_rejects_garbage() {
        let (mut heap, global) = setup();
        let ctor = global_fn(&heap, global, "URL");
        // No arguments.
        assert!(matches!(
            ctor(&mut heap, Value::Undefined, &[]),
            Err(NativeError::TypeError(_))
        ));
        // Unparseable input.
        let bad = arg(&mut heap, "http://");
        assert!(matches!(
            ctor(&mut heap, Value::Undefined, &[bad]),
            Err(NativeError::TypeError(_))
        ));
        // Relative input with no base (no VM, so no document-base fallback).
        let rel = arg(&mut heap, "relative/path");
        assert!(matches!(
            ctor(&mut heap, Value::Undefined, &[rel]),
            Err(NativeError::TypeError(_))
        ));
    }

    #[test]
    fn url_setters_resync_components() {
        let (mut heap, global) = setup();
        let input = arg(&mut heap, "https://example.com/a?x=1#old");
        let u = new_url(&mut heap, global, &[input]);
        let this = Value::Object(u);
        let p = proto(&heap, global, "URL");

        let v = arg(&mut heap, "/x/y");
        call_method(&mut heap, p, "setPathname", this, &[v]);
        assert_eq!(prop_str(&heap, u, "pathname"), "/x/y");
        assert_eq!(
            prop_str(&heap, u, "href"),
            "https://example.com/x/y?x=1#old"
        );

        let v = arg(&mut heap, "a=2");
        call_method(&mut heap, p, "setSearch", this, &[v]);
        assert_eq!(prop_str(&heap, u, "search"), "?a=2");

        let v = arg(&mut heap, "top");
        call_method(&mut heap, p, "setHash", this, &[v]);
        assert_eq!(prop_str(&heap, u, "hash"), "#top");

        let v = arg(&mut heap, "3000");
        call_method(&mut heap, p, "setPort", this, &[v]);
        assert_eq!(prop_str(&heap, u, "port"), "3000");
        assert_eq!(prop_str(&heap, u, "host"), "example.com:3000");

        let v = arg(&mut heap, "example.net");
        call_method(&mut heap, p, "setHostname", this, &[v]);
        assert_eq!(prop_str(&heap, u, "hostname"), "example.net");
        assert_eq!(prop_str(&heap, u, "port"), "3000");

        let v = arg(&mut heap, "http:");
        call_method(&mut heap, p, "setProtocol", this, &[v]);
        assert_eq!(prop_str(&heap, u, "protocol"), "http:");
        assert_eq!(prop_str(&heap, u, "origin"), "http://example.net:3000");

        let v = arg(&mut heap, "https://other.org/final");
        call_method(&mut heap, p, "setHref", this, &[v]);
        assert_eq!(prop_str(&heap, u, "href"), "https://other.org/final");
        assert_eq!(prop_str(&heap, u, "pathname"), "/final");
        assert_eq!(prop_str(&heap, u, "search"), "");
        assert_eq!(prop_str(&heap, u, "hash"), "");

        // setHref with garbage throws and leaves the URL unchanged.
        let v = arg(&mut heap, "http://");
        let r = native_fn(&heap, p, "setHref")(&mut heap, this, &[v]);
        assert!(matches!(r, Err(NativeError::TypeError(_))));
        assert_eq!(prop_str(&heap, u, "href"), "https://other.org/final");
    }

    #[test]
    fn url_set_host_with_port() {
        let (mut heap, global) = setup();
        let input = arg(&mut heap, "https://example.com/");
        let u = new_url(&mut heap, global, &[input]);
        let this = Value::Object(u);
        let p = proto(&heap, global, "URL");
        let v = arg(&mut heap, "example.net:8443");
        call_method(&mut heap, p, "setHost", this, &[v]);
        assert_eq!(prop_str(&heap, u, "host"), "example.net:8443");
        assert_eq!(prop_str(&heap, u, "hostname"), "example.net");
        assert_eq!(prop_str(&heap, u, "port"), "8443");
        assert_eq!(prop_str(&heap, u, "href"), "https://example.net:8443/");
    }

    // -- URLSearchParams ------------------------------------------------

    #[test]
    fn params_get_set_iteration() {
        let (mut heap, global) = setup();
        let q = arg(&mut heap, "a=1&a=2&b=3");
        let ps = new_params(&mut heap, global, &[q]);
        let this = Value::Object(ps);
        let p = proto(&heap, global, "URLSearchParams");

        let n = arg(&mut heap, "a");
        let g = call_method(&mut heap, p, "get", this, &[n]);
        assert_eq!(js_str(&heap, g), "1");

        let all = call_method(&mut heap, p, "getAll", this, &[n]);
        assert_eq!(arr_strs(&heap, all), vec!["1", "2"]);

        let nb = arg(&mut heap, "b");
        let h = call_method(&mut heap, p, "has", this, &[nb]);
        assert_eq!(h, Value::Bool(true));
        let nz = arg(&mut heap, "zzz");
        let h = call_method(&mut heap, p, "has", this, &[nz]);
        assert_eq!(h, Value::Bool(false));
        let missing = call_method(&mut heap, p, "get", this, &[nz]);
        assert_eq!(missing, Value::Null);

        assert_eq!(prop(&heap, ps, "size"), Value::Int(3));

        let keys = call_method(&mut heap, p, "keys", this, &[]);
        assert_eq!(arr_strs(&heap, keys), vec!["a", "a", "b"]);
        let values = call_method(&mut heap, p, "values", this, &[]);
        assert_eq!(arr_strs(&heap, values), vec!["1", "2", "3"]);
        let entries = call_method(&mut heap, p, "entries", this, &[]);
        let entry_strs: Vec<Vec<String>> = arr_items(&heap, entries)
            .into_iter()
            .map(|e| arr_strs(&heap, e))
            .collect();
        assert_eq!(
            entry_strs,
            vec![vec!["a", "1"], vec!["a", "2"], vec!["b", "3"]]
        );

        let t = call_method(&mut heap, p, "toString", this, &[]);
        assert_eq!(js_str(&heap, t), "a=1&a=2&b=3");
    }

    #[test]
    fn params_mutations_and_encoding() {
        let (mut heap, global) = setup();
        let ps = new_params(&mut heap, global, &[]);
        let this = Value::Object(ps);
        let p = proto(&heap, global, "URLSearchParams");

        let k = arg(&mut heap, "q");
        let v = arg(&mut heap, "a b+c");
        call_method(&mut heap, p, "append", this, &[k, v]);
        let t = call_method(&mut heap, p, "toString", this, &[]);
        // Space -> `+`, literal `+` -> `%2B`.
        assert_eq!(js_str(&heap, t), "q=a+b%2Bc");
        // Decoding round-trips.
        let g = call_method(&mut heap, p, "get", this, &[k]);
        assert_eq!(js_str(&heap, g), "a b+c");

        // set replaces all values for the name.
        let v2 = arg(&mut heap, "z");
        call_method(&mut heap, p, "set", this, &[k, v2]);
        let all = call_method(&mut heap, p, "getAll", this, &[k]);
        assert_eq!(arr_strs(&heap, all), vec!["z"]);
        assert_eq!(prop(&heap, ps, "size"), Value::Int(1));

        // delete with a value only removes the matching pair.
        let k2 = arg(&mut heap, "d");
        let v3 = arg(&mut heap, "1");
        let v4 = arg(&mut heap, "2");
        call_method(&mut heap, p, "append", this, &[k2, v3]);
        call_method(&mut heap, p, "append", this, &[k2, v4]);
        call_method(&mut heap, p, "delete", this, &[k2, v3]);
        let all = call_method(&mut heap, p, "getAll", this, &[k2]);
        assert_eq!(arr_strs(&heap, all), vec!["2"]);
        // has with value.
        let h = call_method(&mut heap, p, "has", this, &[k2, v4]);
        assert_eq!(h, Value::Bool(true));
        let h = call_method(&mut heap, p, "has", this, &[k2, v3]);
        assert_eq!(h, Value::Bool(false));
        // delete without value removes the rest.
        call_method(&mut heap, p, "delete", this, &[k2]);
        let h = call_method(&mut heap, p, "has", this, &[k2]);
        assert_eq!(h, Value::Bool(false));

        // sort is stable by name.
        let a = arg(&mut heap, "b=2&a=1&b=1");
        let ps2 = new_params(&mut heap, global, &[a]);
        let this2 = Value::Object(ps2);
        call_method(&mut heap, p, "sort", this2, &[]);
        let t = call_method(&mut heap, p, "toString", this2, &[]);
        assert_eq!(js_str(&heap, t), "a=1&b=2&b=1");
    }

    #[test]
    fn params_init_forms() {
        let (mut heap, global) = setup();
        let p = proto(&heap, global, "URLSearchParams");

        // Leading `?` is stripped; empty segments skipped.
        let q = arg(&mut heap, "?x=1&&y=2");
        let ps = new_params(&mut heap, global, &[q]);
        let t = call_method(&mut heap, p, "toString", Value::Object(ps), &[]);
        assert_eq!(js_str(&heap, t), "x=1&y=2");

        // Sequence of pairs (array-likes).
        let seq = heap.alloc_object(Object::new(ObjectKind::Array)).unwrap();
        for (i, (k, v)) in [("n1", "v1"), ("n2", "v2")].iter().enumerate() {
            let pair = heap.alloc_object(Object::new(ObjectKind::Array)).unwrap();
            let k0 = arg(&mut heap, k);
            let v0 = arg(&mut heap, v);
            heap.define_property_str(pair, "0", Property::data(k0))
                .unwrap();
            heap.define_property_str(pair, "1", Property::data(v0))
                .unwrap();
            heap.define_property_str(pair, "length", Property::data(Value::Int(2)))
                .unwrap();
            heap.define_property_str(seq, &i.to_string(), Property::data(Value::Object(pair)))
                .unwrap();
        }
        heap.define_property_str(seq, "length", Property::data(Value::Int(2)))
            .unwrap();
        let ps = new_params(&mut heap, global, &[Value::Object(seq)]);
        let t = call_method(&mut heap, p, "toString", Value::Object(ps), &[]);
        assert_eq!(js_str(&heap, t), "n1=v1&n2=v2");

        // Record form.
        let rec = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let rv = arg(&mut heap, "rv");
        heap.define_property_str(rec, "rk", Property::data(rv))
            .unwrap();
        let ps = new_params(&mut heap, global, &[Value::Object(rec)]);
        let t = call_method(&mut heap, p, "toString", Value::Object(ps), &[]);
        assert_eq!(js_str(&heap, t), "rk=rv");

        // Copy from another URLSearchParams.
        let c3 = arg(&mut heap, "c=3");
        let src = new_params(&mut heap, global, &[c3]);
        let ps = new_params(&mut heap, global, &[Value::Object(src)]);
        let t = call_method(&mut heap, p, "toString", Value::Object(ps), &[]);
        assert_eq!(js_str(&heap, t), "c=3");
    }

    thread_local! {
        static SEEN: RefCell<Vec<(String, String)>> = const { RefCell::new(Vec::new()) };
    }

    fn record_kv(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
        // forEach passes (value, name, params).
        let v = to_js_string(heap, args[0])?;
        let k = to_js_string(heap, args[1])?;
        SEEN.with(|s| s.borrow_mut().push((k, v)));
        Ok(Value::Undefined)
    }

    #[test]
    fn params_for_each_invokes_callback() {
        let (mut heap, global) = setup();
        let ab = arg(&mut heap, "a=1&b=2");
        let ps = new_params(&mut heap, global, &[ab]);
        let this = Value::Object(ps);
        let p = proto(&heap, global, "URLSearchParams");
        SEEN.with(|s| s.borrow_mut().clear());
        let cb = Value::Object(
            heap.alloc_object(Object::new(ObjectKind::NativeFunction(record_kv)))
                .unwrap(),
        );
        call_method(&mut heap, p, "forEach", this, &[cb]);
        let seen = SEEN.with(|s| s.borrow().clone());
        assert_eq!(
            seen,
            vec![
                ("a".to_string(), "1".to_string()),
                ("b".to_string(), "2".to_string())
            ]
        );
    }

    #[test]
    fn url_searchparams_live_link() {
        let (mut heap, global) = setup();
        let input = arg(&mut heap, "https://example.com/?x=1");
        let u = new_url(&mut heap, global, &[input]);
        let this = Value::Object(u);
        let up = proto(&heap, global, "URL");
        let pp = proto(&heap, global, "URLSearchParams");

        let ps = match prop(&heap, u, "searchParams") {
            Value::Object(o) => o,
            v => panic!("no searchParams: {v:?}"),
        };
        // Params -> URL direction.
        let k = arg(&mut heap, "y");
        let v = arg(&mut heap, "2");
        call_method(&mut heap, pp, "append", Value::Object(ps), &[k, v]);
        assert_eq!(prop_str(&heap, u, "search"), "?x=1&y=2");
        assert_eq!(prop_str(&heap, u, "href"), "https://example.com/?x=1&y=2");
        // URL -> params direction.
        let q = arg(&mut heap, "z=9");
        call_method(&mut heap, up, "setSearch", this, &[q]);
        let gk = arg(&mut heap, "z");
        let g = call_method(&mut heap, pp, "get", Value::Object(ps), &[gk]);
        assert_eq!(js_str(&heap, g), "9");
        assert_eq!(prop(&heap, ps, "size"), Value::Int(1));
        // Identity is stable across URL mutations.
        let ps2 = match prop(&heap, u, "searchParams") {
            Value::Object(o) => o,
            v => panic!("no searchParams: {v:?}"),
        };
        assert_eq!(ps, ps2);
    }

    // -- atob / btoa ----------------------------------------------------

    #[test]
    fn btoa_vectors() {
        let (mut heap, global) = setup();
        let btoa = global_fn(&heap, global, "btoa");
        for (input, expected) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
            ("hello", "aGVsbG8="),
        ] {
            let a = arg(&mut heap, input);
            let r = btoa(&mut heap, Value::Undefined, &[a]).unwrap();
            assert_eq!(js_str(&heap, r), expected, "btoa({input:?})");
        }
        // Latin-1 byte 0xFF.
        let a = arg(&mut heap, "\u{ff}");
        let r = btoa(&mut heap, Value::Undefined, &[a]).unwrap();
        assert_eq!(js_str(&heap, r), "/w==");
        // Outside Latin-1 -> TypeError.
        let a = arg(&mut heap, "€");
        assert!(matches!(
            btoa(&mut heap, Value::Undefined, &[a]),
            Err(NativeError::TypeError(_))
        ));
    }

    #[test]
    fn atob_vectors() {
        let (mut heap, global) = setup();
        let atob = global_fn(&heap, global, "atob");
        for (input, expected) in [
            ("", ""),
            ("Zg==", "f"),
            ("Zm8=", "fo"),
            ("Zm9v", "foo"),
            ("Zm9vYg==", "foob"),
            ("Zm9vYmE=", "fooba"),
            ("Zm9vYmFy", "foobar"),
            ("aGVsbG8=", "hello"),
            ("/w==", "\u{ff}"),
            // Forgiving: ASCII whitespace is stripped.
            (" aG\tVs\nbG8=\r ", "hello"),
            // Missing padding is tolerated.
            ("aGVsbG8", "hello"),
        ] {
            let a = arg(&mut heap, input);
            let r = atob(&mut heap, Value::Undefined, &[a]).unwrap();
            assert_eq!(js_str(&heap, r), expected, "atob({input:?})");
        }
        // Failures.
        for bad in ["a", "!!!", "====", "aGVsbG8==="] {
            let a = arg(&mut heap, bad);
            assert!(
                matches!(
                    atob(&mut heap, Value::Undefined, &[a]),
                    Err(NativeError::TypeError(_))
                ),
                "atob({bad:?}) should fail"
            );
        }
    }

    #[test]
    fn atob_btoa_round_trip() {
        let (mut heap, global) = setup();
        let atob = global_fn(&heap, global, "atob");
        let btoa = global_fn(&heap, global, "btoa");
        let original: String = (0u32..256).filter_map(char::from_u32).collect();
        let a = arg(&mut heap, &original);
        let enc = btoa(&mut heap, Value::Undefined, &[a]).unwrap();
        let dec = atob(&mut heap, Value::Undefined, &[enc]).unwrap();
        assert_eq!(js_str(&heap, dec), original);
    }

    // -- structuredClone ------------------------------------------------

    fn obj(heap: &mut Heap) -> Gc<Object> {
        heap.alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap()
    }

    #[test]
    fn structured_clone_deep_equal_and_independent() {
        let (mut heap, global) = setup();
        let clone = global_fn(&heap, global, "structuredClone");

        // {a: [1, {b: "x"}], n: 5}
        let root = obj(&mut heap);
        let arr = heap.alloc_object(Object::new(ObjectKind::Array)).unwrap();
        let inner = obj(&mut heap);
        let xb = arg(&mut heap, "x");
        heap.define_property_str(inner, "b", Property::data(xb))
            .unwrap();
        heap.define_property_str(arr, "0", Property::data(Value::Int(1)))
            .unwrap();
        heap.define_property_str(arr, "1", Property::data(Value::Object(inner)))
            .unwrap();
        heap.define_property_str(arr, "length", Property::data(Value::Int(2)))
            .unwrap();
        heap.define_property_str(root, "a", Property::data(Value::Object(arr)))
            .unwrap();
        heap.define_property_str(root, "n", Property::data(Value::Int(5)))
            .unwrap();

        let c = match clone(&mut heap, Value::Undefined, &[Value::Object(root)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        assert_ne!(c, root, "clone must be a new object");
        assert_eq!(prop(&heap, c, "n"), Value::Int(5));
        let ca = match prop(&heap, c, "a") {
            Value::Object(o) => o,
            v => panic!("not an array: {v:?}"),
        };
        assert_ne!(ca, arr);
        assert_eq!(prop(&heap, ca, "length"), Value::Int(2));
        assert_eq!(prop(&heap, ca, "0"), Value::Int(1));
        let ci = match prop(&heap, ca, "1") {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        assert_ne!(ci, inner);
        assert_eq!(prop_str(&heap, ci, "b"), "x");
        // Array kind is preserved.
        assert_eq!(heap.get(ca).kind, ObjectKind::Array);

        // Independence: mutate the clone, the original is untouched.
        let changed = arg(&mut heap, "CHANGED");
        heap.define_property_str(ci, "b", Property::data(changed))
            .unwrap();
        assert_eq!(prop_str(&heap, inner, "b"), "x");
        heap.define_property_str(ca, "0", Property::data(Value::Int(99)))
            .unwrap();
        assert_eq!(prop(&heap, arr, "0"), Value::Int(1));
    }

    #[test]
    fn structured_clone_preserves_cycles() {
        let (mut heap, global) = setup();
        let clone = global_fn(&heap, global, "structuredClone");
        let o = obj(&mut heap);
        heap.define_property_str(o, "self", Property::data(Value::Object(o)))
            .unwrap();
        let c = match clone(&mut heap, Value::Undefined, &[Value::Object(o)]).unwrap() {
            Value::Object(x) => x,
            v => panic!("not an object: {v:?}"),
        };
        assert_ne!(c, o);
        assert_eq!(prop(&heap, c, "self"), Value::Object(c));
    }

    #[test]
    fn structured_clone_primitives_and_rejections() {
        let (mut heap, global) = setup();
        let clone = global_fn(&heap, global, "structuredClone");
        for v in [
            Value::Undefined,
            Value::Null,
            Value::Bool(true),
            Value::Int(42),
            Value::Float(1.5),
        ] {
            assert_eq!(clone(&mut heap, Value::Undefined, &[v]).unwrap(), v);
        }
        let s = arg(&mut heap, "shared");
        assert_eq!(clone(&mut heap, Value::Undefined, &[s]).unwrap(), s);

        // Symbols cannot be cloned.
        let sym = Value::Symbol(heap.alloc_symbol(Some(Wtf8::from_str("s"))).unwrap());
        assert!(matches!(
            clone(&mut heap, Value::Undefined, &[sym]),
            Err(NativeError::TypeError(_))
        ));
        // Functions cannot be cloned.
        let f = Value::Object(
            heap.alloc_object(Object::new(ObjectKind::NativeFunction(record_kv)))
                .unwrap(),
        );
        assert!(matches!(
            clone(&mut heap, Value::Undefined, &[f]),
            Err(NativeError::TypeError(_))
        ));
        // A function nested inside an object also fails.
        let o = obj(&mut heap);
        heap.define_property_str(o, "f", Property::data(f)).unwrap();
        assert!(matches!(
            clone(&mut heap, Value::Undefined, &[Value::Object(o)]),
            Err(NativeError::TypeError(_))
        ));
    }

    fn array_buffer(heap: &mut Heap, bytes: &[u8]) -> Gc<Object> {
        let mut b = Object::new(ObjectKind::ArrayBuffer);
        b.set_bytes(Some(bytes.to_vec().into_boxed_slice()));
        let buf = heap.alloc_object(b).unwrap();
        heap.define_property_str(
            buf,
            "byteLength",
            Property {
                value: Value::Int(bytes.len() as i32),
                writable: false,
                enumerable: false,
                configurable: false,
            },
        )
        .unwrap();
        buf
    }

    fn view_buffer(heap: &Heap, v: Gc<Object>) -> Gc<Object> {
        match heap.get(v).kind {
            ObjectKind::TypedArrayView { buffer, .. } => buffer,
            ObjectKind::DataView { buffer, .. } => buffer,
            k => panic!("not a view: {k:?}"),
        }
    }

    #[test]
    fn structured_clone_array_buffer_deep_copies() {
        let (mut heap, global) = setup();
        let clone = global_fn(&heap, global, "structuredClone");
        let buf = array_buffer(&mut heap, &[10, 20, 30, 40]);

        let c = match clone(&mut heap, Value::Undefined, &[Value::Object(buf)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        assert_ne!(c, buf, "clone must be a new object");
        assert_eq!(heap.get(c).kind, ObjectKind::ArrayBuffer);
        assert_eq!(prop(&heap, c, "byteLength"), Value::Int(4));
        let orig = heap.get(buf).bytes().unwrap();
        let dupe = heap.get(c).bytes().unwrap();
        assert_eq!(dupe, &[10u8, 20, 30, 40]);
        // Genuinely independent storage: equal contents, different
        // allocations, so mutating one side cannot affect the other.
        assert_ne!(orig.as_ptr(), dupe.as_ptr());
    }

    #[test]
    fn structured_clone_typed_array_view_rebuilds_over_cloned_buffer() {
        let (mut heap, global) = setup();
        let clone = global_fn(&heap, global, "structuredClone");
        let buf = array_buffer(&mut heap, &[1, 2, 3, 4, 5, 6]);
        let view = heap
            .alloc_object(Object::new(ObjectKind::TypedArrayView {
                buffer: buf,
                byte_offset: 2,
                len_elems: 2,
                etype: crate::value::TypedArrayKind::Uint8,
            }))
            .unwrap();
        heap.define_property_str(view, "length", Property::data(Value::Int(2)))
            .unwrap();

        let c = match clone(&mut heap, Value::Undefined, &[Value::Object(view)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        assert_ne!(c, view);
        match heap.get(c).kind {
            ObjectKind::TypedArrayView {
                byte_offset,
                len_elems,
                etype,
                ..
            } => {
                assert_eq!(byte_offset, 2);
                assert_eq!(len_elems, 2);
                assert_eq!(etype, crate::value::TypedArrayKind::Uint8);
            }
            k => panic!("not a typed array view: {k:?}"),
        }
        let c_buf = view_buffer(&heap, c);
        assert_ne!(c_buf, buf, "view must point at the cloned buffer");
        assert_eq!(heap.get(c_buf).bytes().unwrap(), &[1u8, 2, 3, 4, 5, 6]);
        assert_ne!(
            heap.get(buf).bytes().unwrap().as_ptr(),
            heap.get(c_buf).bytes().unwrap().as_ptr(),
            "buffer storage must be an independent copy"
        );
        // Own properties (length) are cloned onto the new view.
        assert_eq!(prop(&heap, c, "length"), Value::Int(2));
    }

    #[test]
    fn structured_clone_data_view_rebuilds_over_cloned_buffer() {
        let (mut heap, global) = setup();
        let clone = global_fn(&heap, global, "structuredClone");
        let buf = array_buffer(&mut heap, &[9, 8, 7, 6]);
        let dv = heap
            .alloc_object(Object::new(ObjectKind::DataView {
                buffer: buf,
                byte_offset: 1,
                byte_len: 2,
            }))
            .unwrap();

        let c = match clone(&mut heap, Value::Undefined, &[Value::Object(dv)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        assert_ne!(c, dv);
        match heap.get(c).kind {
            ObjectKind::DataView {
                byte_offset,
                byte_len,
                ..
            } => {
                assert_eq!(byte_offset, 1);
                assert_eq!(byte_len, 2);
            }
            k => panic!("not a data view: {k:?}"),
        }
        let c_buf = view_buffer(&heap, c);
        assert_ne!(c_buf, buf);
        assert_eq!(heap.get(c_buf).bytes().unwrap(), &[9u8, 8, 7, 6]);
        assert_ne!(
            heap.get(buf).bytes().unwrap().as_ptr(),
            heap.get(c_buf).bytes().unwrap().as_ptr(),
            "buffer storage must be an independent copy"
        );
    }

    #[test]
    fn structured_clone_date_copies_time_value() {
        let (mut heap, global) = setup();
        let clone = global_fn(&heap, global, "structuredClone");
        let bits = 1_704_067_200_000f64.to_bits();
        let d = heap
            .alloc_object(Object::new(ObjectKind::Date { value_bits: bits }))
            .unwrap();

        let c = match clone(&mut heap, Value::Undefined, &[Value::Object(d)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        assert_ne!(c, d, "clone must be a new object");
        assert_eq!(heap.get(c).kind, ObjectKind::Date { value_bits: bits });
    }

    #[test]
    fn structured_clone_views_sharing_a_buffer_share_the_clone() {
        let (mut heap, global) = setup();
        let clone = global_fn(&heap, global, "structuredClone");
        let buf = array_buffer(&mut heap, &[1, 2, 3, 4]);
        let mk_view = |heap: &mut Heap, offset: u32| {
            heap.alloc_object(Object::new(ObjectKind::TypedArrayView {
                buffer: buf,
                byte_offset: offset,
                len_elems: 2,
                etype: crate::value::TypedArrayKind::Uint8,
            }))
            .unwrap()
        };
        let v1 = mk_view(&mut heap, 0);
        let v2 = mk_view(&mut heap, 2);
        let root = obj(&mut heap);
        heap.define_property_str(root, "a", Property::data(Value::Object(v1)))
            .unwrap();
        heap.define_property_str(root, "b", Property::data(Value::Object(v2)))
            .unwrap();

        let c = match clone(&mut heap, Value::Undefined, &[Value::Object(root)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        let ca = match prop(&heap, c, "a") {
            Value::Object(o) => o,
            v => panic!("not a view: {v:?}"),
        };
        let cb = match prop(&heap, c, "b") {
            Value::Object(o) => o,
            v => panic!("not a view: {v:?}"),
        };
        assert_ne!(ca, v1);
        assert_ne!(cb, v2);
        let ba = view_buffer(&heap, ca);
        let bb = view_buffer(&heap, cb);
        // Per spec, one cloned buffer backs both cloned views.
        assert_eq!(ba, bb);
        assert_ne!(ba, buf);
    }

    #[test]
    fn structured_clone_view_buffer_cycle_terminates() {
        let (mut heap, global) = setup();
        let clone = global_fn(&heap, global, "structuredClone");
        let buf = array_buffer(&mut heap, &[7, 8]);
        let view = heap
            .alloc_object(Object::new(ObjectKind::TypedArrayView {
                buffer: buf,
                byte_offset: 0,
                len_elems: 2,
                etype: crate::value::TypedArrayKind::Uint8,
            }))
            .unwrap();
        // Buffer -> view -> buffer cycle through an own property.
        heap.define_property_str(buf, "view", Property::data(Value::Object(view)))
            .unwrap();

        let c = match clone(&mut heap, Value::Undefined, &[Value::Object(view)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        let c_buf = view_buffer(&heap, c);
        assert_ne!(c_buf, buf);
        // The cycle resolves to the single cloned view, not a duplicate.
        assert_eq!(prop(&heap, c_buf, "view"), Value::Object(c));
    }

    fn resizable_buffer(heap: &mut Heap, bytes: &[u8], max: u32) -> Gc<Object> {
        let mut b = Object::new(ObjectKind::ArrayBuffer);
        b.set_bytes(Some(bytes.to_vec().into_boxed_slice()));
        b.set_ab_max_byte_length(Some(max));
        let buf = heap.alloc_object(b).unwrap();
        heap.define_property_str(
            buf,
            "byteLength",
            Property {
                value: Value::Int(bytes.len() as i32),
                writable: false,
                enumerable: false,
                configurable: false,
            },
        )
        .unwrap();
        buf
    }

    #[test]
    fn structured_clone_resizable_buffer_preserves_max() {
        let (mut heap, global) = setup();
        let clone = global_fn(&heap, global, "structuredClone");
        let buf = resizable_buffer(&mut heap, &[1, 2, 3, 4], 8);

        let c = match clone(&mut heap, Value::Undefined, &[Value::Object(buf)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        assert_ne!(c, buf, "clone must be a new object");
        assert_eq!(heap.get(c).kind, ObjectKind::ArrayBuffer);
        assert_eq!(
            heap.get(c).ab_max_byte_length(),
            Some(8),
            "clone of a resizable buffer must stay resizable with the same ceiling"
        );
        assert_eq!(heap.get(c).bytes().unwrap(), &[1u8, 2, 3, 4]);

        // And a fixed-size buffer clones back fixed-size (no phantom max).
        let fixed = array_buffer(&mut heap, &[5, 6]);
        let cf = match clone(&mut heap, Value::Undefined, &[Value::Object(fixed)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        assert_eq!(heap.get(cf).ab_max_byte_length(), None);
    }

    #[test]
    fn structured_clone_detached_buffer_throws() {
        let (mut heap, global) = setup();
        let clone = global_fn(&heap, global, "structuredClone");
        // `bytes == None` on an ArrayBuffer tag is the detached state.
        let buf = heap
            .alloc_object(Object::new(ObjectKind::ArrayBuffer))
            .unwrap();

        assert!(
            matches!(
                clone(&mut heap, Value::Undefined, &[Value::Object(buf)]),
                Err(NativeError::TypeError(_))
            ),
            "cloning a detached buffer must throw DataCloneError"
        );

        // A view over a detached buffer fails the same way: the buffer
        // clone inside the view arm hits the detached check.
        let view = heap
            .alloc_object(Object::new(ObjectKind::TypedArrayView {
                buffer: buf,
                byte_offset: 0,
                len_elems: 0,
                etype: crate::value::TypedArrayKind::Uint8,
            }))
            .unwrap();
        assert!(
            matches!(
                clone(&mut heap, Value::Undefined, &[Value::Object(view)]),
                Err(NativeError::TypeError(_))
            ),
            "cloning a view over a detached buffer must throw"
        );
    }

    #[test]
    fn structured_clone_views_preserve_length_tracking() {
        let (mut heap, global) = setup();
        let clone = global_fn(&heap, global, "structuredClone");
        // Length-tracking views: omitted length over a resizable buffer.
        let buf = resizable_buffer(&mut heap, &[1, 2, 3, 4], 8);
        let mut ta = Object::new(ObjectKind::TypedArrayView {
            buffer: buf,
            byte_offset: 0,
            len_elems: 4,
            etype: crate::value::TypedArrayKind::Uint8,
        });
        ta.set_view_length_tracking(true);
        let view = heap.alloc_object(ta).unwrap();
        let mut dv = Object::new(ObjectKind::DataView {
            buffer: buf,
            byte_offset: 0,
            byte_len: 4,
        });
        dv.set_view_length_tracking(true);
        let dataview = heap.alloc_object(dv).unwrap();

        let c1 = match clone(&mut heap, Value::Undefined, &[Value::Object(view)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        assert!(
            heap.get(c1).view_length_tracking(),
            "typed-array clone must stay length-tracking"
        );
        // The cloned buffer backing it stays resizable as well.
        let c1_buf = view_buffer(&heap, c1);
        assert_ne!(c1_buf, buf);
        assert_eq!(heap.get(c1_buf).ab_max_byte_length(), Some(8));

        let c2 = match clone(&mut heap, Value::Undefined, &[Value::Object(dataview)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        assert!(
            heap.get(c2).view_length_tracking(),
            "DataView clone must stay length-tracking"
        );
        let c2_buf = view_buffer(&heap, c2);
        assert_ne!(c2_buf, buf);
        assert_eq!(heap.get(c2_buf).ab_max_byte_length(), Some(8));

        // A fixed-length view over a fixed buffer clones back fixed.
        let fixed = array_buffer(&mut heap, &[9, 8, 7, 6]);
        let fv = heap
            .alloc_object(Object::new(ObjectKind::TypedArrayView {
                buffer: fixed,
                byte_offset: 0,
                len_elems: 4,
                etype: crate::value::TypedArrayKind::Uint8,
            }))
            .unwrap();
        let cf = match clone(&mut heap, Value::Undefined, &[Value::Object(fv)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        assert!(!heap.get(cf).view_length_tracking());
    }

    /// Build a SharedArrayBuffer the way `sharedarraybuffer_constructor`
    /// does: an ArrayBuffer-tagged object flagged shared, proto-linked to
    /// the given SAB prototype, with written bytes and a frozen byteLength.
    fn shared_array_buffer(heap: &mut Heap, bytes: &[u8], sab_proto: Gc<Object>) -> Gc<Object> {
        let mut b = Object::new(ObjectKind::ArrayBuffer);
        b.set_bytes(Some(bytes.to_vec().into_boxed_slice()));
        let buf = heap.alloc_object(b).unwrap();
        heap.set_prototype(buf, Value::Object(sab_proto));
        heap.mark_shared_array_buffer(buf);
        heap.define_property_str(
            buf,
            "byteLength",
            Property {
                value: Value::Int(bytes.len() as i32),
                writable: false,
                enumerable: false,
                configurable: false,
            },
        )
        .unwrap();
        buf
    }

    #[test]
    fn structured_clone_shared_buffer_becomes_plain_arraybuffer() {
        let (mut heap, global) = setup();
        let clone = global_fn(&heap, global, "structuredClone");
        // The full VM installs these intrinsics via builtin_reflect_proxy;
        // standalone URL tests link them by hand.
        let ab_proto = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let sab_proto = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.arraybuffer_prototype = Some(ab_proto);

        let sab = shared_array_buffer(&mut heap, &[1, 2, 3, 4, 5, 6, 7, 8], sab_proto);
        assert!(heap.get(sab).ab_shared(), "fixture must be a shared buffer");

        let c = match clone(&mut heap, Value::Undefined, &[Value::Object(sab)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        assert_ne!(c, sab, "clone must be a new object");
        assert_eq!(heap.get(c).kind, ObjectKind::ArrayBuffer);
        // Per HTML StructuredSerialize/Deserialize the SAB deserializes as
        // a fresh non-shared ArrayBuffer: %ArrayBuffer.prototype%, not
        // SharedArrayBuffer.prototype, and IsSharedArrayBuffer false.
        assert_eq!(
            heap.get(c).prototype,
            Value::Object(ab_proto),
            "clone of a SharedArrayBuffer must link to %ArrayBuffer.prototype%"
        );
        assert!(
            !heap.get(c).ab_shared(),
            "clone of a SharedArrayBuffer must not be shared"
        );
        // Bytes are copied, not aliased.
        let dupe = heap.get(c).bytes().unwrap();
        assert_eq!(dupe, &[1u8, 2, 3, 4, 5, 6, 7, 8]);
        assert_ne!(
            heap.get(sab).bytes().unwrap().as_ptr(),
            dupe.as_ptr(),
            "byte storage must be an independent copy"
        );
        assert_eq!(prop(&heap, c, "byteLength"), Value::Int(8));
        // The source is untouched: still shared, still SAB-proto-linked.
        assert!(heap.get(sab).ab_shared());
        assert_eq!(heap.get(sab).prototype, Value::Object(sab_proto));
    }

    #[test]
    fn structured_clone_nonshared_buffer_keeps_own_prototype() {
        let (mut heap, global) = setup();
        let clone = global_fn(&heap, global, "structuredClone");
        // A custom (e.g. subclass) prototype rides on the clone unchanged.
        let custom_proto = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let buf = array_buffer(&mut heap, &[9, 9]);
        heap.set_prototype(buf, Value::Object(custom_proto));

        let c = match clone(&mut heap, Value::Undefined, &[Value::Object(buf)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        assert_ne!(c, buf, "clone must be a new object");
        assert_eq!(
            heap.get(c).prototype,
            Value::Object(custom_proto),
            "non-shared buffer clone keeps the source prototype"
        );
        assert!(!heap.get(c).ab_shared());
        assert_eq!(heap.get(c).bytes().unwrap(), &[9u8, 9]);
    }

    #[test]
    fn structured_clone_view_over_shared_buffer_gets_plain_clone() {
        let (mut heap, global) = setup();
        let clone = global_fn(&heap, global, "structuredClone");
        let ab_proto = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let sab_proto = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.arraybuffer_prototype = Some(ab_proto);

        let sab = shared_array_buffer(&mut heap, &[1, 2, 3, 4], sab_proto);
        let view = heap
            .alloc_object(Object::new(ObjectKind::TypedArrayView {
                buffer: sab,
                byte_offset: 0,
                len_elems: 4,
                etype: crate::value::TypedArrayKind::Uint8,
            }))
            .unwrap();

        let c = match clone(&mut heap, Value::Undefined, &[Value::Object(view)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        // The view's buffer is cloned through `seen` onto a fresh,
        // non-shared ArrayBuffer: the hybrid never leaks into views.
        let cbuf = view_buffer(&heap, c);
        assert_ne!(cbuf, sab, "cloned view must not share the original buffer");
        assert_eq!(heap.get(cbuf).prototype, Value::Object(ab_proto));
        assert!(!heap.get(cbuf).ab_shared());
        assert_eq!(heap.get(cbuf).bytes().unwrap(), &[1u8, 2, 3, 4]);
    }
}
