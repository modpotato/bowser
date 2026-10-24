//! `Reflect`, `Proxy`, `ArrayBuffer`/`SharedArrayBuffer`, `DataView`, and
//! the typed array constructors.
//!
//! Installs everything on the global object via [`install`].
//!
//! ## Self-containment
//!
//! This module is deliberately self-contained: it copies the small helpers
//! it needs (numeric conversions, property helpers, native-function
//! builders) instead of reaching into the sibling builtin modules, which
//! are not wired into the crate yet. The helpers assume only `&mut Heap`
//! context, matching the sibling pattern.
//!
//! ## v1 gaps (documented, not silent)
//!
//! - **No VM re-entry.** Trap callbacks and `Reflect.apply`/`construct`
//!   targets can only be [`ObjectKind::NativeFunction`]; bytecode functions
//!   produce `TypeError`, exactly like the sibling `call_function` choke
//!   point. The single choke point here is [`call_value`].
//! - **No automatic interception.** Without VM support for proxy exotic
//!   objects, `proxy.foo` in bytecode does not consult traps. All proxy
//!   behavior is reachable through the `Reflect.*` entry points, which
//!   dispatch through the trap wrappers in this module.
//! - **No accessor descriptors.** `Property` is data-only, so
//!   `defineProperty` with `get`/`set` throws `TypeError`. Invariant checks
//!   that involve accessor properties are skipped as vacuous.
//! - **No symbols as property keys.** Property keys are strings; symbol
//!   keys throw on coercion.
//! - **Buffer backing store.** `ArrayBuffer` bytes live as ordinary
//!   non-enumerable indexed data properties on the buffer object. No packed
//!   backing store yet, so large buffers pay per-byte property overhead.
//!   Typed array indexed access from bytecode (`view[i]`) does not go
//!   through the element accessors until the VM learns integer-indexed
//!   exotic objects; use the `Reflect` entry points or the `pub(crate)`
//!   `typed_index_get`/`typed_index_set` helpers.
//! - **`SharedArrayBuffer` aliases `ArrayBuffer`.** Same implementation,
//!   separate constructor; no actual shared-memory semantics.
//! - **`Reflect.construct`** supports native and bytecode targets
//!   (bytecode re-enters the VM); the `newTarget` argument is ignored
//!   (v1).

use std::cmp::Ordering;

use crate::builtin_object_array::{
    PrimitiveHint, array_index_key, check_prototype_cycle, new_array_from_opt,
    to_primitive_with_hint, walk_proto_chain,
};
use crate::builtin_string_number::{bn_parse, get_method, with_temp_root};
use crate::gc::{ArrayDefineDesc, ElemKey, Gc, Heap, PropKey, WellKnown};
use crate::value::{
    AccessorProp, BigInt, NativeError, NativeFn, Object, ObjectKind, Property, Symbol,
    TypedArrayKind, Value,
};
use crate::wtf8::{Wtf8, Wtf8Buf};

// ---------------------------------------------------------------------------
// Small helpers (local copies of the sibling-builtin pattern)
// ---------------------------------------------------------------------------

/// Argument `i`, or `undefined` when the caller passed fewer.
fn get_arg(args: &[Value], i: usize) -> Value {
    args.get(i).copied().unwrap_or(Value::Undefined)
}

/// Build the cheapest numeric value: [`Value::Int`] when `n` is integral
/// and fits `i32` (preserving `-0` as `Float`), else [`Value::Float`].
///
/// `pub(crate)` for the sibling typed-array builder module
/// (`crate::builtin_typedarray`), which shapes the same numeric slots.
pub(crate) fn js_num(n: f64) -> Value {
    if n == 0.0 {
        return if n.is_sign_negative() {
            Value::Float(n)
        } else {
            Value::Int(0)
        };
    }
    if n.fract() == 0.0 && n > -2147483648.0 && n < 2147483648.0 {
        Value::Int(n as i32)
    } else {
        Value::Float(n)
    }
}

/// Shortest-roundtrip `Number::toString` (same algorithm as the sibling
/// builtin module).
fn js_number_to_string(n: f64) -> String {
    if n.is_nan() {
        return "NaN".to_owned();
    }
    if n == 0.0 {
        return "0".to_owned();
    }
    if n.is_infinite() {
        return if n > 0.0 {
            "Infinity".to_owned()
        } else {
            "-Infinity".to_owned()
        };
    }
    let mut out = String::new();
    let m = n.abs();
    if n < 0.0 {
        out.push('-');
    }
    let full = format!("{m}");
    let (int_part, frac_part) = match full.split_once('.') {
        Some((i, f)) => (i, f),
        None => (full.as_str(), ""),
    };
    let int_len = int_part.len();
    let combined = format!("{int_part}{frac_part}");
    let bytes = combined.as_bytes();
    let mut first = 0;
    while first < bytes.len() && bytes[first] == b'0' {
        first += 1;
    }
    let mut last = bytes.len();
    while last > first && bytes[last - 1] == b'0' {
        last -= 1;
    }
    let digits = &combined[first..last];
    let count = digits.len() as i64;
    let k = int_len as i64 - first as i64;
    if k >= 1 && k <= count {
        out.push_str(&digits[..k as usize]);
        if k < count {
            out.push('.');
            out.push_str(&digits[k as usize..]);
        }
    } else if (1..=21).contains(&k) {
        out.push_str(digits);
        for _ in 0..(k - count) {
            out.push('0');
        }
    } else if k > -6 {
        out.push_str("0.");
        for _ in 0..-k {
            out.push('0');
        }
        out.push_str(digits);
    } else {
        out.push_str(&digits[..1]);
        if count > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        let exp = k - 1;
        if exp >= 0 {
            out.push('+');
        }
        out.push_str(&exp.to_string());
    }
    out
}

fn is_js_whitespace(c: char) -> bool {
    matches!(
        c,
        '\u{0009}'
            | '\u{000A}'
            | '\u{000B}'
            | '\u{000C}'
            | '\u{000D}'
            | '\u{0020}'
            | '\u{00A0}'
            | '\u{1680}'
            | '\u{2000}'
            ..='\u{200A}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202F}'
                | '\u{205F}'
                | '\u{3000}'
                | '\u{FEFF}'
    )
}

fn parse_js_number(s: &str) -> f64 {
    let t = s.trim_matches(is_js_whitespace);
    if t.is_empty() {
        return 0.0;
    }
    if t == "Infinity" || t == "+Infinity" {
        return f64::INFINITY;
    }
    if t == "-Infinity" {
        return f64::NEG_INFINITY;
    }
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        if hex.is_empty() {
            return f64::NAN;
        }
        let mut acc = 0.0f64;
        for c in hex.chars() {
            let d = match c {
                '0'..='9' => c as u32 - '0' as u32,
                'a'..='f' => c as u32 - 'a' as u32 + 10,
                'A'..='F' => c as u32 - 'A' as u32 + 10,
                _ => return f64::NAN,
            };
            acc = acc * 16.0 + f64::from(d);
        }
        return acc;
    }
    // Strict decimal literal check (minus Rust-only forms like `inf`).
    let b = t.as_bytes();
    let mut i = 0;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let mut digits = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
        digits += 1;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return f64::NAN;
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        i += 1;
        if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        let mut ed = 0;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
            ed += 1;
        }
        if ed == 0 {
            return f64::NAN;
        }
    }
    if i != b.len() {
        return f64::NAN;
    }
    t.parse::<f64>().unwrap_or(f64::NAN)
}

/// `ToNumber`. v1 gap: objects coerce to `NaN` (no `ToPrimitive` dispatch).
fn to_number(v: Value) -> Result<f64, NativeError> {
    match v {
        Value::Undefined => Ok(f64::NAN),
        Value::Null => Ok(0.0),
        Value::Bool(b) => Ok(if b { 1.0 } else { 0.0 }),
        Value::Int(i) => Ok(f64::from(i)),
        Value::Float(f) => Ok(f),
        Value::Str(s) => Ok(s
            .as_wtf8()
            .as_utf8()
            .map(parse_js_number)
            .unwrap_or(f64::NAN)),
        Value::Symbol(_) => Err(NativeError::TypeError(
            "cannot convert symbol to number".into(),
        )),
        Value::BigInt(_) => Err(NativeError::TypeError(
            "cannot convert bigint to number".into(),
        )),
        Value::Object(_) => Ok(f64::NAN),
    }
}

/// `ToString`. For objects, a callable native `toString` found on the
/// prototype chain is invoked; otherwise `"[object Object]"`.
fn to_string_js(heap: &mut Heap, v: Value) -> Result<Wtf8Buf, NativeError> {
    match v {
        Value::Undefined => Ok(Wtf8Buf::from("undefined")),
        Value::Null => Ok(Wtf8Buf::from("null")),
        Value::Bool(b) => Ok(Wtf8Buf::from(if b { "true" } else { "false" })),
        Value::Int(i) => Ok(Wtf8Buf::from(&i.to_string())),
        Value::Float(f) => Ok(Wtf8Buf::from(&js_number_to_string(f))),
        Value::Str(s) => Ok(s.as_wtf8().to_owned()),
        Value::Symbol(_) => Err(NativeError::TypeError(
            "cannot convert symbol to string".into(),
        )),
        Value::BigInt(_) => Err(NativeError::TypeError(
            "cannot convert bigint to string (v1 gap)".into(),
        )),
        Value::Object(o) => {
            let func = get_prop_chain(heap, o, Wtf8::from_str("toString"));
            if func.is_some_and(|f| is_callable(heap, f)) {
                let r = call_value(heap, func.unwrap(), Value::Object(o), &[])?;
                match r {
                    Value::Str(s) => return Ok(s.as_wtf8().to_owned()),
                    Value::Int(i) => return Ok(Wtf8Buf::from(&i.to_string())),
                    Value::Float(f) => return Ok(Wtf8Buf::from(&js_number_to_string(f))),
                    Value::Bool(b) => {
                        return Ok(Wtf8Buf::from(if b { "true" } else { "false" }));
                    }
                    _ => {}
                }
            }
            Ok(Wtf8Buf::from("[object Object]"))
        }
    }
}

/// `ToPropertyKey`. Every symbol — well-known or user-created — passes
/// through as [`ElemKey::Symbol`]: no alias translation, no `TypeError`
/// for symbols.
fn to_property_key(heap: &mut Heap, v: Value) -> Result<ElemKey, NativeError> {
    if let Value::Symbol(s) = v {
        return Ok(ElemKey::Symbol(s));
    }
    let s = to_string_js(heap, v)?;
    Ok(ElemKey::Str(heap.alloc_str_wtf8(&s)?))
}

/// `ToIntegerOrInfinity`: `NaN` -> `+0`, truncates toward zero.
fn to_integer_or_infinity(v: Value) -> Result<f64, NativeError> {
    let n = to_number(v)?;
    if n.is_nan() {
        return Ok(0.0);
    }
    Ok(n.trunc())
}

/// `ToIntegerOrInfinity` with full `ToNumber`: objects go through
/// `ToPrimitive` (so `valueOf`/`toString` hooks run and a mid-call
/// `ArrayBuffer.prototype.resize` is honored). For argument conversions
/// the spec makes observable (`%TypedArray%.prototype.slice` /
/// `subarray` start/end, ...).
fn to_integer_or_infinity_full(heap: &mut Heap, v: Value) -> Result<f64, NativeError> {
    let n = to_number_full(heap, v)?;
    if n.is_nan() {
        return Ok(0.0);
    }
    Ok(n.trunc())
}

/// `ToIndex` with full `ToNumber` (objects go through `ToPrimitive`,
/// so `valueOf`/`toString` hooks run).
fn to_index_full(heap: &mut Heap, v: Value) -> Result<u64, NativeError> {
    const MAX: f64 = 9007199254740991.0;
    let n = to_integer_or_infinity_full(heap, v)?;
    if !(0.0..=MAX).contains(&n) {
        return Err(NativeError::RangeError("index out of range".into()));
    }
    Ok(n as u64)
}

/// Full `ToNumber`: objects go through `ToPrimitive(hint Number)` first
/// (so `valueOf`/`toString` hooks run); primitives use the fast path.
/// The primitive-only [`to_number`] maps objects to `NaN`, which is
/// wrong for e.g. `{ valueOf() { return 5; } }`.
fn to_number_full(heap: &mut Heap, v: Value) -> Result<f64, NativeError> {
    match v {
        Value::Object(o) => {
            let prim = to_primitive_with_hint(heap, o, PrimitiveHint::Number)?;
            to_number(prim)
        }
        _ => to_number(v),
    }
}

/// `ToLength`: clamped to `[0, 2^53 - 1]`.
fn to_length(v: Value) -> Result<u64, NativeError> {
    const MAX: f64 = 9007199254740991.0;
    let n = to_number(v)?;
    if n.is_nan() || n <= 0.0 {
        return Ok(0);
    }
    if n.is_infinite() {
        return Ok(MAX as u64);
    }
    Ok(n.trunc().min(MAX) as u64)
}

/// `ToIndex`: integer in `[0, 2^53 - 1]`, else `RangeError`.
fn to_index(v: Value) -> Result<u64, NativeError> {
    const MAX: f64 = 9007199254740991.0;
    let n = to_integer_or_infinity(v)?;
    if !(0.0..=MAX).contains(&n) {
        return Err(NativeError::RangeError("index out of range".into()));
    }
    Ok(n as u64)
}

/// `Strict Equality Comparison` (`===`).
fn strict_equal(a: Value, b: Value) -> bool {
    match (a, b) {
        (Value::Undefined, Value::Undefined) | (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => x == y,
        (Value::Int(x), Value::Float(y)) => f64::from(x) == y,
        (Value::Float(x), Value::Int(y)) => x == f64::from(y),
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Object(x), Value::Object(y)) => x == y,
        (Value::Symbol(x), Value::Symbol(y)) => x == y,
        (Value::BigInt(x), Value::BigInt(y)) => x == y,
        _ => false,
    }
}

/// `SameValue`: like `===` but `NaN` equals `NaN` and `+0` differs from `-0`.
fn same_value(a: Value, b: Value) -> bool {
    match (a, b) {
        (Value::Float(x), Value::Float(y)) => x.to_bits() == y.to_bits(),
        (Value::Int(0), Value::Float(y)) | (Value::Float(y), Value::Int(0)) => {
            y.to_bits() == 0.0f64.to_bits()
        }
        _ => strict_equal(a, b),
    }
}

/// True when a value is callable: a native or bytecode function object.
fn is_callable(heap: &Heap, v: Value) -> bool {
    match v {
        Value::Object(o) => matches!(
            heap.get(o).kind,
            ObjectKind::NativeFunction(_) | ObjectKind::Function(_)
        ),
        _ => false,
    }
}

/// If `o` carries a proxy link — the [`as_proxy`] fail-closed rule, both
/// the `P_TARGET` and the `P_HANDLER` slot holding objects — return the
/// link's target. A lone or non-object `P_TARGET` slot is script forgery
/// on a plain object, not a proxy, and is ignored here.
fn proxy_link_target(heap: &Heap, o: Gc<Object>) -> Option<Gc<Object>> {
    let target = match heap
        .get_property(o, Wtf8::from_str(P_TARGET))
        .map(|p| p.value)
    {
        Some(Value::Object(t)) => t,
        _ => return None,
    };
    match heap
        .get_property(o, Wtf8::from_str(P_HANDLER))
        .map(|p| p.value)
    {
        Some(Value::Object(_)) => Some(target),
        _ => None,
    }
}

/// `IsCallable`, proxy-transparent: a proxy counts as callable (or a
/// constructor) exactly when its target does. Loops instead of recursing;
/// the walk is bounded by [`MAX_PROXY_UNWRAP_HOPS`] and fails closed on forged
/// or cyclic chains (see [`proxy_link_target`]).
pub(crate) fn is_callable_deep(heap: &Heap, v: Value) -> bool {
    let mut current = v;
    let mut hops = 0u32;
    loop {
        let Value::Object(o) = current else {
            return false;
        };
        // Kind first: proxies are always `Ordinary`-kind, so function
        // values short-circuit without a `P_TARGET` hash probe.
        if matches!(
            heap.get(o).kind,
            ObjectKind::NativeFunction(_) | ObjectKind::Function(_)
        ) {
            return true;
        }
        match proxy_link_target(heap, o) {
            Some(t) => {
                hops += 1;
                if hops > MAX_PROXY_UNWRAP_HOPS {
                    return false;
                }
                current = Value::Object(t);
            }
            None => return false,
        }
    }
}

/// `IsConstructor`, proxy-transparent: a proxy counts as a constructor
/// exactly when its target does. Loops instead of recursing; the walk is
/// bounded by [`MAX_PROXY_UNWRAP_HOPS`] and fails closed on forged or cyclic
/// chains (see [`proxy_link_target`]).
///
/// Bytecode functions report `true` here *as targets* and let the VM's
/// construct path enforce the fine-grained rule (arrows, methods, async
/// functions, and generators throw `TypeError` there): the observable
/// error type is `TypeError` either way, and the template metadata the VM
/// consults lives outside the heap. That leniency covers the *target*
/// position only — a `newTarget` is never itself constructed, so
/// [`is_new_target_constructible`] consults the template metadata
/// directly (ECMA-262 `Reflect.construct` step 3).
///
/// Native functions consult the install-time registry
/// ([`Heap::mark_native_constructor`]): a callable native that was not
/// installed as a constructor (e.g. `Array.isArray`, `Math.max`) is not a
/// constructor, so `SpeciesConstructor`'s step-6 `TypeError` fires instead
/// of silently "constructing" it.
pub(crate) fn is_constructor(heap: &Heap, v: Value) -> bool {
    let mut current = v;
    let mut hops = 0u32;
    loop {
        let Value::Object(o) = current else {
            return false;
        };
        // Kind first: proxies are always `Ordinary`-kind, so function
        // values short-circuit without a `P_TARGET` hash probe (species
        // hot path). A forged `P_TARGET` slot on a function is ignored.
        match heap.get(o).kind {
            ObjectKind::NativeFunction(f) => return heap.is_native_constructor(f),
            ObjectKind::Function(_) => return true,
            _ => {}
        }
        match proxy_link_target(heap, o) {
            Some(t) => {
                hops += 1;
                // `P_TARGET` slots are non-writable engine internals (see
                // [`define_hidden_const`]), so a cycle here can only be
                // forged script-side; bound the walk regardless.
                if hops > MAX_PROXY_UNWRAP_HOPS {
                    return false;
                }
                current = Value::Object(t);
            }
            None => return false,
        }
    }
}

/// Strict `IsConstructor` for `Reflect.construct`'s `newTarget`
/// (ECMA-262 step 3).
///
/// Proxy-transparent and hop-bounded like [`is_constructor`]: a proxy
/// counts exactly when its ultimate target does. Unlike
/// [`is_constructor`] — which reports every bytecode function as a
/// constructor and relies on the VM's construct path to reject
/// arrows/methods/async functions/generators *as targets* — a `newTarget`
/// is never itself constructed, so the native-target arm of
/// `reflect_construct_inner` would otherwise silently accept
/// `Reflect.construct(Array, [], () => {})`. Bytecode functions therefore
/// go through [`crate::vm::is_bytecode_constructible`], which consults
/// the VM's template metadata directly.
fn is_new_target_constructible(heap: &Heap, v: Value) -> bool {
    let mut current = v;
    let mut hops = 0u32;
    loop {
        let Value::Object(o) = current else {
            return false;
        };
        match heap.get(o).kind {
            ObjectKind::NativeFunction(f) => return heap.is_native_constructor(f),
            ObjectKind::Function(_) => {
                return crate::vm::is_bytecode_constructible(heap, Value::Object(o));
            }
            _ => {}
        }
        match proxy_link_target(heap, o) {
            Some(t) => {
                hops += 1;
                if hops > MAX_PROXY_UNWRAP_HOPS {
                    return false;
                }
                current = Value::Object(t);
            }
            None => return false,
        }
    }
}

/// Invoke a callable value: native callees run directly, bytecode
/// ([`ObjectKind::Function`]) callees re-enter the VM through the shared
/// [`crate::builtin_object_array::call_function`] choke point.
///
/// This is the single choke point all trap and `Reflect` dispatch goes
/// through.
fn call_value(
    heap: &mut Heap,
    callee: Value,
    this_arg: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    crate::builtin_object_array::call_function(heap, callee, this_arg, args)
}

/// Allocate a native function object with `name`/`length` installed.
fn new_native_function(
    heap: &mut Heap,
    name: &str,
    func: NativeFn,
    length: u32,
) -> Result<Gc<Object>, NativeError> {
    let f = heap.alloc_object(Object::new(ObjectKind::NativeFunction(func)))?;
    // `f` is not reachable from any root yet; hold it across the string
    // allocation, which may trigger a collection.
    let name_val = with_temp_root(heap, Value::Object(f), |heap| {
        heap.alloc_str(name).map_err(|_| NativeError::OutOfMemory)
    })?;
    heap.define_property_str(
        f,
        "length",
        Property {
            value: js_num(f64::from(length)),
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )?;
    heap.define_property_str(
        f,
        "name",
        Property {
            value: Value::Str(name_val),
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )?;
    Ok(f)
}

/// Allocate a native *constructor* with `name`/`length` installed and
/// record its implementation in the heap's constructor registry, so
/// [`is_constructor`] (ECMA-262 `IsConstructor`) accepts it. Use for
/// every builtin installed as a constructible function (`Array`,
/// `ArrayBuffer`, the typed-array constructors, …); plain callable
/// builtins (`Array.isArray`, `Math.max`, prototype methods) keep using
/// [`new_native_function`] and are correctly rejected as species
/// constructors and `Reflect.construct` targets.
///
/// The single definition for the crate (the former duplicate in
/// `builtin_string_number` was removed): it delegates to
/// [`crate::builtin_string_number::new_native_function`], whose
/// `%Function.prototype%` `[[Prototype]]` link this module's local
/// `new_native_function` does not install.
pub(crate) fn new_native_constructor(
    heap: &mut Heap,
    name: &str,
    func: NativeFn,
    length: u32,
) -> Result<Gc<Object>, NativeError> {
    let f = crate::builtin_string_number::new_native_function(heap, name, func, length)?;
    heap.mark_native_constructor(func);
    Ok(f)
}

/// Define `target[name] = <native function>`.
fn define_native(
    heap: &mut Heap,
    target: Gc<Object>,
    name: &str,
    func: NativeFn,
    length: u32,
) -> Result<(), NativeError> {
    // Hold `target` across the function-object allocation below, which may
    // trigger a collection while the target is not yet otherwise rooted.
    let f = with_temp_root(heap, Value::Object(target), |heap| {
        new_native_function(heap, name, func, length)
    })?;
    heap.define_property_str(target, name, Property::data(Value::Object(f)))?;
    Ok(())
}

/// Define a non-writable, non-enumerable, non-configurable constant.
fn define_const(
    heap: &mut Heap,
    target: Gc<Object>,
    name: &str,
    value: Value,
) -> Result<(), NativeError> {
    heap.define_property_str(
        target,
        name,
        Property {
            value,
            writable: false,
            enumerable: false,
            configurable: false,
        },
    )?;
    Ok(())
}

/// Define a hidden (non-enumerable) data property used for internal slots.
///
/// `pub(crate)` for the `Proxy` machinery below, which keeps its
/// target/handler/revocation links in NUL-prefixed slots. (Typed-array,
/// `ArrayBuffer`, `DataView`, and `Date` internals used to live in such
/// slots too; they are unforgeable [`ObjectKind`] tags now.)
///
/// The slots stay writable: the engine itself rewrites `P_REVOKED` on
/// revoke and the typed-array iterator index on every `next()`.
pub(crate) fn define_hidden(
    heap: &mut Heap,
    target: Gc<Object>,
    name: &str,
    value: Value,
) -> Result<(), NativeError> {
    heap.define_property_str(
        target,
        name,
        Property {
            value,
            writable: true,
            enumerable: false,
            configurable: false,
        },
    )?;
    Ok(())
}

/// Define a hidden internal slot that script must never rewrite:
/// non-writable, non-enumerable, non-configurable.
///
/// Use for engine-owned links that are written once at creation — the
/// proxy `P_TARGET`/`P_HANDLER` slots. With writable slots, script can
/// repoint a live proxy's target: assigning the NUL-prefixed key on a
/// *proxy of* the victim goes through the receiver-based `[[Set]]` and
/// lands on the victim's own slot (`p2["\0bowser:proxy-target"] = evil`
/// rewrites `p1`'s slot when `p2 = new Proxy(p1, {})`), and a
/// self-referential store forges a cyclic `P_TARGET` chain. Ordinary
/// `[[Set]]` and `[[DefineOwnProperty]]` both reject the write once the
/// slot is non-writable + non-configurable (see [`define_ordinary`]).
fn define_hidden_const(
    heap: &mut Heap,
    target: Gc<Object>,
    name: &str,
    value: Value,
) -> Result<(), NativeError> {
    heap.define_property_str(
        target,
        name,
        Property {
            value,
            writable: false,
            enumerable: false,
            configurable: false,
        },
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Property helpers (spec [[Get]]/[[Set]]/[[DefineOwnProperty]]/[[Delete]])
// ---------------------------------------------------------------------------

/// Is `s` a canonical array index (`"0"` .. `"4294967294"`)?
/// Own property descriptor, or `None`.
fn get_own(heap: &Heap, obj: Gc<Object>, key: &Wtf8) -> Option<Property> {
    heap.get_property(obj, key)
}

/// [`get_own`] over a [`PropKey`].
fn get_own_key(heap: &Heap, obj: Gc<Object>, key: PropKey<'_>) -> Option<Property> {
    heap.get_property_key(obj, key)
}

/// `[[Get]]`: own property, else walk the prototype chain (cycle-guarded).
pub(crate) fn get_prop_chain(heap: &Heap, obj: Gc<Object>, key: &Wtf8) -> Option<Value> {
    walk_proto_chain(heap, obj, |o| heap.get_property(o, key).map(|p| p.value))
}

/// `HasProperty`: `[[Get]]` presence, prototype chain included
/// (cycle-guarded).
fn has_prop_chain(heap: &Heap, obj: Gc<Object>, key: &Wtf8) -> bool {
    walk_proto_chain(heap, obj, |o| heap.get_property(o, key)).is_some()
}

/// [`has_prop_chain`] over a [`PropKey`]: string keys take the historical
/// path; symbol keys scan the symbol data-property side table.
fn has_prop_chain_key(heap: &Heap, obj: Gc<Object>, key: PropKey<'_>) -> bool {
    if let PropKey::Str(s) = key {
        return has_prop_chain(heap, obj, s);
    }
    walk_proto_chain(heap, obj, |o| heap.get_property_key(o, key)).is_some()
}

/// Bound on consecutive proxy-delegation hops (`proxy_get`/`proxy_set` ⇄
/// `ordinary_get`/`ordinary_set`). A proxy embedded in a prototype chain
/// can otherwise recurse forever on a self-referential chain (constructible
/// via `Object.setPrototypeOf(t, new Proxy(t, {}))`); legitimate nesting
/// never approaches this. Mirrors the `MAX_NATIVE_RECURSION` idiom in
/// `vm.rs`: exceed it and the operation fails with a `TypeError` instead
/// of overflowing the native stack.
const MAX_PROXY_HOPS: u32 = 128;

/// Bound on iterations of the iterative `P_TARGET`-following loops in
/// [`is_constructor`], [`is_callable_deep`], and
/// [`is_new_target_constructible`].
///
/// Unlike [`MAX_PROXY_HOPS`] — which bounds *recursive* delegation where
/// every hop consumes native stack — these loops are iterative, so a
/// forged cycle costs only hash probes, not stack. The bound must still
/// sit above legitimate nesting: the engine supports thousands of
/// trap-less nested proxies on the call path (which trips its own
/// resource cap past ~1024), so `is_callable_deep` has to answer past
/// that. Past this bound the value fails closed (not a constructor / not
/// callable); script can forge a cyclic `P_TARGET` chain on plain objects
/// (e.g. `p["\0bowser:proxy-target"] = p` through a proxy's
/// receiver-based `[[Set]]`), and those loops run with no fuel check —
/// without the bound, `Reflect.construct` on such an object would wedge
/// the tab.
const MAX_PROXY_UNWRAP_HOPS: u32 = 8192;

/// Ordinary `[[Get]]` (ECMA-262 10.1.8): walks the prototype chain
/// (cycle-guarded); invokes accessor getters with `receiver` as `this`;
/// delegates to nested proxies so inner traps still fire. Used for the
/// no-trap path of `proxy_get` and for `Reflect.get` on ordinary targets.
fn ordinary_get(
    heap: &mut Heap,
    obj: Gc<Object>,
    key: &Wtf8,
    receiver: Value,
    depth: u32,
) -> Result<Value, NativeError> {
    // `obj` is not yet reachable from any root; hold it across the
    // allocations below (a getter call can collect).
    with_temp_root(heap, Value::Object(obj), |heap| {
        let mut current = Value::Object(obj);
        // Every `[[SetPrototypeOf]]` path rejects cycles up front, but
        // this visited set keeps the walk total even if one slips through.
        let mut visited: Vec<Gc<Object>> = Vec::new();
        loop {
            let o = match current {
                Value::Object(o) => o,
                _ => return Ok(Value::Undefined),
            };
            if visited.contains(&o) {
                return Ok(Value::Undefined);
            }
            visited.push(o);
            // A proxy in the chain performs its own `[[Get]]`, traps
            // included, with the same receiver.
            if let Some(inner) = as_proxy(heap, Value::Object(o)) {
                return proxy_get(heap, inner, PropKey::Str(key), receiver, depth + 1);
            }
            if let Some(acc) = heap.get_accessor(o, key) {
                return match acc.get {
                    Value::Undefined => Ok(Value::Undefined),
                    getter => call_value(heap, getter, receiver, &[]),
                };
            }
            if let Some(prop) = heap.get_property(o, key) {
                return Ok(prop.value);
            }
            current = heap.get(o).prototype;
        }
    })
}

/// [`ordinary_get`] over a [`PropKey`]: string keys take the historical
/// path verbatim; symbol keys walk the chain with the symbol side
/// tables. A proxy nested in the prototype chain performs its own
/// `[[Get]]`, traps included, exactly like the string path.
fn ordinary_get_key(
    heap: &mut Heap,
    obj: Gc<Object>,
    key: PropKey<'_>,
    receiver: Value,
    depth: u32,
) -> Result<Value, NativeError> {
    if let PropKey::Str(s) = key {
        return ordinary_get(heap, obj, s, receiver, depth);
    }
    // `obj` is not yet reachable from any root; hold it across the
    // allocations below (a getter call can collect).
    with_temp_root(heap, Value::Object(obj), |heap| {
        let mut current = Value::Object(obj);
        // Every `[[SetPrototypeOf]]` path rejects cycles up front, but
        // this visited set keeps the walk total even if one slips through.
        let mut visited: Vec<Gc<Object>> = Vec::new();
        loop {
            let o = match current {
                Value::Object(o) => o,
                _ => return Ok(Value::Undefined),
            };
            if visited.contains(&o) {
                return Ok(Value::Undefined);
            }
            visited.push(o);
            // A proxy in the chain performs its own `[[Get]]`, traps
            // included, with the same receiver.
            if let Some(inner) = as_proxy(heap, Value::Object(o)) {
                return proxy_get(heap, inner, key, receiver, depth + 1);
            }
            if let Some(acc) = heap.get_accessor_key(o, key) {
                return match acc.get {
                    Value::Undefined => Ok(Value::Undefined),
                    getter => call_value(heap, getter, receiver, &[]),
                };
            }
            if let Some(prop) = heap.get_property_key(o, key) {
                return Ok(prop.value);
            }
            current = heap.get(o).prototype;
        }
    })
}

/// Ordinary `[[Set]]` (ECMA-262 10.1.9, `OrdinarySetWithOwnDescriptor`):
/// walks the prototype chain for the first own descriptor (cycle-guarded);
/// fires inherited setters with `receiver` as `this`; creates or updates a
/// data property on `receiver` (never on the target itself); delegates to
/// nested proxies so inner traps still fire. Returns `Ok(false)` — never a
/// throw — when the set fails (non-writable property, missing setter,
/// non-extensible receiver, ...). Used for the no-trap path of
/// `proxy_set` and for `Reflect.set` on ordinary targets, and by the
/// array species helpers for spec `Set` writes on hostile results.
pub(crate) fn ordinary_set(
    heap: &mut Heap,
    obj: Gc<Object>,
    key: &Wtf8,
    value: Value,
    receiver: Value,
    depth: u32,
) -> Result<bool, NativeError> {
    // `obj` is not yet reachable from any root; hold it across the
    // allocations below (a setter call can collect).
    with_temp_root(heap, Value::Object(obj), |heap| {
        let mut current = Value::Object(obj);
        let mut visited: Vec<Gc<Object>> = Vec::new();
        loop {
            let o = match current {
                Value::Object(o) => o,
                // Null prototype: no descriptor found; create on receiver.
                _ => return define_on_receiver(heap, key, value, receiver),
            };
            if visited.contains(&o) {
                return define_on_receiver(heap, key, value, receiver);
            }
            visited.push(o);
            // A proxy in the chain performs its own `[[Set]]`, traps
            // included, with the same receiver.
            if let Some(inner) = as_proxy(heap, Value::Object(o)) {
                return proxy_set(heap, inner, PropKey::Str(key), value, receiver, depth + 1);
            }
            if let Some(acc) = heap.get_accessor(o, key) {
                return match acc.set {
                    Value::Undefined => Ok(false),
                    setter => {
                        call_value(heap, setter, receiver, &[value])?;
                        Ok(true)
                    }
                };
            }
            if let Some(prop) = heap.get_property(o, key) {
                if !prop.writable {
                    return Ok(false);
                }
                return define_on_receiver(heap, key, value, receiver);
            }
            current = heap.get(o).prototype;
        }
    })
}

/// [`ordinary_set`] over a [`PropKey`]: string keys take the historical
/// path verbatim; symbol keys walk the chain with the symbol side
/// tables. A proxy nested in the prototype chain performs its own
/// `[[Set]]`, traps included, exactly like the string path.
fn ordinary_set_key(
    heap: &mut Heap,
    obj: Gc<Object>,
    key: PropKey<'_>,
    value: Value,
    receiver: Value,
    depth: u32,
) -> Result<bool, NativeError> {
    if let PropKey::Str(s) = key {
        return ordinary_set(heap, obj, s, value, receiver, depth);
    }
    // `obj` is not yet reachable from any root; hold it across the
    // allocations below (a setter call can collect).
    with_temp_root(heap, Value::Object(obj), |heap| {
        let mut current = Value::Object(obj);
        let mut visited: Vec<Gc<Object>> = Vec::new();
        loop {
            let o = match current {
                Value::Object(o) => o,
                // Null prototype: no descriptor found; create on receiver.
                _ => return define_on_receiver_key(heap, key, value, receiver),
            };
            if visited.contains(&o) {
                return define_on_receiver_key(heap, key, value, receiver);
            }
            visited.push(o);
            // A proxy in the chain performs its own `[[Set]]`, traps
            // included, with the same receiver.
            if let Some(inner) = as_proxy(heap, Value::Object(o)) {
                return proxy_set(heap, inner, key, value, receiver, depth + 1);
            }
            if let Some(acc) = heap.get_accessor_key(o, key) {
                return match acc.set {
                    Value::Undefined => Ok(false),
                    setter => {
                        call_value(heap, setter, receiver, &[value])?;
                        Ok(true)
                    }
                };
            }
            if let Some(prop) = heap.get_property_key(o, key) {
                if !prop.writable {
                    return Ok(false);
                }
                return define_on_receiver_key(heap, key, value, receiver);
            }
            current = heap.get(o).prototype;
        }
    })
}

/// Shared tail of [`ordinary_set_key`]: define the data property for
/// `value` on `receiver`. A proxy receiver goes through its own
/// `[[DefineOwnProperty]]`, so a `defineProperty` trap still fires —
/// exactly like the string path.
fn define_on_receiver_key(
    heap: &mut Heap,
    key: PropKey<'_>,
    value: Value,
    receiver: Value,
) -> Result<bool, NativeError> {
    if let PropKey::Str(s) = key {
        return define_on_receiver(heap, s, value, receiver);
    }
    let recv_obj = match receiver {
        Value::Object(o) => o,
        _ => return Ok(false),
    };
    // Hold the receiver across the descriptor allocation below.
    with_temp_root(heap, receiver, |heap| {
        if let Some(inner) = as_proxy(heap, receiver) {
            let desc = descriptor_object(heap, Property::data(value))?;
            return proxy_define_property(heap, inner, key, Value::Object(desc));
        }
        if heap.get_accessor_key(recv_obj, key).is_some() {
            // An accessor already lives on the receiver: redefining it as
            // data is a shape change the ordinary `[[Set]]` path must not
            // perform on its own.
            return Ok(false);
        }
        match heap.get_property_key(recv_obj, key) {
            Some(existing) => {
                if !existing.writable {
                    return Ok(false);
                }
                heap.define_property_key(recv_obj, key, Property { value, ..existing })
                    .map_err(|_| NativeError::OutOfMemory)?;
                Ok(true)
            }
            None => {
                if !heap.get(recv_obj).extensible {
                    return Ok(false);
                }
                heap.define_property_key(recv_obj, key, Property::data(value))
                    .map_err(|_| NativeError::OutOfMemory)?;
                Ok(true)
            }
        }
    })
}

/// Shared tail of [`ordinary_set`]: define the data property for `value`
/// on `receiver` (the spec's `CreateDataProperty`/`DefineOwnProperty`
/// step). A proxy receiver goes through its own `[[DefineOwnProperty]]`,
/// so a `defineProperty` trap still fires and — without one — the property
/// lands on the target, preserving the old observable read-back. An
/// ordinary receiver is defined on directly. A non-object receiver, an
/// accessor already on the receiver, or a non-extensible receiver fails
/// with `Ok(false)`.
fn define_on_receiver(
    heap: &mut Heap,
    key: &Wtf8,
    value: Value,
    receiver: Value,
) -> Result<bool, NativeError> {
    let recv_obj = match receiver {
        Value::Object(o) => o,
        _ => return Ok(false),
    };
    // Hold the receiver across the descriptor allocation below.
    with_temp_root(heap, receiver, |heap| {
        if let Some(inner) = as_proxy(heap, receiver) {
            let desc = descriptor_object(heap, Property::data(value))?;
            return proxy_define_property(heap, inner, PropKey::Str(key), Value::Object(desc));
        }
        if heap.get_accessor(recv_obj, key).is_some() {
            // An accessor already lives on the receiver: redefining it as
            // data is a shape change the ordinary `[[Set]]` path must not
            // perform on its own.
            return Ok(false);
        }
        match heap.get_property(recv_obj, key) {
            Some(existing) => {
                if !existing.writable {
                    return Ok(false);
                }
                heap.define_property(recv_obj, key, Property { value, ..existing })
                    .map_err(|_| NativeError::OutOfMemory)?;
                Ok(true)
            }
            None => {
                if !heap.get(recv_obj).extensible {
                    return Ok(false);
                }
                heap.define_property(recv_obj, key, Property::data(value))
                    .map_err(|_| NativeError::OutOfMemory)?;
                Ok(true)
            }
        }
    })
}

/// Ordinary `[[DefineOwnProperty]]` with validation: rejects incompatible
/// redefinitions of non-configurable properties, rejects definitions on
/// non-extensible objects when the property is new. Returns `Ok(false)`
/// for spec-level failure (so `Reflect.defineProperty` reports `false`);
/// `Err` only for malformed input.
fn define_ordinary(
    heap: &mut Heap,
    obj: Gc<Object>,
    key: &Wtf8,
    prop: Property,
) -> Result<bool, NativeError> {
    match heap.get_property(obj, key) {
        Some(existing) => {
            if !existing.configurable {
                if prop.configurable || prop.enumerable != existing.enumerable {
                    return Ok(false);
                }
                if !existing.writable && (prop.writable || !same_value(prop.value, existing.value))
                {
                    return Ok(false);
                }
            }
            heap.define_property(obj, key, prop)?;
            Ok(true)
        }
        None => {
            if !heap.get(obj).extensible {
                return Ok(false);
            }
            heap.define_property(obj, key, prop)?;
            Ok(true)
        }
    }
}

/// [`define_ordinary`] over a [`PropKey`]: string keys take the
/// historical path verbatim; symbol keys validate against the symbol
/// side tables.
fn define_ordinary_key(
    heap: &mut Heap,
    obj: Gc<Object>,
    key: PropKey<'_>,
    prop: Property,
) -> Result<bool, NativeError> {
    if let PropKey::Str(s) = key {
        return define_ordinary(heap, obj, s, prop);
    }
    match heap.get_property_key(obj, key) {
        Some(existing) => {
            if !existing.configurable {
                if prop.configurable || prop.enumerable != existing.enumerable {
                    return Ok(false);
                }
                if !existing.writable && (prop.writable || !same_value(prop.value, existing.value))
                {
                    return Ok(false);
                }
            }
            heap.define_property_key(obj, key, prop)?;
            Ok(true)
        }
        None => {
            if !heap.get(obj).extensible {
                return Ok(false);
            }
            heap.define_property_key(obj, key, prop)?;
            Ok(true)
        }
    }
}

/// [`define_from_descriptor`] over a [`PropKey`]: string keys take the
/// historical path verbatim (array exotic included); symbol keys never
/// hit the array exotic and define in the symbol side tables.
fn define_from_descriptor_key(
    heap: &mut Heap,
    obj: Gc<Object>,
    key: PropKey<'_>,
    desc: Value,
) -> Result<bool, NativeError> {
    if let PropKey::Str(s) = key {
        return define_from_descriptor(heap, obj, s, desc);
    }
    let parts = read_descriptor(heap, desc)?;
    match heap.get_property_key(obj, key) {
        Some(existing) => {
            let prop = Property {
                value: parts.value.unwrap_or(existing.value),
                writable: parts.writable.unwrap_or(existing.writable),
                enumerable: parts.enumerable.unwrap_or(existing.enumerable),
                configurable: parts.configurable.unwrap_or(existing.configurable),
            };
            define_ordinary_key(heap, obj, key, prop)
        }
        None => {
            if !heap.get(obj).extensible {
                return Ok(false);
            }
            let prop = Property {
                value: parts.value.unwrap_or(Value::Undefined),
                writable: parts.writable.unwrap_or(false),
                enumerable: parts.enumerable.unwrap_or(false),
                configurable: parts.configurable.unwrap_or(false),
            };
            heap.define_property_key(obj, key, prop)?;
            Ok(true)
        }
    }
}

/// Ordinary `[[Delete]]`: `false` for non-configurable own properties.
fn delete_ordinary(heap: &mut Heap, obj: Gc<Object>, key: &Wtf8) -> bool {
    match heap.get_property(obj, key) {
        Some(p) => {
            if !p.configurable {
                return false;
            }
            heap.delete_property(obj, key)
        }
        None => true,
    }
}

/// [`delete_ordinary`] over a [`PropKey`]: string keys take the
/// historical path verbatim; symbol keys delete from the symbol side
/// tables.
fn delete_ordinary_key(heap: &mut Heap, obj: Gc<Object>, key: PropKey<'_>) -> bool {
    if let PropKey::Str(s) = key {
        return delete_ordinary(heap, obj, s);
    }
    match heap.get_property_key(obj, key) {
        Some(p) => {
            if !p.configurable {
                return false;
            }
            heap.delete_property_key(obj, key)
        }
        None => true,
    }
}

/// One `[[OwnPropertyKeys]]` entry in spec order: a string key as owned
/// bytes (never a GC handle, so the string half of a key list needs no
/// rooting) or a symbol key (a GC handle — root the list before any call
/// that can collect).
#[derive(Clone, PartialEq, Eq)]
enum OwnKey {
    Str(Wtf8Buf),
    Sym(Gc<Symbol>),
}

impl OwnKey {
    /// Borrow as a [`PropKey`] for the `*_key` heap APIs.
    fn as_prop_key(&self) -> PropKey<'_> {
        match self {
            OwnKey::Str(s) => PropKey::Str(s.as_wtf8()),
            OwnKey::Sym(sym) => PropKey::Symbol(*sym),
        }
    }
}

/// `true` when `a` and `b` hold the same keys, order-insensitive. Both
/// lists are duplicate-free by construction (trap results are
/// duplicate-checked; target keys are unique).
fn own_key_set_eq(a: &[OwnKey], b: &[OwnKey]) -> bool {
    a.len() == b.len() && a.iter().all(|k| b.contains(k))
}

/// All own keys in spec order: ascending array indices, then other string
/// keys in insertion order, then symbol keys.
///
/// Data and accessor properties live in separate lists with no shared
/// insertion sequence (a data↔accessor conversion moves the key to the end
/// of the other list), so among non-index keys data properties precede
/// accessor properties, each in list order; that is the documented
/// approximation of chronological order. The symbol tail follows the same
/// approximation (data side table, then accessor side table).
fn own_keys_in_order(heap: &Heap, obj: Gc<Object>) -> Vec<OwnKey> {
    let o = heap.get(obj);
    let mut indices: Vec<(u32, Wtf8Buf)> = Vec::new();
    let mut others: Vec<Wtf8Buf> = Vec::new();
    // Data properties first, then accessors (see the doc comment above);
    // each list contributes its keys in list order.
    for k in o
        .properties
        .iter()
        .map(|(k, _)| k.as_wtf8())
        .chain(o.accessors.iter().map(|(k, _)| k.as_wtf8()))
    {
        match array_index_key(k) {
            Some(i) => indices.push((i, k.to_owned())),
            None => others.push(k.to_owned()),
        }
    }
    indices.sort_by_key(|(i, _)| *i);
    let mut out: Vec<OwnKey> = indices
        .into_iter()
        .map(|(_, k)| OwnKey::Str(k))
        .chain(others.into_iter().map(OwnKey::Str))
        .collect();
    // Spec tail: symbol keys after every string key.
    out.extend(heap.own_symbol_keys(obj).into_iter().map(OwnKey::Sym));
    out
}

/// [`own_configurable`] over a [`PropKey`]: string keys take the
/// historical path; symbol keys consult the symbol side tables.
fn own_configurable_key(heap: &Heap, obj: Gc<Object>, key: PropKey<'_>) -> Option<bool> {
    if let Some(p) = heap.get_property_key(obj, key) {
        return Some(p.configurable);
    }
    heap.get_accessor_key(obj, key).map(|a| a.configurable)
}

/// The trap-argument value for a [`PropKey`]: strings are freshly
/// allocated (a trap may observe the key string), symbols pass through
/// as `Value::Symbol`.
fn prop_key_value(heap: &mut Heap, key: PropKey<'_>) -> Result<Value, NativeError> {
    match key {
        PropKey::Str(s) => Ok(Value::Str(heap.alloc_str_wtf8(s)?)),
        PropKey::Symbol(sym) => Ok(Value::Symbol(sym)),
    }
}

/// Allocate an `Array`-kind object holding the given own keys: strings
/// as `Value::Str`, symbols as `Value::Symbol`.
///
/// Elements go through dense storage and are materialized once, so batch
/// construction is O(n) instead of O(n²).
fn new_key_array(heap: &mut Heap, keys: &[OwnKey]) -> Result<Gc<Object>, NativeError> {
    let arr = heap.new_dense_array()?;
    // Root the array and every symbol key across the per-element
    // allocations below (each `alloc_str_wtf8` may collect); the strings
    // are born live and `dense_push` performs no GC-heap allocation.
    let base = heap.temp_roots_len();
    heap.protect(Value::Object(arr));
    for k in keys {
        if let OwnKey::Sym(sym) = k {
            heap.protect(Value::Symbol(*sym));
        }
    }
    for k in keys {
        let v = match k {
            OwnKey::Str(s) => Value::Str(heap.alloc_str_wtf8(s.as_wtf8())?),
            OwnKey::Sym(sym) => Value::Symbol(*sym),
        };
        heap.dense_push(arr, v)?;
    }
    heap.release_temp(base);
    heap.materialize_dense(arr)?;
    Ok(arr)
}

/// Read an element of an array-like object (own or inherited index).
fn array_like_get(heap: &Heap, arr: Gc<Object>, i: u64) -> Value {
    get_prop_chain(heap, arr, Wtf8::from_str(&i.to_string())).unwrap_or(Value::Undefined)
}

/// Spec `Get(arr, i)` for array-like sources in typed-array operations:
/// full `[[Get]]` (invokes accessor getters, propagates abrupt
/// completions), unlike [`array_like_get`] which only reads data
/// properties. Used by `set`/`from`/constructor source reads where
/// test262 asserts getter abrupt completions propagate.
fn array_like_get_ordinary(heap: &mut Heap, arr: Gc<Object>, i: u64) -> Result<Value, NativeError> {
    ordinary_get(
        heap,
        arr,
        Wtf8::from_str(&i.to_string()),
        Value::Object(arr),
        0,
    )
}

/// Spec `Get(obj, "length")` with full `[[Get]]` semantics.
fn array_like_length_ordinary(heap: &mut Heap, obj: Gc<Object>) -> Result<u64, NativeError> {
    let len_val = ordinary_get(heap, obj, Wtf8::from_str("length"), Value::Object(obj), 0)?;
    to_length(len_val)
}

/// Spec `GetMethod(obj, @@iterator)` + iterator-protocol consumption for
/// typed-array construction (`new TA(iterable)`, `TypedArray.from`):
/// returns `Some((values, base))` when `obj` is iterable, `None` when it
/// has no `@@iterator` method (the caller then falls back to the
/// array-like path). Each `next()` result must be an object; `done` /
/// `value` are read per iteration and a throwing `next` propagates. The
/// collected values are temp-rooted and stay rooted until the caller
/// runs `heap.release_temp(base)` after consuming them — `next()` calls
/// and the later array allocation can both trigger a collection.
fn iterable_values(
    heap: &mut Heap,
    obj: Gc<Object>,
) -> Result<Option<(Vec<Value>, usize)>, NativeError> {
    let Some(method) = get_method(heap, Value::Object(obj), WellKnown::Iterator)? else {
        return Ok(None);
    };
    let iterator = call_value(heap, method, Value::Object(obj), &[])?;
    let Value::Object(iter_obj) = iterator else {
        return Err(NativeError::TypeError(
            "iterator method did not return an object".into(),
        ));
    };
    let mut values: Vec<Value> = Vec::new();
    // `iter_obj` is a call argument (rooted by `call_roots`); the values
    // below are not — protect each as it arrives. `base` is the outer
    // slot (holding `iter_obj`), so releasing to it pops `iter_obj` and
    // every value pushed after it in one call.
    let base = heap.protect(Value::Object(iter_obj));
    let out = (|| -> Result<Vec<Value>, NativeError> {
        loop {
            let next_fn = ordinary_get(
                heap,
                iter_obj,
                Wtf8::from_str("next"),
                Value::Object(iter_obj),
                0,
            )?;
            if !is_callable(heap, next_fn) {
                return Err(NativeError::TypeError(
                    "iterator.next is not callable".into(),
                ));
            }
            let result = call_value(heap, next_fn, Value::Object(iter_obj), &[])?;
            let Value::Object(res_obj) = result else {
                return Err(NativeError::TypeError(
                    "iterator.next() did not return an object".into(),
                ));
            };
            let done = ordinary_get(
                heap,
                res_obj,
                Wtf8::from_str("done"),
                Value::Object(res_obj),
                0,
            )?;
            if done.is_truthy() {
                break;
            }
            let v = ordinary_get(
                heap,
                res_obj,
                Wtf8::from_str("value"),
                Value::Object(res_obj),
                0,
            )?;
            heap.protect(v);
            values.push(v);
        }
        Ok(values)
    })();
    match out {
        Ok(values) => Ok(Some((values, base))),
        Err(e) => {
            heap.release_temp(base);
            Err(e)
        }
    }
}

/// Turn an own data [`Property`] into a descriptor object
/// `{value, writable, enumerable, configurable}`.
pub(crate) fn descriptor_object(
    heap: &mut Heap,
    prop: Property,
) -> Result<Gc<Object>, NativeError> {
    let d = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    heap.define_property_str(d, "value", Property::data(prop.value))?;
    heap.define_property_str(d, "writable", Property::data(Value::Bool(prop.writable)))?;
    heap.define_property_str(
        d,
        "enumerable",
        Property::data(Value::Bool(prop.enumerable)),
    )?;
    heap.define_property_str(
        d,
        "configurable",
        Property::data(Value::Bool(prop.configurable)),
    )?;
    Ok(d)
}

/// Fields present on a descriptor object. v1: data descriptors only;
/// `get`/`set` presence is a `TypeError`.
struct DescParts {
    value: Option<Value>,
    writable: Option<bool>,
    enumerable: Option<bool>,
    configurable: Option<bool>,
}

fn read_descriptor(heap: &mut Heap, d: Value) -> Result<DescParts, NativeError> {
    let obj = match d {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "property descriptor must be an object".into(),
            ));
        }
    };
    if get_prop_chain(heap, obj, Wtf8::from_str("get")).is_some()
        || get_prop_chain(heap, obj, Wtf8::from_str("set")).is_some()
    {
        return Err(NativeError::TypeError(
            "accessor descriptors are a v1 gap".into(),
        ));
    }
    let flag = |heap: &Heap, name: &str| -> Option<bool> {
        get_prop_chain(heap, obj, Wtf8::from_str(name)).map(|v| v.is_truthy())
    };
    Ok(DescParts {
        value: get_prop_chain(heap, obj, Wtf8::from_str("value")),
        writable: flag(heap, "writable"),
        enumerable: flag(heap, "enumerable"),
        configurable: flag(heap, "configurable"),
    })
}

/// Apply a descriptor object to an ordinary target. Returns `Ok(false)` on
/// spec-level failure; `Err` only for malformed descriptors.
///
/// Array targets with an exotic key (`"length"` or a canonical index) go
/// through the array exotic `[[DefineOwnProperty]]` (spec 10.4.2):
/// rejected defines (e.g. a shrink blocked by a non-configurable
/// element) return `false`, and a non-Uint32 `length` throws
/// `RangeError`, matching `Reflect.defineProperty`'s contract. Absent
/// descriptor fields stay absent so the exotic completes them against
/// the current property, the same completion `Object.defineProperty`
/// uses. Non-array targets and ordinary keys keep the exact prior
/// behavior.
fn define_from_descriptor(
    heap: &mut Heap,
    obj: Gc<Object>,
    key: &Wtf8,
    desc: Value,
) -> Result<bool, NativeError> {
    let parts = read_descriptor(heap, desc)?;
    if heap.get(obj).kind == ObjectKind::Array
        && (key == "length" || array_index_key(key).is_some())
    {
        return heap.define_array_own_property(
            obj,
            key,
            &ArrayDefineDesc {
                value: parts.value,
                writable: parts.writable,
                enumerable: parts.enumerable,
                configurable: parts.configurable,
                ..Default::default()
            },
        );
    }
    match get_own(heap, obj, key) {
        Some(existing) => {
            let prop = Property {
                value: parts.value.unwrap_or(existing.value),
                writable: parts.writable.unwrap_or(existing.writable),
                enumerable: parts.enumerable.unwrap_or(existing.enumerable),
                configurable: parts.configurable.unwrap_or(existing.configurable),
            };
            define_ordinary(heap, obj, key, prop)
        }
        None => {
            if !heap.get(obj).extensible {
                return Ok(false);
            }
            let prop = Property {
                value: parts.value.unwrap_or(Value::Undefined),
                writable: parts.writable.unwrap_or(false),
                enumerable: parts.enumerable.unwrap_or(false),
                configurable: parts.configurable.unwrap_or(false),
            };
            heap.define_property(obj, key, prop)?;
            Ok(true)
        }
    }
}

/// Require an object argument.
fn require_object(v: Value, name: &str) -> Result<Gc<Object>, NativeError> {
    match v {
        Value::Object(o) => Ok(o),
        _ => Err(NativeError::TypeError(format!(
            "{name}: target must be an object"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Proxy internals
// ---------------------------------------------------------------------------

/// Hidden property names backing proxy exotic objects. NUL-prefixed like
/// every other internal slot in the engine (see `bindings.rs`): ordinary
/// property enumeration never surfaces them, and script code cannot guess
/// them without deliberately writing a NUL into a key.
///
/// Shared `pub(crate)` with `vm.rs`'s trampolined proxy `[[Call]]`
/// protocol ([`crate::vm::Interp::proxy_apply_setup`]): a single source of
/// truth, so the two modules can never disagree on the slot names.
pub(crate) const P_TARGET: &str = "\u{0}bowser:proxy-target";
pub(crate) const P_HANDLER: &str = "\u{0}bowser:proxy-handler";
pub(crate) const P_REVOKED: &str = "\u{0}bowser:proxy-revoked";

/// If `v` is a proxy object, return its handle. Fail closed: both the
/// target and the handler slot must hold objects, so a forged object that
/// merely names one of the hidden keys is treated as an ordinary object.
pub(crate) fn as_proxy(heap: &Heap, v: Value) -> Option<Gc<Object>> {
    match v {
        Value::Object(o) => {
            let target_ok = matches!(
                heap.get_property(o, Wtf8::from_str(P_TARGET))
                    .map(|p| p.value),
                Some(Value::Object(_))
            );
            let handler_ok = matches!(
                heap.get_property(o, Wtf8::from_str(P_HANDLER))
                    .map(|p| p.value),
                Some(Value::Object(_))
            );
            if target_ok && handler_ok {
                Some(o)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn proxy_target(heap: &Heap, proxy: Gc<Object>) -> Option<Gc<Object>> {
    match heap
        .get_property(proxy, Wtf8::from_str(P_TARGET))
        .map(|p| p.value)
    {
        Some(Value::Object(o)) => Some(o),
        _ => None,
    }
}

fn proxy_handler(heap: &Heap, proxy: Gc<Object>) -> Option<Gc<Object>> {
    match heap
        .get_property(proxy, Wtf8::from_str(P_HANDLER))
        .map(|p| p.value)
    {
        Some(Value::Object(o)) => Some(o),
        _ => None,
    }
}

/// Fail-closed error for a proxy whose hidden slots are missing or hold
/// non-objects (forged or corrupted internals): a JS `TypeError`, never a
/// Rust panic.
fn proxy_slot_err() -> NativeError {
    NativeError::TypeError("proxy internal slot is not an object".into())
}

fn ensure_not_revoked(heap: &Heap, proxy: Gc<Object>) -> Result<(), NativeError> {
    let revoked = heap
        .get_property(proxy, Wtf8::from_str(P_REVOKED))
        .is_some_and(|p| p.value.is_truthy());
    if revoked {
        return Err(NativeError::TypeError("proxy has been revoked".into()));
    }
    Ok(())
}

/// `GetMethod(handler, name)`: `undefined` when absent, `TypeError` when
/// present but not callable.
fn get_trap(heap: &Heap, handler: Gc<Object>, name: &str) -> Result<Option<Value>, NativeError> {
    match get_prop_chain(heap, handler, Wtf8::from_str(name)) {
        None | Some(Value::Undefined) | Some(Value::Null) => Ok(None),
        Some(f) => {
            if !is_callable(heap, f) {
                return Err(NativeError::TypeError(format!(
                    "proxy trap '{name}' is not callable"
                )));
            }
            Ok(Some(f))
        }
    }
}

/// `[[Get]]` through a proxy, enforcing 10.5.8 invariants.
fn proxy_get(
    heap: &mut Heap,
    proxy: Gc<Object>,
    key: PropKey<'_>,
    receiver: Value,
    depth: u32,
) -> Result<Value, NativeError> {
    if depth > MAX_PROXY_HOPS {
        return Err(NativeError::TypeError(
            "proxy 'get' delegation too deep".into(),
        ));
    }
    ensure_not_revoked(heap, proxy)?;
    let target = proxy_target(heap, proxy).ok_or_else(proxy_slot_err)?;
    let handler = proxy_handler(heap, proxy).ok_or_else(proxy_slot_err)?;
    let trap = get_trap(heap, handler, "get")?;
    let key_val = prop_key_value(heap, key)?;
    let trap_result = match trap {
        // No trap: the target's own `[[Get]]` with the proxy's receiver —
        // accessors fire, and nested proxies keep their traps.
        None => ordinary_get_key(heap, target, key, receiver, depth)?,
        Some(f) => call_value(
            heap,
            f,
            Value::Object(handler),
            &[Value::Object(target), key_val, receiver],
        )?,
    };
    // Invariant: a non-configurable, non-writable own data property must
    // be reported with its true value.
    if let Some(desc) = get_own_key(heap, target, key)
        && !desc.configurable
        && !desc.writable
        && !same_value(trap_result, desc.value)
    {
        return Err(NativeError::TypeError(format!(
            "proxy 'get' invariant violated for non-configurable, non-writable property '{}'",
            heap.format_key(key)
        )));
    }
    Ok(trap_result)
}

/// `[[Set]]` through a proxy, enforcing 10.5.9 invariants.
pub(crate) fn proxy_set(
    heap: &mut Heap,
    proxy: Gc<Object>,
    key: PropKey<'_>,
    value: Value,
    receiver: Value,
    depth: u32,
) -> Result<bool, NativeError> {
    if depth > MAX_PROXY_HOPS {
        return Err(NativeError::TypeError(
            "proxy 'set' delegation too deep".into(),
        ));
    }
    ensure_not_revoked(heap, proxy)?;
    let target = proxy_target(heap, proxy).ok_or_else(proxy_slot_err)?;
    let handler = proxy_handler(heap, proxy).ok_or_else(proxy_slot_err)?;
    let trap = get_trap(heap, handler, "set")?;
    let key_val = prop_key_value(heap, key)?;
    let trap_result = match trap {
        // No trap: the target's own `[[Set]]` with the proxy's receiver —
        // inherited setters fire with the receiver as `this`, and nested
        // proxies keep their traps.
        None => ordinary_set_key(heap, target, key, value, receiver, depth)?,
        Some(f) => {
            let r = call_value(
                heap,
                f,
                Value::Object(handler),
                &[Value::Object(target), key_val, value, receiver],
            )?;
            if !r.is_truthy() {
                return Ok(false);
            }
            true
        }
    };
    if trap_result
        && let Some(desc) = get_own_key(heap, target, key)
        && !desc.configurable
        && !desc.writable
        && !same_value(value, desc.value)
    {
        return Err(NativeError::TypeError(format!(
            "proxy 'set' invariant violated for non-configurable, non-writable property '{}'",
            heap.format_key(key)
        )));
    }
    Ok(trap_result)
}

/// `[[HasProperty]]` through a proxy, enforcing 10.5.7 invariants.
fn proxy_has(heap: &mut Heap, proxy: Gc<Object>, key: PropKey<'_>) -> Result<bool, NativeError> {
    ensure_not_revoked(heap, proxy)?;
    let target = proxy_target(heap, proxy).ok_or_else(proxy_slot_err)?;
    let handler = proxy_handler(heap, proxy).ok_or_else(proxy_slot_err)?;
    let trap = get_trap(heap, handler, "has")?;
    let key_val = prop_key_value(heap, key)?;
    let trap_result = match trap {
        None => has_prop_chain_key(heap, target, key),
        Some(f) => call_value(
            heap,
            f,
            Value::Object(handler),
            &[Value::Object(target), key_val],
        )?
        .is_truthy(),
    };
    if !trap_result {
        // 10.5.7: a non-configurable own property (data or accessor)
        // may not be reported absent.
        if let Some(false) = own_configurable_key(heap, target, key) {
            return Err(NativeError::TypeError(format!(
                "proxy 'has' invariant violated: non-configurable own property '{}' reported absent",
                heap.format_key(key)
            )));
        }
        if !heap.get(target).extensible {
            return Err(NativeError::TypeError(
                "proxy 'has' invariant violated: property reported absent on non-extensible target"
                    .into(),
            ));
        }
    }
    Ok(trap_result)
}

/// `[[Delete]]` through a proxy, enforcing 10.5.10 invariants.
fn proxy_delete(heap: &mut Heap, proxy: Gc<Object>, key: PropKey<'_>) -> Result<bool, NativeError> {
    ensure_not_revoked(heap, proxy)?;
    let target = proxy_target(heap, proxy).ok_or_else(proxy_slot_err)?;
    let handler = proxy_handler(heap, proxy).ok_or_else(proxy_slot_err)?;
    let trap = get_trap(heap, handler, "deleteProperty")?;
    let key_val = prop_key_value(heap, key)?;
    let trap_result = match trap {
        None => delete_ordinary_key(heap, target, key),
        Some(f) => call_value(
            heap,
            f,
            Value::Object(handler),
            &[Value::Object(target), key_val],
        )?
        .is_truthy(),
    };
    // 10.5.10: a non-configurable own property (data or accessor)
    // may not be reported as deleted.
    if trap_result && let Some(false) = own_configurable_key(heap, target, key) {
        return Err(NativeError::TypeError(format!(
            "proxy 'deleteProperty' invariant violated for non-configurable property '{}'",
            heap.format_key(key)
        )));
    }
    Ok(trap_result)
}

/// `[[OwnPropertyKeys]]` through a proxy, enforcing 10.5.12 invariants.
/// Returns the key list on success.
fn proxy_own_keys(heap: &mut Heap, proxy: Gc<Object>) -> Result<Vec<OwnKey>, NativeError> {
    ensure_not_revoked(heap, proxy)?;
    let target = proxy_target(heap, proxy).ok_or_else(proxy_slot_err)?;
    let handler = proxy_handler(heap, proxy).ok_or_else(proxy_slot_err)?;
    let trap = get_trap(heap, handler, "ownKeys")?;
    let trap_keys: Vec<OwnKey> = match trap {
        None => own_keys_in_order(heap, target),
        Some(f) => {
            let r = call_value(heap, f, Value::Object(handler), &[Value::Object(target)])?;
            let arr = require_object(r, "proxy 'ownKeys' trap")?;
            // Read `length` (own or inherited), then the indexed elements.
            // Each element must be a string or a symbol.
            let len_val =
                get_prop_chain(heap, arr, Wtf8::from_str("length")).unwrap_or(Value::Undefined);
            let len = to_length(len_val)?;
            // The loop below performs no GC-heap allocation, so the
            // symbol handles copied out of `arr` cannot be collected
            // mid-loop.
            let mut keys = Vec::new();
            for i in 0..len {
                let k = array_like_get(heap, arr, i);
                match k {
                    Value::Str(s) => keys.push(OwnKey::Str(s.as_wtf8().to_owned())),
                    Value::Symbol(sym) => keys.push(OwnKey::Sym(sym)),
                    _ => {
                        return Err(NativeError::TypeError(
                            "proxy 'ownKeys' trap must return an array of strings or symbols"
                                .into(),
                        ));
                    }
                }
            }
            // Duplicate check: strings and symbols live in separate
            // namespaces, so a string never duplicates a symbol.
            let mut seen_str = std::collections::HashSet::new();
            let mut seen_sym: Vec<Gc<Symbol>> = Vec::new();
            for k in &keys {
                match k {
                    OwnKey::Str(s) => {
                        if !seen_str.insert(s.clone()) {
                            return Err(NativeError::TypeError(
                                "proxy 'ownKeys' trap returned duplicate keys".into(),
                            ));
                        }
                    }
                    OwnKey::Sym(sym) => {
                        if seen_sym.contains(sym) {
                            return Err(NativeError::TypeError(
                                "proxy 'ownKeys' trap returned duplicate keys".into(),
                            ));
                        }
                        seen_sym.push(*sym);
                    }
                }
            }
            keys
        }
    };
    // Invariants against the target's own keys.
    let target_keys = own_keys_in_order(heap, target);
    let extensible = heap.get(target).extensible;
    for k in &target_keys {
        if let Some(configurable) = own_configurable_key(heap, target, k.as_prop_key())
            && !configurable
            && !trap_keys.contains(k)
        {
            return Err(NativeError::TypeError(format!(
                "proxy 'ownKeys' invariant violated: non-configurable key '{}' missing",
                heap.format_key(k.as_prop_key())
            )));
        }
    }
    if !extensible && !own_key_set_eq(&trap_keys, &target_keys) {
        return Err(NativeError::TypeError(
            "proxy 'ownKeys' invariant violated: keys must exactly match target's on a non-extensible target".into(),
        ));
    }
    Ok(trap_keys)
}

/// `[[GetOwnProperty]]` through a proxy, enforcing 10.5.1 invariants.
fn proxy_get_own_property_descriptor(
    heap: &mut Heap,
    proxy: Gc<Object>,
    key: PropKey<'_>,
) -> Result<Option<Property>, NativeError> {
    ensure_not_revoked(heap, proxy)?;
    let key_name = heap.format_key(key);
    let target = proxy_target(heap, proxy).ok_or_else(proxy_slot_err)?;
    let handler = proxy_handler(heap, proxy).ok_or_else(proxy_slot_err)?;
    let trap = get_trap(heap, handler, "getOwnPropertyDescriptor")?;
    let key_val = prop_key_value(heap, key)?;
    let Some(f) = trap else {
        // No trap: forward to the target. (Data descriptors only; the
        // accessor-target gap on this path predates this change.)
        return Ok(get_own_key(heap, target, key));
    };
    let r = call_value(
        heap,
        f,
        Value::Object(handler),
        &[Value::Object(target), key_val],
    )?;
    let parts: Option<DescParts> = match r {
        Value::Undefined => None,
        Value::Object(d) => Some(read_descriptor(heap, Value::Object(d))?),
        _ => {
            return Err(NativeError::TypeError(
                "proxy 'getOwnPropertyDescriptor' trap must return an object or undefined".into(),
            ));
        }
    };
    // Read the target state after the trap ran: the trap may have
    // mutated the target (spec 10.5.1 step 8).
    let target_data = get_own_key(heap, target, key);
    let target_acc = heap.get_accessor_key(target, key);
    let extensible = heap.get(target).extensible;
    match parts {
        // Spec 10.5.1 step 10: the trap reported the property absent.
        None => {
            if target_data.is_none() && target_acc.is_none() {
                return Ok(None);
            }
            // A non-configurable own property (data or accessor) may not
            // be reported absent.
            if let Some(false) = own_configurable_key(heap, target, key) {
                return Err(NativeError::TypeError(format!(
                    "proxy 'getOwnPropertyDescriptor' invariant violated: non-configurable property '{key_name}' reported absent"
                )));
            }
            if !extensible {
                return Err(NativeError::TypeError(format!(
                    "proxy 'getOwnPropertyDescriptor' invariant violated: own property '{key_name}' of non-extensible target reported absent"
                )));
            }
            Ok(None)
        }
        Some(parts) => {
            // The returned descriptor keeps the fill behavior for the
            // value handed back, but every invariant check below runs on
            // `parts` presence, never on the filled descriptor (spec
            // 10.5.1 steps 13-15: absent fields carry no invariant).
            let filled = Property {
                value: parts
                    .value
                    .or(target_data.map(|p| p.value))
                    .unwrap_or(Value::Undefined),
                writable: parts
                    .writable
                    .or(target_data.map(|p| p.writable))
                    .unwrap_or(false),
                enumerable: parts
                    .enumerable
                    .or(target_data.map(|p| p.enumerable))
                    .unwrap_or(false),
                configurable: parts
                    .configurable
                    .or(target_data.map(|p| p.configurable))
                    .unwrap_or(false),
            };
            // Spec 10.5.1 step 11: a generic descriptor (neither data nor
            // accessor fields present) is returned as-is, no invariants.
            if parts.value.is_none() && parts.writable.is_none() {
                return Ok(Some(filled));
            }
            match (target_data, target_acc) {
                (None, None) => {
                    // Spec 10.5.1 step 13: a new property on a
                    // non-extensible target always throws ...
                    if !extensible {
                        return Err(NativeError::TypeError(format!(
                            "proxy 'getOwnPropertyDescriptor' invariant violated: new property '{key_name}' on non-extensible target"
                        )));
                    }
                    // ... and on an extensible target it must be reported
                    // configurable.
                    if parts.configurable != Some(true) {
                        return Err(NativeError::TypeError(format!(
                            "proxy 'getOwnPropertyDescriptor' invariant violated: non-configurable descriptor for absent target property '{key_name}'"
                        )));
                    }
                }
                (Some(td), _) => {
                    // Spec 10.5.1 step 14: a non-configurable result must
                    // match a non-configurable target.
                    if parts.configurable == Some(false) {
                        if td.configurable {
                            return Err(NativeError::TypeError(format!(
                                "proxy 'getOwnPropertyDescriptor' invariant violated: non-configurable descriptor for configurable target property '{key_name}'"
                            )));
                        }
                        if let Some(e) = parts.enumerable
                            && e != td.enumerable
                        {
                            return Err(NativeError::TypeError(format!(
                                "proxy 'getOwnPropertyDescriptor' invariant violated: enumerable mismatch for non-configurable property '{key_name}'"
                            )));
                        }
                    }
                    // Spec 10.5.1 step 15: both sides are data descriptors
                    // (the trap cannot produce an accessor descriptor;
                    // read_descriptor rejects get/set).
                    if parts.configurable == Some(false) && parts.writable == Some(true) {
                        return Err(NativeError::TypeError(format!(
                            "proxy 'getOwnPropertyDescriptor' invariant violated: non-configurable descriptor reports writable for '{key_name}'"
                        )));
                    }
                    if !td.configurable && !td.writable {
                        if let Some(v) = parts.value
                            && !same_value(v, td.value)
                        {
                            return Err(NativeError::TypeError(format!(
                                "proxy 'getOwnPropertyDescriptor' invariant violated: value mismatch for non-configurable, non-writable property '{key_name}'"
                            )));
                        }
                        if parts.writable == Some(true) {
                            return Err(NativeError::TypeError(format!(
                                "proxy 'getOwnPropertyDescriptor' invariant violated: writable reported for non-configurable, non-writable property '{key_name}'"
                            )));
                        }
                    }
                }
                (None, Some(ta)) => {
                    // Step 14 applies to an accessor targetDesc; step 15
                    // needs both sides data, so it is skipped.
                    if parts.configurable == Some(false) {
                        if ta.configurable {
                            return Err(NativeError::TypeError(format!(
                                "proxy 'getOwnPropertyDescriptor' invariant violated: non-configurable descriptor for configurable accessor property '{key_name}'"
                            )));
                        }
                        if let Some(e) = parts.enumerable
                            && e != ta.enumerable
                        {
                            return Err(NativeError::TypeError(format!(
                                "proxy 'getOwnPropertyDescriptor' invariant violated: enumerable mismatch for non-configurable accessor property '{key_name}'"
                            )));
                        }
                    }
                }
            }
            Ok(Some(filled))
        }
    }
}

/// `[[DefineOwnProperty]]` through a proxy, enforcing 10.5.6 invariants.
pub(crate) fn proxy_define_property(
    heap: &mut Heap,
    proxy: Gc<Object>,
    key: PropKey<'_>,
    desc: Value,
) -> Result<bool, NativeError> {
    ensure_not_revoked(heap, proxy)?;
    let key_name = heap.format_key(key);
    let target = proxy_target(heap, proxy).ok_or_else(proxy_slot_err)?;
    let handler = proxy_handler(heap, proxy).ok_or_else(proxy_slot_err)?;
    let trap = get_trap(heap, handler, "defineProperty")?;
    let key_val = prop_key_value(heap, key)?;
    let trap_result = match trap {
        None => define_from_descriptor_key(heap, target, key, desc)?,
        Some(f) => {
            let r = call_value(
                heap,
                f,
                Value::Object(handler),
                &[Value::Object(target), key_val, desc],
            )?;
            if !r.is_truthy() {
                return Ok(false);
            }
            true
        }
    };
    if trap_result {
        // Spec 10.5.6 step 12: a trap success must be compatible with the
        // target's own property state (data or accessor).
        let parts = read_descriptor(heap, desc)?;
        match (
            get_own_key(heap, target, key),
            heap.get_accessor_key(target, key),
        ) {
            // Step 12.c: adding a new property.
            (None, None) => {
                if !heap.get(target).extensible {
                    return Err(NativeError::TypeError(
                        "proxy 'defineProperty' invariant violated: new property on non-extensible target"
                            .into(),
                    ));
                }
                // Step 12.c.ii: a non-configurable addition is forbidden
                // even on an extensible target.
                if parts.configurable == Some(false) {
                    return Err(NativeError::TypeError(format!(
                        "proxy 'defineProperty' invariant violated: non-configurable addition of '{key_name}'"
                    )));
                }
            }
            // Step 12.d: redefining an existing data property.
            (Some(td), _) => {
                if !td.configurable {
                    if parts.configurable == Some(true) {
                        return Err(NativeError::TypeError(format!(
                            "proxy 'defineProperty' invariant violated for non-configurable property '{key_name}'"
                        )));
                    }
                    if let Some(e) = parts.enumerable
                        && e != td.enumerable
                    {
                        return Err(NativeError::TypeError(format!(
                            "proxy 'defineProperty' invariant violated for non-configurable property '{key_name}'"
                        )));
                    }
                    if !td.writable
                        && (parts.writable == Some(true)
                            || parts.value.is_some_and(|v| !same_value(v, td.value)))
                    {
                        return Err(NativeError::TypeError(format!(
                            "proxy 'defineProperty' invariant violated for non-configurable, non-writable property '{key_name}'"
                        )));
                    }
                }
            }
            // Step 12.d.i for an accessor targetDesc. Step 12.d.ii needs
            // both sides data; Desc is data-or-generic here because
            // read_descriptor rejects get/set.
            (None, Some(ta)) => {
                if !ta.configurable {
                    if parts.configurable == Some(true) {
                        return Err(NativeError::TypeError(format!(
                            "proxy 'defineProperty' invariant violated for non-configurable accessor property '{key_name}'"
                        )));
                    }
                    if let Some(e) = parts.enumerable
                        && e != ta.enumerable
                    {
                        return Err(NativeError::TypeError(format!(
                            "proxy 'defineProperty' invariant violated for non-configurable accessor property '{key_name}'"
                        )));
                    }
                }
            }
        }
    }
    Ok(trap_result)
}

/// `[[GetPrototypeOf]]` through a proxy (10.5.2).
fn proxy_get_prototype_of(heap: &mut Heap, proxy: Gc<Object>) -> Result<Value, NativeError> {
    ensure_not_revoked(heap, proxy)?;
    let target = proxy_target(heap, proxy).ok_or_else(proxy_slot_err)?;
    let handler = proxy_handler(heap, proxy).ok_or_else(proxy_slot_err)?;
    let trap = get_trap(heap, handler, "getPrototypeOf")?;
    let target_proto = heap.get(target).prototype;
    match trap {
        None => Ok(target_proto),
        Some(f) => {
            let r = call_value(heap, f, Value::Object(handler), &[Value::Object(target)])?;
            match r {
                Value::Object(_) | Value::Null => {}
                _ => {
                    return Err(NativeError::TypeError(
                        "proxy 'getPrototypeOf' trap must return an object or null".into(),
                    ));
                }
            }
            if !heap.get(target).extensible && !same_value(r, target_proto) {
                return Err(NativeError::TypeError(
                    "proxy 'getPrototypeOf' invariant violated on non-extensible target".into(),
                ));
            }
            Ok(r)
        }
    }
}

/// `[[SetPrototypeOf]]` through a proxy (10.5.3).
fn proxy_set_prototype_of(
    heap: &mut Heap,
    proxy: Gc<Object>,
    proto: Value,
) -> Result<bool, NativeError> {
    ensure_not_revoked(heap, proxy)?;
    match proto {
        Value::Object(_) | Value::Null => {}
        _ => {
            return Err(NativeError::TypeError(
                "prototype must be an object or null".into(),
            ));
        }
    }
    let target = proxy_target(heap, proxy).ok_or_else(proxy_slot_err)?;
    let handler = proxy_handler(heap, proxy).ok_or_else(proxy_slot_err)?;
    let trap = get_trap(heap, handler, "setPrototypeOf")?;
    let trap_result = match trap {
        None => {
            // No trap: forward to the target's own `[[SetPrototypeOf]]`,
            // which rejects prototype cycles (spec 10.5.3).
            if check_prototype_cycle(heap, target, proto).is_err() {
                return Ok(false);
            }
            heap.set_prototype(target, proto);
            true
        }
        Some(f) => call_value(
            heap,
            f,
            Value::Object(handler),
            &[Value::Object(target), proto],
        )?
        .is_truthy(),
    };
    if trap_result && !heap.get(target).extensible {
        let target_proto = heap.get(target).prototype;
        if !same_value(proto, target_proto) {
            return Err(NativeError::TypeError(
                "proxy 'setPrototypeOf' invariant violated on non-extensible target".into(),
            ));
        }
    }
    Ok(trap_result)
}

/// `[[IsExtensible]]` through a proxy (10.5.4).
fn proxy_is_extensible(heap: &mut Heap, proxy: Gc<Object>) -> Result<bool, NativeError> {
    ensure_not_revoked(heap, proxy)?;
    let target = proxy_target(heap, proxy).ok_or_else(proxy_slot_err)?;
    let handler = proxy_handler(heap, proxy).ok_or_else(proxy_slot_err)?;
    let trap = get_trap(heap, handler, "isExtensible")?;
    let target_extensible = heap.get(target).extensible;
    match trap {
        None => Ok(target_extensible),
        Some(f) => {
            let r =
                call_value(heap, f, Value::Object(handler), &[Value::Object(target)])?.is_truthy();
            if r != target_extensible {
                return Err(NativeError::TypeError(
                    "proxy 'isExtensible' invariant violated: trap result must match target".into(),
                ));
            }
            Ok(r)
        }
    }
}

/// `[[PreventExtensions]]` through a proxy (10.5.5).
fn proxy_prevent_extensions(heap: &mut Heap, proxy: Gc<Object>) -> Result<bool, NativeError> {
    ensure_not_revoked(heap, proxy)?;
    let target = proxy_target(heap, proxy).ok_or_else(proxy_slot_err)?;
    let handler = proxy_handler(heap, proxy).ok_or_else(proxy_slot_err)?;
    let trap = get_trap(heap, handler, "preventExtensions")?;
    let trap_result = match trap {
        None => {
            heap.prevent_extensions(target);
            true
        }
        Some(f) => {
            call_value(heap, f, Value::Object(handler), &[Value::Object(target)])?.is_truthy()
        }
    };
    if trap_result && heap.get(target).extensible {
        return Err(NativeError::TypeError(
            "proxy 'preventExtensions' invariant violated: target still extensible".into(),
        ));
    }
    Ok(trap_result)
}

/// `[[Construct]]` through a proxy (10.5.13).
pub(crate) fn proxy_construct(
    heap: &mut Heap,
    proxy: Gc<Object>,
    new_target: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    ensure_not_revoked(heap, proxy)?;
    let target = proxy_target(heap, proxy).ok_or_else(proxy_slot_err)?;
    let handler = proxy_handler(heap, proxy).ok_or_else(proxy_slot_err)?;
    let trap = get_trap(heap, handler, "construct")?;
    match trap {
        None => reflect_construct_inner(heap, Value::Object(target), new_target, args),
        Some(f) => {
            let arg_array = new_value_array(heap, args)?;
            call_value(
                heap,
                f,
                Value::Object(handler),
                &[Value::Object(target), Value::Object(arg_array)],
            )
        }
    }
}

/// `[[Call]]` through a proxy (10.5.14).
pub(crate) fn proxy_apply(
    heap: &mut Heap,
    proxy: Gc<Object>,
    this_arg: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    ensure_not_revoked(heap, proxy)?;
    let target = proxy_target(heap, proxy).ok_or_else(proxy_slot_err)?;
    let handler = proxy_handler(heap, proxy).ok_or_else(proxy_slot_err)?;
    let trap = get_trap(heap, handler, "apply")?;
    match trap {
        None => reflect_apply_inner(heap, Value::Object(target), this_arg, args),
        Some(f) => {
            let arg_array = new_value_array(heap, args)?;
            call_value(
                heap,
                f,
                Value::Object(handler),
                &[Value::Object(target), this_arg, Value::Object(arg_array)],
            )
        }
    }
}

/// Allocate an `Array`-kind object holding the given values.
///
/// Elements go through dense storage and are materialized once, so batch
/// construction is O(n) instead of O(n²).
fn new_value_array(heap: &mut Heap, elems: &[Value]) -> Result<Gc<Object>, NativeError> {
    let mut base = None::<usize>;
    for &e in elems {
        base.get_or_insert(heap.protect(e));
    }
    let arr = heap.new_dense_array()?;
    if let Some(b) = base {
        heap.release_temp(b);
    }
    for &e in elems {
        heap.dense_push(arr, e)?;
    }
    heap.materialize_dense(arr)?;
    Ok(arr)
}

// ---------------------------------------------------------------------------
// Reflect internals (proxy-aware ordinary operations)
// ---------------------------------------------------------------------------

fn reflect_apply_inner(
    heap: &mut Heap,
    target: Value,
    this_arg: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    if !is_callable_deep(heap, target) {
        return Err(NativeError::TypeError(
            "Reflect.apply: target is not callable".into(),
        ));
    }
    match as_proxy(heap, target) {
        Some(proxy) => proxy_apply(heap, proxy, this_arg, args),
        None => call_value(heap, target, this_arg, args),
    }
}

pub(crate) fn reflect_construct_inner(
    heap: &mut Heap,
    target: Value,
    new_target: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let target_obj = require_object(target, "Reflect.construct")?;
    // ECMA-262 `Reflect.construct` step 2 / `Construct` precondition:
    // `IsConstructor`, not mere callability. A callable-but-not-
    // constructible native (e.g. `Array.isArray`) throws `TypeError`
    // here; non-constructible bytecode functions (arrows, methods,
    // async functions, generators) pass this gate and throw `TypeError`
    // from the VM's construct path below.
    if !is_constructor(heap, target) {
        return Err(NativeError::TypeError(
            "Reflect.construct: target is not a constructor".into(),
        ));
    }
    // ECMA-262 `Reflect.construct` step 3: `IsConstructor(newTarget)`.
    // This must be the *strict* check, not `is_constructor`: that gate
    // deliberately reports every bytecode function as a constructor and
    // relies on the VM's construct path to reject arrows/methods/async
    // functions/generators *as targets*, but a `newTarget` is never
    // itself constructed — the native-target arm below would otherwise
    // silently accept `Reflect.construct(Array, [], () => {})`.
    if !is_new_target_constructible(heap, new_target) {
        return Err(NativeError::TypeError(
            "Reflect.construct: newTarget is not a constructor".into(),
        ));
    }
    if let Some(proxy) = as_proxy(heap, target) {
        return proxy_construct(heap, proxy, new_target, args);
    }
    // Native targets construct through the v1 ordinary-object path;
    // bytecode targets re-enter the VM with the explicit `newTarget`.
    // The new `this` for the native path is an ordinary object whose
    // prototype is `newTarget.prototype` when that is an object.
    match heap.get(target_obj).kind {
        ObjectKind::Function(_) => {
            crate::vm::reenter_construct_with_target(heap, target, new_target, args)
        }
        ObjectKind::NativeFunction(f) => {
            let nt_obj = match new_target {
                Value::Object(o) => o,
                _ => target_obj,
            };
            let proto =
                get_prop_chain(heap, nt_obj, Wtf8::from_str("prototype")).unwrap_or(Value::Null);
            let proto = match proto {
                Value::Object(_) => proto,
                _ => Value::Null,
            };
            let this_obj = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
            heap.set_prototype(this_obj, proto);
            // Natives that allocate their own instance (e.g.
            // `object_constructor`, `array_constructor`) observe
            // construction through `active_new_target`, so stash the
            // construction's newTarget exactly as
            // `Interp::invoke_inner` does on the VM's own construct
            // path; save/restore nests with any outer construction.
            let prev = crate::vm::set_new_target(Some(new_target));
            let result = f(heap, Value::Object(this_obj), args);
            crate::vm::set_new_target(prev);
            match result? {
                Value::Object(o) => Ok(Value::Object(o)),
                _ => Ok(Value::Object(this_obj)),
            }
        }
        _ => Err(NativeError::TypeError(
            "Reflect.construct: target is not a constructor".into(),
        )),
    }
}

fn reflect_get_inner(
    heap: &mut Heap,
    target: Value,
    key: PropKey<'_>,
    receiver: Value,
) -> Result<Value, NativeError> {
    let target_obj = require_object(target, "Reflect.get")?;
    match as_proxy(heap, target) {
        Some(proxy) => proxy_get(heap, proxy, key, receiver, 0),
        None => ordinary_get_key(heap, target_obj, key, receiver, 0),
    }
}

fn reflect_set_inner(
    heap: &mut Heap,
    target: Value,
    key: PropKey<'_>,
    value: Value,
    receiver: Value,
) -> Result<bool, NativeError> {
    let target_obj = require_object(target, "Reflect.set")?;
    match as_proxy(heap, target) {
        Some(proxy) => proxy_set(heap, proxy, key, value, receiver, 0),
        None => ordinary_set_key(heap, target_obj, key, value, receiver, 0),
    }
}

fn reflect_has_inner(
    heap: &mut Heap,
    target: Value,
    key: PropKey<'_>,
) -> Result<bool, NativeError> {
    let target_obj = require_object(target, "Reflect.has")?;
    match as_proxy(heap, target) {
        Some(proxy) => proxy_has(heap, proxy, key),
        None => Ok(has_prop_chain_key(heap, target_obj, key)),
    }
}

fn reflect_delete_inner(
    heap: &mut Heap,
    target: Value,
    key: PropKey<'_>,
) -> Result<bool, NativeError> {
    let target_obj = require_object(target, "Reflect.deleteProperty")?;
    match as_proxy(heap, target) {
        Some(proxy) => proxy_delete(heap, proxy, key),
        None => Ok(delete_ordinary_key(heap, target_obj, key)),
    }
}

fn reflect_own_keys_inner(heap: &mut Heap, target: Value) -> Result<Gc<Object>, NativeError> {
    let target_obj = require_object(target, "Reflect.ownKeys")?;
    let keys = match as_proxy(heap, target) {
        Some(proxy) => proxy_own_keys(heap, proxy)?,
        None => own_keys_in_order(heap, target_obj),
    };
    new_key_array(heap, &keys)
}

fn reflect_get_own_property_descriptor_inner(
    heap: &mut Heap,
    target: Value,
    key: PropKey<'_>,
) -> Result<Value, NativeError> {
    let target_obj = require_object(target, "Reflect.getOwnPropertyDescriptor")?;
    let desc = match as_proxy(heap, target) {
        Some(proxy) => proxy_get_own_property_descriptor(heap, proxy, key)?,
        None => get_own_key(heap, target_obj, key),
    };
    match desc {
        Some(p) => Ok(Value::Object(descriptor_object(heap, p)?)),
        None => Ok(Value::Undefined),
    }
}

fn reflect_define_property_inner(
    heap: &mut Heap,
    target: Value,
    key: PropKey<'_>,
    desc: Value,
) -> Result<bool, NativeError> {
    let target_obj = require_object(target, "Reflect.defineProperty")?;
    match as_proxy(heap, target) {
        Some(proxy) => proxy_define_property(heap, proxy, key, desc),
        None => define_from_descriptor_key(heap, target_obj, key, desc),
    }
}

fn reflect_get_prototype_of_inner(heap: &mut Heap, target: Value) -> Result<Value, NativeError> {
    let target_obj = require_object(target, "Reflect.getPrototypeOf")?;
    match as_proxy(heap, target) {
        Some(proxy) => proxy_get_prototype_of(heap, proxy),
        None => Ok(heap.get(target_obj).prototype),
    }
}

fn reflect_set_prototype_of_inner(
    heap: &mut Heap,
    target: Value,
    proto: Value,
) -> Result<bool, NativeError> {
    let target_obj = require_object(target, "Reflect.setPrototypeOf")?;
    match proto {
        Value::Object(_) | Value::Null => {}
        _ => {
            return Err(NativeError::TypeError(
                "Reflect.setPrototypeOf: prototype must be an object or null".into(),
            ));
        }
    }
    match as_proxy(heap, target) {
        Some(proxy) => proxy_set_prototype_of(heap, proxy, proto),
        None => {
            // Unlike `Object.setPrototypeOf` (which throws),
            // `Reflect.setPrototypeOf` reports failure as `false`.
            if check_prototype_cycle(heap, target_obj, proto).is_err() {
                return Ok(false);
            }
            heap.set_prototype(target_obj, proto);
            Ok(true)
        }
    }
}

fn reflect_is_extensible_inner(heap: &mut Heap, target: Value) -> Result<bool, NativeError> {
    let target_obj = require_object(target, "Reflect.isExtensible")?;
    match as_proxy(heap, target) {
        Some(proxy) => proxy_is_extensible(heap, proxy),
        None => Ok(heap.get(target_obj).extensible),
    }
}

fn reflect_prevent_extensions_inner(heap: &mut Heap, target: Value) -> Result<bool, NativeError> {
    let target_obj = require_object(target, "Reflect.preventExtensions")?;
    match as_proxy(heap, target) {
        Some(proxy) => proxy_prevent_extensions(heap, proxy),
        None => {
            heap.prevent_extensions(target_obj);
            Ok(true)
        }
    }
}

// ---------------------------------------------------------------------------
// Reflect.* native functions
// ---------------------------------------------------------------------------

/// Read the arguments array of `Reflect.apply` into a `Vec`.
fn apply_arg_list(heap: &Heap, v: Value) -> Result<Vec<Value>, NativeError> {
    let arr = require_object(v, "Reflect.apply")?;
    let len =
        to_length(get_prop_chain(heap, arr, Wtf8::from_str("length")).unwrap_or(Value::Undefined))?;
    let mut out = Vec::new();
    for i in 0..len {
        out.push(array_like_get(heap, arr, i));
    }
    Ok(out)
}

fn reflect_apply(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let target = get_arg(args, 0);
    let this_arg = get_arg(args, 1);
    let arg_list = apply_arg_list(heap, get_arg(args, 2))?;
    reflect_apply_inner(heap, target, this_arg, &arg_list)
}

fn reflect_construct(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let target = get_arg(args, 0);
    let arg_list = apply_arg_list(heap, get_arg(args, 1))?;
    // `newTarget` defaults to `target`; it must be a constructor.
    let new_target = match get_arg(args, 2) {
        Value::Undefined => target,
        nt => nt,
    };
    // The `IsConstructor(newTarget)` gate (ECMA-262 step 3) lives in
    // `reflect_construct_inner`, which enforces it strictly (arrows and
    // friends rejected via the VM's template metadata) on every path —
    // direct, proxy-without-`construct`-trap, and species.
    reflect_construct_inner(heap, target, new_target, &arg_list)
}

fn reflect_define_property(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let target = get_arg(args, 0);
    let key = to_property_key(heap, get_arg(args, 1))?;
    let desc = get_arg(args, 2);
    Ok(Value::Bool(reflect_define_property_inner(
        heap,
        target,
        key.as_prop_key(),
        desc,
    )?))
}

fn reflect_delete_property(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let target = get_arg(args, 0);
    let key = to_property_key(heap, get_arg(args, 1))?;
    Ok(Value::Bool(reflect_delete_inner(
        heap,
        target,
        key.as_prop_key(),
    )?))
}

fn reflect_get(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let target = get_arg(args, 0);
    let key = to_property_key(heap, get_arg(args, 1))?;
    let receiver = get_arg(args, 2);
    let receiver = if receiver.is_undefined() {
        target
    } else {
        receiver
    };
    reflect_get_inner(heap, target, key.as_prop_key(), receiver)
}

fn reflect_get_own_property_descriptor(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let target = get_arg(args, 0);
    let key = to_property_key(heap, get_arg(args, 1))?;
    reflect_get_own_property_descriptor_inner(heap, target, key.as_prop_key())
}

fn reflect_get_prototype_of(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    reflect_get_prototype_of_inner(heap, get_arg(args, 0))
}

fn reflect_has(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let target = get_arg(args, 0);
    let key = to_property_key(heap, get_arg(args, 1))?;
    Ok(Value::Bool(reflect_has_inner(
        heap,
        target,
        key.as_prop_key(),
    )?))
}

fn reflect_is_extensible(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    Ok(Value::Bool(reflect_is_extensible_inner(
        heap,
        get_arg(args, 0),
    )?))
}

fn reflect_own_keys(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(Value::Object(reflect_own_keys_inner(
        heap,
        get_arg(args, 0),
    )?))
}

fn reflect_prevent_extensions(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    Ok(Value::Bool(reflect_prevent_extensions_inner(
        heap,
        get_arg(args, 0),
    )?))
}

fn reflect_set(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let target = get_arg(args, 0);
    let key = to_property_key(heap, get_arg(args, 1))?;
    let value = get_arg(args, 2);
    // `Reflect.set(target, key, value, receiver)`: the receiver defaults
    // to the target and is what setters observe as `this`.
    let receiver = get_arg(args, 3);
    let receiver = if receiver.is_undefined() {
        target
    } else {
        receiver
    };
    Ok(Value::Bool(reflect_set_inner(
        heap,
        target,
        key.as_prop_key(),
        value,
        receiver,
    )?))
}

fn reflect_set_prototype_of(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let target = get_arg(args, 0);
    let proto = get_arg(args, 1);
    Ok(Value::Bool(reflect_set_prototype_of_inner(
        heap, target, proto,
    )?))
}

// ---------------------------------------------------------------------------
// Proxy constructor + Proxy.revocable
// ---------------------------------------------------------------------------

fn proxy_constructor(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let target = require_object(get_arg(args, 0), "Proxy")?;
    let handler = require_object(get_arg(args, 1), "Proxy")?;
    let proxy = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    define_hidden_const(heap, proxy, P_TARGET, Value::Object(target))?;
    define_hidden_const(heap, proxy, P_HANDLER, Value::Object(handler))?;
    define_hidden(heap, proxy, P_REVOKED, Value::Bool(false))?;
    Ok(Value::Object(proxy))
}

/// Hidden slots backing the bound `revoke` function built by
/// `proxy_revocable`. A `NativeFn` cannot observe its callee, so the proxy
/// is captured as the function's bound `this` at creation — the
/// `[[RevocableProxy]]` slot, in effect. The `INVOKE_BOUND` interception in
/// the VM and in `call_function` substitutes it before `proxy_revoke` runs,
/// whatever the call-site receiver is. The strings must match
/// `vm::bound_parts` / `Function.prototype.bind` exactly; this module stays
/// self-contained per its header docs, so they are copied here rather than
/// imported.
const BOUND_TARGET: &str = "\u{0}bowser:bound-target";
const BOUND_THIS: &str = "\u{0}bowser:bound-this";
const BOUND_ARGS: &str = "\u{0}bowser:bound-args";

/// Unreachable fallback for the bound `revoke` wrapper's kind: the
/// `INVOKE_BOUND` interception runs before kind dispatch on every call
/// path, so reaching this is a bug, not a slow path.
fn bound_revoke_unreachable(
    _heap: &mut Heap,
    _this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    Err(NativeError::TypeError(
        "bound revoke dispatched without interception".into(),
    ))
}

fn proxy_revoke(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    // `this` is the proxy captured as the bound `this` at creation (see
    // `proxy_revocable`): the bound hop runs before this body on every
    // call path, so the call-site receiver is irrelevant — detached calls
    // and `Reflect.apply(revoke, otherReceiver, [])` revoke this proxy.
    let proxy = require_object(this, "revoke")?;
    heap.define_property_str(
        proxy,
        P_REVOKED,
        Property {
            value: Value::Bool(true),
            writable: true,
            enumerable: false,
            configurable: false,
        },
    )?;
    Ok(Value::Undefined)
}

fn proxy_revocable(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let proxy_val = proxy_constructor(heap, Value::Undefined, args)?;
    let proxy = require_object(proxy_val, "Proxy.revocable")?;
    // The proxy and the fresh objects below are not yet linked into any
    // traced root; hold them across the allocations below, which may
    // collect. One base covers the whole batch; everything is installed
    // into `result` before the release.
    let base = heap.protect(Value::Object(proxy));
    let target = new_native_function(heap, "revoke", proxy_revoke, 0)?;
    heap.protect(Value::Object(target));
    let bound_args = heap
        .new_dense_array()
        .map_err(|_| NativeError::OutOfMemory)?;
    heap.protect(Value::Object(bound_args));
    let proto = heap
        .function_prototype
        .map(Value::Object)
        .unwrap_or(Value::Null);
    let revoke = heap
        .alloc_object(Object {
            kind: ObjectKind::NativeFunction(bound_revoke_unreachable),
            prototype: proto,
            invoke_flags: Object::INVOKE_BOUND,
            ..Default::default()
        })
        .map_err(|_| NativeError::OutOfMemory)?;
    heap.protect(Value::Object(revoke));
    let hidden = Property {
        value: Value::Null,
        writable: false,
        enumerable: false,
        configurable: false,
    };
    heap.define_property_str(
        revoke,
        BOUND_TARGET,
        Property {
            value: Value::Object(target),
            ..hidden
        },
    )?;
    heap.define_property_str(
        revoke,
        BOUND_THIS,
        Property {
            value: Value::Object(proxy),
            ..hidden
        },
    )?;
    heap.define_property_str(
        revoke,
        BOUND_ARGS,
        Property {
            value: Value::Object(bound_args),
            ..hidden
        },
    )?;
    // Spec-shaped bound name/length: `"bound revoke"`, length 0.
    let name_val = with_temp_root(heap, Value::Object(revoke), |heap| {
        heap.alloc_str("bound revoke")
            .map_err(|_| NativeError::OutOfMemory)
    })?;
    heap.define_property_str(
        revoke,
        "name",
        Property {
            value: Value::Str(name_val),
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )?;
    heap.define_property_str(
        revoke,
        "length",
        Property {
            value: js_num(0.0),
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )?;
    let result = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    heap.define_property_str(result, "proxy", Property::data(Value::Object(proxy)))?;
    heap.define_property_str(result, "revoke", Property::data(Value::Object(revoke)))?;
    // Installed: `result` now keeps the proxy and revoke function alive.
    heap.release_temp(base);
    Ok(Value::Object(result))
}

// ---------------------------------------------------------------------------
// ArrayBuffer / SharedArrayBuffer
//
// v1 storage: the buffer's bytes live in a single zeroed `Box<[u8]>` on
// the buffer object (`ObjectCold::bytes`), not as one property per byte.
// Views share the buffer object, so writes through one view are visible
// through every other view. `new ArrayBuffer(200_000_000)` is one 200 MB
// allocation, not 200 M hash inserts.
// ---------------------------------------------------------------------------

/// True when `v` is an ArrayBuffer or SharedArrayBuffer object: the
/// unforgeable [`ObjectKind::ArrayBuffer`] tag (replacing the old hidden
/// `"\u{0}bowser:arraybuffer"` marker, which script could forge).
fn is_array_buffer(heap: &Heap, v: Value) -> bool {
    matches!(
        v,
        Value::Object(o) if matches!(heap.get(o).kind, ObjectKind::ArrayBuffer)
    )
}

/// True when `buf` is a `SharedArrayBuffer`: same
/// [`ObjectKind::ArrayBuffer`] tag as a plain `ArrayBuffer`, told apart
/// by the [`ObjectCold::ab_shared`][crate::value::ObjectCold] flag the
/// `SharedArrayBuffer` constructor sets. `IsDetachedBuffer` is always
/// false for shared buffers (they can never detach), and several
/// `ArrayBuffer.prototype` methods throw `TypeError` on them.
fn is_shared_array_buffer(heap: &Heap, buf: Gc<Object>) -> bool {
    let o = heap.get(buf);
    matches!(o.kind, ObjectKind::ArrayBuffer) && o.ab_shared()
}

fn array_buffer_length(heap: &Heap, buf: Gc<Object>) -> u64 {
    heap.object_bytes(buf).map(|b| b.len() as u64).unwrap_or(0)
}

fn buffer_get_byte(heap: &Heap, buf: Gc<Object>, i: u64) -> u8 {
    heap.object_bytes(buf)
        .and_then(|b| b.get(i as usize))
        .copied()
        .unwrap_or(0)
}

fn buffer_set_byte(heap: &mut Heap, buf: Gc<Object>, i: u64, b: u8) {
    if let Some(bytes) = heap.object_bytes_mut(buf)
        && let Some(slot) = bytes.get_mut(i as usize)
    {
        *slot = b;
        // Out-of-range writes are ignored (spec: silent on views).
    }
}

/// True when `buf` is a detached `ArrayBuffer`: the
/// [`ObjectKind::ArrayBuffer`] tag with no backing store. A zero-length
/// buffer still carries `Some([])`, so `None` is unambiguous. One tag
/// read plus one pointer deref — cheap enough for the element-access
/// hot paths.
fn is_detached_buffer(heap: &Heap, buf: Gc<Object>) -> bool {
    let o = heap.get(buf);
    matches!(o.kind, ObjectKind::ArrayBuffer) && o.bytes().is_none()
}

/// Engine detach op behind `$262.detachArrayBuffer` (test262 harness):
/// drop the backing store and rewrite the `byteLength` own data
/// property to 0 (the property is non-writable/non-configurable to
/// script, but this unvalidated internal define is the one sanctioned
/// writer). Views observe the detach lazily — every view accessor
/// consults [`is_detached_buffer`] on the buffer at read time — so no
/// view registry is needed. Idempotent: detaching twice is a no-op.
fn detach_array_buffer_op(heap: &mut Heap, buf: Gc<Object>) -> Result<(), NativeError> {
    heap.detach_array_buffer(buf);
    heap.define_property_str(
        buf,
        "byteLength",
        Property {
            value: js_num(0.0),
            writable: false,
            enumerable: false,
            configurable: false,
        },
    )?;
    Ok(())
}

/// `$262.detachArrayBuffer(buffer)`: the test262 harness hook the
/// vendored `harness/detachArrayBuffer.js` prefers when present.
/// A non-`ArrayBuffer` argument throws `TypeError` (mirrors
/// `RequireInternalSlot`); detaching an already-detached buffer is a
/// no-op. Returns `undefined`.
fn test262_detach_array_buffer(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let buf = match get_arg(args, 0) {
        Value::Object(o) if is_array_buffer(heap, Value::Object(o)) => o,
        _ => {
            return Err(NativeError::TypeError(
                "$262.detachArrayBuffer: argument must be an ArrayBuffer".into(),
            ));
        }
    };
    // Shared buffers can never detach (`IsDetachedBuffer` is always false
    // for them): a no-op, so `$262.detachArrayBuffer(sab)` leaves every
    // view over the buffer intact.
    if is_shared_array_buffer(heap, buf) {
        return Ok(Value::Undefined);
    }
    detach_array_buffer_op(heap, buf)?;
    Ok(Value::Undefined)
}

/// Install the test262 `$262` helper object (`$262.detachArrayBuffer`)
/// on the realm's global object.
///
/// Test-harness only: the browser embedder never calls this, so real
/// pages never see `$262`. Public so the test262 runner — an external
/// consumer of the crate — can opt in per realm; the runner installs it
/// in `Evaluator::new` *before* snapshotting baseline globals, so the
/// per-test scrub treats `$262` as a builtin binding rather than test
/// debris.
pub fn install_test262_helpers(vm: &mut crate::vm::Vm) -> Result<(), NativeError> {
    let global = vm
        .get_global("globalThis")
        .expect("install_test262_helpers: globalThis missing");
    let Value::Object(global_obj) = global else {
        return Err(NativeError::TypeError(
            "install_test262_helpers: globalThis is not an object".into(),
        ));
    };
    let heap = vm.heap_mut();
    // Hold the fresh `$262` object across the function allocation, which
    // may trigger a collection; the fresh function is held too, until it
    // is linked into `$262`'s properties below. Releasing the `$262`
    // root before both `define_property_str` calls (each of which can
    // allocate and collect) would leave both objects unrooted mid-install.
    let obj_262 = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    with_temp_root(heap, Value::Object(obj_262), |heap| {
        let detach =
            new_native_function(heap, "detachArrayBuffer", test262_detach_array_buffer, 1)?;
        with_temp_root(heap, Value::Object(detach), |heap| {
            heap.define_property_str(
                obj_262,
                "detachArrayBuffer",
                Property::data(Value::Object(detach)),
            )?;
            heap.define_property_str(global_obj, "$262", Property::data(Value::Object(obj_262)))?;
            Ok(())
        })
    })?;
    Ok(())
}

/// Allocate an ArrayBuffer object of `byte_len` zeroed bytes: a single
/// fallible zeroed allocation, not one property per byte.
///
/// `max_byte_length` is `Some(max)` for a resizable buffer (created
/// with `{ maxByteLength: max }`), `None` for fixed-size.
fn new_array_buffer(
    heap: &mut Heap,
    byte_len: u64,
    max_byte_length: Option<u32>,
) -> Result<Gc<Object>, NativeError> {
    // Spec (CreateByteDataBlock): throw RangeError if the Data Block cannot
    // be created. 4 GiB matches common implementation limits; it also keeps
    // `byte_len as usize` exact on 64-bit.
    if byte_len > u32::MAX as u64 {
        return Err(NativeError::RangeError(
            "ArrayBuffer length exceeds maximum".into(),
        ));
    }
    if let Some(max) = max_byte_length
        && byte_len > u64::from(max)
    {
        return Err(NativeError::RangeError(
            "ArrayBuffer maxByteLength is less than the byte length".into(),
        ));
    }
    // Fallible: `vec![0; n]` would abort the process on OOM instead of
    // reporting `AllocError`.
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(byte_len as usize)
        .map_err(|_| NativeError::OutOfMemory)?;
    bytes.resize(byte_len as usize, 0);
    let mut obj = Object::new(ObjectKind::ArrayBuffer);
    obj.set_bytes(Some(bytes.into_boxed_slice()));
    obj.set_ab_max_byte_length(max_byte_length);
    // Link backing stores to `%ArrayBuffer.prototype%` so
    // `buf instanceof ArrayBuffer` holds (spec: AllocateArrayBuffer).
    if let Some(ab_proto) = heap.arraybuffer_prototype {
        obj.prototype = Value::Object(ab_proto);
    }
    let buf = heap.alloc_object(obj)?;
    heap.define_property_str(
        buf,
        "byteLength",
        Property {
            value: js_num(byte_len as f64),
            writable: false,
            enumerable: false,
            configurable: false,
        },
    )?;
    Ok(buf)
}

/// Parse the `ArrayBuffer` constructor's options bag (`{
/// maxByteLength }`, ECMA-262 25.1.3.1 step 2): `Some(max)` makes a
/// resizable buffer, `None` (or absent/`undefined` options) a fixed-size
/// one. `null` options throw `TypeError` (GetOptionsObject); a
/// `maxByteLength` below `byte_len` throws `RangeError`. The
/// `maxByteLength` read is a spec `Get` (getters run) and the index
/// conversion is a full `ToIndex` (objects go through `ToPrimitive`).
fn parse_max_byte_length(
    heap: &mut Heap,
    options: Value,
    byte_len: u64,
) -> Result<Option<u32>, NativeError> {
    if options.is_undefined() {
        return Ok(None);
    }
    if options.is_null() {
        return Err(NativeError::TypeError(
            "ArrayBuffer: options must be an object".into(),
        ));
    }
    let Value::Object(opts) = options else {
        // GetArrayBufferMaxByteLengthOption step 1: not an object → empty.
        return Ok(None);
    };
    let mbl = ordinary_get(
        heap,
        opts,
        Wtf8::from_str("maxByteLength"),
        Value::Object(opts),
        0,
    )?;
    if mbl.is_undefined() {
        return Ok(None);
    }
    let max = to_index_full(heap, mbl)?;
    if max > u64::from(u32::MAX) {
        return Err(NativeError::RangeError(
            "ArrayBuffer maxByteLength exceeds maximum".into(),
        ));
    }
    if max < byte_len {
        return Err(NativeError::RangeError(
            "ArrayBuffer maxByteLength is less than the byte length".into(),
        ));
    }
    Ok(Some(max as u32))
}

fn arraybuffer_constructor(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    // Full `ToIndex`: a `valueOf` hook runs (`new ArrayBuffer({
    // valueOf: () => 8 })` yields 8 bytes, not 0).
    let byte_len = to_index_full(heap, get_arg(args, 0))?;
    let max_byte_length = parse_max_byte_length(heap, get_arg(args, 1), byte_len)?;
    let buf = new_array_buffer(heap, byte_len, max_byte_length)?;
    // ECMA-262 `AllocateArrayBuffer(newTarget)`: a derived
    // `class D extends ArrayBuffer` calling `super()` must get
    // `D.prototype` as the instance's prototype (same pattern as
    // `array_constructor`); without this, species construction through
    // `ArrayBuffer.prototype.slice` could never yield a genuine
    // subclass instance.
    //
    // `buf` is freshly allocated and not yet reachable from any root:
    // hold it across the `prototype` lookup and `set_prototype` below.
    // The lookup itself is side-effect-free today (`get_prop_chain`
    // performs raw property reads), but any future allocating or
    // user-code-invoking step inserted here would collect the unrooted
    // buffer out from under the `set_prototype`.
    with_temp_root(heap, Value::Object(buf), |heap| {
        if let Some(Value::Object(nt)) = crate::vm::active_new_target()
            && let Some(Value::Object(p)) = get_prop_chain(heap, nt, Wtf8::from_str("prototype"))
        {
            heap.set_prototype(buf, Value::Object(p));
        }
        Ok::<(), NativeError>(())
    })?;
    Ok(Value::Object(buf))
}

/// `SharedArrayBuffer` v1: aliases the `ArrayBuffer` implementation.
/// Documented gap: no actual shared-memory semantics. The result is
/// flagged shared ([`is_shared_array_buffer`]): shared buffers can never
/// detach, and `transfer` / `slice` / `resize` plus the `resizable` /
/// `maxByteLength` / `detached` getters throw `TypeError` on them.
fn sharedarraybuffer_constructor(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let Value::Object(buf) = arraybuffer_constructor(heap, _this, args)? else {
        return Err(NativeError::TypeError(
            "SharedArrayBuffer: construction failed".into(),
        ));
    };
    heap.mark_shared_array_buffer(buf);
    Ok(Value::Object(buf))
}

fn arraybuffer_isview(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(Value::Bool(
        is_typed_array(heap, get_arg(args, 0)) || is_data_view(heap, get_arg(args, 0)),
    ))
}

fn arraybuffer_slice(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    const METHOD: &str = "ArrayBuffer.prototype.slice";
    let buf = match this {
        Value::Object(o) if is_array_buffer(heap, Value::Object(o)) => o,
        _ => {
            return Err(NativeError::TypeError(format!(
                "{METHOD} called on incompatible receiver"
            )));
        }
    };
    if is_detached_buffer(heap, buf) {
        return Err(NativeError::TypeError(format!(
            "{METHOD} called on detached ArrayBuffer"
        )));
    }
    if is_shared_array_buffer(heap, buf) {
        return Err(NativeError::TypeError(format!(
            "{METHOD} called on a SharedArrayBuffer"
        )));
    }
    let len = array_buffer_length(heap, buf);
    let rel = |n: f64| -> u64 {
        if n < 0.0 {
            (len as f64 + n).max(0.0) as u64
        } else {
            n.min(len as f64) as u64
        }
    };
    // Full `ToIntegerOrInfinity`: `valueOf`/`toString` hooks run
    // (test262 `slice/number-conversion.js`).
    let from = rel(to_integer_or_infinity_full(heap, get_arg(args, 0))?);
    let end_arg = get_arg(args, 1);
    let to = if end_arg.is_undefined() {
        len
    } else {
        rel(to_integer_or_infinity_full(heap, end_arg)?)
    };
    let (from, to) = if from > to { (to, to) } else { (from, to) };
    let new_len = to - from;
    // ECMA-262 25.1.5.3: `Let C be ? SpeciesConstructor(O,
    // %ArrayBuffer%).` A non-object `O.constructor` or a
    // non-constructor `@@species` throws `TypeError` here — the
    // species lookup used to be skipped entirely, so hostile
    // constructors never errored.
    let default = heap.arraybuffer_ctor.map(Value::Object);
    let ctor = species_constructor(heap, buf, default, METHOD)?;
    // `Let new be ? Construct(C, « newLen »).` Slicing always yields
    // a fixed-size buffer: the spec constructs the result with a
    // single length argument (no options bag), so a resizable source
    // does *not* propagate its ceiling.
    let new = reflect_construct_inner(heap, ctor, ctor, &[js_num(new_len as f64)])?;
    // `RequireInternalSlot(new, [[ArrayBufferData]])`: the species
    // constructor must return a genuine buffer.
    if !is_array_buffer(heap, new) {
        return Err(NativeError::TypeError(format!(
            "{METHOD}: species constructor did not return an ArrayBuffer"
        )));
    }
    let new_obj = match new {
        Value::Object(o) => o,
        _ => unreachable!("is_array_buffer implies an object"),
    };
    if is_detached_buffer(heap, new_obj) {
        return Err(NativeError::TypeError(format!(
            "{METHOD}: species constructor returned a detached ArrayBuffer"
        )));
    }
    // `SameValue(new.[[ArrayBufferData]], O.[[ArrayBufferData]])`:
    // the copy must not alias the source's storage. Guarded by
    // `new_len > 0`: two distinct empty boxes can share a dangling
    // pointer.
    if new_len > 0 {
        let src_ptr = heap.get(buf).bytes().map(|b| b.as_ptr());
        let dst_ptr = heap.get(new_obj).bytes().map(|b| b.as_ptr());
        if src_ptr == dst_ptr {
            return Err(NativeError::TypeError(format!(
                "{METHOD}: species constructor returned the source buffer"
            )));
        }
    }
    // `new.[[ArrayBufferByteLength]] < newLen` throws `TypeError`.
    if array_buffer_length(heap, new_obj) < new_len {
        return Err(NativeError::TypeError(format!(
            "{METHOD}: species constructor returned a short ArrayBuffer"
        )));
    }
    // One memmove, not one property write per byte. `this` (`buf`) is
    // rooted by the VM's `call_roots` across the constructions above.
    heap.copy_object_bytes(buf, new_obj, from as usize, 0, new_len as usize);
    Ok(Value::Object(new_obj))
}

/// Shared core of `ArrayBuffer.prototype.transfer` and
/// `transferToFixedLength` (ECMA-262 25.1.5.4 / 25.1.5.3): copy this
/// buffer's bytes into a fresh buffer of `new_len` bytes, then detach
/// the source. `preserve_resizable` selects `transfer` (the new buffer
/// keeps the source's `maxByteLength`) vs `transferToFixedLength` (the
/// new buffer is fixed-size).
fn arraybuffer_transfer_impl(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
    method: &str,
    preserve_resizable: bool,
) -> Result<Value, NativeError> {
    let buf = match this {
        Value::Object(o) if is_array_buffer(heap, Value::Object(o)) => o,
        _ => {
            return Err(NativeError::TypeError(format!(
                "{method} called on incompatible receiver"
            )));
        }
    };
    if is_detached_buffer(heap, buf) {
        return Err(NativeError::TypeError(format!(
            "{method} called on detached ArrayBuffer"
        )));
    }
    if is_shared_array_buffer(heap, buf) {
        return Err(NativeError::TypeError(format!(
            "{method} called on a SharedArrayBuffer"
        )));
    }
    let old_len = array_buffer_length(heap, buf);
    let new_len_arg = get_arg(args, 0);
    let new_len = if new_len_arg.is_undefined() {
        old_len
    } else {
        // Spec step 6 is a full `ToIntegerOrInfinity` (a `valueOf` hook
        // runs; test262 `transfer/new-length-non-number.js`), and step 7
        // rejects values outside `[0, 2^53 - 1]` — without it a negative
        // would saturate-cast to 0 instead of throwing `RangeError`.
        let n = to_integer_or_infinity_full(heap, new_len_arg)?;
        if !(0.0..=9007199254740991.0).contains(&n) {
            return Err(NativeError::RangeError(format!(
                "{method}: new length out of range"
            )));
        }
        n as u64
    };
    let max = if preserve_resizable {
        heap.get(buf).ab_max_byte_length()
    } else {
        None
    };
    let out = new_array_buffer(heap, new_len, max)?;
    // `out` is fresh, so `src`/`dst` are distinct as `copy_object_bytes`
    // requires; `this` (`buf`) is rooted by the VM's `call_roots` across
    // the allocation above (same as `arraybuffer_slice`).
    heap.copy_object_bytes(buf, out, 0, 0, old_len.min(new_len) as usize);
    // Spec: the source is detached (its `byteLength` own data property
    // drops to 0 with it).
    detach_array_buffer_op(heap, buf)?;
    Ok(Value::Object(out))
}

/// `ArrayBuffer.prototype.transfer([newLength])` (ECMA-262 25.1.5.4):
/// copy this buffer's bytes into a fresh buffer of `newLength` bytes
/// (default: the current length), clamping the copy to the shorter of
/// the two lengths, then detach the source. Transferring a resizable
/// buffer yields a resizable buffer with the same `maxByteLength`.
fn arraybuffer_transfer(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    arraybuffer_transfer_impl(heap, this, args, "ArrayBuffer.prototype.transfer", true)
}

/// `ArrayBuffer.prototype.transferToFixedLength([newLength])`
/// (ECMA-262 25.1.5.3): like `transfer`, but the new buffer is always
/// fixed-size, and the source is detached.
fn arraybuffer_transfer_to_fixed_length(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    arraybuffer_transfer_impl(
        heap,
        this,
        args,
        "ArrayBuffer.prototype.transferToFixedLength",
        false,
    )
}

/// `ArrayBuffer.prototype.resize(newLength)` (ECMA-262 25.1.5.2): grow
/// (zero-fill) or shrink a resizable buffer in place, then rewrite the
/// `byteLength` own data property. Views observe the new length lazily
/// through the buffer, so no view updates are needed here. Throws
/// `TypeError` on a non-`ArrayBuffer` receiver, a detached buffer, or a
/// fixed-size buffer; `RangeError` when `newLength` exceeds
/// `maxByteLength`.
fn arraybuffer_resize(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let buf = match this {
        Value::Object(o) if is_array_buffer(heap, Value::Object(o)) => o,
        _ => {
            return Err(NativeError::TypeError(
                "ArrayBuffer.prototype.resize called on incompatible receiver".into(),
            ));
        }
    };
    // Spec order per test262's `resize/coerced-new-length-detach.js`: the
    // resizable check first, then `ToIndex(newLength)` (a `valueOf` hook
    // runs even on an already-detached buffer and may detach mid-call),
    // then the single detached check, then the `maxByteLength` bound.
    let max = heap.get(buf).ab_max_byte_length().ok_or_else(|| {
        NativeError::TypeError(
            "ArrayBuffer.prototype.resize called on non-resizable ArrayBuffer".into(),
        )
    })?;
    // Shared buffers are never resizable in the observable sense:
    // `resize` throws `TypeError` on them (before the `newLength`
    // conversion, so a `valueOf` hook does not run).
    if is_shared_array_buffer(heap, buf) {
        return Err(NativeError::TypeError(
            "ArrayBuffer.prototype.resize called on a SharedArrayBuffer".into(),
        ));
    }
    let new_len = to_index_full(heap, get_arg(args, 0))?;
    if is_detached_buffer(heap, buf) {
        return Err(NativeError::TypeError(
            "ArrayBuffer.prototype.resize called on detached ArrayBuffer".into(),
        ));
    }
    if new_len > u64::from(max) {
        return Err(NativeError::RangeError(
            "ArrayBuffer.prototype.resize: new length exceeds maxByteLength".into(),
        ));
    }
    let old_len = array_buffer_length(heap, buf);
    if new_len != old_len {
        // Fallible allocation first (no heap allocation happens here, so
        // the shared borrow of the old bytes below cannot observe a
        // collection); then swap the store wholesale.
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(new_len as usize)
            .map_err(|_| NativeError::OutOfMemory)?;
        bytes.resize(new_len as usize, 0);
        let keep = old_len.min(new_len) as usize;
        if keep > 0
            && let Some(old) = heap.object_bytes(buf)
        {
            bytes[..keep].copy_from_slice(&old[..keep]);
        }
        heap.set_array_buffer_bytes(buf, bytes.into_boxed_slice());
        // Keep the `byteLength` snapshot (non-writable/non-configurable
        // to script) in sync via the sanctioned internal rewrite.
        heap.define_property_str(
            buf,
            "byteLength",
            Property {
                value: js_num(new_len as f64),
                writable: false,
                enumerable: false,
                configurable: false,
            },
        )?;
    }
    Ok(Value::Undefined)
}

/// `get ArrayBuffer.prototype.maxByteLength` (ECMA-262 25.1.5.3): the
/// resizable ceiling, or 0 for fixed-size and detached buffers.
fn arraybuffer_max_byte_length_get(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let buf = match this {
        Value::Object(o) if is_array_buffer(heap, Value::Object(o)) => o,
        _ => {
            return Err(NativeError::TypeError(
                "get maxByteLength called on incompatible receiver".into(),
            ));
        }
    };
    if is_shared_array_buffer(heap, buf) {
        return Err(NativeError::TypeError(
            "get maxByteLength called on a SharedArrayBuffer".into(),
        ));
    }
    if is_detached_buffer(heap, buf) {
        return Ok(js_num(0.0));
    }
    Ok(js_num(
        heap.get(buf).ab_max_byte_length().map_or(0.0, f64::from),
    ))
}

/// `get ArrayBuffer.prototype.resizable` (ECMA-262 25.1.5.4): whether
/// the buffer has a `[[ArrayBufferMaxByteLength]]` slot — which a
/// detached buffer keeps, so a detached resizable buffer still reports
/// `true`.
fn arraybuffer_resizable_get(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let buf = match this {
        Value::Object(o) if is_array_buffer(heap, Value::Object(o)) => o,
        _ => {
            return Err(NativeError::TypeError(
                "get resizable called on incompatible receiver".into(),
            ));
        }
    };
    if is_shared_array_buffer(heap, buf) {
        return Err(NativeError::TypeError(
            "get resizable called on a SharedArrayBuffer".into(),
        ));
    }
    Ok(Value::Bool(heap.get(buf).ab_max_byte_length().is_some()))
}

/// `get ArrayBuffer.prototype.detached`: whether the buffer is detached.
fn arraybuffer_detached_get(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let buf = match this {
        Value::Object(o) if is_array_buffer(heap, Value::Object(o)) => o,
        _ => {
            return Err(NativeError::TypeError(
                "get detached called on incompatible receiver".into(),
            ));
        }
    };
    if is_shared_array_buffer(heap, buf) {
        return Err(NativeError::TypeError(
            "get detached called on a SharedArrayBuffer".into(),
        ));
    }
    Ok(Value::Bool(is_detached_buffer(heap, buf)))
}

// ---------------------------------------------------------------------------
// Typed arrays
// ---------------------------------------------------------------------------

/// (constructor name, element size in bytes, kind). `TypedArrayKind` lives
/// in `crate::value` so the [`ObjectKind::TypedArrayView`] tag can name
/// the element type.
const ETYPES: &[(&str, usize, TypedArrayKind)] = &[
    ("Int8Array", 1, TypedArrayKind::Int8),
    ("Uint8Array", 1, TypedArrayKind::Uint8),
    ("Uint8ClampedArray", 1, TypedArrayKind::Uint8Clamped),
    ("Int16Array", 2, TypedArrayKind::Int16),
    ("Uint16Array", 2, TypedArrayKind::Uint16),
    ("Int32Array", 4, TypedArrayKind::Int32),
    ("Uint32Array", 4, TypedArrayKind::Uint32),
    ("Float32Array", 4, TypedArrayKind::Float32),
    ("Float64Array", 8, TypedArrayKind::Float64),
    ("BigInt64Array", 8, TypedArrayKind::BigInt64),
    ("BigUint64Array", 8, TypedArrayKind::BigUint64),
];

/// `ETYPES` index of `Uint8Array`. The single source of truth for the
/// constructor table below (which is still index-addressed).
pub(crate) const UINT8_ETYPE: usize = 1;

/// Range-checked `ETYPES` lookup: an out-of-range constructor index fails
/// closed instead of indexing out of bounds.
fn etype_parts(etype: usize) -> Option<(&'static str, usize, TypedArrayKind)> {
    ETYPES.get(etype).copied()
}

/// True when `v` is a genuine typed-array view: the unforgeable
/// [`ObjectKind::TypedArrayView`] tag. A plain object with NUL-prefixed
/// properties of the same names is *not* a view — the old hidden-slot
/// scheme was forgeable from script, so the tag is the only check.
fn is_typed_array(heap: &Heap, v: Value) -> bool {
    matches!(v, Value::Object(o) if is_typed_array_view(heap, o))
}

/// True when `o` is a genuine typed-array view of any element type.
/// `pub(crate)` for the sibling modules (`crate::builtin_typedarray`,
/// `crate::bindings_encoding`) that need the `ArrayBuffer.isView`
/// semantics without going through a `Value`.
pub(crate) fn is_typed_array_view(heap: &Heap, o: Gc<Object>) -> bool {
    matches!(heap.get(o).kind, ObjectKind::TypedArrayView { .. })
}

/// Decompose any typed-array view into `(buffer, byte_offset, len_elems,
/// etype)`: one tag read, no property lookups. `pub(crate)` for the
/// sibling typed-array builder module's
/// [`view_buffer`](crate::builtin_typedarray::view_buffer), which recovers
/// the backing buffer for every element type.
pub(crate) fn typed_array_parts(
    heap: &Heap,
    view: Gc<Object>,
) -> Option<(Gc<Object>, u64, u64, TypedArrayKind)> {
    match heap.get(view).kind {
        ObjectKind::TypedArrayView {
            buffer,
            byte_offset,
            len_elems,
            etype,
        } => Some((buffer, u64::from(byte_offset), u64::from(len_elems), etype)),
        _ => None,
    }
}

/// Resolve a typed-array view against its buffer's *current* state
/// (ECMA-262 `IsTypedArrayOutOfBounds` + length-tracking): `None` when
/// the buffer is detached or the view is out of bounds of the buffer's
/// current byte length (getters report 0, indexed access is dead),
/// `Some(effective_len)` otherwise — recomputed from the buffer for
/// length-tracking views, the fixed snapshot for the rest.
///
/// One `heap.get(buf)` deref plus a few integer ops: cheap enough for
/// the element-access hot paths.
fn typed_array_resolve(
    heap: &Heap,
    buf: Gc<Object>,
    byte_offset: u64,
    len_elems: u64,
    size: u64,
    length_tracking: bool,
) -> Option<u64> {
    let bytes = heap.object_bytes(buf)?;
    let buf_len = bytes.len() as u64;
    if length_tracking {
        // Length-tracking views shrink (and grow) with the buffer. Per
        // `IsTypedArrayOutOfBounds`, an offset past the buffer's current
        // end is out of bounds (getters report 0) rather than an empty
        // in-bounds view.
        if byte_offset > buf_len {
            return None;
        }
        return Some(buf_len.saturating_sub(byte_offset) / size);
    }
    // Fixed-length view: out of bounds (buffer shrank past the view, or
    // the offset itself is past the end) reads as zero-length.
    let end = byte_offset.saturating_add(len_elems.saturating_mul(size));
    if byte_offset > buf_len || end > buf_len {
        return None;
    }
    Some(len_elems)
}

/// Resolve a `DataView` against its buffer's *current* state
/// (`IsDataViewOutOfBounds` + length-tracking): `Ok(effective_byte_len)`
/// — recomputed for length-tracking views, the fixed snapshot
/// otherwise — or `TypeError` when the buffer is detached or the view
/// is out of bounds (per spec, `DataView` getters and element
/// accessors throw rather than reporting 0).
fn dataview_resolve(
    heap: &Heap,
    buf: Gc<Object>,
    byte_offset: u64,
    byte_len: u64,
    length_tracking: bool,
) -> Result<u64, NativeError> {
    let bytes = heap.object_bytes(buf).ok_or_else(|| {
        NativeError::TypeError("DataView operation on detached ArrayBuffer".into())
    })?;
    let buf_len = bytes.len() as u64;
    if byte_offset > buf_len {
        return Err(NativeError::TypeError(
            "DataView is out of bounds of its ArrayBuffer".into(),
        ));
    }
    if length_tracking {
        return Ok(buf_len - byte_offset);
    }
    if byte_offset.saturating_add(byte_len) > buf_len {
        return Err(NativeError::TypeError(
            "DataView is out of bounds of its ArrayBuffer".into(),
        ));
    }
    Ok(byte_len)
}

/// ECMAScript `ToInt32`.
fn to_int32(n: f64) -> i32 {
    if !n.is_finite() || n == 0.0 {
        return 0;
    }
    let m = n.trunc() % 4294967296.0;
    let m = if m < 0.0 { m + 4294967296.0 } else { m };
    if m >= 2147483648.0 {
        (m - 4294967296.0) as i32
    } else {
        m as i32
    }
}

/// ECMAScript `ToUint8Clamp`.
fn to_uint8_clamp(n: f64) -> u8 {
    if n.is_nan() || n <= 0.0 {
        return 0;
    }
    if n >= 255.0 {
        return 255;
    }
    let f = n.floor();
    let frac = n - f;
    let r = if frac > 0.5 || (frac == 0.5 && (f as u64) % 2 == 1) {
        f + 1.0
    } else {
        f
    };
    r as u8
}

/// Exact `f64` → `(negative, magnitude-digits)` for finite integral
/// values; `None` for NaN/infinity/fractions. The loop divides out 2^32
/// words: every step is exact (power-of-two division of an
/// exactly-representable value), so the digits are exact even past 2^53.
fn bigint_digits_from_f64(f: f64) -> Option<(bool, Vec<u32>)> {
    if !f.is_finite() || f.fract() != 0.0 {
        return None;
    }
    if f == 0.0 {
        return Some((false, vec![0]));
    }
    let negative = f < 0.0;
    let mut m = f.abs();
    let mut digits = Vec::new();
    while m >= 1.0 {
        digits.push((m % 4294967296.0) as u32);
        m = (m / 4294967296.0).floor();
    }
    Some((negative, digits))
}

/// Normalize `(negative, magnitude-digits)` into a heap `BigInt` value:
/// strip high zero words so zero never carries a sign.
fn bigint_normalized(
    heap: &mut Heap,
    negative: bool,
    mut digits: Vec<u32>,
) -> Result<Value, NativeError> {
    while digits.len() > 1 && digits[digits.len() - 1] == 0 {
        digits.pop();
    }
    if digits.is_empty() {
        digits.push(0);
    }
    let negative = negative && digits.iter().any(|&d| d != 0);
    heap.alloc_bigint(negative, digits)
        .map(Value::BigInt)
        .map_err(|_| NativeError::OutOfMemory)
}

/// Spec `ToBigInt`: `ToPrimitive(value, number)` for objects, then the
/// primitive conversion (mirrors `BigInt(v)` in `builtin_string_number`,
/// which is call-shaped rather than conversion-shaped).
fn to_bigint(heap: &mut Heap, v: Value) -> Result<Value, NativeError> {
    let prim = match v {
        Value::Object(o) => to_primitive_with_hint(heap, o, PrimitiveHint::Number)?,
        _ => v,
    };
    let (negative, digits): (bool, Vec<u32>) = match prim {
        Value::BigInt(g) => return Ok(Value::BigInt(g)),
        Value::Bool(b) => (false, vec![u32::from(b)]),
        Value::Int(i) => {
            let m = i64::from(i).unsigned_abs();
            (i < 0, vec![m as u32, (m >> 32) as u32])
        }
        Value::Float(f) => bigint_digits_from_f64(f).ok_or_else(|| {
            NativeError::RangeError(format!(
                "The number {f} cannot be converted to a BigInt because it is not an integer"
            ))
        })?,
        Value::Str(s) => {
            let b: BigInt = bn_parse(s.as_wtf8()).ok_or_else(|| {
                NativeError::SyntaxError(format!("Cannot convert {} to a BigInt", s.as_wtf8()))
            })?;
            (b.negative, b.digits)
        }
        Value::Null => {
            return Err(NativeError::TypeError(
                "Cannot convert null to a BigInt".into(),
            ));
        }
        Value::Undefined => {
            return Err(NativeError::TypeError(
                "Cannot convert undefined to a BigInt".into(),
            ));
        }
        Value::Symbol(_) => {
            return Err(NativeError::TypeError(
                "Cannot convert a Symbol value to a BigInt".into(),
            ));
        }
        // Unreachable: `to_primitive_with_hint` only yields primitives.
        Value::Object(_) => {
            return Err(NativeError::TypeError(
                "Cannot convert object to a BigInt".into(),
            ));
        }
    };
    bigint_normalized(heap, negative, digits)
}

/// Low 64 bits of a bigint's two's-complement representation, for the
/// 8-byte bigint element write path.
fn bigint_low64(g: Gc<BigInt>) -> u64 {
    let lo = u64::from(g.digits.first().copied().unwrap_or(0))
        | (u64::from(g.digits.get(1).copied().unwrap_or(0)) << 32);
    if g.negative { lo.wrapping_neg() } else { lo }
}

/// Semantic equality of two heap bigints (normalized constructors never
/// produce `-0`, but the comparison does not rely on it).
fn bigint_eq(a: Gc<BigInt>, b: Gc<BigInt>) -> bool {
    let az = a.digits.iter().all(|&d| d == 0);
    let bz = b.digits.iter().all(|&d| d == 0);
    if az && bz {
        return true;
    }
    if az || bz || a.negative != b.negative {
        return false;
    }
    let mut x = a.digits.as_slice();
    let mut y = b.digits.as_slice();
    while x.last() == Some(&0) {
        x = &x[..x.len() - 1];
    }
    while y.last() == Some(&0) {
        y = &y[..y.len() - 1];
    }
    x == y
}

/// Exact decimal rendering of a heap bigint (for `join`/`toString` on
/// bigint typed arrays).
fn bigint_to_decimal(g: Gc<BigInt>) -> String {
    if g.digits.iter().all(|&d| d == 0) {
        return "0".to_string();
    }
    let mut m: Vec<u32> = g.digits.clone();
    let mut rev: Vec<u8> = Vec::new();
    while m.iter().any(|&d| d != 0) {
        let mut rem: u64 = 0;
        for w in m.iter_mut().rev() {
            let cur = (rem << 32) | u64::from(*w);
            *w = (cur / 10) as u32;
            rem = cur % 10;
        }
        rev.push(b'0' + rem as u8);
    }
    let mut s = String::with_capacity(rev.len() + 1);
    if g.negative {
        s.push('-');
    }
    s.extend(rev.iter().rev().map(|&c| c as char));
    s
}

/// `ToString` applied to a typed-array element value: elements are only
/// ever numbers or bigints, so the object branch of `to_string_js` is
/// unreachable here.
fn elem_js_string(v: Value) -> Wtf8Buf {
    match v {
        Value::Int(i) => Wtf8Buf::from(&i.to_string()),
        Value::Float(f) => Wtf8Buf::from(&js_number_to_string(f)),
        Value::BigInt(g) => Wtf8Buf::from(bigint_to_decimal(g).as_str()),
        _ => Wtf8Buf::from("undefined"),
    }
}

/// Read one element: bytes are stored little-endian in the buffer.
/// Now `&mut Heap` + `Result`: bigint elements materialize a heap
/// `BigInt` on every read.
fn elem_get(
    heap: &mut Heap,
    buf: Gc<Object>,
    addr: u64,
    kind: TypedArrayKind,
) -> Result<Value, NativeError> {
    // Raw little-endian value first; the immutable buffer borrows end
    // before the bigint arms below reborrow `heap` mutably.
    let mut raw: u64 = 0;
    for i in 0..kind.size() {
        raw |= u64::from(buffer_get_byte(heap, buf, addr + i as u64)) << (8 * i);
    }
    match kind {
        TypedArrayKind::Int8 => Ok(Value::Int(raw as u8 as i8 as i32)),
        TypedArrayKind::Uint8 => Ok(Value::Int(raw as i32)),
        TypedArrayKind::Uint8Clamped => Ok(Value::Int(raw as i32)),
        TypedArrayKind::Int16 => Ok(Value::Int(raw as u16 as i16 as i32)),
        TypedArrayKind::Uint16 => Ok(js_num(raw as f64)),
        TypedArrayKind::Int32 => Ok(Value::Int(raw as u32 as i32)),
        TypedArrayKind::Uint32 => Ok(js_num(raw as f64)),
        TypedArrayKind::Float32 => Ok(Value::Float(f64::from(f32::from_bits(raw as u32)))),
        TypedArrayKind::Float64 => Ok(Value::Float(f64::from_bits(raw))),
        TypedArrayKind::BigInt64 => {
            let m = (raw as i64).unsigned_abs();
            bigint_normalized(heap, raw >> 63 != 0, vec![m as u32, (m >> 32) as u32])
        }
        TypedArrayKind::BigUint64 => {
            bigint_normalized(heap, false, vec![raw as u32, (raw >> 32) as u32])
        }
    }
}

/// Element value → 8 little-endian bytes (only `kind.size()` are stored).
/// Bigint kinds go through `ToBigInt` (low 64 bits of the two's-complement
/// representation); the rest through `ToNumber` with the per-kind
/// conversion.
fn elem_bytes(heap: &mut Heap, kind: TypedArrayKind, v: Value) -> Result<[u8; 8], NativeError> {
    if kind.is_bigint() {
        let b = to_bigint(heap, v)?;
        let Value::BigInt(g) = b else {
            return Err(NativeError::TypeError(
                "ToBigInt did not produce a BigInt".into(),
            ));
        };
        return Ok(bigint_low64(g).to_le_bytes());
    }
    let n = to_number_full(heap, v)?;
    Ok(match kind {
        TypedArrayKind::Int8 => [to_int32(n) as i8 as u8, 0, 0, 0, 0, 0, 0, 0],
        TypedArrayKind::Uint8 => [to_int32(n) as u8, 0, 0, 0, 0, 0, 0, 0],
        TypedArrayKind::Uint8Clamped => [to_uint8_clamp(n), 0, 0, 0, 0, 0, 0, 0],
        TypedArrayKind::Int16 => {
            let x = to_int32(n) as i16 as u16;
            [x as u8, (x >> 8) as u8, 0, 0, 0, 0, 0, 0]
        }
        TypedArrayKind::Uint16 => {
            let x = to_int32(n) as u16;
            [x as u8, (x >> 8) as u8, 0, 0, 0, 0, 0, 0]
        }
        TypedArrayKind::Int32 => {
            let x = to_int32(n);
            [
                x as u8,
                (x >> 8) as u8,
                (x >> 16) as u8,
                (x >> 24) as u8,
                0,
                0,
                0,
                0,
            ]
        }
        TypedArrayKind::Uint32 => {
            let x = to_int32(n) as u32;
            [
                x as u8,
                (x >> 8) as u8,
                (x >> 16) as u8,
                (x >> 24) as u8,
                0,
                0,
                0,
                0,
            ]
        }
        TypedArrayKind::Float32 => {
            let bits = (n as f32).to_bits();
            [
                bits as u8,
                (bits >> 8) as u8,
                (bits >> 16) as u8,
                (bits >> 24) as u8,
                0,
                0,
                0,
                0,
            ]
        }
        TypedArrayKind::Float64 => n.to_bits().to_le_bytes(),
        TypedArrayKind::BigInt64 | TypedArrayKind::BigUint64 => {
            unreachable!("bigint kinds convert via ToBigInt above")
        }
    })
}

/// Write one element: value is converted per element kind, bytes stored
/// little-endian.
fn elem_set(
    heap: &mut Heap,
    buf: Gc<Object>,
    addr: u64,
    kind: TypedArrayKind,
    v: Value,
) -> Result<(), NativeError> {
    let bytes = elem_bytes(heap, kind, v)?;
    let size = kind.size();
    for (i, b) in bytes.iter().take(size).enumerate() {
        buffer_set_byte(heap, buf, addr + i as u64, *b);
    }
    Ok(())
}

/// Integer-indexed `[[Get]]` for a typed array view: out of bounds reads
/// `undefined`. Now `&mut Heap` + `Result`: bigint elements materialize a
/// heap `BigInt` on every read.
///
/// Called by `crate::builtin_typedarray`'s `uint8_get` (the native-level
/// integer-indexed exotic path) and by the VM's indexed load path.
pub(crate) fn typed_index_get(
    heap: &mut Heap,
    view: Gc<Object>,
    i: u64,
) -> Result<Value, NativeError> {
    let Some((buf, offset, length, kind)) = typed_array_parts(heap, view) else {
        return Ok(Value::Undefined);
    };
    // Detached or out-of-bounds (shrunk buffer): the index is not
    // addressable, so the read yields `undefined` (spec
    // `IntegerIndexedElementGet`).
    let tracking = heap.get(view).view_length_tracking();
    let Some(n) = typed_array_resolve(heap, buf, offset, length, kind.size() as u64, tracking)
    else {
        return Ok(Value::Undefined);
    };
    if i >= n {
        return Ok(Value::Undefined);
    }
    elem_get(heap, buf, offset + i * kind.size() as u64, kind)
}

/// Integer-indexed `[[Set]]` for a typed array view: out of bounds writes
/// are ignored.
///
/// Called by `crate::builtin_typedarray`'s `uint8_set` (the native-level
/// integer-indexed exotic path), which the VM's indexed store path routes
/// through.
pub(crate) fn typed_index_set(
    heap: &mut Heap,
    view: Gc<Object>,
    i: u64,
    v: Value,
) -> Result<(), NativeError> {
    let Some((buf, offset, length, kind)) = typed_array_parts(heap, view) else {
        return Err(NativeError::TypeError(
            "typed array operation on incompatible receiver".into(),
        ));
    };
    // Detached or out-of-bounds: out-of-bounds writes are ignored (spec
    // `IntegerIndexedElementSet`).
    let tracking = heap.get(view).view_length_tracking();
    let Some(n) = typed_array_resolve(heap, buf, offset, length, kind.size() as u64, tracking)
    else {
        return Ok(());
    };
    if i >= n {
        return Ok(());
    }
    elem_set(heap, buf, offset + i * kind.size() as u64, kind, v)
}

/// Allocate a typed array view over `buf` at `byte_offset` with
/// `len_elems` elements. The view metadata (buffer, offset, length,
/// element type) lives in the unforgeable [`ObjectKind::TypedArrayView`]
/// tag — not in hidden properties — so user script can neither forge a
/// view from scratch nor retarget a real one. The tag edge keeps `buf`
/// alive from the moment of allocation (the tracer follows it).
///
/// `length_tracking` marks a view created with an omitted length over a
/// resizable buffer: `length` / `byteLength` recompute from the buffer's
/// current byte length (see [`typed_array_resolve`]). It lives on the
/// object (not the tag) so the tag's struct shape — constructed in
/// sibling modules too — stays untouched.
///
/// `length`, `byteLength`, and `byteOffset` are *not* own properties:
/// they resolve through the `get %TypedArray%.prototype.length` (etc.)
/// accessors, per ECMA-262 22.2.3.16–18 — which read the same tag, so
/// there is exactly one source of truth.
fn new_typed_array(
    heap: &mut Heap,
    kind: TypedArrayKind,
    buf: Gc<Object>,
    byte_offset: u64,
    len_elems: u64,
    length_tracking: bool,
    proto: Value,
) -> Result<Gc<Object>, NativeError> {
    // Buffers are capped at `u32::MAX` bytes and every caller keeps
    // `byte_offset + len_elems * size` inside the buffer, so these fit;
    // fail closed rather than truncate.
    let byte_offset = u32::try_from(byte_offset)
        .map_err(|_| NativeError::RangeError("typed array: byte offset out of range".into()))?;
    let len_elems = u32::try_from(len_elems)
        .map_err(|_| NativeError::RangeError("typed array: length out of range".into()))?;
    let mut obj = Object::new(ObjectKind::TypedArrayView {
        buffer: buf,
        byte_offset,
        len_elems,
        etype: kind,
    });
    obj.set_view_length_tracking(length_tracking);
    let view = heap.alloc_object(obj)?;
    heap.set_prototype(view, proto);
    Ok(view)
}

/// Shared construction logic for every typed array constructor.
fn typed_array_construct(
    heap: &mut Heap,
    etype: usize,
    args: &[Value],
    proto: Value,
) -> Result<Value, NativeError> {
    let Some((_, size, kind)) = etype_parts(etype) else {
        return Err(NativeError::RangeError(
            "typed array: unknown element type".into(),
        ));
    };
    let arg0 = get_arg(args, 0);
    // Case 1: an ArrayBuffer (or SharedArrayBuffer) view.
    if is_array_buffer(heap, arg0) {
        let buf = require_object(arg0, "typed array constructor")?;
        // Spec order (`InitializeTypedArrayFromArrayBuffer` steps 2-5):
        // `ToIndex(byteOffset)` — its `valueOf` hook runs even on a
        // detached buffer — then the offset-alignment `RangeError`, then
        // `ToIndex(length)`, and only then the single detached check
        // after *both* conversions. So `new Int16Array(detachedBuf, 1)`
        // throws `RangeError`, while
        // `new TA(buf, 0, { valueOf() { $DETACHBUFFER(buf); return 1; } })`
        // throws `TypeError`.
        let offset_arg = get_arg(args, 1);
        let byte_offset = if offset_arg.is_undefined() {
            0
        } else {
            to_index_full(heap, offset_arg)?
        };
        let buf_len = array_buffer_length(heap, buf);
        if byte_offset % size as u64 != 0 {
            return Err(NativeError::RangeError(
                "byte offset must be a multiple of the element size".into(),
            ));
        }
        let len_arg = get_arg(args, 2);
        // Length-tracking (ECMA-262 `auto` length): omitted length over a
        // resizable buffer. The view's `length` / `byteLength` recompute
        // from the buffer's current byte length (see
        // [`typed_array_resolve`]).
        let length_tracking =
            len_arg.is_undefined() && heap.get(buf).ab_max_byte_length().is_some();
        let len_elems = if len_arg.is_undefined() {
            // `byte_offset > buf_len` is rejected just below; saturate
            // here so a mid-conversion shrink cannot underflow the
            // subtraction before that check runs.
            buf_len.saturating_sub(byte_offset) / size as u64
        } else {
            to_index_full(heap, len_arg)?
        };
        if is_detached_buffer(heap, buf) {
            return Err(NativeError::TypeError(
                "typed array constructor: ArrayBuffer is detached".into(),
            ));
        }
        // Spec (InitializeTypedArrayFromArrayBuffer, step 9.1): only when
        // `length` is omitted AND the buffer is fixed-length
        // (length-tracking views over resizable buffers take the step-8
        // auto path) — `bufferByteLength mod elementSize ≠ 0` throws
        // `RangeError`. `buf_len` is still fresh here: in the omitted-
        // length branch no user code ran between the read above and here,
        // and `byte_offset` was already required to be aligned, so
        // checking `buf_len.is_multiple_of(size)` is equivalent to
        // checking `(buf_len - byte_offset)`.
        if len_arg.is_undefined() && !length_tracking && !buf_len.is_multiple_of(size as u64) {
            return Err(NativeError::RangeError(
                "buffer byte length is not a multiple of the element size".into(),
            ));
        }
        if byte_offset > buf_len {
            return Err(NativeError::RangeError("byte offset out of range".into()));
        }
        // Checked: `len_elems * size` can overflow `u64` (both come from
        // user input via `to_index`), which would wrap small and pass the
        // range check below.
        let view_bytes = len_elems
            .checked_mul(size as u64)
            .ok_or_else(|| NativeError::RangeError("typed array view length overflow".into()))?;
        let end = byte_offset
            .checked_add(view_bytes)
            .ok_or_else(|| NativeError::RangeError("typed array view length overflow".into()))?;
        if end > buf_len {
            return Err(NativeError::RangeError(
                "typed array view out of buffer range".into(),
            ));
        }
        return Ok(Value::Object(new_typed_array(
            heap,
            kind,
            buf,
            byte_offset,
            len_elems,
            length_tracking,
            proto,
        )?));
    }
    // Case 2: copy from another typed array.
    if is_typed_array(heap, arg0) {
        let src = require_object(arg0, "typed array constructor")?;
        let Some((src_buf, src_offset, src_len, src_kind)) = typed_array_parts(heap, src) else {
            return Err(NativeError::TypeError(
                "typed array constructor: bad source".into(),
            ));
        };
        // Checked: `src_len` near 2^32 with a large element size would wrap
        // the multiply small, pass the length cap, and then hang iterating.
        let byte_len = src_len
            .checked_mul(size as u64)
            .ok_or_else(|| NativeError::RangeError("typed array length overflow".into()))?;
        let buf = new_array_buffer(heap, byte_len, None)?;
        let view = new_typed_array(heap, kind, buf, 0, src_len, false, proto)?;
        // `src_len` is bounded by `u32::MAX / 1` via `new_array_buffer`'s
        // cap, so `i * elem_size` below cannot overflow.
        for i in 0..src_len {
            let v = elem_get(
                heap,
                src_buf,
                src_offset + i * src_kind.size() as u64,
                src_kind,
            )?;
            elem_set(heap, buf, i * size as u64, kind, v)?;
        }
        return Ok(Value::Object(view));
    }
    // Case 3: length, an iterable, or an array-like object (spec 22.2.5.1:
    // the `@@iterator` method is consulted before the array-like path).
    if let Value::Object(o) = arg0
        && let Some((values, base)) = iterable_values(heap, o)?
    {
        let len_elems = values.len() as u64;
        let byte_len = len_elems
            .checked_mul(size as u64)
            .ok_or_else(|| NativeError::RangeError("typed array length overflow".into()))?;
        let buf = new_array_buffer(heap, byte_len, None)?;
        let view = new_typed_array(heap, kind, buf, 0, len_elems, false, proto)?;
        // `view` is not yet reachable; hold it while element conversion
        // (BigInt allocation especially) may collect. `values` stays
        // rooted via `base` until the copy completes.
        let out = with_temp_root(heap, Value::Object(view), |heap| {
            for (i, v) in values.iter().enumerate() {
                typed_index_set(heap, view, i as u64, *v)?;
            }
            Ok(Value::Object(view))
        });
        heap.release_temp(base);
        return out;
    }
    let (len_elems, src_obj) = match arg0 {
        Value::Undefined => (0, None),
        Value::Object(o) => {
            let len = array_like_length_ordinary(heap, o)?;
            (len, Some(o))
        }
        _ => (to_index(arg0)?, None),
    };
    // Checked: `to_index`/`to_length` admit values near 2^53, whose product
    // with `size` wraps `u64`.
    let byte_len = len_elems
        .checked_mul(size as u64)
        .ok_or_else(|| NativeError::RangeError("typed array length overflow".into()))?;
    let buf = new_array_buffer(heap, byte_len, None)?;
    let view = new_typed_array(heap, kind, buf, 0, len_elems, false, proto)?;
    if let Some(src) = src_obj {
        // `view` is fresh and unrooted; hold it while the element reads
        // (getters) and conversions may collect.
        with_temp_root(heap, Value::Object(view), |heap| {
            for i in 0..len_elems {
                let v = array_like_get_ordinary(heap, src, i)?;
                let g = heap.protect(v);
                let r = typed_index_set(heap, view, i, v);
                heap.release_temp(g);
                r?;
            }
            Ok(())
        })?;
    }
    Ok(Value::Object(view))
}

// ---------------------------------------------------------------------------
// `%Uint8Array.prototype%` methods
//
// v1 installs a real prototype object only for `Uint8Array` — the type the
// native builders (`TextEncoder.encode`, `uint8array_from_bytes`) produce.
// The methods are written generically over the receiver's element type via
// `typed_array_parts`, so they stay correct if per-type prototypes land
// later.
// ---------------------------------------------------------------------------

/// Shared receiver check for the typed-array prototype methods: the view
/// object plus its decomposed `(buffer, byte_offset, len_elems, etype)`.
/// Non-views fail closed with a `TypeError` (forgery is impossible: the
/// metadata lives in the kind tag, not in properties).
type CheckedView = (Gc<Object>, Gc<Object>, u64, u64, TypedArrayKind);

/// `ValidateTypedArray` (ECMA-262 22.2.3.24): decompose a genuine view,
/// throwing `TypeError` on a detached buffer or an out-of-bounds view
/// (buffer shrank past a fixed-length view). The returned length is the
/// *effective* one — recomputed from the buffer's current byte length
/// for length-tracking views — so prototype methods iterate what is
/// actually addressable. The `%TypedArray%.prototype` getters and the
/// integer-indexed paths do *not* use this (they report 0 / `undefined`
/// instead of throwing); see [`typed_array_resolve`].
fn require_typed_array_view(
    heap: &Heap,
    this: Value,
    method: &str,
) -> Result<CheckedView, NativeError> {
    let Value::Object(view) = this else {
        return Err(NativeError::TypeError(format!(
            "{method} called on incompatible receiver"
        )));
    };
    let Some((buf, offset, len, etype)) = typed_array_parts(heap, view) else {
        return Err(NativeError::TypeError(format!(
            "{method} called on incompatible receiver"
        )));
    };
    let tracking = heap.get(view).view_length_tracking();
    let Some(len) = typed_array_resolve(heap, buf, offset, len, etype.size() as u64, tracking)
    else {
        return Err(NativeError::TypeError(format!(
            "{method} called on detached or out-of-bounds ArrayBuffer"
        )));
    };
    Ok((view, buf, offset, len, etype))
}

/// `%TypedArray%.prototype.slice(start, end)`: a new view of the same
/// element type over a freshly copied buffer. One `memmove`, not one
/// property write per byte.
fn typed_array_slice(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (view, buf, offset, len, etype) = require_typed_array_view(heap, this, "slice")?;
    let size = etype.size();
    // Relative start/end, mirroring `arraybuffer_slice`. The full
    // conversion runs `valueOf` hooks, so a mid-call `resize` is honored
    // (and the `copy_object_bytes` fast path below clamps to the
    // buffer's then-current length).
    let rel = |n: f64| -> u64 {
        if n < 0.0 {
            (len as f64 + n).max(0.0) as u64
        } else {
            n.min(len as f64) as u64
        }
    };
    let from = rel(to_integer_or_infinity_full(heap, get_arg(args, 0))?);
    let end_arg = get_arg(args, 1);
    let to = if end_arg.is_undefined() {
        len
    } else {
        rel(to_integer_or_infinity_full(heap, end_arg)?)
    };
    let (from, to) = if from > to { (to, to) } else { (from, to) };
    let count = to - from;
    // `TypedArraySpeciesCreate` (ECMA-262 22.2.3.23): the copy's element
    // type stays `etype`, but the constructor comes from
    // `O.constructor[@@species]` (subclasses propagate).
    let av = species_construct(heap, view, count, "slice")?;
    let aobj = require_object(av, "TypedArray.prototype.slice")?;
    // Fast path: the species result is a genuine view of the same element
    // type over a *different* buffer — one `memmove` into it.
    if let Some((abuf, aoff, alen, akind)) = typed_array_parts(heap, aobj)
        && akind == etype
        && alen >= count
        && abuf != buf
    {
        let byte_len = count
            .checked_mul(size as u64)
            .ok_or_else(|| NativeError::RangeError("typed array length overflow".into()))?;
        // `abuf != buf`, so `copy_object_bytes`' distinct-src/dst
        // requirement holds; the copy clamps defensively, so a forged
        // out-of-range view cannot overrun. `buf` is rooted by the VM's
        // `call_roots` across the species construction above, and `av` is
        // rooted below for the slow path.
        heap.copy_object_bytes(
            buf,
            abuf,
            (offset + from * size as u64) as usize,
            aoff as usize,
            byte_len as usize,
        );
        return Ok(av);
    }
    // Slow path: a hostile species result (non-view, wrong kind, short,
    // or aliasing the source buffer) — element-wise copy with element
    // conversion, exactly like `map`'s store.
    with_temp_root(heap, av, |heap| {
        with_temp_root(heap, this, |heap| {
            for i in 0..count {
                let v = typed_index_get(heap, view, from + i)?;
                let g = heap.protect(v);
                let r = set_constructed_elem(heap, aobj, i, v);
                heap.release_temp(g);
                r?;
            }
            Ok(())
        })
    })?;
    Ok(av)
}

/// `%TypedArray%.prototype.set(source, offset)`: copy elements from a
/// typed-array view or an array-like into this view at `offset`
/// (default `0`). Throws `RangeError` when the source does not fit.
fn typed_array_set(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    // `typed_array_parts` already range-checked `etype`, so `size` lookups
    // below are infallible; `typed_index_get`/`typed_index_set` apply the
    // views' offsets and element conversions.
    let (view, buf, _offset, len, _etype) = require_typed_array_view(heap, this, "set")?;
    // `get_arg` yields `Undefined` for a missing offset. Spec (ES2027
    // §23.2.3.26 step 4) converts with `ToIntegerOrInfinity`, so the
    // full conversion runs here: an object offset's `valueOf`/`toString`
    // hooks fire (primitive-only `to_index` would map objects to NaN→0
    // and swallow throwing hooks).
    let dst_off = to_index_full(heap, get_arg(args, 1))?;
    if dst_off > len {
        return Err(NativeError::RangeError("set: offset out of range".into()));
    }
    let src = get_arg(args, 0);
    if is_typed_array(heap, src) {
        let src_obj = require_object(src, "set")?;
        // `ValidateTypedArray(src)`: a detached or out-of-bounds source
        // throws `TypeError` (ECMA-262 22.2.3.21 step 4) instead of
        // silently filling the target with zeros. The resolved `src_len`
        // is the source's current effective length.
        let (_, src_buf, _, src_len, _) = require_typed_array_view(heap, src, "set")?;
        if src_len > len - dst_off {
            return Err(NativeError::RangeError("set: source does not fit".into()));
        }
        // Same-buffer copies go through a scratch vector first (spec copies
        // the source list before writing, so overlapping ranges work).
        // `typed_index_get`/`typed_index_set` already apply the views'
        // offsets and element conversions. Scratch values are temp-rooted:
        // bigint reads materialize fresh heap bigints, and the write loop
        // below can run user code (`valueOf`) that collects.
        if src_buf == buf {
            let mut scratch = Vec::new();
            scratch
                .try_reserve(src_len as usize)
                .map_err(|_| NativeError::OutOfMemory)?;
            let base = heap.protect(Value::Object(view));
            for i in 0..src_len {
                let v = typed_index_get(heap, src_obj, i)?;
                heap.protect(v);
                scratch.push(v);
            }
            for (i, v) in scratch.into_iter().enumerate() {
                typed_index_set(heap, view, dst_off + i as u64, v)?;
            }
            heap.release_temp(base);
        } else {
            for i in 0..src_len {
                let v = typed_index_get(heap, src_obj, i)?;
                let g = heap.protect(v);
                let r = typed_index_set(heap, view, dst_off + i, v);
                heap.release_temp(g);
                r?;
            }
        }
    } else {
        // Array-like (or, per spec `ToObject`, primitive) source:
        // `ToLength(Get(src, "length"))` then indexed reads. `null` /
        // `undefined` throw; other primitives box (`this_object`).
        let src_obj = match src {
            Value::Object(o) => o,
            Value::Undefined | Value::Null => {
                return Err(NativeError::TypeError(
                    "set: source must be an object".into(),
                ));
            }
            _ => crate::builtin_object_array::this_object(heap, src)?,
        };
        let src_len = array_like_length_ordinary(heap, src_obj)?;
        if src_len > len - dst_off {
            return Err(NativeError::RangeError("set: source does not fit".into()));
        }
        for i in 0..src_len {
            let v = array_like_get_ordinary(heap, src_obj, i)?;
            // `v` may come from a getter; root it across the store, which
            // can run user code (`valueOf`) that collects.
            let g = heap.protect(v);
            let r = typed_index_set(heap, view, dst_off + i, v);
            heap.release_temp(g);
            r?;
        }
    }
    Ok(Value::Undefined)
}

/// `%TypedArray%.prototype.subarray(begin, end)`: a new view of the same
/// element type over the *same* buffer (no copy). Writes through one view
/// are visible through the other.
fn typed_array_subarray(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    const METHOD: &str = "TypedArray.prototype.subarray";
    // No `ValidateTypedArray` here (unlike `slice`): per ECMA-262
    // 22.2.3.27, `subarray` reads the view's *current* length — 0 when
    // detached or out of bounds — converts `begin`/`end` observably
    // against it, and only then runs the species construction, which
    // builds the result over the same buffer. Construction over a
    // detached buffer throws `TypeError` (via the default constructor's
    // detached check); a mid-call `resize` back into bounds (e.g. from a
    // `valueOf`) is honored.
    let Value::Object(view) = this else {
        return Err(NativeError::TypeError(
            "subarray called on incompatible receiver".into(),
        ));
    };
    let Some((buf, offset, len, etype)) = typed_array_parts(heap, view) else {
        return Err(NativeError::TypeError(
            "subarray called on incompatible receiver".into(),
        ));
    };
    let size = etype.size();
    let tracking = heap.get(view).view_length_tracking();
    // The source length as of entry (0 when detached/out of bounds); both
    // clamps below use this single snapshot.
    let src_len = typed_array_resolve(heap, buf, offset, len, size as u64, tracking).unwrap_or(0);
    // Relative begin/end, mirroring `typed_array_slice`. The full
    // conversion runs `valueOf` hooks observably (test262's
    // `subarray/detached-buffer.js` requires the conversions to run
    // before the detached-buffer `TypeError`).
    let rel = |n: f64| -> u64 {
        if n < 0.0 {
            (src_len as f64 + n).max(0.0) as u64
        } else {
            n.min(src_len as f64) as u64
        }
    };
    let from = rel(to_integer_or_infinity_full(heap, get_arg(args, 0))?);
    let end_arg = get_arg(args, 1);
    let to = if end_arg.is_undefined() {
        src_len
    } else {
        rel(to_integer_or_infinity_full(heap, end_arg)?)
    };
    let (from, to) = if from > to { (to, to) } else { (from, to) };
    let new_len = to - from;
    let begin_byte_offset = offset + from * size as u64;
    // `TypedArraySpeciesCreate(O, « buffer, beginByteOffset, newLen »)`
    // (ECMA-262 22.2.3.27): the species lookup runs after the observable
    // begin/end conversions above — a throwing `valueOf` there propagates
    // before any construction error — and `SpeciesConstructor` throws
    // `TypeError` for a non-object `O.constructor` or a non-constructor
    // `@@species`. The species constructor itself builds the result view
    // over the *same* buffer (`new S(buffer, byteOffset, length)`), so a
    // custom species is genuinely used — its return value is the result —
    // and the default path shares the buffer exactly as before. A
    // length-tracking source with omitted `end` passes an omitted length
    // so the result stays length-tracking; otherwise the result is
    // fixed-length. The detached-buffer `TypeError` surfaces from the
    // default constructor's own detached check (a custom species that
    // ignores its arguments observes no detached error, matching
    // engines); likewise the `byteOffset > bufferByteLength` `RangeError`
    // comes from the constructor.
    let len_arg = if tracking && end_arg.is_undefined() {
        Value::Undefined
    } else {
        js_num(new_len as f64)
    };
    // `buf` stays alive through the view's kind-tag edge, reachable from
    // the rooted `this` across the species construction below.
    let av = species_construct_args(
        heap,
        view,
        &[
            Value::Object(buf),
            js_num(begin_byte_offset as f64),
            len_arg,
        ],
        METHOD,
    )?;
    let aobj = require_object(av, METHOD)?;
    // The species result must be a genuine typed array view over a
    // non-detached buffer: a hostile species returning a plain object —
    // or a detached view — throws `TypeError`. (A different element kind
    // is fine; the constructor already built a valid view.)
    let Some((abuf, _, _, _)) = typed_array_parts(heap, aobj) else {
        return Err(NativeError::TypeError(format!(
            "{METHOD}: species constructor did not return a typed array"
        )));
    };
    if is_detached_buffer(heap, abuf) {
        return Err(NativeError::TypeError(format!(
            "{METHOD}: species constructor returned a detached view"
        )));
    }
    Ok(av)
}

/// Spec `Relative` index clamp for `start`/`end` arguments: negative
/// counts from `len`, result in `0..=len`.
fn clamp_index(n: f64, len: u64) -> u64 {
    if n < 0.0 {
        (len as f64 + n).max(0.0) as u64
    } else {
        n.min(len as f64) as u64
    }
}

/// Spec `fromIndex` clamp for `includes`/`indexOf`: negative counts from
/// `len`, result in `0..=len`.
fn clamp_from_index(v: Value, len: u64) -> Result<u64, NativeError> {
    let n = to_integer_or_infinity(v)?;
    Ok(if n < 0.0 {
        (len as f64 + n).max(0.0) as u64
    } else {
        n.min(len as f64) as u64
    })
}

/// Numeric ordering of two heap bigints.
fn bigint_cmp(a: Gc<BigInt>, b: Gc<BigInt>) -> Ordering {
    let az = a.digits.iter().all(|&d| d == 0);
    let bz = b.digits.iter().all(|&d| d == 0);
    if az && bz {
        return Ordering::Equal;
    }
    if az {
        return if b.negative {
            Ordering::Greater
        } else {
            Ordering::Less
        };
    }
    if bz {
        return if a.negative {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    match (a.negative, b.negative) {
        (true, true) => cmp_mag(&b.digits, &a.digits),
        (false, false) => cmp_mag(&a.digits, &b.digits),
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
    }
}

/// Magnitude ordering of two base-2^32 digit slices (least significant
/// first), ignoring high zero words.
fn cmp_mag(a: &[u32], b: &[u32]) -> Ordering {
    let (mut x, mut y) = (a, b);
    while x.len() > 1 && x[x.len() - 1] == 0 {
        x = &x[..x.len() - 1];
    }
    while y.len() > 1 && y[y.len() - 1] == 0 {
        y = &y[..y.len() - 1];
    }
    x.len().cmp(&y.len()).then_with(|| {
        for (&dx, &dy) in x.iter().rev().zip(y.iter().rev()) {
            match dx.cmp(&dy) {
                Ordering::Equal => continue,
                o => return o,
            }
        }
        Ordering::Equal
    })
}

/// A typed-array element as an `f64` for the default sort comparator.
/// Bigints that fit in 64 bits convert exactly; anything bigger saturates
/// (only reachable for hostile hand-built values — genuine bigint
/// elements always fit).
fn ta_value_as_f64(v: Value) -> f64 {
    match v {
        Value::Int(i) => f64::from(i),
        Value::Float(f) => f,
        Value::BigInt(g) => {
            let lo = bigint_low64(g);
            if g.digits.len() <= 2 {
                if g.negative { -(lo as f64) } else { lo as f64 }
            } else if g.negative {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
        }
        _ => f64::NAN,
    }
}

/// The default typed-array sort comparator (ECMA-262 22.2.4.25):
/// numeric ascending, `NaN` sorts after everything.
fn default_ta_cmp(a: Value, b: Value) -> Ordering {
    if let (Value::BigInt(x), Value::BigInt(y)) = (a, b) {
        return bigint_cmp(x, y);
    }
    let x = ta_value_as_f64(a);
    let y = ta_value_as_f64(b);
    if x.is_nan() {
        if y.is_nan() {
            Ordering::Equal
        } else {
            Ordering::Greater
        }
    } else if y.is_nan() {
        Ordering::Less
    } else {
        x.partial_cmp(&y).unwrap_or(Ordering::Equal)
    }
}

/// `%TypedArray%.prototype.at(index)`: the element at the relative
/// index, or `undefined` when out of bounds.
fn typed_array_at(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (view, _, _, len, _) = require_typed_array_view(heap, this, "at")?;
    let n = to_integer_or_infinity(get_arg(args, 0))?;
    let k = if n < 0.0 { len as f64 + n } else { n };
    if k < 0.0 || k >= len as f64 {
        return Ok(Value::Undefined);
    }
    typed_index_get(heap, view, k as u64)
}

/// `%TypedArray%.prototype.copyWithin(target, start, end)`: copies at
/// the byte level through a scratch buffer (spec snapshots the source
/// range first, so overlapping ranges work). Returns `this`.
fn typed_array_copy_within(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let (_view, buf, offset, len, etype) = require_typed_array_view(heap, this, "copyWithin")?;
    let size = etype.size() as u64;
    let target = clamp_index(to_integer_or_infinity(get_arg(args, 0))?, len);
    let start = clamp_index(to_integer_or_infinity(get_arg(args, 1))?, len);
    let end_arg = get_arg(args, 2);
    let end = if end_arg.is_undefined() {
        len
    } else {
        clamp_index(to_integer_or_infinity(end_arg)?, len)
    };
    let count = end.saturating_sub(start).min(len.saturating_sub(target));
    if count == 0 {
        return Ok(this);
    }
    let byte_count = count
        .checked_mul(size)
        .ok_or_else(|| NativeError::RangeError("copyWithin: length overflow".into()))?;
    // Byte traffic only — no `Value`s, so no GC rooting is needed across
    // the copy.
    let mut scratch: Vec<u8> = Vec::new();
    scratch
        .try_reserve(byte_count as usize)
        .map_err(|_| NativeError::OutOfMemory)?;
    scratch.resize(byte_count as usize, 0);
    for (i, b) in scratch.iter_mut().enumerate() {
        *b = buffer_get_byte(heap, buf, offset + start * size + i as u64);
    }
    for (i, b) in scratch.iter().enumerate() {
        buffer_set_byte(heap, buf, offset + target * size + i as u64, *b);
    }
    Ok(this)
}

/// `%TypedArray%.prototype.fill(value, start, end)`: converts `value`
/// once, then stamps the bytes. Returns `this`.
fn typed_array_fill(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (_view, buf, offset, len, etype) = require_typed_array_view(heap, this, "fill")?;
    let bytes = elem_bytes(heap, etype, get_arg(args, 0))?;
    let start = clamp_index(to_integer_or_infinity(get_arg(args, 1))?, len);
    let end_arg = get_arg(args, 2);
    let end = if end_arg.is_undefined() {
        len
    } else {
        clamp_index(to_integer_or_infinity(end_arg)?, len)
    };
    let size = etype.size() as u64;
    for i in start..end.max(start) {
        let addr = offset + i * size;
        for k in 0..size {
            buffer_set_byte(heap, buf, addr + k, bytes[k as usize]);
        }
    }
    Ok(this)
}

/// Shared callback-method prelude: validates the callback and the
/// receiver, returning `(view, len, callback, this_arg)`.
fn ta_callback_prelude(
    heap: &Heap,
    this: Value,
    args: &[Value],
    method: &str,
) -> Result<(Gc<Object>, u64, Value, Value), NativeError> {
    let cb = crate::builtin_object_array::expect_callback(args, method)?;
    let (view, _, _, len, _) = require_typed_array_view(heap, this, method)?;
    Ok((view, len, cb, get_arg(args, 1)))
}

/// `%TypedArray%.prototype.every(callback, thisArg)`.
fn typed_array_every(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (view, len, cb, this_arg) =
        ta_callback_prelude(heap, this, args, "TypedArray.prototype.every")?;
    with_temp_root(heap, this, |heap| {
        for k in 0..len {
            let v = typed_index_get(heap, view, k)?;
            // `call_value` roots its arguments across the call; `v` is
            // dead after this iteration.
            if !call_value(heap, cb, this_arg, &[v, js_num(k as f64), this])?.is_truthy() {
                return Ok(Value::Bool(false));
            }
        }
        Ok(Value::Bool(true))
    })
}

/// `%TypedArray%.prototype.some(callback, thisArg)`.
fn typed_array_some(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (view, len, cb, this_arg) =
        ta_callback_prelude(heap, this, args, "TypedArray.prototype.some")?;
    with_temp_root(heap, this, |heap| {
        for k in 0..len {
            let v = typed_index_get(heap, view, k)?;
            if call_value(heap, cb, this_arg, &[v, js_num(k as f64), this])?.is_truthy() {
                return Ok(Value::Bool(true));
            }
        }
        Ok(Value::Bool(false))
    })
}

/// `%TypedArray%.prototype.forEach(callback, thisArg)`.
fn typed_array_for_each(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let (view, len, cb, this_arg) =
        ta_callback_prelude(heap, this, args, "TypedArray.prototype.forEach")?;
    with_temp_root(heap, this, |heap| {
        for k in 0..len {
            let v = typed_index_get(heap, view, k)?;
            call_value(heap, cb, this_arg, &[v, js_num(k as f64), this])?;
        }
        Ok(Value::Undefined)
    })
}

/// `%TypedArray%.prototype.find(callback, thisArg)`.
fn typed_array_find(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (view, len, cb, this_arg) =
        ta_callback_prelude(heap, this, args, "TypedArray.prototype.find")?;
    with_temp_root(heap, this, |heap| {
        for k in 0..len {
            let v = typed_index_get(heap, view, k)?;
            if call_value(heap, cb, this_arg, &[v, js_num(k as f64), this])?.is_truthy() {
                return Ok(v);
            }
        }
        Ok(Value::Undefined)
    })
}

/// `%TypedArray%.prototype.findIndex(callback, thisArg)`.
fn typed_array_find_index(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let (view, len, cb, this_arg) =
        ta_callback_prelude(heap, this, args, "TypedArray.prototype.findIndex")?;
    with_temp_root(heap, this, |heap| {
        for k in 0..len {
            let v = typed_index_get(heap, view, k)?;
            if call_value(heap, cb, this_arg, &[v, js_num(k as f64), this])?.is_truthy() {
                return Ok(js_num(k as f64));
            }
        }
        Ok(Value::Int(-1))
    })
}

/// `%TypedArray%.prototype.findLast(callback, thisArg)`.
fn typed_array_find_last(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let (view, len, cb, this_arg) =
        ta_callback_prelude(heap, this, args, "TypedArray.prototype.findLast")?;
    with_temp_root(heap, this, |heap| {
        for k in (0..len).rev() {
            let v = typed_index_get(heap, view, k)?;
            if call_value(heap, cb, this_arg, &[v, js_num(k as f64), this])?.is_truthy() {
                return Ok(v);
            }
        }
        Ok(Value::Undefined)
    })
}

/// `%TypedArray%.prototype.findLastIndex(callback, thisArg)`.
fn typed_array_find_last_index(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let (view, len, cb, this_arg) =
        ta_callback_prelude(heap, this, args, "TypedArray.prototype.findLastIndex")?;
    with_temp_root(heap, this, |heap| {
        for k in (0..len).rev() {
            let v = typed_index_get(heap, view, k)?;
            if call_value(heap, cb, this_arg, &[v, js_num(k as f64), this])?.is_truthy() {
                return Ok(js_num(k as f64));
            }
        }
        Ok(Value::Int(-1))
    })
}

/// ECMA-262 `SpeciesConstructor(O, defaultConstructor)`: `C` is
/// `Get(O, "constructor")` — `undefined` returns `defaultConstructor`,
/// a non-object `C` throws `TypeError` — then `S` is `Get(C, @@species)`
/// (getters fire, so `static get [Symbol.species]` accessors run),
/// `undefined`/`null` returns `defaultConstructor`, and a
/// non-constructor `S` throws `TypeError` (`IsConstructor`, not mere
/// callability: a callable non-constructor native like `Array.isArray`
/// throws here instead of being silently "constructed"). Returns the
/// constructor value `S`; the caller `Construct`s it.
pub(crate) fn species_constructor(
    heap: &mut Heap,
    o: Gc<Object>,
    default: Option<Value>,
    method: &str,
) -> Result<Value, NativeError> {
    let ctor = ordinary_get(heap, o, Wtf8::from_str("constructor"), Value::Object(o), 0)?;
    let ctor_obj = match ctor {
        Value::Undefined => {
            return default.ok_or_else(|| {
                NativeError::TypeError(format!("{method}: no default constructor"))
            });
        }
        Value::Object(c) => c,
        _ => {
            return Err(NativeError::TypeError(format!(
                "{method}: constructor is not an object"
            )));
        }
    };
    let species_sym = heap
        .well_known_symbol(WellKnown::Species)
        .map_err(|_| NativeError::OutOfMemory)?;
    let sp = ordinary_get_key(
        heap,
        ctor_obj,
        PropKey::Symbol(species_sym),
        Value::Object(ctor_obj),
        0,
    )?;
    match sp {
        Value::Undefined | Value::Null => default
            .ok_or_else(|| NativeError::TypeError(format!("{method}: no default constructor"))),
        s if is_constructor(heap, s) => Ok(s),
        _ => Err(NativeError::TypeError(format!(
            "{method}: @@species is not a constructor"
        ))),
    }
}

/// `TypedArraySpeciesCreate(O, argumentList)` (ECMA-262 22.2.4.7):
/// `SpeciesConstructor` with the intrinsic default constructor for the
/// view's element type, then `Construct(S, argumentList)`.
fn species_construct_args(
    heap: &mut Heap,
    view: Gc<Object>,
    args: &[Value],
    method: &str,
) -> Result<Value, NativeError> {
    // Default constructor: the intrinsic for the view's element type.
    let default_ctor: Option<Value> = typed_array_parts(heap, view).and_then(|(_, _, _, kind)| {
        let idx = match kind {
            TypedArrayKind::Int8 => 0,
            TypedArrayKind::Uint8 => 1,
            TypedArrayKind::Uint8Clamped => 2,
            TypedArrayKind::Int16 => 3,
            TypedArrayKind::Uint16 => 4,
            TypedArrayKind::Int32 => 5,
            TypedArrayKind::Uint32 => 6,
            TypedArrayKind::Float32 => 7,
            TypedArrayKind::Float64 => 8,
            TypedArrayKind::BigInt64 => 9,
            TypedArrayKind::BigUint64 => 10,
        };
        heap.typed_array_ctors[idx].map(Value::Object)
    });
    let target = species_constructor(heap, view, default_ctor, method)?;
    reflect_construct_inner(heap, target, target, args)
}

/// `TypedArraySpeciesCreate(O, « len »)`: the single-argument (fresh
/// length) form, used by the copying methods (`map`, `filter`, `slice`).
fn species_construct(
    heap: &mut Heap,
    view: Gc<Object>,
    len: u64,
    method: &str,
) -> Result<Value, NativeError> {
    species_construct_args(heap, view, &[js_num(len as f64)], method)
}

/// `%TypedArray%.prototype.map(callback, thisArg)`: species-constructs
/// the result, then stores each mapped value with element conversion
/// (or as a plain indexed property when the species constructor
/// returned a hostile non-view, mirroring `of`/`from`).
fn typed_array_map(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (view, len, cb, this_arg) =
        ta_callback_prelude(heap, this, args, "TypedArray.prototype.map")?;
    let av = species_construct(heap, view, len, "map")?;
    let aobj = require_object(av, "TypedArray.prototype.map")?;
    with_temp_root(heap, av, |heap| {
        with_temp_root(heap, this, |heap| {
            for k in 0..len {
                let v = typed_index_get(heap, view, k)?;
                let mapped = call_value(heap, cb, this_arg, &[v, js_num(k as f64), this])?;
                // `mapped` is fresh from the call and unrooted; the store
                // below can run user code (`valueOf`) that collects.
                let g = heap.protect(mapped);
                let r = set_constructed_elem(heap, aobj, k, mapped);
                heap.release_temp(g);
                r?;
            }
            Ok(())
        })
    })?;
    Ok(av)
}

/// `%TypedArray%.prototype.filter(callback, thisArg)`: collects the
/// kept values first (spec sizes the species result to the kept count),
/// then species-constructs and stores them.
fn typed_array_filter(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (view, len, cb, this_arg) =
        ta_callback_prelude(heap, this, args, "TypedArray.prototype.filter")?;
    // Kept values are temp-rooted as they accumulate: later callbacks
    // can trigger collections.
    let base = heap.protect(this);
    let mut kept: Vec<Value> = Vec::new();
    for k in 0..len {
        let v = typed_index_get(heap, view, k)?;
        if call_value(heap, cb, this_arg, &[v, js_num(k as f64), this])?.is_truthy() {
            heap.protect(v);
            kept.push(v);
        }
    }
    let out = (|| -> Result<Value, NativeError> {
        let av = species_construct(heap, view, kept.len() as u64, "filter")?;
        let aobj = require_object(av, "TypedArray.prototype.filter")?;
        let g = heap.protect(av);
        for (k, kv) in kept.iter().enumerate() {
            let g2 = heap.protect(*kv);
            let r = set_constructed_elem(heap, aobj, k as u64, *kv);
            heap.release_temp(g2);
            r?;
        }
        heap.release_temp(g);
        Ok(av)
    })();
    heap.release_temp(base);
    out
}

/// `%TypedArray%.prototype.reduce(callback, initialValue?)`.
fn typed_array_reduce(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    typed_array_reduce_dir(heap, this, args, "TypedArray.prototype.reduce", false)
}

/// `%TypedArray%.prototype.reduceRight(callback, initialValue?)`.
fn typed_array_reduce_right(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    typed_array_reduce_dir(heap, this, args, "TypedArray.prototype.reduceRight", true)
}

fn typed_array_reduce_dir(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
    method: &str,
    right: bool,
) -> Result<Value, NativeError> {
    let cb = crate::builtin_object_array::expect_callback(args, method)?;
    let (view, _, _, len, _) = require_typed_array_view(heap, this, method)?;
    // An explicitly-passed `undefined` initial value counts as present.
    let (mut acc, mut k) = if args.len() > 1 {
        (get_arg(args, 1), if right { len } else { 0 })
    } else {
        if len == 0 {
            return Err(NativeError::TypeError(format!(
                "{method} of empty typed array with no initial value"
            )));
        }
        if right {
            (typed_index_get(heap, view, len - 1)?, len - 1)
        } else {
            (typed_index_get(heap, view, 0)?, 1)
        }
    };
    with_temp_root(heap, this, |heap| {
        loop {
            let idx = if right {
                if k == 0 {
                    break;
                }
                k -= 1;
                k
            } else {
                if k >= len {
                    break;
                }
                let i = k;
                k += 1;
                i
            };
            // `acc` is live across the read below, which can collect.
            let g = heap.protect(acc);
            let v = typed_index_get(heap, view, idx)?;
            heap.release_temp(g);
            // `call_value` roots its arguments; the fresh `acc` is dead
            // until the next loop-top protect (no allocation between).
            acc = call_value(
                heap,
                cb,
                Value::Undefined,
                &[acc, v, js_num(idx as f64), this],
            )?;
        }
        Ok(acc)
    })
}

/// `%TypedArray%.prototype.reverse()`: in-place byte-level reversal.
/// Returns `this`.
fn typed_array_reverse(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let (_view, buf, offset, len, etype) = require_typed_array_view(heap, this, "reverse")?;
    let size = etype.size() as u64;
    // Byte traffic only — no `Value`s, so no GC rooting is needed.
    let mut tmp = [0u8; 8];
    for i in 0..len / 2 {
        let j = len - 1 - i;
        for k in 0..size {
            tmp[k as usize] = buffer_get_byte(heap, buf, offset + i * size + k);
        }
        for k in 0..size {
            let b = buffer_get_byte(heap, buf, offset + j * size + k);
            buffer_set_byte(heap, buf, offset + i * size + k, b);
        }
        for k in 0..size {
            buffer_set_byte(heap, buf, offset + j * size + k, tmp[k as usize]);
        }
    }
    Ok(this)
}

/// Snapshot the view's elements and sort them: the default numeric
/// comparator, or `comparefn` (called as `comparefn(x, y)`, `NaN` → +0).
/// Every pushed value is temp-rooted immediately: a user `comparefn` can
/// trigger collections mid-sort.
fn sorted_snapshot(
    heap: &mut Heap,
    view: Gc<Object>,
    len: u64,
    cmp: Value,
    method: &str,
) -> Result<Vec<Value>, NativeError> {
    if !matches!(cmp, Value::Undefined) && !is_callable(heap, cmp) {
        return Err(NativeError::TypeError(format!(
            "{method}: comparefn is not callable"
        )));
    }
    let mut vals: Vec<Value> = Vec::new();
    vals.try_reserve(len as usize)
        .map_err(|_| NativeError::OutOfMemory)?;
    for i in 0..len {
        let v = typed_index_get(heap, view, i)?;
        heap.protect(v);
        vals.push(v);
    }
    if matches!(cmp, Value::Undefined) {
        // Stable: `sort_by` is a stable merge sort.
        vals.sort_by(|&a, &b| default_ta_cmp(a, b));
    } else {
        let mut cmp_err: Option<NativeError> = None;
        vals.sort_by(|&a, &b| {
            if cmp_err.is_some() {
                return Ordering::Equal;
            }
            match call_value(heap, cmp, Value::Undefined, &[a, b]) {
                Err(e) => {
                    cmp_err = Some(e);
                    Ordering::Equal
                }
                Ok(v) => match to_number(v) {
                    Err(e) => {
                        cmp_err = Some(e);
                        Ordering::Equal
                    }
                    Ok(n) => {
                        let n = if n.is_nan() { 0.0 } else { n };
                        n.partial_cmp(&0.0).unwrap_or(Ordering::Equal)
                    }
                },
            }
        });
        if let Some(e) = cmp_err {
            return Err(e);
        }
    }
    Ok(vals)
}

/// Write sorted snapshot values into `target` (a genuine view or a
/// hostile species result — `set_constructed_elem` covers both).
/// Every value is already temp-rooted by the caller.
fn store_sorted(heap: &mut Heap, target: Gc<Object>, vals: &[Value]) -> Result<(), NativeError> {
    for (i, v) in vals.iter().enumerate() {
        set_constructed_elem(heap, target, i as u64, *v)?;
    }
    Ok(())
}

/// `%TypedArray%.prototype.sort(comparefn?)`: in-place stable sort.
/// Returns `this`.
fn typed_array_sort(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    const METHOD: &str = "TypedArray.prototype.sort";
    let (view, _, _, len, _) = require_typed_array_view(heap, this, METHOD)?;
    let base = heap.protect(this);
    let out = sorted_snapshot(heap, view, len, get_arg(args, 0), METHOD)
        .and_then(|vals| store_sorted(heap, view, &vals));
    heap.release_temp(base);
    out?;
    Ok(this)
}

/// Same-type result construction for the change-array-by-copy methods
/// (`with`, `toSorted`, `toReversed`): a fresh buffer, the receiver's
/// element kind, the receiver's own prototype. Species and
/// `.constructor` play no role (ECMA-262 `TypedArrayCreateSameType`).
fn new_same_type_array(
    heap: &mut Heap,
    view: Gc<Object>,
    kind: TypedArrayKind,
    len: u64,
    method: &str,
) -> Result<Gc<Object>, NativeError> {
    let proto = heap.get(view).prototype;
    let byte_len = len
        .checked_mul(kind.size() as u64)
        .ok_or_else(|| NativeError::RangeError(format!("{method}: length overflow")))?;
    let new_buf = new_array_buffer(heap, byte_len, None)?;
    new_typed_array(heap, kind, new_buf, 0, len, false, proto)
}

/// `%TypedArray%.prototype.toSorted(comparefn?)`: returns a same-type
/// sorted copy; `this` is untouched. Species is ignored entirely — the
/// result's prototype is the receiver's own prototype — and
/// `.constructor` is never read.
fn typed_array_to_sorted(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    const METHOD: &str = "TypedArray.prototype.toSorted";
    let (view, _, _, len, kind) = require_typed_array_view(heap, this, METHOD)?;
    let base = heap.protect(this);
    let out = (|| -> Result<Value, NativeError> {
        let vals = sorted_snapshot(heap, view, len, get_arg(args, 0), METHOD)?;
        let res = new_same_type_array(heap, view, kind, len, METHOD)?;
        let g = heap.protect(Value::Object(res));
        let r = store_sorted(heap, res, &vals);
        heap.release_temp(g);
        r?;
        Ok(Value::Object(res))
    })();
    heap.release_temp(base);
    out
}

/// `%TypedArray%.prototype.toReversed()`: returns a same-type reversed
/// copy; `this` is untouched. Species is ignored entirely — the result's
/// prototype is the receiver's own prototype — and `.constructor` is
/// never read.
fn typed_array_to_reversed(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    const METHOD: &str = "TypedArray.prototype.toReversed";
    let (view, _, _, len, kind) = require_typed_array_view(heap, this, METHOD)?;
    let res = new_same_type_array(heap, view, kind, len, METHOD)?;
    let av = Value::Object(res);
    with_temp_root(heap, av, |heap| {
        with_temp_root(heap, this, |heap| {
            for i in 0..len {
                let v = typed_index_get(heap, view, i)?;
                let g = heap.protect(v);
                let r = typed_index_set(heap, res, len - 1 - i, v);
                heap.release_temp(g);
                r?;
            }
            Ok(())
        })
    })?;
    Ok(av)
}

/// `%TypedArray%.prototype.with(index, value)` (change-array-by-copy):
/// returns a same-type copy with one element replaced. Ignores species
/// entirely — the result's prototype is the receiver's own prototype —
/// and never reads `.constructor`.
fn typed_array_with(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    const METHOD: &str = "TypedArray.prototype.with";
    let (view, buf, byte_offset, len, kind) = require_typed_array_view(heap, this, METHOD)?;
    // `ToIntegerOrInfinity(index)`, then the negative-index wrap; the
    // spec validates against the current length.
    let rel = to_integer_or_infinity(get_arg(args, 0))?;
    let actual = if rel >= 0.0 { rel } else { len as f64 + rel };
    if actual < 0.0 || actual >= len as f64 {
        return Err(NativeError::RangeError(format!(
            "{METHOD}: index out of range"
        )));
    }
    let actual = actual as u64;
    // Convert the replacement value before creating the result (spec
    // order): `ToBigInt` for bigint kinds, `ToNumber` otherwise, with
    // `valueOf` hooks honored.
    let converted = if kind.is_bigint() {
        to_bigint(heap, get_arg(args, 1))?
    } else {
        Value::Float(to_number_full(heap, get_arg(args, 1))?)
    };
    // Same-type result: same element kind, receiver's own prototype —
    // species and `.constructor` play no role.
    let proto = heap.get(view).prototype;
    let byte_len = len
        .checked_mul(kind.size() as u64)
        .ok_or_else(|| NativeError::RangeError(format!("{METHOD}: length overflow")))?;
    let new_buf = new_array_buffer(heap, byte_len, None)?;
    let res = new_typed_array(heap, kind, new_buf, 0, len, false, proto)?;
    with_temp_root(heap, Value::Object(res), |heap| {
        with_temp_root(heap, converted, |heap| {
            // Same-kind byte copy, then the single converted element.
            for i in 0..byte_len {
                let b = buffer_get_byte(heap, buf, byte_offset + i);
                buffer_set_byte(heap, new_buf, i, b);
            }
            typed_index_set(heap, res, actual, converted)?;
            Ok(())
        })
    })?;
    Ok(Value::Object(res))
}

/// `%TypedArray%.prototype.join(separator?)`: `ToString` per element.
/// `toString` is `join(",")`; `toLocaleString` renders the same way for
/// numeric/bigint elements (v1: no locale-specific number formatting).
///
/// `pub(crate)`: `Array.prototype.join` delegates typed-array receivers
/// here, since `%TypedArray%.prototype.toString` is the same function
/// object as `Array.prototype.join` (ECMA-262 22.1.3.27 / 22.2.3.32).
pub(crate) fn typed_array_join(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let (view, _, _, len, _) = require_typed_array_view(heap, this, "join")?;
    let sep_arg = get_arg(args, 0);
    let sep = if matches!(sep_arg, Value::Undefined) {
        Wtf8Buf::from(",")
    } else {
        to_string_js(heap, sep_arg)?
    };
    with_temp_root(heap, this, |heap| {
        let mut out = Wtf8Buf::new();
        for k in 0..len {
            if k > 0 {
                out.push_wtf8(&sep);
            }
            let v = typed_index_get(heap, view, k)?;
            let s = elem_js_string(v);
            out.push_wtf8(&s);
        }
        let s = heap
            .alloc_str_wtf8(&out)
            .map_err(|_| NativeError::OutOfMemory)?;
        Ok(Value::Str(s))
    })
}

/// `%TypedArray%.prototype.toString()`: `Array.prototype.toString`
/// semantics — the same function object behavior as `join(",")`
/// (ECMA-262 22.2.3.32: the initial value is `Array.prototype.toString`).
fn typed_array_to_string(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    typed_array_join(heap, this, &[])
}

/// `%TypedArray%.prototype.toLocaleString()`.
fn typed_array_to_locale_string(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    typed_array_join(heap, this, &[])
}

/// `%TypedArray%.prototype.includes(searchElement, fromIndex?)`:
/// `SameValueZero` comparison (`NaN` matches `NaN`).
fn typed_array_includes(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let (view, _, _, len, etype) = require_typed_array_view(heap, this, "includes")?;
    let from = clamp_from_index(get_arg(args, 1), len)?;
    if etype.is_bigint() {
        let search = to_bigint(heap, get_arg(args, 0))?;
        let Value::BigInt(sg) = search else {
            return Err(NativeError::TypeError(
                "ToBigInt did not produce a BigInt".into(),
            ));
        };
        let base = heap.protect(Value::BigInt(sg));
        let found = with_temp_root(heap, this, |heap| {
            for k in from..len {
                let v = typed_index_get(heap, view, k)?;
                if let Value::BigInt(g) = v
                    && bigint_eq(g, sg)
                {
                    return Ok(true);
                }
            }
            Ok(false)
        })?;
        heap.release_temp(base);
        Ok(Value::Bool(found))
    } else {
        let search = to_number(get_arg(args, 0))?;
        with_temp_root(heap, this, |heap| {
            for k in from..len {
                let v = typed_index_get(heap, view, k)?;
                let n = match v {
                    Value::Int(i) => f64::from(i),
                    Value::Float(f) => f,
                    _ => continue,
                };
                if n == search || (n.is_nan() && search.is_nan()) {
                    return Ok(Value::Bool(true));
                }
            }
            Ok(Value::Bool(false))
        })
    }
}

/// `%TypedArray%.prototype.indexOf(searchElement, fromIndex?)`:
/// `StrictEqual` comparison (`NaN` never matches).
fn typed_array_index_of(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    typed_array_index_of_dir(heap, this, args, "indexOf", false)
}

/// `%TypedArray%.prototype.lastIndexOf(searchElement, fromIndex?)`.
fn typed_array_last_index_of(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    typed_array_index_of_dir(heap, this, args, "lastIndexOf", true)
}

fn typed_array_index_of_dir(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
    method: &str,
    last: bool,
) -> Result<Value, NativeError> {
    let (view, _, _, len, etype) = require_typed_array_view(heap, this, method)?;
    if len == 0 {
        return Ok(Value::Int(-1));
    }
    // Search bounds: forward scans `from..len`, backward scans `0..=from`.
    let (fwd_from, bwd_from): (u64, u64) = if last {
        let from_arg = get_arg(args, 1);
        let bwd = if from_arg.is_undefined() {
            len - 1
        } else {
            let n = to_integer_or_infinity(from_arg)?;
            if n < 0.0 {
                let c = len as f64 + n;
                if c < 0.0 {
                    return Ok(Value::Int(-1));
                }
                (c as u64).min(len - 1)
            } else {
                (n as u64).min(len - 1)
            }
        };
        (0, bwd)
    } else {
        (clamp_from_index(get_arg(args, 1), len)?, 0)
    };
    if etype.is_bigint() {
        let search = to_bigint(heap, get_arg(args, 0))?;
        let Value::BigInt(sg) = search else {
            return Err(NativeError::TypeError(
                "ToBigInt did not produce a BigInt".into(),
            ));
        };
        let base = heap.protect(Value::BigInt(sg));
        let found = with_temp_root(heap, this, |heap| {
            if last {
                for k in (0..=bwd_from).rev() {
                    let v = typed_index_get(heap, view, k)?;
                    if let Value::BigInt(g) = v
                        && bigint_eq(g, sg)
                    {
                        return Ok(Some(k));
                    }
                }
            } else {
                for k in fwd_from..len {
                    let v = typed_index_get(heap, view, k)?;
                    if let Value::BigInt(g) = v
                        && bigint_eq(g, sg)
                    {
                        return Ok(Some(k));
                    }
                }
            }
            Ok(None)
        })?;
        heap.release_temp(base);
        Ok(match found {
            Some(k) => js_num(k as f64),
            None => Value::Int(-1),
        })
    } else {
        let search = to_number(get_arg(args, 0))?;
        with_temp_root(heap, this, |heap| {
            let hit = |v: Value| match v {
                // `==` on `f64`: `-0 == +0`, `NaN != NaN` — exactly
                // `StrictEqual` for numbers.
                Value::Int(i) => f64::from(i) == search,
                Value::Float(f) => f == search,
                _ => false,
            };
            if last {
                for k in (0..=bwd_from).rev() {
                    if hit(typed_index_get(heap, view, k)?) {
                        return Ok(js_num(k as f64));
                    }
                }
            } else {
                for k in fwd_from..len {
                    if hit(typed_index_get(heap, view, k)?) {
                        return Ok(js_num(k as f64));
                    }
                }
            }
            Ok(Value::Int(-1))
        })
    }
}

// ---------------------------------------------------------------------------
// `%TypedArray%` iterator objects (`keys`/`values`/`entries`/`@@iterator`)
// ---------------------------------------------------------------------------

/// Hidden slots on a `%TypedArray%` iterator object (same pattern as the
/// `Map`/`Set` iterators in `builtin_collections`).
const TA_ITER_VIEW: &str = "\u{0}bowser:ta-iter-view";
/// The viewed typed array object.
const TA_ITER_INDEX: &str = "\u{0}bowser:ta-iter-index";
/// `Int`/`Float` next index (float past 2^31).
const TA_ITER_KIND: &str = "\u{0}bowser:ta-iter-kind";
/// `Int`: 0 = keys, 1 = values, 2 = entries.
/// Allocate a `%TypedArray%` iterator over `view` (already validated).
fn new_ta_iterator(heap: &mut Heap, view: Gc<Object>, kind: u8) -> Result<Value, NativeError> {
    let iter = heap
        .alloc_object(Object::new(ObjectKind::Ordinary))
        .map_err(|_| NativeError::OutOfMemory)?;
    heap.set_prototype(iter, Value::Null);
    with_temp_root(heap, Value::Object(iter), |heap| {
        with_temp_root(heap, Value::Object(view), |heap| {
            define_hidden(heap, iter, TA_ITER_VIEW, Value::Object(view))?;
            define_hidden(heap, iter, TA_ITER_INDEX, Value::Int(0))?;
            define_hidden(heap, iter, TA_ITER_KIND, Value::Int(i32::from(kind)))?;
            // Non-enumerable `next`, like every other builtin method.
            define_ta_method(heap, iter, "next", ta_iter_next, 0)?;
            // Iterators are iterable: `for (x of ta.values())` — the
            // real symbol key, so a computed `it[Symbol.iterator]` key
            // resolves to the same property.
            let self_fn = new_native_function(heap, "[Symbol.iterator]", ta_iter_self, 0)?;
            let iterator_sym = heap
                .well_known_symbol(WellKnown::Iterator)
                .map_err(|_| NativeError::OutOfMemory)?;
            heap.define_property_key(
                iter,
                PropKey::Symbol(iterator_sym),
                Property::data(Value::Object(self_fn)),
            )
            .map_err(|_| NativeError::OutOfMemory)?;
            Ok(())
        })
    })?;
    Ok(Value::Object(iter))
}

/// Build a `{value, done}` result object (writable/enumerable/
/// configurable, per spec `CreateIterResultObject`).
fn iter_result(heap: &mut Heap, value: Value, done: bool) -> Result<Value, NativeError> {
    let obj = heap
        .alloc_object(Object::new(ObjectKind::Ordinary))
        .map_err(|_| NativeError::OutOfMemory)?;
    heap.set_prototype(obj, Value::Null);
    let g = heap.protect(Value::Object(obj));
    let r = (|| -> Result<Value, NativeError> {
        heap.define_property_str(
            obj,
            "value",
            Property {
                value,
                writable: true,
                enumerable: true,
                configurable: true,
            },
        )
        .map_err(|_| NativeError::OutOfMemory)?;
        heap.define_property_str(
            obj,
            "done",
            Property {
                value: Value::Bool(done),
                writable: true,
                enumerable: true,
                configurable: true,
            },
        )
        .map_err(|_| NativeError::OutOfMemory)?;
        Ok(Value::Object(obj))
    })();
    heap.release_temp(g);
    r
}

fn ta_iter_self(_heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    match this {
        Value::Object(_) => Ok(this),
        _ => Err(NativeError::TypeError(
            "@@iterator called on incompatible receiver".into(),
        )),
    }
}

fn ta_iter_next(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let incompatible =
        || NativeError::TypeError("iterator.next called on incompatible receiver".into());
    let Value::Object(iter) = this else {
        return Err(incompatible());
    };
    let slot = |name: &str| {
        heap.get_property(iter, Wtf8::from_str(name))
            .map(|p| p.value)
    };
    let (view, index, kind) = match (slot(TA_ITER_VIEW), slot(TA_ITER_INDEX), slot(TA_ITER_KIND)) {
        (Some(Value::Object(v)), idx, Some(Value::Int(k))) => {
            let i = match idx {
                Some(Value::Int(n)) => n.max(0) as u64,
                Some(Value::Float(f)) => f.max(0.0) as u64,
                _ => return Err(incompatible()),
            };
            (v, i, k)
        }
        _ => return Err(incompatible()),
    };
    let Some((_buf, _offset, len, _etype)) = typed_array_parts(heap, view) else {
        return Err(incompatible());
    };
    if index >= len {
        return iter_result(heap, Value::Undefined, true);
    }
    let value = match kind {
        0 => js_num(index as f64),
        1 => typed_index_get(heap, view, index)?,
        _ => {
            let v = typed_index_get(heap, view, index)?;
            let g = heap.protect(v);
            let pair = new_array_from_opt(heap, &[Some(js_num(index as f64)), Some(v)])
                .map_err(|_| NativeError::OutOfMemory)?;
            heap.release_temp(g);
            Value::Object(pair)
        }
    };
    // `value` may hold a fresh bigint; hold it across the slot write and
    // the result allocation below.
    let g = heap.protect(value);
    define_hidden(heap, iter, TA_ITER_INDEX, js_num(index as f64 + 1.0))?;
    let r = iter_result(heap, value, false);
    heap.release_temp(g);
    r
}

/// `%TypedArray%.prototype.keys()`.
fn typed_array_keys(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let (view, _, _, _, _) = require_typed_array_view(heap, this, "keys")?;
    new_ta_iterator(heap, view, 0)
}

/// `%TypedArray%.prototype.values()`.
fn typed_array_values(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let (view, _, _, _, _) = require_typed_array_view(heap, this, "values")?;
    new_ta_iterator(heap, view, 1)
}

/// `%TypedArray%.prototype.entries()`.
fn typed_array_entries(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let (view, _, _, _, _) = require_typed_array_view(heap, this, "entries")?;
    new_ta_iterator(heap, view, 2)
}

// ---------------------------------------------------------------------------
// `%TypedArray%.prototype` length/byteLength/byteOffset accessors
// ---------------------------------------------------------------------------

/// `get %TypedArray%.prototype.length` (ECMA-262 22.2.3.18). No
/// `ValidateTypedArray`: a detached or out-of-bounds view reports 0
/// (never throws); length-tracking views recompute from the buffer's
/// current byte length.
fn typed_array_length_get(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let Value::Object(view) = this else {
        return Err(NativeError::TypeError(
            "get length called on incompatible receiver".into(),
        ));
    };
    let Some((buf, offset, len, etype)) = typed_array_parts(heap, view) else {
        return Err(NativeError::TypeError(
            "get length called on incompatible receiver".into(),
        ));
    };
    let tracking = heap.get(view).view_length_tracking();
    let n = typed_array_resolve(heap, buf, offset, len, etype.size() as u64, tracking).unwrap_or(0);
    Ok(js_num(n as f64))
}

/// `get %TypedArray%.prototype.byteLength` (ECMA-262 22.2.3.16): like
/// `length`, in bytes.
fn typed_array_byte_length_get(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let Value::Object(view) = this else {
        return Err(NativeError::TypeError(
            "get byteLength called on incompatible receiver".into(),
        ));
    };
    let Some((buf, offset, len, etype)) = typed_array_parts(heap, view) else {
        return Err(NativeError::TypeError(
            "get byteLength called on incompatible receiver".into(),
        ));
    };
    let tracking = heap.get(view).view_length_tracking();
    let n = typed_array_resolve(heap, buf, offset, len, etype.size() as u64, tracking).unwrap_or(0);
    Ok(js_num(n as f64 * etype.size() as f64))
}

/// `get %TypedArray%.prototype.byteOffset` (ECMA-262 22.2.3.17): the
/// fixed offset, or 0 when detached/out of bounds.
fn typed_array_byte_offset_get(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let Value::Object(view) = this else {
        return Err(NativeError::TypeError(
            "get byteOffset called on incompatible receiver".into(),
        ));
    };
    let Some((buf, offset, len, etype)) = typed_array_parts(heap, view) else {
        return Err(NativeError::TypeError(
            "get byteOffset called on incompatible receiver".into(),
        ));
    };
    let tracking = heap.get(view).view_length_tracking();
    if typed_array_resolve(heap, buf, offset, len, etype.size() as u64, tracking).is_none() {
        return Ok(js_num(0.0));
    }
    Ok(js_num(offset as f64))
}

/// Define `target[name]` as a getter-only accessor (no setter), for the
/// `%TypedArray%.prototype` accessor family. The getter function's
/// `name` is `"get <name>"`, per ECMA-262's accessor naming.
fn define_getter(
    heap: &mut Heap,
    target: Gc<Object>,
    name: &str,
    getter: NativeFn,
) -> Result<(), NativeError> {
    // Hold `target` across the function allocation, which may collect.
    let f = with_temp_root(heap, Value::Object(target), |heap| {
        new_native_function(heap, &format!("get {name}"), getter, 0)
    })?;
    heap.define_accessor_str(
        target,
        name,
        AccessorProp {
            get: Value::Object(f),
            set: Value::Undefined,
            enumerable: false,
            configurable: true,
        },
    )
    .map_err(|_| NativeError::OutOfMemory)?;
    Ok(())
}

/// Spec `IsConstructor(C)` gate shared by `%TypedArray%.of`/`from`:
/// `this` must be a constructor. Bytecode arrows and friends pass this
/// gate and throw `TypeError` from the VM's construct path; callable
/// non-constructor natives (e.g. `Array.isArray`) throw here.
fn require_typed_array_ctor(heap: &Heap, ctor: Value, method: &str) -> Result<(), NativeError> {
    if !matches!(ctor, Value::Object(_)) || !is_constructor(heap, ctor) {
        return Err(NativeError::TypeError(format!(
            "TypedArray.{method}: `this` is not a constructor"
        )));
    }
    Ok(())
}

/// Store one element into a fresh `of`/`from` result: the typed array's
/// integer-indexed [[Set]] (element conversion applied) on a real view;
/// a plain data property when `this` was a hostile constructor whose
/// result is not a view (spec `Set`).
fn set_constructed_elem(
    heap: &mut Heap,
    obj: Gc<Object>,
    i: u64,
    v: Value,
) -> Result<(), NativeError> {
    if is_typed_array_view(heap, obj) {
        typed_index_set(heap, obj, i, v)
    } else {
        heap.define_property_str(obj, &i.to_string(), Property::data(v))?;
        Ok(())
    }
}

/// Shared `%TypedArray%.of`/`from` construction: the instance comes from
/// `Construct(C, « len »)` (via `reflect_construct_inner`), so
/// subclassing is respected — `class D extends Uint8Array {}; D.of(1)`
/// runs `D`'s constructor with `newTarget = D`, and the view gets
/// `D.prototype` through the `NEW_TARGET` thread-local.
fn construct_typed_array_static(
    heap: &mut Heap,
    ctor: Value,
    len: u64,
) -> Result<Value, NativeError> {
    reflect_construct_inner(heap, ctor, ctor, &[js_num(len as f64)])
}

/// `%TypedArray%.of(...items)` (ECMA-262 22.2.5): `C` is `this`;
/// construct `C(len)` and store each argument with element conversion.
fn typed_array_of(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    require_typed_array_ctor(heap, this, "of")?;
    let new_obj = construct_typed_array_static(heap, this, args.len() as u64)?;
    let view = require_object(new_obj, "TypedArray.of")?;
    // The fresh view is not yet reachable from any root, and `to_number`
    // inside `typed_index_set` can run user code (`valueOf`) that
    // collects: hold it across the stores.
    with_temp_root(heap, new_obj, |heap| {
        for (i, v) in args.iter().enumerate() {
            set_constructed_elem(heap, view, i as u64, *v)?;
        }
        Ok(())
    })?;
    Ok(new_obj)
}

/// `%TypedArray%.from(source, mapfn, thisArg)` (ECMA-262 22.2.6):
/// typed-array sources read through the integer-indexed exotic path;
/// array-like sources go through `ToLength(length)` plus indexed gets
/// (v1: no iterator protocol, mirroring `Array.from`). Construction goes
/// through `this` exactly like `of`, so subclassing is respected.
fn typed_array_from(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    require_typed_array_ctor(heap, this, "from")?;
    let map_fn = get_arg(args, 1);
    let has_map = !matches!(map_fn, Value::Undefined);
    if has_map && !is_callable(heap, map_fn) {
        return Err(NativeError::TypeError(
            "TypedArray.from: map function is not callable".into(),
        ));
    }
    let this_arg = get_arg(args, 2);
    let src = match get_arg(args, 0) {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "TypedArray.from: source must be an object".into(),
            ));
        }
    };
    // Spec `%TypedArray%.from`: the `@@iterator` method is consulted
    // before the array-like path. An iterable source is fully consumed
    // first (so `len` is known), then the result is constructed.
    if let Some((values, base)) = iterable_values(heap, src)? {
        let len = values.len() as u64;
        let new_obj = construct_typed_array_static(heap, this, len)?;
        let view = require_object(new_obj, "TypedArray.from")?;
        let out = with_temp_root(heap, new_obj, |heap| {
            for (k, v) in values.iter().enumerate() {
                let mut v = *v;
                if has_map {
                    v = call_value(heap, map_fn, this_arg, &[v, js_num(k as f64)])?;
                }
                // `v` may be a fresh unrooted value (mapfn return), and
                // the store below can collect (interning/shape
                // transitions); root it across the store (see
                // `typed_array_map`).
                let g = heap.protect(v);
                let r = set_constructed_elem(heap, view, k as u64, v);
                heap.release_temp(g);
                r?;
            }
            Ok(new_obj)
        });
        heap.release_temp(base);
        return out;
    }
    // Length first, so the constructor sees the spec's `« 𝔽(len) »`: a
    // typed-array source reports its element count via the kind tag; an
    // array-like goes through `ToLength(Get(length))` (getters fire).
    let (len, from_view) = match typed_array_parts(heap, src) {
        Some((_, _, src_len, _)) => (src_len, true),
        None => (array_like_length_ordinary(heap, src)?, false),
    };
    let new_obj = construct_typed_array_static(heap, this, len)?;
    let view = require_object(new_obj, "TypedArray.from")?;
    // The fresh view is not yet reachable from any root, and the map
    // function below can allocate and collect: hold it across the loop.
    // `src` is `args[0]`, rooted by the VM's `call_roots`.
    with_temp_root(heap, new_obj, |heap| {
        for k in 0..len {
            let mut v = if from_view {
                typed_index_get(heap, src, k)?
            } else {
                array_like_get_ordinary(heap, src, k)?
            };
            if has_map {
                v = call_value(heap, map_fn, this_arg, &[v, js_num(k as f64)])?;
            }
            // `v` may be a fresh unrooted value (mapfn return, or a
            // materialized BigInt from `typed_index_get`); the store
            // below can collect, so root it across the store (see
            // `typed_array_map`).
            let g = heap.protect(v);
            let r = set_constructed_elem(heap, view, k, v);
            heap.release_temp(g);
            r?;
        }
        Ok(())
    })?;
    Ok(new_obj)
}

/// `%TypedArray%.prototype.buffer` getter: the view's backing store.
/// Browsers expose `buffer` as a prototype accessor, not as an own data
/// property — which is why neither construction path defines one.
fn typed_array_buffer_get(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    // The buffer is returned even when detached or out of bounds (spec:
    // no detached/OOB check in `get %TypedArray%.prototype.buffer`).
    let Value::Object(view) = this else {
        return Err(NativeError::TypeError(
            "get buffer called on incompatible receiver".into(),
        ));
    };
    let Some((buf, _, _, _)) = typed_array_parts(heap, view) else {
        return Err(NativeError::TypeError(
            "get buffer called on incompatible receiver".into(),
        ));
    };
    Ok(Value::Object(buf))
}

/// The installed `%Uint8Array.prototype%`, resolved from the global
/// (`globalThis.Uint8Array.prototype`): the canonical prototype native
/// view builders proto-link to. `Value::Null` when the typed-array install
/// did not run, mirroring the v1 constructor fallback.
pub(crate) fn uint8array_prototype(heap: &Heap, global: Gc<Object>) -> Value {
    match heap.get_property(global, Wtf8::from_str("Uint8Array")) {
        Some(p) => match p.value {
            Value::Object(ctor) => match heap.get_property(ctor, Wtf8::from_str("prototype")) {
                Some(pp) if matches!(pp.value, Value::Object(_)) => pp.value,
                _ => Value::Null,
            },
            _ => Value::Null,
        },
        None => Value::Null,
    }
}

/// GetPrototypeFromConstructor(newTarget, `%C.prototype%`): resolve the
/// instance prototype from the active `new.target`'s `"prototype"`
/// property. A direct `new C(...)` resolves the constructor's installed
/// prototype; `class D extends C` subclass construction resolves
/// `D.prototype` (so instances link to the subclass prototype per
/// ECMA-262 10.2.2 OrdinaryCreateFromConstructor). `None` covers plain
/// native calls in tests and yields `Null`.
fn proto_from_new_target(heap: &Heap) -> Value {
    match crate::vm::active_new_target() {
        Some(Value::Object(nt)) => match get_prop_chain(heap, nt, Wtf8::from_str("prototype")) {
            Some(Value::Object(p)) => Value::Object(p),
            _ => Value::Null,
        },
        _ => Value::Null,
    }
}

macro_rules! ta_constructor {
    ($name:ident, $idx:expr) => {
        fn $name(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
            // GetPrototypeFromConstructor(newTarget, "%<T>Array.prototype%"):
            // `class D extends Uint8Array` instances get `D.prototype`;
            // direct construction resolves the installed per-type
            // prototype (all nine typed-array types get one from the
            // constructor install loop below). The VM stashes newTarget
            // in a thread-local for native constructors (see
            // `NEW_TARGET` in vm.rs); `None` covers direct native calls
            // in tests.
            let proto = proto_from_new_target(heap);
            typed_array_construct(heap, $idx, args, proto)
        }
    };
}

ta_constructor!(int8array_constructor, 0);
ta_constructor!(uint8array_constructor, UINT8_ETYPE);
ta_constructor!(uint8clampedarray_constructor, 2);
ta_constructor!(int16array_constructor, 3);
ta_constructor!(uint16array_constructor, 4);
ta_constructor!(int32array_constructor, 5);
ta_constructor!(uint32array_constructor, 6);
ta_constructor!(float32array_constructor, 7);
ta_constructor!(float64array_constructor, 8);
ta_constructor!(bigint64array_constructor, 9);
ta_constructor!(biguint64array_constructor, 10);

/// The abstract `%TypedArray%` constructor (ECMA-262 22.2.2): calling or
/// constructing it always throws a `TypeError`. Installed with `length`
/// 0, `name` `"TypedArray"`, and `prototype` → the shared
/// `%TypedArray%.prototype`.
fn typed_array_abstract_constructor(
    _heap: &mut Heap,
    _this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    Err(NativeError::TypeError(
        "Abstract class TypedArray not directly constructible".into(),
    ))
}

/// `static get [Symbol.species]` on each concrete typed-array
/// constructor (ECMA-262 22.2.6.2): returns `this`, so subclass
/// constructors propagate through `map`/`filter`/`slice`/etc.
fn typed_array_species_get(
    _heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    Ok(this)
}

// ---------------------------------------------------------------------------
// DataView
// ---------------------------------------------------------------------------

/// True when `v` is a genuine `DataView`: the unforgeable
/// [`ObjectKind::DataView`] tag. A plain object with NUL-prefixed
/// properties of the old slot names is *not* a `DataView`.
fn is_data_view(heap: &Heap, v: Value) -> bool {
    matches!(v, Value::Object(o) if is_data_view_obj(heap, o))
}

/// True when `o` is a genuine `DataView`. `pub(crate)` for
/// `crate::bindings_encoding`'s `ArrayBuffer.isView` semantics.
pub(crate) fn is_data_view_obj(heap: &Heap, o: Gc<Object>) -> bool {
    matches!(heap.get(o).kind, ObjectKind::DataView { .. })
}

/// Decompose a `DataView` into `(buffer, byte_offset, byte_len)`: one tag
/// read, no property lookups.
fn data_view_parts(heap: &Heap, view: Gc<Object>) -> Option<(Gc<Object>, u64, u64)> {
    match heap.get(view).kind {
        ObjectKind::DataView {
            buffer,
            byte_offset,
            byte_len,
        } => Some((buffer, u64::from(byte_offset), u64::from(byte_len))),
        _ => None,
    }
}

fn dataview_constructor(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let buf_val = get_arg(args, 0);
    if !is_array_buffer(heap, buf_val) {
        return Err(NativeError::TypeError(
            "DataView: argument must be an ArrayBuffer".into(),
        ));
    }
    let buf = require_object(buf_val, "DataView")?;
    // Spec order: `ToIndex(byteOffset)` first — its `valueOf` hook runs
    // even on a detached buffer — then the detached check, then the
    // range check. (After the `byteLength` conversion the order flips:
    // the stale range check runs before the second detached check, per
    // ES2026 25.3.2.1 steps 9.2/11.)
    let offset_arg = get_arg(args, 1);
    let byte_offset = if offset_arg.is_undefined() {
        0
    } else {
        to_index_full(heap, offset_arg)?
    };
    if is_detached_buffer(heap, buf) {
        return Err(NativeError::TypeError(
            "DataView: ArrayBuffer is detached".into(),
        ));
    }
    let buf_len = array_buffer_length(heap, buf);
    if byte_offset > buf_len {
        return Err(NativeError::RangeError(
            "DataView: byte offset out of range".into(),
        ));
    }
    let len_arg = get_arg(args, 2);
    // Length-tracking (ECMA-262 `undefined` [[ByteLength]]): omitted
    // length over a resizable buffer. `byteLength` recomputes from the
    // buffer's current byte length (see [`dataview_resolve`]).
    let length_tracking = len_arg.is_undefined() && heap.get(buf).ab_max_byte_length().is_some();
    let byte_len = if len_arg.is_undefined() {
        buf_len - byte_offset
    } else {
        to_index_full(heap, len_arg)?
    };
    // Spec order (ES2026 25.3.2.1 steps 9.2/11): the range check against
    // the *stale* `bufferByteLength` read above runs before the second
    // detached check.
    if byte_offset + byte_len > buf_len {
        return Err(NativeError::RangeError(
            "DataView: view out of buffer range".into(),
        ));
    }
    if is_detached_buffer(heap, buf) {
        return Err(NativeError::TypeError(
            "DataView: ArrayBuffer is detached".into(),
        ));
    }
    let mut obj = Object::new(ObjectKind::DataView {
        buffer: buf,
        byte_offset: u32::try_from(byte_offset)
            .map_err(|_| NativeError::RangeError("DataView: byte offset out of range".into()))?,
        byte_len: u32::try_from(byte_len)
            .map_err(|_| NativeError::RangeError("DataView: length out of range".into()))?,
    });
    obj.set_view_length_tracking(length_tracking);
    let view = heap.alloc_object(obj)?;
    // GetPrototypeFromConstructor(newTarget, "%DataView.prototype%"):
    // a direct `new DataView(ab)` links to `dv_proto` (the constructor's
    // installed prototype); `class D extends DataView` instances link to
    // `D.prototype`. The tag edge (`ObjectKind::DataView`) still carries
    // the view metadata, so GC reachability is unchanged.
    heap.set_prototype(view, proto_from_new_target(heap));
    // The own `buffer` data property stays: `crate::builtin_typedarray`'s
    // `view_buffer` recovers a DataView's buffer from it (typed-array
    // views carry no own `buffer`; they resolve through the prototype
    // accessor). `byteLength` / `byteOffset` are *not* own properties:
    // they resolve through the `get DataView.prototype.*` accessors
    // (ECMA-262 24.3.4.1–3), which consult the buffer's detached and
    // out-of-bounds state on every read.
    heap.define_property_str(
        view,
        "buffer",
        Property {
            value: Value::Object(buf),
            writable: false,
            enumerable: false,
            configurable: false,
        },
    )?;
    Ok(Value::Object(view))
}

/// `get DataView.prototype.buffer` (ECMA-262 24.3.4.1): the viewed
/// buffer, returned even when detached.
fn dataview_buffer_get(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let view = match this {
        Value::Object(o) if is_data_view_obj(heap, o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "get buffer called on incompatible receiver".into(),
            ));
        }
    };
    let Some((buf, _, _)) = data_view_parts(heap, view) else {
        return Err(NativeError::TypeError("bad DataView".into()));
    };
    Ok(Value::Object(buf))
}

/// `get DataView.prototype.byteLength` (ECMA-262 24.3.4.2): throws
/// `TypeError` when detached or out of bounds; recomputes for
/// length-tracking views.
fn dataview_byte_length_get(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let view = match this {
        Value::Object(o) if is_data_view_obj(heap, o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "get byteLength called on incompatible receiver".into(),
            ));
        }
    };
    let Some((buf, offset, len)) = data_view_parts(heap, view) else {
        return Err(NativeError::TypeError("bad DataView".into()));
    };
    let tracking = heap.get(view).view_length_tracking();
    Ok(js_num(
        dataview_resolve(heap, buf, offset, len, tracking)? as f64
    ))
}

/// `get DataView.prototype.byteOffset` (ECMA-262 24.3.4.3): throws
/// `TypeError` when detached or out of bounds, else the fixed offset.
fn dataview_byte_offset_get(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let view = match this {
        Value::Object(o) if is_data_view_obj(heap, o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "get byteOffset called on incompatible receiver".into(),
            ));
        }
    };
    let Some((buf, offset, len)) = data_view_parts(heap, view) else {
        return Err(NativeError::TypeError("bad DataView".into()));
    };
    let tracking = heap.get(view).view_length_tracking();
    dataview_resolve(heap, buf, offset, len, tracking)?;
    Ok(js_num(offset as f64))
}

/// Element flavors for DataView accessors.
#[derive(Clone, Copy)]
enum DvKind {
    Int8,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Float32,
    Float64,
}

impl DvKind {
    fn size(self) -> usize {
        match self {
            DvKind::Int8 | DvKind::Uint8 => 1,
            DvKind::Int16 | DvKind::Uint16 => 2,
            DvKind::Int32 | DvKind::Uint32 | DvKind::Float32 => 4,
            DvKind::Float64 => 8,
        }
    }
}

/// Resolve `(view, byteIndex)` with bounds checking. The little-endian
/// flag lives at `le_arg`: index 1 for getters, index 2 for setters.
fn dv_args(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
    kind: DvKind,
    le_arg: usize,
) -> Result<(Gc<Object>, u64, u64, bool), NativeError> {
    let view = match this {
        Value::Object(o) if is_data_view_obj(heap, o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "DataView method called on incompatible receiver".into(),
            ));
        }
    };
    let Some((buf, offset, length)) = data_view_parts(heap, view) else {
        return Err(NativeError::TypeError("bad DataView".into()));
    };
    // Spec `GetViewValue`/`SetViewValue` order: full `ToIndex(requestIndex)`
    // first — a `valueOf` hook runs, and its `RangeError` precedes the
    // detached-buffer `TypeError` — then the buffer's detached/out-of-bounds
    // state (which also resolves the effective length for length-tracking
    // views).
    let byte_index = to_index_full(heap, get_arg(args, 0))?;
    let tracking = heap.get(view).view_length_tracking();
    let length = dataview_resolve(heap, buf, offset, length, tracking)?;
    let size = kind.size() as u64;
    // Checked: `byte_index` comes from user input and `length` is bounded,
    // so plain `+` cannot overflow `u64` here, but fail closed anyway.
    let end = byte_index
        .checked_add(size)
        .ok_or_else(|| NativeError::RangeError("DataView: offset out of bounds".into()))?;
    if end > length {
        return Err(NativeError::RangeError(
            "DataView: offset out of bounds".into(),
        ));
    }
    let little_endian = get_arg(args, le_arg).is_truthy();
    let addr = offset
        .checked_add(byte_index)
        .ok_or_else(|| NativeError::RangeError("DataView: offset out of bounds".into()))?;
    Ok((buf, addr, size, little_endian))
}

fn dv_get(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
    kind: DvKind,
) -> Result<Value, NativeError> {
    let (buf, addr, size, le) = dv_args(heap, this, args, kind, 1)?;
    // Bytes are stored little-endian; reassemble per requested endianness.
    let mut raw: u64 = 0;
    for i in 0..size {
        let b = u64::from(buffer_get_byte(heap, buf, addr + i));
        if le {
            raw |= b << (8 * i);
        } else {
            raw = (raw << 8) | b;
        }
    }
    let v = match kind {
        DvKind::Int8 => Value::Int(raw as u8 as i8 as i32),
        DvKind::Uint8 => Value::Int(raw as u8 as i32),
        DvKind::Int16 => Value::Int(raw as u16 as i16 as i32),
        DvKind::Uint16 => js_num(raw as u16 as f64),
        DvKind::Int32 => Value::Int(raw as u32 as i32),
        DvKind::Uint32 => js_num(raw as u32 as f64),
        DvKind::Float32 => Value::Float(f64::from(f32::from_bits(raw as u32))),
        DvKind::Float64 => Value::Float(f64::from_bits(raw)),
    };
    Ok(v)
}

fn dv_set(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
    kind: DvKind,
) -> Result<Value, NativeError> {
    let (buf, addr, size, le) = dv_args(heap, this, args, kind, 2)?;
    // Convert the value through the element kind first (spec: SetValueInBuffer
    // with the type's conversion), then emit bytes in the requested endianness.
    let n = to_number(get_arg(args, 1))?;
    let raw: u64 = match kind {
        DvKind::Int8 => u64::from(to_int32(n) as i8 as u8),
        DvKind::Uint8 => u64::from(to_int32(n) as u8),
        DvKind::Int16 => u64::from(to_int32(n) as i16 as u16),
        DvKind::Uint16 => u64::from(to_int32(n) as u16),
        DvKind::Int32 => u64::from(to_int32(n) as u32),
        DvKind::Uint32 => u64::from(to_int32(n) as u32),
        DvKind::Float32 => u64::from((n as f32).to_bits()),
        DvKind::Float64 => n.to_bits(),
    };
    for i in 0..size {
        let shift = if le { 8 * i } else { 8 * (size - 1 - i) };
        buffer_set_byte(heap, buf, addr + i, ((raw >> shift) & 0xFF) as u8);
    }
    Ok(Value::Undefined)
}

macro_rules! dv_getter {
    ($name:ident, $kind:expr) => {
        fn $name(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
            dv_get(heap, this, args, $kind)
        }
    };
}

macro_rules! dv_setter {
    ($name:ident, $kind:expr) => {
        fn $name(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
            dv_set(heap, this, args, $kind)
        }
    };
}

dv_getter!(dv_get_int8, DvKind::Int8);
dv_getter!(dv_get_uint8, DvKind::Uint8);
dv_getter!(dv_get_int16, DvKind::Int16);
dv_getter!(dv_get_uint16, DvKind::Uint16);
dv_getter!(dv_get_int32, DvKind::Int32);
dv_getter!(dv_get_uint32, DvKind::Uint32);
dv_getter!(dv_get_float32, DvKind::Float32);
dv_getter!(dv_get_float64, DvKind::Float64);
dv_setter!(dv_set_int8, DvKind::Int8);
dv_setter!(dv_set_uint8, DvKind::Uint8);
dv_setter!(dv_set_int16, DvKind::Int16);
dv_setter!(dv_set_uint16, DvKind::Uint16);
dv_setter!(dv_set_int32, DvKind::Int32);
dv_setter!(dv_set_uint32, DvKind::Uint32);
dv_setter!(dv_set_float32, DvKind::Float32);
dv_setter!(dv_set_float64, DvKind::Float64);

// ---------------------------------------------------------------------------
// install
// ---------------------------------------------------------------------------

/// Install `Reflect`, `Proxy`, `ArrayBuffer`, `SharedArrayBuffer`,
/// `DataView`, and all typed array constructors on `global`.
///
/// Called by the VM bootstrap ([`Vm::new`](crate::vm::Vm::new)) when it
/// installs the builtin modules on the fresh global object.
pub fn install(heap: &mut Heap, global: Gc<Object>) -> Result<(), NativeError> {
    install_inner(heap, global)
}

fn install_inner(heap: &mut Heap, global: Gc<Object>) -> Result<(), NativeError> {
    // Reflect.
    let reflect = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    define_native(heap, reflect, "apply", reflect_apply, 3)?;
    define_native(heap, reflect, "construct", reflect_construct, 2)?;
    define_native(heap, reflect, "defineProperty", reflect_define_property, 3)?;
    define_native(heap, reflect, "deleteProperty", reflect_delete_property, 2)?;
    define_native(heap, reflect, "get", reflect_get, 2)?;
    define_native(
        heap,
        reflect,
        "getOwnPropertyDescriptor",
        reflect_get_own_property_descriptor,
        2,
    )?;
    define_native(heap, reflect, "getPrototypeOf", reflect_get_prototype_of, 1)?;
    define_native(heap, reflect, "has", reflect_has, 2)?;
    define_native(heap, reflect, "isExtensible", reflect_is_extensible, 1)?;
    define_native(heap, reflect, "ownKeys", reflect_own_keys, 1)?;
    define_native(
        heap,
        reflect,
        "preventExtensions",
        reflect_prevent_extensions,
        1,
    )?;
    define_native(heap, reflect, "set", reflect_set, 3)?;
    define_native(heap, reflect, "setPrototypeOf", reflect_set_prototype_of, 2)?;
    heap.define_property_str(global, "Reflect", Property::data(Value::Object(reflect)))?;

    // Proxy.
    let proxy_ctor = new_native_constructor(heap, "Proxy", proxy_constructor, 2)?;
    define_native(heap, proxy_ctor, "revocable", proxy_revocable, 2)?;
    heap.define_property_str(global, "Proxy", Property::data(Value::Object(proxy_ctor)))?;

    // ArrayBuffer, with a prototype carrying `slice`.
    let ab_proto = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    // `ab_proto` is not reachable from any traced root until
    // `ArrayBuffer.prototype` is linked below; hold it across the
    // constructor allocation, which may trigger a collection.
    with_temp_root(heap, Value::Object(ab_proto), |heap| {
        define_native(heap, ab_proto, "slice", arraybuffer_slice, 2)?;
        define_native(heap, ab_proto, "transfer", arraybuffer_transfer, 1)?;
        define_native(
            heap,
            ab_proto,
            "transferToFixedLength",
            arraybuffer_transfer_to_fixed_length,
            1,
        )?;
        define_native(heap, ab_proto, "resize", arraybuffer_resize, 1)?;
        define_getter(
            heap,
            ab_proto,
            "maxByteLength",
            arraybuffer_max_byte_length_get,
        )?;
        define_getter(heap, ab_proto, "resizable", arraybuffer_resizable_get)?;
        define_getter(heap, ab_proto, "detached", arraybuffer_detached_get)?;
        let ab_ctor = new_native_constructor(heap, "ArrayBuffer", arraybuffer_constructor, 1)?;
        heap.arraybuffer_ctor = Some(ab_ctor);
        define_native(heap, ab_ctor, "isView", arraybuffer_isview, 1)?;
        heap.define_property_str(
            ab_ctor,
            "prototype",
            Property {
                value: Value::Object(ab_proto),
                writable: false,
                enumerable: false,
                configurable: false,
            },
        )?;
        heap.define_property_str(
            global,
            "ArrayBuffer",
            Property::data(Value::Object(ab_ctor)),
        )?;
        // Backing stores allocated by `new_array_buffer` link to
        // `%ArrayBuffer.prototype%` via this intrinsic (traced as a root).
        heap.arraybuffer_prototype = Some(ab_proto);
        // `%ArrayBuffer.prototype%[Symbol.toStringTag]` = `"ArrayBuffer"`
        // (ECMA-262 25.1.4.3): the real well-known symbol key.
        let tag_sym = heap
            .well_known_symbol(WellKnown::ToStringTag)
            .map_err(|_| NativeError::OutOfMemory)?;
        let tag_str = heap
            .alloc_str("ArrayBuffer")
            .map_err(|_| NativeError::OutOfMemory)?;
        heap.define_property_key(
            ab_proto,
            PropKey::Symbol(tag_sym),
            Property {
                value: Value::Str(tag_str),
                writable: false,
                enumerable: false,
                configurable: true,
            },
        )
        .map_err(|_| NativeError::OutOfMemory)?;
        Ok(())
    })?;

    // SharedArrayBuffer v1: aliases the ArrayBuffer implementation.
    let sab_ctor =
        new_native_constructor(heap, "SharedArrayBuffer", sharedarraybuffer_constructor, 1)?;
    heap.define_property_str(
        global,
        "SharedArrayBuffer",
        Property::data(Value::Object(sab_ctor)),
    )?;

    // DataView, with a prototype carrying the 16 accessors.
    let dv_proto = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    // `DataView.prototype`'s [[Prototype]] is `%Object.prototype%`
    // (ECMA-262 24.3.3). `builtin_object_array::install` runs before this
    // install in the full VM (see vm.rs), so the intrinsic is available
    // there; native-level tests install this module standalone, in which
    // case no prototype links (matching the previous behavior).
    heap.set_prototype(
        dv_proto,
        heap.object_prototype
            .map(Value::Object)
            .unwrap_or(Value::Null),
    );
    // `dv_proto` is not reachable from any traced root until
    // `DataView.prototype` is linked below; hold it across the constructor
    // allocation, which may trigger a collection.
    with_temp_root(heap, Value::Object(dv_proto), |heap| {
        define_native(heap, dv_proto, "getInt8", dv_get_int8, 1)?;
        define_native(heap, dv_proto, "getUint8", dv_get_uint8, 1)?;
        define_native(heap, dv_proto, "getInt16", dv_get_int16, 2)?;
        define_native(heap, dv_proto, "getUint16", dv_get_uint16, 2)?;
        define_native(heap, dv_proto, "getInt32", dv_get_int32, 2)?;
        define_native(heap, dv_proto, "getUint32", dv_get_uint32, 2)?;
        define_native(heap, dv_proto, "getFloat32", dv_get_float32, 2)?;
        define_native(heap, dv_proto, "getFloat64", dv_get_float64, 2)?;
        define_native(heap, dv_proto, "setInt8", dv_set_int8, 2)?;
        define_native(heap, dv_proto, "setUint8", dv_set_uint8, 2)?;
        define_native(heap, dv_proto, "setInt16", dv_set_int16, 3)?;
        define_native(heap, dv_proto, "setUint16", dv_set_uint16, 3)?;
        define_native(heap, dv_proto, "setInt32", dv_set_int32, 3)?;
        define_native(heap, dv_proto, "setUint32", dv_set_uint32, 3)?;
        define_native(heap, dv_proto, "setFloat32", dv_set_float32, 3)?;
        define_native(heap, dv_proto, "setFloat64", dv_set_float64, 3)?;
        // `%DataView.prototype`'s `buffer` / `byteLength` / `byteOffset`
        // accessors (ECMA-262 24.3.4.1–3). Instances keep an own `buffer`
        // data property (read by `crate::builtin_typedarray`'s
        // `view_buffer`); `byteLength` / `byteOffset` resolve through
        // these, so detach and resize are observed on every read.
        define_getter(heap, dv_proto, "buffer", dataview_buffer_get)?;
        define_getter(heap, dv_proto, "byteLength", dataview_byte_length_get)?;
        define_getter(heap, dv_proto, "byteOffset", dataview_byte_offset_get)?;
        let dv_ctor = new_native_constructor(heap, "DataView", dataview_constructor, 1)?;
        heap.define_property_str(
            dv_ctor,
            "prototype",
            Property {
                value: Value::Object(dv_proto),
                writable: false,
                enumerable: false,
                configurable: false,
            },
        )?;
        // `DataView.prototype.constructor` (ECMA-262 24.3.4): non-writable,
        // non-enumerable, configurable.
        heap.define_property_str(
            dv_proto,
            "constructor",
            Property {
                value: Value::Object(dv_ctor),
                writable: false,
                enumerable: false,
                configurable: true,
            },
        )?;
        heap.define_property_str(global, "DataView", Property::data(Value::Object(dv_ctor)))?;
        Ok(())
    })?;

    // Typed arrays: the `%TypedArray%` intrinsic plus one concrete
    // constructor per element type.
    //
    // - `%TypedArray%.prototype` (the shared `ta_proto` below) carries
    //   the full kind-generic method set plus the `length` /
    //   `byteLength` / `byteOffset` / `buffer` accessors; its
    //   [[Prototype]] is `%Object.prototype%`.
    // - `%TypedArray%` itself (`ta_ctor`) is abstract: calling or
    //   constructing it throws. It is deliberately *not* installed on
    //   the global (spec: no global `TypedArray` binding).
    // - Each concrete constructor's [[Prototype]] is `%TypedArray%`, so
    //   `Object.getPrototypeOf(Int8Array)` — the test262 harness's
    //   `testTypedArray.js` idiom — yields the intrinsic.
    // - Each `%<T>Array.prototype%`'s [[Prototype]] is the shared
    //   `%TypedArray%.prototype`.
    //
    // Views built by the JS constructors resolve their prototype via
    // GetPrototypeFromConstructor (see `ta_constructor`); native builders
    // (e.g. `TextEncoder.encode`) proto-link to `%Uint8Array.prototype%`
    // via `uint8array_prototype`, so `instanceof Uint8Array` and the
    // prototype methods work on both paths.
    //
    // NOTE (symbol phase 2): `@@iterator` / `@@species` / `@@toStringTag`
    // below use the real well-known symbol keys
    // (`heap.well_known_symbol`), never `"@@…"` string keys.
    const CTORS: [NativeFn; 11] = [
        int8array_constructor,
        uint8array_constructor,
        uint8clampedarray_constructor,
        int16array_constructor,
        uint16array_constructor,
        int32array_constructor,
        uint32array_constructor,
        float32array_constructor,
        float64array_constructor,
        bigint64array_constructor,
        biguint64array_constructor,
    ];
    // (`name`, `func`, `length`) — ECMA-262 22.2.3 method `length`s.
    const TA_METHODS: [(&str, NativeFn, u32); 31] = [
        ("at", typed_array_at, 1),
        ("copyWithin", typed_array_copy_within, 2),
        ("entries", typed_array_entries, 0),
        ("every", typed_array_every, 1),
        ("fill", typed_array_fill, 1),
        ("filter", typed_array_filter, 1),
        ("find", typed_array_find, 1),
        ("findIndex", typed_array_find_index, 1),
        ("findLast", typed_array_find_last, 1),
        ("findLastIndex", typed_array_find_last_index, 1),
        ("forEach", typed_array_for_each, 1),
        ("includes", typed_array_includes, 1),
        ("indexOf", typed_array_index_of, 1),
        ("join", typed_array_join, 1),
        ("keys", typed_array_keys, 0),
        ("lastIndexOf", typed_array_last_index_of, 1),
        ("map", typed_array_map, 1),
        ("reduce", typed_array_reduce, 1),
        ("reduceRight", typed_array_reduce_right, 1),
        ("reverse", typed_array_reverse, 0),
        ("set", typed_array_set, 2),
        ("slice", typed_array_slice, 2),
        ("some", typed_array_some, 1),
        ("sort", typed_array_sort, 1),
        ("subarray", typed_array_subarray, 2),
        ("toLocaleString", typed_array_to_locale_string, 0),
        ("toReversed", typed_array_to_reversed, 0),
        ("toSorted", typed_array_to_sorted, 1),
        ("toString", typed_array_to_string, 0),
        ("values", typed_array_values, 0),
        ("with", typed_array_with, 2),
    ];
    // Shared `%TypedArray%.prototype`: a fresh ordinary object,
    // [[Prototype]] → `%Object.prototype%` (native-level tests install
    // this module standalone, mirroring the DataView install above).
    let ta_proto = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    heap.set_prototype(
        ta_proto,
        heap.object_prototype
            .map(Value::Object)
            .unwrap_or(Value::Null),
    );
    // Root `ta_proto` across the abstract-constructor allocation below,
    // which may trigger a collection.
    let ta_ctor = with_temp_root(heap, Value::Object(ta_proto), |heap| {
        new_native_function(heap, "TypedArray", typed_array_abstract_constructor, 0)
    })?;
    // Install the method set, the iterator, and the accessors on the
    // shared prototype; link the abstract constructor both ways.
    with_temp_root(heap, Value::Object(ta_proto), |heap| {
        with_temp_root(heap, Value::Object(ta_ctor), |heap| {
            for (name, func, length) in TA_METHODS {
                define_ta_method(heap, ta_proto, name, func, length)?;
            }
            // `%TypedArray%.prototype.toString` is the same function object
            // as `Array.prototype.toString` (ECMA-262 22.2.3.32), which
            // `builtin_object_array` installs as `Array.prototype.join`.
            // In a standalone native-test install the array prototype may
            // be absent; then the `typed_array_to_string` installed with
            // the method set above stands in.
            if let Some(ap) = heap.array_prototype
                && let Some(p) = heap.get_property(ap, Wtf8::from_str("toString"))
            {
                heap.define_property_str(
                    ta_proto,
                    "toString",
                    Property {
                        value: p.value,
                        writable: true,
                        enumerable: false,
                        configurable: true,
                    },
                )?;
            }
            // `%TypedArray%.prototype[@@iterator]` === `values` (real
            // symbol key): {writable, non-enumerable, configurable}.
            let iterator_sym = heap
                .well_known_symbol(WellKnown::Iterator)
                .map_err(|_| NativeError::OutOfMemory)?;
            let values_fn = heap
                .get_property(ta_proto, Wtf8::from_str("values"))
                .map(|p| p.value)
                .unwrap_or(Value::Undefined);
            heap.define_property_key(
                ta_proto,
                PropKey::Symbol(iterator_sym),
                Property {
                    value: values_fn,
                    writable: true,
                    enumerable: false,
                    configurable: true,
                },
            )
            .map_err(|_| NativeError::OutOfMemory)?;
            // The accessor family: `length`, `byteLength`, `byteOffset`
            // (ECMA-262 22.2.3.16–18) and the pre-existing `buffer`
            // getter, all getter-only, non-enumerable, configurable.
            define_getter(heap, ta_proto, "length", typed_array_length_get)?;
            define_getter(heap, ta_proto, "byteLength", typed_array_byte_length_get)?;
            define_getter(heap, ta_proto, "byteOffset", typed_array_byte_offset_get)?;
            define_getter(heap, ta_proto, "buffer", typed_array_buffer_get)?;
            // `%TypedArray%.prototype.constructor` → `%TypedArray%`
            // (writable, non-enumerable, configurable).
            heap.define_property_str(
                ta_proto,
                "constructor",
                Property {
                    value: Value::Object(ta_ctor),
                    writable: true,
                    enumerable: false,
                    configurable: true,
                },
            )?;
            // `%TypedArray%.prototype[@@toStringTag]` = `"TypedArray"`.
            let tag_sym = heap
                .well_known_symbol(WellKnown::ToStringTag)
                .map_err(|_| NativeError::OutOfMemory)?;
            let tag_str = with_temp_root(heap, Value::Object(ta_proto), |heap| {
                heap.alloc_str("TypedArray")
                    .map_err(|_| NativeError::OutOfMemory)
            })?;
            heap.define_property_key(
                ta_proto,
                PropKey::Symbol(tag_sym),
                Property {
                    value: Value::Str(tag_str),
                    writable: false,
                    enumerable: false,
                    configurable: true,
                },
            )
            .map_err(|_| NativeError::OutOfMemory)?;
            // `%TypedArray%.prototype` link on the constructor
            // (non-writable, per ECMA-262 22.2.2).
            heap.define_property_str(
                ta_ctor,
                "prototype",
                Property {
                    value: Value::Object(ta_proto),
                    writable: false,
                    enumerable: false,
                    configurable: false,
                },
            )?;
            // `%TypedArray%.from` / `%TypedArray%.of` (ECMA-262 22.2.8.1–2):
            // statics on the abstract constructor; concrete constructors
            // inherit them via [[Prototype]] → `%TypedArray%`. Both are
            // generic over `this`, so subclass constructors work through
            // the shared `reflect_construct_inner` path.
            define_ta_method(heap, ta_ctor, "of", typed_array_of, 0)?;
            define_ta_method(heap, ta_ctor, "from", typed_array_from, 1)?;
            // `get %TypedArray% [ @@species ]` → `this` (real symbol key;
            // ECMA-262 22.2.8.3). Concrete constructors inherit it.
            let species_sym = heap
                .well_known_symbol(WellKnown::Species)
                .map_err(|_| NativeError::OutOfMemory)?;
            let species_get = with_temp_root(heap, Value::Object(ta_ctor), |heap| {
                new_native_function(heap, "get [Symbol.species]", typed_array_species_get, 0)
            })?;
            heap.define_accessor_key(
                ta_ctor,
                PropKey::Symbol(species_sym),
                AccessorProp {
                    get: Value::Object(species_get),
                    set: Value::Undefined,
                    enumerable: false,
                    configurable: true,
                },
            )
            .map_err(|_| NativeError::OutOfMemory)?;
            Ok(())
        })
    })?;
    // One concrete constructor per element type. `ta_proto`/`ta_ctor`
    // stay temp-rooted across the loop: they become reachable from the
    // global only as each constructor links up.
    with_temp_root(heap, Value::Object(ta_proto), |heap| {
        with_temp_root(heap, Value::Object(ta_ctor), |heap| {
            for (i, (name, size, _)) in ETYPES.iter().enumerate() {
                // ECMA-262 22.2.5: each `%TypedArray%` constructor's
                // `length` is 3 (`(buffer, byteOffset, length)`).
                let ctor = new_native_constructor(heap, name, CTORS[i], 3)?;
                // `ctor` is not reachable from any traced root until it
                // is linked on the global below; hold it across the
                // allocating installs.
                with_temp_root(heap, Value::Object(ctor), |heap| {
                    define_const(heap, ctor, "BYTES_PER_ELEMENT", js_num(*size as f64))?;
                    // `%<T>Array.prototype%`: fresh ordinary object,
                    // [[Prototype]] → shared `%TypedArray%.prototype`,
                    // carrying `constructor` and `@@toStringTag`.
                    crate::builtin_typedarray::install_typed_array_prototype(
                        heap, ctor, ta_proto, name,
                    )?;
                    // Concrete constructor's [[Prototype]] →
                    // `%TypedArray%` (inherits `of`/`from`/`@@species`).
                    heap.set_prototype(ctor, Value::Object(ta_ctor));
                    heap.define_property_str(global, name, Property::data(Value::Object(ctor)))?;
                    // Remember the intrinsic constructor for the spec
                    // `TypedArraySpeciesCreate` default-constructor fallback.
                    heap.typed_array_ctors[i] = Some(ctor);
                    Ok(())
                })?;
            }
            Ok(())
        })
    })?;
    // `Array.prototype[@@iterator]` (ECMA-262 22.1.4.1): installed here,
    // after `builtin_collections::install` has created the canonical
    // well-known symbols. (Installing it in `builtin_object_array`
    // would allocate a pre-canonical symbol that gets overwritten.)
    // Not `values` (a v1 array-returning deviation): a real iterator.
    if let Some(array_proto) = heap.array_prototype {
        let iterator_sym = heap
            .well_known_symbol(WellKnown::Iterator)
            .map_err(|_| NativeError::OutOfMemory)?;
        let iter_fn = with_temp_root(heap, Value::Object(array_proto), |heap| {
            new_native_function(
                heap,
                "[Symbol.iterator]",
                crate::builtin_object_array::array_iterator,
                0,
            )
        })?;
        heap.define_property_key(
            array_proto,
            PropKey::Symbol(iterator_sym),
            Property {
                value: Value::Object(iter_fn),
                writable: true,
                enumerable: false,
                configurable: true,
            },
        )
        .map_err(|_| NativeError::OutOfMemory)?;
    }
    Ok(())
}

/// Define `target[name]` as a builtin method with spec attributes
/// (`writable`, non-`enumerable`, `configurable`): the installer for the
/// `%TypedArray%` method family. (The module-level `define_native` uses
/// `Property::data` — enumerable — which matches the older installs but
/// not ECMA-262's method descriptors; the typed-array install does not
/// reuse it.)
fn define_ta_method(
    heap: &mut Heap,
    target: Gc<Object>,
    name: &str,
    func: NativeFn,
    length: u32,
) -> Result<(), NativeError> {
    // Hold `target` across the function-object allocation below, which
    // may trigger a collection while the target is not yet otherwise
    // rooted.
    let f = with_temp_root(heap, Value::Object(target), |heap| {
        new_native_function(heap, name, func, length)
    })?;
    heap.define_property_str(
        target,
        name,
        Property {
            value: Value::Object(f),
            writable: true,
            enumerable: false,
            configurable: true,
        },
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gc::Heap;

    fn setup() -> (Heap, Value) {
        let mut heap = Heap::new();
        let global = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        install(&mut heap, global).unwrap();
        (heap, Value::Object(global))
    }

    fn call(heap: &mut Heap, f: Value, this: Value, args: &[Value]) -> Result<Value, NativeError> {
        match f {
            Value::Object(o) => match heap.get(o).kind {
                ObjectKind::NativeFunction(nf) => nf(heap, this, args),
                _ => panic!("not a native function"),
            },
            _ => panic!("not a function"),
        }
    }

    /// Invoke a `%TypedArray%.prototype` accessor (`length` / `byteLength` /
    /// `byteOffset` / `buffer`) on a view: walk the prototype chain for the
    /// accessor and call its getter with the view as receiver.
    fn ta_acc(heap: &mut Heap, view: Gc<Object>, name: &str) -> Value {
        let mut current = Value::Object(view);
        loop {
            let o = match current {
                Value::Object(o) => o,
                _ => panic!("ta_acc: no accessor '{name}' on the chain"),
            };
            if let Some(acc) = heap.get_accessor(o, Wtf8::from_str(name)) {
                let Value::Object(f) = acc.get else {
                    panic!("ta_acc: '{name}' has no getter");
                };
                return call(heap, Value::Object(f), Value::Object(view), &[]).unwrap();
            }
            current = heap.get(o).prototype;
        }
    }

    /// Walk a property path from a value.
    fn mget(heap: &Heap, mut v: Value, path: &[&str]) -> Value {
        for p in path {
            let o = match v {
                Value::Object(o) => o,
                _ => panic!("mget: not an object at '{p}'"),
            };
            v = get_prop_chain(heap, o, Wtf8::from_str(p))
                .unwrap_or_else(|| panic!("mget: missing '{p}'"));
        }
        v
    }

    /// Call `global.path...`, sequencing borrows (no nested `&mut heap`).
    fn call_path(
        heap: &mut Heap,
        global: Value,
        path: &[&str],
        this: Value,
        args: &[Value],
    ) -> Result<Value, NativeError> {
        let f = mget(heap, global, path);
        call(heap, f, this, args)
    }

    fn sval(heap: &mut Heap, s: &str) -> Value {
        Value::Str(heap.alloc_str(s).unwrap())
    }

    fn new_plain(heap: &mut Heap) -> Gc<Object> {
        heap.alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap()
    }

    /// Handler object with a single trap installed.
    fn handler_with(heap: &mut Heap, name: &str, trap: NativeFn, length: u32) -> Gc<Object> {
        let h = new_plain(heap);
        let f = new_native_function(heap, name, trap, length).unwrap();
        heap.define_property_str(h, name, Property::data(Value::Object(f)))
            .unwrap();
        h
    }

    fn trap_true(_heap: &mut Heap, _this: Value, _args: &[Value]) -> Result<Value, NativeError> {
        Ok(Value::Bool(true))
    }

    fn trap_false(_heap: &mut Heap, _this: Value, _args: &[Value]) -> Result<Value, NativeError> {
        Ok(Value::Bool(false))
    }

    fn trap_undef(_heap: &mut Heap, _this: Value, _args: &[Value]) -> Result<Value, NativeError> {
        Ok(Value::Undefined)
    }

    fn trap_get_wrong(
        _heap: &mut Heap,
        _this: Value,
        _args: &[Value],
    ) -> Result<Value, NativeError> {
        Ok(Value::Int(999))
    }

    fn trap_new_object(
        heap: &mut Heap,
        _this: Value,
        _args: &[Value],
    ) -> Result<Value, NativeError> {
        Ok(Value::Object(new_plain(heap)))
    }

    fn trap_ownkeys_empty(
        heap: &mut Heap,
        _this: Value,
        _args: &[Value],
    ) -> Result<Value, NativeError> {
        Ok(Value::Object(new_key_array(heap, &[])?))
    }

    fn native_add(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
        let _ = heap;
        Ok(js_num(
            to_number(get_arg(args, 0))? + to_number(get_arg(args, 1))?,
        ))
    }

    /// Build `new Proxy(target, handler)` via the installed constructor.
    fn new_proxy(heap: &mut Heap, global: Value, target: Gc<Object>, handler: Gc<Object>) -> Value {
        let ctor = mget(heap, global, &["Proxy"]);
        call(
            heap,
            ctor,
            Value::Undefined,
            &[Value::Object(target), Value::Object(handler)],
        )
        .unwrap()
    }

    fn fixed_target(heap: &mut Heap) -> Gc<Object> {
        let t = new_plain(heap);
        heap.define_property_str(
            t,
            "fixed",
            Property {
                value: Value::Int(1),
                writable: false,
                enumerable: true,
                configurable: false,
            },
        )
        .unwrap();
        t
    }

    #[test]
    fn reflect_ordinary_object_ops() {
        let (mut heap, global) = setup();
        let target = Value::Object(new_plain(&mut heap));
        let kx = sval(&mut heap, "x");
        let ky = sval(&mut heap, "y");
        let kmissing = sval(&mut heap, "missing");

        let desc = new_plain(&mut heap);
        heap.define_property_str(desc, "value", Property::data(Value::Int(42)))
            .unwrap();
        heap.define_property_str(desc, "writable", Property::data(Value::Bool(true)))
            .unwrap();
        heap.define_property_str(desc, "enumerable", Property::data(Value::Bool(true)))
            .unwrap();
        heap.define_property_str(desc, "configurable", Property::data(Value::Bool(true)))
            .unwrap();
        let d = Value::Object(desc);
        let undef = Value::Undefined;

        let r = call_path(
            &mut heap,
            global,
            &["Reflect", "defineProperty"],
            undef,
            &[target, kx, d],
        )
        .unwrap();
        assert_eq!(r, Value::Bool(true));

        let got = call_path(&mut heap, global, &["Reflect", "get"], undef, &[target, kx]).unwrap();
        assert_eq!(got, Value::Int(42));

        let ok = call_path(
            &mut heap,
            global,
            &["Reflect", "set"],
            undef,
            &[target, kx, Value::Int(7)],
        )
        .unwrap();
        assert_eq!(ok, Value::Bool(true));
        let got = call_path(&mut heap, global, &["Reflect", "get"], undef, &[target, kx]).unwrap();
        assert_eq!(got, Value::Int(7));

        let h = call_path(&mut heap, global, &["Reflect", "has"], undef, &[target, kx]).unwrap();
        assert_eq!(h, Value::Bool(true));
        let h = call_path(&mut heap, global, &["Reflect", "has"], undef, &[target, ky]).unwrap();
        assert_eq!(h, Value::Bool(false));

        let dd = call_path(
            &mut heap,
            global,
            &["Reflect", "getOwnPropertyDescriptor"],
            undef,
            &[target, kx],
        )
        .unwrap();
        let dd = require_object(dd, "t").unwrap();
        assert_eq!(
            get_prop_chain(&heap, dd, Wtf8::from_str("value")),
            Some(Value::Int(7))
        );
        let dd = call_path(
            &mut heap,
            global,
            &["Reflect", "getOwnPropertyDescriptor"],
            undef,
            &[target, kmissing],
        )
        .unwrap();
        assert_eq!(dd, Value::Undefined);

        let keys = call_path(&mut heap, global, &["Reflect", "ownKeys"], undef, &[target]).unwrap();
        let keys = require_object(keys, "t").unwrap();
        assert!(strict_equal(array_like_get(&heap, keys, 0), kx));
        assert_eq!(
            get_prop_chain(&heap, keys, Wtf8::from_str("length")),
            Some(Value::Int(1))
        );

        let del = call_path(
            &mut heap,
            global,
            &["Reflect", "deleteProperty"],
            undef,
            &[target, kx],
        )
        .unwrap();
        assert_eq!(del, Value::Bool(true));
        let h = call_path(&mut heap, global, &["Reflect", "has"], undef, &[target, kx]).unwrap();
        assert_eq!(h, Value::Bool(false));

        let e = call_path(
            &mut heap,
            global,
            &["Reflect", "isExtensible"],
            undef,
            &[target],
        )
        .unwrap();
        assert_eq!(e, Value::Bool(true));
        let e = call_path(
            &mut heap,
            global,
            &["Reflect", "preventExtensions"],
            undef,
            &[target],
        )
        .unwrap();
        assert_eq!(e, Value::Bool(true));
        let e = call_path(
            &mut heap,
            global,
            &["Reflect", "isExtensible"],
            undef,
            &[target],
        )
        .unwrap();
        assert_eq!(e, Value::Bool(false));

        let sp = call_path(
            &mut heap,
            global,
            &["Reflect", "setPrototypeOf"],
            undef,
            &[target, Value::Null],
        )
        .unwrap();
        assert_eq!(sp, Value::Bool(true));
        let gp = call_path(
            &mut heap,
            global,
            &["Reflect", "getPrototypeOf"],
            undef,
            &[target],
        )
        .unwrap();
        assert_eq!(gp, Value::Null);

        // Non-object targets throw.
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "get"],
            undef,
            &[Value::Int(1), kx],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));
    }

    #[test]
    fn proxy_forwards_without_traps() {
        let (mut heap, global) = setup();
        let target = new_plain(&mut heap);
        heap.define_property_str(target, "a", Property::data(Value::Int(11)))
            .unwrap();
        let handler = new_plain(&mut heap);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let ka = sval(&mut heap, "a");
        let kb = sval(&mut heap, "b");
        let undef = Value::Undefined;

        let got = call_path(&mut heap, global, &["Reflect", "get"], undef, &[proxy, ka]).unwrap();
        assert_eq!(got, Value::Int(11));
        let ok = call_path(
            &mut heap,
            global,
            &["Reflect", "set"],
            undef,
            &[proxy, kb, Value::Int(2)],
        )
        .unwrap();
        assert_eq!(ok, Value::Bool(true));
        let got = call_path(&mut heap, global, &["Reflect", "get"], undef, &[proxy, kb]).unwrap();
        assert_eq!(got, Value::Int(2));
    }

    #[test]
    fn proxy_get_set_invariants() {
        let (mut heap, global) = setup();
        let target = fixed_target(&mut heap);
        let kfixed = sval(&mut heap, "fixed");
        let kother = sval(&mut heap, "other");
        let undef = Value::Undefined;

        // get trap returning the wrong value for a non-configurable,
        // non-writable property throws.
        let handler = handler_with(&mut heap, "get", trap_get_wrong, 3);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "get"],
            undef,
            &[proxy, kfixed],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));

        // set trap succeeding on a non-configurable, non-writable property
        // with a different value throws.
        let handler = handler_with(&mut heap, "set", trap_true, 4);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "set"],
            undef,
            &[proxy, kfixed, Value::Int(2)],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));

        // set trap returning false reports false.
        let handler = handler_with(&mut heap, "set", trap_false, 4);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let r = call_path(
            &mut heap,
            global,
            &["Reflect", "set"],
            undef,
            &[proxy, kother, Value::Int(1)],
        )
        .unwrap();
        assert_eq!(r, Value::Bool(false));
    }

    #[test]
    fn proxy_has_delete_ownkeys_gopd_invariants() {
        let (mut heap, global) = setup();
        let target = fixed_target(&mut heap);
        let kfixed = sval(&mut heap, "fixed");
        let undef = Value::Undefined;

        // has trap denying a non-configurable own property throws.
        let handler = handler_with(&mut heap, "has", trap_false, 2);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "has"],
            undef,
            &[proxy, kfixed],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));

        // deleteProperty trap succeeding on a non-configurable property throws.
        let handler = handler_with(&mut heap, "deleteProperty", trap_true, 2);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "deleteProperty"],
            undef,
            &[proxy, kfixed],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));

        // ownKeys trap omitting a non-configurable key throws.
        let handler = handler_with(&mut heap, "ownKeys", trap_ownkeys_empty, 1);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let err =
            call_path(&mut heap, global, &["Reflect", "ownKeys"], undef, &[proxy]).unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));

        // getOwnPropertyDescriptor trap returning undefined for a
        // non-configurable property throws.
        let handler = handler_with(&mut heap, "getOwnPropertyDescriptor", trap_undef, 2);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "getOwnPropertyDescriptor"],
            undef,
            &[proxy, kfixed],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));

        // defineProperty trap claiming success while changing a
        // non-configurable, non-writable property throws.
        let handler = handler_with(&mut heap, "defineProperty", trap_true, 3);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let desc = new_plain(&mut heap);
        heap.define_property_str(desc, "value", Property::data(Value::Int(2)))
            .unwrap();
        let d = Value::Object(desc);
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "defineProperty"],
            undef,
            &[proxy, kfixed, d],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));

        // Without a trap, defineProperty on the non-configurable property
        // reports false instead of throwing.
        let handler = new_plain(&mut heap);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let r = call_path(
            &mut heap,
            global,
            &["Reflect", "defineProperty"],
            undef,
            &[proxy, kfixed, d],
        )
        .unwrap();
        assert_eq!(r, Value::Bool(false));
    }

    /// Target with a single accessor property `x` (native getter).
    fn accessor_target(heap: &mut Heap, configurable: bool) -> Gc<Object> {
        let t = new_plain(heap);
        let get = new_native_function(heap, "getx", trap_undef, 0).unwrap();
        heap.define_accessor(
            t,
            Wtf8::from_str("x"),
            crate::value::AccessorProp {
                get: Value::Object(get),
                set: Value::Undefined,
                enumerable: true,
                configurable,
            },
        )
        .unwrap();
        t
    }

    fn trap_ownkeys_x(
        heap: &mut Heap,
        _this: Value,
        _args: &[Value],
    ) -> Result<Value, NativeError> {
        Ok(Value::Object(new_key_array(
            heap,
            &[OwnKey::Str(Wtf8Buf::from("x"))],
        )?))
    }

    #[test]
    fn reflect_own_keys_lists_accessor_keys() {
        // Accessor keys are own keys: `Reflect.ownKeys` must list them.
        let (mut heap, global) = setup();
        let target = Value::Object(accessor_target(&mut heap, true));
        let keys = call_path(
            &mut heap,
            global,
            &["Reflect", "ownKeys"],
            Value::Undefined,
            &[target],
        )
        .unwrap();
        let keys = require_object(keys, "t").unwrap();
        assert_eq!(
            get_prop_chain(&heap, keys, Wtf8::from_str("length")),
            Some(Value::Int(1))
        );
        assert!(strict_equal(
            array_like_get(&heap, keys, 0),
            sval(&mut heap, "x")
        ));
    }

    #[test]
    fn reflect_own_keys_merges_accessors_in_spec_order() {
        // Integer indices first (ascending), then other string keys:
        // data properties in insertion order, then accessors.
        let (mut heap, global) = setup();
        let t = new_plain(&mut heap);
        heap.define_property_str(t, "b", Property::data(Value::Int(1)))
            .unwrap();
        let get = new_native_function(&mut heap, "geta", trap_undef, 0).unwrap();
        heap.define_accessor(
            t,
            Wtf8::from_str("a"),
            crate::value::AccessorProp {
                get: Value::Object(get),
                set: Value::Undefined,
                enumerable: true,
                configurable: true,
            },
        )
        .unwrap();
        heap.define_property_str(t, "2", Property::data(Value::Int(2)))
            .unwrap();
        heap.define_property_str(t, "10", Property::data(Value::Int(10)))
            .unwrap();
        let keys = call_path(
            &mut heap,
            global,
            &["Reflect", "ownKeys"],
            Value::Undefined,
            &[Value::Object(t)],
        )
        .unwrap();
        let keys = require_object(keys, "t").unwrap();
        let got: Vec<String> = (0..4)
            .map(|i| match array_like_get(&heap, keys, i) {
                Value::Str(s) => s.as_wtf8().to_utf8_lossy().into_owned(),
                v => panic!("expected string key, got {v:?}"),
            })
            .collect();
        assert_eq!(got, vec!["2", "10", "b", "a"]);
    }

    #[test]
    fn reflect_own_keys_spec_order_with_symbols() {
        // Integer indices, then string keys, then symbol keys — symbols
        // never interleave with strings.
        assert_eq!(
            eval_str(
                "var s1 = Symbol('one'), s2 = Symbol('two'); \
                 var o = {b: 1}; \
                 o['10'] = 'x'; o['2'] = 'y'; \
                 o[s1] = 1; o[s2] = 2; \
                 var keys = Reflect.ownKeys(o); \
                 keys.length + '|' + keys[0] + '|' + keys[1] + '|' + keys[2] \
                   + '|' + (keys[3] === s1) + '|' + (keys[4] === s2)"
            ),
            "5|2|10|b|true|true"
        );
    }

    #[test]
    fn proxy_traps_support_symbol_keys() {
        // Every per-key trap fires with the symbol (not a string), and
        // the no-trap path forwards to the target's symbol side tables.
        assert_eq!(
            eval_str(
                "var s = Symbol('k'); \
                 var target = {}; target[s] = 'v0'; \
                 var seen = []; \
                 var p = new Proxy(target, { \
                   get: function(t, k, r) { seen.push('get'); return t[k]; }, \
                   set: function(t, k, v, r) { seen.push('set'); t[k] = v; return true; }, \
                   has: function(t, k) { seen.push('has'); return k in t; }, \
                   deleteProperty: function(t, k) { seen.push('delete'); delete t[k]; return true; }, \
                   getOwnPropertyDescriptor: function(t, k) { \
                     seen.push('gopd'); return Object.getOwnPropertyDescriptor(t, k); }, \
                   defineProperty: function(t, k, d) { \
                     seen.push('define'); Object.defineProperty(t, k, d); return true; }, \
                   ownKeys: function(t) { seen.push('ownKeys'); return Reflect.ownKeys(t); } \
                 }); \
                 var r = []; \
                 r.push(Reflect.get(p, s)); \
                 Reflect.set(p, s, 'v1'); \
                 r.push(target[s]); \
                 r.push(Reflect.has(p, s)); \
                 r.push(Reflect.getOwnPropertyDescriptor(p, s).value); \
                 Reflect.defineProperty(p, s, {value: 'v2', enumerable: true, configurable: true}); \
                 r.push(target[s]); \
                 r.push(Reflect.ownKeys(p).length); \
                 r.push(Reflect.ownKeys(p)[0] === s); \
                 Reflect.deleteProperty(p, s); \
                 r.push(s in target); \
                 r.push(seen.join(',')); \
                 r.join('|')"
            ),
            "v0|v1|true|v1|v2|1|true|false|get,set,has,gopd,define,ownKeys,ownKeys,delete"
        );
    }

    #[test]
    fn proxy_no_trap_forwards_symbol_keys_to_target() {
        assert_eq!(
            eval_str(
                "var s = Symbol('k'); \
                 var target = {}; target[s] = 11; \
                 var p = new Proxy(target, {}); \
                 var r = []; \
                 r.push(Reflect.get(p, s)); \
                 r.push(Reflect.has(p, s)); \
                 r.push(Reflect.getOwnPropertyDescriptor(p, s).value); \
                 r.push(Reflect.ownKeys(p)[0] === s); \
                 Reflect.set(p, s, 12); \
                 r.push(target[s]); \
                 Reflect.deleteProperty(p, s); \
                 r.push(Reflect.has(p, s)); \
                 r.join(',')"
            ),
            "11,true,11,true,12,false"
        );
    }

    #[test]
    fn proxy_get_invariant_applies_to_symbol_keys() {
        // A get trap lying about a non-configurable, non-writable symbol
        // property throws, exactly like the string-key rule.
        assert_eq!(
            eval_str(
                "var s = Symbol('k'); \
                 var target = {}; \
                 Object.defineProperty(target, s, \
                   {value: 1, writable: false, configurable: false}); \
                 var p = new Proxy(target, { \
                   get: function(t, k, r) { return 999; } \
                 }); \
                 var r = 'no-throw'; \
                 try { Reflect.get(p, s); } catch (e) { r = 'threw'; } \
                 r"
            ),
            "threw"
        );
    }

    #[test]
    fn proxy_ownkeys_trap_returning_symbols_passes_invariants() {
        // An ownKeys trap may return symbols; duplicates and missing
        // non-configurable keys still throw.
        assert_eq!(
            eval_str(
                "var s = Symbol('k'); \
                 var target = {}; target[s] = 1; target.x = 2; \
                 var p = new Proxy(target, { \
                   ownKeys: function(t) { return [s, 'x']; } \
                 }); \
                 var keys = Reflect.ownKeys(p); \
                 keys.length + ',' + (keys[0] === s) + ',' + (keys[1] === 'x')"
            ),
            "2,true,true"
        );
        assert_eq!(
            eval_str(
                "var s = Symbol('k'); \
                 var target = {}; target[s] = 1; \
                 var p = new Proxy(target, { \
                   ownKeys: function(t) { return [s, s]; } \
                 }); \
                 var r = 'no-throw'; \
                 try { Reflect.ownKeys(p); } catch (e) { r = 'threw'; } \
                 r"
            ),
            "threw"
        );
    }

    #[test]
    fn proxy_ownkeys_trap_returning_accessor_key_passes() {
        // A trap that faithfully reports the target's accessor key must
        // not throw, even on a non-extensible target (the key is part of
        // the target's own keys, so the exact-match invariant holds).
        let (mut heap, global) = setup();
        let target = accessor_target(&mut heap, true);
        heap.prevent_extensions(target);
        let handler = handler_with(&mut heap, "ownKeys", trap_ownkeys_x, 1);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let keys = call_path(
            &mut heap,
            global,
            &["Reflect", "ownKeys"],
            Value::Undefined,
            &[proxy],
        )
        .unwrap();
        let keys = require_object(keys, "t").unwrap();
        assert!(strict_equal(
            array_like_get(&heap, keys, 0),
            sval(&mut heap, "x")
        ));
    }

    #[test]
    fn proxy_ownkeys_trap_omitting_nonconfigurable_accessor_throws() {
        // Omitting a non-configurable accessor key violates the
        // `[[OwnPropertyKeys]]` invariant, exactly like a data key.
        let (mut heap, global) = setup();
        let target = accessor_target(&mut heap, false);
        let handler = handler_with(&mut heap, "ownKeys", trap_ownkeys_empty, 1);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "ownKeys"],
            Value::Undefined,
            &[proxy],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));
    }

    fn trap_gopd_empty(
        heap: &mut Heap,
        _this: Value,
        _args: &[Value],
    ) -> Result<Value, NativeError> {
        Ok(Value::Object(new_plain(heap)))
    }

    fn trap_gopd_value_only(
        heap: &mut Heap,
        _this: Value,
        _args: &[Value],
    ) -> Result<Value, NativeError> {
        let d = new_plain(heap);
        heap.define_property_str(d, "value", Property::data(Value::Int(1)))
            .unwrap();
        Ok(Value::Object(d))
    }

    fn trap_gopd_nonconf_writable(
        heap: &mut Heap,
        _this: Value,
        _args: &[Value],
    ) -> Result<Value, NativeError> {
        let d = new_plain(heap);
        heap.define_property_str(d, "configurable", Property::data(Value::Bool(false)))
            .unwrap();
        heap.define_property_str(d, "writable", Property::data(Value::Bool(true)))
            .unwrap();
        heap.define_property_str(d, "value", Property::data(Value::Int(1)))
            .unwrap();
        Ok(Value::Object(d))
    }

    fn trap_gopd_nonconf_only(
        heap: &mut Heap,
        _this: Value,
        _args: &[Value],
    ) -> Result<Value, NativeError> {
        let d = new_plain(heap);
        heap.define_property_str(d, "configurable", Property::data(Value::Bool(false)))
            .unwrap();
        Ok(Value::Object(d))
    }

    #[test]
    fn proxy_gopd_absent_fields_carry_no_invariant() {
        // Spec 10.5.1 steps 11/15: absent descriptor fields carry no
        // invariant, so `{}` and `{value:1}` on a non-configurable,
        // non-writable target must NOT throw (previously they did).
        let (mut heap, global) = setup();
        let target = fixed_target(&mut heap);
        let kfixed = sval(&mut heap, "fixed");
        let undef = Value::Undefined;
        for trap in [trap_gopd_empty, trap_gopd_value_only] {
            let handler = handler_with(&mut heap, "getOwnPropertyDescriptor", trap, 2);
            let proxy = new_proxy(&mut heap, global, target, handler);
            let r = call_path(
                &mut heap,
                global,
                &["Reflect", "getOwnPropertyDescriptor"],
                undef,
                &[proxy, kfixed],
            )
            .unwrap();
            assert!(matches!(r, Value::Object(_)));
        }
    }

    #[test]
    fn proxy_gopd_writable_true_on_nonwritable_throws() {
        // Spec 10.5.1 step 15.a: a non-configurable result that reports
        // writable on a non-writable target throws (previously missed).
        let (mut heap, global) = setup();
        let target = fixed_target(&mut heap);
        let kfixed = sval(&mut heap, "fixed");
        let handler = handler_with(
            &mut heap,
            "getOwnPropertyDescriptor",
            trap_gopd_nonconf_writable,
            2,
        );
        let proxy = new_proxy(&mut heap, global, target, handler);
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "getOwnPropertyDescriptor"],
            Value::Undefined,
            &[proxy, kfixed],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));
    }

    #[test]
    fn proxy_gopd_undef_on_nonconfigurable_accessor_throws() {
        // Spec 10.5.1 step 10.b: reporting undefined for a
        // non-configurable accessor target property throws (previously
        // fell through to the (None, None) case).
        let (mut heap, global) = setup();
        let kx = sval(&mut heap, "x");
        let target = accessor_target(&mut heap, false);
        let handler = handler_with(&mut heap, "getOwnPropertyDescriptor", trap_undef, 2);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "getOwnPropertyDescriptor"],
            Value::Undefined,
            &[proxy, kx],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));

        // A configurable accessor reported absent is fine.
        let target = accessor_target(&mut heap, true);
        let handler = handler_with(&mut heap, "getOwnPropertyDescriptor", trap_undef, 2);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let r = call_path(
            &mut heap,
            global,
            &["Reflect", "getOwnPropertyDescriptor"],
            Value::Undefined,
            &[proxy, kx],
        )
        .unwrap();
        assert_eq!(r, Value::Undefined);
    }

    #[test]
    fn proxy_gopd_new_property_rules() {
        // Spec 10.5.1 step 13: on an extensible target, a data descriptor
        // for an absent key must report configurable:true ...
        let (mut heap, global) = setup();
        let kmissing = sval(&mut heap, "missing");
        let target = new_plain(&mut heap);
        let handler = handler_with(
            &mut heap,
            "getOwnPropertyDescriptor",
            trap_gopd_value_only,
            2,
        );
        let proxy = new_proxy(&mut heap, global, target, handler);
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "getOwnPropertyDescriptor"],
            Value::Undefined,
            &[proxy, kmissing],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));

        // ... but a generic descriptor (`{}` or `{configurable:false}`,
        // neither data nor accessor) is returned as-is with no invariants
        // (step 11), even when it reports configurable:false ...
        let handler = handler_with(&mut heap, "getOwnPropertyDescriptor", trap_gopd_empty, 2);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let r = call_path(
            &mut heap,
            global,
            &["Reflect", "getOwnPropertyDescriptor"],
            Value::Undefined,
            &[proxy, kmissing],
        )
        .unwrap();
        assert!(matches!(r, Value::Object(_)));
        let handler = handler_with(
            &mut heap,
            "getOwnPropertyDescriptor",
            trap_gopd_nonconf_only,
            2,
        );
        let proxy = new_proxy(&mut heap, global, target, handler);
        let r = call_path(
            &mut heap,
            global,
            &["Reflect", "getOwnPropertyDescriptor"],
            Value::Undefined,
            &[proxy, kmissing],
        )
        .unwrap();
        assert!(matches!(r, Value::Object(_)));

        // A data descriptor for an absent key on a non-extensible
        // target always throws (step 13.a); a generic descriptor still
        // returns as-is (step 11 precedes step 13).
        heap.prevent_extensions(target);
        let handler = handler_with(
            &mut heap,
            "getOwnPropertyDescriptor",
            trap_gopd_value_only,
            2,
        );
        let proxy = new_proxy(&mut heap, global, target, handler);
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "getOwnPropertyDescriptor"],
            Value::Undefined,
            &[proxy, kmissing],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));
        let handler = handler_with(&mut heap, "getOwnPropertyDescriptor", trap_gopd_empty, 2);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let r = call_path(
            &mut heap,
            global,
            &["Reflect", "getOwnPropertyDescriptor"],
            Value::Undefined,
            &[proxy, kmissing],
        )
        .unwrap();
        assert!(matches!(r, Value::Object(_)));
    }

    #[test]
    fn proxy_has_false_on_nonconfigurable_accessor_throws() {
        // Spec 10.5.7: a `has` trap denying a non-configurable accessor
        // property throws.
        let (mut heap, global) = setup();
        let kx = sval(&mut heap, "x");
        let target = accessor_target(&mut heap, false);
        let handler = handler_with(&mut heap, "has", trap_false, 2);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "has"],
            Value::Undefined,
            &[proxy, kx],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));
    }

    #[test]
    fn proxy_delete_true_on_nonconfigurable_accessor_throws() {
        // Spec 10.5.10: a `deleteProperty` trap succeeding on a
        // non-configurable accessor property throws.
        let (mut heap, global) = setup();
        let kx = sval(&mut heap, "x");
        let target = accessor_target(&mut heap, false);
        let handler = handler_with(&mut heap, "deleteProperty", trap_true, 2);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "deleteProperty"],
            Value::Undefined,
            &[proxy, kx],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));
    }

    fn desc_with(heap: &mut Heap, fields: &[(&str, Value)]) -> Value {
        let d = new_plain(heap);
        for (name, v) in fields {
            heap.define_property_str(d, name, Property::data(*v))
                .unwrap();
        }
        Value::Object(d)
    }

    #[test]
    fn proxy_define_nonconfigurable_addition_on_extensible_throws() {
        // Spec 10.5.6 step 12.c.ii: a trap claiming success for a
        // non-configurable addition on an extensible target throws, even
        // though the target has no such property.
        let (mut heap, global) = setup();
        let kn = sval(&mut heap, "n");
        let target = new_plain(&mut heap);
        let handler = handler_with(&mut heap, "defineProperty", trap_true, 3);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let d = desc_with(&mut heap, &[("configurable", Value::Bool(false))]);
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "defineProperty"],
            Value::Undefined,
            &[proxy, kn, d],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));

        // A configurable addition still succeeds.
        let d = desc_with(&mut heap, &[("configurable", Value::Bool(true))]);
        let r = call_path(
            &mut heap,
            global,
            &["Reflect", "defineProperty"],
            Value::Undefined,
            &[proxy, kn, d],
        )
        .unwrap();
        assert_eq!(r, Value::Bool(true));
    }

    #[test]
    fn proxy_define_accessor_target_invariants() {
        // Spec 10.5.6 step 12.d.i on an accessor targetDesc: flipping
        // configurable to true, or changing enumerable, of a
        // non-configurable accessor throws.
        let (mut heap, global) = setup();
        let undef = Value::Undefined;
        let kx = sval(&mut heap, "x");
        for fields in [
            &[("configurable", Value::Bool(true))][..],
            &[("enumerable", Value::Bool(false))][..],
        ] {
            let target = accessor_target(&mut heap, false);
            let handler = handler_with(&mut heap, "defineProperty", trap_true, 3);
            let proxy = new_proxy(&mut heap, global, target, handler);
            let d = desc_with(&mut heap, fields);
            let err = call_path(
                &mut heap,
                global,
                &["Reflect", "defineProperty"],
                undef,
                &[proxy, kx, d],
            )
            .unwrap_err();
            assert!(matches!(err, NativeError::TypeError(_)));
        }

        // A compatible redefinition of a non-configurable accessor (no
        // configurable/enumerable change) succeeds.
        let target = accessor_target(&mut heap, false);
        let handler = handler_with(&mut heap, "defineProperty", trap_true, 3);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let d = desc_with(&mut heap, &[("value", Value::Int(7))]);
        let r = call_path(
            &mut heap,
            global,
            &["Reflect", "defineProperty"],
            undef,
            &[proxy, kx, d],
        )
        .unwrap();
        assert_eq!(r, Value::Bool(true));
    }

    #[test]
    fn proxy_extensibility_and_prototype_invariants() {
        let (mut heap, global) = setup();
        let undef = Value::Undefined;

        // isExtensible trap disagreeing with the target throws.
        let target = fixed_target(&mut heap);
        let handler = handler_with(&mut heap, "isExtensible", trap_true, 1);
        let proxy = new_proxy(&mut heap, global, target, handler);
        call_path(
            &mut heap,
            global,
            &["Reflect", "preventExtensions"],
            undef,
            &[proxy],
        )
        .unwrap();
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "isExtensible"],
            undef,
            &[proxy],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));

        // preventExtensions trap succeeding without preventing throws.
        let target = fixed_target(&mut heap);
        let handler = handler_with(&mut heap, "preventExtensions", trap_true, 1);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "preventExtensions"],
            undef,
            &[proxy],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));

        // getPrototypeOf trap returning a different object on a
        // non-extensible target throws.
        let target = fixed_target(&mut heap);
        heap.prevent_extensions(target);
        let handler = handler_with(&mut heap, "getPrototypeOf", trap_new_object, 1);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "getPrototypeOf"],
            undef,
            &[proxy],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));

        // setPrototypeOf trap succeeding with a different proto on a
        // non-extensible target throws.
        let handler = handler_with(&mut heap, "setPrototypeOf", trap_true, 2);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let other = Value::Object(new_plain(&mut heap));
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "setPrototypeOf"],
            undef,
            &[proxy, other],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));
    }

    #[test]
    fn proxy_revocable() {
        let (mut heap, global) = setup();
        let target = new_plain(&mut heap);
        heap.define_property_str(target, "x", Property::data(Value::Int(5)))
            .unwrap();
        let handler = new_plain(&mut heap);
        let kx = sval(&mut heap, "x");
        let undef = Value::Undefined;

        let pair = call_path(
            &mut heap,
            global,
            &["Proxy", "revocable"],
            undef,
            &[Value::Object(target), Value::Object(handler)],
        )
        .unwrap();
        let pair_obj = require_object(pair, "t").unwrap();
        let proxy = get_prop_chain(&heap, pair_obj, Wtf8::from_str("proxy")).unwrap();
        let revoke = get_prop_chain(&heap, pair_obj, Wtf8::from_str("revoke")).unwrap();

        let got = call_path(&mut heap, global, &["Reflect", "get"], undef, &[proxy, kx]).unwrap();
        assert_eq!(got, Value::Int(5));
        // The revoke function is bound: the call-site receiver is replaced
        // by the captured proxy before `proxy_revoke` runs.
        let r = call_value(&mut heap, revoke, pair, &[]).unwrap();
        assert_eq!(r, Value::Undefined);
        let err =
            call_path(&mut heap, global, &["Reflect", "get"], undef, &[proxy, kx]).unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));
    }

    /// Getter probe: returns its `this`, so tests can assert the receiver.
    fn trap_return_this(
        _heap: &mut Heap,
        this: Value,
        _args: &[Value],
    ) -> Result<Value, NativeError> {
        Ok(this)
    }

    /// Setter probe: stashes the observed `this` as `sawThis` on the first
    /// argument (a caller-owned object), so tests can assert the receiver.
    fn trap_stash_this_on_arg(
        heap: &mut Heap,
        this: Value,
        args: &[Value],
    ) -> Result<Value, NativeError> {
        let v = require_object(get_arg(args, 0), "stash")?;
        heap.define_property_str(v, "sawThis", Property::data(this))?;
        Ok(Value::Undefined)
    }

    /// Set-trap probe: records the trap's receiver argument (`args[3]`) as
    /// `sawReceiver` on the handler (`this`), and reports success.
    fn trap_stash_receiver_on_this(
        heap: &mut Heap,
        this: Value,
        args: &[Value],
    ) -> Result<Value, NativeError> {
        let h = require_object(this, "stash")?;
        heap.define_property_str(h, "sawReceiver", Property::data(get_arg(args, 3)))?;
        Ok(Value::Bool(true))
    }

    fn trap_get_marker(
        _heap: &mut Heap,
        _this: Value,
        _args: &[Value],
    ) -> Result<Value, NativeError> {
        Ok(Value::Int(4242))
    }

    #[test]
    fn proxy_no_trap_set_fires_inherited_setter() {
        let (mut heap, global) = setup();
        let undef = Value::Undefined;
        // Prototype carries accessor `x` whose setter stashes its `this`.
        let proto = new_plain(&mut heap);
        let setter = new_native_function(&mut heap, "setx", trap_stash_this_on_arg, 1).unwrap();
        heap.define_accessor(
            proto,
            Wtf8::from_str("x"),
            crate::value::AccessorProp {
                get: Value::Undefined,
                set: Value::Object(setter),
                enumerable: true,
                configurable: true,
            },
        )
        .unwrap();
        let target = new_plain(&mut heap);
        heap.set_prototype(target, Value::Object(proto));
        let handler = new_plain(&mut heap);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let kx = sval(&mut heap, "x");
        let v = Value::Object(new_plain(&mut heap));

        let ok = call_path(
            &mut heap,
            global,
            &["Reflect", "set"],
            undef,
            &[proxy, kx, v],
        )
        .unwrap();
        assert_eq!(ok, Value::Bool(true));
        // The inherited setter ran with the proxy as its receiver.
        let v_obj = require_object(v, "t").unwrap();
        let saw = get_prop_chain(&heap, v_obj, Wtf8::from_str("sawThis")).unwrap();
        assert!(strict_equal(saw, proxy));
    }

    #[test]
    fn proxy_no_trap_get_invokes_accessor() {
        let (mut heap, global) = setup();
        let undef = Value::Undefined;
        let target = new_plain(&mut heap);
        let getter = new_native_function(&mut heap, "getx", trap_return_this, 0).unwrap();
        heap.define_accessor(
            target,
            Wtf8::from_str("x"),
            crate::value::AccessorProp {
                get: Value::Object(getter),
                set: Value::Undefined,
                enumerable: true,
                configurable: true,
            },
        )
        .unwrap();
        let handler = new_plain(&mut heap);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let kx = sval(&mut heap, "x");

        // Default receiver is the proxy itself.
        let got = call_path(&mut heap, global, &["Reflect", "get"], undef, &[proxy, kx]).unwrap();
        assert!(strict_equal(got, proxy));
        // An explicit receiver is honored by the accessor.
        let recv = Value::Object(new_plain(&mut heap));
        let got = call_path(
            &mut heap,
            global,
            &["Reflect", "get"],
            undef,
            &[proxy, kx, recv],
        )
        .unwrap();
        assert!(strict_equal(got, recv));
    }

    #[test]
    fn proxy_no_trap_delegates_to_nested_proxy_traps() {
        let (mut heap, global) = setup();
        let undef = Value::Undefined;
        // Inner proxy with get/set traps.
        let inner_target = new_plain(&mut heap);
        let inner_handler = handler_with(&mut heap, "get", trap_get_marker, 3);
        let set_trap =
            new_native_function(&mut heap, "set", trap_stash_receiver_on_this, 4).unwrap();
        heap.define_property_str(
            inner_handler,
            "set",
            Property::data(Value::Object(set_trap)),
        )
        .unwrap();
        let inner = new_proxy(&mut heap, global, inner_target, inner_handler);
        // Outer proxy with no traps, targeting the inner proxy.
        let inner_obj = require_object(inner, "t").unwrap();
        let outer_handler = new_plain(&mut heap);
        let outer = new_proxy(&mut heap, global, inner_obj, outer_handler);
        let kx = sval(&mut heap, "x");

        // The inner get trap fires through the no-trap outer proxy.
        let got = call_path(&mut heap, global, &["Reflect", "get"], undef, &[outer, kx]).unwrap();
        assert_eq!(got, Value::Int(4242));
        // The inner set trap fires too, observing the outer proxy as receiver.
        let ok = call_path(
            &mut heap,
            global,
            &["Reflect", "set"],
            undef,
            &[outer, kx, Value::Int(1)],
        )
        .unwrap();
        assert_eq!(ok, Value::Bool(true));
        let saw = get_prop_chain(&heap, inner_handler, Wtf8::from_str("sawReceiver")).unwrap();
        assert!(strict_equal(saw, outer));
    }

    #[test]
    fn reflect_set_forwards_receiver_to_set_trap() {
        let (mut heap, global) = setup();
        let undef = Value::Undefined;
        let target = new_plain(&mut heap);
        let handler = handler_with(&mut heap, "set", trap_stash_receiver_on_this, 4);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let kx = sval(&mut heap, "k");
        let recv = Value::Object(new_plain(&mut heap));

        // `Reflect.set(t, k, v, receiver)`: the trap sees the passed
        // receiver, not the proxy.
        let ok = call_path(
            &mut heap,
            global,
            &["Reflect", "set"],
            undef,
            &[proxy, kx, Value::Int(7), recv],
        )
        .unwrap();
        assert_eq!(ok, Value::Bool(true));
        let saw = get_prop_chain(&heap, handler, Wtf8::from_str("sawReceiver")).unwrap();
        assert!(strict_equal(saw, recv));
    }

    #[test]
    fn reflect_set_explicit_receiver_reaches_inherited_setter() {
        let (mut heap, global) = setup();
        let undef = Value::Undefined;
        let proto = new_plain(&mut heap);
        let setter = new_native_function(&mut heap, "setx", trap_stash_this_on_arg, 1).unwrap();
        heap.define_accessor(
            proto,
            Wtf8::from_str("x"),
            crate::value::AccessorProp {
                get: Value::Undefined,
                set: Value::Object(setter),
                enumerable: true,
                configurable: true,
            },
        )
        .unwrap();
        let target = new_plain(&mut heap);
        heap.set_prototype(target, Value::Object(proto));
        let handler = new_plain(&mut heap);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let kx = sval(&mut heap, "x");
        let v = Value::Object(new_plain(&mut heap));
        let recv = Value::Object(new_plain(&mut heap));

        let ok = call_path(
            &mut heap,
            global,
            &["Reflect", "set"],
            undef,
            &[proxy, kx, v, recv],
        )
        .unwrap();
        assert_eq!(ok, Value::Bool(true));
        let v_obj = require_object(v, "t").unwrap();
        let saw = get_prop_chain(&heap, v_obj, Wtf8::from_str("sawThis")).unwrap();
        assert!(strict_equal(saw, recv));
    }

    #[test]
    fn proxy_no_trap_set_nonwritable_reports_false() {
        let (mut heap, global) = setup();
        let undef = Value::Undefined;
        // `fixed` is non-configurable, non-writable on the target.
        let target = fixed_target(&mut heap);
        let handler = new_plain(&mut heap);
        let proxy = new_proxy(&mut heap, global, target, handler);
        let kfixed = sval(&mut heap, "fixed");

        // Spec `[[Set]]`: a failed ordinary set reports `false`, it does
        // not throw.
        let r = call_path(
            &mut heap,
            global,
            &["Reflect", "set"],
            undef,
            &[proxy, kfixed, Value::Int(2)],
        )
        .unwrap();
        assert_eq!(r, Value::Bool(false));
    }

    #[test]
    fn proxy_revoke_detached_and_ignores_receiver() {
        let (mut heap, global) = setup();
        let undef = Value::Undefined;
        let kx = sval(&mut heap, "x");

        let target_a = new_plain(&mut heap);
        heap.define_property_str(target_a, "x", Property::data(Value::Int(5)))
            .unwrap();
        let handler_a = new_plain(&mut heap);
        let pair_a = call_path(
            &mut heap,
            global,
            &["Proxy", "revocable"],
            undef,
            &[Value::Object(target_a), Value::Object(handler_a)],
        )
        .unwrap();
        let pair_a_obj = require_object(pair_a, "t").unwrap();
        let proxy_a = get_prop_chain(&heap, pair_a_obj, Wtf8::from_str("proxy")).unwrap();
        let revoke_a = get_prop_chain(&heap, pair_a_obj, Wtf8::from_str("revoke")).unwrap();

        let target_b = new_plain(&mut heap);
        heap.define_property_str(target_b, "x", Property::data(Value::Int(7)))
            .unwrap();
        let handler_b = new_plain(&mut heap);
        let pair_b = call_path(
            &mut heap,
            global,
            &["Proxy", "revocable"],
            undef,
            &[Value::Object(target_b), Value::Object(handler_b)],
        )
        .unwrap();
        let pair_b_obj = require_object(pair_b, "t").unwrap();
        let proxy_b = get_prop_chain(&heap, pair_b_obj, Wtf8::from_str("proxy")).unwrap();
        let revoke_b = get_prop_chain(&heap, pair_b_obj, Wtf8::from_str("revoke")).unwrap();

        // `Reflect.apply(a.revoke, bPair, [])` revokes A's proxy — the one
        // captured at creation — not B's.
        let no_args = Value::Object(new_value_array(&mut heap, &[]).unwrap());
        let r = call_path(
            &mut heap,
            global,
            &["Reflect", "apply"],
            undef,
            &[revoke_a, pair_b, no_args],
        )
        .unwrap();
        assert_eq!(r, Value::Undefined);
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "get"],
            undef,
            &[proxy_a, kx],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));
        // B's proxy is untouched and still functional.
        let got = call_path(
            &mut heap,
            global,
            &["Reflect", "get"],
            undef,
            &[proxy_b, kx],
        )
        .unwrap();
        assert_eq!(got, Value::Int(7));

        // Detached call (`this` is undefined) still revokes.
        let r = call_value(&mut heap, revoke_b, Value::Undefined, &[]).unwrap();
        assert_eq!(r, Value::Undefined);
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "get"],
            undef,
            &[proxy_b, kx],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));
    }

    #[test]
    fn proxy_apply_construct_forwarding() {
        let (mut heap, global) = setup();
        // The proxy target must be a constructor for the construct half:
        // `Reflect.construct` on a proxy of a non-constructor throws
        // `TypeError` (proxy `[[Construct]]` / `IsConstructor`).
        let add = new_native_constructor(&mut heap, "add", native_add, 2).unwrap();
        let handler = new_plain(&mut heap);
        let proxy = new_proxy(&mut heap, global, add, handler);
        let undef = Value::Undefined;

        let arg_list = new_value_array(&mut heap, &[Value::Int(2), Value::Int(3)]).unwrap();
        let al = Value::Object(arg_list);
        let r = call_path(
            &mut heap,
            global,
            &["Reflect", "apply"],
            undef,
            &[proxy, undef, al],
        )
        .unwrap();
        assert_eq!(r, Value::Int(5));

        // Reflect.apply on a non-callable throws.
        let plain = Value::Object(new_plain(&mut heap));
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "apply"],
            undef,
            &[plain, undef, al],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));

        // Reflect.construct through a proxy without a construct trap.
        let empty = new_value_array(&mut heap, &[]).unwrap();
        let inst = call_path(
            &mut heap,
            global,
            &["Reflect", "construct"],
            undef,
            &[proxy, Value::Object(empty)],
        )
        .unwrap();
        assert!(matches!(inst, Value::Object(_)));

        // A proxy of a non-constructible target is not a constructor
        // either (`IsConstructor` is proxy-transparent):
        // `Reflect.construct` throws `TypeError`.
        let plain_target = new_plain(&mut heap);
        let handler2 = new_plain(&mut heap);
        let proxy2 = new_proxy(&mut heap, global, plain_target, handler2);
        let err = call_path(
            &mut heap,
            global,
            &["Reflect", "construct"],
            undef,
            &[proxy2, Value::Object(empty)],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));
    }

    #[test]
    fn typed_array_aliasing_and_conversions() {
        let (mut heap, global) = setup();
        let undef = Value::Undefined;

        let buf = call_path(&mut heap, global, &["ArrayBuffer"], undef, &[Value::Int(8)]).unwrap();

        let u8 = call_path(&mut heap, global, &["Uint8Array"], undef, &[buf]).unwrap();
        let u16 = call_path(&mut heap, global, &["Uint16Array"], undef, &[buf]).unwrap();
        let u8o = require_object(u8, "t").unwrap();
        let u16o = require_object(u16, "t").unwrap();
        // Direct native calls leave `newTarget` unset, so the views get a
        // null prototype; link the installed per-type prototypes by hand
        // (what the VM's GetPrototypeFromConstructor does for `new`).
        for (o, name) in [(u8o, "Uint8Array"), (u16o, "Uint16Array")] {
            let proto = mget(&heap, global, &[name, "prototype"]);
            heap.set_prototype(o, proto);
        }

        // Write bytes through the Uint8 view, read a Uint16 through the
        // other view: shared backing store.
        typed_index_set(&mut heap, u8o, 0, Value::Int(0x34)).unwrap();
        typed_index_set(&mut heap, u8o, 1, Value::Int(0x12)).unwrap();
        assert_eq!(
            typed_index_get(&mut heap, u16o, 0).unwrap(),
            Value::Int(0x1234)
        );
        // And back: write the Uint16, read bytes.
        typed_index_set(&mut heap, u16o, 1, Value::Int(0xABCD)).unwrap();
        assert_eq!(
            typed_index_get(&mut heap, u8o, 2).unwrap(),
            Value::Int(0xCD)
        );
        assert_eq!(
            typed_index_get(&mut heap, u8o, 3).unwrap(),
            Value::Int(0xAB)
        );

        // length / byteLength / byteOffset resolve through the shared
        // `%TypedArray%.prototype` accessors (no longer own data
        // properties, per ECMA-262 22.2.3.16-18).
        for (o, name) in [
            (u8o, "length"),
            (u16o, "length"),
            (u16o, "byteLength"),
            (u16o, "byteOffset"),
        ] {
            assert!(
                heap.get_property(o, Wtf8::from_str(name)).is_none(),
                "{name} must not be an own data property"
            );
        }
        assert_eq!(ta_acc(&mut heap, u8o, "length"), Value::Int(8));
        assert_eq!(ta_acc(&mut heap, u16o, "length"), Value::Int(4));
        assert_eq!(ta_acc(&mut heap, u16o, "byteLength"), Value::Int(8));
        assert_eq!(ta_acc(&mut heap, u16o, "byteOffset"), Value::Int(0));

        // Int8 sign-extends.
        let i8 = call_path(&mut heap, global, &["Int8Array"], undef, &[buf]).unwrap();
        let i8o = require_object(i8, "t").unwrap();
        typed_index_set(&mut heap, i8o, 0, Value::Int(-1)).unwrap();
        assert_eq!(typed_index_get(&mut heap, u8o, 0).unwrap(), Value::Int(255));
        assert_eq!(typed_index_get(&mut heap, i8o, 0).unwrap(), Value::Int(-1));

        // Uint8Clamped clamping and round-half-to-even.
        let cl = call_path(
            &mut heap,
            global,
            &["Uint8ClampedArray"],
            undef,
            &[Value::Int(4)],
        )
        .unwrap();
        let clo = require_object(cl, "t").unwrap();
        typed_index_set(&mut heap, clo, 0, Value::Int(300)).unwrap();
        typed_index_set(&mut heap, clo, 1, Value::Int(-5)).unwrap();
        typed_index_set(&mut heap, clo, 2, Value::Float(2.5)).unwrap();
        typed_index_set(&mut heap, clo, 3, Value::Float(3.5)).unwrap();
        assert_eq!(typed_index_get(&mut heap, clo, 0).unwrap(), Value::Int(255));
        assert_eq!(typed_index_get(&mut heap, clo, 1).unwrap(), Value::Int(0));
        assert_eq!(typed_index_get(&mut heap, clo, 2).unwrap(), Value::Int(2));
        assert_eq!(typed_index_get(&mut heap, clo, 3).unwrap(), Value::Int(4));

        // Out-of-bounds reads undefined; writes ignored.
        assert_eq!(
            typed_index_get(&mut heap, clo, 9).unwrap(),
            Value::Undefined
        );
        typed_index_set(&mut heap, clo, 9, Value::Int(1)).unwrap();

        // Construct from a length.
        let f32 = call_path(
            &mut heap,
            global,
            &["Float32Array"],
            undef,
            &[Value::Int(2)],
        )
        .unwrap();
        let f32o = require_object(f32, "t").unwrap();
        heap.set_prototype(f32o, mget(&heap, global, &["Float32Array", "prototype"]));
        assert_eq!(ta_acc(&mut heap, f32o, "byteLength"), Value::Int(8));
        typed_index_set(&mut heap, f32o, 0, Value::Float(0.5)).unwrap();
        assert_eq!(
            typed_index_get(&mut heap, f32o, 0).unwrap(),
            Value::Float(0.5)
        );

        // Construct from an array-like.
        let src = new_value_array(&mut heap, &[Value::Int(10), Value::Int(20)]).unwrap();
        let cpy = call_path(
            &mut heap,
            global,
            &["Uint8Array"],
            undef,
            &[Value::Object(src)],
        )
        .unwrap();
        let cpyo = require_object(cpy, "t").unwrap();
        assert_eq!(typed_index_get(&mut heap, cpyo, 0).unwrap(), Value::Int(10));
        assert_eq!(typed_index_get(&mut heap, cpyo, 1).unwrap(), Value::Int(20));

        // Misaligned byte offset throws.
        let err = call_path(
            &mut heap,
            global,
            &["Uint16Array"],
            undef,
            &[buf, Value::Int(1)],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::RangeError(_)));

        // BYTES_PER_ELEMENT.
        let bpe = mget(&heap, global, &["Float64Array", "BYTES_PER_ELEMENT"]);
        assert_eq!(bpe, Value::Int(8));
    }

    #[test]
    fn dataview_endianness() {
        let (mut heap, global) = setup();
        let undef = Value::Undefined;

        let buf = call_path(&mut heap, global, &["ArrayBuffer"], undef, &[Value::Int(8)]).unwrap();
        let dv = call_path(&mut heap, global, &["DataView"], undef, &[buf]).unwrap();

        // Big-endian (default): 0x1234 -> [0x12, 0x34].
        call_path(
            &mut heap,
            global,
            &["DataView", "prototype", "setUint16"],
            dv,
            &[Value::Int(0), Value::Int(0x1234), Value::Bool(false)],
        )
        .unwrap();
        let b0 = call_path(
            &mut heap,
            global,
            &["DataView", "prototype", "getUint8"],
            dv,
            &[Value::Int(0)],
        )
        .unwrap();
        assert_eq!(b0, Value::Int(0x12));
        let b1 = call_path(
            &mut heap,
            global,
            &["DataView", "prototype", "getUint8"],
            dv,
            &[Value::Int(1)],
        )
        .unwrap();
        assert_eq!(b1, Value::Int(0x34));
        let v = call_path(
            &mut heap,
            global,
            &["DataView", "prototype", "getUint16"],
            dv,
            &[Value::Int(0), Value::Bool(false)],
        )
        .unwrap();
        assert_eq!(v, Value::Int(0x1234));

        // Little-endian: 0x1234 -> [0x34, 0x12].
        call_path(
            &mut heap,
            global,
            &["DataView", "prototype", "setUint16"],
            dv,
            &[Value::Int(2), Value::Int(0x1234), Value::Bool(true)],
        )
        .unwrap();
        let b2 = call_path(
            &mut heap,
            global,
            &["DataView", "prototype", "getUint8"],
            dv,
            &[Value::Int(2)],
        )
        .unwrap();
        assert_eq!(b2, Value::Int(0x34));
        let b3 = call_path(
            &mut heap,
            global,
            &["DataView", "prototype", "getUint8"],
            dv,
            &[Value::Int(3)],
        )
        .unwrap();
        assert_eq!(b3, Value::Int(0x12));
        let v = call_path(
            &mut heap,
            global,
            &["DataView", "prototype", "getUint16"],
            dv,
            &[Value::Int(2), Value::Bool(true)],
        )
        .unwrap();
        assert_eq!(v, Value::Int(0x1234));

        // Float64 round-trip.
        call_path(
            &mut heap,
            global,
            &["DataView", "prototype", "setFloat64"],
            dv,
            &[Value::Int(0), Value::Float(1.5), Value::Bool(true)],
        )
        .unwrap();
        let f = call_path(
            &mut heap,
            global,
            &["DataView", "prototype", "getFloat64"],
            dv,
            &[Value::Int(0), Value::Bool(true)],
        )
        .unwrap();
        assert_eq!(f, Value::Float(1.5));

        // Out-of-bounds access throws RangeError.
        let err = call_path(
            &mut heap,
            global,
            &["DataView", "prototype", "getUint16"],
            dv,
            &[Value::Int(7)],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::RangeError(_)));

        // DataView requires an ArrayBuffer.
        let plain = Value::Object(new_plain(&mut heap));
        let err = call_path(&mut heap, global, &["DataView"], undef, &[plain]).unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));
    }

    #[test]
    fn arraybuffer_isview_slice_and_shared() {
        let (mut heap, global) = setup();
        let undef = Value::Undefined;

        let buf = call_path(&mut heap, global, &["ArrayBuffer"], undef, &[Value::Int(8)]).unwrap();
        let u8 = call_path(&mut heap, global, &["Uint8Array"], undef, &[buf]).unwrap();
        let u8o = require_object(u8, "t").unwrap();
        for i in 0..8u64 {
            typed_index_set(&mut heap, u8o, i, Value::Int(i as i32)).unwrap();
        }
        let dv = call_path(&mut heap, global, &["DataView"], undef, &[buf]).unwrap();

        let r = call_path(&mut heap, global, &["ArrayBuffer", "isView"], undef, &[u8]).unwrap();
        assert_eq!(r, Value::Bool(true));
        let r = call_path(&mut heap, global, &["ArrayBuffer", "isView"], undef, &[dv]).unwrap();
        assert_eq!(r, Value::Bool(true));
        let r = call_path(&mut heap, global, &["ArrayBuffer", "isView"], undef, &[buf]).unwrap();
        assert_eq!(r, Value::Bool(false));
        let plain = Value::Object(new_plain(&mut heap));
        let r = call_path(
            &mut heap,
            global,
            &["ArrayBuffer", "isView"],
            undef,
            &[plain],
        )
        .unwrap();
        assert_eq!(r, Value::Bool(false));

        // slice via ArrayBuffer.prototype.
        let s = call_path(
            &mut heap,
            global,
            &["ArrayBuffer", "prototype", "slice"],
            buf,
            &[Value::Int(2), Value::Int(5)],
        )
        .unwrap();
        let so = require_object(s, "t").unwrap();
        assert_eq!(array_buffer_length(&heap, so), 3);
        assert_eq!(buffer_get_byte(&heap, so, 0), 2);
        assert_eq!(buffer_get_byte(&heap, so, 2), 4);

        // SharedArrayBuffer constructs like ArrayBuffer (v1 aliasing).
        let sbuf = call_path(
            &mut heap,
            global,
            &["SharedArrayBuffer"],
            undef,
            &[Value::Int(4)],
        )
        .unwrap();
        assert!(is_array_buffer(&heap, sbuf));
        assert_eq!(
            array_buffer_length(&heap, require_object(sbuf, "t").unwrap()),
            4
        );

        // Negative length throws.
        let err = call_path(
            &mut heap,
            global,
            &["ArrayBuffer"],
            undef,
            &[Value::Int(-1)],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::RangeError(_)));
    }

    // -- VM re-entry regression tests ------------------------------------

    /// Evaluate JS in a full VM and stringify the completion value.
    fn eval_str(source: &str) -> String {
        let mut vm = crate::vm::Vm::new();
        let v = vm
            .eval(source)
            .unwrap_or_else(|e| panic!("eval failed: {e}"));
        crate::builtin_object_array::to_string_js(vm.heap_mut(), v)
            .map(|w| w.to_string())
            .unwrap_or_else(|e| panic!("to_string failed: {e:?}"))
    }

    #[test]
    fn typed_array_prototypes_installed_for_all_types() {
        // Every constructor's `prototype` must be the prototype the
        // corresponding instances link to (spec `%<T>Array.prototype%`).
        let r = eval_str(
            "var names = ['Int8Array','Uint8Array','Uint8ClampedArray','Int16Array',\
             'Uint16Array','Int32Array','Uint32Array','Float32Array','Float64Array']; \
             names.map(function(n) { \
               return Object.getPrototypeOf(new globalThis[n]()) === globalThis[n].prototype; \
             }).join(',')",
        );
        assert_eq!(r, "true,true,true,true,true,true,true,true,true");
    }

    #[test]
    fn dataview_prototype_chain_linked() {
        // `DataView.prototype`'s [[Prototype]] is `%Object.prototype%`
        // (ECMA-262 24.3.3), and instances link to it via
        // GetPrototypeFromConstructor, so `instanceof` and method
        // dispatch work. Regression: `dv_proto` was allocated with no
        // prototype and instances were allocated with none, so
        // `new DataView(ab).getInt8(0)` threw undefined-is-not-a-function.
        let r = eval_str(
            "var ab = new ArrayBuffer(4); \
             var dv = new DataView(ab); \
             dv.setInt8(0, 42); \
             (dv.getInt8(0) === 42) + ',' + \
             (dv instanceof DataView) + ',' + \
             (Object.getPrototypeOf(dv) === DataView.prototype) + ',' + \
             (Object.getPrototypeOf(DataView.prototype) === Object.prototype) + ',' + \
             (dv.constructor === DataView)",
        );
        assert_eq!(r, "true,true,true,true,true");
    }

    #[test]
    fn dataview_subclass_instances_link_to_subclass_prototype() {
        // `class D extends DataView` construction resolves the prototype
        // from new.target's `prototype` (GetPrototypeFromConstructor), so
        // subclass instances get `D.prototype` and subclass methods.
        let r = eval_str(
            "class MyDV extends DataView { doubleGet(i) { return this.getInt8(i) * 2; } } \
             var dv = new MyDV(new ArrayBuffer(2)); \
             dv.setInt8(1, 21); \
             (dv.doubleGet(1) === 42) + ',' + \
             (dv instanceof DataView) + ',' + \
             (dv instanceof MyDV) + ',' + \
             (Object.getPrototypeOf(dv) === MyDV.prototype) + ',' + \
             (Object.getPrototypeOf(MyDV.prototype) === DataView.prototype)",
        );
        assert_eq!(r, "true,true,true,true,true");
    }

    #[test]
    fn typed_array_prototype_methods_present_for_all_types() {
        // The kind-generic methods plus the `buffer` accessor live on each
        // type's prototype; `'buffer' in p` checks presence without
        // invoking the getter (it throws on a non-view).
        let r = eval_str(
            "var names = ['Int8Array','Uint8Array','Uint8ClampedArray','Int16Array',\
             'Uint16Array','Int32Array','Uint32Array','Float32Array','Float64Array']; \
             names.map(function(n) { var p = globalThis[n].prototype; \
               return typeof p.slice === 'function' && typeof p.set === 'function' \
                 && typeof p.subarray === 'function' && ('buffer' in p); \
             }).join(',')",
        );
        assert_eq!(r, "true,true,true,true,true,true,true,true,true");
    }

    #[test]
    fn typed_array_buffer_getter_works_for_all_types() {
        // End-to-end: the `buffer` getter on each installed prototype
        // returns the view's backing ArrayBuffer.
        let r = eval_str(
            "var names = ['Int8Array','Uint8Array','Uint8ClampedArray','Int16Array',\
             'Uint16Array','Int32Array','Uint32Array','Float32Array','Float64Array']; \
             names.map(function(n) { return new globalThis[n](4).buffer instanceof ArrayBuffer; }) \
             .join(',')",
        );
        assert_eq!(r, "true,true,true,true,true,true,true,true,true");
    }

    #[test]
    fn proxy_get_trap_accepts_bytecode_function() {
        // Note: exercised through `Reflect.get`; the VM's own property
        // load path does not consult proxies yet (separate engine gap).
        let r = eval_str("let p = new Proxy({x: 1}, { get: (t, k) => 42 }); Reflect.get(p, 'x')");
        assert_eq!(r, "42");
    }

    #[test]
    fn proxy_set_trap_accepts_bytecode_function() {
        let r = eval_str(
            "let seen = ''; let p = new Proxy({}, { set: (t, k, v) => { seen = k + '=' + v; return true; } }); Reflect.set(p, 'a', 7); seen",
        );
        assert_eq!(r, "a=7");
    }

    #[test]
    fn proxy_apply_trap_accepts_bytecode_function() {
        let r = eval_str(
            "function f(a) { return a * 2; } let p = new Proxy(f, { apply: (t, th, args) => 99 }); Reflect.apply(p, null, [1])",
        );
        assert_eq!(r, "99");
    }

    #[test]
    fn reflect_apply_accepts_bytecode_target() {
        let r = eval_str("Reflect.apply((a, b) => a + b, null, [2, 3])");
        assert_eq!(r, "5");
    }

    #[test]
    fn reflect_construct_accepts_bytecode_target() {
        let r = eval_str("function C(a) { this.a = a; } Reflect.construct(C, [7]).a");
        assert_eq!(r, "7");
    }

    #[test]
    fn reflect_construct_native_target_observes_new_target() {
        // `Reflect.construct(Object, [], D)` must proto-link the instance
        // to D.prototype: the native path stashes newTarget in the
        // `NEW_TARGET` thread-local exactly like the VM construct path.
        let r = eval_str(
            "function D() {} \
             Object.getPrototypeOf(Reflect.construct(Object, [], D)) === D.prototype",
        );
        assert_eq!(r, "true");
    }

    #[test]
    fn reflect_construct_native_array_target_observes_new_target() {
        let r = eval_str(
            "function D() {} \
             Object.getPrototypeOf(Reflect.construct(Array, [], D)) === D.prototype",
        );
        assert_eq!(r, "true");
    }

    #[test]
    fn reflect_construct_native_target_defaults_new_target() {
        // Omitted newTarget: defaults to the target itself.
        let r =
            eval_str("Object.getPrototypeOf(Reflect.construct(Object, [])) === Object.prototype");
        assert_eq!(r, "true");
        let r = eval_str("Object.getPrototypeOf(Reflect.construct(Array, [])) === Array.prototype");
        assert_eq!(r, "true");
    }

    #[test]
    fn reflect_construct_proxy_wrapped_native_target_observes_new_target() {
        // A proxy around a native target with no construct trap forwards
        // to the same native path, so the newTarget linkage is identical.
        let r = eval_str(
            "function D() {} \
             var p = new Proxy(Object, {}); \
             Object.getPrototypeOf(Reflect.construct(p, [], D)) === D.prototype",
        );
        assert_eq!(r, "true");
    }

    #[test]
    fn reflect_construct_clears_new_target_afterwards() {
        // A stale newTarget must not leak into a later plain construction.
        let r = eval_str(
            "function D() {} \
             Reflect.construct(Object, [], D); \
             Object.getPrototypeOf(new Object()) === Object.prototype",
        );
        assert_eq!(r, "true");
    }

    // -- Prototype-cycle hang regressions ----------------------------------

    #[test]
    fn reflect_set_prototype_of_rejects_two_cycle() {
        // The hang report's trigger: the second call must fail (return
        // false) instead of installing the cycle.
        let r = eval_str(
            "var a = {}, b = {}; \
             var first = Reflect.setPrototypeOf(a, b); \
             var second = Reflect.setPrototypeOf(b, a); \
             first + ',' + second",
        );
        assert_eq!(r, "true,false");
    }

    #[test]
    fn reflect_set_prototype_of_rejects_self_cycle() {
        let r = eval_str("var a = {}; Reflect.setPrototypeOf(a, a)");
        assert_eq!(r, "false");
    }

    #[test]
    fn reflect_set_prototype_of_allows_acyclic_chain() {
        let r = eval_str(
            "var a = {}, b = {}; \
             Reflect.setPrototypeOf(a, b); \
             Reflect.getPrototypeOf(a) === b",
        );
        assert_eq!(r, "true");
    }

    #[test]
    fn string_on_rejected_cycle_terminates() {
        // Full trigger from the hang report: `Reflect.setPrototypeOf(b, a)`
        // now fails, so `String(a)` must terminate with the default.
        let r = eval_str(
            "var a = {}, b = {}; \
             Reflect.setPrototypeOf(a, b); \
             Reflect.setPrototypeOf(b, a); \
             String(a)",
        );
        assert_eq!(r, "[object Object]");
    }

    #[test]
    fn proxy_set_prototype_of_no_trap_rejects_cycle() {
        // No-trap path forwards to the target's `[[SetPrototypeOf]]`,
        // which must reject the cycle just like the direct path.
        let r = eval_str(
            "var a = {}, b = {}; \
             var p = new Proxy(a, {}); \
             Reflect.setPrototypeOf(b, a); \
             Reflect.setPrototypeOf(p, b)",
        );
        assert_eq!(r, "false");
    }

    #[test]
    fn get_prop_chain_terminates_on_prototype_cycle() {
        // A 2-cycle built directly on the heap, bypassing the
        // `[[SetPrototypeOf]]` cycle checks: the guarded walks must
        // terminate (returning "not found") instead of looping forever.
        let mut heap = Heap::new();
        let a = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let b = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.set_prototype(a, Value::Object(b));
        heap.set_prototype(b, Value::Object(a));
        assert_eq!(get_prop_chain(&heap, a, Wtf8::from_str("toString")), None);
        assert!(!has_prop_chain(&heap, a, Wtf8::from_str("toString")));
    }

    // -- hidden-slot forging and numeric hardening regressions ------------

    #[test]
    fn proxy_forged_target_slot_is_ordinary_object() {
        // JS strings can contain NUL, so the hidden key alone cannot stop
        // forging: a forged `\0bowser:proxy-target` slot must not make an
        // ordinary object behave like a proxy. No panic, ordinary `[[Get]]`.
        let r = eval_str(
            "var o = {}; o[\"\\0bowser:proxy-target\"] = 0; o.x = 7; Reflect.get(o, \"x\")",
        );
        assert_eq!(r, "7");
    }

    #[test]
    fn proxy_forged_partial_slots_are_ordinary_objects() {
        // Object-valued target but no handler: still not a proxy.
        let r =
            eval_str("var o = {}; o[\"\\0bowser:proxy-target\"] = {}; Reflect.get(o, \"missing\")");
        assert_eq!(r, "undefined");
    }

    #[test]
    fn typed_array_forged_etype_fails_closed() {
        // Forgery via NUL-prefixed property names is impossible now that
        // view identity is a kind tag: the object fails the typed-array
        // predicate, so the constructor treats it as an array-like of
        // length 0.
        let r = eval_str(
            "var o = {}; \
             o[\"\\0bowser:ta-buffer\"] = new ArrayBuffer(8); \
             o[\"\\0bowser:ta-offset\"] = 0; \
             o[\"\\0bowser:ta-length\"] = 2; \
             o[\"\\0bowser:ta-etype\"] = 99; \
             var v = new Int32Array(o); v.length",
        );
        assert_eq!(r, "0");
    }

    #[test]
    fn typed_array_forged_negative_slots_fail_closed() {
        // NUL-prefixed slot names on a plain object are inert properties,
        // not view metadata: the object is not a view.
        let r = eval_str(
            "var o = {}; \
             o[\"\\0bowser:ta-buffer\"] = new ArrayBuffer(8); \
             o[\"\\0bowser:ta-offset\"] = -1; \
             o[\"\\0bowser:ta-length\"] = -1; \
             o[\"\\0bowser:ta-etype\"] = -1; \
             var v = new Int32Array(o); v.length",
        );
        assert_eq!(r, "0");
    }

    #[test]
    fn typed_array_forged_fractional_slots_fail_closed() {
        // Same: NUL-prefixed names do not make a view.
        let r = eval_str(
            "var o = {}; \
             o[\"\\0bowser:ta-buffer\"] = new ArrayBuffer(8); \
             o[\"\\0bowser:ta-offset\"] = 0.5; \
             o[\"\\0bowser:ta-length\"] = 1.5; \
             o[\"\\0bowser:ta-etype\"] = 5; \
             var v = new Int32Array(o); v.length",
        );
        assert_eq!(r, "0");
    }

    #[test]
    fn view_identity_is_unforgeable_from_script() {
        // A plain object naming every old hidden-slot key is not a view:
        // `ArrayBuffer.isView` is false, `DataView` construction throws,
        // and `TextDecoder` (via `is_genuine_view`) rejects it.
        for src in [
            "var o = {}; \
             o[\"\\0bowser:ta-buffer\"] = new ArrayBuffer(8); \
             o[\"\\0bowser:ta-offset\"] = 0; \
             o[\"\\0bowser:ta-length\"] = 8; \
             o[\"\\0bowser:ta-etype\"] = 1; \
             ArrayBuffer.isView(o)",
            "var o = {}; \
             o[\"\\0bowser:dv-buffer\"] = new ArrayBuffer(8); \
             o[\"\\0bowser:dv-offset\"] = 0; \
             o[\"\\0bowser:dv-length\"] = 8; \
             ArrayBuffer.isView(o)",
            "var o = {}; o[\"\\0bowser:arraybuffer\"] = true; ArrayBuffer.isView(o)",
            // Genuine views still pass.
            "ArrayBuffer.isView(new Uint8Array(4))",
            "ArrayBuffer.isView(new DataView(new ArrayBuffer(4)))",
        ] {
            let expect = if src.starts_with("ArrayBuffer.isView(new") {
                "true"
            } else {
                "false"
            };
            assert_eq!(eval_str(src), expect, "src: {src}");
        }
        // A forged "buffer" is not an ArrayBuffer: `DataView` rejects it
        // with a TypeError, and the typed-array constructor treats it as
        // an ordinary array-like (length 0) — in neither case is it
        // recognized as a buffer.
        assert_eq!(
            eval_str(
                "var ok = false; var o = {}; o[\"\\0bowser:arraybuffer\"] = true; \
                 try { new DataView(o); } catch (e) { ok = e instanceof TypeError; } ok"
            ),
            "true"
        );
        assert_eq!(
            eval_str(
                "var v = new Uint8Array((() => { var o = {}; \
                 o[\"\\0bowser:arraybuffer\"] = true; return o; })()); v.length"
            ),
            "0"
        );
    }

    #[test]
    fn view_metadata_cannot_be_tampered_from_script() {
        // Writing the old slot names onto a real view creates inert
        // ordinary properties; the view's behavior is unchanged.
        let r = eval_str(
            "var v = new Uint8Array([10, 20, 30]); \
             v[\"\\0bowser:ta-length\"] = 999; \
             v[\"\\0bowser:ta-etype\"] = 5; \
             v[\"\\0bowser:ta-offset\"] = 2; \
             v.length + ',' + v[0] + ',' + v[2]",
        );
        assert_eq!(r, "3,10,30");
    }

    #[test]
    fn dataview_metadata_cannot_be_tampered() {
        // Writing the old slot name onto a real DataView creates an inert
        // ordinary property; the view's behavior is unchanged. (Native
        // `call_path`: methods are invoked off `DataView.prototype`
        // directly; the instance's own prototype link is exercised by the
        // `dataview_prototype_chain_linked` eval test.)
        let (mut heap, global) = setup();
        let undef = Value::Undefined;
        let b = call_path(&mut heap, global, &["ArrayBuffer"], undef, &[Value::Int(4)]).unwrap();
        let dv = call_path(&mut heap, global, &["DataView"], undef, &[b]).unwrap();
        let view = match dv {
            Value::Object(o) => o,
            _ => panic!("DataView() did not return an object"),
        };
        heap.define_property_str(
            view,
            "\u{0}bowser:dv-length",
            Property {
                value: Value::Int(999),
                writable: true,
                enumerable: false,
                configurable: false,
            },
        )
        .unwrap();
        let v = call_path(
            &mut heap,
            global,
            &["DataView", "prototype", "getUint8"],
            dv,
            &[Value::Int(3)],
        )
        .unwrap();
        assert_eq!(v, Value::Int(0));
        assert!(matches!(
            call_path(
                &mut heap,
                global,
                &["DataView", "prototype", "getUint8"],
                dv,
                &[Value::Int(4)],
            ),
            Err(NativeError::RangeError(_))
        ));
    }

    #[test]
    fn view_tostring_tags() {
        // The kind tags drive `Object.prototype.toString` per spec.
        let r = eval_str(
            "var t = Object.prototype.toString; \
             t.call(new Uint8Array(1)) + '|' + t.call(new Int32Array(1)) + '|' + \
             t.call(new Float64Array(1)) + '|' + t.call(new ArrayBuffer(1)) + '|' + \
             t.call(new DataView(new ArrayBuffer(1)))",
        );
        assert_eq!(
            r,
            "[object Uint8Array]|[object Int32Array]|[object Float64Array]|[object ArrayBuffer]|[object DataView]"
        );
    }

    #[test]
    fn date_value_slot_is_unforgeable_from_script() {
        // A plain object with the old hidden slot name is not a Date:
        // prototype methods brand-check the kind tag and throw.
        let r = eval_str(
            "var o = {}; o[\"\\0bowser:date-value\"] = 0; \
             var ok = false; \
             try { Date.prototype.getTime.call(o); } \
             catch (e) { ok = e instanceof TypeError; } \
             ok",
        );
        assert_eq!(r, "true");
        // Writing the slot name onto a real Date does not change its value.
        let r = eval_str(
            "var d = new Date(12345); \
             d[\"\\0bowser:date-value\"] = 99999; \
             d.getTime()",
        );
        assert_eq!(r, "12345");
    }

    #[test]
    fn arraybuffer_huge_length_fails_fast() {
        // Must throw RangeError immediately, not hang zeroing or OOM.
        for src in [
            "var ok = false; try { new ArrayBuffer(2 ** 32); } catch (e) { ok = e instanceof RangeError; } ok",
            "var ok = false; try { new Uint8Array(2 ** 53); } catch (e) { ok = e instanceof RangeError; } ok",
            "var ok = false; try { new Int32Array(2 ** 53); } catch (e) { ok = e instanceof RangeError; } ok",
        ] {
            assert_eq!(eval_str(src), "true", "src: {src}");
        }
    }

    #[test]
    fn arraybuffer_backing_store_is_shared_not_propertied() {
        // Views share one backing store: writes through one DataView are
        // visible through another, and no indexed byte properties exist on
        // the buffer object itself. (`call_path`: `new ArrayBuffer()`
        // instances are proto-linked to `ArrayBuffer.prototype` via the
        // heap intrinsic, so method syntax works too.)
        let (mut heap, global) = setup();
        let undef = Value::Undefined;
        let buf = call_path(&mut heap, global, &["ArrayBuffer"], undef, &[Value::Int(4)]).unwrap();
        let buf_obj = require_object(buf, "t").unwrap();
        let dv = call_path(&mut heap, global, &["DataView"], undef, &[buf]).unwrap();
        for (i, b) in [(0, 0x12), (1, 0x34)] {
            call_path(
                &mut heap,
                global,
                &["DataView", "prototype", "setUint8"],
                dv,
                &[Value::Int(i), Value::Int(b)],
            )
            .unwrap();
        }
        let dv2 = call_path(&mut heap, global, &["DataView"], undef, &[buf]).unwrap();
        let v = call_path(
            &mut heap,
            global,
            &["DataView", "prototype", "getUint16"],
            dv2,
            &[Value::Int(0)],
        )
        .unwrap();
        assert_eq!(v, Value::Int(0x1234));
        // Not one property per byte.
        assert_eq!(get_prop_chain(&heap, buf_obj, Wtf8::from_str("0")), None);
        assert_eq!(
            get_prop_chain(&heap, buf_obj, Wtf8::from_str("byteLength")),
            Some(Value::Int(4))
        );
    }

    #[test]
    fn arraybuffer_slice_copies_bytes() {
        let (mut heap, global) = setup();
        let undef = Value::Undefined;
        let buf = call_path(&mut heap, global, &["ArrayBuffer"], undef, &[Value::Int(4)]).unwrap();
        let dv = call_path(&mut heap, global, &["DataView"], undef, &[buf]).unwrap();
        for (i, b) in [(0, 1), (1, 2), (2, 3), (3, 4)] {
            call_path(
                &mut heap,
                global,
                &["DataView", "prototype", "setUint8"],
                dv,
                &[Value::Int(i), Value::Int(b)],
            )
            .unwrap();
        }
        let s = call_path(
            &mut heap,
            global,
            &["ArrayBuffer", "prototype", "slice"],
            buf,
            &[Value::Int(1), Value::Int(3)],
        )
        .unwrap();
        let s_obj = require_object(s, "t").unwrap();
        assert_eq!(
            get_prop_chain(&heap, s_obj, Wtf8::from_str("byteLength")),
            Some(Value::Int(2))
        );
        let sdv = call_path(&mut heap, global, &["DataView"], undef, &[s]).unwrap();
        for (i, want) in [(0, 2), (1, 3)] {
            let v = call_path(
                &mut heap,
                global,
                &["DataView", "prototype", "getUint8"],
                sdv,
                &[Value::Int(i)],
            )
            .unwrap();
            assert_eq!(v, Value::Int(want));
        }
    }

    #[test]
    fn arraybuffer_slice_observable_start_end_conversion() {
        // `start`/`end` go through a full `ToIntegerOrInfinity`:
        // `valueOf` hooks run (test262 `slice/number-conversion.js`).
        let r = eval_str(
            "var ab = new ArrayBuffer(8); \
             var s = ab.slice({ valueOf: function() { return 2; } }, \
                              { valueOf: function() { return 6; } }); \
             s.byteLength",
        );
        assert_eq!(r, "4");
    }

    #[test]
    fn arraybuffer_moderate_size_is_fast() {
        // 1 MiB allocates as one zeroed block, not 1M property inserts;
        // the last byte is addressable and the rest is zero.
        let (mut heap, global) = setup();
        let undef = Value::Undefined;
        let n = 1 << 20;
        let buf = call_path(&mut heap, global, &["ArrayBuffer"], undef, &[Value::Int(n)]).unwrap();
        let buf_obj = require_object(buf, "t").unwrap();
        let dv = call_path(&mut heap, global, &["DataView"], undef, &[buf]).unwrap();
        call_path(
            &mut heap,
            global,
            &["DataView", "prototype", "setUint8"],
            dv,
            &[Value::Int(n - 1), Value::Int(7)],
        )
        .unwrap();
        let last = call_path(
            &mut heap,
            global,
            &["DataView", "prototype", "getUint8"],
            dv,
            &[Value::Int(n - 1)],
        )
        .unwrap();
        let first = call_path(
            &mut heap,
            global,
            &["DataView", "prototype", "getUint8"],
            dv,
            &[Value::Int(0)],
        )
        .unwrap();
        assert_eq!(last, Value::Int(7));
        assert_eq!(first, Value::Int(0));
        assert_eq!(
            get_prop_chain(&heap, buf_obj, Wtf8::from_str("byteLength")),
            Some(Value::Int(n))
        );
    }

    // -- Regression tests: builtin result arrays inherit Array.prototype --
    //
    // `Reflect.ownKeys` results are `CreateArrayFromList` arrays, so they
    // must inherit `%Array.prototype%` (a null-prototype regression once
    // made `.join`/`instanceof Array` fail on them).

    #[test]
    fn reflect_own_keys_result_is_real_array() {
        assert_eq!(
            eval_str("Reflect.ownKeys({a:1,b:2}) instanceof Array"),
            "true"
        );
        assert_eq!(eval_str("Reflect.ownKeys({a:1,b:2}).join(',')"), "a,b");
        assert_eq!(
            eval_str("Reflect.ownKeys(new Proxy({a:1}, {})).join(',')"),
            "a"
        );
    }

    // -- Reflect.defineProperty on arrays goes through the array exotic --
    //
    // `Reflect.defineProperty` must surface the array exotic
    // `[[DefineOwnProperty]]` outcomes (spec 10.4.2): `false` when the
    // define is rejected, `RangeError` for a non-Uint32 `length`,
    // `true` on success.

    #[test]
    fn reflect_define_property_array_length_shrink_returns_false() {
        // Shrinking past a non-configurable element fails: `false` is
        // returned and `length` is restored to failed_index + 1.
        let r = eval_str(
            "let a = [1,2,3]; \
             Object.defineProperty(a, '1', {value: 2, configurable: false}); \
             let ok = Reflect.defineProperty(a, 'length', {value: 0}); \
             ok + ',' + a.length + ',' + ('1' in a) + ',' + a[1]",
        );
        assert_eq!(r, "false,2,true,2");
    }

    #[test]
    fn reflect_define_property_array_invalid_length_throws_range_error() {
        let r = eval_str(
            "let a = [1,2,3]; \
             let out; \
             try { Reflect.defineProperty(a, 'length', {value: -1}); out = 'no-throw'; } \
             catch (e) { out = (e instanceof RangeError) ? 'RangeError' : 'other'; } \
             out + ',' + a.length",
        );
        assert_eq!(r, "RangeError,3");
    }

    #[test]
    fn reflect_define_property_array_index_extends_length() {
        let r = eval_str(
            "let a = [1]; \
             let ok = Reflect.defineProperty(a, '3', {value: 42}); \
             ok + ',' + a.length + ',' + a[3]",
        );
        assert_eq!(r, "true,4,42");
    }

    // Note: the install path must survive a collection on every allocation
    // (threshold 1): the `ArrayBuffer`/`DataView` prototypes are allocated
    // before their constructors, so without the temp-roots they would be
    // freed mid-install and the constructors' `prototype` links would
    // dangle.
    // (Covered by `install_paths_survive_threshold_1_collections` in
    // `tests/vm.rs`.)

    #[test]
    fn typed_array_of_from_statics_present_for_all_types() {
        // `%TypedArray%.of` (length 0) and `%TypedArray%.from` (length 1)
        // on every concrete constructor.
        let r = eval_str(
            "var names = ['Int8Array','Uint8Array','Uint8ClampedArray','Int16Array',\
             'Uint16Array','Int32Array','Uint32Array','Float32Array','Float64Array']; \
             names.map(function(n) { \
               var C = globalThis[n]; \
               return (typeof C.of === 'function' && C.of.length === 0 \
                 && typeof C.from === 'function' && C.from.length === 1) ? '1' : '0'; \
             }).join(',')",
        );
        assert_eq!(r, "1,1,1,1,1,1,1,1,1");
    }

    #[test]
    fn typed_array_of_builds_converted_view() {
        // Element conversion applies (300 wraps to 44 in Uint8); the
        // result links to `%Uint8Array.prototype%`.
        let r = eval_str(
            "var a = Uint8Array.of(1, 2, 300); \
             a.length + ',' + a[0] + ',' + a[1] + ',' + a[2] + ',' + \
             (Object.getPrototypeOf(a) === Uint8Array.prototype)",
        );
        assert_eq!(r, "3,1,2,44,true");
    }

    #[test]
    fn typed_array_of_respects_subclass_this() {
        // `D.of` is inherited from `Uint8Array`; `this` is `D`, so the
        // instance is a `D` (Uint8 elements, `D.prototype`).
        let r = eval_str(
            "class D extends Uint8Array {} \
             var d = D.of(7, 8); \
             (d instanceof D) + ',' + (d instanceof Uint8Array) + ',' + \
             (Object.getPrototypeOf(d) === D.prototype) + ',' + \
             d.length + ',' + d[0] + ',' + d[1]",
        );
        assert_eq!(r, "true,true,true,2,7,8");
    }

    #[test]
    fn typed_array_of_retargeted_this_picks_target_type() {
        // `of` is generic: the *called-on* constructor decides the type.
        let r = eval_str(
            "var a = Uint8Array.of.call(Int16Array, 258); \
             (a instanceof Int16Array) + ',' + a.length + ',' + a[0]",
        );
        assert_eq!(r, "true,1,258");
    }

    #[test]
    fn typed_array_of_rejects_non_constructor_this() {
        let r = eval_str(
            "var out; \
             try { Uint8Array.of.call({}, 1); out = 'no-throw'; } \
             catch (e) { out = (e instanceof TypeError) ? 'TypeError' : 'other'; } \
             out",
        );
        assert_eq!(r, "TypeError");
    }

    #[test]
    fn typed_array_from_array_like_and_typed_array_sources() {
        let r = eval_str(
            "var a = Uint16Array.from({0: 10, 1: 20, length: 2}); \
             var c = Float64Array.from(new Uint8Array(2)); \
             a.length + ',' + a[0] + ',' + a[1] + '|' + c.length + ',' + c[0] + ',' + c[1]",
        );
        assert_eq!(r, "2,10,20|2,0,0");
    }

    #[test]
    fn typed_array_from_applies_mapfn_with_this_arg() {
        // mapfn receives (value, index); `thisArg` is the call receiver.
        let r = eval_str(
            "var b = Int32Array.from([1, 2, 3], function(v, k) { return v * 10 + k + this.n; }, {n: 100}); \
             b.length + ',' + b[0] + ',' + b[1] + ',' + b[2]",
        );
        assert_eq!(r, "3,110,121,132");
    }

    #[test]
    fn typed_array_from_respects_subclass_this() {
        let r = eval_str(
            "class D extends Uint8Array {} \
             var d = D.from([5, 6]); \
             (d instanceof D) + ',' + d.length + ',' + d[0] + ',' + d[1]",
        );
        assert_eq!(r, "true,2,5,6");
    }

    #[test]
    fn typed_array_from_rejects_bad_source_and_mapfn() {
        let r = eval_str(
            "var out1, out2; \
             try { Uint8Array.from(null); out1 = 'no-throw'; } \
             catch (e) { out1 = (e instanceof TypeError) ? 'TypeError' : 'other'; } \
             try { Uint8Array.from([1], 42); out2 = 'no-throw'; } \
             catch (e) { out2 = (e instanceof TypeError) ? 'TypeError' : 'other'; } \
             try { Uint8Array.from.call(7, [1]); out2 += ',ctor'; } \
             catch (e) { out2 += ',' + ((e instanceof TypeError) ? 'TypeError' : 'other'); } \
             out1 + '|' + out2",
        );
        assert_eq!(r, "TypeError|TypeError,TypeError");
    }

    #[test]
    fn typed_array_from_roots_stored_value_across_gc_during_define() {
        // G2 regression: `typed_array_from` passed an unrooted `v` to
        // `set_constructed_elem`. The non-view branch defines a
        // property (interning / shape transitions), which can collect
        // and free the referent of a fresh mapfn return or a
        // materialized BigInt, leaving a dangling handle in the
        // constructed object. Pinning the GC threshold to 1 forces a
        // collection on nearly every allocation, including inside the
        // define path; the stored values must survive verbatim.
        let mut vm = crate::vm::Vm::new();
        vm.heap_mut().set_gc_threshold(1);
        let v = vm
            .eval(
                "function Evil() { return {}; } \
                 var bi = BigInt64Array.from.call(Evil, new BigInt64Array([1n, 2n]), function(v) { return v + 10n; }); \
                 var al = Uint8Array.from.call(Evil, {0: 7, 1: 8, length: 2}, function(v) { return v * 2; }); \
                 var it = Int32Array.from.call(Evil, [3, 4], function(v, k) { return v * 10 + k; }); \
                 bi[0] + ',' + bi[1] + '|' + al[0] + ',' + al[1] + '|' + it[0] + ',' + it[1]",
            )
            .unwrap_or_else(|e| panic!("eval failed: {e}"));
        let s = crate::builtin_object_array::to_string_js(vm.heap_mut(), v)
            .map(|w| w.to_string())
            .unwrap_or_else(|e| panic!("to_string failed: {e:?}"));
        assert_eq!(s, "11,12|14,16|30,41");
    }

    #[test]
    fn arraybuffer_transfer_copies_bytes() {
        // Byte contents verified through Uint8Array views (integer-indexed
        // reads).
        let r = eval_str(
            "var src = Uint8Array.of(1, 2, 3, 4); \
             var t = src.buffer.transfer(); \
             var tv = new Uint8Array(t); \
             (t instanceof ArrayBuffer) + ',' + t.byteLength + ',' + \
             tv[0] + ',' + tv[1] + ',' + tv[2] + ',' + tv[3] + ',' + \
             (t !== src.buffer)",
        );
        assert_eq!(r, "true,4,1,2,3,4,true");
    }

    #[test]
    fn arraybuffer_transfer_resizes_and_clamps() {
        // Shorter: truncates. Longer: zero-pads. The source is detached
        // by each transfer (spec), so the two transfers use fresh
        // buffers.
        let r = eval_str(
            "var s0 = Uint8Array.of(9, 8, 7, 6).buffer; \
             var t2 = s0.transfer(2); var sv = new Uint8Array(t2); \
             var g0 = Uint8Array.of(9, 8, 7, 6).buffer; \
             var g = g0.transfer(6); var gv = new Uint8Array(g); \
             t2.byteLength + ',' + sv[0] + ',' + sv[1] + '|' + \
             g.byteLength + ',' + gv[3] + ',' + gv[4] + ',' + gv[5] + '|' + \
             s0.byteLength + ',' + g0.byteLength",
        );
        assert_eq!(r, "2,9,8|6,6,0,0|0,0");
    }

    #[test]
    fn arraybuffer_transfer_rejects_bad_receiver() {
        let r = eval_str(
            "var out; \
             try { ArrayBuffer.prototype.transfer.call({}, 1); out = 'no-throw'; } \
             catch (e) { out = (e instanceof TypeError) ? 'TypeError' : 'other'; } \
             out",
        );
        assert_eq!(r, "TypeError");
    }

    #[test]
    fn arraybuffer_transfer_new_length_observable_conversion() {
        // Spec step 6 is a full `ToIntegerOrInfinity`: a `valueOf` hook
        // runs (test262 `transfer/new-length-non-number.js`); step 7
        // rejects out-of-range values with `RangeError`.
        let r = eval_str(&format!(
            "{throw_kind} \
             var hooked = {{ valueOf: function() {{ return 2; }} }}; \
             var xfer = new ArrayBuffer(8).transfer(hooked); \
             xfer.byteLength + ',' + \
             t(function() {{ new ArrayBuffer(8).transfer(-1); }}) + ',' + \
             t(function() {{ new ArrayBuffer(8).transfer(9007199254740992); }})",
            throw_kind = throw_kind()
        ));
        assert_eq!(r, "2,RangeError,RangeError");
    }

    /// Evaluate `source` on an existing `Vm`, returning the completion
    /// value stringified (like [`eval_str`]).
    fn eval_on(vm: &mut crate::vm::Vm, source: &str) -> String {
        let v = vm
            .eval(source)
            .unwrap_or_else(|e| panic!("eval failed: {e}"));
        crate::builtin_object_array::to_string_js(vm.heap_mut(), v)
            .map(|w| w.to_string())
            .unwrap_or_else(|e| panic!("to_string failed: {e:?}"))
    }

    /// Detach the `ArrayBuffer` held in the global `name` via the engine
    /// op (the unit-test stand-in for `$262.detachArrayBuffer`).
    fn detach_global(vm: &mut crate::vm::Vm, name: &str) {
        let buf = vm.get_global(name).unwrap();
        let Value::Object(bo) = buf else {
            panic!("{name} is not an object");
        };
        detach_array_buffer_op(vm.heap_mut(), bo).unwrap();
    }

    /// Classify a thrown error for the assertions below.
    fn throw_kind() -> &'static str {
        "function t(f) { \
           try { f(); return 'no-throw'; } \
           catch (e) { \
             return e instanceof RangeError ? 'RangeError' \
                  : e instanceof TypeError ? 'TypeError' : 'other'; \
           } \
         }"
    }

    #[test]
    fn detach_array_buffer_views_observe_detach() {
        let mut vm = crate::vm::Vm::new();
        vm.eval(
            "var buf = new ArrayBuffer(8); \
             var ta = new Uint8Array(buf); ta[0] = 42; ta[1] = 7; \
             var dv = new DataView(buf);",
        )
        .unwrap();
        detach_global(&mut vm, "buf");
        // The buffer's own `byteLength` snapshot drops to 0.
        assert_eq!(eval_on(&mut vm, "buf.byteLength"), "0");
        // Typed-array getters report 0 (never throw).
        assert_eq!(
            eval_on(
                &mut vm,
                "ta.length + ',' + ta.byteLength + ',' + ta.byteOffset"
            ),
            "0,0,0"
        );
        // Integer-indexed reads yield `undefined`; writes are ignored.
        assert_eq!(eval_on(&mut vm, "ta[0]"), "undefined");
        assert_eq!(eval_on(&mut vm, "ta[1] = 99; ta[1]"), "undefined");
        // DataView element accessors throw `TypeError`.
        assert_eq!(
            eval_on(
                &mut vm,
                &format!(
                    "{throw_kind} t(function() {{ dv.getUint8(0); }})",
                    throw_kind = throw_kind()
                )
            ),
            "TypeError"
        );
        assert_eq!(
            eval_on(
                &mut vm,
                &format!(
                    "{throw_kind} t(function() {{ dv.setUint8(0, 1); }})",
                    throw_kind = throw_kind()
                )
            ),
            "TypeError"
        );
        // DataView `byteLength` / `byteOffset` throw; `buffer` still
        // answers the (detached) buffer.
        assert_eq!(
            eval_on(
                &mut vm,
                &format!(
                    "{throw_kind} t(function() {{ return dv.byteLength; }}) + ',' + \
                     t(function() {{ return dv.byteOffset; }})",
                    throw_kind = throw_kind()
                )
            ),
            "TypeError,TypeError"
        );
        assert_eq!(eval_on(&mut vm, "dv.buffer === buf"), "true");
        // Constructing views over a detached buffer throws.
        assert_eq!(
            eval_on(
                &mut vm,
                &format!(
                    "{throw_kind} t(function() {{ new Uint8Array(buf); }}) + ',' + \
                     t(function() {{ new DataView(buf); }})",
                    throw_kind = throw_kind()
                )
            ),
            "TypeError,TypeError"
        );
        // `slice` / `transfer` on a detached buffer throw.
        assert_eq!(
            eval_on(
                &mut vm,
                &format!(
                    "{throw_kind} t(function() {{ buf.slice(); }}) + ',' + \
                     t(function() {{ buf.transfer(); }})",
                    throw_kind = throw_kind()
                )
            ),
            "TypeError,TypeError"
        );
        // Prototype methods validate (`ValidateTypedArray`).
        assert_eq!(
            eval_on(
                &mut vm,
                &format!(
                    "{throw_kind} t(function() {{ ta.slice(); }}) + ',' + \
                     t(function() {{ ta.map(function(x) {{ return x; }}); }}) + ',' + \
                     t(function() {{ ta.at(0); }}) + ',' + \
                     t(function() {{ ta.includes(42); }})",
                    throw_kind = throw_kind()
                )
            ),
            "TypeError,TypeError,TypeError,TypeError"
        );
        // Detaching twice is a no-op.
        detach_global(&mut vm, "buf");
        assert_eq!(eval_on(&mut vm, "buf.byteLength"), "0");
    }

    #[test]
    fn test262_detach_array_buffer_native_fn() {
        let mut vm = crate::vm::Vm::new();
        vm.eval("var buf = new ArrayBuffer(4);").unwrap();
        let buf = vm.get_global("buf").unwrap();
        // Non-`ArrayBuffer` (and missing) arguments throw `TypeError`.
        assert!(matches!(
            test262_detach_array_buffer(vm.heap_mut(), Value::Undefined, &[Value::Int(1)]),
            Err(NativeError::TypeError(_))
        ));
        assert!(matches!(
            test262_detach_array_buffer(vm.heap_mut(), Value::Undefined, &[]),
            Err(NativeError::TypeError(_))
        ));
        // Detach works, returns `undefined`, and is idempotent.
        assert_eq!(
            test262_detach_array_buffer(vm.heap_mut(), Value::Undefined, &[buf]).unwrap(),
            Value::Undefined
        );
        assert_eq!(eval_on(&mut vm, "buf.byteLength"), "0");
        test262_detach_array_buffer(vm.heap_mut(), Value::Undefined, &[buf]).unwrap();
        assert_eq!(eval_on(&mut vm, "buf.byteLength"), "0");
        // A `SharedArrayBuffer` can never detach: the helper is a no-op
        // and every view over the buffer stays intact.
        vm.eval("var sab = new SharedArrayBuffer(4); var sta = new Uint8Array(sab);")
            .unwrap();
        let sab = vm.get_global("sab").unwrap();
        assert_eq!(
            test262_detach_array_buffer(vm.heap_mut(), Value::Undefined, &[sab]).unwrap(),
            Value::Undefined
        );
        assert_eq!(eval_on(&mut vm, "sab.byteLength + ',' + sta.length"), "4,4");
    }

    #[test]
    fn install_test262_helpers_installs_262() {
        let mut vm = crate::vm::Vm::new();
        install_test262_helpers(&mut vm).unwrap();
        assert_eq!(
            eval_on(&mut vm, "typeof $262.detachArrayBuffer"),
            "function"
        );
        assert_eq!(
            eval_on(
                &mut vm,
                "var b = new ArrayBuffer(2); $262.detachArrayBuffer(b); b.byteLength"
            ),
            "0"
        );
        assert_eq!(
            eval_on(
                &mut vm,
                &format!(
                    "{throw_kind} t(function() {{ $262.detachArrayBuffer(1); }})",
                    throw_kind = throw_kind()
                )
            ),
            "TypeError"
        );
    }

    #[test]
    fn resizable_array_buffer_basics() {
        let r = eval_str(
            "var ab = new ArrayBuffer(4, { maxByteLength: 8 }); \
             ab.resizable + ',' + ab.maxByteLength + ',' + ab.byteLength + '|' + \
             new ArrayBuffer(4).resizable + ',' + new ArrayBuffer(4).maxByteLength + '|' + \
             new ArrayBuffer(0, { maxByteLength: 0 }).resizable",
        );
        assert_eq!(r, "true,8,4|false,0|true");
    }

    #[test]
    fn shared_array_buffer_guards_throw() {
        // Shared buffers can never detach and reject the
        // resizable-buffer protocol: `transfer` / `transferToFixedLength`
        // / `slice` / `resize` and the `resizable` / `maxByteLength` /
        // `detached` getters all throw `TypeError` on them.
        let r = eval_str(&format!(
            "{throw_kind} \
             var sab = new SharedArrayBuffer(4); \
             t(function() {{ sab.slice(0); }}) + ',' + \
             t(function() {{ sab.transfer(); }}) + ',' + \
             t(function() {{ sab.transferToFixedLength(); }}) + ',' + \
             t(function() {{ sab.resize(2); }}) + ',' + \
             t(function() {{ return sab.resizable; }}) + ',' + \
             t(function() {{ return sab.maxByteLength; }}) + ',' + \
             t(function() {{ return sab.detached; }})",
            throw_kind = throw_kind()
        ));
        assert_eq!(
            r,
            "TypeError,TypeError,TypeError,TypeError,TypeError,TypeError,TypeError"
        );
    }

    #[test]
    fn shared_array_buffer_views_still_work() {
        // The guards above reject the buffer *protocol*; views over a
        // shared buffer keep working, and `byteLength` still reads.
        let r = eval_str(
            "var sab = new SharedArrayBuffer(4); \
             var ta = new Uint8Array(sab); ta[0] = 7; \
             ta[0] + ',' + ta.length + ',' + sab.byteLength",
        );
        assert_eq!(r, "7,4,4");
    }

    #[test]
    fn arraybuffer_constructor_options_validation() {
        let r = eval_str(&format!(
            "{throw_kind} \
             t(function() {{ new ArrayBuffer(8, {{ maxByteLength: 4 }}); }}) + ',' + \
             t(function() {{ new ArrayBuffer(8, null); }}) + ',' + \
             t(function() {{ new ArrayBuffer(8, {{ maxByteLength: 'foo' }}); }}) + ',' + \
             t(function() {{ new ArrayBuffer(8, {{ maxByteLength: 4294967296 }}); }}) + ',' + \
             t(function() {{ \
               var poisoned = {{ get maxByteLength() {{ throw {{ custom: true }}; }} }}; \
               new ArrayBuffer(0, poisoned); \
             }})",
            throw_kind = throw_kind()
        ));
        assert_eq!(r, "RangeError,TypeError,RangeError,RangeError,other");
    }

    #[test]
    fn arraybuffer_constructor_options_observable_conversion() {
        // `maxByteLength` goes through `Get` + `ToIndex` with full
        // `ToPrimitive`: `valueOf`/`toString` hooks run in order, and a
        // non-primitive result throws `TypeError`.
        let r = eval_str(
            "var log = []; \
             var options = { maxByteLength: { \
               valueOf: function() { log.push('valueOf'); return {}; }, \
               toString: function() { log.push('toString'); return {}; } \
             } }; \
             var kind; \
             try { new ArrayBuffer(0, options); kind = 'no-throw'; } \
             catch (e) { kind = (e instanceof TypeError) ? 'TypeError' : 'other'; } \
             kind + ',' + log.join(',')",
        );
        assert_eq!(r, "TypeError,valueOf,toString");
    }

    #[test]
    fn arraybuffer_constructor_length_observable_conversion() {
        // The `length` argument goes through a full `ToIndex`: a
        // `valueOf` hook runs, so `new ArrayBuffer({ valueOf: () => 8 })`
        // yields 8 bytes, not 0.
        let r = eval_str("new ArrayBuffer({ valueOf: function() { return 8; } }).byteLength");
        assert_eq!(r, "8");
    }

    #[test]
    fn arraybuffer_resize_grows_zero_fills_and_shrinks() {
        let r = eval_str(
            "var ab = new ArrayBuffer(4, { maxByteLength: 8 }); \
             var ta = new Uint8Array(ab); ta[0] = 9; ta[3] = 7; \
             ab.resize(6); \
             var grown = ab.byteLength + ',' + ta.length + ',' + ta[0] + ',' + \
                         ta[3] + ',' + ta[4] + ',' + ta[5]; \
             ab.resize(2); \
             var shrunk = ab.byteLength + ',' + ta.length + ',' + ta[0] + ',' + ta[1]; \
             grown + '|' + shrunk",
        );
        assert_eq!(r, "6,6,9,7,0,0|2,2,9,0");
    }

    #[test]
    fn arraybuffer_resize_rejects() {
        let r = eval_str(&format!(
            "{throw_kind} \
             var ab = new ArrayBuffer(4, {{ maxByteLength: 8 }}); \
             var fixed = new ArrayBuffer(4); \
             var d = new ArrayBuffer(4, {{ maxByteLength: 8 }}); d.transfer(); \
             t(function() {{ ab.resize(9); }}) + ',' + \
             t(function() {{ fixed.resize(2); }}) + ',' + \
             t(function() {{ ab.resize(-1); }}) + ',' + \
             t(function() {{ d.resize(2); }})",
            throw_kind = throw_kind()
        ));
        assert_eq!(r, "RangeError,TypeError,RangeError,TypeError");
    }

    #[test]
    fn length_tracking_typed_array_getters() {
        let r = eval_str(
            "var ab = new ArrayBuffer(8, { maxByteLength: 16 }); \
             var ta = new Uint16Array(ab); \
             var fixed = new Uint16Array(ab, 0, 2); \
             var off = new Uint16Array(ab, 4, 2); \
             function lens() { \
               return ta.length + ',' + ta.byteLength + ',' + \
                      fixed.length + ',' + fixed.byteLength + ',' + off.byteOffset; \
             } \
             var a = lens(); \
             ab.resize(16); var b = lens(); \
             ab.resize(6); var c = lens(); \
             ab.resize(4); var d = lens(); \
             a + '|' + b + '|' + c + '|' + d",
        );
        assert_eq!(r, "4,8,2,4,4|8,16,2,4,4|3,6,2,4,0|2,4,2,4,0");
    }

    #[test]
    fn length_tracking_dataview_getters() {
        let r = eval_str(&format!(
            "{throw_kind} \
             var ab = new ArrayBuffer(8, {{ maxByteLength: 16 }}); \
             var dv = new DataView(ab); \
             var fixed = new DataView(ab, 2, 4); \
             var a = dv.byteLength + ',' + fixed.byteLength; \
             ab.resize(12); var b = dv.byteLength + ',' + fixed.byteLength; \
             ab.resize(5); \
             var c = dv.byteLength + ',' + t(function() {{ return fixed.byteLength; }}); \
             a + '|' + b + '|' + c",
            throw_kind = throw_kind()
        ));
        assert_eq!(r, "8,4|12,4|5,TypeError");
    }

    #[test]
    fn dataview_element_access_tracks_resize() {
        let r = eval_str(&format!(
            "{throw_kind} \
             var ab = new ArrayBuffer(8, {{ maxByteLength: 16 }}); \
             var dv = new DataView(ab); dv.setUint8(7, 171); \
             ab.resize(16); \
             var a = dv.getUint8(7) + ',' + dv.byteLength; \
             ab.resize(4); \
             var b = t(function() {{ return dv.getUint8(7); }}); \
             var c = dv.getUint8(3); \
             var ab2 = new ArrayBuffer(8, {{ maxByteLength: 16 }}); \
             var fdv = new DataView(ab2, 0, 8); \
             ab2.resize(4); \
             var d = t(function() {{ return fdv.getUint8(0); }}); \
             a + '|' + b + '|' + c + '|' + d",
            throw_kind = throw_kind()
        ));
        assert_eq!(r, "171,16|RangeError|0|TypeError");
    }

    #[test]
    fn transfer_preserves_resizable_and_detaches() {
        let r = eval_str(
            "var src = new ArrayBuffer(4, { maxByteLength: 8 }); \
             var t = src.transfer(6); \
             var a = t.resizable + ',' + t.maxByteLength + ',' + t.byteLength + ',' + \
                     src.byteLength; \
             var src2 = new ArrayBuffer(4, { maxByteLength: 8 }); \
             var f = src2.transferToFixedLength(2); \
             var b = f.resizable + ',' + f.maxByteLength + ',' + f.byteLength + ',' + \
                     src2.byteLength; \
             a + '|' + b",
        );
        assert_eq!(r, "true,8,6,0|false,0,2,0");
    }

    #[test]
    fn slice_yields_fixed_length_buffer() {
        // ECMA-262 25.1.6.7 step 16: the slice result is constructed with
        // a single length argument (no options bag), so it is always a
        // fixed-size buffer — a resizable source does NOT propagate its
        // ceiling.
        let r = eval_str(
            "var ab = new ArrayBuffer(8, { maxByteLength: 16 }); \
             var s = ab.slice(2, 6); \
             s.resizable + ',' + s.maxByteLength + ',' + s.byteLength",
        );
        assert_eq!(r, "false,0,4");
    }

    #[test]
    fn subarray_detached_throws_after_observable_conversion() {
        // Mirrors test262's `subarray/detached-buffer.js`: the
        // `begin`/`end` conversions run before the `TypeError`.
        let r = eval_str(
            "var sample = new Uint8Array(2); \
             var begin = false, end = false; \
             var o1 = { valueOf: function() { begin = true; return 0; } }; \
             var o2 = { valueOf: function() { end = true; return 1; } }; \
             sample.buffer.transfer(); \
             function t(f) { \
               try { f(); return 'no-throw'; } \
               catch (e) { return e instanceof TypeError ? 'TypeError' : 'other'; } \
             } \
             t(function() { sample.subarray(o1, o2); }) + ',' + begin + ',' + end",
        );
        assert_eq!(r, "TypeError,true,true");
    }

    #[test]
    fn typed_array_constructor_detached_after_length_conversion() {
        // `InitializeTypedArrayFromArrayBuffer` performs its single
        // detached check *after* both conversions: a `valueOf` hook on
        // `length` that detaches the buffer throws `TypeError`.
        let mut vm = crate::vm::Vm::new();
        install_test262_helpers(&mut vm).unwrap();
        vm.eval("var buf = new ArrayBuffer(8);").unwrap();
        let r = eval_on(
            &mut vm,
            "function t(f) { \
               try { f(); return 'no-throw'; } \
               catch (e) { return e instanceof TypeError ? 'TypeError' : 'other'; } \
             } \
             t(function() { \
               new Int8Array(buf, 0, { valueOf: function() { \
                 $262.detachArrayBuffer(buf); return 1; } }); \
             })",
        );
        assert_eq!(r, "TypeError");
    }

    #[test]
    fn typed_array_constructor_misaligned_offset_on_detached_throws_range_error() {
        // Spec order: the offset-alignment `RangeError` precedes the
        // detached check, so `new Int16Array(detachedBuf, 1)` throws
        // `RangeError`, not `TypeError`.
        let mut vm = crate::vm::Vm::new();
        vm.eval("var buf = new ArrayBuffer(8);").unwrap();
        detach_global(&mut vm, "buf");
        let r = eval_on(
            &mut vm,
            "function t(f) { \
               try { f(); return 'no-throw'; } \
               catch (e) { \
                 return e instanceof RangeError ? 'RangeError' \
                      : e instanceof TypeError ? 'TypeError' : 'other'; \
               } \
             } \
             t(function() { new Int16Array(buf, 1); })",
        );
        assert_eq!(r, "RangeError");
    }

    #[test]
    fn subarray_out_of_bounds_uses_current_length() {
        // Mirrors test262's `subarray/result-byteOffset-from-out-of-bounds.js`.
        let r = eval_str(
            "var rab = new ArrayBuffer(10, { maxByteLength: 10 }); \
             var autoLength = new Int8Array(rab, 4); \
             var start = { valueOf: function() { rab.resize(10); return 1; } }; \
             rab.resize(0); \
             var r = autoLength.subarray(start); \
             r.byteOffset + ',' + r.length",
        );
        assert_eq!(r, "4,6");
    }

    #[test]
    fn length_tracking_byte_offset_zero_when_out_of_bounds() {
        // Per `IsTypedArrayOutOfBounds`, a length-tracking view whose
        // offset lies past the buffer's current end is out of bounds:
        // `byteOffset` reports 0 (not the stale offset).
        let r = eval_str(
            "var rab = new ArrayBuffer(10, { maxByteLength: 10 }); \
             var ta = new Int8Array(rab, 4); \
             rab.resize(2); \
             ta.byteOffset + ',' + ta.length + ',' + ta.byteLength",
        );
        assert_eq!(r, "0,0,0");
    }

    #[test]
    fn subarray_tracking_omitted_end_shrunk_mid_call_throws() {
        // Length-tracking source, omitted `end`, and a `valueOf` hook
        // that shrinks the buffer past the computed begin: the species
        // construction's `TypedArray` constructor would throw `RangeError`
        // (`offset > bufferByteLength`), so `subarray` does too instead
        // of building a view with `byte_offset > buf_len`.
        let r = eval_str(&format!(
            "{throw_kind} \
             var rab = new ArrayBuffer(10, {{ maxByteLength: 10 }}); \
             var ta = new Int8Array(rab); \
             var shrink = {{ valueOf: function() {{ rab.resize(2); return 8; }} }}; \
             t(function() {{ ta.subarray(shrink); }})",
            throw_kind = throw_kind()
        ));
        assert_eq!(r, "RangeError");
    }

    #[test]
    fn typed_array_set_detached_source_throws() {
        // `%TypedArray%.prototype.set` validates the source view
        // (`ValidateTypedArray(src)`): a detached source throws
        // `TypeError` instead of silently filling the target with zeros.
        let mut vm = crate::vm::Vm::new();
        vm.eval(
            "var src = new Uint8Array(4); src[0] = 1; \
             var dst = new Uint8Array(4); \
             var srcBuf = src.buffer;",
        )
        .unwrap();
        detach_global(&mut vm, "srcBuf");
        let r = eval_on(
            &mut vm,
            "function t(f) { \
               try { f(); return 'no-throw'; } \
               catch (e) { return e instanceof TypeError ? 'TypeError' : 'other'; } \
             } \
             t(function() { dst.set(src); }) + ',' + dst[0]",
        );
        assert_eq!(r, "TypeError,0");
    }

    #[test]
    fn dataview_index_observable_conversion() {
        // `GetViewValue`/`SetViewValue` use a full `ToIndex`: a `valueOf`
        // hook on the index runs.
        let r = eval_str(
            "var dv = new DataView(new ArrayBuffer(4)); \
             dv.setUint8({ valueOf: function() { return 1; } }, 42); \
             dv.getUint8({ valueOf: function() { return 1; } })",
        );
        assert_eq!(r, "42");
    }

    #[test]
    fn arraybuffer_detached_getter() {
        let r = eval_str(
            "var ab = new ArrayBuffer(1); \
             var before = ab.detached; \
             ab.transfer(); \
             before + ',' + ab.detached",
        );
        assert_eq!(r, "false,true");
    }

    #[test]
    fn arraybuffer_resizable_survives_detach() {
        // `resizable` reports the `[[ArrayBufferMaxByteLength]]` slot,
        // which detach does not clear; `maxByteLength` reports 0.
        let r = eval_str(
            "var fixed = new ArrayBuffer(1); fixed.transfer(); \
             var rab = new ArrayBuffer(1, { maxByteLength: 1 }); rab.transfer(); \
             fixed.resizable + ',' + rab.resizable + ',' + rab.maxByteLength",
        );
        assert_eq!(r, "false,true,0");
    }

    #[test]
    fn arraybuffer_resize_coerces_before_detach_check() {
        // Mirrors test262's `resize/coerced-new-length-detach.js`: one
        // detach check *after* argument coercion.
        let r = eval_str(&format!(
            "{throw_kind} \
             var rab = new ArrayBuffer(64, {{ maxByteLength: 1024 }}); \
             var called1 = false; \
             var r1 = t(function() {{ \
               rab.resize({{ valueOf: function() {{ \
                 rab.transfer(); called1 = true; return 8; \
               }} }}); \
             }}); \
             var rab2 = new ArrayBuffer(64, {{ maxByteLength: 1024 }}); \
             rab2.transfer(); \
             var called2 = false; \
             var r2 = t(function() {{ \
               rab2.resize({{ valueOf: function() {{ called2 = true; return 8; }} }}); \
             }}); \
             r1 + ',' + called1 + '|' + r2 + ',' + called2",
            throw_kind = throw_kind()
        ));
        assert_eq!(r, "TypeError,true|TypeError,true");
    }

    #[test]
    fn dataview_constructor_coerces_before_detach_check() {
        // `ToNumber(byteOffset)` runs before the detached-buffer throw.
        let r = eval_str(
            "var toNumberOffset = 0; \
             var obj = { valueOf: function() { toNumberOffset += 1; return 0; } }; \
             var ab = new ArrayBuffer(42); \
             ab.transfer(); \
             var kind; \
             try { new DataView(ab, obj); kind = 'no-throw'; } \
             catch (e) { kind = (e instanceof TypeError) ? 'TypeError' : 'other'; } \
             kind + ',' + toNumberOffset",
        );
        assert_eq!(r, "TypeError,1");
    }
}
