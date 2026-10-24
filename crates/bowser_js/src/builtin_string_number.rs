//! `String`, `Number`, `BigInt`, `Math`, and `JSON` builtins.
//!
//! # String encoding model
//!
//! JS strings are sequences of UTF-16 code units; this engine stores them
//! as Rust UTF-8 (`GcStr`). Every method that is specified in terms of
//! code units converts at the method boundary (`s.encode_utf16()`), works
//! on the `Vec<u16>`, and converts back. The mapping is documented per
//! method; the O(n) conversion cost is a v1 simplification, not the goal.
//!
//! # Lone surrogates
//!
//! A Rust `String` cannot hold an unpaired surrogate, but JS strings can
//! (`"𝌆".charAt(0)` is `"\uD834"`). v1 policy: whenever a lone surrogate
//! would be produced (`charAt`, `fromCharCode`, ...), it is replaced with
//! U+FFFD. This is spec-divergent and documented here instead of hidden.
//!
//! # Deliberate v1 deviations
//!
//! * `String.prototype.normalize` throws `TypeError("not implemented")`.
//! * `String.prototype.localeCompare` is a raw UTF-16 code-unit comparison.
//! * `String.prototype[Symbol.iterator]` is absent (needs the iterator
//!   protocol from the interpreter worker).
//! * `String.prototype.match` / `matchAll` / `search` follow the spec's
//!   well-known-symbol dispatch: `GetMethod(pattern, Symbol.match)` (etc.) is
//!   invoked via [`call_function`](crate::builtin_object_array::call_function)
//!   when callable, covering both `RegExp` patterns (through
//!   `RegExp.prototype`) and custom hook objects; otherwise the pattern is
//!   coerced with `RegExpCreate` (so `"ab".match("b")` works). v1 gaps:
//!   `matchAll` returns an `Array` of match arrays instead of a
//!   `RegExpStringIterator` (no iterator protocol yet), and primitives are
//!   not boxed for the `GetMethod` lookup.
//! * `String.prototype.replace` / `replaceAll` with a `RegExp` search value
//!   and a *callable* replacement throw `TypeError`: per-match invocation
//!   needs match data the regexp worker does not expose yet.
//! * `String.prototype.includes` / `startsWith` / `endsWith` throw
//!   `TypeError` when the search value is a `RegExp`, per spec.
//! * Constructors (`String()`, `Number()`, `BigInt()`) implement the *call*
//!   form only. The interpreter does not pass `new.target` to [`NativeFn`]
//!   yet, so `new String("x")` currently returns the primitive; wrapper
//!   objects arrive with `new.target` support.
//! * `Number.prototype.toString(radix)` for radix != 10 uses the exact
//!   (terminating or 1100-digit-capped) expansion rather than the spec's
//!   shortest-rounding algorithm; outputs agree for terminating expansions.
//! * `JSON.parse` maps lone surrogates in `\u` escapes to U+FFFD (same
//!   UTF-8 limitation as above) and caps nesting at 1000.
//! * `JSON.stringify` property order follows the heap's hash order, not
//!   insertion order (see [`Object`][crate::value::Object] docs); `toJSON`
//!   and replacer *functions* work only when they are native functions;
//!   array serialization caps at 2^24 elements.
//! * `Math.random` is xorshift64* seeded from the system clock: not
//!   cryptographic, not seedable.
//! * `BigInt` division/modulo and shifts are O(n^2); a Knuth-D division
//!   replaces the binary long division when profiling says so.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrd};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::builtin_object_array::{
    PrimitiveHint, call_function, is_callable, to_primitive_with_hint, to_string_js_str,
    walk_proto_chain,
};
use crate::builtin_reflect_proxy::new_native_constructor;
use crate::gc::{Gc, Heap, PropKey, WellKnown};
use crate::value::{BigInt, NativeError, NativeFn, Object, ObjectKind, Property, Value};
use crate::wtf8::{Wtf8, Wtf8Buf};

// ---------------------------------------------------------------------------
// Shared helpers (pub(crate) so sibling builtin workers can reuse them).
// ---------------------------------------------------------------------------

/// Hidden slot holding a primitive wrapper's `[[PrimitiveValue]]`.
///
/// Sloppy-mode `this` boxing (`Interp::coerce_this`) wraps a primitive
/// receiver in an ordinary object carrying the matching prototype; the
/// wrapped primitive itself lives here (non-enumerable, like the `Date`
/// time-value slot) so `valueOf` and friends can unwrap it.
pub(crate) const PRIM_VALUE: &str = "\u{0}bowser:prim-value";

/// Read the hidden primitive slot of a wrapper object. `None` when the
/// object is not a primitive wrapper.
pub(crate) fn prim_slot(heap: &Heap, obj: Gc<Object>) -> Option<Value> {
    heap.get_property(obj, Wtf8::from_str(PRIM_VALUE))
        .map(|p| p.value)
}

/// `args[i]` or `undefined`.
pub(crate) fn arg(args: &[Value], i: usize) -> Value {
    args.get(i).copied().unwrap_or(Value::Undefined)
}

/// Wrap an f64 as the cheapest numeric value: [`Value::Int`] when it is an
/// integral value that fits `i32` (preserving `-0` as `Float`), else
/// [`Value::Float`].
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

/// The `StrWhiteSpace` code points (spec 12.2), shared by `trim`,
/// `parseInt`, `parseFloat`, `Number()`, `BigInt()`, and the JSON parser.
pub(crate) fn is_js_whitespace(c: char) -> bool {
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

/// Strip JS whitespace from the start.
///
/// ASCII fast path: the leading ASCII-whitespace run is scanned as bytes;
/// only a non-ASCII edge falls back to the char check.
pub(crate) fn trim_js_start(s: &str) -> &str {
    let b = s.as_bytes();
    let n = b.len();
    let mut start = 0;
    while start < n {
        let c = b[start];
        if c == b' ' || (0x09..=0x0D).contains(&c) {
            start += 1;
        } else if c < 0x80 {
            break;
        } else {
            match s[start..].chars().next() {
                Some(ch) if is_js_whitespace(ch) => start += ch.len_utf8(),
                _ => break,
            }
        }
    }
    &s[start..]
}

/// Strip JS whitespace from the end.
///
/// ASCII fast path: the trailing ASCII-whitespace run is scanned as
/// bytes; only a non-ASCII edge falls back to the char check.
pub(crate) fn trim_js_end(s: &str) -> &str {
    let b = s.as_bytes();
    let n = b.len();
    let mut end = n;
    while end > 0 {
        let c = b[end - 1];
        if c == b' ' || (0x09..=0x0D).contains(&c) {
            end -= 1;
        } else if c < 0x80 {
            break;
        } else {
            // Step back over continuation bytes to the char start
            // (`0` is always a char boundary, so this terminates).
            let mut cs = end - 1;
            while cs > 0 && b[cs] & 0xC0 == 0x80 {
                cs -= 1;
            }
            match s[cs..].chars().next() {
                Some(ch) if is_js_whitespace(ch) => end = cs,
                _ => break,
            }
        }
    }
    &s[..end]
}

/// Strip JS whitespace from both ends.
pub(crate) fn trim_js(s: &str) -> &str {
    trim_js_end(trim_js_start(s))
}

/// Strip JS whitespace from the start of WTF-8 (lone surrogates are never
/// whitespace, so they always stop the scan).
pub(crate) fn trim_js_start_wtf8(s: &Wtf8) -> &Wtf8 {
    let b = s.as_bytes();
    let n = b.len();
    let mut start = 0;
    while start < n {
        let c = b[start];
        if c == b' ' || (0x09..=0x0D).contains(&c) {
            start += 1;
        } else if c < 0x80 {
            break;
        } else {
            match s.code_point_at(start) {
                Some((cp, len)) if char::from_u32(cp).is_some_and(is_js_whitespace) => {
                    start += len;
                }
                _ => break,
            }
        }
    }
    // SAFETY: `start` advanced only over ASCII bytes or whole WTF-8
    // sequences, so it is always a valid WTF-8 boundary.
    unsafe { Wtf8::from_bytes_unchecked(&b[start..]) }
}

/// Strip JS whitespace from the end of WTF-8.
pub(crate) fn trim_js_end_wtf8(s: &Wtf8) -> &Wtf8 {
    let b = s.as_bytes();
    let n = b.len();
    let mut end = n;
    while end > 0 {
        let c = b[end - 1];
        if c == b' ' || (0x09..=0x0D).contains(&c) {
            end -= 1;
        } else if c < 0x80 {
            break;
        } else {
            // Step back over continuation bytes to the sequence start
            // (`0` is always a boundary, so this terminates).
            let mut cs = end - 1;
            while cs > 0 && b[cs] & 0xC0 == 0x80 {
                cs -= 1;
            }
            match s.code_point_at(cs) {
                Some((cp, _)) if char::from_u32(cp).is_some_and(is_js_whitespace) => {
                    end = cs;
                }
                _ => break,
            }
        }
    }
    // SAFETY: `end` retreated only over ASCII bytes or whole WTF-8
    // sequences, so it is always a valid WTF-8 boundary.
    unsafe { Wtf8::from_bytes_unchecked(&b[..end]) }
}

/// Strip JS whitespace from both ends of WTF-8.
pub(crate) fn trim_js_wtf8(s: &Wtf8) -> &Wtf8 {
    trim_js_end_wtf8(trim_js_start_wtf8(s))
}

/// JS `ToNumber`. Throws on `Symbol`; objects go through a v1
/// `ToPrimitive` (no user-defined `valueOf`/`toString` yet).
pub(crate) fn js_to_number(v: Value) -> Result<f64, NativeError> {
    match v {
        Value::Undefined => Ok(f64::NAN),
        Value::Null => Ok(0.0),
        Value::Bool(b) => Ok(if b { 1.0 } else { 0.0 }),
        Value::Int(i) => Ok(i as f64),
        Value::Float(f) => Ok(f),
        Value::Str(g) => Ok(string_to_number(g.as_wtf8())),
        // Spec 7.1.4: `ToNumber` on a BigInt throws (this is why
        // `Math.abs(1n)` and `+1n` throw); only `Number()` converts, via
        // its explicit arm in `number_construct` below.
        Value::BigInt(_) => Err(NativeError::TypeError(
            "Cannot convert a BigInt value to a number".to_string(),
        )),
        Value::Symbol(_) => Err(NativeError::TypeError(
            "Cannot convert a Symbol value to a number".to_string(),
        )),
        Value::Object(o) => Ok(string_to_number(&js_to_string_rust(Value::Object(o))?)),
    }
}

/// JS `ToIntegerOrInfinity`: `NaN` -> `+0`, infinities pass through.
pub(crate) fn js_to_integer_or_infinity(v: Value) -> Result<f64, NativeError> {
    let n = js_to_number(v)?;
    if n.is_nan() {
        Ok(0.0)
    } else if n.is_infinite() {
        Ok(n)
    } else {
        Ok(n.trunc())
    }
}

/// JS `ToLength`: integer clamped to `[0, 2^53 - 1]`.
pub(crate) fn js_to_length(v: Value) -> Result<u64, NativeError> {
    let n = js_to_integer_or_infinity(v)?;
    if n <= 0.0 {
        Ok(0)
    } else if n >= 9007199254740991.0 {
        Ok(9007199254740991)
    } else {
        Ok(n as u64)
    }
}

/// JS `ToUint32` (modulo 2^32, infinities/NaN -> 0).
pub(crate) fn js_to_uint32(v: Value) -> Result<u32, NativeError> {
    let n = js_to_number(v)?;
    if !n.is_finite() {
        return Ok(0);
    }
    let m = n % 4294967296.0;
    let m = if m < 0.0 { m + 4294967296.0 } else { m };
    Ok(m as u32)
}

/// JS `ToInt32` via `ToUint32` reinterpretation.
pub(crate) fn js_to_int32(v: Value) -> Result<i32, NativeError> {
    let u = js_to_uint32(v)?;
    Ok(if u >= 0x8000_0000 {
        (u as i64 - 0x1_0000_0000) as i32
    } else {
        u as i32
    })
}

/// JS `ToString` as an owned Rust string. Objects: wrappers unwrap, arrays
/// join with `,`, everything else is `"[object Object]"` (v1: no
/// user-defined `toString`/`valueOf` yet).
pub(crate) fn js_to_string_rust(v: Value) -> Result<Wtf8Buf, NativeError> {
    js_to_string_depth(v, 0)
}

fn js_to_string_depth(v: Value, depth: usize) -> Result<Wtf8Buf, NativeError> {
    if depth > 64 {
        return Err(NativeError::TypeError(
            "ToString: too much nesting".to_string(),
        ));
    }
    match v {
        Value::Undefined => Ok(Wtf8Buf::from("undefined")),
        Value::Null => Ok(Wtf8Buf::from("null")),
        Value::Bool(b) => Ok(Wtf8Buf::from(if b { "true" } else { "false" })),
        Value::Int(i) => Ok(Wtf8Buf::from(&i.to_string())),
        Value::Float(f) => Ok(Wtf8Buf::from(&js_number_to_string(f))),
        Value::Str(g) => Ok(g.as_wtf8().to_owned()),
        Value::BigInt(g) => Ok(Wtf8Buf::from(&bn_to_string_radix(&g, 10))),
        Value::Symbol(_) => Err(NativeError::TypeError(
            "Cannot convert a Symbol value to a string".to_string(),
        )),
        Value::Object(o) => {
            if matches!(o.kind, ObjectKind::Array) {
                let len = array_length(o);
                let mut out = Wtf8Buf::new();
                for i in 0..len {
                    if i > 0 {
                        out.push_char(',');
                    }
                    match prop_get(o, Wtf8::from_str(&i.to_string())) {
                        None | Some(Value::Undefined) | Some(Value::Null) => {}
                        Some(e) => out.push_wtf8(js_to_string_depth(e, depth + 1)?.as_wtf8()),
                    }
                }
                Ok(out)
            } else {
                Ok(Wtf8Buf::from("[object Object]"))
            }
        }
    }
}

/// JS `ToString` as a heap string.
pub(crate) fn js_to_string_val(heap: &mut Heap, v: Value) -> Result<Value, NativeError> {
    if let Value::Str(g) = v {
        return Ok(Value::Str(g));
    }
    match v {
        // Arrays keep the v1 comma-join (`Array.prototype.toString` is not
        // implemented); everything else goes through `to_string_js_str` and
        // the returned `Str` handle is handed straight to the caller — no
        // `Wtf8Buf` temporary, no second copy.
        //
        // # Rooting
        //
        // `to_string_js_str` may hand back an unrooted `Str::Heap` (a
        // `toString` result whose temporary roots `call_function` already
        // released, or a fresh `alloc_str` handle), but no heap allocation
        // runs between that return and this one, so no collection can free
        // it in between. The caller receives the same unrooted-fresh
        // posture the old `alloc_wtf8_owned` copy had. Short results
        // (numbers, `"true"`, ...) pack inline via `alloc_str`: no heap
        // allocation at all.
        Value::Object(o) if matches!(o.kind, ObjectKind::Array) => {
            let s = js_to_string_rust(v)?;
            Ok(Value::Str(heap.alloc_wtf8_owned(s)?))
        }
        _ => Ok(Value::Str(to_string_js_str(heap, v)?)),
    }
}

/// Own-property lookup by name (no prototype walk).
pub(crate) fn prop_get(o: Gc<Object>, name: &Wtf8) -> Option<Value> {
    o.properties
        .iter()
        .find(|(k, _)| k.as_wtf8() == name)
        .map(|(_, p)| p.value)
}

/// The `"length"` of an array-like object, sanitized.
pub(crate) fn array_length(o: Gc<Object>) -> u64 {
    match prop_get(o, Wtf8::from_str("length")) {
        Some(Value::Int(n)) if n >= 0 => n as u64,
        Some(Value::Float(f)) if f.is_finite() && f >= 0.0 => f.min(9007199254740991.0) as u64,
        _ => 0,
    }
}

/// Build a JS `Array` (indexed properties + `length`) from elements.
///
/// Elements are pushed through dense storage and materialized once, so
/// batch construction is O(n) instead of O(n²) (no per-element string
/// key, interning, or linear property scan).
pub(crate) fn new_array(heap: &mut Heap, elems: Vec<Value>) -> Result<Value, NativeError> {
    // The elements live only in this untraced vec; root them across the
    // allocation below, whose collection would otherwise free them before
    // they are stored on the array. (`dense_push` performs no GC-heap
    // allocation, so the guard is only needed for this first allocation.)
    let mut base = None::<usize>;
    for &e in &elems {
        base.get_or_insert(heap.protect(e));
    }
    let arr = heap.new_dense_array()?;
    if let Some(b) = base {
        heap.release_temp(b);
    }
    for e in elems {
        heap.dense_push(arr, e)?;
    }
    // Materialize before the array becomes user-visible: direct readers
    // of `properties` (enumeration, cloning, `JSON.stringify`, …) must
    // see ordinary index properties.
    heap.materialize_dense(arr)?;
    Ok(Value::Object(arr))
}

/// Allocate a native function object with `name`/`length` installed.
///
/// The function's [[Prototype]] is `%Function.prototype%` (spec 10.2.4):
/// install-time builtins created before the Function intrinsics exist get a
/// null prototype here and are repaired by
/// [`Heap::fixup_native_function_prototypes`][crate::gc::Heap::fixup_native_function_prototypes]
/// once installation completes.
pub(crate) fn new_native_function(
    heap: &mut Heap,
    name: &str,
    func: NativeFn,
    length: u32,
) -> Result<Gc<Object>, NativeError> {
    let proto = heap
        .function_prototype
        .map(Value::Object)
        .unwrap_or(Value::Null);
    let f = heap.alloc_object(Object {
        kind: ObjectKind::NativeFunction(func),
        prototype: proto,
        ..Default::default()
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
    let name_val = Value::Str({
        // `f` is not reachable from any root yet; hold it across the
        // string allocation, which may trigger a collection.
        let base = heap.protect(Value::Object(f));
        let name_res = heap.alloc_str(name);
        heap.release_temp(base);
        name_res?
    });
    heap.define_property_str(
        f,
        "name",
        Property {
            value: name_val,
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )?;
    Ok(f)
}

/// Define `target[name] = <native function>` as a builtin method:
/// writable, non-enumerable, configurable (spec 17).
pub(crate) fn define_native(
    heap: &mut Heap,
    target: Gc<Object>,
    name: &str,
    func: NativeFn,
    length: u32,
) -> Result<(), NativeError> {
    // Hold `target` across the function-object allocation below, which may
    // trigger a collection while the target is not yet otherwise rooted.
    let base = heap.protect(Value::Object(target));
    let f = new_native_function(heap, name, func, length);
    heap.release_temp(base);
    let f = f?;
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

/// [`define_native`] generalized over the key kind: installs a native
/// function under a [`PropKey`], for well-known-symbol methods
/// (`Map.prototype[Symbol.iterator]`, `RegExp.prototype[Symbol.replace]`,
/// …). `name` is the function's `name` property
/// (`"[Symbol.iterator]"`, …).
pub(crate) fn define_native_key(
    heap: &mut Heap,
    target: Gc<Object>,
    key: PropKey<'_>,
    name: &str,
    func: NativeFn,
    length: u32,
) -> Result<(), NativeError> {
    // Hold `target` — and the symbol key, when it is not otherwise
    // rooted — across the function-object allocation below, which may
    // trigger a collection while the install is still wiring things up.
    let base = heap.protect(Value::Object(target));
    let key_base = match key {
        PropKey::Symbol(s) => Some(heap.protect(Value::Symbol(s))),
        PropKey::Str(_) => None,
    };
    let r = new_native_function(heap, name, func, length);
    heap.release_temp(base);
    if let Some(kb) = key_base {
        heap.release_temp(kb);
    }
    let f = r?;
    heap.define_property_key(
        target,
        key,
        Property {
            value: Value::Object(f),
            writable: true,
            enumerable: false,
            configurable: true,
        },
    )?;
    Ok(())
}

/// Run `f` with `root` held as a temporary GC root, releasing it on both
/// the success and error paths.
///
/// Fresh allocations (prototypes, constructors, intrinsic tables) are not
/// reachable from any traced root until they are linked into the object
/// graph; without this, a threshold-triggered collection mid-install would
/// free them while the install is still wiring them up.
pub(crate) fn with_temp_root<T>(
    heap: &mut Heap,
    root: Value,
    f: impl FnOnce(&mut Heap) -> Result<T, NativeError>,
) -> Result<T, NativeError> {
    let base = heap.protect(root);
    let out = f(heap);
    heap.release_temp(base);
    out
}

/// Define a non-writable, non-enumerable, non-configurable constant.
pub(crate) fn define_const(
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

/// Allocate a heap string and wrap it as a value.
fn str_val(heap: &mut Heap, s: &Wtf8) -> Result<Value, NativeError> {
    Ok(Value::Str(heap.alloc_str_wtf8(s)?))
}

/// If `v` is a callable object (native or bytecode), return it for invocation
/// through the shared [`crate::builtin_object_array::call_function`] choke
/// point (bytecode callees re-enter the VM via `vm::reenter_call`).
fn callable_value(v: Value) -> Option<Value> {
    match v {
        Value::Object(o) => match crate::gc::object_kind(o) {
            ObjectKind::NativeFunction(_) | ObjectKind::Function(_) => Some(v),
            _ => None,
        },
        _ => None,
    }
}

/// Spec `String.prototype.replace` / `replaceAll` with a callable
/// replacement and a RegExp search value: delegate to the `@@replace`
/// The [`NativeFn`] behind a native-function object, if `o` is one.
///
/// (Reads the object header directly; no heap borrow needed.)
fn native_fn_of(o: Gc<Object>) -> Option<NativeFn> {
    match crate::gc::object_kind(o) {
        ObjectKind::NativeFunction(f) => Some(f),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Number <-> string conversions.
// ---------------------------------------------------------------------------

/// Parse a string per `StringNumericLiteral` (spec 7.1.4.1): surrounding
/// whitespace, optional sign, then `Infinity`, a hex/binary/octal integer,
/// or a strict decimal literal. Anything else is `NaN`; empty is `0`.
/// JS `StringNumericLiteral` to `f64`. Operates on WTF-8 bytes: the grammar
/// is ASCII-only, and any lone surrogate in the input fails the strict
/// decimal check, yielding `NaN` naturally.
pub(crate) fn string_to_number(s: &Wtf8) -> f64 {
    let t = trim_js_wtf8(s);
    if t.as_bytes().is_empty() {
        return 0.0;
    }
    // Sign bytes are ASCII; WTF-8 never encodes ASCII as multibyte, so
    // stripping one byte cannot split a sequence.
    let (negative, signed, rest) = if let Some(r) = t.strip_ascii_prefix(b"-") {
        (true, true, r)
    } else if let Some(r) = t.strip_ascii_prefix(b"+") {
        (false, true, r)
    } else {
        (false, false, t)
    };
    if rest == Wtf8::from_str("Infinity") {
        return if negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    // The `StringNumericLiteral` grammar gives hex/binary/octal no sign
    // slot: a sign is only valid before `Infinity` or a decimal literal,
    // so `+0x1` and `-0b11` are `NaN`.
    if !signed {
        if let Some(hex) = rest
            .strip_ascii_prefix(b"0x")
            .or_else(|| rest.strip_ascii_prefix(b"0X"))
        {
            return prefixed_int(hex, 16, negative);
        }
        if let Some(bin) = rest
            .strip_ascii_prefix(b"0b")
            .or_else(|| rest.strip_ascii_prefix(b"0B"))
        {
            return prefixed_int(bin, 2, negative);
        }
        if let Some(oct) = rest
            .strip_ascii_prefix(b"0o")
            .or_else(|| rest.strip_ascii_prefix(b"0O"))
        {
            return prefixed_int(oct, 8, negative);
        }
    }
    if !is_strict_decimal(rest) {
        return f64::NAN;
    }
    // `is_strict_decimal` passed, so the input is ASCII-only: valid UTF-8.
    let mut v: f64 = rest.as_utf8().unwrap_or("").parse().unwrap_or(f64::NAN);
    if negative {
        v = -v;
    }
    v
}

fn prefixed_int(digits: &Wtf8, radix: u32, negative: bool) -> f64 {
    if digits.as_bytes().is_empty() {
        return f64::NAN;
    }
    let mut v = 0.0f64;
    let r = f64::from(radix);
    // `char::to_digit` only accepts ASCII digits, so scanning bytes is
    // exactly equivalent and skips UTF-8 decoding.
    for &b in digits.as_bytes() {
        match (b as char).to_digit(radix) {
            Some(d) => v = v * r + f64::from(d),
            None => return f64::NAN,
        }
    }
    if negative { -v } else { v }
}

/// Strict decimal literal covering the whole input (no partial parses).
/// Byte-level: the grammar is ASCII-only, so WTF-8 needs no decoding.
fn is_strict_decimal(s: &Wtf8) -> bool {
    let b = s.as_bytes();
    let mut i = 0;
    let n = b.len();
    let int_start = i;
    while i < n && b[i].is_ascii_digit() {
        i += 1;
    }
    let has_int = i > int_start;
    let mut has_frac = false;
    if i < n && b[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < n && b[i].is_ascii_digit() {
            i += 1;
        }
        has_frac = i > frac_start;
    }
    if !has_int && !has_frac {
        return false;
    }
    if i < n && (b[i] == b'e' || b[i] == b'E') {
        i += 1;
        if i < n && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < n && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_start {
            return false;
        }
    }
    i == n
}

/// Shortest-round-trip decimal digits of `|x|` (finite, nonzero) plus the
/// exponent `n` with `value = 0.d1..dk * 10^n` and `d1 != '0'`.
///
/// Relies on Rust's `Display` for `f64`, which is shortest-round-trip and
/// never uses exponent notation.
pub(crate) fn decimal_digits(x: f64) -> (String, i32) {
    debug_assert!(x.is_finite() && x != 0.0);
    let plain = format!("{}", x.abs());
    let mut digits = String::with_capacity(plain.len());
    let mut n: i32;
    if let Some(dot) = plain.find('.') {
        let intpart = &plain[..dot];
        let stripped = intpart.trim_start_matches('0');
        n = stripped.len() as i32;
        digits.push_str(stripped);
        digits.push_str(&plain[dot + 1..]);
    } else {
        n = plain.len() as i32;
        digits.push_str(plain.trim_start_matches('0'));
    }
    // Normalize leading zeros (e.g. "0.0001" -> digits "00001", n 0).
    let leading = digits.len() - digits.trim_start_matches('0').len();
    n -= leading as i32;
    digits = digits.trim_start_matches('0').to_string();
    // Shortest reprs never carry trailing zeros; strip defensively (`n`
    // is unaffected by trailing zeros).
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    if digits.is_empty() {
        digits.push('0');
    }
    (digits, n)
}

/// Spec `Number::toString(x)` for radix 10.
pub(crate) fn js_number_to_string(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x == 0.0 {
        return "0".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        };
    }
    let (digits, n) = decimal_digits(x);
    let k = digits.len() as i32;
    // Each branch builds the body in one allocation; the old code used
    // `format!` over a `"0".repeat(..)` temporary (two allocations).
    let body = if k <= n && n <= 21 {
        let mut s = String::with_capacity(digits.len() + (n - k) as usize);
        s.push_str(&digits);
        s.extend(std::iter::repeat_n('0', (n - k) as usize));
        s
    } else if 0 < n && n <= 21 {
        let mut s = String::with_capacity(digits.len() + 1);
        s.push_str(&digits[..n as usize]);
        s.push('.');
        s.push_str(&digits[n as usize..]);
        s
    } else if -6 < n && n <= 0 {
        let mut s = String::with_capacity(2 + (-n) as usize + digits.len());
        s.push_str("0.");
        s.extend(std::iter::repeat_n('0', (-n) as usize));
        s.push_str(&digits);
        s
    } else {
        let exp = n - 1;
        let mantissa = if k == 1 {
            digits
        } else {
            let mut m = String::with_capacity(digits.len() + 1);
            m.push_str(&digits[..1]);
            m.push('.');
            m.push_str(&digits[1..]);
            m
        };
        format!("{mantissa}e{exp:+}")
    };
    if x < 0.0 {
        let mut s = String::with_capacity(body.len() + 1);
        s.push('-');
        s.push_str(&body);
        s
    } else {
        body
    }
}

/// Exact binary decomposition: `|x| = m * 2^e`, `m` a u64 including the
/// hidden mantissa bit. Used by the exact `toFixed`/`toExponential`.
fn decompose(x: f64) -> (u64, i32) {
    debug_assert!(x.is_finite() && x != 0.0);
    let bits = x.abs().to_bits();
    let raw_exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & 0x000f_ffff_ffff_ffff;
    if raw_exp == 0 {
        (frac, -1074)
    } else {
        (frac | 0x0010_0000_0000_0000, raw_exp - 1075)
    }
}

// ---------------------------------------------------------------------------
// BigInt core: sign-magnitude, base 2^32 little-endian.
// `BigInt { negative, digits }` is normalized by construction: no trailing
// zero words, and zero is always `negative == false`.
// ---------------------------------------------------------------------------

fn bn_is_zero(d: &[u32]) -> bool {
    d.iter().all(|&w| w == 0)
}

fn eff_len(d: &[u32]) -> usize {
    let mut n = d.len();
    while n > 0 && d[n - 1] == 0 {
        n -= 1;
    }
    n
}

fn bn_normalize(negative: bool, mut d: Vec<u32>) -> BigInt {
    while d.last() == Some(&0) {
        d.pop();
    }
    BigInt {
        negative: negative && !d.is_empty(),
        digits: d,
    }
}

fn bn_zero() -> BigInt {
    BigInt {
        negative: false,
        digits: Vec::new(),
    }
}

fn bn_one() -> BigInt {
    BigInt {
        negative: false,
        digits: vec![1],
    }
}

fn bn_from_i64(v: i64) -> BigInt {
    if v == 0 {
        return bn_zero();
    }
    let m = v.unsigned_abs();
    bn_normalize(v < 0, vec![m as u32, (m >> 32) as u32])
}

/// Exact conversion of a finite f64, truncating toward zero.
/// `None` for NaN/infinity.
fn bn_from_f64(v: f64) -> Option<BigInt> {
    if v.is_nan() || v.is_infinite() {
        return None;
    }
    if v == 0.0 {
        return Some(bn_zero());
    }
    let negative = v < 0.0;
    let (m, e) = decompose(v); // |v| = m * 2^e, exact
    let d = vec![(m & 0xffff_ffff) as u32, (m >> 32) as u32];
    let b = if e >= 0 {
        // e < 2048 for every finite f64, so this always fits.
        bn_shl_mag(&d, e as u64).expect("finite f64 shift always fits")
    } else {
        bn_shr_mag(&d, (-e) as u64)
    };
    Some(bn_normalize(negative, b))
}

/// Nearest f64. Iterative rounding: correctly rounded in the common case,
/// documented as possibly 1 ulp off for values needing > 53 bits.
pub(crate) fn bn_to_f64(x: &BigInt) -> f64 {
    let mut v = 0.0f64;
    for &w in x.digits.iter().rev() {
        v = v * 4294967296.0 + f64::from(w);
    }
    if x.negative { -v } else { v }
}

/// Parse per the `BigInt()` string grammar: whitespace, one optional sign,
/// then decimal digits or a `0x`/`0o`/`0b` integer. `None` = SyntaxError.
/// Parse a `BigInt` literal. Byte-level: the grammar is ASCII-only, and
/// any lone surrogate fails digit parsing, yielding `None` (SyntaxError
/// at the call sites).
pub(crate) fn bn_parse(s: &Wtf8) -> Option<BigInt> {
    let t = trim_js_wtf8(s);
    if t.as_bytes().is_empty() {
        return Some(bn_zero());
    }
    // Sign bytes are ASCII; stripping one byte cannot split a sequence.
    let (negative, rest) = if let Some(r) = t.strip_ascii_prefix(b"-") {
        (true, r)
    } else if let Some(r) = t.strip_ascii_prefix(b"+") {
        (false, r)
    } else {
        (false, t)
    };
    let (radix, digits): (u32, &Wtf8) = if let Some(h) = rest
        .strip_ascii_prefix(b"0x")
        .or_else(|| rest.strip_ascii_prefix(b"0X"))
    {
        (16, h)
    } else if let Some(b) = rest
        .strip_ascii_prefix(b"0b")
        .or_else(|| rest.strip_ascii_prefix(b"0B"))
    {
        (2, b)
    } else if let Some(o) = rest
        .strip_ascii_prefix(b"0o")
        .or_else(|| rest.strip_ascii_prefix(b"0O"))
    {
        (8, o)
    } else {
        (10, rest)
    };
    if digits.as_bytes().is_empty() {
        return None;
    }
    // ~8 hex digits per u32; an overestimate for binary, harmless.
    let mut d: Vec<u32> = Vec::with_capacity(digits.as_bytes().len() / 8 + 1);
    // `char::to_digit` only accepts ASCII, so scanning bytes is exactly
    // equivalent and skips UTF-8 decoding.
    for &b in digits.as_bytes() {
        let v = (b as char).to_digit(radix)?;
        bn_mul_add_small(&mut d, radix, v);
    }
    Some(bn_normalize(negative, d))
}

/// `*d = *d * m + a`, in place. `bn_parse` feeds digits one at a time;
/// fusing the multiply and the add into a single pass halves the Vec
/// traffic of the old `bn_add_small(&bn_mul_small(..))` nest, and the
/// result stays normalized (no trailing zero words).
fn bn_mul_add_small(d: &mut Vec<u32>, m: u32, a: u32) {
    let mut carry = u64::from(a);
    for w in d.iter_mut() {
        let t = u64::from(*w) * u64::from(m) + carry;
        *w = t as u32;
        carry = t >> 32;
    }
    while carry != 0 {
        d.push(carry as u32);
        carry >>= 32;
    }
}

fn bn_cmp_mag(a: &[u32], b: &[u32]) -> Ordering {
    let (na, nb) = (eff_len(a), eff_len(b));
    if na != nb {
        return na.cmp(&nb);
    }
    for i in (0..na).rev() {
        if a[i] != b[i] {
            return a[i].cmp(&b[i]);
        }
    }
    Ordering::Equal
}

fn bn_add_mag(a: &[u32], b: &[u32]) -> Vec<u32> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n + 1);
    let mut carry: u64 = 0;
    for i in 0..n {
        let t = u64::from(a.get(i).copied().unwrap_or(0))
            + u64::from(b.get(i).copied().unwrap_or(0))
            + carry;
        out.push(t as u32);
        carry = t >> 32;
    }
    if carry != 0 {
        out.push(carry as u32);
    }
    out
}

/// Requires `|a| >= |b|`.
fn bn_sub_mag(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len());
    let mut borrow: i64 = 0;
    for (i, &av) in a.iter().enumerate() {
        let t = i64::from(av) - i64::from(b.get(i).copied().unwrap_or(0)) - borrow;
        out.push(t as u32); // wraps mod 2^32, correct for t in [-2^32, 2^32)
        borrow = i64::from(t < 0);
    }
    debug_assert!(borrow == 0);
    out
}

/// Schoolbook O(n*m). Fine for v1; Karatsuba/FFT when profiling demands it.
fn bn_mul_mag(a: &[u32], b: &[u32]) -> Vec<u32> {
    if bn_is_zero(a) || bn_is_zero(b) {
        return Vec::new();
    }
    let mut out = vec![0u32; a.len() + b.len()];
    for (i, &av) in a.iter().enumerate() {
        if av == 0 {
            continue;
        }
        let mut carry: u64 = 0;
        for (j, &bv) in b.iter().enumerate() {
            let t = u64::from(out[i + j]) + u64::from(av) * u64::from(bv) + carry;
            out[i + j] = t as u32;
            carry = t >> 32;
        }
        let mut k = i + b.len();
        while carry != 0 {
            let t = u64::from(out[k]) + carry;
            out[k] = t as u32;
            carry = t >> 32;
            k += 1;
        }
    }
    out
}

fn bit_len_mag(d: &[u32]) -> u64 {
    let n = eff_len(d);
    if n == 0 {
        return 0;
    }
    (n as u64 - 1) * 32 + u64::from(32 - d[n - 1].leading_zeros())
}

/// Binary long division, O(n^2) word ops. Correct and simple; Knuth
/// Algorithm D replaces it when division shows up in profiles.
/// Requires `|b| != 0`.
fn bn_divmod_mag(a: &[u32], b: &[u32]) -> (Vec<u32>, Vec<u32>) {
    debug_assert!(!bn_is_zero(b));
    if bn_cmp_mag(a, b) == Ordering::Less {
        return (Vec::new(), a.to_vec());
    }
    let nbits = bit_len_mag(a);
    let mut q = vec![0u32; nbits.div_ceil(32) as usize];
    let mut r: Vec<u32> = Vec::new();
    for i in (0..nbits).rev() {
        // r = (r << 1) | bit(a, i)
        let mut incoming = u64::from((a[(i / 32) as usize] >> (i % 32)) & 1);
        for w in r.iter_mut() {
            let t = (u64::from(*w) << 1) | incoming;
            *w = t as u32;
            incoming = t >> 32;
        }
        if incoming != 0 {
            r.push(incoming as u32);
        }
        if bn_cmp_mag(&r, b) != Ordering::Less {
            r = bn_sub_mag(&r, b);
            q[(i / 32) as usize] |= 1 << (i % 32);
        }
    }
    (q, r)
}

/// `d / div`, `d % div` for small divisors. Used by radix conversion.
fn bn_divmod_small(d: &[u32], div: u32) -> (Vec<u32>, u32) {
    debug_assert!(div >= 2);
    let n = eff_len(d);
    let mut q = vec![0u32; n];
    let mut rem: u64 = 0;
    for i in (0..n).rev() {
        let cur = (rem << 32) | u64::from(d[i]);
        q[i] = (cur / u64::from(div)) as u32;
        rem = cur % u64::from(div);
    }
    (q, rem as u32)
}

/// Cap on shift results: 2^28 words (~1 GiB). Larger shifts return `None`
/// and the builtin raises `RangeError`.
const MAX_SHIFT_WORDS: u64 = 1 << 28;

fn bn_shl_mag(d: &[u32], bits: u64) -> Option<Vec<u32>> {
    let word_shift = bits / 32;
    if word_shift > MAX_SHIFT_WORDS {
        return None;
    }
    let bit_shift = (bits % 32) as u32;
    let n = eff_len(d);
    if n == 0 {
        return Some(Vec::new());
    }
    let ws = word_shift as usize;
    let mut out = vec![0u32; ws + n + 1];
    if bit_shift == 0 {
        out[ws..ws + n].copy_from_slice(&d[..n]);
    } else {
        let mut carry: u64 = 0;
        for i in 0..n {
            let t = (u64::from(d[i]) << bit_shift) | carry;
            out[ws + i] = t as u32;
            carry = t >> 32;
        }
        out[ws + n] = carry as u32;
    }
    Some(out)
}

fn bn_shr_mag(d: &[u32], bits: u64) -> Vec<u32> {
    let word_shift = bits / 32;
    let bit_shift = (bits % 32) as u32;
    let n = eff_len(d);
    let ws = word_shift as usize;
    if ws >= n {
        return Vec::new();
    }
    let mut out = vec![0u32; n - ws];
    if bit_shift == 0 {
        out.copy_from_slice(&d[ws..n]);
    } else {
        for i in 0..out.len() {
            let lo = d[ws + i] >> bit_shift;
            let hi = if ws + i + 1 < n {
                d[ws + i + 1] << (32 - bit_shift)
            } else {
                0
            };
            out[i] = lo | hi;
        }
    }
    out
}

// --- signed operations ---

/// `pub(crate)` for the vm worker: BigInt arithmetic lives here.
pub(crate) fn bn_cmp(x: &BigInt, y: &BigInt) -> Ordering {
    if bn_is_zero(&x.digits) && bn_is_zero(&y.digits) {
        return Ordering::Equal;
    }
    if x.negative != y.negative {
        return if x.negative {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let m = bn_cmp_mag(&x.digits, &y.digits);
    if x.negative { m.reverse() } else { m }
}

/// `pub(crate)` for the vm worker: BigInt arithmetic lives here.
pub(crate) fn bn_neg(x: &BigInt) -> BigInt {
    if bn_is_zero(&x.digits) {
        bn_zero()
    } else {
        BigInt {
            negative: !x.negative,
            digits: x.digits.clone(),
        }
    }
}

/// `pub(crate)` for the vm worker: BigInt arithmetic lives here.
pub(crate) fn bn_add(x: &BigInt, y: &BigInt) -> BigInt {
    if x.negative == y.negative {
        bn_normalize(x.negative, bn_add_mag(&x.digits, &y.digits))
    } else {
        match bn_cmp_mag(&x.digits, &y.digits) {
            Ordering::Greater => bn_normalize(x.negative, bn_sub_mag(&x.digits, &y.digits)),
            Ordering::Less => bn_normalize(y.negative, bn_sub_mag(&y.digits, &x.digits)),
            Ordering::Equal => bn_zero(),
        }
    }
}

/// `pub(crate)` for the vm worker: BigInt arithmetic lives here.
pub(crate) fn bn_sub(x: &BigInt, y: &BigInt) -> BigInt {
    bn_add(x, &bn_neg(y))
}

/// `pub(crate)` for the vm worker: BigInt arithmetic lives here.
pub(crate) fn bn_mul(x: &BigInt, y: &BigInt) -> BigInt {
    bn_normalize(x.negative != y.negative, bn_mul_mag(&x.digits, &y.digits))
}

/// Truncating division: quotient rounds toward zero, remainder takes the
/// dividend's sign. `None` on division by zero.
/// `pub(crate)` for the vm worker: BigInt arithmetic lives here.
pub(crate) fn bn_divmod(x: &BigInt, y: &BigInt) -> Option<(BigInt, BigInt)> {
    if bn_is_zero(&y.digits) {
        return None;
    }
    let (q, r) = bn_divmod_mag(&x.digits, &y.digits);
    Some((
        bn_normalize(x.negative != y.negative, q),
        bn_normalize(x.negative, r),
    ))
}

/// `pub(crate)` for the vm worker: BigInt arithmetic lives here.
///
/// Returns `None` when the result would exceed [`MAX_SHIFT_WORDS`] words
/// (~1 GiB): `10n ** 1000000000n` must raise `RangeError`, not hang or OOM.
pub(crate) fn bn_pow(mut base: BigInt, mut exp: u32) -> Option<BigInt> {
    let mut result = bn_one();
    while exp > 0 {
        if exp & 1 == 1 {
            result = bn_mul(&result, &base);
            if eff_len(&result.digits) as u64 > MAX_SHIFT_WORDS {
                return None;
            }
        }
        exp >>= 1;
        if exp > 0 {
            base = bn_mul(&base, &base);
            if eff_len(&base.digits) as u64 > MAX_SHIFT_WORDS {
                return None;
            }
        }
    }
    Some(result)
}

/// `pub(crate)` for the vm worker: BigInt arithmetic lives here.
pub(crate) fn bn_shl(x: &BigInt, bits: u64) -> Option<BigInt> {
    bn_shl_mag(&x.digits, bits).map(|d| bn_normalize(x.negative, d))
}

/// Arithmetic right shift: `floor(x / 2^bits)`.
/// `pub(crate)` for the vm worker: BigInt arithmetic lives here.
pub(crate) fn bn_shr(x: &BigInt, bits: u64) -> BigInt {
    if !x.negative {
        return bn_normalize(false, bn_shr_mag(&x.digits, bits));
    }
    let m = &x.digits;
    let q = bn_shr_mag(m, bits);
    // floor(-m / 2^bits) = -(q + 1) when the shifted-out bits are nonzero.
    let word_shift = (bits / 32) as usize;
    let bit_shift = (bits % 32) as u32;
    let n = eff_len(m);
    let mut rem_nonzero = false;
    for &w in &m[..word_shift.min(n)] {
        if w != 0 {
            rem_nonzero = true;
            break;
        }
    }
    if !rem_nonzero && bit_shift != 0 && word_shift < n && (m[word_shift] << (32 - bit_shift)) != 0
    {
        rem_nonzero = true;
    }
    if rem_nonzero {
        bn_normalize(true, bn_add_mag(&q, &[1]))
    } else {
        bn_normalize(true, q)
    }
}

/// Two's-complement words of `x`, widened to `nwords` (infinite sign
/// extension). For `x = -m < 0`: `!((m - 1) words)`.
fn bn_twos_comp(x: &BigInt, nwords: usize) -> Vec<u32> {
    let mut w = vec![0u32; nwords];
    if !x.negative {
        let n = x.digits.len().min(nwords);
        w[..n].copy_from_slice(&x.digits[..n]);
    } else {
        let mut m1 = x.digits.clone();
        let mut i = 0;
        while i < m1.len() && m1[i] == 0 {
            m1[i] = u32::MAX;
            i += 1;
        }
        // |x| >= 1, so the borrow always terminates.
        if i < m1.len() {
            m1[i] -= 1;
        }
        for (slot, wi) in w.iter_mut().enumerate() {
            *wi = if slot < m1.len() { !m1[slot] } else { u32::MAX };
        }
    }
    w
}

/// Inverse of [`bn_twos_comp`]; drops the sign-extension word.
fn bn_from_twos_comp(w: &[u32], negative: bool) -> BigInt {
    debug_assert!(!w.is_empty());
    let mut d = w[..w.len() - 1].to_vec();
    if negative {
        for x in d.iter_mut() {
            *x = !*x;
        }
        let mut carry: u64 = 1;
        for x in d.iter_mut() {
            let t = u64::from(*x) + carry;
            *x = t as u32;
            carry = t >> 32;
            if carry == 0 {
                break;
            }
        }
    }
    bn_normalize(negative, d)
}

/// `pub(crate)` for the vm worker: BigInt arithmetic lives here.
pub(crate) fn bn_bitop(x: &BigInt, y: &BigInt, op: fn(u32, u32) -> u32) -> BigInt {
    let words = ((bit_len_mag(&x.digits).max(bit_len_mag(&y.digits)) + 32) / 32) as usize + 1;
    let wx = bn_twos_comp(x, words);
    let wy = bn_twos_comp(y, words);
    let w: Vec<u32> = wx.iter().zip(wy.iter()).map(|(&a, &b)| op(a, b)).collect();
    // The spare top word is pure sign extension.
    let negative = w[words - 1] == u32::MAX;
    debug_assert!(w[words - 1] == 0 || w[words - 1] == u32::MAX);
    bn_from_twos_comp(&w, negative)
}

/// `~x == -x - 1`.
/// `pub(crate)` for the vm worker: BigInt arithmetic lives here.
pub(crate) fn bn_bitnot(x: &BigInt) -> BigInt {
    bn_sub(&bn_neg(x), &bn_one())
}

// (Awaiting BigInt literal wiring: only used by the not-yet-registered
// bigint builtins; the allow keeps `cargo clippy -- -D warnings` green.)
#[allow(dead_code)]
fn bn_bit_len(x: &BigInt) -> u64 {
    bit_len_mag(&x.digits)
}

/// Base 2..=36 rendering, `0-9a-z`.
fn bn_to_string_radix(x: &BigInt, radix: u32) -> String {
    debug_assert!((2..=36).contains(&radix));
    if bn_is_zero(&x.digits) {
        return "0".to_string();
    }
    let mut m = x.digits.clone();
    // One u32 word holds at most ceil(32 / log2(radix)) digits; 32 is the
    // tight bound for radix 2 and an overestimate elsewhere.
    let mut rev: Vec<u8> = Vec::with_capacity(m.len() * 32);
    while !bn_is_zero(&m) {
        let (q, r) = bn_divmod_small(&m, radix);
        rev.push(char::from_digit(r, radix).unwrap_or('?') as u8);
        m = q;
    }
    let mut s = String::with_capacity(rev.len() + 1);
    if x.negative {
        s.push('-');
    }
    s.extend(rev.iter().rev().map(|&c| c as char));
    s
}

// ---------------------------------------------------------------------------
// String
// ---------------------------------------------------------------------------

/// UTF-16 code units of `s`. JS strings are indexed in these units.
/// Lone surrogates are preserved (WTF-8 storage).
fn utf16_units(s: &Wtf8) -> Vec<u16> {
    s.to_units()
}

/// Length of `s` in UTF-16 code units. ASCII fast path: code units are
/// bytes, and `is_ascii` is SIMD-scanned.
fn units_len(s: &Wtf8) -> usize {
    s.units_len()
}

/// Build a JS string from code units. Lone surrogates are preserved as
/// WTF-8 (the old v1 code lossy-converted them to U+FFFD).
fn units_to_string(units: &[u16]) -> Wtf8Buf {
    Wtf8Buf::from_units(units)
}

/// The `i`-th UTF-16 code unit of `s` without materializing the whole
/// string (`None` when out of range).
fn nth_unit(s: &Wtf8, i: usize) -> Option<u16> {
    s.nth_unit(i)
}

/// `true` when the code units of `s` at `pos` match `pat` exactly.
/// Caller guarantees `pos + pat.len() <= units_len(s)`.
fn units_have_prefix(s: &Wtf8, pat: &[u16], pos: usize) -> bool {
    pat.iter()
        .enumerate()
        .all(|(j, &b)| s.nth_unit(pos + j) == Some(b))
}

/// JS `s[start..end]` by UTF-16 code-unit indices (caller passes
/// `start <= end`; clamped to the string). Borrowed when the window
/// aligns to code-point boundaries; a window that splits a surrogate
/// pair materializes the lone surrogate half.
fn slice_units(s: &Wtf8, start: usize, end: usize) -> Wtf8Buf {
    s.slice_units(start, end).into_owned()
}

/// A code-unit view of a JS string for the substitution helpers:
/// borrowed bytes when ASCII (code unit == byte), borrowed UTF-16 units
/// otherwise. Lets the ASCII fast paths skip the UTF-16 round trip while
/// sharing the `$`-substitution logic.
enum UnitSlice<'a> {
    Bytes(&'a [u8]),
    Units(&'a [u16]),
}

impl<'a> UnitSlice<'a> {
    fn len(&self) -> usize {
        match self {
            UnitSlice::Bytes(b) => b.len(),
            UnitSlice::Units(u) => u.len(),
        }
    }

    /// `self[a..b]` as a JS string (code-unit indices).
    fn range_to_string(&self, a: usize, b: usize) -> Wtf8Buf {
        match self {
            // ASCII bytes are valid WTF-8.
            UnitSlice::Bytes(s) => Wtf8::from_bytes(&s[a..b])
                .expect("UnitSlice::Bytes holds ASCII")
                .to_owned(),
            UnitSlice::Units(u) => units_to_string(&u[a..b]),
        }
    }
}

fn is_lead_surrogate(u: u16) -> bool {
    (0xD800..0xDC00).contains(&u)
}

fn is_trail_surrogate(u: u16) -> bool {
    (0xDC00..0xE000).contains(&u)
}

/// Unwrap a primitive wrapper's `[[PrimitiveValue]]` (see [`PRIM_VALUE`]);
/// any other value passes through unchanged.
fn unwrap_prim(heap: &Heap, v: Value) -> Value {
    match v {
        Value::Object(o) => prim_slot(heap, o).unwrap_or(v),
        _ => v,
    }
}

/// The `this` value as a JS string: primitives via `ToString`, `String`
/// wrappers unwrap, `null`/`undefined` throw.
fn this_str(heap: &Heap, v: Value) -> Result<Wtf8Buf, NativeError> {
    match unwrap_prim(heap, v) {
        Value::Null | Value::Undefined => Err(NativeError::TypeError(
            "String prototype method called on null or undefined".to_string(),
        )),
        Value::Str(g) => Ok(g.as_wtf8().to_owned()),
        v => js_to_string_rust(v),
    }
}

/// Borrow the string content of `this` for a read-only phase: the string
/// is borrowed directly for `Value::Str` (no clone) — inline bytes from
/// the value itself, heap bytes from the allocation — other primitives
/// are coerced into `owned`, `null`/`undefined` throw.
///
/// Discipline: the borrow must not be *used* after any `heap` mutation
/// call (an allocation may trigger a mark-sweep collection). Extract
/// owned values/indices first; using the borrow as the direct argument
/// to `alloc_str` is fine (it is copied before any collection runs).
/// See the [`crate::gc`] module docs for the rooting contract.
fn this_str_borrow<'a>(
    heap: &Heap,
    this: &'a Value,
    owned: &'a mut Wtf8Buf,
) -> Result<&'a Wtf8, NativeError> {
    match this {
        Value::Str(g) => Ok(g.as_wtf8()),
        Value::Null | Value::Undefined => Err(NativeError::TypeError(
            "String prototype method called on null or undefined".to_string(),
        )),
        Value::Object(o) => {
            // Primitive wrapper (`coerce_this` boxing): unwrap the hidden
            // slot. The handle is copied out of the property, so the
            // string cannot borrow from `this`; clone into `owned`.
            match prim_slot(heap, *o) {
                Some(Value::Str(g)) => {
                    *owned = g.as_wtf8().to_owned();
                    Ok(owned.as_wtf8())
                }
                Some(v) => {
                    *owned = js_to_string_rust(v)?;
                    Ok(owned.as_wtf8())
                }
                // Not a wrapper: keep the old object coercion.
                None => {
                    *owned = js_to_string_rust(Value::Object(*o))?;
                    Ok(owned.as_wtf8())
                }
            }
        }
        v => {
            *owned = js_to_string_rust(*v)?;
            Ok(owned.as_wtf8())
        }
    }
}

/// `ToIntegerOrInfinity` clamped into `[0, len]` treating negatives as
/// `len + n` floored at 0 (the `slice`/`indexOf` convention).
fn clamp_index(n: f64, len: f64) -> f64 {
    if n < 0.0 {
        (len + n).max(0.0)
    } else {
        n.min(len)
    }
}

fn string_construct(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    // v1: call form only; `new` needs new.target from the interpreter.
    match arg(args, 0) {
        Value::Undefined => str_val(heap, Wtf8::from_str("")),
        v => js_to_string_val(heap, v),
    }
}

fn string_value_of(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    Ok(Value::Str(heap.alloc_wtf8_owned(this_str(heap, this)?)?))
}

fn string_char_at(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let s = this_str_borrow(heap, &this, &mut owned)?;
    let p = js_to_integer_or_infinity(arg(args, 0))?;
    // The unit is extracted into an owned value before any heap
    // allocation runs; the borrow never crosses one.
    let unit = if p < 0.0 {
        None
    } else {
        nth_unit(s, p as usize)
    };
    match unit {
        Some(u) => str_val(heap, &units_to_string(&[u])),
        None => str_val(heap, Wtf8::from_str("")),
    }
}

fn string_char_code_at(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let s = this_str_borrow(heap, &this, &mut owned)?;
    let p = js_to_integer_or_infinity(arg(args, 0))?;
    // No heap allocation at all on this path.
    let unit = if p < 0.0 {
        None
    } else {
        nth_unit(s, p as usize)
    };
    Ok(match unit {
        Some(u) => js_num(f64::from(u)),
        None => js_num(f64::NAN),
    })
}

fn string_code_point_at(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let s = this_str_borrow(heap, &this, &mut owned)?;
    let p = js_to_integer_or_infinity(arg(args, 0))?;
    // No heap allocation at all on this path.
    if p < 0.0 {
        return Ok(Value::Undefined);
    }
    let i = p as usize;
    let Some(first) = nth_unit(s, i) else {
        return Ok(Value::Undefined);
    };
    if is_lead_surrogate(first)
        && let Some(lo) = nth_unit(s, i + 1)
        && is_trail_surrogate(lo)
    {
        let cp = 0x10000 + (u64::from(first - 0xD800) << 10) + u64::from(lo - 0xDC00);
        return Ok(js_num(cp as f64));
    }
    Ok(js_num(f64::from(first)))
}

fn string_concat(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut parts: Vec<Wtf8Buf> = Vec::with_capacity(args.len() + 1);
    parts.push(this_str(heap, this)?);
    for a in args {
        parts.push(js_to_string_rust(*a)?);
    }
    // Size the output once instead of growing it per push.
    let total: usize = parts.iter().map(|p| p.as_wtf8().as_bytes().len()).sum();
    let mut out = Wtf8Buf::with_capacity(total);
    for p in &parts {
        out.push_wtf8(p);
    }
    str_val(heap, &out)
}

fn string_slice(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let s = this_str_borrow(heap, &this, &mut owned)?;
    let len = units_len(s) as f64;
    let from = clamp_index(js_to_integer_or_infinity(arg(args, 0))?, len);
    let to = match arg(args, 1) {
        Value::Undefined => len,
        v => clamp_index(js_to_integer_or_infinity(v)?, len),
    };
    if from >= to {
        return str_val(heap, Wtf8::from_str(""));
    }
    // Borrow ends here: `slice_units` returns an owned string.
    str_val(heap, &slice_units(s, from as usize, to as usize))
}

fn string_substring(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let s = this_str_borrow(heap, &this, &mut owned)?;
    let len = units_len(s) as f64;
    let norm = |v: Value| -> Result<f64, NativeError> {
        let n = js_to_integer_or_infinity(v)?;
        Ok(n.max(0.0).min(len))
    };
    let a = norm(arg(args, 0))?;
    let b = match arg(args, 1) {
        Value::Undefined => len,
        v => norm(v)?,
    };
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    // Borrow ends here: `slice_units` returns an owned string.
    str_val(heap, &slice_units(s, lo as usize, hi as usize))
}

/// Annex B legacy `substr(start, length)`.
fn string_substr(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let s = this_str_borrow(heap, &this, &mut owned)?;
    let size = units_len(s) as f64;
    let int_start = js_to_integer_or_infinity(arg(args, 0))?;
    let start = if int_start < 0.0 {
        (size + int_start).max(0.0)
    } else {
        int_start.min(size)
    };
    let len = match arg(args, 1) {
        Value::Undefined => size - start,
        v => js_to_integer_or_infinity(v)?.max(0.0).min(size - start),
    };
    // Borrow ends here: `slice_units` returns an owned string.
    str_val(
        heap,
        &slice_units(s, start as usize, (start + len) as usize),
    )
}

/// Code-unit search used by `includes` and the index functions (and by
/// `find_all` for `replaceAll`/`split`).
///
/// The first-unit pre-check skips the full `memcmp`-style slice comparison
/// at positions that cannot match, which keeps the common "needle absent /
/// rare first unit" cases cheap without the setup cost of a skip table.
fn units_find(hay: &[u16], needle: &[u16], from: usize) -> Option<usize> {
    if needle.is_empty() {
        return Some(from.min(hay.len()));
    }
    if needle.len() > hay.len() || from > hay.len() - needle.len() {
        return None;
    }
    let first = needle[0];
    let last_start = hay.len() - needle.len();
    let mut i = from;
    while i <= last_start {
        if hay[i] == first && &hay[i..i + needle.len()] == needle {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn string_starts_with(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let s = this_str_borrow(heap, &this, &mut owned)?;
    let search_v = arg(args, 0);
    reject_regexp_arg(search_v, "startsWith")?;
    let search = js_to_string_rust(search_v)?;
    // No heap mutation below: the borrow is safe.
    let len = units_len(s);
    let pos = js_to_integer_or_infinity(arg(args, 1))?
        .max(0.0)
        .min(len as f64) as usize;
    let found = if s.is_ascii() && search.is_ascii() {
        // Code units are bytes: compare the byte windows directly.
        let (sb, pb) = (s.as_bytes(), search.as_bytes());
        pos + pb.len() <= sb.len() && &sb[pos..pos + pb.len()] == pb
    } else {
        let pat = utf16_units(&search);
        pos + pat.len() <= len && units_have_prefix(s, &pat, pos)
    };
    Ok(Value::Bool(found))
}

fn string_ends_with(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let s = this_str_borrow(heap, &this, &mut owned)?;
    let search_v = arg(args, 0);
    reject_regexp_arg(search_v, "endsWith")?;
    let search = js_to_string_rust(search_v)?;
    // No heap mutation below: the borrow is safe.
    let len = units_len(s);
    let end = match arg(args, 1) {
        Value::Undefined => len as f64,
        v => js_to_integer_or_infinity(v)?.max(0.0).min(len as f64),
    } as usize;
    let found = if s.is_ascii() && search.is_ascii() {
        let (sb, pb) = (s.as_bytes(), search.as_bytes());
        pb.len() <= end && &sb[end - pb.len()..end] == pb
    } else {
        let pat = utf16_units(&search);
        pat.len() <= end && units_have_prefix(s, &pat, end - pat.len())
    };
    Ok(Value::Bool(found))
}

fn string_includes(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let s = this_str_borrow(heap, &this, &mut owned)?;
    let search_v = arg(args, 0);
    reject_regexp_arg(search_v, "includes")?;
    let search = js_to_string_rust(search_v)?;
    // No heap mutation below: the borrow is safe.
    let len = units_len(s);
    let pos = js_to_integer_or_infinity(arg(args, 1))?
        .max(0.0)
        .min(len as f64) as usize;
    let found = if s.is_ascii() && search.is_ascii() {
        // `str::find` is memchr-accelerated; byte indices are code units.
        search.is_empty()
            || s.as_ascii_str()[pos..]
                .find(search.as_ascii_str())
                .is_some()
    } else {
        let u = utf16_units(s);
        units_find(&u, &utf16_units(&search), pos).is_some()
    };
    Ok(Value::Bool(found))
}

fn string_index_of(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let s = this_str_borrow(heap, &this, &mut owned)?;
    let search = js_to_string_rust(arg(args, 0))?;
    // No heap mutation below: the borrow is safe.
    let len = units_len(s);
    let pos = js_to_integer_or_infinity(arg(args, 1))?
        .max(0.0)
        .min(len as f64) as usize;
    let found = if s.is_ascii() && search.is_ascii() {
        if search.is_empty() {
            Some(pos)
        } else {
            s.as_ascii_str()[pos..]
                .find(search.as_ascii_str())
                .map(|i| pos + i)
        }
    } else {
        let u = utf16_units(s);
        units_find(&u, &utf16_units(&search), pos)
    };
    Ok(js_num(found.map_or(-1.0, |i| i as f64)))
}

fn string_last_index_of(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let s = this_str_borrow(heap, &this, &mut owned)?;
    let search = js_to_string_rust(arg(args, 0))?;
    // No heap mutation below: the borrow is safe.
    let len = units_len(s);
    let from = match arg(args, 1) {
        Value::Undefined => len as f64,
        v => js_to_integer_or_infinity(v)?.max(0.0).min(len as f64),
    } as usize;
    if search.is_empty() {
        return Ok(js_num(from as f64));
    }
    let found = if s.is_ascii() && search.is_ascii() {
        let (sb, pb) = (s.as_bytes(), search.as_bytes());
        if pb.len() > sb.len() {
            None
        } else {
            // Any match starting at `<= top` lies fully inside the
            // truncated slice, so the memchr-accelerated `rfind` on it
            // is exact. (ASCII: every byte index is a char boundary.)
            let top = from.min(sb.len() - pb.len());
            s.as_ascii_str()[..top + pb.len()].rfind(search.as_ascii_str())
        }
    } else {
        let u = utf16_units(s);
        let pat = utf16_units(&search);
        if pat.len() > u.len() {
            None
        } else {
            let top = from.min(u.len() - pat.len());
            (0..=top).rev().find(|&i| u[i..i + pat.len()] == *pat)
        }
    };
    Ok(js_num(found.map_or(-1.0, |i| i as f64)))
}

/// Shared `padStart`/`padEnd` implementation.
fn string_pad(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
    at_start: bool,
) -> Result<Value, NativeError> {
    let s = this_str(heap, this)?;
    let len = units_len(&s) as u64;
    let target = js_to_length(arg(args, 0))?;
    if target <= len {
        return str_val(heap, &s);
    }
    let filler = match arg(args, 1) {
        Value::Undefined => Wtf8Buf::from(" "),
        v => js_to_string_rust(v)?,
    };
    if filler.is_empty() {
        return str_val(heap, &s);
    }
    let fill_len = target - len;
    // Memory-safety guard: no JS string may exceed 2^28-1 code units here.
    if target > (1u64 << 28) - 1 {
        return Err(NativeError::RangeError("Invalid string length".to_string()));
    }
    let fu = utf16_units(&filler);
    let mut units: Vec<u16> = Vec::with_capacity(fill_len as usize);
    while units.len() < fill_len as usize {
        units.extend_from_slice(&fu);
    }
    units.truncate(fill_len as usize);
    let fill = units_to_string(&units);
    let mut out = Wtf8Buf::with_capacity(s.len() + fill.len());
    if at_start {
        out.push_wtf8(&fill);
        out.push_wtf8(&s);
    } else {
        out.push_wtf8(&s);
        out.push_wtf8(&fill);
    }
    str_val(heap, &out)
}

fn string_pad_start(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    string_pad(heap, this, args, true)
}

fn string_pad_end(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    string_pad(heap, this, args, false)
}

fn string_repeat(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let s = this_str(heap, this)?;
    let n = js_to_integer_or_infinity(arg(args, 0))?;
    if n < 0.0 || n.is_infinite() {
        return Err(NativeError::RangeError("Invalid count value".to_string()));
    }
    if n == 0.0 || s.is_empty() {
        return str_val(heap, Wtf8::from_str(""));
    }
    let len = units_len(&s) as f64;
    if n * len > ((1u64 << 28) - 1) as f64 {
        return Err(NativeError::RangeError("Invalid string length".to_string()));
    }
    // Preallocate the full result once instead of pushing one copy per
    // iteration.
    let count = n as usize;
    let mut out = Wtf8Buf::with_capacity(s.as_wtf8().as_bytes().len() * count);
    for _ in 0..count {
        out.push_wtf8(&s);
    }
    str_val(heap, &out)
}

/// `GetSubstitution` for string-pattern replacement: `$$`, `$&`, `$\``,
/// `$'`, `$1`..`$99`. `$<name>` has no named groups in v1 and stays literal.
/// The template is scanned as code points (lone surrogates preserved);
/// only ASCII `$` sigils are special.
fn get_substitution(
    template: &Wtf8,
    whole: &UnitSlice,
    pos: usize,
    mlen: usize,
    captures: &[Wtf8Buf],
) -> Wtf8Buf {
    let mut out = Wtf8Buf::with_capacity(template.len());
    let mut chars = template.chars_wtf8().peekable();
    let m = captures.len().saturating_sub(1); // captures[0] is the whole match
    let wlen = whole.len();
    while let Some(c) = chars.next() {
        if c != '$' as u32 {
            push_code_point(&mut out, c);
            continue;
        }
        match chars.peek().copied() {
            Some(c2) if c2 == '$' as u32 => {
                out.push_char('$');
                chars.next();
            }
            Some(c2) if c2 == '&' as u32 => {
                out.push_wtf8(whole.range_to_string(pos, pos + mlen).as_wtf8());
                chars.next();
            }
            Some(c2) if c2 == '`' as u32 => {
                out.push_wtf8(whole.range_to_string(0, pos).as_wtf8());
                chars.next();
            }
            Some(c2) if c2 == '\'' as u32 => {
                out.push_wtf8(whole.range_to_string(pos + mlen, wlen).as_wtf8());
                chars.next();
            }
            Some(c2) if c2 < 0x80 && (c2 as u8).is_ascii_digit() => {
                let d1 = (c2 as u8 - b'0') as usize;
                chars.next();
                let mut num = d1;
                let mut consumed_two = false;
                if let Some(d2) = chars.peek().copied()
                    && d2 < 0x80
                    && (d2 as u8).is_ascii_digit()
                {
                    let nn = d1 * 10 + (d2 as u8 - b'0') as usize;
                    if nn >= 1 && nn <= m {
                        num = nn;
                        consumed_two = true;
                        chars.next();
                    }
                }
                if !consumed_two && (num < 1 || num > m) {
                    // Not a valid group reference: emit literally.
                    out.push_char('$');
                    out.push_char((b'0' + d1 as u8) as char);
                } else {
                    out.push_wtf8(captures[num].as_wtf8());
                }
            }
            // `$<name>`: no named groups in v1 -> literal. Trailing/unknown
            // `$x` -> literal `$`.
            _ => out.push_char('$'),
        }
    }
    out
}

/// Push one Unicode scalar value (or lone surrogate, as WTF-8) from a
/// [`Wtf8::chars_wtf8`] code point.
pub(crate) fn push_code_point(out: &mut Wtf8Buf, c: u32) {
    if c <= 0xFFFF {
        // Includes lone surrogates: preserved, not replaced.
        out.push_unit(c as u16);
    } else {
        let v = c - 0x10000;
        out.push_unit(0xD800 | (v >> 10) as u16);
        out.push_unit(0xDC00 | (v & 0x3FF) as u16);
    }
}

/// All non-overlapping occurrences of `pat` in `u` at/after `from`.
fn find_all(u: &[u16], pat: &[u16], from: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = from.min(u.len());
    if pat.is_empty() {
        return out;
    }
    while let Some(p) = units_find(u, pat, i) {
        out.push((p, pat.len()));
        i = p + pat.len().max(1);
    }
    out
}

fn string_replace(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let s = this_str(heap, this)?;
    let search_v = arg(args, 0);
    let repl_v = arg(args, 1);
    // Spec 22.1.3.19: delegate to the search value's `@@replace` when it
    // is callable. This covers RegExp (via `RegExp.prototype[Symbol.replace]`)
    // and any user object defining the hook.
    if let Value::Object(o) = search_v
        && let Some(m) = well_known_get(heap, o, WellKnown::Replace)
        && callable_value(m).is_some()
    {
        let s_val = Value::Str(heap.alloc_str_wtf8(s.as_wtf8())?);
        return crate::builtin_object_array::call_function(heap, m, search_v, &[s_val, repl_v]);
    }
    // Null-prototype regexp literals skip the method lookup above; run the
    // engine directly for them.
    if let Value::Object(o) = search_v
        && is_regexp(search_v)
    {
        let (source, flags) = regexp_source_flags(o);
        if callable_value(repl_v).is_some() {
            // Callable replacement goes through the shared `@@replace`
            // worker directly (no method to look up on a null proto).
            let s_val = Value::Str(heap.alloc_str_wtf8(s.as_wtf8())?);
            return crate::builtin_regexp::regexp_symbol_replace(heap, search_v, &[s_val, repl_v]);
        }
        let template = js_to_string_rust(repl_v)?;
        // The engine compiles the exact WTF-8 pattern (lone surrogates
        // match as-is); the subject and the result stay exact WTF-8.
        let out = crate::builtin_regexp::regexp_replace(
            source.as_wtf8(),
            flags.as_wtf8(),
            s.as_wtf8(),
            template.as_wtf8(),
        )
        .map_err(regex_native_error)?;
        return str_val(heap, out.as_wtf8());
    }
    let search = js_to_string_rust(search_v)?;
    // ASCII fast path: byte indices are code-unit indices, so the whole
    // operation runs on bytes with no UTF-16 round trip.
    if s.is_ascii() && search.is_ascii() {
        return string_replace_ascii(heap, s.as_wtf8(), search.as_wtf8(), repl_v);
    }
    let u = s.to_units();
    let pat = search.to_units();
    let (pos, mlen) = match units_find(&u, &pat, 0) {
        Some(p) => (p, pat.len()),
        None => return str_val(heap, s.as_wtf8()),
    };
    let whole_g = heap.alloc_str_wtf8(s.as_wtf8())?;
    let replacement = if callable_value(repl_v).is_some() {
        let matched = heap.alloc_wtf8_owned(Wtf8Buf::from_units(&u[pos..pos + mlen]))?;
        let r = crate::builtin_object_array::call_function(
            heap,
            repl_v,
            Value::Undefined,
            &[Value::Str(matched), js_num(pos as f64), Value::Str(whole_g)],
        )?;
        js_to_string_rust(r)?
    } else {
        let template = js_to_string_rust(repl_v)?;
        // String patterns have no capture groups: `$1` etc. stay literal,
        // so no per-match capture allocation is needed.
        get_substitution(template.as_wtf8(), &UnitSlice::Units(&u), pos, mlen, &[])
    };
    // One code-unit buffer, one conversion.
    let mut units: Vec<u16> = Vec::with_capacity(u.len() + replacement.units_len());
    units.extend_from_slice(&u[..pos]);
    units.extend_from_slice(&replacement.to_units());
    units.extend_from_slice(&u[pos + mlen..]);
    str_val(heap, units_to_string(&units).as_wtf8())
}

/// `String.prototype.replace` for ASCII `s` and ASCII string patterns.
/// (The replacement itself may be non-ASCII; only `s`/`search` are.)
fn string_replace_ascii(
    heap: &mut Heap,
    s: &Wtf8,
    search: &Wtf8,
    repl_v: Value,
) -> Result<Value, NativeError> {
    // Callers guarantee ASCII, hence valid UTF-8.
    let s = s.as_utf8().expect("string_replace_ascii: ascii s");
    let search = search
        .as_utf8()
        .expect("string_replace_ascii: ascii search");
    // `str::find("")` is `Some(0)`: the empty pattern matches first,
    // exactly like the code-unit search.
    let Some(pos) = s.find(search) else {
        return str_val(heap, Wtf8::from_str(s));
    };
    let mlen = search.len();
    let whole = UnitSlice::Bytes(s.as_bytes());
    let replacement = if callable_value(repl_v).is_some() {
        // Owned before the call: the borrow must not cross it.
        let matched = s[pos..pos + mlen].to_owned();
        let matched_s = Value::Str(heap.alloc_str(&matched)?);
        let whole_s = Value::Str(heap.alloc_str(s)?);
        let r = crate::builtin_object_array::call_function(
            heap,
            repl_v,
            Value::Undefined,
            &[matched_s, js_num(pos as f64), whole_s],
        )?;
        js_to_string_rust(r)?
    } else {
        let template = js_to_string_rust(repl_v)?;
        // String patterns have no capture groups: `$1` etc. stay literal,
        // so no per-match capture allocation is needed.
        get_substitution(template.as_wtf8(), &whole, pos, mlen, &[])
    };
    let mut out = Wtf8Buf::with_capacity(s.len() + replacement.len());
    out.push_wtf8_bytes(&s.as_bytes()[..pos]);
    out.push_wtf8(replacement.as_wtf8());
    out.push_wtf8_bytes(&s.as_bytes()[pos + mlen..]);
    str_val(heap, out.as_wtf8())
}

fn string_replace_all(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let s = this_str(heap, this)?;
    let search_v = arg(args, 0);
    let repl_v = arg(args, 1);
    // RegExp search value: must be global (spec 22.1.3.19).
    if let Value::Object(o) = search_v
        && is_regexp(search_v)
    {
        let (_, flags) = regexp_source_flags(o);
        if !flags.as_bytes().contains(&b'g') {
            return Err(NativeError::TypeError(
                "String.prototype.replaceAll called with a non-global RegExp".to_string(),
            ));
        }
    }
    // Spec: delegate to the search value's `@@replace` when callable.
    if let Value::Object(o) = search_v
        && let Some(m) = well_known_get(heap, o, WellKnown::Replace)
        && callable_value(m).is_some()
    {
        let s_val = Value::Str(heap.alloc_str_wtf8(s.as_wtf8())?);
        let out_v =
            crate::builtin_object_array::call_function(heap, m, search_v, &[s_val, repl_v])?;
        let out = js_to_string_rust(out_v)?;
        if out.units_len() > (1 << 28) - 1 {
            return Err(NativeError::RangeError("Invalid string length".to_string()));
        }
        return str_val(heap, out.as_wtf8());
    }
    // Null-prototype regexp literals: run the engine directly.
    if let Value::Object(o) = search_v
        && is_regexp(search_v)
    {
        let (source, flags) = regexp_source_flags(o);
        if callable_value(repl_v).is_some() {
            let s_val = Value::Str(heap.alloc_str_wtf8(s.as_wtf8())?);
            let out_v =
                crate::builtin_regexp::regexp_symbol_replace(heap, search_v, &[s_val, repl_v])?;
            let out = js_to_string_rust(out_v)?;
            if out.units_len() > (1 << 28) - 1 {
                return Err(NativeError::RangeError("Invalid string length".to_string()));
            }
            return str_val(heap, out.as_wtf8());
        }
        let template = js_to_string_rust(repl_v)?;
        // See `string_replace`: the pattern stays exact, subject and
        // result stay exact.
        let out = crate::builtin_regexp::regexp_replace(
            source.as_wtf8(),
            flags.as_wtf8(),
            s.as_wtf8(),
            template.as_wtf8(),
        )
        .map_err(regex_native_error)?;
        if out.units_len() > (1 << 28) - 1 {
            return Err(NativeError::RangeError("Invalid string length".to_string()));
        }
        return str_val(heap, out.as_wtf8());
    }
    let search = js_to_string_rust(search_v)?;
    // ASCII fast path: byte indices are code-unit indices, so the whole
    // operation runs on bytes with no UTF-16 round trip.
    if s.is_ascii() && search.is_ascii() {
        return string_replace_all_ascii(heap, s.as_wtf8(), search.as_wtf8(), repl_v);
    }
    let u = s.to_units();
    let pat = search.to_units();
    let repl_fn = callable_value(repl_v);
    let template = if repl_fn.is_none() {
        Some(js_to_string_rust(repl_v)?)
    } else {
        None
    };
    // Plain template (no `$`): skip substitution processing and the
    // per-match capture allocation entirely.
    let plain = match template.as_deref() {
        Some(t) if !t.as_bytes().contains(&b'$') => Some(t),
        _ => None,
    };

    // Occurrence list. Empty pattern matches between every code unit.
    let occurrences: Vec<(usize, usize)> = if pat.is_empty() {
        (0..=u.len()).map(|p| (p, 0)).collect()
    } else {
        find_all(&u, &pat, 0)
    };

    let whole = UnitSlice::Units(&u);
    // `whole_g` only exists on the callable path; allocate it lazily.
    let mut whole_g: Option<Value> = None;
    // One code-unit buffer, one conversion at the end (the old code
    // converted every segment separately).
    let mut out: Vec<u16> = Vec::with_capacity(u.len() + occurrences.len() * 4);
    let mut last = 0;
    for (pos, mlen) in occurrences {
        out.extend_from_slice(&u[last..pos]);
        if let Some(p) = plain {
            // No allocation: the template is used directly.
            out.extend(p.to_units());
        } else {
            let replacement = if let Some(f) = repl_fn {
                let matched = heap.alloc_wtf8_owned(units_to_string(&u[pos..pos + mlen]))?;
                let wg = match whole_g {
                    Some(v) => v,
                    None => {
                        let v = Value::Str(heap.alloc_str_wtf8(s.as_wtf8())?);
                        whole_g = Some(v);
                        v
                    }
                };
                let r = crate::builtin_object_array::call_function(
                    heap,
                    f,
                    Value::Undefined,
                    &[Value::Str(matched), js_num(pos as f64), wg],
                )?;
                js_to_string_rust(r)?
            } else {
                get_substitution(
                    template.as_deref().unwrap_or(Wtf8::from_str("")),
                    &whole,
                    pos,
                    mlen,
                    &[],
                )
            };
            out.extend_from_slice(&replacement.to_units());
        }
        last = pos + mlen;
    }
    out.extend_from_slice(&u[last..]);
    // Length guard, same policy as repeat/pad.
    if out.len() > (1 << 28) - 1 {
        return Err(NativeError::RangeError("Invalid string length".to_string()));
    }
    str_val(heap, units_to_string(&out).as_wtf8())
}

/// `String.prototype.replaceAll` for ASCII `s` and ASCII string patterns.
/// (Templates and replacements may still be non-ASCII.)
fn string_replace_all_ascii(
    heap: &mut Heap,
    s: &Wtf8,
    search: &Wtf8,
    repl_v: Value,
) -> Result<Value, NativeError> {
    // Callers guarantee ASCII, hence valid UTF-8.
    let s = s.as_utf8().expect("string_replace_all_ascii: ascii s");
    let search = search
        .as_utf8()
        .expect("string_replace_all_ascii: ascii search");
    let repl_fn = callable_value(repl_v);
    let template = if repl_fn.is_none() {
        Some(js_to_string_rust(repl_v)?)
    } else {
        None
    };
    // Fast path: a template without `$` needs no substitution processing
    // and no per-match capture allocation (string patterns have no groups,
    // so captures would be unused anyway).
    let plain = match template.as_deref() {
        Some(t) if !t.as_bytes().contains(&b'$') => Some(t),
        _ => None,
    };

    let sb = s.as_bytes();
    let pb = search.as_bytes();
    let whole = UnitSlice::Bytes(sb);
    let mut whole_g: Option<Value> = None;
    // Single pass: scan bytes directly instead of looping `str::find` on
    // shrinking suffixes (whose per-call searcher setup dominates when
    // there are many matches).
    let mut out = Wtf8Buf::with_capacity(s.len() + 64);
    let mut i = 0;
    if pb.is_empty() {
        // Empty pattern matches between every code unit (== every byte).
        while i <= sb.len() {
            if let Some(p) = plain {
                out.push_wtf8(p);
            } else {
                let replacement = if let Some(f) = repl_fn {
                    let wg = match whole_g {
                        Some(v) => v,
                        None => {
                            let v = Value::Str(heap.alloc_str(s)?);
                            whole_g = Some(v);
                            v
                        }
                    };
                    let matched_s = Value::Str(heap.alloc_str("")?);
                    let r = crate::builtin_object_array::call_function(
                        heap,
                        f,
                        Value::Undefined,
                        &[matched_s, js_num(i as f64), wg],
                    )?;
                    js_to_string_rust(r)?
                } else {
                    get_substitution(
                        template.as_deref().unwrap_or(Wtf8::from_str("")),
                        &whole,
                        i,
                        0,
                        &[],
                    )
                };
                out.push_wtf8(replacement.as_wtf8());
            }
            if i < sb.len() {
                out.push_char(sb[i] as char);
            }
            i += 1;
        }
    } else {
        while i + pb.len() <= sb.len() {
            if &sb[i..i + pb.len()] == pb {
                if let Some(p) = plain {
                    // No allocation: the template is used directly.
                    out.push_wtf8(p);
                } else {
                    let replacement = if let Some(f) = repl_fn {
                        let matched = s[i..i + pb.len()].to_owned();
                        let wg = match whole_g {
                            Some(v) => v,
                            None => {
                                let v = Value::Str(heap.alloc_str(s)?);
                                whole_g = Some(v);
                                v
                            }
                        };
                        let matched_s = Value::Str(heap.alloc_str(&matched)?);
                        let r = crate::builtin_object_array::call_function(
                            heap,
                            f,
                            Value::Undefined,
                            &[matched_s, js_num(i as f64), wg],
                        )?;
                        js_to_string_rust(r)?
                    } else {
                        get_substitution(
                            template.as_deref().unwrap_or(Wtf8::from_str("")),
                            &whole,
                            i,
                            pb.len(),
                            &[],
                        )
                    };
                    out.push_wtf8(replacement.as_wtf8());
                }
                i += pb.len();
            } else {
                out.push_char(sb[i] as char);
                i += 1;
            }
        }
        out.push_wtf8_bytes(&sb[i..]);
    }
    // Length guard, same policy as repeat/pad.
    if out.units_len() > (1 << 28) - 1 {
        return Err(NativeError::RangeError("Invalid string length".to_string()));
    }
    str_val(heap, out.as_wtf8())
}

fn string_split(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let lim = match arg(args, 1) {
        Value::Undefined => u32::MAX,
        v => js_to_uint32(v)?,
    };
    if lim == 0 {
        return new_array(heap, Vec::new());
    }
    let s = this_str(heap, this)?;
    let sep_v = arg(args, 0);
    if sep_v == Value::Undefined {
        let only = Value::Str(heap.alloc_str_wtf8(s.as_wtf8())?);
        return new_array(heap, vec![only]);
    }
    // Spec 22.1.3.23: delegate to the separator's `@@split` when callable.
    // This covers RegExp (via `RegExp.prototype[Symbol.split]`) and any user
    // object defining the hook.
    if let Value::Object(o) = sep_v
        && let Some(m) = well_known_get(heap, o, WellKnown::Split)
        && callable_value(m).is_some()
    {
        let s_val = Value::Str(heap.alloc_str_wtf8(s.as_wtf8())?);
        let lim_v = arg(args, 1);
        return crate::builtin_object_array::call_function(heap, m, sep_v, &[s_val, lim_v]);
    }
    // Null-prototype regexp literals: run the engine directly.
    if let Value::Object(o) = sep_v
        && is_regexp(sep_v)
    {
        let (source, flags) = regexp_source_flags(o);
        // See `string_replace`: the pattern stays exact, subject and
        // parts stay exact.
        let parts = crate::builtin_regexp::regexp_split(
            source.as_wtf8(),
            flags.as_wtf8(),
            s.as_wtf8(),
            lim,
        )
        .map_err(regex_native_error)?;
        let mut elems = Vec::with_capacity(parts.len());
        for p in parts {
            elems.push(Value::Str(heap.alloc_wtf8_owned(p)?));
        }
        return new_array(heap, elems);
    }
    let sep = js_to_string_rust(sep_v)?;
    // ASCII fast path: byte indices are code-unit indices, so the whole
    // operation runs on bytes with no UTF-16 round trip. (`s` is an owned
    // Rust string here, so borrowing it across the allocations below is
    // always sound.)
    if s.is_ascii() && sep.is_ascii() {
        // ASCII implies valid UTF-8, so the byte view borrows for free.
        let (s, sep) = (s.as_utf8().unwrap(), sep.as_utf8().unwrap());
        if sep.is_empty() {
            // Split into individual code units (== bytes).
            let count = (lim as usize).min(s.len());
            let mut elems = Vec::with_capacity(count);
            for b in s.bytes().take(count) {
                elems.push(Value::Str(heap.alloc_str(&(b as char).to_string())?));
            }
            return new_array(heap, elems);
        }
        // Stop after `lim` parts instead of splitting the whole string and
        // truncating: `Split` is lazy, so a small limit skips most of the
        // scan. (Taking the first `lim` parts is exactly truncation.)
        let parts: Vec<&str> = s.split(sep).take(lim as usize).collect();
        let mut elems = Vec::with_capacity(parts.len());
        for p in parts {
            elems.push(Value::Str(heap.alloc_str(p)?));
        }
        return new_array(heap, elems);
    }
    let u = s.to_units();
    let pat = sep.to_units();
    let limit = lim as usize;
    let mut parts: Vec<Wtf8Buf> = Vec::new();
    if pat.is_empty() {
        // Split into individual code units.
        parts.extend(u.iter().take(limit).map(|&c| units_to_string(&[c])));
    } else {
        let mut i = 0;
        // Search incrementally and stop once the limit is reached: a small
        // limit skips most of the scan instead of finding every occurrence
        // upfront and truncating afterwards.
        while parts.len() < limit {
            match units_find(&u, &pat, i) {
                Some(p) => {
                    parts.push(units_to_string(&u[i..p]));
                    i = p + pat.len().max(1);
                }
                None => break,
            }
        }
        // The trailing segment is always a part unless the limit is hit.
        if parts.len() < limit {
            parts.push(units_to_string(&u[i..]));
        }
    }
    let mut elems = Vec::with_capacity(parts.len());
    for p in parts {
        elems.push(Value::Str(heap.alloc_wtf8_owned(p)?));
    }
    new_array(heap, elems)
}

fn string_trim(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let t = trim_js_wtf8(this_str_borrow(heap, &this, &mut owned)?);
    str_val(heap, t)
}

fn string_trim_start(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let t = trim_js_start_wtf8(this_str_borrow(heap, &this, &mut owned)?);
    str_val(heap, t)
}

fn string_trim_end(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let t = trim_js_end_wtf8(this_str_borrow(heap, &this, &mut owned)?);
    str_val(heap, t)
}

/// Unicode simple case mapping over code points; lone surrogates pass
/// through unchanged (they have no case mapping).
fn case_map(s: &Wtf8, upper: bool) -> Wtf8Buf {
    let mut out = Wtf8Buf::with_capacity(s.as_bytes().len());
    for cp in s.chars_wtf8() {
        if (0xd800..0xe000).contains(&cp) {
            out.push_unit(cp as u16);
            continue;
        }
        let c = char::from_u32(cp).expect("non-surrogate code point decodes");
        if upper {
            for uc in c.to_uppercase() {
                out.push_char(uc);
            }
        } else {
            for lc in c.to_lowercase() {
                out.push_char(lc);
            }
        }
    }
    out
}

fn string_to_upper_case(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let s = this_str_borrow(heap, &this, &mut owned)?;
    let upper = case_map(s, true);
    str_val(heap, &upper)
}

fn string_to_lower_case(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let s = this_str_borrow(heap, &this, &mut owned)?;
    let lower = case_map(s, false);
    str_val(heap, &lower)
}

/// Spec 22.1.3.33 `String.prototype.isWellFormed`: `true` iff the string
/// has no lone surrogates in its UTF-16 encoding.
fn string_is_well_formed(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let s = this_str_borrow(heap, &this, &mut owned)?;
    Ok(Value::Bool(s.is_well_formed()))
}

/// Spec 22.1.3.34 `String.prototype.toWellFormed`: returns the string with
/// every lone surrogate replaced by U+FFFD. Well-formed input is returned
/// unchanged (same handle when `this` is already a string).
fn string_to_well_formed(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let s = this_str_borrow(heap, &this, &mut owned)?;
    if s.is_well_formed() {
        return Ok(match this {
            // `this` was already a `Value::Str`; reusing the handle
            // avoids a copy when no replacement is needed.
            Value::Str(g) => Value::Str(g),
            _ => Value::Str(heap.alloc_str_wtf8(s)?),
        });
    }
    let fixed = s.to_well_formed();
    str_val(heap, &fixed)
}

/// Spec 22.1.3.21 `String.prototype.normalize`: UAX #15 normalization
/// (NFC/NFD/NFKC/NFKD). An invalid form throws `RangeError`; ASCII input is
/// already normalized in every form and reuses the handle.
fn string_normalize(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let s = this_str_borrow(heap, &this, &mut owned)?;
    let form = match arg(args, 0) {
        Value::Undefined => crate::normalize::Form::Nfc,
        v => {
            let name = js_to_string_rust(v)?;
            crate::normalize::Form::from_name(name.as_wtf8().as_bytes()).ok_or_else(|| {
                NativeError::RangeError(
                    "String.prototype.normalize: invalid normalization form".to_string(),
                )
            })?
        }
    };
    if s.is_ascii() {
        return Ok(match this {
            // `this` was already a `Value::Str`; reusing the handle
            // avoids a copy when nothing can change.
            Value::Str(g) => Value::Str(g),
            _ => Value::Str(heap.alloc_str_wtf8(s)?),
        });
    }
    let cps: Vec<u32> = s.chars_wtf8().collect();
    let normed = crate::normalize::normalize_cps(&cps, form);
    // Back to UTF-16 code units (lone surrogates pass through as themselves),
    // then to the crate's WTF-8 string encoding.
    let mut units = Vec::with_capacity(normed.len());
    for cp in normed {
        if cp < 0x10000 {
            units.push(cp as u16);
        } else {
            let c = cp - 0x10000;
            units.push(0xD800 + (c >> 10) as u16);
            units.push(0xDC00 + (c & 0x3FF) as u16);
        }
    }
    let buf = Wtf8Buf::from_units(&units);
    str_val(heap, buf.as_wtf8())
}

/// v1: raw UTF-16 code-unit comparison (-1/0/1), not locale-aware.
fn string_locale_compare(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let mut owned = Wtf8Buf::new();
    let a = this_str_borrow(heap, &this, &mut owned)?;
    let b = js_to_string_rust(arg(args, 0))?;
    // No heap mutation below: the borrow is safe. WTF-8 byte order is
    // UTF-16 code-unit order, so this is the spec's code-unit comparison.
    let ord = a.cmp(b.as_wtf8());
    Ok(js_num(match ord {
        Ordering::Less => -1.0,
        Ordering::Equal => 0.0,
        Ordering::Greater => 1.0,
    }))
}

// ---------------------------------------------------------------------------
// RegExp delegation bridge.
//
// `String.prototype.replace` / `replaceAll` / `split` / `match` / `search`
// route their regular-expression paths to the real engine in
// [`crate::builtin_regexp`]:
//
// * `replace` / `replaceAll` -> [`crate::builtin_regexp::regexp_replace`]
//   (template replacements); callable replacements delegate to the search
//   value's `@@replace` via `call_function`, and string-pattern callable
//   replacements are invoked directly through `call_function`.
// * `split`                 -> [`crate::builtin_regexp::regexp_split`]
// * `match`                 -> `@@match` via `GetMethod`, else
//   [`crate::builtin_regexp::regexp_construct`] +
//   [`crate::builtin_regexp::regexp_symbol_match`]
// * `matchAll`              -> `@@matchAll` via `GetMethod`, else
//   [`crate::builtin_regexp::regexp_construct`] +
//   [`crate::builtin_regexp::regexp_symbol_match_all`] (v1: returns an
//   `Array` of match arrays, not a `RegExpStringIterator`)
// * `search`                -> `@@search` via `GetMethod`, else
//   [`crate::builtin_regexp::regexp_construct`] +
//   [`crate::builtin_regexp::regexp_symbol_search`]
// ---------------------------------------------------------------------------

/// Spec `IsRegExp`: true when `v` is a `RegExp` object.
fn is_regexp(v: Value) -> bool {
    matches!(
        v,
        Value::Object(o) if matches!(crate::gc::object_kind(o), ObjectKind::RegExp)
    )
}

/// The `source`/`flags` string data properties of a `RegExp` object.
fn regexp_source_flags(o: Gc<Object>) -> (Wtf8Buf, Wtf8Buf) {
    let source = match prop_get(o, Wtf8::from_str("source")) {
        Some(Value::Str(g)) => g.as_wtf8().to_owned(),
        _ => Wtf8Buf::new(),
    };
    let flags = match prop_get(o, Wtf8::from_str("flags")) {
        Some(Value::Str(g)) => g.as_wtf8().to_owned(),
        _ => Wtf8Buf::new(),
    };
    (source, flags)
}

/// Map a [`crate::regex_engine::RegexError`] to the JS error the spec
/// mandates: `SyntaxError` for an invalid pattern/flags, `RangeError` when
/// the engine's backtracking budget is exhausted.
fn regex_native_error(e: crate::regex_engine::RegexError) -> NativeError {
    use crate::regex_engine::RegexError as RE;
    match e {
        RE::ParseError(msg) => NativeError::SyntaxError(msg),
        RE::BacktrackLimit => {
            NativeError::RangeError("regex backtracking limit exceeded".to_string())
        }
    }
}

/// Throw when the first argument must not be a `RegExp`
/// (`includes`/`startsWith`/`endsWith`, spec 22.1.3.x step 2).
fn reject_regexp_arg(v: Value, method: &str) -> Result<(), NativeError> {
    if is_regexp(v) {
        return Err(NativeError::TypeError(format!(
            "String.prototype.{method} called with a RegExp argument"
        )));
    }
    Ok(())
}

/// Chain-resolved property: a data value, or an accessor's getter.
#[derive(Clone, Copy)]
enum ChainSlot {
    Data(Value),
    Getter(Value),
}

thread_local! {
    /// `%String.prototype%` / `%Number.prototype%` / `%BigInt.prototype%`
    /// per heap, recorded by [`install_inner`]. `get_method` needs the
    /// wrapper prototype for the spec `GetMethod` `ToObject` step on
    /// primitive patterns: a hook installed on `String.prototype` must be
    /// visible to `"ab".matchAll("a")`. The native cannot reach the global
    /// object (or its own callee) from the `NativeFn` signature, so the
    /// install-time record is the stand-in — the same rationale as
    /// `REGEXP_PROTO` in `crate::builtin_regexp`. Keyed by heap id: `Gc`
    /// handles are heap-local and must never cross heaps. (`%Symbol.prototype%`
    /// is already a heap intrinsic; v1 has no `%Boolean.prototype%`, so
    /// boolean patterns keep the "no hook" fallback, mirroring the VM's
    /// `primitive_proto`.)
    static WRAPPER_PROTOS: RefCell<Option<(u64, WrapperProtos)>> =
        const { RefCell::new(None) };
}

/// Install-time record of the primitive wrapper prototypes (see
/// `WRAPPER_PROTOS`).
#[derive(Clone, Copy)]
struct WrapperProtos {
    string: Gc<Object>,
    number: Gc<Object>,
    bigint: Gc<Object>,
}

/// The wrapper prototype used to box a primitive for the spec `GetMethod`
/// `ToObject` step (`None` for `undefined`/`null`, which the callers treat
/// as "no hook", and for primitives whose prototype is not installed or
/// not recorded for this heap).
fn wrapper_proto_for(heap: &Heap, pattern: Value) -> Option<Gc<Object>> {
    match pattern {
        Value::Symbol(_) => heap.symbol_prototype,
        Value::Str(_) | Value::Int(_) | Value::Float(_) | Value::BigInt(_) => {
            WRAPPER_PROTOS.with(|c| {
                let guard = c.borrow();
                let (id, protos) = (*guard).as_ref()?;
                if *id != heap.id() {
                    return None;
                }
                Some(match pattern {
                    Value::Str(_) => protos.string,
                    Value::BigInt(_) => protos.bigint,
                    // `Int` / `Float`: `%Number.prototype%`.
                    _ => protos.number,
                })
            })
        }
        _ => None,
    }
}

/// Spec `GetMethod(V, P)` for the well-known-symbol dispatch used by
/// `match` / `matchAll` / `search`: `None` when the property is absent,
/// `undefined`, or `null`; `TypeError` when present but not callable.
///
/// `GetV` performs the ordinary `[[Get]]`: the first own-or-inherited
/// property supplies the candidate — a data value directly, or an
/// accessor getter invoked with `V` as receiver. The VM's own `[[Get]]`
/// (`Interp::get_obj_prop` in `crate::vm`) is private, so this is the
/// closest correct local equivalent: the same data-then-accessor
/// per-level order and cycle-guarded prototype walk as
/// [`walk_proto_chain`], with the getter invoked through
/// [`call_function`].
///
/// Spec step 1 is `ToObject(V)`: for a primitive pattern the walk starts
/// at the corresponding wrapper prototype ([`wrapper_proto_for`]) instead
/// of the primitive itself — a fresh wrapper has no own properties, so
/// this is exactly `GetV` on the boxed value. `undefined` / `null` keep
/// the "no hook" behavior (the `RegExpCreate` fallback coerces them), as
/// do primitives whose wrapper prototype is not installed for this heap.
pub(crate) fn get_method(
    heap: &mut Heap,
    pattern: Value,
    wk: WellKnown,
) -> Result<Option<Value>, NativeError> {
    let obj = match pattern {
        Value::Object(o) => o,
        // `undefined` / `null`: no hook (the `RegExpCreate` fallback
        // coerces them). Primitives: box via the wrapper prototype.
        _ => match wrapper_proto_for(heap, pattern) {
            Some(proto) => proto,
            None => return Ok(None),
        },
    };
    // The hook lives under the real well-known symbol key. Symbols are
    // always installed by the time string methods run, so `None` here
    // means "no hook", never "builtins missing".
    let Some(sym) = heap.well_known[wk as usize] else {
        return Ok(None);
    };
    let key = PropKey::Symbol(sym);
    // Ordinary `[[Get]]`: own property first, then the prototype chain;
    // per level, a data property wins over an accessor for the same key
    // (matching the VM's `find_prop_in_chain` order).
    let found = walk_proto_chain(heap, obj, |o| {
        if let Some(p) = heap.get_property_key(o, key) {
            Some(ChainSlot::Data(p.value))
        } else {
            heap.get_accessor_key(o, key)
                .map(|a| ChainSlot::Getter(a.get))
        }
    });
    let func = match found {
        None => return Ok(None),
        Some(ChainSlot::Data(v)) => v,
        Some(ChainSlot::Getter(get)) => {
            // Spec `OrdinaryGet` for an accessor: no getter means
            // `undefined`; a non-callable getter throws `TypeError`.
            if get.is_undefined() {
                return Ok(None);
            }
            if !is_callable(heap, get) {
                return Err(NativeError::TypeError(format!(
                    "String method: getter for {key:?} is not callable"
                )));
            }
            // `GetV` passes the original value as receiver.
            call_function(heap, get, pattern, &[])?
        }
    };
    match func {
        Value::Undefined | Value::Null => Ok(None),
        f if is_callable(heap, f) => Ok(Some(f)),
        _ => Err(NativeError::TypeError(format!(
            "String method: property {key:?} is not callable"
        ))),
    }
}

/// `get_prop` under a real well-known symbol key. `None` when the symbol
/// is not installed yet (string methods only run after builtins are).
fn well_known_get(heap: &Heap, o: Gc<Object>, wk: WellKnown) -> Option<Value> {
    let sym = heap.well_known[wk as usize]?;
    crate::builtin_object_array::get_prop_key(heap, o, PropKey::Symbol(sym))
}

/// Spec 22.1.3.15 `String.prototype.match`: `GetMethod(pattern, @@match)`
/// and `Call(matcher, pattern, « S »)` when it is callable, otherwise
/// `RegExpCreate(pattern, undefined)` and `@@match` on the result (so
/// `"ab".match("b")` coerces the string to a regexp instead of throwing).
fn string_match(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let s = this_str(heap, this)?;
    let pattern = arg(args, 0);
    if let Some(matcher) = get_method(heap, pattern, WellKnown::Match)? {
        let s_val = Value::Str(heap.alloc_str_wtf8(&s)?);
        return call_function(heap, matcher, pattern, &[s_val]);
    }
    let rx = crate::builtin_regexp::regexp_construct(heap, Value::Undefined, &[pattern])?;
    let s_val = Value::Str(heap.alloc_str_wtf8(&s)?);
    crate::builtin_regexp::regexp_symbol_match(heap, rx, &[s_val])
}

/// Spec 22.1.3.16 `String.prototype.matchAll`, minus the iterator: when
/// `GetMethod(pattern, @@matchAll)` is callable it is invoked; otherwise
/// the pattern is coerced with `RegExpCreate` and the regexp worker runs.
/// v1 returns an `Array` of match arrays instead of a `RegExpStringIterator`
/// (no iterator protocol yet). Note the step order: a regexp pattern
/// inherits `@@matchAll` from `%RegExp.prototype%` (literals get the
/// prototype since the finding-4 fix), so step 3b dispatches the hook and
/// a non-global regexp *throws* `TypeError` there — step 4's
/// `RegExpCreate(pattern, "g")` never runs for it. The `"g"` in the
/// fallback only coerces hook-less patterns (strings, plain objects).
fn string_match_all(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let s = this_str(heap, this)?;
    let pattern = arg(args, 0);
    if let Some(matcher) = get_method(heap, pattern, WellKnown::MatchAll)? {
        let s_val = Value::Str(heap.alloc_str_wtf8(&s)?);
        return call_function(heap, matcher, pattern, &[s_val]);
    }
    // Spec 22.1.3.16 step 4: `RegExpCreate(pattern, "g")`. A string
    // pattern (no `@@matchAll` hook, so step 3b does not fire) coerces to
    // a global regexp and matches instead of throwing.
    let flags = Value::Str(heap.alloc_str("g")?);
    let rx = crate::builtin_regexp::regexp_construct(heap, Value::Undefined, &[pattern, flags])?;
    let s_val = Value::Str(heap.alloc_str_wtf8(&s)?);
    crate::builtin_regexp::regexp_symbol_match_all(heap, rx, &[s_val])
}

/// Spec 22.1.3.25 `String.prototype.search`: `GetMethod(pattern, @@search)`
/// and `Call(searcher, pattern, « S »)` when callable, otherwise
/// `RegExpCreate(pattern, undefined)` and `@@search` on the result. Returns
/// the code-unit index of the first match, or `-1`.
fn string_search(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let s = this_str(heap, this)?;
    let pattern = arg(args, 0);
    if let Some(searcher) = get_method(heap, pattern, WellKnown::Search)? {
        let s_val = Value::Str(heap.alloc_str_wtf8(&s)?);
        return call_function(heap, searcher, pattern, &[s_val]);
    }
    let rx = crate::builtin_regexp::regexp_construct(heap, Value::Undefined, &[pattern])?;
    let s_val = Value::Str(heap.alloc_str_wtf8(&s)?);
    crate::builtin_regexp::regexp_symbol_search(heap, rx, &[s_val])
}

fn string_from_char_code(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let mut units: Vec<u16> = Vec::with_capacity(args.len());
    for a in args {
        units.push((js_to_uint32(*a)? % 65536) as u16);
    }
    str_val(heap, &units_to_string(&units))
}

fn string_from_code_point(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let mut out = Wtf8Buf::new();
    for a in args {
        let n = js_to_number(*a)?;
        if !(0.0..=1114111.0).contains(&n) || n.fract() != 0.0 {
            return Err(NativeError::RangeError(format!(
                "{n} is not a valid code point"
            )));
        }
        // 0xD800..0xE000 are not valid scalar values; char::from_u32
        // rejects them, and the spec throws RangeError for them too.
        match char::from_u32(n as u32) {
            Some(c) => out.push_char(c),
            None => {
                return Err(NativeError::RangeError(format!(
                    "{n} is not a valid code point"
                )));
            }
        }
    }
    str_val(heap, &out)
}

/// `String.raw(callSite, ...substitutions)`.
fn string_raw(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let call_site = arg(args, 0);
    let raw_obj = match call_site {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "String.raw requires a template object".to_string(),
            ));
        }
    };
    let raw = match prop_get(raw_obj, Wtf8::from_str("raw")) {
        Some(Value::Object(o)) => o,
        _ => {
            return Err(NativeError::TypeError(
                "String.raw requires a template object with a raw array".to_string(),
            ));
        }
    };
    let len = array_length(raw).min(1_000_000);
    let mut out = Wtf8Buf::new();
    for i in 0..len {
        if let Some(v) = prop_get(raw, Wtf8::from_str(&i.to_string())) {
            out.push_wtf8(&js_to_string_rust(v)?);
        }
        if i + 1 < len && (i as usize) < args.len().saturating_sub(1) {
            out.push_wtf8(&js_to_string_rust(arg(args, i as usize + 1))?);
        }
    }
    str_val(heap, &out)
}

// ---------------------------------------------------------------------------
// Number
// ---------------------------------------------------------------------------

fn number_construct(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    // v1: call form only; `new` needs new.target from the interpreter.
    match arg(args, 0) {
        Value::Undefined => Ok(js_num(0.0)),
        // Spec 20.1.1.1 / 7.1.4: `Number(1n)` is `1` — the only
        // `ToNumber` caller that converts BigInts instead of throwing.
        Value::BigInt(g) => Ok(js_num(bn_to_f64(&g))),
        // Spec 20.1.1.1 / 7.1.4: `Number(obj)` is
        // `ToNumber(ToPrimitive(obj, number))`, so user `valueOf` /
        // `@@toPrimitive` hooks run with the number hint here, while the
        // unary `+` in the VM runs them with its own hint. (The v1 object
        // path inside `js_to_number` stays hook-free for the other
        // `ToNumber` callers.)
        Value::Object(o) => {
            let prim = to_primitive_with_hint(heap, o, PrimitiveHint::Number)?;
            // The primitive may itself be a BigInt (e.g. via
            // `@@toPrimitive`); its Number value applies here too.
            let n = match prim {
                Value::BigInt(g) => bn_to_f64(&g),
                v => js_to_number(v)?,
            };
            Ok(js_num(n))
        }
        v => Ok(js_num(js_to_number(v)?)),
    }
}

/// The numeric `this` value: primitives directly, `Number` wrappers via
/// the hidden [`PRIM_VALUE`] slot (`new Number()` still returns the
/// primitive in v1, but sloppy `this` boxing creates real wrappers).
fn this_number(heap: &Heap, v: Value) -> Result<f64, NativeError> {
    match unwrap_prim(heap, v) {
        Value::Int(i) => Ok(f64::from(i)),
        Value::Float(f) => Ok(f),
        _ => Err(NativeError::TypeError(
            "Number prototype method called on incompatible receiver".to_string(),
        )),
    }
}

fn number_value_of(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(this_number(heap, this)?))
}

/// `Number.prototype.toString(radix)`. Radix 10 uses the shortest
/// round-trip rendering; other radices use the exact expansion
/// (terminating, or capped at 1100 fraction digits).
fn number_to_string(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let x = this_number(heap, this)?;
    let r = match arg(args, 0) {
        Value::Undefined => 10,
        v => {
            let ri = js_to_integer_or_infinity(v)?;
            if !(2.0..=36.0).contains(&ri) {
                return Err(NativeError::RangeError(
                    "toString() radix argument must be between 2 and 36".to_string(),
                ));
            }
            ri as u32
        }
    };
    if r == 10 {
        return str_val(heap, Wtf8::from_str(&js_number_to_string(x)));
    }
    if x.is_nan() {
        return str_val(heap, Wtf8::from_str("NaN"));
    }
    if x == 0.0 {
        return str_val(heap, Wtf8::from_str("0"));
    }
    if x.is_infinite() {
        return str_val(
            heap,
            Wtf8::from_str(if x > 0.0 { "Infinity" } else { "-Infinity" }),
        );
    }
    let neg = x < 0.0;
    let ax = x.abs();
    let int_part = bn_from_f64(ax.trunc()).expect("finite trunc converts");
    let mut s = String::with_capacity(32);
    if neg {
        // Sign first: avoids the O(n) shift of `insert(0, '-')`.
        s.push('-');
    }
    s.push_str(&bn_to_string_radix(&int_part, r));
    let mut frac = ax - ax.trunc();
    if frac != 0.0 {
        s.push('.');
        let mut count = 0;
        while frac != 0.0 && count < 1100 {
            frac *= f64::from(r);
            let d = frac.floor();
            s.push(char::from_digit(d as u32, r).unwrap_or('?'));
            frac -= d;
            count += 1;
        }
    }
    str_val(heap, Wtf8::from_str(&s))
}

/// Exact `toFixed`: `round(|x| * 10^f)` computed on the exact binary value
/// via the [`BigInt`] core, ties away from zero (spec: pick the larger `n`).
fn fmt_fixed(x: f64, f: usize) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    let ax = x.abs();
    if ax >= 1e21 {
        return js_number_to_string(x);
    }
    if ax == 0.0 {
        if f == 0 {
            return "0".to_string();
        }
        // One allocation instead of `format!("0.{}", "0".repeat(f))`.
        let mut z = String::with_capacity(f + 2);
        z.push_str("0.");
        z.extend(std::iter::repeat_n('0', f));
        return z;
    }
    let (m, e2) = decompose(x);
    let mut num = BigInt {
        negative: false,
        digits: vec![(m & 0xffff_ffff) as u32, (m >> 32) as u32],
    };
    let mut den = bn_one();
    if e2 >= 0 {
        num = bn_shl(&num, e2 as u64).expect("finite f64 shift fits");
    } else {
        den = bn_shl(&bn_one(), (-e2) as u64).expect("finite f64 shift fits");
    }
    if f > 0 {
        // `f` is the f64 fraction-digit count (bounded), so this cannot hit
        // the `bn_pow` size cap.
        num = bn_mul(
            &num,
            &bn_pow(bn_from_i64(10), f as u32).expect("f64 digits bounded"),
        );
    }
    let (mut q, r) = bn_divmod(&num, &den).expect("denominator is nonzero");
    // Round half up on the exact value.
    if bn_cmp(&bn_add(&r, &r), &den) != Ordering::Less {
        q = bn_add(&q, &bn_one());
    }
    let mut digits = bn_to_string_radix(&q, 10);
    while digits.len() <= f {
        digits.insert(0, '0');
    }
    let body = if f == 0 {
        digits
    } else {
        let (head, tail) = digits.split_at(digits.len() - f);
        format!("{head}.{tail}")
    };
    if x < 0.0 { format!("-{body}") } else { body }
}

/// Exact `toExponential` with `f` fraction digits (or shortest form when
/// `f` is `None`).
fn fmt_exponential(x: f64, f: Option<usize>) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        };
    }
    let ax = x.abs();
    if ax == 0.0 {
        return match f {
            None => "0e+0".to_string(),
            Some(0) => "0e+0".to_string(),
            Some(f) => format!("0.{}e+0", "0".repeat(f)),
        };
    }
    let sign = if x < 0.0 { "-" } else { "" };
    match f {
        None => {
            let (digits, n) = decimal_digits(x);
            let e = n - 1;
            let mantissa = if digits.len() == 1 {
                digits
            } else {
                format!("{}.{}", &digits[..1], &digits[1..])
            };
            format!("{sign}{mantissa}e{e:+}")
        }
        Some(f) => {
            let (_, n) = decimal_digits(x);
            let mut e = n - 1; // == floor(log10(ax)), exact (see decimal_digits)
            let (m, e2) = decompose(x);
            let mut num = BigInt {
                negative: false,
                digits: vec![(m & 0xffff_ffff) as u32, (m >> 32) as u32],
            };
            let mut den = bn_one();
            if e2 >= 0 {
                num = bn_shl(&num, e2 as u64).expect("finite f64 shift fits");
            } else {
                den = bn_shl(&bn_one(), (-e2) as u64).expect("finite f64 shift fits");
            }
            // q = round(ax * 10^(f-e))
            let p = f as i64 - e as i64;
            // `|p|` is bounded by the f64 decimal-exponent range, far below
            // the `bn_pow` size cap.
            if p >= 0 {
                num = bn_mul(
                    &num,
                    &bn_pow(bn_from_i64(10), p as u32).expect("f64 exponent bounded"),
                );
            } else {
                den = bn_mul(
                    &den,
                    &bn_pow(bn_from_i64(10), (-p) as u32).expect("f64 exponent bounded"),
                );
            }
            let (mut q, r) = bn_divmod(&num, &den).expect("denominator is nonzero");
            if bn_cmp(&bn_add(&r, &r), &den) != Ordering::Less {
                q = bn_add(&q, &bn_one());
            }
            // Rounding may carry into 10^(f+1): renormalize (`f` bounded, so
            // the `bn_pow` cap cannot trigger).
            if bn_cmp(
                &q,
                &bn_pow(bn_from_i64(10), f as u32 + 1).expect("f64 digits bounded"),
            ) == Ordering::Equal
            {
                q = bn_pow(bn_from_i64(10), f as u32).expect("f64 digits bounded");
                e += 1;
            }
            let mut dq = bn_to_string_radix(&q, 10);
            while dq.len() < f + 1 {
                dq.insert(0, '0');
            }
            let mantissa = if f == 0 {
                dq
            } else {
                format!("{}.{}", &dq[..1], &dq[1..])
            };
            format!("{sign}{mantissa}e{e:+}")
        }
    }
}

fn digits_arg(args: &[Value], method: &str, lo: f64, hi: f64) -> Result<usize, NativeError> {
    let d = js_to_integer_or_infinity(arg(args, 0))?;
    if !(lo..=hi).contains(&d) {
        return Err(NativeError::RangeError(format!(
            "{method}() digits argument must be between {} and {}",
            lo as i32, hi as i32
        )));
    }
    Ok(d as usize)
}

fn number_to_fixed(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let x = this_number(heap, this)?;
    let f = digits_arg(args, "toFixed", 0.0, 100.0)?;
    str_val(heap, Wtf8::from_str(&fmt_fixed(x, f)))
}

fn number_to_exponential(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let x = this_number(heap, this)?;
    let f = match arg(args, 0) {
        Value::Undefined => None,
        _ => Some(digits_arg(args, "toExponential", 0.0, 100.0)?),
    };
    str_val(heap, Wtf8::from_str(&fmt_exponential(x, f)))
}

fn number_to_precision(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let x = this_number(heap, this)?;
    let p = digits_arg(args, "toPrecision", 1.0, 100.0)?;
    if x == 0.0 {
        return str_val(
            heap,
            Wtf8::from_str(
                &(if p == 1 {
                    "0".to_string()
                } else {
                    // One allocation instead of `format!("0.{}", "0".repeat(..))`.
                    let mut z = String::with_capacity(p + 1);
                    z.push_str("0.");
                    z.extend(std::iter::repeat_n('0', p - 1));
                    z
                }),
            ),
        );
    }
    if x.is_nan() || x.is_infinite() {
        return str_val(heap, Wtf8::from_str(&js_number_to_string(x)));
    }
    let (_, n) = decimal_digits(x);
    let e = n - 1;
    if e < -6 || e >= p as i32 {
        str_val(heap, Wtf8::from_str(&fmt_exponential(x, Some(p - 1))))
    } else {
        str_val(
            heap,
            Wtf8::from_str(&fmt_fixed(x, (p as i32 - e - 1) as usize)),
        )
    }
}

/// `parseInt(string, radix)`: spec grammar incl. `0x` detection, no legacy
/// octal, longest valid digit prefix.
fn number_parse_int(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let input = js_to_string_rust(arg(args, 0))?;
    let mut s = trim_js_wtf8(&input);
    let mut sign = 1.0;
    if let Some(rest) = s.strip_ascii_prefix(b"-") {
        sign = -1.0;
        s = rest;
    } else if let Some(rest) = s.strip_ascii_prefix(b"+") {
        s = rest;
    }
    let radix_mv = js_to_integer_or_infinity(arg(args, 1))?;
    let r = if radix_mv == 0.0 { 10.0 } else { radix_mv };
    if !(2.0..=36.0).contains(&r) {
        return Ok(js_num(f64::NAN));
    }
    let mut r = r as u32;
    // Strip 0x/0X when radix was 0/undefined or is 16. A 0x prefix with an
    // omitted radix switches to base 16 per spec.
    if (radix_mv == 0.0 || r == 16)
        && let Some(hex) = s
            .strip_ascii_prefix(b"0x")
            .or_else(|| s.strip_ascii_prefix(b"0X"))
    {
        s = hex;
        if radix_mv == 0.0 {
            r = 16;
        }
    }
    let mut v = 0.0f64;
    let mut any = false;
    let rf = f64::from(r);
    // `char::to_digit` only accepts ASCII, so scanning bytes is exactly
    // equivalent and skips UTF-8 decoding.
    for &b in s.as_bytes() {
        match (b as char).to_digit(r) {
            Some(d) => {
                v = v * rf + f64::from(d);
                any = true;
            }
            None => break,
        }
    }
    if !any {
        return Ok(js_num(f64::NAN));
    }
    Ok(js_num(sign * v))
}

/// `parseFloat(string)`: longest decimal-literal/`Infinity` prefix after
/// leading whitespace and an optional sign.
fn number_parse_float(
    _heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let input = js_to_string_rust(arg(args, 0))?;
    let t = trim_js_start_wtf8(&input);
    let (negative, rest) = match t.strip_ascii_prefix(b"-") {
        Some(r) => (true, r),
        None => (false, t.strip_ascii_prefix(b"+").unwrap_or(t)),
    };
    if rest.as_bytes().starts_with(b"Infinity") {
        return Ok(js_num(if negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        }));
    }
    // Longest prefix matching DecimalLiteral.
    let b = rest.as_bytes();
    let mut i = 0;
    let n = b.len();
    let mut digits = 0;
    while i < n && b[i].is_ascii_digit() {
        i += 1;
        digits += 1;
    }
    if i < n && b[i] == b'.' {
        i += 1;
        while i < n && b[i].is_ascii_digit() {
            i += 1;
            digits += 1;
        }
    }
    if digits > 0 && i < n && (b[i] == b'e' || b[i] == b'E') {
        let mut j = i + 1;
        if j < n && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        let k = j;
        while j < n && b[j].is_ascii_digit() {
            j += 1;
        }
        if j > k {
            i = j;
        }
    }
    if digits == 0 {
        return Ok(js_num(f64::NAN));
    }
    let mut v: f64 = std::str::from_utf8(&rest.as_bytes()[..i])
        .unwrap_or("")
        .parse()
        .unwrap_or(f64::NAN);
    if negative {
        v = -v;
    }
    Ok(js_num(v))
}

fn number_is_finite(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(Value::Bool(match arg(args, 0) {
        Value::Int(_) => true,
        Value::Float(f) => f.is_finite(),
        _ => false,
    }))
}

fn number_is_nan(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(Value::Bool(
        matches!(arg(args, 0), Value::Float(f) if f.is_nan()),
    ))
}

/// Global `isNaN(x)`: unlike `Number.isNaN`, coerces via `ToNumber` first.
fn global_is_nan(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(Value::Bool(js_to_number(arg(args, 0))?.is_nan()))
}

/// Global `isFinite(x)`: unlike `Number.isFinite`, coerces via `ToNumber` first.
fn global_is_finite(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(Value::Bool(js_to_number(arg(args, 0))?.is_finite()))
}

fn number_is_integer(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(Value::Bool(match arg(args, 0) {
        Value::Int(_) => true,
        Value::Float(f) => f.is_finite() && f.fract() == 0.0,
        _ => false,
    }))
}

fn number_is_safe_integer(
    _heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    Ok(Value::Bool(match arg(args, 0) {
        Value::Int(_) => true,
        Value::Float(f) => f.is_finite() && f.fract() == 0.0 && f.abs() <= 9007199254740991.0,
        _ => false,
    }))
}

// ---------------------------------------------------------------------------
// Boolean
// ---------------------------------------------------------------------------

fn boolean_construct(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let b = arg(args, 0).is_truthy();
    // `new Boolean(x)`: wrapper object carrying [[BooleanData]] in the
    // hidden `PRIM_VALUE` slot; the call form returns the primitive.
    if crate::vm::active_new_target().is_some() {
        let proto = heap
            .boolean_prototype
            .map(Value::Object)
            .unwrap_or(Value::Null);
        let obj = heap.alloc_object(Object {
            kind: ObjectKind::Ordinary,
            prototype: proto,
            ..Default::default()
        })?;
        heap.define_property_str(obj, PRIM_VALUE, Property::data(Value::Bool(b)))?;
        Ok(Value::Object(obj))
    } else {
        Ok(Value::Bool(b))
    }
}

/// The `this` value as a boolean: primitives directly, `Boolean` wrappers
/// via the hidden [`PRIM_VALUE`] slot.
fn this_boolean(heap: &Heap, v: Value) -> Result<bool, NativeError> {
    match unwrap_prim(heap, v) {
        Value::Bool(b) => Ok(b),
        _ => Err(NativeError::TypeError(
            "Boolean prototype method called on incompatible receiver".to_string(),
        )),
    }
}

fn boolean_to_string(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    str_val(
        heap,
        Wtf8::from_str(if this_boolean(heap, this)? {
            "true"
        } else {
            "false"
        }),
    )
}

fn boolean_value_of(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    Ok(Value::Bool(this_boolean(heap, this)?))
}

// ---------------------------------------------------------------------------
// BigInt
// ---------------------------------------------------------------------------

/// Copy a heap bigint's parts out (the core works on owned values).
fn bigint_parts(g: Gc<BigInt>) -> BigInt {
    BigInt {
        negative: g.negative,
        digits: g.digits.clone(),
    }
}

fn bigint_arg(v: Value, which: &str) -> Result<BigInt, NativeError> {
    match v {
        Value::BigInt(g) => Ok(bigint_parts(g)),
        _ => Err(NativeError::TypeError(format!(
            "Cannot mix BigInt and other types: {which} operand is not a BigInt"
        ))),
    }
}

fn alloc_bigint_val(heap: &mut Heap, b: BigInt) -> Result<Value, NativeError> {
    Ok(Value::BigInt(heap.alloc_bigint(b.negative, b.digits)?))
}

fn bigint_construct(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    // v1: call form only; `new BigInt()` should throw but new.target is not
    // passed to NativeFn yet (documented at module top).
    let v = arg(args, 0);
    // Spec 20.2.1.1 / 7.1.6: `BigInt(value)` is `ToBigInt(ToPrimitive(value,
    // number))`, so user `valueOf` / `@@toPrimitive` hooks run with the
    // number hint for objects.
    if let Value::Object(o) = v {
        let prim = to_primitive_with_hint(heap, o, PrimitiveHint::Number)?;
        return bigint_from_primitive(heap, prim);
    }
    bigint_from_primitive(heap, v)
}

/// `BigInt(v)` for a primitive `v` (spec `ToBigInt`).
fn bigint_from_primitive(heap: &mut Heap, v: Value) -> Result<Value, NativeError> {
    let b = match v {
        Value::BigInt(g) => return Ok(Value::BigInt(g)),
        Value::Bool(true) => bn_one(),
        Value::Bool(false) => bn_zero(),
        Value::Int(i) => bn_from_i64(i64::from(i)),
        Value::Float(f) => {
            if !f.is_finite() || f.fract() != 0.0 {
                return Err(NativeError::RangeError(format!(
                    "The number {f} cannot be converted to a BigInt because it is not an integer"
                )));
            }
            bn_from_f64(f).expect("finite integer converts")
        }
        Value::Str(g) => match bn_parse(g.as_wtf8()) {
            Some(b) => b,
            None => {
                return Err(NativeError::SyntaxError(format!(
                    "Cannot convert {} to a BigInt",
                    g.as_wtf8()
                )));
            }
        },
        Value::Null => {
            return Err(NativeError::TypeError(
                "Cannot convert null to a BigInt".to_string(),
            ));
        }
        Value::Undefined => {
            return Err(NativeError::TypeError(
                "Cannot convert undefined to a BigInt".to_string(),
            ));
        }
        Value::Symbol(_) => {
            return Err(NativeError::TypeError(
                "Cannot convert a Symbol value to a BigInt".to_string(),
            ));
        }
        // Unreachable: `to_primitive_with_hint` only yields primitives.
        Value::Object(_) => {
            return Err(NativeError::TypeError(
                "Cannot convert object to a BigInt".to_string(),
            ));
        }
    };
    alloc_bigint_val(heap, b)
}

fn this_bigint(heap: &Heap, v: Value) -> Result<BigInt, NativeError> {
    match unwrap_prim(heap, v) {
        Value::BigInt(g) => Ok(bigint_parts(g)),
        _ => Err(NativeError::TypeError(
            "BigInt prototype method called on incompatible receiver".to_string(),
        )),
    }
}

fn bigint_to_string(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let x = this_bigint(heap, this)?;
    let radix = match arg(args, 0) {
        Value::Undefined => 10,
        v => {
            let r = js_to_integer_or_infinity(v)?;
            if !(2.0..=36.0).contains(&r) {
                return Err(NativeError::RangeError(
                    "toString() radix argument must be between 2 and 36".to_string(),
                ));
            }
            r as u32
        }
    };
    str_val(heap, Wtf8::from_str(&bn_to_string_radix(&x, radix)))
}

fn bigint_value_of(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    alloc_bigint_val(heap, this_bigint(heap, this)?)
}

macro_rules! bigint_binop {
    ($name:ident, $op:expr, $zero_msg:literal) => {
        // (Awaiting BigInt literal wiring: the generated BigInt.prototype
        // builtins are not registered yet; the allow keeps clippy green.)
        #[allow(dead_code)]
        fn $name(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
            let a = bigint_arg(arg(args, 0), "left")?;
            let b = bigint_arg(arg(args, 1), "right")?;
            match $op(&a, &b) {
                Some(r) => alloc_bigint_val(heap, r),
                None => Err(NativeError::RangeError($zero_msg.to_string())),
            }
        }
    };
}

bigint_binop!(bigint_add, |a: &BigInt, b: &BigInt| Some(bn_add(a, b)), "");
bigint_binop!(bigint_sub, |a: &BigInt, b: &BigInt| Some(bn_sub(a, b)), "");
bigint_binop!(bigint_mul, |a: &BigInt, b: &BigInt| Some(bn_mul(a, b)), "");
bigint_binop!(
    bigint_div,
    |a: &BigInt, b: &BigInt| bn_divmod(a, b).map(|(q, _)| q),
    "Division by zero"
);
bigint_binop!(
    bigint_mod,
    |a: &BigInt, b: &BigInt| bn_divmod(a, b).map(|(_, r)| r),
    "Division by zero"
);

// (Awaiting BigInt literal wiring: not registered as a builtin yet;
// the allow keeps `cargo clippy -- -D warnings` green until then.)
#[allow(dead_code)]
fn bigint_pow(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let base = bigint_arg(arg(args, 0), "base")?;
    let exp = bigint_arg(arg(args, 1), "exponent")?;
    if exp.negative {
        return Err(NativeError::RangeError(
            "Exponent must be non-negative".to_string(),
        ));
    }
    if bn_bit_len(&exp) > 32 {
        return Err(NativeError::RangeError("Exponent too large".to_string()));
    }
    let e = exp.digits.first().copied().unwrap_or(0);
    let r = bn_pow(base, e)
        .ok_or_else(|| NativeError::RangeError("BigInt result too large".to_string()))?;
    alloc_bigint_val(heap, r)
}

// (Awaiting BigInt literal wiring: not registered as a builtin yet;
// the allow keeps `cargo clippy -- -D warnings` green until then.)
#[allow(dead_code)]
fn bigint_shl(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let a = bigint_arg(arg(args, 0), "left")?;
    let b = bigint_arg(arg(args, 1), "shift count")?;
    if b.negative {
        return Err(NativeError::RangeError(
            "BigInt shift count must be non-negative".to_string(),
        ));
    }
    let bits = shift_amount(&b)?;
    match bn_shl(&a, bits) {
        Some(r) => alloc_bigint_val(heap, r),
        None => Err(NativeError::RangeError(
            "BigInt shift count too large".to_string(),
        )),
    }
}

// (Awaiting BigInt literal wiring: not registered as a builtin yet;
// the allow keeps `cargo clippy -- -D warnings` green until then.)
#[allow(dead_code)]
fn bigint_shr(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let a = bigint_arg(arg(args, 0), "left")?;
    let b = bigint_arg(arg(args, 1), "shift count")?;
    if b.negative {
        return Err(NativeError::RangeError(
            "BigInt shift count must be non-negative".to_string(),
        ));
    }
    let bits = shift_amount(&b)?;
    alloc_bigint_val(heap, bn_shr(&a, bits))
}

/// A non-negative bigint as u64, rejecting absurd magnitudes.
// (Awaiting BigInt literal wiring: only called from the not-yet-registered
// shift builtins; the allow keeps `cargo clippy -- -D warnings` green.)
#[allow(dead_code)]
fn shift_amount(b: &BigInt) -> Result<u64, NativeError> {
    if bn_bit_len(b) > 64 {
        return Err(NativeError::RangeError(
            "BigInt shift count too large".to_string(),
        ));
    }
    let mut v: u64 = 0;
    for (i, &w) in b.digits.iter().enumerate() {
        v |= u64::from(w) << (i * 32);
    }
    Ok(v)
}

macro_rules! bigint_bitop {
    ($name:ident, $op:expr) => {
        // (Awaiting BigInt literal wiring: the generated BigInt.prototype
        // builtins are not registered yet; the allow keeps clippy green.)
        #[allow(dead_code)]
        fn $name(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
            let a = bigint_arg(arg(args, 0), "left")?;
            let b = bigint_arg(arg(args, 1), "right")?;
            alloc_bigint_val(heap, bn_bitop(&a, &b, $op))
        }
    };
}

bigint_bitop!(bigint_bitand, |a, b| a & b);
bigint_bitop!(bigint_bitor, |a, b| a | b);
bigint_bitop!(bigint_bitxor, |a, b| a ^ b);

// (Awaiting BigInt literal wiring: not registered as a builtin yet;
// the allow keeps `cargo clippy -- -D warnings` green until then.)
#[allow(dead_code)]
fn bigint_bitnot(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    alloc_bigint_val(heap, bn_bitnot(&bigint_arg(arg(args, 0), "operand")?))
}

// (Awaiting BigInt literal wiring: not registered as a builtin yet;
// the allow keeps `cargo clippy -- -D warnings` green until then.)
#[allow(dead_code)]
fn bigint_neg(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    alloc_bigint_val(heap, bn_neg(&bigint_arg(arg(args, 0), "operand")?))
}

/// Shared `asIntN`/`asUintN`.
fn bigint_as_int_n_impl(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
    signed: bool,
) -> Result<Value, NativeError> {
    // ToIndex(bits): integer in [0, 2^53 - 1].
    let n = js_to_integer_or_infinity(arg(args, 0))?;
    if n < 0.0 || n > 9007199254740991.0 {
        return Err(NativeError::RangeError(
            "asIntN/asUintN bit count out of range".to_string(),
        ));
    }
    let bits = n as u64;
    let x = bigint_arg(arg(args, 1), "value")?;
    if bits == 0 {
        return alloc_bigint_val(heap, bn_zero());
    }
    if bits > (1 << 24) {
        // 2^bits would exceed ~2 MiB of digits; cap for sanity.
        return Err(NativeError::RangeError(
            "asIntN/asUintN bit count too large".to_string(),
        ));
    }
    let modulo = bn_shl(&bn_one(), bits).expect("capped shift fits");
    let (_, mut r) = bn_divmod(&x, &modulo).expect("nonzero divisor");
    if r.negative {
        r = bn_add(&r, &modulo);
    }
    if signed {
        let half = bn_shr(&modulo, 1); // 2^(bits-1)
        if bn_cmp(&r, &half) != Ordering::Less {
            r = bn_sub(&r, &modulo);
        }
    }
    alloc_bigint_val(heap, r)
}

fn bigint_as_int_n(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    bigint_as_int_n_impl(_heap, _this, args, true)
}

fn bigint_as_uint_n(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    bigint_as_int_n_impl(_heap, _this, args, false)
}

// ---------------------------------------------------------------------------
// Math
// ---------------------------------------------------------------------------

fn math_abs(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.abs()))
}

fn math_acos(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.acos()))
}

fn math_acosh(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.acosh()))
}

fn math_asin(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.asin()))
}

fn math_asinh(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.asinh()))
}

fn math_atan(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.atan()))
}

fn math_atanh(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.atanh()))
}

fn math_atan2(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let y = js_to_number(arg(args, 0))?;
    let x = js_to_number(arg(args, 1))?;
    Ok(js_num(y.atan2(x)))
}

fn math_cbrt(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.cbrt()))
}

fn math_ceil(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.ceil()))
}

fn math_clz32(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(f64::from(
        js_to_uint32(arg(args, 0))?.leading_zeros(),
    )))
}

fn math_cos(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.cos()))
}

fn math_cosh(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.cosh()))
}

fn math_exp(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.exp()))
}

fn math_expm1(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.exp_m1()))
}

fn math_floor(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.floor()))
}

fn math_fround(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(f64::from(js_to_number(arg(args, 0))? as f32)))
}

fn math_hypot(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut acc = 0.0f64;
    for a in args {
        acc = acc.hypot(js_to_number(*a)?);
    }
    Ok(js_num(acc))
}

fn math_imul(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let a = js_to_int32(arg(args, 0))?;
    let b = js_to_int32(arg(args, 1))?;
    Ok(js_num(f64::from(a.wrapping_mul(b))))
}

fn math_log(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.ln()))
}

fn math_log1p(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.ln_1p()))
}

fn math_log2(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.log2()))
}

fn math_log10(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.log10()))
}

fn math_max(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut m = f64::NEG_INFINITY;
    for a in args {
        let x = js_to_number(*a)?;
        if x.is_nan() {
            return Ok(js_num(f64::NAN));
        }
        if x > m || (x == 0.0 && m == 0.0 && !x.is_sign_negative()) {
            m = x; // +0 beats -0
        }
    }
    Ok(js_num(m))
}

fn math_min(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut m = f64::INFINITY;
    for a in args {
        let x = js_to_number(*a)?;
        if x.is_nan() {
            return Ok(js_num(f64::NAN));
        }
        if x < m || (x == 0.0 && m == 0.0 && x.is_sign_negative()) {
            m = x; // -0 beats +0
        }
    }
    Ok(js_num(m))
}

fn math_pow(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let x = js_to_number(arg(args, 0))?;
    let y = js_to_number(arg(args, 1))?;
    Ok(js_num(x.powf(y)))
}

/// xorshift64* state for `Math.random`. Seeded once from the system clock.
/// NOT cryptographic; NOT seedable (spec leaves the algorithm
/// implementation-defined).
static RNG_STATE: AtomicU64 = AtomicU64::new(0);

fn math_random(_heap: &mut Heap, _this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let mut s = RNG_STATE.load(AtomicOrd::Relaxed);
    if s == 0 {
        s = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E3779B97F4A7C15);
        // xorshift64* requires nonzero state.
        if s == 0 {
            s = 0x9E3779B97F4A7C15;
        }
    }
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    RNG_STATE.store(s, AtomicOrd::Relaxed);
    let r = s.wrapping_mul(0x2545F4914F6CDD1D);
    Ok(Value::Float((r >> 11) as f64 / (1u64 << 53) as f64))
}

fn math_round(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let x = js_to_number(arg(args, 0))?;
    if x.is_nan() || x.is_infinite() || x == 0.0 {
        return Ok(js_num(x)); // preserves -0 and NaN
    }
    if x > 0.0 {
        Ok(js_num((x + 0.5).floor()))
    } else if x >= -0.5 {
        Ok(js_num(-0.0))
    } else {
        Ok(js_num((x + 0.5).ceil()))
    }
}

fn math_sign(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.signum()))
}

fn math_sin(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.sin()))
}

fn math_sinh(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.sinh()))
}

fn math_sqrt(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.sqrt()))
}

fn math_tan(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.tan()))
}

fn math_tanh(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.tanh()))
}

fn math_trunc(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(js_to_number(arg(args, 0))?.trunc()))
}

// ---------------------------------------------------------------------------
// JSON: hand-written recursive-descent parser (no serde) and a spec-shaped
// serializer with replacer support, toJSON hooks (native fns only), indent,
// and circular-structure detection.
// ---------------------------------------------------------------------------

/// Max nesting for both parse and stringify (stack-safety guard).
const JSON_MAX_DEPTH: u32 = 1000;

/// Cap on serialized array length (hostile `"length"` guard).
const JSON_ARRAY_CAP: u64 = 1 << 24;

fn json_syntax(msg: &str) -> NativeError {
    NativeError::SyntaxError(format!("JSON.parse: {msg}"))
}

struct JsonParser<'a> {
    s: &'a Wtf8,
    b: &'a [u8],
    pos: usize,
    depth: u32,
}

fn hex_val(d: u8) -> Option<u32> {
    match d {
        b'0'..=b'9' => Some(u32::from(d - b'0')),
        b'a'..=b'f' => Some(u32::from(d - b'a') + 10),
        b'A'..=b'F' => Some(u32::from(d - b'A') + 10),
        _ => None,
    }
}

impl<'a> JsonParser<'a> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    /// Decode the code point at the current byte position, advancing past
    /// it. Returns `(code point, byte length)`; lone surrogates decode to
    /// `0xD800..=0xDFFF`.
    fn next_code_point(&mut self) -> Option<(u32, usize)> {
        let (cp, len) = self.s.code_point_at(self.pos)?;
        self.pos += len;
        Some((cp, len))
    }

    fn parse_hex4(&mut self) -> Result<u32, NativeError> {
        let mut v = 0u32;
        for _ in 0..4 {
            let d = self
                .peek()
                .and_then(hex_val)
                .ok_or_else(|| json_syntax("bad \\u escape"))?;
            v = v * 16 + d;
            self.pos += 1;
        }
        Ok(v)
    }

    /// If the upcoming input is `\uHHHH`, return `HHHH` without consuming.
    fn peek_hex4(&self) -> Option<u32> {
        let b = self.b.get(self.pos..self.pos + 6)?;
        if b[0] != b'\\' || b[1] != b'u' {
            return None;
        }
        let mut v = 0u32;
        for &d in &b[2..6] {
            v = v * 16 + hex_val(d)?;
        }
        Some(v)
    }

    fn parse_string(&mut self) -> Result<Wtf8Buf, NativeError> {
        debug_assert_eq!(self.peek(), Some(b'"'));
        self.pos += 1;
        let mut out = Wtf8Buf::new();
        loop {
            let (cp, _) = self
                .next_code_point()
                .ok_or_else(|| json_syntax("unterminated string"))?;
            // Push a decoded code point: lone surrogates are preserved as
            // code units, never replaced.
            let push_cp = |cp: u32, out: &mut Wtf8Buf| {
                if (0xd800..0xe000).contains(&cp) {
                    out.push_unit(cp as u16);
                } else {
                    out.push_char(char::from_u32(cp).expect("non-surrogate decodes"));
                }
            };
            match cp {
                0x22 => return Ok(out), // '"'
                0x5c => {
                    // '\\'
                    let (e, _) = self
                        .next_code_point()
                        .ok_or_else(|| json_syntax("unterminated escape"))?;
                    match e {
                        0x22 => out.push_char('"'),
                        0x5c => out.push_char('\\'),
                        0x2f => out.push_char('/'),
                        0x62 => out.push_char('\u{8}'),
                        0x66 => out.push_char('\u{c}'),
                        0x6e => out.push_char('\n'),
                        0x72 => out.push_char('\r'),
                        0x74 => out.push_char('\t'),
                        0x75 => {
                            // 'u'
                            let h = self.parse_hex4()?;
                            if (0xD800..0xDC00).contains(&h)
                                && let Some(lo) = self.peek_hex4()
                                && (0xDC00..0xE000).contains(&lo)
                            {
                                self.pos += 6;
                                let cp = 0x10000 + ((h - 0xD800) << 10) + (lo - 0xDC00);
                                out.push_char(char::from_u32(cp).expect("astral scalar"));
                                continue;
                            }
                            // Lone `\u` surrogates are preserved, per spec.
                            push_cp(h, &mut out);
                        }
                        _ => return Err(json_syntax("bad escape")),
                    }
                }
                cp if cp < 0x20 => {
                    return Err(json_syntax("unescaped control character"));
                }
                cp => push_cp(cp, &mut out),
            }
        }
    }

    fn parse_number(&mut self) -> Result<Value, NativeError> {
        let start = self.pos;
        let b = self.b;
        let mut i = self.pos;
        if b[i] == b'-' {
            i += 1;
        }
        if i < b.len() && b[i] == b'0' {
            i += 1;
        } else {
            if i >= b.len() || !b[i].is_ascii_digit() {
                return Err(json_syntax("bad number"));
            }
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
        }
        if i < b.len() && b[i] == b'.' {
            i += 1;
            if i >= b.len() || !b[i].is_ascii_digit() {
                return Err(json_syntax("bad number"));
            }
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
        }
        if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
            i += 1;
            if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                i += 1;
            }
            if i >= b.len() || !b[i].is_ascii_digit() {
                return Err(json_syntax("bad number"));
            }
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
        }
        // Grammar-validated ASCII: the byte slice is valid UTF-8.
        // Rust parses it (overflow -> infinity).
        let text = std::str::from_utf8(&self.b[start..i]).map_err(|_| json_syntax("bad number"))?;
        self.pos = i;
        let v: f64 = text.parse().map_err(|_| json_syntax("bad number"))?;
        Ok(js_num(v))
    }

    fn parse_literal(&mut self, word: &str, v: Value) -> Result<Value, NativeError> {
        if self.b[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(v)
        } else {
            Err(json_syntax("unexpected token"))
        }
    }

    fn parse_value(&mut self, heap: &mut Heap) -> Result<Value, NativeError> {
        self.depth += 1;
        if self.depth > JSON_MAX_DEPTH {
            return Err(json_syntax("too much nesting"));
        }
        let r = match self.peek() {
            Some(b'{') => self.parse_object(heap),
            Some(b'[') => self.parse_array(heap),
            Some(b'"') => Ok(Value::Str(heap.alloc_str_wtf8(&self.parse_string()?)?)),
            Some(b't') => self.parse_literal("true", Value::Bool(true)),
            Some(b'f') => self.parse_literal("false", Value::Bool(false)),
            Some(b'n') => self.parse_literal("null", Value::Null),
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            _ => Err(json_syntax("unexpected token")),
        };
        self.depth -= 1;
        r
    }

    fn parse_array(&mut self, heap: &mut Heap) -> Result<Value, NativeError> {
        self.pos += 1; // [
        let mut elems: Vec<Value> = Vec::new();
        // Elements accumulate in this untraced Rust vec while later elements
        // are parsed; root each one as it arrives so a collection triggered
        // by a later `parse_value` cannot free an earlier element.
        // (`new_array` roots them across its own allocation; this guard
        // covers the parse loop.)
        let mut base = None::<usize>;
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return new_array(heap, elems);
        }
        loop {
            self.skip_ws();
            let v = match self.parse_value(heap) {
                Ok(v) => v,
                Err(e) => {
                    if let Some(b) = base {
                        heap.release_temp(b);
                    }
                    return Err(e);
                }
            };
            base.get_or_insert(heap.protect(v));
            elems.push(v);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => {
                    if let Some(b) = base {
                        heap.release_temp(b);
                    }
                    return Err(json_syntax("expected ',' or ']'"));
                }
            }
        }
        let r = new_array(heap, elems);
        if let Some(b) = base {
            heap.release_temp(b);
        }
        r
    }

    fn parse_object(&mut self, heap: &mut Heap) -> Result<Value, NativeError> {
        self.pos += 1; // {
        let obj = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
        // `obj` is not yet linked into any traced root; hold it across the
        // `parse_value` calls below, whose allocations may collect and free
        // it before the properties are defined. (`define_property` roots
        // `obj` across its own key interning, so the guard only needs to
        // span the value parsing.)
        let base = heap.protect(Value::Object(obj));
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            heap.release_temp(base);
            return Ok(Value::Object(obj));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                heap.release_temp(base);
                return Err(json_syntax("expected string key"));
            }
            let key = match self.parse_string() {
                Ok(k) => k,
                Err(e) => {
                    heap.release_temp(base);
                    return Err(e);
                }
            };
            self.skip_ws();
            if self.peek() != Some(b':') {
                heap.release_temp(base);
                return Err(json_syntax("expected ':'"));
            }
            self.pos += 1;
            self.skip_ws();
            let val = match self.parse_value(heap) {
                Ok(v) => v,
                Err(e) => {
                    heap.release_temp(base);
                    return Err(e);
                }
            };
            // Duplicate keys: last wins, per spec.
            if let Err(e) = heap.define_property(obj, key.as_wtf8(), Property::data(val)) {
                heap.release_temp(base);
                return Err(e.into());
            }
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => {
                    heap.release_temp(base);
                    return Err(json_syntax("expected ',' or '}'"));
                }
            }
        }
        heap.release_temp(base);
        Ok(Value::Object(obj))
    }
}

fn json_parse(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let text = js_to_string_rust(arg(args, 0))?;
    let mut p = JsonParser {
        s: &text,
        b: text.as_bytes(),
        pos: 0,
        depth: 0,
    };
    p.skip_ws();
    let v = p.parse_value(heap)?;
    p.skip_ws();
    if p.pos != p.b.len() {
        return Err(json_syntax("unexpected token"));
    }
    Ok(v)
}

// --- stringify ---

struct StringifyState<'h> {
    heap: &'h mut Heap,
    replacer_fn: Option<NativeFn>,
    prop_list: Option<Vec<Wtf8Buf>>,
    gap: Wtf8Buf,
    stack: Vec<Gc<Object>>,
    depth: u32,
}

/// Quote a JS string as a JSON string literal. Lone surrogates pass
/// through raw, per spec (the output is a JS string, not UTF-8 text).
fn quote_json_str(s: &Wtf8) -> Wtf8Buf {
    let mut out = Wtf8Buf::with_capacity(s.as_bytes().len() + 2);
    out.push_char('"');
    for cp in s.chars_wtf8() {
        // Lone surrogates have no escape: emit the raw code unit.
        if (0xd800..0xe000).contains(&cp) {
            out.push_unit(cp as u16);
            continue;
        }
        let c = char::from_u32(cp).expect("non-surrogate decodes");
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                // Manual `\u00XX`: avoids a `format!` allocation per char.
                let v = c as u32;
                out.push_str("\\u00");
                out.push_char(char::from_digit(v >> 4, 16).unwrap_or('?'));
                out.push_char(char::from_digit(v & 0xF, 16).unwrap_or('?'));
            }
            c => out.push_char(c),
        }
    }
    out.push_char('"');
    out
}

/// `SerializeJSONProperty`: toJSON hook, replacer call, then the value.
fn serialize_prop(
    state: &mut StringifyState,
    key: &Wtf8,
    holder: Gc<Object>,
    indent: &Wtf8,
) -> Result<Option<Wtf8Buf>, NativeError> {
    state.depth += 1;
    if state.depth > JSON_MAX_DEPTH {
        return Err(NativeError::RangeError(
            "JSON.stringify: too much nesting".to_string(),
        ));
    }
    let r = serialize_prop_inner(state, key, holder, indent);
    state.depth -= 1;
    r
}

fn serialize_prop_inner(
    state: &mut StringifyState,
    key: &Wtf8,
    holder: Gc<Object>,
    indent: &Wtf8,
) -> Result<Option<Wtf8Buf>, NativeError> {
    let mut value = prop_get(holder, key).unwrap_or(Value::Undefined);
    // 1. toJSON hook. v1: only native functions are callable from Rust.
    if let Value::Object(o) = value
        && let Some(tp) = prop_get(o, Wtf8::from_str("toJSON"))
        && let Value::Object(f) = tp
        && let Some(func) = native_fn_of(f)
    {
        let k = state.heap.alloc_str_wtf8(key)?;
        value = func(state.heap, value, &[Value::Str(k)])?;
    }
    // 2. replacer function.
    if let Some(rf) = state.replacer_fn {
        let k = state.heap.alloc_str_wtf8(key)?;
        value = rf(state.heap, Value::Object(holder), &[Value::Str(k), value])?;
    }
    serialize_value(state, value, indent)
}

fn serialize_value(
    state: &mut StringifyState,
    value: Value,
    indent: &Wtf8,
) -> Result<Option<Wtf8Buf>, NativeError> {
    match value {
        Value::Undefined => Ok(None),
        Value::Null => Ok(Some(Wtf8Buf::from("null"))),
        Value::Bool(b) => Ok(Some(Wtf8Buf::from(if b { "true" } else { "false" }))),
        Value::Int(i) => Ok(Some(Wtf8Buf::from(&i.to_string()))),
        Value::Float(f) => Ok(Some(if f.is_finite() {
            Wtf8Buf::from(&js_number_to_string(f))
        } else {
            Wtf8Buf::from("null")
        })),
        Value::Str(g) => Ok(Some(quote_json_str(g.as_wtf8()))),
        Value::Symbol(_) => Ok(None),
        Value::BigInt(_) => Err(NativeError::TypeError(
            "Do not know how to serialize a BigInt".to_string(),
        )),
        Value::Object(o) => {
            if matches!(
                o.kind,
                ObjectKind::NativeFunction(_) | ObjectKind::Function(_)
            ) {
                return Ok(None);
            }
            if state.stack.contains(&o) {
                return Err(NativeError::TypeError(
                    "Converting circular structure to JSON".to_string(),
                ));
            }
            state.stack.push(o);
            let r = if matches!(o.kind, ObjectKind::Array) {
                serialize_array(state, o, indent)
            } else {
                serialize_object(state, o, indent)
            };
            state.stack.pop();
            r
        }
    }
}

fn serialize_array(
    state: &mut StringifyState,
    arr: Gc<Object>,
    indent: &Wtf8,
) -> Result<Option<Wtf8Buf>, NativeError> {
    let len = array_length(arr).min(JSON_ARRAY_CAP);
    let gap = state.gap.clone();
    let mut new_indent = Wtf8Buf::new();
    new_indent.push_wtf8(indent);
    new_indent.push_wtf8(&gap);
    let mut out = Wtf8Buf::from("[");
    let mut first = true;
    for i in 0..len {
        if !first {
            out.push_char(',');
        }
        first = false;
        if !gap.is_empty() {
            out.push_char('\n');
            out.push_wtf8(&new_indent);
        }
        // Array elements go through toJSON/replacer too (key = index).
        let key = Wtf8Buf::from(&i.to_string());
        match serialize_prop(state, &key, arr, &new_indent)? {
            Some(s) => out.push_wtf8(&s),
            None => out.push_str("null"),
        }
    }
    if !gap.is_empty() && len > 0 {
        out.push_char('\n');
        out.push_wtf8(indent);
    }
    out.push_char(']');
    Ok(Some(out))
}

fn serialize_object(
    state: &mut StringifyState,
    obj: Gc<Object>,
    indent: &Wtf8,
) -> Result<Option<Wtf8Buf>, NativeError> {
    // Own enumerable keys, optionally filtered by the replacer property list.
    let keys: Vec<Wtf8Buf> = match &state.prop_list {
        Some(list) => {
            let mut ks = Vec::new();
            for k in list {
                if let Some(p) = obj
                    .properties
                    .iter()
                    .find(|(key, _)| key.as_wtf8() == k.as_wtf8())
                    && p.1.enumerable
                {
                    ks.push(k.clone());
                }
            }
            ks
        }
        None => obj
            .properties
            .iter()
            .filter(|(_, p)| p.enumerable)
            .map(|(k, _)| k.as_wtf8().to_owned())
            .collect(),
    };
    let gap = state.gap.clone();
    let mut new_indent = Wtf8Buf::new();
    new_indent.push_wtf8(indent);
    new_indent.push_wtf8(&gap);
    let mut out = Wtf8Buf::from("{");
    let mut first = true;
    for k in &keys {
        let v = match serialize_prop(state, k, obj, &new_indent)? {
            Some(s) => s,
            None => continue,
        };
        if !first {
            out.push_char(',');
        }
        first = false;
        if !gap.is_empty() {
            out.push_char('\n');
            out.push_wtf8(&new_indent);
        }
        out.push_wtf8(&quote_json_str(k));
        out.push_char(':');
        if !gap.is_empty() {
            out.push_char(' ');
        }
        out.push_wtf8(&v);
    }
    if !gap.is_empty() && !first {
        out.push_char('\n');
        out.push_wtf8(indent);
    }
    out.push_char('}');
    Ok(Some(out))
}

fn json_stringify(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let value = arg(args, 0);
    let mut state = StringifyState {
        heap,
        replacer_fn: None,
        prop_list: None,
        gap: Wtf8Buf::new(),
        stack: Vec::new(),
        depth: 0,
    };
    // Replacer: native function, or array of property names.
    if let Value::Object(o) = arg(args, 1) {
        if let Some(f) = native_fn_of(o) {
            state.replacer_fn = Some(f);
        } else if matches!(o.kind, ObjectKind::Array) {
            let len = array_length(o).min(JSON_ARRAY_CAP);
            let mut list = Vec::new();
            for i in 0..len {
                match prop_get(o, Wtf8::from_str(&i.to_string())) {
                    Some(Value::Str(g)) => list.push(g.as_wtf8().to_owned()),
                    Some(Value::Int(n)) => list.push(Wtf8Buf::from(&n.to_string())),
                    Some(Value::Float(f)) if f.is_finite() && f.fract() == 0.0 => {
                        list.push(Wtf8Buf::from(&js_number_to_string(f)));
                    }
                    _ => {}
                }
            }
            state.prop_list = Some(list);
        }
    }
    // Gap: number -> spaces (max 10), string -> first 10 code units.
    state.gap = match arg(args, 2) {
        Value::Int(_) | Value::Float(_) => {
            let n = js_to_integer_or_infinity(arg(args, 2))?.clamp(0.0, 10.0);
            Wtf8Buf::from(&" ".repeat(n as usize))
        }
        // First 10 UTF-16 code units (lone surrogates preserved).
        Value::Str(g) => g.as_wtf8().slice_units(0, 10).into_owned(),
        _ => Wtf8Buf::new(),
    };
    // The spec wraps the top-level value in a holder: { "": value }.
    let holder = state.heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    state
        .heap
        .define_property_str(holder, "", Property::data(value))?;
    let indent = Wtf8Buf::new();
    match serialize_prop(&mut state, Wtf8::from_str(""), holder, &indent)? {
        Some(s) => Ok(Value::Str(state.heap.alloc_wtf8_owned(s)?)),
        None => Ok(Value::Undefined),
    }
}

// ---------------------------------------------------------------------------
// install
// ---------------------------------------------------------------------------

/// Link `proto`'s `[[Prototype]]` to `%Object.prototype%` when it is
/// installed (install order between builtin workers is not fixed).
fn link_to_object_prototype(heap: &mut Heap, proto: Gc<Object>) {
    if let Some(op) = heap.object_prototype {
        heap.set_prototype(proto, Value::Object(op));
    }
}

/// Make `ctor`/`proto` a constructor/prototype pair:
/// `ctor.prototype = proto` ({writable:false, enumerable:false,
/// configurable:false}) and `proto.constructor = ctor`
/// ({writable:true, enumerable:false, configurable:true}).
fn wire_ctor_proto(
    heap: &mut Heap,
    ctor: Gc<Object>,
    proto: Gc<Object>,
) -> Result<(), NativeError> {
    heap.define_property_str(
        ctor,
        "prototype",
        Property {
            value: Value::Object(proto),
            writable: false,
            enumerable: false,
            configurable: false,
        },
    )?;
    heap.define_property_str(
        proto,
        "constructor",
        Property {
            value: Value::Object(ctor),
            writable: true,
            enumerable: false,
            configurable: true,
        },
    )?;
    Ok(())
}

/// Install `String`, `Number`, `BigInt`, `Math`, `JSON`, and the `parseInt` /
/// `parseFloat` globals on `global`.
pub fn install(heap: &mut Heap, global: Gc<Object>) -> Result<(), NativeError> {
    // Root `global` for the whole install: with an embedder-enabled GC
    // threshold, the allocations below could otherwise trigger a
    // collection while the global object (and the intrinsics being linked
    // into it) is still unreachable from any root.
    let base = heap.protect(Value::Object(global));
    let result = install_inner(heap, global);
    heap.release_temp(base);
    result
}

fn install_inner(heap: &mut Heap, global: Gc<Object>) -> Result<(), NativeError> {
    // ---- String ---------------------------------------------------------
    let string_proto = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    // Not reachable from any root until linked into `global` at the end of
    // this section; hold it across the allocations below. (On the error
    // path `install` aborts via `expect`, so the temporary root never
    // outlives the install.)
    let string_base = heap.protect(Value::Object(string_proto));
    link_to_object_prototype(heap, string_proto);
    define_native(heap, string_proto, "charAt", string_char_at, 1)?;
    define_native(heap, string_proto, "charCodeAt", string_char_code_at, 1)?;
    define_native(heap, string_proto, "codePointAt", string_code_point_at, 1)?;
    define_native(heap, string_proto, "concat", string_concat, 1)?;
    define_native(heap, string_proto, "endsWith", string_ends_with, 1)?;
    define_native(heap, string_proto, "includes", string_includes, 1)?;
    define_native(heap, string_proto, "isWellFormed", string_is_well_formed, 0)?;
    define_native(heap, string_proto, "indexOf", string_index_of, 1)?;
    define_native(heap, string_proto, "lastIndexOf", string_last_index_of, 1)?;
    define_native(
        heap,
        string_proto,
        "localeCompare",
        string_locale_compare,
        1,
    )?;
    define_native(heap, string_proto, "match", string_match, 1)?;
    define_native(heap, string_proto, "matchAll", string_match_all, 1)?;
    define_native(heap, string_proto, "normalize", string_normalize, 0)?;
    define_native(heap, string_proto, "padEnd", string_pad_end, 1)?;
    define_native(heap, string_proto, "padStart", string_pad_start, 1)?;
    define_native(heap, string_proto, "repeat", string_repeat, 1)?;
    define_native(heap, string_proto, "replace", string_replace, 2)?;
    define_native(heap, string_proto, "replaceAll", string_replace_all, 2)?;
    define_native(heap, string_proto, "search", string_search, 1)?;
    define_native(heap, string_proto, "slice", string_slice, 2)?;
    define_native(heap, string_proto, "split", string_split, 2)?;
    define_native(heap, string_proto, "startsWith", string_starts_with, 1)?;
    define_native(heap, string_proto, "substr", string_substr, 2)?;
    define_native(heap, string_proto, "substring", string_substring, 2)?;
    define_native(heap, string_proto, "toLowerCase", string_to_lower_case, 0)?;
    define_native(heap, string_proto, "toUpperCase", string_to_upper_case, 0)?;
    define_native(heap, string_proto, "toWellFormed", string_to_well_formed, 0)?;
    define_native(heap, string_proto, "toString", string_value_of, 0)?;
    define_native(heap, string_proto, "valueOf", string_value_of, 0)?;
    define_native(heap, string_proto, "trim", string_trim, 0)?;
    define_native(heap, string_proto, "trimStart", string_trim_start, 0)?;
    define_native(heap, string_proto, "trimEnd", string_trim_end, 0)?;
    let string_ctor = new_native_constructor(heap, "String", string_construct, 1)?;
    define_native(heap, string_ctor, "fromCharCode", string_from_char_code, 1)?;
    define_native(
        heap,
        string_ctor,
        "fromCodePoint",
        string_from_code_point,
        1,
    )?;
    define_native(heap, string_ctor, "raw", string_raw, 1)?;
    wire_ctor_proto(heap, string_ctor, string_proto)?;
    heap.define_property_str(global, "String", Property::data(Value::Object(string_ctor)))?;
    heap.release_temp(string_base);

    // ---- Number ---------------------------------------------------------
    let number_proto = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    let number_base = heap.protect(Value::Object(number_proto));
    link_to_object_prototype(heap, number_proto);
    define_native(
        heap,
        number_proto,
        "toExponential",
        number_to_exponential,
        1,
    )?;
    define_native(heap, number_proto, "toFixed", number_to_fixed, 1)?;
    define_native(heap, number_proto, "toPrecision", number_to_precision, 1)?;
    define_native(heap, number_proto, "toString", number_to_string, 1)?;
    define_native(heap, number_proto, "valueOf", number_value_of, 0)?;
    let number_ctor = new_native_constructor(heap, "Number", number_construct, 1)?;
    define_native(heap, number_ctor, "isFinite", number_is_finite, 1)?;
    define_native(heap, number_ctor, "isInteger", number_is_integer, 1)?;
    define_native(heap, number_ctor, "isNaN", number_is_nan, 1)?;
    define_native(
        heap,
        number_ctor,
        "isSafeInteger",
        number_is_safe_integer,
        1,
    )?;
    define_native(heap, number_ctor, "parseFloat", number_parse_float, 1)?;
    define_native(heap, number_ctor, "parseInt", number_parse_int, 2)?;
    define_const(heap, number_ctor, "EPSILON", js_num(f64::EPSILON))?;
    define_const(
        heap,
        number_ctor,
        "MAX_SAFE_INTEGER",
        js_num(9007199254740991.0),
    )?;
    define_const(heap, number_ctor, "MAX_VALUE", Value::Float(f64::MAX))?;
    define_const(
        heap,
        number_ctor,
        "MIN_SAFE_INTEGER",
        js_num(-9007199254740991.0),
    )?;
    define_const(
        heap,
        number_ctor,
        "MIN_VALUE",
        Value::Float(f64::MIN_POSITIVE),
    )?;
    define_const(heap, number_ctor, "NaN", Value::Float(f64::NAN))?;
    define_const(
        heap,
        number_ctor,
        "NEGATIVE_INFINITY",
        Value::Float(f64::NEG_INFINITY),
    )?;
    define_const(
        heap,
        number_ctor,
        "POSITIVE_INFINITY",
        Value::Float(f64::INFINITY),
    )?;
    wire_ctor_proto(heap, number_ctor, number_proto)?;
    heap.define_property_str(global, "Number", Property::data(Value::Object(number_ctor)))?;
    heap.release_temp(number_base);

    // ---- Boolean --------------------------------------------------------
    let boolean_proto = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    let boolean_base = heap.protect(Value::Object(boolean_proto));
    link_to_object_prototype(heap, boolean_proto);
    define_native(heap, boolean_proto, "toString", boolean_to_string, 0)?;
    define_native(heap, boolean_proto, "valueOf", boolean_value_of, 0)?;
    let boolean_ctor = new_native_constructor(heap, "Boolean", boolean_construct, 1)?;
    wire_ctor_proto(heap, boolean_ctor, boolean_proto)?;
    heap.boolean_prototype = Some(boolean_proto);
    heap.define_property_str(
        global,
        "Boolean",
        Property::data(Value::Object(boolean_ctor)),
    )?;
    heap.release_temp(boolean_base);

    // ---- BigInt ---------------------------------------------------------
    let bigint_proto = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    let bigint_base = heap.protect(Value::Object(bigint_proto));
    link_to_object_prototype(heap, bigint_proto);
    define_native(heap, bigint_proto, "toString", bigint_to_string, 1)?;
    define_native(heap, bigint_proto, "valueOf", bigint_value_of, 0)?;
    let bigint_ctor = new_native_function(heap, "BigInt", bigint_construct, 1)?;
    define_native(heap, bigint_ctor, "asIntN", bigint_as_int_n, 2)?;
    define_native(heap, bigint_ctor, "asUintN", bigint_as_uint_n, 2)?;
    wire_ctor_proto(heap, bigint_ctor, bigint_proto)?;
    heap.define_property_str(global, "BigInt", Property::data(Value::Object(bigint_ctor)))?;
    heap.release_temp(bigint_base);

    // Record the wrapper prototypes for `get_method`'s spec `GetMethod`
    // `ToObject` step on primitive patterns (see `WRAPPER_PROTOS`).
    WRAPPER_PROTOS.with(|c| {
        *c.borrow_mut() = Some((
            heap.id(),
            WrapperProtos {
                string: string_proto,
                number: number_proto,
                bigint: bigint_proto,
            },
        ))
    });

    // ---- Math -----------------------------------------------------------
    let math = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    let math_base = heap.protect(Value::Object(math));
    link_to_object_prototype(heap, math);
    define_native(heap, math, "abs", math_abs, 1)?;
    define_native(heap, math, "acos", math_acos, 1)?;
    define_native(heap, math, "acosh", math_acosh, 1)?;
    define_native(heap, math, "asin", math_asin, 1)?;
    define_native(heap, math, "asinh", math_asinh, 1)?;
    define_native(heap, math, "atan", math_atan, 1)?;
    define_native(heap, math, "atanh", math_atanh, 1)?;
    define_native(heap, math, "atan2", math_atan2, 2)?;
    define_native(heap, math, "cbrt", math_cbrt, 1)?;
    define_native(heap, math, "ceil", math_ceil, 1)?;
    define_native(heap, math, "clz32", math_clz32, 1)?;
    define_native(heap, math, "cos", math_cos, 1)?;
    define_native(heap, math, "cosh", math_cosh, 1)?;
    define_native(heap, math, "exp", math_exp, 1)?;
    define_native(heap, math, "expm1", math_expm1, 1)?;
    define_native(heap, math, "floor", math_floor, 1)?;
    define_native(heap, math, "fround", math_fround, 1)?;
    define_native(heap, math, "hypot", math_hypot, 2)?;
    define_native(heap, math, "imul", math_imul, 2)?;
    define_native(heap, math, "log", math_log, 1)?;
    define_native(heap, math, "log1p", math_log1p, 1)?;
    define_native(heap, math, "log2", math_log2, 1)?;
    define_native(heap, math, "log10", math_log10, 1)?;
    define_native(heap, math, "max", math_max, 2)?;
    define_native(heap, math, "min", math_min, 2)?;
    define_native(heap, math, "pow", math_pow, 2)?;
    define_native(heap, math, "random", math_random, 0)?;
    define_native(heap, math, "round", math_round, 1)?;
    define_native(heap, math, "sign", math_sign, 1)?;
    define_native(heap, math, "sin", math_sin, 1)?;
    define_native(heap, math, "sinh", math_sinh, 1)?;
    define_native(heap, math, "sqrt", math_sqrt, 1)?;
    define_native(heap, math, "tan", math_tan, 1)?;
    define_native(heap, math, "tanh", math_tanh, 1)?;
    define_native(heap, math, "trunc", math_trunc, 1)?;
    define_const(heap, math, "E", Value::Float(std::f64::consts::E))?;
    define_const(heap, math, "LN10", Value::Float(std::f64::consts::LN_10))?;
    define_const(heap, math, "LN2", Value::Float(std::f64::consts::LN_2))?;
    define_const(
        heap,
        math,
        "LOG10E",
        Value::Float(std::f64::consts::LOG10_E),
    )?;
    define_const(heap, math, "LOG2E", Value::Float(std::f64::consts::LOG2_E))?;
    define_const(heap, math, "PI", Value::Float(std::f64::consts::PI))?;
    define_const(
        heap,
        math,
        "SQRT1_2",
        Value::Float(std::f64::consts::FRAC_1_SQRT_2),
    )?;
    define_const(heap, math, "SQRT2", Value::Float(std::f64::consts::SQRT_2))?;
    heap.define_property_str(global, "Math", Property::data(Value::Object(math)))?;
    heap.release_temp(math_base);

    // ---- JSON -----------------------------------------------------------
    let json = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    let json_base = heap.protect(Value::Object(json));
    link_to_object_prototype(heap, json);
    define_native(heap, json, "parse", json_parse, 1)?;
    define_native(heap, json, "stringify", json_stringify, 3)?;
    heap.define_property_str(global, "JSON", Property::data(Value::Object(json)))?;
    heap.release_temp(json_base);

    // ---- Globals --------------------------------------------------------
    define_native(heap, global, "parseInt", number_parse_int, 2)?;
    define_native(heap, global, "parseFloat", number_parse_float, 1)?;
    define_native(heap, global, "isNaN", global_is_nan, 1)?;
    define_native(heap, global, "isFinite", global_is_finite, 1)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Heap, Gc<Object>) {
        let mut heap = Heap::new();
        let global = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        install(&mut heap, global).unwrap();
        (heap, global)
    }

    // Naive reference implementations for the fused-multiply-add test.
    fn bn_mul_small(d: &[u32], m: u32) -> Vec<u32> {
        if m == 0 || d.iter().all(|&w| w == 0) {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(d.len() + 1);
        let mut carry: u64 = 0;
        for &w in d {
            let t = u64::from(w) * u64::from(m) + carry;
            out.push(t as u32);
            carry = t >> 32;
        }
        if carry != 0 {
            out.push(carry as u32);
        }
        out
    }

    fn bn_add_small(d: &[u32], a: u32) -> Vec<u32> {
        let mut out = d.to_vec();
        let mut carry = u64::from(a);
        let mut i = 0;
        while carry != 0 {
            if i == out.len() {
                out.push(0);
            }
            let t = u64::from(out[i]) + carry;
            out[i] = t as u32;
            carry = t >> 32;
            i += 1;
        }
        out
    }

    fn s(heap: &mut Heap, text: &str) -> Value {
        Value::Str(heap.alloc_str(text).unwrap())
    }

    fn as_str(v: Value) -> String {
        match v {
            Value::Str(g) => g.as_wtf8().to_string(),
            v => panic!("expected string, got {v:?}"),
        }
    }

    fn as_num(v: Value) -> f64 {
        match v {
            Value::Int(i) => i as f64,
            Value::Float(f) => f,
            v => panic!("expected number, got {v:?}"),
        }
    }

    fn call(f: NativeFn, heap: &mut Heap, this: Value, args: &[Value]) -> Value {
        f(heap, this, args).expect("native call failed")
    }

    #[test]
    fn install_exposes_constructors_and_globals() {
        let (heap, global) = setup();
        for name in [
            "String",
            "Number",
            "BigInt",
            "Math",
            "JSON",
            "parseInt",
            "parseFloat",
        ] {
            let p = heap
                .get_property(global, Wtf8::from_str(name))
                .unwrap_or_else(|| panic!("{name} not installed"));
            assert!(matches!(p.value, Value::Object(_)), "{name} is an object");
        }
    }

    #[test]
    fn string_prototype_methods_installed() {
        let (heap, global) = setup();
        let ctor = match heap
            .get_property(global, Wtf8::from_str("String"))
            .unwrap()
            .value
        {
            Value::Object(o) => o,
            v => panic!("String is not an object: {v:?}"),
        };
        let proto = match heap
            .get_property(ctor, Wtf8::from_str("prototype"))
            .unwrap()
            .value
        {
            Value::Object(o) => o,
            v => panic!("String.prototype is not an object: {v:?}"),
        };
        for name in [
            "charAt",
            "slice",
            "indexOf",
            "toUpperCase",
            "trim",
            "split",
            "concat",
            "replace",
        ] {
            assert!(
                heap.get_property(proto, Wtf8::from_str(name)).is_some(),
                "String.prototype.{name} installed"
            );
        }
    }

    #[test]
    fn char_at() {
        let (mut heap, _) = setup();
        let this = s(&mut heap, "hello");
        assert_eq!(
            as_str(call(string_char_at, &mut heap, this, &[Value::Int(1)])),
            "e"
        );
        let this = s(&mut heap, "hello");
        assert_eq!(
            as_str(call(string_char_at, &mut heap, this, &[Value::Int(99)])),
            ""
        );
    }

    #[test]
    fn slice() {
        let (mut heap, _) = setup();
        let this = s(&mut heap, "hello");
        assert_eq!(
            as_str(call(
                string_slice,
                &mut heap,
                this,
                &[Value::Int(1), Value::Int(3)]
            )),
            "el"
        );
        let this = s(&mut heap, "hello");
        assert_eq!(
            as_str(call(
                string_slice,
                &mut heap,
                this,
                &[Value::Int(-3), Value::Undefined]
            )),
            "llo"
        );
    }

    #[test]
    fn index_of() {
        let (mut heap, _) = setup();
        let this = s(&mut heap, "hello world");
        let pat = s(&mut heap, "o");
        assert_eq!(as_num(call(string_index_of, &mut heap, this, &[pat])), 4.0);
        let this = s(&mut heap, "hello world");
        let pat = s(&mut heap, "z");
        assert_eq!(as_num(call(string_index_of, &mut heap, this, &[pat])), -1.0);
    }

    #[test]
    fn to_upper_case() {
        let (mut heap, _) = setup();
        let this = s(&mut heap, "aBc");
        assert_eq!(
            as_str(call(string_to_upper_case, &mut heap, this, &[])),
            "ABC"
        );
    }

    #[test]
    fn is_well_formed() {
        // Plain and empty strings are well-formed.
        assert_eq!(eval_str("'abc'.isWellFormed() + ''"), "true");
        assert_eq!(eval_str("''.isWellFormed() + ''"), "true");
        // A surrogate pair (astral character) is well-formed.
        assert_eq!(eval_str("'\\u{1F600}'.isWellFormed() + ''"), "true");
        assert_eq!(
            eval_str("String.fromCharCode(0xD800, 0xDC00).isWellFormed() + ''"),
            "true"
        );
        // Concatenating a lead and a trail across the `+` boundary pairs
        // them (maximal pairing), so the result is well-formed.
        assert_eq!(
            eval_str(
                "(String.fromCharCode(0xD83D) + String.fromCharCode(0xDE00)).isWellFormed() + ''"
            ),
            "true"
        );
        // Lone surrogates are ill-formed per spec
        // (`IsStringWellFormedUnicode`); the WTF-8 string model preserves
        // them instead of replacing with U+FFFD at ingress.
        assert_eq!(
            eval_str("String.fromCharCode(0xD800).isWellFormed() + ''"),
            "false"
        );
        assert_eq!(
            eval_str("String.fromCharCode(0xDC00).isWellFormed() + ''"),
            "false"
        );
        // Receiver is coerced with `ToString`. `null`/`undefined`
        // receivers throw `TypeError` (RequireObjectCoercible), per spec —
        // explicit `.call` shapes pass thisArg through unsubstituted.
        assert_eq!(
            eval_str("String.prototype.isWellFormed.call(42) + ''"),
            "true"
        );
        assert_eq!(
            eval_str(
                "var r; try { String.prototype.isWellFormed.call(null); r = 'no-throw'; } \
                 catch (e) { r = e instanceof TypeError ? 'TypeError' : 'wrong'; } r"
            ),
            "TypeError"
        );
        assert_eq!(
            eval_str(
                "var r; try { String.prototype.isWellFormed.call(undefined); r = 'no-throw'; } \
                 catch (e) { r = e instanceof TypeError ? 'TypeError' : 'wrong'; } r"
            ),
            "TypeError"
        );
        // Spec `length` is 0.
        assert_eq!(eval_str("String.prototype.isWellFormed.length + ''"), "0");
    }

    #[test]
    fn to_well_formed() {
        // Well-formed input is returned unchanged.
        assert_eq!(eval_str("'abc'.toWellFormed()"), "abc");
        assert_eq!(eval_str("'\\u{1F600}'.toWellFormed()"), "\u{1F600}");
        // Lone surrogates are preserved by the WTF-8 string model, so
        // `toWellFormed` replaces each with U+FFFD (spec `String.prototype.toWellFormed`).
        assert_eq!(
            eval_str("String.fromCharCode(0xD800).toWellFormed()"),
            "\u{FFFD}"
        );
        assert_eq!(
            eval_str("String.fromCharCode(0xD800, 0xDC00).toWellFormed()"),
            "\u{10000}"
        );
        // Each lone surrogate becomes exactly one U+FFFD.
        assert_eq!(
            eval_str("String.fromCharCode(0xD800).toWellFormed().length + ''"),
            "1"
        );
        assert_eq!(
            eval_str("String.fromCharCode(0xD800, 0x41, 0xDC00).toWellFormed().length + ''"),
            "3"
        );
        // Non-string receivers are coerced. `null`/`undefined` receivers
        // throw `TypeError` (RequireObjectCoercible), per spec — explicit
        // `.call` shapes pass thisArg through unsubstituted.
        assert_eq!(eval_str("String.prototype.toWellFormed.call(42)"), "42");
        assert_eq!(
            eval_str(
                "var r; try { String.prototype.toWellFormed.call(null); r = 'no-throw'; } \
                 catch (e) { r = e instanceof TypeError ? 'TypeError' : 'wrong'; } r"
            ),
            "TypeError"
        );
        // Spec `length` is 0.
        assert_eq!(eval_str("String.prototype.toWellFormed.length + ''"), "0");
    }

    #[test]
    fn lone_surrogate_round_trip() {
        // Lone surrogates survive every ingress and report one code unit.
        assert_eq!(eval_str("'\\uD800'.length + ''"), "1");
        assert_eq!(eval_str("String.fromCharCode(0xD800).length + ''"), "1");
        assert_eq!(
            eval_str("String.fromCharCode(0xD800).charCodeAt(0) + ''"),
            "55296"
        );
        assert_eq!(
            eval_str("var s = String.fromCharCode(0xD800); (s === '\\uD800') + ''"),
            "true"
        );
        // JSON \\u escapes preserve lone surrogates (V8 behavior).
        assert_eq!(eval_str("JSON.parse('\"\\\\uD800\"').length + ''"), "1");
        assert_eq!(
            eval_str("JSON.parse('\"\\\\uD800\\\\uDC00\"').length + ''"),
            "2"
        );
        // String iterator yields lone surrogates as themselves.
        assert_eq!(eval_str("[...'\\uD800'].length + ''"), "1");
        assert_eq!(eval_str("([...'\\uD800'][0] === '\\uD800') + ''"), "true");
        // Index access on a primitive string.
        assert_eq!(eval_str("'abc'[1]"), "b");
        assert_eq!(eval_str("('\\uD800'[0] === '\\uD800') + ''"), "true");
        assert_eq!(eval_str("'abc'.length + ''"), "3");
        // Surrogate pairs stay paired through concatenation.
        assert_eq!(
            eval_str("String.fromCharCode(0xD83D, 0xDE00).length + ''"),
            "2"
        );
        assert_eq!(
            eval_str(
                "(String.fromCharCode(0xD83D) + String.fromCharCode(0xDE00)).isWellFormed() + ''"
            ),
            "true"
        );
        // Regex replace/split on lone surrogates.
        assert_eq!(eval_str("'a\\uD800b'.replace('\\uD800', 'X')"), "aXb");
        assert_eq!(eval_str("'a\\uD800b'.split('\\uD800').length + ''"), "2");
    }

    #[test]
    fn null_proto_regexp_lone_surrogate_source_stays_exact() {
        // A null-prototype regexp with a lone-surrogate source must take
        // the same exact-compile path as one with a prototype: the
        // pattern matches the lone surrogate itself, not U+FFFD.
        const MK: &str =
            "var re = Object.setPrototypeOf(new RegExp(String.fromCharCode(0xD800)), null);";
        const MK_PROTO: &str = "var re = new RegExp(String.fromCharCode(0xD800));";
        const SUBJECT: &str = "var s = String.fromCharCode(0x41, 0xD800, 0x42);";
        // Sanity: the source property really is the lone surrogate.
        assert_eq!(
            eval_str(&format!("{MK} re.source.charCodeAt(0) + ''")),
            "55296"
        );
        // replace: null-prototype fast path matches the surrogate.
        let replace_js = |mk: &str| {
            eval_str(&format!(
                "{mk} {SUBJECT} var r = s.replace(re, 'X'); \
                 r.charCodeAt(0) + ',' + r.charCodeAt(1) + ',' + r.charCodeAt(2) + ',' + r.length"
            ))
        };
        assert_eq!(replace_js(MK), "65,88,66,3");
        assert_eq!(replace_js(MK_PROTO), "65,88,66,3");
        // split: null-prototype fast path splits on the surrogate.
        let split_js = |mk: &str| {
            eval_str(&format!(
                "{mk} {SUBJECT} var p = s.split(re); \
                 p.length + ':' + p[0].charCodeAt(0) + ':' + p[1].charCodeAt(0)"
            ))
        };
        assert_eq!(split_js(MK), "2:65:66");
        assert_eq!(split_js(MK_PROTO), "2:65:66");
    }

    #[test]
    fn trim() {
        let (mut heap, _) = setup();
        let this = s(&mut heap, "  hi\t\n");
        assert_eq!(as_str(call(string_trim, &mut heap, this, &[])), "hi");
    }

    #[test]
    fn split() {
        let (mut heap, _) = setup();
        let this = s(&mut heap, "a,b,c");
        let sep = s(&mut heap, ",");
        let arr = match call(string_split, &mut heap, this, &[sep]) {
            Value::Object(o) => o,
            v => panic!("split did not return an array: {v:?}"),
        };
        let len = as_num(
            heap.get_property(arr, Wtf8::from_str("length"))
                .expect("length")
                .value,
        );
        assert_eq!(len, 3.0);
        for (i, want) in ["a", "b", "c"].iter().enumerate() {
            let got = heap
                .get_property(arr, Wtf8::from_str(&i.to_string()))
                .unwrap_or_else(|| panic!("missing index {i}"))
                .value;
            assert_eq!(as_str(got), *want);
        }
    }

    fn split_parts(this: &str, sep: &str, lim: Value) -> Vec<String> {
        let (mut h, _) = setup();
        let this_v = s(&mut h, this);
        let sep_v = s(&mut h, sep);
        let arr_v = call(string_split, &mut h, this_v, &[sep_v, lim]);
        let arr = match arr_v {
            Value::Object(o) => o,
            v => panic!("split did not return an array: {v:?}"),
        };
        let len = as_num(
            h.get_property(arr, Wtf8::from_str("length"))
                .expect("length")
                .value,
        ) as usize;
        (0..len)
            .map(|i| {
                as_str(
                    h.get_property(arr, Wtf8::from_str(&i.to_string()))
                        .unwrap_or_else(|| panic!("missing index {i}"))
                        .value,
                )
                .to_owned()
            })
            .collect()
    }

    #[test]
    fn split_limit() {
        // Limit smaller than the part count: no trailing segment.
        assert_eq!(
            split_parts("a,b,c", ",", Value::Int(2)),
            vec!["a".to_string(), "b".to_string()]
        );
        // Limit larger than the part count: full split incl. trailing part.
        assert_eq!(
            split_parts("a,b", ",", Value::Int(5)),
            vec!["a".to_string(), "b".to_string()]
        );
        // Empty separator with a limit splits into code units.
        assert_eq!(
            split_parts("abcd", "", Value::Int(2)),
            vec!["a".to_string(), "b".to_string()]
        );
        // Non-ASCII path honors the limit the same way.
        assert_eq!(
            split_parts("héllo,wörld", ",", Value::Int(1)),
            vec!["héllo".to_string()]
        );
        assert_eq!(
            split_parts("a,b", ",", Value::Undefined),
            vec!["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn string_construct_coerces() {
        let (mut heap, _) = setup();
        assert_eq!(
            as_str(call(string_construct, &mut heap, Value::Undefined, &[])),
            ""
        );
        assert_eq!(
            as_str(call(
                string_construct,
                &mut heap,
                Value::Undefined,
                &[Value::Int(42)]
            )),
            "42"
        );
        // Arrays keep the comma-join; error objects dispatch
        // `Error.prototype.toString`.
        assert_eq!(eval_str("String([1, 2, 3]);"), "1,2,3");
        assert_eq!(eval_str("String(new TypeError('bad'));"), "TypeError: bad");
    }

    /// Native `toString` for [`string_long_result_survives_forced_collect`]:
    /// forces a collection mid-conversion (freeing any unrooted string
    /// handle the builtin is holding), then allocates garbage in the same
    /// size class so a freed block is observably reused. Returns a long
    /// (>15 byte) heap string.
    fn gc_stress_collecting_to_string(
        heap: &mut Heap,
        _this: Value,
        _args: &[Value],
    ) -> Result<Value, NativeError> {
        heap.collect();
        for _ in 0..200 {
            let _ = heap.alloc_str("GARBAGEGARBAGEGA")?;
        }
        Ok(Value::Str(heap.alloc_str("0123456789ABCDEF")?))
    }

    #[test]
    fn string_long_result_survives_forced_collect() {
        // Rooting test for `String(x)`: the object arm of `js_to_string_val`
        // converts via `to_string_js_str`, whose `Str::Heap` handle for a
        // `toString` result is unrooted once `call_function` releases its
        // temporary roots. The hook forces a collection mid-conversion; the
        // handle must flow from conversion to the caller with no
        // intervening heap allocation, otherwise the garbage allocs reuse
        // its block and the result comes back corrupted.
        let (mut heap, _global) = setup();
        let obj = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let obase = heap.protect(Value::Object(obj));
        let to_string_fn = Value::Object(
            new_native_function(&mut heap, "toString", gc_stress_collecting_to_string, 0).unwrap(),
        );
        heap.define_property_str(obj, "toString", Property::data(to_string_fn))
            .unwrap();
        let v = call(
            string_construct,
            &mut heap,
            Value::Undefined,
            &[Value::Object(obj)],
        );
        heap.release_temp(obase);
        assert_eq!(as_str(v), "0123456789ABCDEF");
        // Long heap-resident string input takes the `Value::Str` fast path
        // (free handle copy): pin that it also survives collections around
        // the call, the way the interpreter's rooted argument does.
        let long = heap.alloc_str(&"x".repeat(64)).unwrap();
        let lbase = heap.protect(Value::Str(long));
        let v = call(
            string_construct,
            &mut heap,
            Value::Undefined,
            &[Value::Str(long)],
        );
        heap.collect();
        for _ in 0..64 {
            let _ = heap.alloc_str("GARBAGEGARBAGEGA").unwrap();
        }
        assert_eq!(as_str(v), "x".repeat(64));
        heap.release_temp(lbase);
    }

    /// Native `toString` for [`string_construct_handoff_allocates_nothing`]:
    /// allocates exactly one long (>15 byte) heap string and nothing else.
    fn single_alloc_to_string(
        heap: &mut Heap,
        _this: Value,
        _args: &[Value],
    ) -> Result<Value, NativeError> {
        Ok(Value::Str(heap.alloc_str("0123456789ABCDEF")?))
    }

    #[test]
    fn string_construct_handoff_allocates_nothing() {
        // The `to_string_js_str` -> `js_to_string_val` hand-off carries an
        // unrooted `Str::Heap`; it is only sound because zero heap
        // allocations run between conversion and consumption. Pin that: a
        // `toString` hook allocating exactly one string must be the only
        // allocation in the whole `String(obj)` call. (Automatic collection
        // is disabled by default, so the count is exact.)
        let (mut heap, _global) = setup();
        let obj = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let obase = heap.protect(Value::Object(obj));
        let to_string_fn = Value::Object(
            new_native_function(&mut heap, "toString", single_alloc_to_string, 0).unwrap(),
        );
        heap.define_property_str(obj, "toString", Property::data(to_string_fn))
            .unwrap();
        let before = heap.alloc_count();
        let v = call(
            string_construct,
            &mut heap,
            Value::Undefined,
            &[Value::Object(obj)],
        );
        let after = heap.alloc_count();
        heap.release_temp(obase);
        assert_eq!(
            after,
            before + 1,
            "String(obj) allocated in the conversion hand-off gap"
        );
        assert_eq!(as_str(v), "0123456789ABCDEF");
    }

    #[test]
    fn new_array_roots_elements_across_alloc() {
        // Elements arrive in an untraced vec; `new_array` must root them
        // across its own `alloc_object`, whose collection would otherwise
        // free them before they are stored on the array.
        let mut heap = Heap::new();
        // Long enough to heap-allocate: short strings would pack inline
        // and the test would prove nothing about rooting.
        let e1 = heap.alloc_str("element-one-xxxxxxxxxxxxxxxxxxxx").unwrap();
        let e2 = heap.alloc_str("element-two-yyyyyyyyyyyyyyyyyyyy").unwrap();
        let before = heap.alloc_count();
        // Force the collection inside `alloc_object` to actually run.
        heap.set_gc_threshold(1);
        let arr = new_array(&mut heap, vec![Value::Str(e1), Value::Str(e2)]).unwrap();
        let after = heap.alloc_count();
        // Fresh heap: just the array object. The "0", "1", and "length"
        // keys pack inline and allocate nothing; both elements survived
        // the collection.
        assert_eq!(
            after,
            before + 1,
            "elements were freed by new_array's collection"
        );
        let arr = match arr {
            Value::Object(o) => o,
            other => panic!("new_array did not return an object: {other:?}"),
        };
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("0")).unwrap().value,
            Value::Str(e1)
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("1")).unwrap().value,
            Value::Str(e2)
        );
    }

    #[test]
    fn parse_array_roots_elements_across_values() {
        // Elements accumulate in an untraced vec while later elements parse;
        // each must be rooted as it arrives so a collection triggered by a
        // later `parse_value` cannot free an earlier element.
        let mut heap = Heap::new();
        // Long enough to heap-allocate: short strings would inline and the
        // test would prove nothing about rooting.
        let text = Wtf8Buf::from(
            r#"["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","cccccccccccccccccccccccccccccccc"]"#,
        );
        let mut p = JsonParser {
            s: &text,
            b: text.as_bytes(),
            pos: 0,
            depth: 0,
        };
        heap.set_gc_threshold(1);
        let v = p.parse_array(&mut heap).unwrap();
        // Fresh heap: 3 element strings + the array. The "length", "0",
        // "1", "2" keys pack inline and allocate nothing; nothing was
        // swept mid-parse.
        assert_eq!(heap.alloc_count(), 4, "an element was freed mid-parse");
        let arr = match v {
            Value::Object(o) => o,
            other => panic!("parse_array did not return an object: {other:?}"),
        };
        for (i, want) in ["a".repeat(32), "b".repeat(32), "c".repeat(32)]
            .iter()
            .enumerate()
        {
            let got = heap
                .get_property(arr, Wtf8::from_str(&i.to_string()))
                .unwrap()
                .value;
            match got {
                Value::Str(g) => assert_eq!(g.as_wtf8(), Wtf8::from_str(want)),
                other => panic!("element {i} is not a string: {other:?}"),
            }
        }
    }

    #[test]
    fn parse_object_roots_self_across_values() {
        // The fresh object is unlinked until its properties are defined; it
        // must be rooted across the `parse_value` calls whose allocations
        // may collect and free it before the properties are defined.
        let mut heap = Heap::new();
        // Long values heap-allocate: short strings would inline and the
        // value-rooting half of the test would prove nothing.
        let text = Wtf8Buf::from(
            r#"{"a":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx","b":"yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"}"#,
        );
        let mut p = JsonParser {
            s: &text,
            b: text.as_bytes(),
            pos: 0,
            depth: 0,
        };
        heap.set_gc_threshold(1);
        let v = p.parse_object(&mut heap).unwrap();
        // Fresh heap: the object + 2 value strings. The "a"/"b" keys pack
        // inline and allocate nothing; the object was never swept mid-parse.
        assert_eq!(heap.alloc_count(), 3, "object was freed mid-parse");
        let obj = match v {
            Value::Object(o) => o,
            other => panic!("parse_object did not return an object: {other:?}"),
        };
        let x = heap.get_property(obj, Wtf8::from_str("a")).unwrap().value;
        assert!(
            matches!(x, Value::Str(g) if g.as_wtf8() == Wtf8::from_str(&"x".repeat(32))),
            "wrong value for \"a\": {x:?}"
        );
        let y = heap.get_property(obj, Wtf8::from_str("b")).unwrap().value;
        assert!(
            matches!(y, Value::Str(g) if g.as_wtf8() == Wtf8::from_str(&"y".repeat(32))),
            "wrong value for \"b\": {y:?}"
        );
    }

    #[test]
    fn json_parse_multi_element_survives_gc_pressure() {
        // End to end: every value in a parsed multi-element payload must
        // survive the collections triggered while later values parse.
        let mut vm = crate::Vm::new();
        vm.heap_mut().set_gc_threshold(8);
        let r = vm
            .eval(
                r#"
                var a = JSON.parse('["alpha","beta","gamma","delta","epsilon","zeta","eta","theta"]');
                var o = JSON.parse('{"a":{"x":1},"b":{"y":2},"c":[true,false]}');
                [a[0], a[3], a[7], a.length, o.a.x, o.b.y, o.c[1]].join("|");
                "#,
            )
            .expect("eval failed");
        let joined = match r {
            Value::Str(g) => g.as_wtf8().to_string(),
            other => panic!("expected string result, got {other:?}"),
        };
        assert_eq!(joined, "alpha|delta|theta|8|1|2|false");
    }

    #[test]
    fn parse_int_global() {
        let (mut heap, _) = setup();
        let arg = s(&mut heap, "42px");
        assert_eq!(
            as_num(call(number_parse_int, &mut heap, Value::Undefined, &[arg])),
            42.0
        );
        let arg = s(&mut heap, "0x10");
        assert_eq!(
            as_num(call(number_parse_int, &mut heap, Value::Undefined, &[arg])),
            16.0
        );
        let arg = s(&mut heap, "abc");
        assert!(as_num(call(number_parse_int, &mut heap, Value::Undefined, &[arg])).is_nan());
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn parse_float_global() {
        let (mut heap, _) = setup();
        let arg = s(&mut heap, "3.14abc");
        assert_eq!(
            as_num(call(
                number_parse_float,
                &mut heap,
                Value::Undefined,
                &[arg]
            )),
            3.14
        );
    }

    #[test]
    fn number_to_string_radix() {
        let (mut heap, _) = setup();
        assert_eq!(
            as_str(call(
                number_to_string,
                &mut heap,
                Value::Int(255),
                &[Value::Int(16)]
            )),
            "ff"
        );
        assert_eq!(
            as_str(call(
                number_to_string,
                &mut heap,
                Value::Int(10),
                &[Value::Int(2)]
            )),
            "1010"
        );
        assert_eq!(
            as_str(call(
                number_to_string,
                &mut heap,
                Value::Int(42),
                &[Value::Undefined]
            )),
            "42"
        );
    }

    // -- replace / replaceAll with callable replacements -------------------
    // Regression: bytecode (arrow) replacers must re-enter the VM through
    // the shared call_function choke point instead of throwing TypeError.

    /// Run `src` in a fresh VM and return its string completion value.
    fn eval_str(src: &str) -> String {
        let mut vm = crate::Vm::new();
        match vm.eval(src) {
            Ok(v) => match v {
                Value::Str(g) => g.as_wtf8().to_string(),
                v => panic!("expected string result for {src:?}, got {v:?}"),
            },
            Err(e) => panic!("eval failed for {src:?}: {e:?}"),
        }
    }

    #[test]
    fn replace_string_pattern_arrow_replacer() {
        assert_eq!(
            eval_str(r#"'hello'.replace('l', (m, off, s) => '[' + m + off + ']');"#),
            "he[l2]lo"
        );
    }

    #[test]
    fn replace_regexp_pattern_arrow_replacer() {
        assert_eq!(
            eval_str(r#"'hello'.replace(/l/, (m) => m.toUpperCase());"#),
            "heLlo"
        );
    }

    #[test]
    fn replace_all_regexp_pattern_arrow_replacer() {
        assert_eq!(
            eval_str(r#"'aaa'.replaceAll(/a/g, (m, off) => String(off));"#),
            "012"
        );
    }

    #[test]
    fn replace_replacer_throw_propagates() {
        let mut vm = crate::Vm::new();
        let err = vm
            .eval(r#"'a'.replace('a', () => { throw 'boom'; });"#)
            .expect_err("throwing replacer must propagate");
        assert!(
            format!("{err:?}").contains("boom"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn replace_regexp_replacer_throw_propagates() {
        let mut vm = crate::Vm::new();
        let err = vm
            .eval(r#"'a'.replace(/a/, () => { throw 'rxboom'; });"#)
            .expect_err("throwing regexp replacer must propagate");
        assert!(
            format!("{err:?}").contains("rxboom"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn replace_regexp_named_group_args() {
        assert_eq!(
            eval_str(
                r#"'John Smith'.replace(/(?<first>\w+) (?<last>\w+)/, (m, f, l, off, str, g) => g.last + ', ' + g.first);"#
            ),
            "Smith, John"
        );
    }

    // Focused tests for the allocation/ASCII-fast-path helpers.
    #[test]
    fn trim_js_helpers() {
        assert_eq!(trim_js("  abc  "), "abc");
        assert_eq!(trim_js("\t\nabc\r\n"), "abc");
        assert_eq!(trim_js(""), "");
        assert_eq!(trim_js("   "), "");
        assert_eq!(trim_js("abc"), "abc");
        // Non-ASCII JS whitespace on both edges.
        assert_eq!(trim_js("\u{a0}\u{3000}abc\u{2003}"), "abc");
        assert_eq!(trim_js_start("\u{feff}  abc  "), "abc  ");
        assert_eq!(trim_js_end("  abc\u{a0}"), "  abc");
        assert_eq!(trim_js_start(""), "");
        assert_eq!(trim_js_end(""), "");
    }

    #[test]
    fn units_len_counts_code_units() {
        assert_eq!(units_len(Wtf8::from_str("")), 0);
        assert_eq!(units_len(Wtf8::from_str("hello")), 5);
        assert_eq!(units_len(Wtf8::from_str("héllo")), 5);
        // Astral character: one char, two code units.
        assert_eq!(units_len(Wtf8::from_str("a\u{1f600}b")), 4);
    }

    #[test]
    fn nth_unit_samples() {
        assert_eq!(nth_unit(Wtf8::from_str("hello"), 1), Some(b'e' as u16));
        assert_eq!(nth_unit(Wtf8::from_str("hello"), 5), None);
        assert_eq!(nth_unit(Wtf8::from_str(""), 0), None);
        // Surrogate pair halves.
        assert_eq!(nth_unit(Wtf8::from_str("\u{1f600}"), 0), Some(0xD83D));
        assert_eq!(nth_unit(Wtf8::from_str("\u{1f600}"), 1), Some(0xDE00));
        assert_eq!(nth_unit(Wtf8::from_str("\u{1f600}"), 2), None);
    }

    #[test]
    fn slice_units_windows() {
        assert_eq!(
            slice_units(Wtf8::from_str("hello"), 1, 4).as_wtf8(),
            Wtf8::from_str("ell")
        );
        assert_eq!(
            slice_units(Wtf8::from_str("hello"), 0, 100).as_wtf8(),
            Wtf8::from_str("hello")
        );
        assert_eq!(
            slice_units(Wtf8::from_str("hello"), 3, 3).as_wtf8(),
            Wtf8::from_str("")
        );
        assert_eq!(
            slice_units(Wtf8::from_str("hello"), 4, 2).as_wtf8(),
            Wtf8::from_str("")
        );
        // Non-ASCII: only the wanted window is encoded.
        assert_eq!(
            slice_units(Wtf8::from_str("a\u{1f600}bc"), 1, 3).as_wtf8(),
            Wtf8::from_str("\u{1f600}")
        );
        assert_eq!(
            slice_units(Wtf8::from_str("a\u{1f600}bc"), 0, 1).as_wtf8(),
            Wtf8::from_str("a")
        );
    }

    #[test]
    fn units_find_skips() {
        let hay: Vec<u16> = "abcabc".encode_utf16().collect();
        let pat: Vec<u16> = "bc".encode_utf16().collect();
        assert_eq!(units_find(&hay, &pat, 0), Some(1));
        assert_eq!(units_find(&hay, &pat, 2), Some(4));
        let missing: Vec<u16> = "z".encode_utf16().collect();
        assert_eq!(units_find(&hay, &missing, 0), None);
        let empty: Vec<u16> = Vec::new();
        assert_eq!(units_find(&hay, &empty, 3), Some(3));
        assert_eq!(units_find(&hay, &empty, 99), Some(6));
        // Needle longer than haystack.
        assert_eq!(units_find(&empty, &pat, 0), None);
    }

    #[test]
    fn bn_mul_add_small_matches_naive() {
        // Fused in-place multiply-add must equal the old nest's result.
        let mut d = vec![0xFFFF_FFFFu32, 0x1234_5678];
        let mut expect = bn_add_small(&bn_mul_small(&d, 10), 7);
        bn_mul_add_small(&mut d, 10, 7);
        assert_eq!(d, expect);
        // Stays normalized: leading zeros are impossible.
        let mut z: Vec<u32> = Vec::new();
        bn_mul_add_small(&mut z, 10, 0);
        assert!(z.is_empty());
        bn_mul_add_small(&mut z, 10, 5);
        assert_eq!(z, vec![5]);
        expect = bn_add_small(&bn_mul_small(&z, 16), 0xFF);
        bn_mul_add_small(&mut z, 16, 0xFF);
        assert_eq!(z, expect);
    }

    #[test]
    fn bn_parse_round_trip() {
        for (s, want) in [
            ("0", "0"),
            ("7", "7"),
            (
                "123456789012345678901234567890",
                "123456789012345678901234567890",
            ),
            ("-987654321987654321", "-987654321987654321"),
            ("0xff", "255"),
            ("0b101010", "42"),
            ("0o777", "511"),
        ] {
            let b = bn_parse(Wtf8::from_str(s)).unwrap_or_else(|| panic!("parse {s}"));
            assert_eq!(bn_to_string_radix(&b, 10), want, "input {s}");
        }
        assert!(bn_parse(Wtf8::from_str("")).is_some());
        assert!(bn_parse(Wtf8::from_str("xyz")).is_none());
    }

    #[test]
    fn get_substitution_ascii_whole() {
        let whole = UnitSlice::Bytes(b"hello world");
        assert_eq!(
            get_substitution(
                Wtf8::from_str("[$&][$`][$']"),
                &whole,
                6,
                5,
                &[Wtf8Buf::from("world")]
            ),
            "[world][hello ][]"
        );
        assert_eq!(
            get_substitution(Wtf8::from_str("$$"), &whole, 0, 0, &[Wtf8Buf::from("")]),
            "$"
        );
        // String patterns have no groups: empty captures keep `$&`/`` `$` ``/`$'`
        // working and leave `$1` literal.
        assert_eq!(
            get_substitution(Wtf8::from_str("[$&][$`][$'][$1][$$]"), &whole, 6, 5, &[]),
            "[world][hello ][][$1][$]"
        );
    }

    #[test]
    fn number_ctor_and_unary_plus_agree_on_prefixed_strings() {
        // `Number('0b101')` (via `string_to_number`) and `+'0b101'` (via
        // `to_number`/`parse_js_number`) must use the same parser.
        for (digits, want) in [
            ("0b101", "5"),
            ("0B101", "5"),
            ("0o17", "15"),
            ("0x1F", "31"),
            ("  0b101  ", "5"),
        ] {
            assert_eq!(
                eval_str(&format!("String(Number('{digits}'));")),
                want,
                "Number('{digits}')"
            );
            assert_eq!(
                eval_str(&format!("String(+'{digits}');")),
                want,
                "+'{digits}'"
            );
        }
    }

    #[test]
    fn ascii_fast_path_end_to_end() {
        assert_eq!(eval_str(r#"'aaa'.replaceAll('a', 'b');"#), "bbb");
        assert_eq!(eval_str(r#"'aaa'.replaceAll('a', '$&');"#), "aaa");
        assert_eq!(eval_str(r#"'aaa'.replaceAll('a', '$1');"#), "$1$1$1");
        assert_eq!(eval_str(r#"'abc'.replaceAll('', 'X');"#), "XaXbXcX");
        assert_eq!(eval_str(r#"'a,b,c'.split(',').length.toString();"#), "3");
        assert_eq!(eval_str(r#"'hello'.lastIndexOf('l').toString();"#), "3");
        assert_eq!(eval_str(r#"'hello'.indexOf('l', 3).toString();"#), "3");
        assert_eq!(eval_str(r#"'héllo'.slice(1, 4);"#), "éll");
        assert_eq!(eval_str(r#"'a😀b'.charAt(1);"#), "\u{fffd}");
    }

    #[test]
    fn number_rejects_sign_before_non_decimal_prefixes() {
        // Spec 7.1.4.1: the `StringNumericLiteral` grammar gives
        // hex/binary/octal no sign slot; a sign is only valid before
        // `Infinity` or a decimal literal.
        for src in [
            "Number('+0x1')",
            "Number('-0x1')",
            "Number('+0Xff')",
            "Number('-0b11')",
            "Number('+0o17')",
            "Number('-0o17')",
            "Number('  +0x1  ')",
        ] {
            let mut vm = crate::Vm::new();
            match vm
                .eval(src)
                .unwrap_or_else(|e| panic!("{src} threw: {e:?}"))
            {
                Value::Float(f) => assert!(f.is_nan(), "{src} should be NaN, got {f}"),
                v => panic!("{src} should be NaN, got {v:?}"),
            }
        }
        // Unsigned prefixes and signed decimals are unaffected.
        let mut vm = crate::Vm::new();
        assert_eq!(vm.eval("Number('0x1')").unwrap(), Value::Int(1));
        assert_eq!(vm.eval("Number('0b11')").unwrap(), Value::Int(3));
        assert_eq!(vm.eval("Number('0o17')").unwrap(), Value::Int(15));
        assert_eq!(vm.eval("Number('-12.5')").unwrap(), Value::Float(-12.5));
        assert_eq!(
            vm.eval("Number('+Infinity')").unwrap(),
            Value::Float(f64::INFINITY)
        );
    }

    #[test]
    fn number_of_bigint_returns_number() {
        // Spec 20.1.1.1 / 7.1.4: `Number(1n)` is `1` — the constructor is
        // the one `ToNumber` caller that converts BigInts instead of
        // throwing.
        let mut vm = crate::Vm::new();
        assert_eq!(vm.eval("Number(1n)").unwrap(), Value::Int(1));
        assert_eq!(vm.eval("Number(10n)").unwrap(), Value::Int(10));
        assert_eq!(vm.eval("Number(-3n)").unwrap(), Value::Int(-3));
        assert_eq!(
            vm.eval("typeof Number(1n) === 'number'").unwrap(),
            Value::Bool(true)
        );
        // No regression: unary negation on a BigInt is a BigInt op, not a
        // `ToNumber` conversion, and mixed BigInt/number arithmetic still
        // throws.
        assert!(matches!(vm.eval("-1n").unwrap(), Value::BigInt(_)));
        assert_eq!(vm.eval("String(-1n) === '-1'").unwrap(), Value::Bool(true));
        assert!(vm.eval("1n + 1").is_err());
        // Only `Number()` converts BigInts: every other `ToNumber` caller
        // (spec 7.1.4) still throws. (`+1n`/`-1n` are BigInt ops in the VM,
        // not `ToNumber` conversions, so they stay BigInts.)
        assert!(vm.eval("Math.abs(1n)").is_err());
    }

    #[test]
    fn number_object_uses_toprimitive_number_hint() {
        // Spec 20.1.1.1 / 7.1.4: `Number(obj)` is
        // `ToNumber(ToPrimitive(obj, number))`, so `valueOf` runs before
        // `toString` and `@@toPrimitive` sees the "number" hint.
        let mut vm = crate::Vm::new();
        for src in [
            "var o1 = { valueOf: () => 1, toString: () => 2 }; Number(o1) === 1 && +o1 === 1;",
            "var o2 = { toString: () => 2 }; Number(o2) === 2 && +o2 === 2;",
            "var o3 = { [Symbol.toPrimitive]: (h) => h === 'number' ? 42 : 'str' }; Number(o3) === 42;",
        ] {
            assert_eq!(vm.eval(src).unwrap(), Value::Bool(true), "{src}");
        }
    }

    // -- Regression tests: builtin result arrays inherit Array.prototype --
    //
    // `String.prototype.split` results are `CreateArrayFromList` arrays,
    // so `.map`/`instanceof Array` must work on them (a null-prototype
    // regression once made `"a,b".split(",").map(x => x)` throw).

    /// Evaluate JS in a full VM and stringify the completion value
    /// (via `toString`, so non-string results like booleans work too).
    fn eval_to_string(source: &str) -> String {
        let mut vm = crate::Vm::new();
        let v = vm
            .eval(source)
            .unwrap_or_else(|e| panic!("eval failed: {e}"));
        crate::builtin_object_array::to_string_js(vm.heap_mut(), v)
            .map(|w| w.to_string())
            .unwrap_or_else(|e| panic!("to_string failed: {e:?}"))
    }

    #[test]
    fn split_result_is_real_array() {
        assert_eq!(
            eval_to_string(r#""a,b".split(",") instanceof Array"#),
            "true"
        );
        assert_eq!(
            eval_to_string(r#""a,b".split(",").map(x => x.toUpperCase()).join("-")"#),
            "A-B"
        );
    }

    // -- Regression tests: match / matchAll / search -----------------------
    //
    // `String.prototype.match` used to return a boolean ("v1: boolean core
    // only") and threw `TypeError` for non-`RegExp` patterns; `matchAll`
    // was an unimplemented stub; `search` returned a boolean too. They now
    // follow the spec's `@@match` / `@@matchAll` / `@@search` dispatch with
    // `RegExpCreate` coercion for patterns without the hook.

    #[test]
    fn match_returns_match_array() {
        // Non-global: `[whole, groups...]` (extras like `index`/`input`
        // are on the array too; `join` shows the indexed elements).
        assert_eq!(
            eval_to_string(r#""ab".match(/(a)(b)/).join(",")"#),
            "ab,a,b"
        );
        // Global: every whole match.
        assert_eq!(eval_to_string(r#""aba".match(/a/g).join(",")"#), "a,a");
        // No match: `null`.
        assert_eq!(eval_to_string(r#"String("ab".match(/z/))"#), "null");
        // The extras ride on the result array.
        assert_eq!(eval_to_string(r#""ab".match(/(b)/).index"#), "1");
        assert_eq!(eval_to_string(r#""ab".match(/(b)/).input"#), "ab");
        // Results are real arrays.
        assert_eq!(
            eval_to_string(r#""ab".match(/b/) instanceof Array"#),
            "true"
        );
    }

    #[test]
    fn match_coerces_non_regexp_pattern() {
        // Used to throw `TypeError`; the spec coerces via `RegExpCreate`.
        assert_eq!(eval_to_string(r#""ab".match("b").join(",")"#), "b");
        assert_eq!(eval_to_string(r#"String("ab".match("z"))"#), "null");
        assert_eq!(eval_to_string(r#""ab".match().join(",")"#), "");
    }

    #[test]
    fn match_dispatches_to_match_hook() {
        // Custom `@@match` hook (string-key alias of the well-known symbol).
        assert_eq!(
            eval_to_string(
                r#"var o = {}; o[Symbol.match] = function(s) { return "hooked:" + s; }; "ab".match(o)"#
            ),
            "hooked:ab"
        );
        // Overriding `@@match` on a regexp dispatches to the override,
        // with the regexp as receiver.
        assert_eq!(
            eval_to_string(
                r#"var rx = /b/; rx[Symbol.match] = function(s) { return this === rx ? "mine:" + s : "wrong"; }; "ab".match(rx)"#
            ),
            "mine:ab"
        );
        // Present-but-not-callable hook: spec `GetMethod` throws.
        let mut vm = crate::Vm::new();
        assert!(
            vm.eval(r#"var o = {}; o[Symbol.match] = 5; "ab".match(o)"#)
                .is_err(),
            "non-callable @@match should throw"
        );
    }

    #[test]
    fn match_all_returns_array_of_match_arrays() {
        // v1: an `Array` of match arrays instead of a
        // `RegExpStringIterator` (no iterator protocol yet).
        assert_eq!(
            eval_to_string(r#""aba".matchAll(/a/g).map(m => m.join(":")).join(";")"#),
            "a;a"
        );
        assert_eq!(
            eval_to_string(r#""ab".matchAll(/(a)(b)/g)[0].join(":")"#),
            "ab:a:b"
        );
        // No matches: empty array.
        assert_eq!(eval_to_string(r#""ab".matchAll(/z/g).length"#), "0");
        assert_eq!(
            eval_to_string(r#""ab".matchAll(/z/g) instanceof Array"#),
            "true"
        );
    }

    #[test]
    fn match_all_coerces_with_global_flag() {
        // Spec 22.1.3.16 step 4: `RegExpCreate(pattern, "g")`. A string
        // pattern has no `@@matchAll` hook, so it coerces to a global
        // regexp and matches instead of throwing.
        assert_eq!(
            eval_to_string(r#""aba".matchAll("a").map(m => m.join(":")).join(";")"#),
            "a;a"
        );
        // The constructed regexp is genuinely global: capture groups ride
        // along and every occurrence is found.
        assert_eq!(
            eval_to_string(r#""ab".matchAll("a(b)").map(m => m.join(":")).join(";")"#),
            "ab:b"
        );
    }

    #[test]
    fn match_all_non_global_regexp_throws() {
        // Spec 22.1.3.16 step 3b: a regexp pattern inherits `@@matchAll`
        // from `%RegExp.prototype%` (literals chain to it since the
        // prototype-wiring fix), so the hook runs and its global-only
        // check throws `TypeError` — step 4's `RegExpCreate(pattern, "g")`
        // never applies to regexp patterns.
        for src in [
            r#""aba".matchAll(/a/)"#,
            r#""aba".matchAll(new RegExp("a"))"#,
            r#""aba".matchAll(/(a)(b)/)"#,
        ] {
            let mut vm = crate::Vm::new();
            assert!(
                vm.eval(src).is_err(),
                "non-global regexp pattern must throw: {src}"
            );
        }
        // Global regexps still match, including literals.
        assert_eq!(
            eval_to_string(r#""aba".matchAll(/a/g).map(m => m.join(":")).join(";")"#),
            "a;a"
        );
        assert_eq!(
            eval_to_string(r#""ab".matchAll(/(a)(b)/g)[0].join(":")"#),
            "ab:a:b"
        );
        // An explicit `@@matchAll` hook still wins over the inherited one.
        assert_eq!(
            eval_to_string(
                r#"var o = {}; o[Symbol.matchAll] = function(s) { return "hooked:" + s; }; "ab".matchAll(o)"#
            ),
            "hooked:ab"
        );
    }

    #[test]
    fn get_method_invokes_accessor_hooks() {
        // A getter-defined `@@matchAll` hook is invoked, not skipped.
        assert_eq!(
            eval_to_string(
                r#"var o = {};
                   Object.defineProperty(o, Symbol.matchAll, { get: function() { return function(s) { return "hooked:" + s; }; } });
                   "ab".matchAll(o)"#
            ),
            "hooked:ab"
        );
        // The hook resolves through the prototype chain, with the pattern
        // as receiver.
        assert_eq!(
            eval_to_string(
                r#"var p = {};
                   Object.defineProperty(p, Symbol.match, { get: function() { return function(s) { return this === o ? "mine:" + s : "wrong"; }; } });
                   var o = Object.create(p);
                   "ab".match(o)"#
            ),
            "mine:ab"
        );
        // A getter returning `undefined` behaves like an absent hook: the
        // `RegExpCreate` fallback runs instead of throwing (here the
        // coerced `"[object Object]"` finds nothing in `"xyz"`).
        assert_eq!(
            eval_to_string(
                r#"var o = {};
                   Object.defineProperty(o, Symbol.search, { get: function() { return undefined; } });
                   "xyz".search(o)"#
            ),
            "-1"
        );
    }

    #[test]
    fn get_method_rejects_non_callable_getter() {
        // `Object.defineProperty` itself rejects non-callable getters, so
        // install the accessor directly: spec `OrdinaryGet` throws
        // `TypeError` for a non-callable getter.
        let (mut heap, _global) = setup();
        let o = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let sym = heap.well_known_symbol(WellKnown::Match).unwrap();
        crate::builtin_object_array::define_accessor_prop(
            &mut heap,
            o,
            PropKey::Symbol(sym),
            crate::value::AccessorProp::new(Value::Int(5), Value::Undefined),
        )
        .unwrap();
        assert!(
            matches!(
                get_method(&mut heap, Value::Object(o), WellKnown::Match),
                Err(NativeError::TypeError(_))
            ),
            "non-callable getter should throw TypeError"
        );
    }

    #[test]
    fn match_dispatches_to_string_prototype_hook_for_primitive_pattern() {
        // Spec `GetMethod` boxes the primitive pattern (`ToObject`), so a
        // hook installed on `String.prototype` fires for string patterns.
        assert_eq!(
            eval_to_string(
                r#"String.prototype[Symbol.match] = function(s) { return "m:" + s; }; "ab".match("x")"#
            ),
            "m:ab"
        );
    }

    #[test]
    fn match_all_dispatches_to_string_prototype_hook_for_primitive_pattern() {
        // Spec `GetMethod` boxes the primitive pattern (`ToObject`), so a
        // hook installed on `String.prototype` fires for string patterns.
        assert_eq!(
            eval_to_string(
                r#"String.prototype[Symbol.matchAll] = function(s) { return "hooked:" + s; }; "ab".matchAll("a")"#
            ),
            "hooked:ab"
        );
    }

    #[test]
    fn search_dispatches_to_string_prototype_hook_for_primitive_pattern() {
        // Spec `GetMethod` boxes the primitive pattern (`ToObject`), so a
        // hook installed on `String.prototype` fires for string patterns.
        assert_eq!(
            eval_to_string(
                r#"String.prototype[Symbol.search] = function(s) { return 42; }; "abc".search("b")"#
            ),
            "42"
        );
    }

    #[test]
    fn search_returns_index() {
        assert_eq!(eval_to_string(r#""abc".search(/b/)"#), "1");
        // String patterns coerce (used to throw `TypeError`).
        assert_eq!(eval_to_string(r#""abc".search("b")"#), "1");
        assert_eq!(eval_to_string(r#""abc".search(/z/)"#), "-1");
        // `@@search` starts at 0 and restores `lastIndex` (spec 22.2.9.13).
        assert_eq!(
            eval_to_string(
                r#"var rx = /b/g; rx.lastIndex = 2; "abc".search(rx) + "|" + rx.lastIndex;"#
            ),
            "1|2"
        );
        // Custom `@@search` hook.
        assert_eq!(
            eval_to_string(
                r#"var o = {}; o[Symbol.search] = function(s) { return 42; }; "abc".search(o)"#
            ),
            "42"
        );
    }
}
