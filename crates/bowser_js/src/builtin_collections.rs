//! `Map`, `Set`, `WeakMap`, `WeakSet`, and `Symbol` builtins.
//!
//! Installs the five constructors on the global object via [`install`].
//! Collection storage is heap-allocated and GC-visible: each instance keeps
//! its entries in a hidden, non-enumerable slot (a `"\0bowser:..."`-prefixed
//! string key) holding an [`ObjectKind::Array`] of alternating keys/values
//! (Map, WeakMap) or plain values (Set, WeakSet). Because the entries array
//! is a regular object property, the [`Tracer`](crate::gc::Tracer) marks
//! every stored key and value transitively. The intrinsic prototypes and
//! the `Symbol.for` registry live in [`Heap`] fields and are traced as roots
//! by [`Heap::collect`](crate::gc::Heap::collect).
//!
//! ## Rooting
//!
//! The collector is real: any allocation may trigger one. Every native
//! function below that allocates wraps itself in a [`CallRoots`] guard and
//! roots `this`, its arguments, and every fresh handle it keeps using
//! across a later allocation point, per the rooting discipline in
//! [`crate::gc`]. Helpers that only read (`map_find`, `dense_get`, ...)
//! allocate nothing and need no guard; helpers that allocate
//! (`map_set_impl`, `dense_push`, ...) must be called with the relevant
//! handles already rooted by their caller.
//!
//! Key equality is `SameValueZero`: `NaN` equals `NaN`, `-0` equals `+0`,
//! strings compare by contents, objects/symbols by identity. Iteration
//! follows insertion order: updates rewrite values in place, deletes remove
//! the pair, and iteration skips nothing because deleted pairs are
//! physically removed.
//!
//! ## v1 gaps (for the vm worker and future workers)
//!
//! - **WeakMap/WeakSet hold strong references.** There is no ephemeron
//!   support: entries are never collected (the arena GC never collects
//!   anything yet either). Known gap vs. the spec.
//! - **`size` is a synced data property on `Map`/`Set` only**, not an
//!   accessor. It is non-writable/non-enumerable and rewritten on every
//!   mutation. `WeakMap`/`WeakSet` have no `size` at all (per spec).
//! - **Iterator objects for `Symbol.iterator` only.** `Map.prototype[Symbol.iterator]`
//!   and `Set.prototype[Symbol.iterator]` return real iterator objects
//!   (`next()` + `Symbol.iterator`, so `for`-`of`, spread, and destructuring all
//!   work). `keys`/`values`/`entries` themselves still return fresh `Array`s
//!   (v1, following the sibling `builtin_object_array` convention), so
//!   `Map.prototype[Symbol.iterator] === Map.prototype.entries` does not
//!   hold yet.
//! - **`new.target` is invisible.** `NativeFn` cannot distinguish
//!   `new Map()` from `Map()` (both take the construct path), nor
//!   `new Symbol()` from `Symbol()` (the spec throws for `new Symbol()`).
//! - **Callback re-entry via the shared choke point.** `forEach` invokes
//!   callbacks through the sibling array worker's
//!   [`crate::builtin_object_array::call_function`]: native callbacks run
//!   directly, bytecode callbacks re-enter the VM via
//!   [`crate::vm::reenter_call`].
//! - **Constructor iterables are array-likes only.** All four constructors
//!   accept objects with a `length` and indexed properties (`new Map(pairs)`,
//!   `new Set(values)`, `new WeakMap(pairs)`, `new WeakSet(values)`); there
//!   is no iterator protocol to consume yet. WeakMap keys and WeakSet
//!   values must be objects.
//! - **No `%Object.prototype%` linkage yet.** Prototype objects created here
//!   chain to `null` until the object worker's intrinsics land; result
//!   arrays are `Array`-kind objects with a `null` prototype for the same
//!   reason.
//! - **`Symbol.prototype` accessors.** `description` is a getter-only
//!   accessor (non-enumerable, configurable); `toString`/`valueOf` are
//!   provided. `Symbol.prototype[Symbol.toStringTag]` is `"Symbol"` under
//!   the real well-known symbol key. `Symbol()` never boxes into a
//!   wrapper object.
//! - **BigInt keys** compare by sign plus trailing-zero-trimmed magnitude,
//!   which matches `SameValueZero` for normalized bigints.

use crate::builtin_string_number::{define_native_key, prim_slot};
use crate::gc::{AllocError, Gc, HandleScope, Heap, PropKey, WellKnown};
use crate::value::{
    AccessorProp, BigInt, NativeError, NativeFn, Object, ObjectKind, Property, ShapeId, Str, Value,
};
use crate::wtf8::{Wtf8, Wtf8Buf};

// ---------------------------------------------------------------------------
// Internal slots (hidden, non-enumerable own properties)
// ---------------------------------------------------------------------------

/// Map instance -> entries array (`[k0, v0, k1, v1, ...]`).
const MAP_ENTRIES: &str = "\u{0}bowser:map-entries";
/// Set instance -> values array.
const SET_VALUES: &str = "\u{0}bowser:set-values";
/// WeakMap instance -> entries array (`[k0, v0, ...]`).
const WEAKMAP_ENTRIES: &str = "\u{0}bowser:weakmap-entries";
/// WeakSet instance -> values array.
const WEAKSET_VALUES: &str = "\u{0}bowser:weakset-values";

// ---------------------------------------------------------------------------
// Small helpers (module-private; siblings define their own copies)
// ---------------------------------------------------------------------------

/// RAII root scope for native implementations.
///
/// Any handle a builtin keeps using across an allocation it performs must be
/// rooted: allocations can trigger [`Heap::collect`], which frees everything
/// unreachable from a root. Handles added here stay alive until the guard is
/// dropped at the end of the native call.
struct CallRoots {
    scope: HandleScope,
}

impl CallRoots {
    fn new(heap: &mut Heap) -> Self {
        // SAFETY: `heap` is a valid, live, single-threaded `Heap`. No
        // collection can run concurrently with scope setup, and the scope is
        // dropped before the native call returns, hence strictly before the
        // heap itself can be dropped. `addr_of_mut!(*heap)` (not a `&mut`
        // reborrow) per `HandleScope::new`'s contract, so later `&mut heap`
        // uses by the caller do not pop the reborrow tag the scope holds.
        let scope = unsafe { HandleScope::new(std::ptr::addr_of_mut!(*heap)) };
        CallRoots { scope }
    }

    /// Root `g`; the referent stays alive until `self` is dropped.
    fn root<T>(&mut self, g: Gc<T>)
    where
        Gc<T>: Into<Value>,
    {
        self.scope.add(g);
    }

    /// Root whatever heap payload `v` carries, if any.
    fn root_value(&mut self, v: Value) {
        self.scope.add_value(v);
    }

    /// Root `this` and every argument: the standard prologue for a native
    /// function that allocates before reading them again.
    fn root_call(&mut self, this: Value, args: &[Value]) {
        self.root_value(this);
        for &a in args {
            self.root_value(a);
        }
    }
}

/// Argument `i`, or `undefined` when the caller passed fewer.
fn get_arg(args: &[Value], i: usize) -> Value {
    args.get(i).copied().unwrap_or(Value::Undefined)
}

/// Build the cheapest numeric value: [`Value::Int`] when `n` is integral and
/// fits `i32` (preserving `-0` as `Float`), else [`Value::Float`].
fn num(n: f64) -> Value {
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

/// An ordinary object with an explicit prototype (`Object::new` only makes
/// null-prototype ones).
fn ordinary_object(prototype: Value) -> Object {
    Object {
        kind: ObjectKind::Ordinary,
        prototype,
        properties: Vec::new(),
        extensible: true,
        accessors: Vec::new(),
        symbol_properties: Vec::new(),
        symbol_accessors: Vec::new(),
        cold: None,
        invoke_flags: Object::INVOKE_PLAIN,
        shape: ShapeId::ROOT,
    }
}

/// Allocate a native function object with spec-shaped `name`/`length`.
fn native_fn(
    heap: &mut Heap,
    name: &str,
    arity: u16,
    f: NativeFn,
) -> Result<Gc<Object>, AllocError> {
    let mut roots = CallRoots::new(heap);
    let func = heap.alloc_object(Object::new(ObjectKind::NativeFunction(f)))?;
    roots.root(func);
    let name_str = heap.alloc_str(name)?;
    roots.root_value(Value::Str(name_str));
    heap.define_property_str(
        func,
        "name",
        Property {
            value: Value::Str(name_str),
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )?;
    heap.define_property_str(
        func,
        "length",
        Property {
            value: Value::Int(i32::from(arity)),
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )?;
    Ok(func)
}

/// Define a prototype method: writable, non-enumerable, configurable.
fn define_method(
    heap: &mut Heap,
    proto: Gc<Object>,
    name: &str,
    arity: u16,
    f: NativeFn,
) -> Result<(), AllocError> {
    let mut roots = CallRoots::new(heap);
    roots.root(proto);
    let func = native_fn(heap, name, arity, f)?;
    roots.root(func);
    heap.define_property_str(
        proto,
        name,
        Property {
            value: Value::Object(func),
            writable: true,
            enumerable: false,
            configurable: true,
        },
    )
}

/// Install `proto[Symbol.toStringTag] = tag`: a non-writable,
/// non-enumerable, configurable data property under the real well-known
/// symbol key (ECMA-262 20.4.3.x for `%Symbol.prototype%`, 24.x for the
/// collection prototypes). `Object.prototype.toString` reads it via
/// [`get_prop_key`](crate::builtin_object_array::get_prop_key).
fn define_to_string_tag(heap: &mut Heap, proto: Gc<Object>, tag: &str) -> Result<(), AllocError> {
    let sym = heap.well_known_symbol(WellKnown::ToStringTag)?;
    let tag_str = heap.alloc_str(tag)?;
    heap.define_property_key(
        proto,
        PropKey::Symbol(sym),
        Property {
            value: Value::Str(tag_str),
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )
}

/// Build a constructor: a prototype object holding `methods`, statics on the
/// constructor, `prototype`/`constructor` links both ways, and the
/// constructor installed on `global` as a non-enumerable property.
fn install_ctor(
    heap: &mut Heap,
    global: Gc<Object>,
    name: &str,
    ctor_arity: u16,
    ctor: NativeFn,
    methods: &[(&str, u16, NativeFn)],
    statics: &[(&str, u16, NativeFn)],
) -> Result<(Gc<Object>, Gc<Object>), AllocError> {
    let mut roots = CallRoots::new(heap);
    roots.root(global);
    let proto = heap.alloc_object(ordinary_object(Value::Null))?;
    roots.root(proto);
    for (mname, arity, f) in methods {
        define_method(heap, proto, mname, *arity, *f)?;
    }
    let ctor_obj = native_fn(heap, name, ctor_arity, ctor)?;
    roots.root(ctor_obj);
    // `Map`/`Set`/`WeakMap`/`WeakSet` are constructors (ECMA-262
    // `IsConstructor`): record the implementation in the heap's
    // constructor registry.
    heap.mark_native_constructor(ctor);
    heap.define_property_str(
        ctor_obj,
        "prototype",
        Property {
            value: Value::Object(proto),
            writable: false,
            enumerable: false,
            configurable: false,
        },
    )?;
    for (sname, arity, f) in statics {
        let sf = native_fn(heap, sname, *arity, *f)?;
        heap.define_property_str(
            ctor_obj,
            sname,
            Property {
                value: Value::Object(sf),
                writable: true,
                enumerable: false,
                configurable: true,
            },
        )?;
    }
    heap.define_property_str(
        proto,
        "constructor",
        Property {
            value: Value::Object(ctor_obj),
            writable: true,
            enumerable: false,
            configurable: true,
        },
    )?;
    heap.define_property_str(
        global,
        name,
        Property {
            value: Value::Object(ctor_obj),
            writable: true,
            enumerable: false,
            configurable: true,
        },
    )?;
    Ok((ctor_obj, proto))
}

/// Read the hidden internal slot of a collection instance.
fn internal_slot(
    heap: &Heap,
    obj: Gc<Object>,
    slot: &str,
    owner: &str,
) -> Result<Gc<Object>, NativeError> {
    match heap.get_property(obj, Wtf8::from_str(slot)) {
        Some(p) => match p.value {
            Value::Object(arr) => Ok(arr),
            _ => Err(NativeError::TypeError(format!(
                "{owner}: internal slot corrupted"
            ))),
        },
        None => Err(NativeError::TypeError(format!(
            "{owner} called on incompatible receiver"
        ))),
    }
}

/// Keep the `size` data property in sync after a mutation.
fn set_size(heap: &mut Heap, obj: Gc<Object>, n: u64) -> Result<(), AllocError> {
    heap.define_property_str(
        obj,
        "size",
        Property {
            value: num(n as f64),
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )
}

// ---------------------------------------------------------------------------
// SameValueZero
// ---------------------------------------------------------------------------

fn trim_zeros(d: &[u32]) -> &[u32] {
    let mut end = d.len();
    while end > 0 && d[end - 1] == 0 {
        end -= 1;
    }
    &d[..end]
}

/// BigInt key equality: `-0n` equals `0n`; otherwise sign plus
/// trailing-zero-trimmed magnitude must match.
fn bigint_key_eq(x: Gc<BigInt>, y: Gc<BigInt>) -> bool {
    let xd = trim_zeros(&x.digits);
    let yd = trim_zeros(&y.digits);
    if xd.is_empty() && yd.is_empty() {
        return true;
    }
    x.negative == y.negative && xd == yd
}

/// `SameValueZero`: like `===`, but `NaN` equals `NaN` (and `-0` equals
/// `+0`, which Rust's `==` already gives). Strings compare by contents, not
/// by handle identity.
fn same_value_zero(a: Value, b: Value) -> bool {
    match (a, b) {
        (Value::Undefined, Value::Undefined) | (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Int(x), Value::Float(y)) => f64::from(x) == y,
        (Value::Float(x), Value::Int(y)) => x == f64::from(y),
        (Value::Float(x), Value::Float(y)) => x == y || (x.is_nan() && y.is_nan()),
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Object(x), Value::Object(y)) => x == y,
        (Value::Symbol(x), Value::Symbol(y)) => x == y,
        (Value::BigInt(x), Value::BigInt(y)) => bigint_key_eq(x, y),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Dense array storage (Array-kind objects with indexed properties)
//
// The entries/values arrays behind every collection are written only by
// this module, which upholds two invariants the readers below rely on:
//
// * `"length"` is defined first and only ever overwritten in place, so it
//   stays at properties index 0 and [`dense_len`] is O(1).
// * Numeric keys are dense (`0..len`), canonical (`u64::to_string` output),
//   and stored in ascending order behind `"length"`, so index `i` usually
//   lives at properties position `i + 1` (the [`dense_get`] fast path).
//
// Every reader parses keys with [`parse_index`] and ignores indices `>=
// len`, so they stay correct even if a stale key ever lingers.
// ---------------------------------------------------------------------------

/// Parse a canonical array-index key (exactly what `u64::to_string`
/// produces) without allocating. `None` for anything else (`"length"`,
/// hidden slot names, ...).
fn parse_index(key: &Wtf8) -> Option<u64> {
    let digits = key.as_bytes();
    if digits.is_empty() || digits.len() > 20 {
        return None;
    }
    if digits[0] == b'0' {
        // Canonical form has no leading zeros: only `"0"` itself parses.
        return if digits.len() == 1 { Some(0) } else { None };
    }
    let mut n: u64 = 0;
    for &d in digits {
        if !d.is_ascii_digit() {
            return None;
        }
        n = n.checked_mul(10)?.checked_add(u64::from(d - b'0'))?;
    }
    Some(n)
}

/// Format `n` as decimal into a stack buffer; the result borrows `buf`.
/// No allocation: [`Heap::define_property`] interns the key itself.
fn fmt_index(n: u64, buf: &mut [u8; 20]) -> &str {
    let mut i = buf.len();
    let mut m = n;
    loop {
        i -= 1;
        buf[i] = b'0' + (m % 10) as u8;
        m /= 10;
        if m == 0 {
            break;
        }
    }
    std::str::from_utf8(&buf[i..]).expect("fmt_index emits ASCII digits")
}

/// A fresh `Array`-kind object with a `length` data property.
fn new_dense_array(heap: &mut Heap) -> Result<Gc<Object>, AllocError> {
    let arr = heap.alloc_object(Object {
        kind: ObjectKind::Array,
        prototype: Value::Null,
        properties: Vec::new(),
        extensible: true,
        accessors: Vec::new(),
        symbol_properties: Vec::new(),
        symbol_accessors: Vec::new(),
        cold: None,
        invoke_flags: Object::INVOKE_PLAIN,
        shape: ShapeId::ROOT,
    })?;
    dense_set_len(heap, arr, 0)?;
    Ok(arr)
}

/// Read a `length`-style numeric value as `u64`.
fn len_value(v: Value) -> u64 {
    match v {
        Value::Int(i) if i >= 0 => i as u64,
        Value::Float(f) if f >= 0.0 => f as u64,
        _ => 0,
    }
}

/// Current `length` of a dense array: O(1) through the `"length"`-at-index-0
/// invariant, with a full scan as backup.
fn dense_len(heap: &Heap, arr: Gc<Object>) -> u64 {
    let props = &heap.get(arr).properties;
    if let Some(p) = props
        .first()
        .filter(|(k, _)| k.as_wtf8() == "length")
        .map(|(_, p)| p)
    {
        return len_value(p.value);
    }
    props
        .iter()
        .find(|(k, _)| k.as_wtf8() == "length")
        .map(|(_, p)| len_value(p.value))
        .unwrap_or(0)
}

/// Read index `i`; `None` when absent. Never allocates: the positional fast
/// path covers the dense layout our writers maintain, and the fallback is a
/// single parse-and-compare scan.
fn dense_get(heap: &Heap, arr: Gc<Object>, i: u64) -> Option<Value> {
    let props = &heap.get(arr).properties;
    if let Some(v) = usize::try_from(i)
        .ok()
        .and_then(|i| i.checked_add(1))
        .and_then(|pos| props.get(pos))
        .filter(|(k, _)| parse_index(k.as_wtf8()) == Some(i))
        .map(|(_, p)| p.value)
    {
        return Some(v);
    }
    props
        .iter()
        .find_map(|(k, p)| (parse_index(k.as_wtf8()) == Some(i)).then_some(p.value))
}

/// Overwrite the `length` data property (standard attributes).
fn dense_set_len(heap: &mut Heap, arr: Gc<Object>, len: u64) -> Result<(), AllocError> {
    heap.define_property_str(
        arr,
        "length",
        Property {
            value: num(len as f64),
            writable: true,
            enumerable: false,
            configurable: false,
        },
    )
}

/// Append `v` at the end. The key is formatted on a stack buffer and
/// interned by `define_property` (a repeat index reuses the interned key),
/// so no `String` is allocated.
fn dense_push(heap: &mut Heap, arr: Gc<Object>, v: Value) -> Result<(), AllocError> {
    let len = dense_len(heap, arr);
    let mut buf = [0u8; 20];
    // `buf` is a caller local: no borrow conflict with `&mut heap`.
    heap.define_property_str(arr, fmt_index(len, &mut buf), Property::data(v))?;
    dense_set_len(heap, arr, len + 1)
}

/// Remove the element at `idx`, shifting later elements down. Allocation
/// free: values are read through the O(1) [`dense_get`] fast path and keys
/// are formatted on a stack buffer.
fn dense_remove_at(heap: &mut Heap, arr: Gc<Object>, idx: u64) -> Result<(), AllocError> {
    let len = dense_len(heap, arr);
    if idx >= len {
        return Ok(());
    }
    for i in (idx + 1)..len {
        let v = dense_get(heap, arr, i).unwrap_or(Value::Undefined);
        let mut buf = [0u8; 20];
        heap.define_property_str(arr, fmt_index(i - 1, &mut buf), Property::data(v))?;
    }
    let mut buf = [0u8; 20];
    heap.delete_property(arr, Wtf8::from_str(fmt_index(len - 1, &mut buf)));
    dense_set_len(heap, arr, len - 1)
}

/// First live index whose element satisfies `want`, scanning the raw
/// property list once. No per-element lookup, no allocation.
fn dense_find(
    heap: &Heap,
    arr: Gc<Object>,
    len: u64,
    mut want: impl FnMut(u64, Value) -> bool,
) -> Option<u64> {
    heap.get(arr).properties.iter().find_map(|(k, p)| {
        let i = parse_index(k.as_wtf8())?;
        (i < len && want(i, p.value)).then_some(i)
    })
}

/// A fresh dense array holding exactly `values`, in order: one object
/// allocation, keys packed inline (no interning, no heap allocation:
/// `"length"` and decimal indices are all short). Every element of
/// `values` must be reachable from a GC root held by the caller across
/// this call, since allocation may trigger a collection. The keys
/// themselves need no rooting: inline keys carry no heap handle, and any
/// heap key is rooted in `roots` via [`Str::heap_handle`] as it is built.
fn dense_from_values(
    heap: &mut Heap,
    roots: &mut CallRoots,
    values: &[Value],
) -> Result<Gc<Object>, AllocError> {
    let mut props: Vec<(Str, Property)> = Vec::with_capacity(values.len() + 1);
    let length_key = heap.intern_str("length")?;
    // Only `Str::Heap` keys need rooting; inline keys are values.
    roots.root_value(Value::from(length_key));
    props.push((
        length_key,
        Property {
            value: num(values.len() as f64),
            writable: true,
            enumerable: false,
            configurable: false,
        },
    ));
    for (i, &v) in values.iter().enumerate() {
        let mut buf = [0u8; 20];
        let key = heap.intern_str(fmt_index(i as u64, &mut buf))?;
        roots.root_value(Value::from(key));
        props.push((key, Property::data(v)));
    }
    let out = heap.alloc_object(Object {
        kind: ObjectKind::Array,
        prototype: Value::Null,
        properties: props,
        extensible: true,
        accessors: Vec::new(),
        symbol_properties: Vec::new(),
        symbol_accessors: Vec::new(),
        cold: None,
        invoke_flags: Object::INVOKE_PLAIN,
        shape: ShapeId::ROOT,
    })?;
    // The lists were pre-filled: derive the canonical shape for the
    // layout instead of leaving ROOT behind.
    heap.shape_rebuild(out);
    roots.root(out);
    Ok(out)
}

/// Read an own `"length"` property as a `ToLength`-style `u64`.
fn own_length(heap: &Heap, obj: Gc<Object>) -> u64 {
    const MAX: u64 = 9007199254740991;
    match heap.get_property(obj, Wtf8::from_str("length")) {
        Some(p) => match p.value {
            Value::Int(i) if i >= 0 => (i as u64).min(MAX),
            Value::Float(f) if f >= 0.0 && !f.is_nan() => (f as u64).min(MAX),
            Value::Bool(b) => u64::from(b),
            _ => 0,
        },
        None => 0,
    }
}

/// Invoke a callback through the shared choke point: native callees run
/// directly, bytecode ([`ObjectKind::Function`]) callees re-enter the VM via
/// [`crate::vm::reenter_call`].
fn call_native(
    heap: &mut Heap,
    callee: Value,
    this_arg: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    crate::builtin_object_array::call_function(heap, callee, this_arg, args)
}

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------

/// The `(map, entries)` pair behind a `this` value, or `TypeError`.
fn map_parts(heap: &Heap, this: Value) -> Result<(Gc<Object>, Gc<Object>), NativeError> {
    match this {
        Value::Object(o) => Ok((o, internal_slot(heap, o, MAP_ENTRIES, "Map")?)),
        _ => Err(NativeError::TypeError(
            "Map method called on incompatible receiver".into(),
        )),
    }
}

/// Pair index of `key` in the entries array, if present: one pass over the
/// raw property list, no allocation.
fn map_find(heap: &Heap, entries: Gc<Object>, key: Value) -> Option<u64> {
    let len = dense_len(heap, entries);
    dense_find(heap, entries, len, |i, k| {
        i % 2 == 0 && same_value_zero(k, key)
    })
    .map(|i| i / 2)
}

fn map_set_impl(
    heap: &mut Heap,
    map: Gc<Object>,
    entries: Gc<Object>,
    key: Value,
    value: Value,
    track_size: bool,
) -> Result<(), AllocError> {
    match map_find(heap, entries, key) {
        // Update in place: insertion order is unchanged.
        Some(pair) => {
            let mut buf = [0u8; 20];
            heap.define_property_str(
                entries,
                fmt_index(pair * 2 + 1, &mut buf),
                Property::data(value),
            )?;
        }
        None => {
            dense_push(heap, entries, key)?;
            dense_push(heap, entries, value)?;
            // WeakMap has no `size`.
            if track_size {
                set_size(heap, map, dense_len(heap, entries) / 2)?;
            }
        }
    }
    Ok(())
}

fn map_delete_impl(
    heap: &mut Heap,
    map: Gc<Object>,
    entries: Gc<Object>,
    key: Value,
    track_size: bool,
) -> Result<bool, AllocError> {
    match map_find(heap, entries, key) {
        Some(pair) => {
            dense_remove_at(heap, entries, pair * 2)?;
            // The value shifted into the key's slot; remove it too.
            dense_remove_at(heap, entries, pair * 2)?;
            // WeakMap has no `size`.
            if track_size {
                set_size(heap, map, dense_len(heap, entries) / 2)?;
            }
            Ok(true)
        }
        None => Ok(false),
    }
}

fn map_constructor(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    for &a in args {
        roots.root_value(a);
    }
    let proto = heap.map_prototype.map(Value::Object).unwrap_or(Value::Null);
    let map = heap.alloc_object(ordinary_object(proto))?;
    roots.root(map);
    let entries = new_dense_array(heap)?;
    roots.root(entries);
    heap.define_property_str(
        map,
        MAP_ENTRIES,
        Property {
            value: Value::Object(entries),
            writable: false,
            enumerable: false,
            configurable: false,
        },
    )?;
    set_size(heap, map, 0)?;
    // v1: the iterable must be an array-like of `[k, v]` pairs.
    match get_arg(args, 0) {
        Value::Undefined | Value::Null => {}
        Value::Object(o) => {
            let len = own_length(heap, o);
            for i in 0..len {
                let pair = dense_get(heap, o, i).unwrap_or(Value::Undefined);
                let pair_obj = match pair {
                    Value::Object(p) => p,
                    _ => {
                        return Err(NativeError::TypeError(
                            "Map constructor: each entry must be an object".into(),
                        ));
                    }
                };
                let k = dense_get(heap, pair_obj, 0).unwrap_or(Value::Undefined);
                let v = dense_get(heap, pair_obj, 1).unwrap_or(Value::Undefined);
                map_set_impl(heap, map, entries, k, v, true)?;
            }
        }
        _ => {
            return Err(NativeError::TypeError(
                "Map constructor: iterable must be an object (v1: array-like of pairs)".into(),
            ));
        }
    }
    Ok(Value::Object(map))
}

fn map_set(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_call(this, args);
    let (map, entries) = map_parts(heap, this)?;
    map_set_impl(heap, map, entries, get_arg(args, 0), get_arg(args, 1), true)?;
    Ok(this)
}

fn map_get(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (_, entries) = map_parts(heap, this)?;
    Ok(match map_find(heap, entries, get_arg(args, 0)) {
        Some(pair) => dense_get(heap, entries, pair * 2 + 1).unwrap_or(Value::Undefined),
        None => Value::Undefined,
    })
}

fn map_has(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (_, entries) = map_parts(heap, this)?;
    Ok(Value::Bool(
        map_find(heap, entries, get_arg(args, 0)).is_some(),
    ))
}

fn map_delete(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_call(this, args);
    let (map, entries) = map_parts(heap, this)?;
    Ok(Value::Bool(map_delete_impl(
        heap,
        map,
        entries,
        get_arg(args, 0),
        true,
    )?))
}

fn map_clear(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_value(this);
    let (map, _) = map_parts(heap, this)?;
    // Swap in a fresh entries array: O(1), and the dropped elements become
    // collectible instead of lingering as stale properties.
    let entries = new_dense_array(heap)?;
    roots.root(entries);
    heap.define_property_str(
        map,
        MAP_ENTRIES,
        Property {
            value: Value::Object(entries),
            writable: false,
            enumerable: false,
            configurable: false,
        },
    )?;
    set_size(heap, map, 0)?;
    Ok(Value::Undefined)
}

fn map_keys(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_value(this);
    let (_, entries) = map_parts(heap, this)?;
    let len = dense_len(heap, entries);
    // One pass; reads hit the O(1) `dense_get` fast path.
    let mut keys = Vec::with_capacity((len / 2) as usize);
    for i in (0..len).step_by(2) {
        if let Some(k) = dense_get(heap, entries, i) {
            keys.push(k);
        }
    }
    Ok(Value::Object(dense_from_values(heap, &mut roots, &keys)?))
}

fn map_values(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_value(this);
    let (_, entries) = map_parts(heap, this)?;
    let len = dense_len(heap, entries);
    let mut vals = Vec::with_capacity((len / 2) as usize);
    for i in (1..len).step_by(2) {
        if let Some(v) = dense_get(heap, entries, i) {
            vals.push(v);
        }
    }
    Ok(Value::Object(dense_from_values(heap, &mut roots, &vals)?))
}

fn map_entries(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_value(this);
    let (_, entries) = map_parts(heap, this)?;
    let len = dense_len(heap, entries);
    let mut out = Vec::with_capacity((len / 2) as usize);
    for i in (0..len).step_by(2) {
        let k = dense_get(heap, entries, i).unwrap_or(Value::Undefined);
        let v = dense_get(heap, entries, i + 1).unwrap_or(Value::Undefined);
        // `dense_from_values` roots each pair for the rest of this call.
        let pair = dense_from_values(heap, &mut roots, &[k, v])?;
        out.push(Value::Object(pair));
    }
    Ok(Value::Object(dense_from_values(heap, &mut roots, &out)?))
}

fn map_for_each(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_call(this, args);
    let (_, entries) = map_parts(heap, this)?;
    let cb = get_arg(args, 0);
    if matches!(cb, Value::Undefined) {
        return Err(NativeError::TypeError(
            "Map.prototype.forEach requires a callback".into(),
        ));
    }
    let this_arg = get_arg(args, 1);
    // Snapshot the pairs first: `Value` is `Copy`, so the callback may
    // freely mutate the map during iteration. Reads hit the O(1)
    // `dense_get` fast path.
    let len = dense_len(heap, entries);
    let mut pairs: Vec<(Value, Value)> = Vec::with_capacity((len / 2) as usize);
    for i in (0..len).step_by(2) {
        pairs.push((
            dense_get(heap, entries, i).unwrap_or(Value::Undefined),
            dense_get(heap, entries, i + 1).unwrap_or(Value::Undefined),
        ));
    }
    for (k, v) in pairs {
        call_native(heap, cb, this_arg, &[v, k, this])?;
    }
    Ok(Value::Undefined)
}

// ---------------------------------------------------------------------------
// Set
// ---------------------------------------------------------------------------

/// The `(set, values)` pair behind a `this` value, or `TypeError`.
fn set_parts(heap: &Heap, this: Value) -> Result<(Gc<Object>, Gc<Object>), NativeError> {
    match this {
        Value::Object(o) => Ok((o, internal_slot(heap, o, SET_VALUES, "Set")?)),
        _ => Err(NativeError::TypeError(
            "Set method called on incompatible receiver".into(),
        )),
    }
}

/// Index of `v` in the values array, if present: one pass over the raw
/// property list, no allocation.
fn set_find(heap: &Heap, values: Gc<Object>, v: Value) -> Option<u64> {
    let len = dense_len(heap, values);
    dense_find(heap, values, len, |_, e| same_value_zero(e, v))
}

fn set_add_impl(
    heap: &mut Heap,
    set: Gc<Object>,
    values: Gc<Object>,
    v: Value,
    track_size: bool,
) -> Result<(), AllocError> {
    if set_find(heap, values, v).is_none() {
        dense_push(heap, values, v)?;
        // WeakSet has no `size`.
        if track_size {
            set_size(heap, set, dense_len(heap, values))?;
        }
    }
    Ok(())
}

fn set_constructor(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    for &a in args {
        roots.root_value(a);
    }
    let proto = heap.set_prototype.map(Value::Object).unwrap_or(Value::Null);
    let set = heap.alloc_object(ordinary_object(proto))?;
    roots.root(set);
    let values = new_dense_array(heap)?;
    roots.root(values);
    heap.define_property_str(
        set,
        SET_VALUES,
        Property {
            value: Value::Object(values),
            writable: false,
            enumerable: false,
            configurable: false,
        },
    )?;
    set_size(heap, set, 0)?;
    // v1: the iterable must be an array-like of values.
    match get_arg(args, 0) {
        Value::Undefined | Value::Null => {}
        Value::Object(o) => {
            let len = own_length(heap, o);
            for i in 0..len {
                let v = dense_get(heap, o, i).unwrap_or(Value::Undefined);
                set_add_impl(heap, set, values, v, true)?;
            }
        }
        _ => {
            return Err(NativeError::TypeError(
                "Set constructor: iterable must be an object (v1: array-like)".into(),
            ));
        }
    }
    Ok(Value::Object(set))
}

fn set_add(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_call(this, args);
    let (set, values) = set_parts(heap, this)?;
    set_add_impl(heap, set, values, get_arg(args, 0), true)?;
    Ok(this)
}

fn set_has(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (_, values) = set_parts(heap, this)?;
    Ok(Value::Bool(
        set_find(heap, values, get_arg(args, 0)).is_some(),
    ))
}

fn set_delete(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_call(this, args);
    let (set, values) = set_parts(heap, this)?;
    match set_find(heap, values, get_arg(args, 0)) {
        Some(i) => {
            dense_remove_at(heap, values, i)?;
            set_size(heap, set, dense_len(heap, values))?;
            Ok(Value::Bool(true))
        }
        None => Ok(Value::Bool(false)),
    }
}

fn set_clear(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_value(this);
    let (set, _) = set_parts(heap, this)?;
    // Swap in a fresh values array: O(1), and the dropped elements become
    // collectible instead of lingering as stale properties.
    let values = new_dense_array(heap)?;
    roots.root(values);
    heap.define_property_str(
        set,
        SET_VALUES,
        Property {
            value: Value::Object(values),
            writable: false,
            enumerable: false,
            configurable: false,
        },
    )?;
    set_size(heap, set, 0)?;
    Ok(Value::Undefined)
}

/// `keys` and `values` are identical for sets; both return insertion order.
fn set_keys_values(heap: &mut Heap, this: Value) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_value(this);
    let (_, values) = set_parts(heap, this)?;
    let len = dense_len(heap, values);
    // One pass; reads hit the O(1) `dense_get` fast path.
    let mut out = Vec::with_capacity(len as usize);
    for i in 0..len {
        if let Some(v) = dense_get(heap, values, i) {
            out.push(v);
        }
    }
    Ok(Value::Object(dense_from_values(heap, &mut roots, &out)?))
}

fn set_keys(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    set_keys_values(heap, this)
}

fn set_values(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    set_keys_values(heap, this)
}

fn set_entries(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_value(this);
    let (_, values) = set_parts(heap, this)?;
    let len = dense_len(heap, values);
    let mut out = Vec::with_capacity(len as usize);
    for i in 0..len {
        let v = dense_get(heap, values, i).unwrap_or(Value::Undefined);
        // `dense_from_values` roots each pair for the rest of this call.
        let pair = dense_from_values(heap, &mut roots, &[v, v])?;
        out.push(Value::Object(pair));
    }
    Ok(Value::Object(dense_from_values(heap, &mut roots, &out)?))
}

fn set_for_each(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_call(this, args);
    let (_, values) = set_parts(heap, this)?;
    let cb = get_arg(args, 0);
    if matches!(cb, Value::Undefined) {
        return Err(NativeError::TypeError(
            "Set.prototype.forEach requires a callback".into(),
        ));
    }
    let this_arg = get_arg(args, 1);
    // Snapshot first: `Value` is `Copy`, so the callback may freely mutate
    // the set during iteration. Reads hit the O(1) `dense_get` fast path.
    let len = dense_len(heap, values);
    let mut snapshot: Vec<Value> = Vec::with_capacity(len as usize);
    for i in 0..len {
        snapshot.push(dense_get(heap, values, i).unwrap_or(Value::Undefined));
    }
    for v in snapshot {
        call_native(heap, cb, this_arg, &[v, v, this])?;
    }
    Ok(Value::Undefined)
}

// ---------------------------------------------------------------------------
// Map/Set iterator objects (`Symbol.iterator`)
// ---------------------------------------------------------------------------
//
// `Map.prototype[Symbol.iterator]` (=== `entries`) and
// `Set.prototype[Symbol.iterator]` (=== `values`) return iterator objects
// with `next()`/`Symbol.iterator`, so `for`-`of`, spread, and destructuring
// consume Maps and Sets through the iterator protocol. `keys`/`values`/
// `entries` themselves still return fresh `Array`s (v1, covered by the
// existing tests); the `Symbol.iterator` methods are separate natives with the
// same iteration behavior, which is the documented deviation from
// `Map.prototype[Symbol.iterator] === Map.prototype.entries`.
//
// Hidden slots on a collection iterator object:
const ITER_COLLECTION: &str = "\u{0}bowser:iter-collection";
/// The Map/Set instance: the entries array is re-read on every `next()`,
/// so `clear()` (which swaps the array) and `delete()` (which splices in
/// place) stay live, per spec.
const ITER_INDEX: &str = "\u{0}bowser:iter-index";
/// `Int`: 0 = keys, 1 = values, 2 = entries.
const ITER_KIND: &str = "\u{0}bowser:iter-kind";
/// `Int`: 1 = Map layout (`[k0, v0, ...]`), 0 = Set layout (`[v0, ...]`).
const ITER_IS_MAP: &str = "\u{0}bowser:iter-is-map";

/// Allocate a collection iterator over `collection` (already validated).
fn new_collection_iterator(
    heap: &mut Heap,
    collection: Gc<Object>,
    kind: u8,
    is_map: bool,
) -> Result<Gc<Object>, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root(collection);
    let iter = heap
        .alloc_object(ordinary_object(Value::Null))
        .map_err(|_| NativeError::OutOfMemory)?;
    roots.root(iter);
    for (slot, value) in [
        (ITER_COLLECTION, Value::Object(collection)),
        (ITER_INDEX, Value::Int(0)),
        (ITER_KIND, Value::Int(kind as i32)),
        (ITER_IS_MAP, Value::Int(is_map as i32)),
    ] {
        heap.define_property_str(
            iter,
            slot,
            Property {
                value,
                writable: false,
                enumerable: false,
                configurable: false,
            },
        )
        .map_err(|_| NativeError::OutOfMemory)?;
    }
    define_method(heap, iter, "next", 0, collection_iter_next)
        .map_err(|_| NativeError::OutOfMemory)?;
    // Iterators are iterable: `for (x of map[Symbol.iterator]())` — the
    // real symbol key, so a computed `it[Symbol.iterator]` key resolves
    // to the same property.
    let iter_self = native_fn(heap, "[Symbol.iterator]", 0, collection_iter_self)
        .map_err(|_| NativeError::OutOfMemory)?;
    roots.root(iter_self);
    let iterator_sym = heap.well_known_symbol(WellKnown::Iterator)?;
    heap.define_property_key(
        iter,
        PropKey::Symbol(iterator_sym),
        Property {
            value: Value::Object(iter_self),
            writable: true,
            enumerable: false,
            configurable: true,
        },
    )
    .map_err(|_| NativeError::OutOfMemory)?;
    Ok(iter)
}

/// Build a `{value, done}` result object (writable/enumerable/configurable,
/// per spec `CreateIterResultObject`).
fn iter_result(
    heap: &mut Heap,
    roots: &mut CallRoots,
    value: Value,
    done: bool,
) -> Result<Value, NativeError> {
    roots.root_value(value);
    let obj = heap
        .alloc_object(ordinary_object(Value::Null))
        .map_err(|_| NativeError::OutOfMemory)?;
    roots.root(obj);
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
}

fn collection_iter_self(
    _heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    match this {
        Value::Object(_) => Ok(this),
        _ => Err(NativeError::TypeError(
            "@@iterator called on incompatible receiver".into(),
        )),
    }
}

fn collection_iter_next(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_value(this);
    let incompatible =
        || NativeError::TypeError("iterator.next called on incompatible receiver".into());
    let iter = match this {
        Value::Object(o) => o,
        _ => return Err(incompatible()),
    };
    let slot = |name: &str| {
        heap.get_property(iter, Wtf8::from_str(name))
            .map(|p| p.value)
    };
    let (collection, index, kind, is_map) = match (
        slot(ITER_COLLECTION),
        slot(ITER_INDEX),
        slot(ITER_KIND),
        slot(ITER_IS_MAP),
    ) {
        (Some(Value::Object(c)), Some(Value::Int(i)), Some(Value::Int(k)), Some(Value::Int(m))) => {
            (c, i.max(0) as u64, k, m != 0)
        }
        _ => return Err(incompatible()),
    };
    roots.root(collection);
    // Live read: `clear()` swaps the entries array, `delete()` splices it
    // in place — both stay visible to an in-flight iterator.
    let entries = internal_slot(
        heap,
        collection,
        if is_map { MAP_ENTRIES } else { SET_VALUES },
        "iterator",
    )?;
    roots.root(entries);
    let len = dense_len(heap, entries);
    if index >= len {
        return iter_result(heap, &mut roots, Value::Undefined, true);
    }
    let step: u64 = if is_map { 2 } else { 1 };
    let value = match (is_map, kind) {
        (true, 0) => dense_get(heap, entries, index).unwrap_or(Value::Undefined),
        (true, 1) => dense_get(heap, entries, index + 1).unwrap_or(Value::Undefined),
        (true, 2) => {
            let k = dense_get(heap, entries, index).unwrap_or(Value::Undefined);
            let v = dense_get(heap, entries, index + 1).unwrap_or(Value::Undefined);
            Value::Object(dense_from_values(heap, &mut roots, &[k, v])?)
        }
        (false, 2) => {
            let v = dense_get(heap, entries, index).unwrap_or(Value::Undefined);
            Value::Object(dense_from_values(heap, &mut roots, &[v, v])?)
        }
        // Set keys/values, and any out-of-range kind (defensive): the value.
        _ => dense_get(heap, entries, index).unwrap_or(Value::Undefined),
    };
    roots.root_value(value);
    heap.define_property_str(
        iter,
        ITER_INDEX,
        Property {
            value: num((index + step) as f64),
            writable: false,
            enumerable: false,
            configurable: false,
        },
    )
    .map_err(|_| NativeError::OutOfMemory)?;
    iter_result(heap, &mut roots, value, false)
}

/// `Map.prototype[Symbol.iterator]`: an entries iterator.
fn map_iterator(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_value(this);
    let (map, _) = map_parts(heap, this)?;
    roots.root(map);
    new_collection_iterator(heap, map, 2, true).map(Value::Object)
}

/// `Set.prototype[Symbol.iterator]`: a values iterator.
fn set_iterator(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_value(this);
    let (set, _) = set_parts(heap, this)?;
    roots.root(set);
    new_collection_iterator(heap, set, 1, false).map(Value::Object)
}

// ---------------------------------------------------------------------------
// WeakMap (v1: strong references; see module docs)
// ---------------------------------------------------------------------------

fn weakmap_parts(heap: &Heap, this: Value) -> Result<(Gc<Object>, Gc<Object>), NativeError> {
    match this {
        Value::Object(o) => Ok((o, internal_slot(heap, o, WEAKMAP_ENTRIES, "WeakMap")?)),
        _ => Err(NativeError::TypeError(
            "WeakMap method called on incompatible receiver".into(),
        )),
    }
}

fn weakmap_set(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_call(this, args);
    let (map, entries) = weakmap_parts(heap, this)?;
    match get_arg(args, 0) {
        Value::Object(_) => {}
        _ => {
            return Err(NativeError::TypeError(
                "WeakMap key must be an object".into(),
            ));
        }
    }
    map_set_impl(
        heap,
        map,
        entries,
        get_arg(args, 0),
        get_arg(args, 1),
        false,
    )?;
    Ok(this)
}

fn weakmap_get(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let key = get_arg(args, 0);
    if !matches!(key, Value::Object(_)) {
        return Ok(Value::Undefined);
    }
    let (_, entries) = weakmap_parts(heap, this)?;
    Ok(match map_find(heap, entries, key) {
        Some(pair) => dense_get(heap, entries, pair * 2 + 1).unwrap_or(Value::Undefined),
        None => Value::Undefined,
    })
}

fn weakmap_has(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let key = get_arg(args, 0);
    if !matches!(key, Value::Object(_)) {
        return Ok(Value::Bool(false));
    }
    let (_, entries) = weakmap_parts(heap, this)?;
    Ok(Value::Bool(map_find(heap, entries, key).is_some()))
}

fn weakmap_delete(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_call(this, args);
    let key = get_arg(args, 0);
    if !matches!(key, Value::Object(_)) {
        return Ok(Value::Bool(false));
    }
    let (map, entries) = weakmap_parts(heap, this)?;
    Ok(Value::Bool(map_delete_impl(
        heap, map, entries, key, false,
    )?))
}

fn weakmap_constructor(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    for &a in args {
        roots.root_value(a);
    }
    let proto = heap
        .weakmap_prototype
        .map(Value::Object)
        .unwrap_or(Value::Null);
    let map = heap.alloc_object(ordinary_object(proto))?;
    roots.root(map);
    let entries = new_dense_array(heap)?;
    roots.root(entries);
    heap.define_property_str(
        map,
        WEAKMAP_ENTRIES,
        Property {
            value: Value::Object(entries),
            writable: false,
            enumerable: false,
            configurable: false,
        },
    )?;
    // v1: array-like of `[k, v]` pairs only; keys must be objects.
    match get_arg(args, 0) {
        Value::Undefined | Value::Null => {}
        Value::Object(o) => {
            let len = own_length(heap, o);
            for i in 0..len {
                let pair = dense_get(heap, o, i).unwrap_or(Value::Undefined);
                let pair_obj = match pair {
                    Value::Object(p) => p,
                    _ => {
                        return Err(NativeError::TypeError(
                            "WeakMap constructor: each entry must be an object".into(),
                        ));
                    }
                };
                let k = dense_get(heap, pair_obj, 0).unwrap_or(Value::Undefined);
                if !matches!(k, Value::Object(_)) {
                    return Err(NativeError::TypeError(
                        "WeakMap key must be an object".into(),
                    ));
                }
                let v = dense_get(heap, pair_obj, 1).unwrap_or(Value::Undefined);
                map_set_impl(heap, map, entries, k, v, false)?;
            }
        }
        _ => {
            return Err(NativeError::TypeError(
                "WeakMap constructor: iterable must be an object (v1: array-like of pairs)".into(),
            ));
        }
    }
    Ok(Value::Object(map))
}

// ---------------------------------------------------------------------------
// WeakSet (v1: strong references; see module docs)
// ---------------------------------------------------------------------------

fn weakset_parts(heap: &Heap, this: Value) -> Result<(Gc<Object>, Gc<Object>), NativeError> {
    match this {
        Value::Object(o) => Ok((o, internal_slot(heap, o, WEAKSET_VALUES, "WeakSet")?)),
        _ => Err(NativeError::TypeError(
            "WeakSet method called on incompatible receiver".into(),
        )),
    }
}

fn weakset_add(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_call(this, args);
    let (set, values) = weakset_parts(heap, this)?;
    match get_arg(args, 0) {
        Value::Object(_) => {}
        _ => {
            return Err(NativeError::TypeError(
                "WeakSet value must be an object".into(),
            ));
        }
    }
    set_add_impl(heap, set, values, get_arg(args, 0), false)?;
    Ok(this)
}

fn weakset_has(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let v = get_arg(args, 0);
    if !matches!(v, Value::Object(_)) {
        return Ok(Value::Bool(false));
    }
    let (_, values) = weakset_parts(heap, this)?;
    Ok(Value::Bool(set_find(heap, values, v).is_some()))
}

fn weakset_delete(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_call(this, args);
    let v = get_arg(args, 0);
    if !matches!(v, Value::Object(_)) {
        return Ok(Value::Bool(false));
    }
    let (_, values) = weakset_parts(heap, this)?;
    match set_find(heap, values, v) {
        Some(i) => {
            dense_remove_at(heap, values, i)?;
            // WeakSet has no `size`.
            Ok(Value::Bool(true))
        }
        None => Ok(Value::Bool(false)),
    }
}

fn weakset_constructor(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    for &a in args {
        roots.root_value(a);
    }
    let proto = heap
        .weakset_prototype
        .map(Value::Object)
        .unwrap_or(Value::Null);
    let set = heap.alloc_object(ordinary_object(proto))?;
    roots.root(set);
    let values = new_dense_array(heap)?;
    roots.root(values);
    heap.define_property_str(
        set,
        WEAKSET_VALUES,
        Property {
            value: Value::Object(values),
            writable: false,
            enumerable: false,
            configurable: false,
        },
    )?;
    // v1: the iterable must be an array-like of values; each must be an object.
    // (WeakSet has no `size`.)
    match get_arg(args, 0) {
        Value::Undefined | Value::Null => {}
        Value::Object(o) => {
            let len = own_length(heap, o);
            for i in 0..len {
                let v = dense_get(heap, o, i).unwrap_or(Value::Undefined);
                if !matches!(v, Value::Object(_)) {
                    return Err(NativeError::TypeError(
                        "WeakSet value must be an object".into(),
                    ));
                }
                set_add_impl(heap, set, values, v, false)?;
            }
        }
        _ => {
            return Err(NativeError::TypeError(
                "WeakSet constructor: iterable must be an object (v1: array-like)".into(),
            ));
        }
    }
    Ok(Value::Object(set))
}

// ---------------------------------------------------------------------------
// Symbol
// ---------------------------------------------------------------------------

/// Shortest-roundtrip `Number::toString` for symbol descriptions. (The full
/// spec digit-placement lives in the sibling string/number worker; a
/// description string does not need it.)
fn number_to_string(n: f64) -> String {
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
    if n.fract() == 0.0 && n.abs() < 1e21 {
        // Integral in the exact-integer range of f64.
        return format!("{}", n as i64);
    }
    format!("{n}")
}

/// `ToString` for symbol descriptions. v1 gaps: objects have no `ToPrimitive`
/// dispatch, bigints are unsupported.
fn symbol_desc_string(v: Value) -> Result<Wtf8Buf, NativeError> {
    match v {
        Value::Undefined => Ok(Wtf8Buf::from("undefined")),
        Value::Null => Ok(Wtf8Buf::from("null")),
        Value::Bool(b) => Ok(Wtf8Buf::from(if b { "true" } else { "false" })),
        Value::Int(i) => Ok(Wtf8Buf::from(&i.to_string())),
        Value::Float(f) => Ok(Wtf8Buf::from(&number_to_string(f))),
        Value::Str(s) => Ok(s.as_wtf8().to_owned()),
        Value::Symbol(_) => Err(NativeError::TypeError(
            "cannot convert symbol to string".into(),
        )),
        Value::BigInt(_) => Err(NativeError::TypeError(
            "cannot convert bigint to string (v1 gap)".into(),
        )),
        Value::Object(_) => Err(NativeError::TypeError(
            "cannot convert object to string (v1 gap: no ToPrimitive)".into(),
        )),
    }
}

fn symbol_constructor(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    for &a in args {
        roots.root_value(a);
    }
    // v1 gap: `new.target` is invisible, so `new Symbol()` (which the spec
    // forbids) takes the same path as `Symbol()`.
    let desc = match get_arg(args, 0) {
        Value::Undefined => None,
        v => Some(symbol_desc_string(v)?),
    };
    Ok(Value::Symbol(heap.alloc_symbol(desc.as_deref())?))
}

fn symbol_for(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    for &a in args {
        roots.root_value(a);
    }
    let key = symbol_desc_string(get_arg(args, 0))?;
    let reg = heap
        .symbol_registry
        .ok_or_else(|| NativeError::TypeError("Symbol registry not installed".into()))?;
    if let Some(p) = heap.get_property(reg, key.as_wtf8()) {
        return Ok(p.value);
    }
    let sym = heap.alloc_symbol(Some(key.as_wtf8()))?;
    roots.root(sym);
    heap.define_property(reg, key.as_wtf8(), Property::data(Value::Symbol(sym)))?;
    Ok(Value::Symbol(sym))
}

fn symbol_key_for(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let s = match get_arg(args, 0) {
        Value::Symbol(s) => s,
        _ => {
            return Err(NativeError::TypeError(
                "Symbol.keyFor: argument must be a symbol".into(),
            ));
        }
    };
    let reg = heap
        .symbol_registry
        .ok_or_else(|| NativeError::TypeError("Symbol registry not installed".into()))?;
    // Copy the pairs out; the scan itself needs no heap access.
    let props: Vec<(Str, Value)> = heap
        .get(reg)
        .properties
        .iter()
        .map(|(k, p)| (*k, p.value))
        .collect();
    for (k, v) in props {
        if v == Value::Symbol(s) {
            return Ok(Value::from(k));
        }
    }
    Ok(Value::Undefined)
}

fn symbol_proto_to_string(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_value(this);
    let s = match this {
        Value::Symbol(s) => s,
        // Sloppy-`this` boxing: unwrap the hidden primitive slot.
        Value::Object(o) => match prim_slot(heap, o) {
            Some(Value::Symbol(s)) => s,
            _ => {
                return Err(NativeError::TypeError(
                    "Symbol.prototype.toString called on incompatible receiver".into(),
                ));
            }
        },
        _ => {
            return Err(NativeError::TypeError(
                "Symbol.prototype.toString called on incompatible receiver".into(),
            ));
        }
    };
    let desc = heap.get(s).description.clone().unwrap_or_default();
    let mut text = Wtf8Buf::from("Symbol(");
    text.push_wtf8(desc.as_wtf8());
    text.push_char(')');
    Ok(Value::Str(heap.alloc_wtf8_owned(text)?))
}

fn symbol_proto_value_of(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    match this {
        Value::Symbol(_) => Ok(this),
        // Sloppy-`this` boxing wraps the symbol in an ordinary object
        // carrying the hidden primitive slot; unwrap it.
        Value::Object(o) => match prim_slot(heap, o) {
            Some(v @ Value::Symbol(_)) => Ok(v),
            _ => Err(NativeError::TypeError(
                "Symbol.prototype.valueOf called on incompatible receiver".into(),
            )),
        },
        _ => Err(NativeError::TypeError(
            "Symbol.prototype.valueOf called on incompatible receiver".into(),
        )),
    }
}

/// `get Symbol.prototype.description`: the symbol's description string,
/// or `undefined` for anonymous symbols. Same receiver rule as
/// `toString`/`valueOf` (a symbol primitive or its boxed wrapper).
fn symbol_proto_description(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let mut roots = CallRoots::new(heap);
    roots.root_value(this);
    let s = match this {
        Value::Symbol(s) => s,
        Value::Object(o) => match prim_slot(heap, o) {
            Some(Value::Symbol(s)) => s,
            _ => {
                return Err(NativeError::TypeError(
                    "Symbol.prototype.description called on incompatible receiver".into(),
                ));
            }
        },
        _ => {
            return Err(NativeError::TypeError(
                "Symbol.prototype.description called on incompatible receiver".into(),
            ));
        }
    };
    match heap.get(s).description.clone() {
        Some(desc) => Ok(Value::Str(heap.alloc_wtf8_owned(desc)?)),
        None => Ok(Value::Undefined),
    }
}

// ---------------------------------------------------------------------------
// install
// ---------------------------------------------------------------------------

/// Well-known symbols installed on the `Symbol` constructor. These are NOT
/// entered in the `Symbol.for` registry (per spec, `Symbol.keyFor` of a
/// well-known symbol is `undefined`).
const WELL_KNOWN_SYMBOLS: &[&str] = &[
    "asyncDispose",
    "asyncIterator",
    "dispose",
    "hasInstance",
    "isConcatSpreadable",
    "iterator",
    "match",
    "matchAll",
    "replace",
    "search",
    "species",
    "split",
    "toPrimitive",
    "toStringTag",
    "unscopables",
];

/// Install `Map`, `Set`, `WeakMap`, `WeakSet`, and `Symbol` on `global`,
/// stashing the intrinsic prototypes and the `Symbol.for` registry on the
/// heap for the constructors to find.
///
/// Call once per heap; not idempotent. Called by the VM bootstrap
/// ([`Vm::new`](crate::vm::Vm::new)) when it installs the builtin modules
/// on the fresh global object.
pub fn install(heap: &mut Heap, global: Gc<Object>) -> Result<(), NativeError> {
    install_inner(heap, global)?;
    Ok(())
}

fn install_inner(heap: &mut Heap, global: Gc<Object>) -> Result<(), AllocError> {
    let mut roots = CallRoots::new(heap);
    roots.root(global);
    // Symbol (first; nothing else depends on it, but the registry must exist
    // before any `Symbol.for` call).
    let symbol_proto = heap.alloc_object(ordinary_object(Value::Null))?;
    roots.root(symbol_proto);
    define_method(heap, symbol_proto, "toString", 0, symbol_proto_to_string)?;
    define_method(heap, symbol_proto, "valueOf", 0, symbol_proto_value_of)?;
    // `get Symbol.prototype.description` (accessor: getter only,
    // non-enumerable, configurable).
    let desc_getter = native_fn(heap, "get description", 0, symbol_proto_description)?;
    roots.root(desc_getter);
    heap.define_accessor(
        symbol_proto,
        Wtf8::from_str("description"),
        AccessorProp {
            get: Value::Object(desc_getter),
            set: Value::Undefined,
            enumerable: false,
            configurable: true,
        },
    )?;
    let symbol_ctor = native_fn(heap, "Symbol", 0, symbol_constructor)?;
    roots.root(symbol_ctor);
    heap.define_property_str(
        symbol_ctor,
        "prototype",
        Property {
            value: Value::Object(symbol_proto),
            writable: false,
            enumerable: false,
            configurable: false,
        },
    )?;
    heap.define_property_str(
        symbol_proto,
        "constructor",
        Property {
            value: Value::Object(symbol_ctor),
            writable: true,
            enumerable: false,
            configurable: true,
        },
    )?;
    for (sname, arity, f) in [
        ("for", 1, symbol_for as NativeFn),
        ("keyFor", 1, symbol_key_for),
    ] {
        let sf = native_fn(heap, sname, arity, f)?;
        heap.define_property_str(
            symbol_ctor,
            sname,
            Property {
                value: Value::Object(sf),
                writable: true,
                enumerable: false,
                configurable: true,
            },
        )?;
    }
    // `WellKnown` discriminants match this table's order; the heap roots
    // the symbols so the abstract operations can consult them under real
    // symbol keys without going through the `Symbol` constructor.
    debug_assert!(WELL_KNOWN_SYMBOLS.len() == WellKnown::COUNT);
    for (i, name) in WELL_KNOWN_SYMBOLS.iter().enumerate() {
        // `well_known_symbol` allocates lazily on first use, so an
        // installer that runs before this module (e.g.
        // `%Array.prototype%[@@toStringTag]`) may already have populated
        // the slot: reuse that symbol so the constructor property and
        // every install-time key are the identical symbol (same
        // description either way; identity is what matters).
        let sym = match heap.well_known[i] {
            Some(sym) => sym,
            None => {
                let desc = format!("Symbol.{name}");
                let sym = heap.alloc_symbol(Some(Wtf8::from_str(&desc)))?;
                heap.well_known[i] = Some(sym);
                sym
            }
        };
        heap.define_property_str(
            symbol_ctor,
            name,
            Property {
                value: Value::Symbol(sym),
                writable: false,
                enumerable: false,
                configurable: false,
            },
        )?;
    }
    // `Symbol.prototype[Symbol.toStringTag] = "Symbol"` (non-writable,
    // non-enumerable, configurable data property, under the real
    // well-known symbol key — installed here, after the loop above
    // populates `heap.well_known`, so the key is `Symbol.toStringTag`
    // itself and not a second same-description symbol).
    define_to_string_tag(heap, symbol_proto, "Symbol")?;
    heap.define_property_str(
        global,
        "Symbol",
        Property {
            value: Value::Object(symbol_ctor),
            writable: true,
            enumerable: false,
            configurable: true,
        },
    )?;
    heap.symbol_prototype = Some(symbol_proto);
    let registry = heap.alloc_object(ordinary_object(Value::Null))?;
    roots.root(registry);
    heap.symbol_registry = Some(registry);

    // Map.
    let map_methods: &[(&str, u16, NativeFn)] = &[
        ("set", 2, map_set),
        ("get", 1, map_get),
        ("has", 1, map_has),
        ("delete", 1, map_delete),
        ("clear", 0, map_clear),
        ("keys", 0, map_keys),
        ("values", 0, map_values),
        ("entries", 0, map_entries),
        ("forEach", 1, map_for_each),
    ];
    let (_, map_proto) = install_ctor(heap, global, "Map", 0, map_constructor, map_methods, &[])?;
    roots.root(map_proto);
    heap.map_prototype = Some(map_proto);
    // `Map.prototype[Symbol.iterator]`: installed under the real symbol
    // key (a computed `o[Symbol.iterator]` key resolves to the same
    // property; the `"@@iterator"` string key no longer aliases it).
    let iterator_sym = heap.well_known_symbol(WellKnown::Iterator)?;
    define_native_key(
        heap,
        map_proto,
        PropKey::Symbol(iterator_sym),
        "[Symbol.iterator]",
        map_iterator,
        0,
    )
    .map_err(|_| AllocError)?;
    // `%Map.prototype%[Symbol.toStringTag]` = `"Map"`.
    define_to_string_tag(heap, map_proto, "Map")?;

    // Set.
    let set_methods: &[(&str, u16, NativeFn)] = &[
        ("add", 1, set_add),
        ("has", 1, set_has),
        ("delete", 1, set_delete),
        ("clear", 0, set_clear),
        ("keys", 0, set_keys),
        ("values", 0, set_values),
        ("entries", 0, set_entries),
        ("forEach", 1, set_for_each),
    ];
    let (_, set_proto) = install_ctor(heap, global, "Set", 0, set_constructor, set_methods, &[])?;
    roots.root(set_proto);
    heap.set_prototype = Some(set_proto);
    // `Set.prototype[Symbol.iterator]`: real symbol key (see the Map note).
    let iterator_sym = heap.well_known_symbol(WellKnown::Iterator)?;
    define_native_key(
        heap,
        set_proto,
        PropKey::Symbol(iterator_sym),
        "[Symbol.iterator]",
        set_iterator,
        0,
    )
    .map_err(|_| AllocError)?;
    // `%Set.prototype%[Symbol.toStringTag]` = `"Set"`.
    define_to_string_tag(heap, set_proto, "Set")?;

    // WeakMap.
    let weakmap_methods: &[(&str, u16, NativeFn)] = &[
        ("set", 2, weakmap_set),
        ("get", 1, weakmap_get),
        ("has", 1, weakmap_has),
        ("delete", 1, weakmap_delete),
    ];
    let (_, weakmap_proto) = install_ctor(
        heap,
        global,
        "WeakMap",
        0,
        weakmap_constructor,
        weakmap_methods,
        &[],
    )?;
    roots.root(weakmap_proto);
    heap.weakmap_prototype = Some(weakmap_proto);
    // `%WeakMap.prototype%[Symbol.toStringTag]` = `"WeakMap"`.
    define_to_string_tag(heap, weakmap_proto, "WeakMap")?;

    // WeakSet.
    let weakset_methods: &[(&str, u16, NativeFn)] = &[
        ("add", 1, weakset_add),
        ("has", 1, weakset_has),
        ("delete", 1, weakset_delete),
    ];
    let (_, weakset_proto) = install_ctor(
        heap,
        global,
        "WeakSet",
        0,
        weakset_constructor,
        weakset_methods,
        &[],
    )?;
    roots.root(weakset_proto);
    heap.weakset_prototype = Some(weakset_proto);
    // `%WeakSet.prototype%[Symbol.toStringTag]` = `"WeakSet"`.
    define_to_string_tag(heap, weakset_proto, "WeakSet")?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Symbol;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn installed() -> (Heap, Gc<Object>) {
        let mut heap = Heap::new();
        let global = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        install(&mut heap, global).unwrap();
        (heap, global)
    }

    fn global_ctor(heap: &Heap, global: Gc<Object>, name: &str) -> Gc<Object> {
        match heap
            .get_property(global, Wtf8::from_str(name))
            .expect("ctor missing")
            .value
        {
            Value::Object(o) => o,
            v => panic!("{name} is not an object: {v:?}"),
        }
    }

    fn ctor_proto(heap: &Heap, ctor: Gc<Object>) -> Gc<Object> {
        match heap
            .get_property(ctor, Wtf8::from_str("prototype"))
            .expect("prototype missing")
            .value
        {
            Value::Object(o) => o,
            _ => panic!("prototype is not an object"),
        }
    }

    fn call_ctor(heap: &mut Heap, ctor: Gc<Object>, args: &[Value]) -> Result<Value, NativeError> {
        match heap.get(ctor).kind {
            ObjectKind::NativeFunction(f) => f(heap, Value::Undefined, args),
            _ => panic!("constructor is not a native function"),
        }
    }

    fn method_fn(heap: &Heap, proto: Gc<Object>, name: &str) -> NativeFn {
        match heap
            .get_property(proto, Wtf8::from_str(name))
            .expect("method missing")
            .value
        {
            Value::Object(o) => match heap.get(o).kind {
                ObjectKind::NativeFunction(f) => f,
                _ => panic!("{name} is not a native function"),
            },
            _ => panic!("{name} is not an object"),
        }
    }

    /// [`method_fn`] over a [`PropKey`]: fetches a symbol-keyed method
    /// (e.g. `Map.prototype[Symbol.iterator]`) for direct invocation.
    fn method_fn_key(heap: &Heap, proto: Gc<Object>, key: PropKey<'_>) -> NativeFn {
        match heap
            .get_property_key(proto, key)
            .expect("method missing")
            .value
        {
            Value::Object(o) => match heap.get(o).kind {
                ObjectKind::NativeFunction(f) => f,
                _ => panic!("symbol-keyed method is not a native function"),
            },
            _ => panic!("symbol-keyed method is not an object"),
        }
    }

    /// The well-known `Symbol.iterator` for installs/assertions.
    fn iterator_sym(heap: &mut Heap) -> Gc<Symbol> {
        heap.well_known_symbol(WellKnown::Iterator).unwrap()
    }

    fn call_method(
        heap: &mut Heap,
        proto: Gc<Object>,
        name: &str,
        this: Value,
        args: &[Value],
    ) -> Result<Value, NativeError> {
        method_fn(heap, proto, name)(heap, this, args)
    }

    fn call_method_key(
        heap: &mut Heap,
        proto: Gc<Object>,
        key: PropKey<'_>,
        this: Value,
        args: &[Value],
    ) -> Result<Value, NativeError> {
        method_fn_key(heap, proto, key)(heap, this, args)
    }

    fn str_val(heap: &mut Heap, s: &str) -> Value {
        Value::Str(heap.alloc_str(s).unwrap())
    }

    fn size_of(heap: &Heap, coll: Gc<Object>) -> Value {
        heap.get_property(coll, Wtf8::from_str("size"))
            .expect("size missing")
            .value
    }

    /// Build an array-like `{0: v0, ..., length: n}` for constructor tests.
    fn array_like(heap: &mut Heap, items: &[Value]) -> Gc<Object> {
        let o = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        for (i, v) in items.iter().enumerate() {
            heap.define_property_str(o, &i.to_string(), Property::data(*v))
                .unwrap();
        }
        heap.define_property_str(
            o,
            "length",
            Property {
                value: Value::Int(items.len() as i32),
                writable: true,
                enumerable: false,
                configurable: false,
            },
        )
        .unwrap();
        o
    }

    /// Read a dense array into a Vec (holes become `undefined`).
    fn dense_to_vec(heap: &Heap, arr: Gc<Object>) -> Vec<Value> {
        let len = dense_len(heap, arr);
        (0..len)
            .map(|i| dense_get(heap, arr, i).unwrap_or(Value::Undefined))
            .collect()
    }

    #[test]
    fn constructors_are_installed_as_functions() {
        let (heap, global) = installed();
        for name in ["Map", "Set", "WeakMap", "WeakSet", "Symbol"] {
            let ctor = global_ctor(&heap, global, name);
            let v = Value::Object(ctor);
            assert_eq!(v.type_of(), "function", "{name} should be a function");
            // prototype <-> constructor link both ways.
            let proto = ctor_proto(&heap, ctor);
            match heap
                .get_property(proto, Wtf8::from_str("constructor"))
                .unwrap()
                .value
            {
                Value::Object(back) => assert_eq!(back, ctor),
                _ => panic!("constructor back-link missing"),
            }
        }
        // Heap intrinsics stashed for the constructors.
        assert!(heap.map_prototype.is_some());
        assert!(heap.set_prototype.is_some());
        assert!(heap.weakmap_prototype.is_some());
        assert!(heap.weakset_prototype.is_some());
        assert!(heap.symbol_prototype.is_some());
        assert!(heap.symbol_registry.is_some());
    }

    #[test]
    fn map_set_get_has_delete() {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Map");
        let proto = ctor_proto(&heap, ctor);
        let map = match call_ctor(&mut heap, ctor, &[]).unwrap() {
            Value::Object(o) => o,
            _ => panic!("not an object"),
        };
        let this = Value::Object(map);
        let k = str_val(&mut heap, "a");
        assert_eq!(
            call_method(&mut heap, proto, "set", this, &[k, Value::Int(1)]).unwrap(),
            this,
            "set returns the map"
        );
        assert_eq!(
            call_method(&mut heap, proto, "get", this, &[k]).unwrap(),
            Value::Int(1)
        );
        assert_eq!(
            call_method(&mut heap, proto, "has", this, &[k]).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(size_of(&heap, map), Value::Int(1));
        assert_eq!(
            call_method(&mut heap, proto, "delete", this, &[k]).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            call_method(&mut heap, proto, "get", this, &[k]).unwrap(),
            Value::Undefined
        );
        assert_eq!(
            call_method(&mut heap, proto, "has", this, &[k]).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            call_method(&mut heap, proto, "delete", this, &[k]).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(size_of(&heap, map), Value::Int(0));
    }

    #[test]
    fn map_nan_keys_use_same_value_zero() {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Map");
        let proto = ctor_proto(&heap, ctor);
        let map = match call_ctor(&mut heap, ctor, &[]).unwrap() {
            Value::Object(o) => o,
            _ => panic!("not an object"),
        };
        let this = Value::Object(map);
        let nan = Value::Float(f64::NAN);
        call_method(&mut heap, proto, "set", this, &[nan, Value::Int(7)]).unwrap();
        // A *different* NaN payload must hit the same entry.
        assert_eq!(
            call_method(&mut heap, proto, "get", this, &[Value::Float(f64::NAN)]).unwrap(),
            Value::Int(7)
        );
        assert_eq!(
            call_method(&mut heap, proto, "has", this, &[nan]).unwrap(),
            Value::Bool(true)
        );
        // Overwriting does not grow the map.
        call_method(&mut heap, proto, "set", this, &[nan, Value::Int(8)]).unwrap();
        assert_eq!(size_of(&heap, map), Value::Int(1));
        assert_eq!(
            call_method(&mut heap, proto, "get", this, &[nan]).unwrap(),
            Value::Int(8)
        );
    }

    #[test]
    fn map_minus_zero_equals_plus_zero() {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Map");
        let proto = ctor_proto(&heap, ctor);
        let map = match call_ctor(&mut heap, ctor, &[]).unwrap() {
            Value::Object(o) => o,
            _ => panic!("not an object"),
        };
        let this = Value::Object(map);
        let neg = str_val(&mut heap, "neg");
        call_method(&mut heap, proto, "set", this, &[Value::Float(-0.0), neg]).unwrap();
        assert_eq!(
            call_method(&mut heap, proto, "get", this, &[Value::Int(0)]).unwrap(),
            neg
        );
        assert_eq!(size_of(&heap, map), Value::Int(1));
    }

    #[test]
    fn map_iteration_follows_insertion_order() {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Map");
        let proto = ctor_proto(&heap, ctor);
        let map = match call_ctor(&mut heap, ctor, &[]).unwrap() {
            Value::Object(o) => o,
            _ => panic!("not an object"),
        };
        let this = Value::Object(map);
        let ka = str_val(&mut heap, "a");
        let kb = str_val(&mut heap, "b");
        let kc = str_val(&mut heap, "c");
        call_method(&mut heap, proto, "set", this, &[ka, Value::Int(1)]).unwrap();
        call_method(&mut heap, proto, "set", this, &[kb, Value::Int(2)]).unwrap();
        call_method(&mut heap, proto, "set", this, &[kc, Value::Int(3)]).unwrap();
        // Update in place: order must not change.
        call_method(&mut heap, proto, "set", this, &[kb, Value::Int(22)]).unwrap();
        // Delete + re-add moves the key to the end.
        call_method(&mut heap, proto, "delete", this, &[ka]).unwrap();
        call_method(&mut heap, proto, "set", this, &[ka, Value::Int(11)]).unwrap();

        let keys = match call_method(&mut heap, proto, "keys", this, &[]).unwrap() {
            Value::Object(o) => dense_to_vec(&heap, o),
            _ => panic!("keys not an array"),
        };
        assert_eq!(keys.len(), 3);
        assert!(matches!(keys[0], Value::Str(s) if s.as_wtf8() == Wtf8::from_str("b")));
        assert!(matches!(keys[1], Value::Str(s) if s.as_wtf8() == Wtf8::from_str("c")));
        assert!(matches!(keys[2], Value::Str(s) if s.as_wtf8() == Wtf8::from_str("a")));

        let entries = match call_method(&mut heap, proto, "entries", this, &[]).unwrap() {
            Value::Object(o) => dense_to_vec(&heap, o),
            _ => panic!("entries not an array"),
        };
        assert_eq!(entries.len(), 3);
        let pair0 = match entries[0] {
            Value::Object(o) => dense_to_vec(&heap, o),
            _ => panic!("entry not an array"),
        };
        assert!(matches!(pair0[0], Value::Str(s) if s.as_wtf8() == Wtf8::from_str("b")));
        assert_eq!(pair0[1], Value::Int(22));
    }

    #[test]
    fn map_clear_and_constructor_pairs() {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Map");
        let proto = ctor_proto(&heap, ctor);
        // Build an array-like of pairs: [[k1, v1], [k2, v2]].
        let pairs = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let pair_objs: Vec<Gc<Object>> = [("k1", 10), ("k2", 20)]
            .iter()
            .map(|(k, v)| {
                let p = heap
                    .alloc_object(Object::new(ObjectKind::Ordinary))
                    .unwrap();
                let kv = str_val(&mut heap, k);
                heap.define_property_str(p, "0", Property::data(kv))
                    .unwrap();
                heap.define_property_str(p, "1", Property::data(Value::Int(*v)))
                    .unwrap();
                heap.define_property_str(
                    p,
                    "length",
                    Property {
                        value: Value::Int(2),
                        writable: true,
                        enumerable: false,
                        configurable: false,
                    },
                )
                .unwrap();
                p
            })
            .collect();
        for (i, p) in pair_objs.iter().enumerate() {
            heap.define_property_str(pairs, &i.to_string(), Property::data(Value::Object(*p)))
                .unwrap();
        }
        heap.define_property_str(
            pairs,
            "length",
            Property {
                value: Value::Int(2),
                writable: true,
                enumerable: false,
                configurable: false,
            },
        )
        .unwrap();

        let map = match call_ctor(&mut heap, ctor, &[Value::Object(pairs)]).unwrap() {
            Value::Object(o) => o,
            _ => panic!("not an object"),
        };
        let this = Value::Object(map);
        assert_eq!(size_of(&heap, map), Value::Int(2));
        let k2 = str_val(&mut heap, "k2");
        assert_eq!(
            call_method(&mut heap, proto, "get", this, &[k2]).unwrap(),
            Value::Int(20)
        );
        call_method(&mut heap, proto, "clear", this, &[]).unwrap();
        assert_eq!(size_of(&heap, map), Value::Int(0));
        let k1 = str_val(&mut heap, "k1");
        assert_eq!(
            call_method(&mut heap, proto, "has", this, &[k1]).unwrap(),
            Value::Bool(false)
        );
    }

    static FOREACH_CALLS: AtomicUsize = AtomicUsize::new(0);

    fn counting_cb(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
        assert_eq!(args.len(), 3);
        FOREACH_CALLS.fetch_add(1, Ordering::SeqCst);
        Ok(Value::Undefined)
    }

    #[test]
    fn map_for_each_visits_every_entry() {
        FOREACH_CALLS.store(0, Ordering::SeqCst);
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Map");
        let proto = ctor_proto(&heap, ctor);
        let map = match call_ctor(&mut heap, ctor, &[]).unwrap() {
            Value::Object(o) => o,
            _ => panic!("not an object"),
        };
        let this = Value::Object(map);
        for i in 0..5 {
            call_method(
                &mut heap,
                proto,
                "set",
                this,
                &[Value::Int(i), Value::Int(i * 10)],
            )
            .unwrap();
        }
        let cb = heap
            .alloc_object(Object::new(ObjectKind::NativeFunction(counting_cb)))
            .unwrap();
        call_method(&mut heap, proto, "forEach", this, &[Value::Object(cb)]).unwrap();
        assert_eq!(FOREACH_CALLS.load(Ordering::SeqCst), 5);
        // Non-function callbacks throw.
        assert!(matches!(
            call_method(&mut heap, proto, "forEach", this, &[Value::Int(1)]),
            Err(NativeError::TypeError(_))
        ));
    }

    #[test]
    fn map_rejects_incompatible_receivers() {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Map");
        let proto = ctor_proto(&heap, ctor);
        let plain = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        assert!(matches!(
            call_method(&mut heap, proto, "get", Value::Object(plain), &[]),
            Err(NativeError::TypeError(_))
        ));
        assert!(matches!(
            call_method(&mut heap, proto, "set", Value::Undefined, &[]),
            Err(NativeError::TypeError(_))
        ));
    }

    #[test]
    fn set_add_has_delete_dedups() {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Set");
        let proto = ctor_proto(&heap, ctor);
        let set = match call_ctor(&mut heap, ctor, &[]).unwrap() {
            Value::Object(o) => o,
            _ => panic!("not an object"),
        };
        let this = Value::Object(set);
        // add returns the set itself.
        assert_eq!(
            call_method(&mut heap, proto, "add", this, &[Value::Int(1)]).unwrap(),
            this
        );
        call_method(&mut heap, proto, "add", this, &[Value::Int(1)]).unwrap();
        call_method(&mut heap, proto, "add", this, &[Value::Float(f64::NAN)]).unwrap();
        call_method(&mut heap, proto, "add", this, &[Value::Float(f64::NAN)]).unwrap();
        assert_eq!(size_of(&heap, set), Value::Int(2));
        assert_eq!(
            call_method(&mut heap, proto, "has", this, &[Value::Float(f64::NAN)]).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            call_method(&mut heap, proto, "delete", this, &[Value::Int(1)]).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            call_method(&mut heap, proto, "delete", this, &[Value::Int(1)]).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(size_of(&heap, set), Value::Int(1));
        // Insertion order is preserved.
        let values = match call_method(&mut heap, proto, "values", this, &[]).unwrap() {
            Value::Object(o) => dense_to_vec(&heap, o),
            _ => panic!("values not an array"),
        };
        assert_eq!(values.len(), 1);
        assert!(matches!(values[0], Value::Float(f) if f.is_nan()));
        // keys() mirrors values() for sets.
        let keys = match call_method(&mut heap, proto, "keys", this, &[]).unwrap() {
            Value::Object(o) => dense_to_vec(&heap, o),
            _ => panic!("keys not an array"),
        };
        assert_eq!(keys.len(), 1);
    }

    #[test]
    fn set_constructor_takes_array_like() {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Set");
        let proto = ctor_proto(&heap, ctor);
        let src = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.define_property_str(src, "0", Property::data(Value::Int(3)))
            .unwrap();
        heap.define_property_str(src, "1", Property::data(Value::Int(3)))
            .unwrap();
        heap.define_property_str(src, "2", Property::data(Value::Int(4)))
            .unwrap();
        heap.define_property_str(
            src,
            "length",
            Property {
                value: Value::Int(3),
                writable: true,
                enumerable: false,
                configurable: false,
            },
        )
        .unwrap();
        let set = match call_ctor(&mut heap, ctor, &[Value::Object(src)]).unwrap() {
            Value::Object(o) => o,
            _ => panic!("not an object"),
        };
        assert_eq!(size_of(&heap, set), Value::Int(2));
        let this = Value::Object(set);
        call_method(&mut heap, proto, "clear", this, &[]).unwrap();
        assert_eq!(size_of(&heap, set), Value::Int(0));
    }

    #[test]
    fn weakmap_requires_object_keys() {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "WeakMap");
        let proto = ctor_proto(&heap, ctor);
        let wm = match call_ctor(&mut heap, ctor, &[]).unwrap() {
            Value::Object(o) => o,
            _ => panic!("not an object"),
        };
        let this = Value::Object(wm);
        let key = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let key_v = Value::Object(key);
        assert_eq!(
            call_method(&mut heap, proto, "set", this, &[key_v, Value::Int(9)]).unwrap(),
            this
        );
        assert_eq!(
            call_method(&mut heap, proto, "get", this, &[key_v]).unwrap(),
            Value::Int(9)
        );
        assert_eq!(
            call_method(&mut heap, proto, "has", this, &[key_v]).unwrap(),
            Value::Bool(true)
        );
        // Non-object keys: set throws, the rest return empty results.
        assert!(matches!(
            call_method(
                &mut heap,
                proto,
                "set",
                this,
                &[Value::Int(1), Value::Int(2)]
            ),
            Err(NativeError::TypeError(_))
        ));
        assert_eq!(
            call_method(&mut heap, proto, "get", this, &[Value::Int(1)]).unwrap(),
            Value::Undefined
        );
        let x = Value::Str(heap.alloc_str("x").unwrap());
        assert_eq!(
            call_method(&mut heap, proto, "has", this, &[x]).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            call_method(&mut heap, proto, "delete", this, &[Value::Null]).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            call_method(&mut heap, proto, "delete", this, &[key_v]).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            call_method(&mut heap, proto, "get", this, &[key_v]).unwrap(),
            Value::Undefined
        );
    }

    #[test]
    fn weakset_requires_object_values() {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "WeakSet");
        let proto = ctor_proto(&heap, ctor);
        let ws = match call_ctor(&mut heap, ctor, &[]).unwrap() {
            Value::Object(o) => o,
            _ => panic!("not an object"),
        };
        let this = Value::Object(ws);
        let v = Value::Object(
            heap.alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap(),
        );
        assert_eq!(
            call_method(&mut heap, proto, "add", this, &[v]).unwrap(),
            this
        );
        assert_eq!(
            call_method(&mut heap, proto, "has", this, &[v]).unwrap(),
            Value::Bool(true)
        );
        assert!(matches!(
            call_method(&mut heap, proto, "add", this, &[Value::Bool(true)]),
            Err(NativeError::TypeError(_))
        ));
        assert_eq!(
            call_method(&mut heap, proto, "has", this, &[Value::Bool(true)]).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            call_method(&mut heap, proto, "delete", this, &[v]).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            call_method(&mut heap, proto, "has", this, &[v]).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn weakset_constructor_takes_array_like() {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "WeakSet");
        let proto = ctor_proto(&heap, ctor);
        let a = Value::Object(
            heap.alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap(),
        );
        let b = Value::Object(
            heap.alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap(),
        );
        let src = array_like(&mut heap, &[a, b, a]);
        let ws = match call_ctor(&mut heap, ctor, &[Value::Object(src)]).unwrap() {
            Value::Object(o) => o,
            _ => panic!("not an object"),
        };
        let this = Value::Object(ws);
        assert_eq!(
            call_method(&mut heap, proto, "has", this, &[a]).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            call_method(&mut heap, proto, "has", this, &[b]).unwrap(),
            Value::Bool(true)
        );
        // Non-object values throw.
        let bad = array_like(&mut heap, &[Value::Int(1)]);
        assert!(matches!(
            call_ctor(&mut heap, ctor, &[Value::Object(bad)]),
            Err(NativeError::TypeError(_))
        ));
    }

    #[test]
    fn weak_collections_have_no_size() {
        let (mut heap, global) = installed();
        let wm_ctor = global_ctor(&heap, global, "WeakMap");
        let wm_proto = ctor_proto(&heap, wm_ctor);
        let ws_ctor = global_ctor(&heap, global, "WeakSet");
        let ws_proto = ctor_proto(&heap, ws_ctor);
        let wm = match call_ctor(&mut heap, wm_ctor, &[]).unwrap() {
            Value::Object(o) => o,
            _ => panic!("not an object"),
        };
        let ws = match call_ctor(&mut heap, ws_ctor, &[]).unwrap() {
            Value::Object(o) => o,
            _ => panic!("not an object"),
        };
        let k = Value::Object(
            heap.alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap(),
        );
        call_method(
            &mut heap,
            wm_proto,
            "set",
            Value::Object(wm),
            &[k, Value::Int(1)],
        )
        .unwrap();
        call_method(&mut heap, ws_proto, "add", Value::Object(ws), &[k]).unwrap();
        assert!(heap.get_property(wm, Wtf8::from_str("size")).is_none());
        assert!(heap.get_property(ws, Wtf8::from_str("size")).is_none());
    }

    #[test]
    fn symbols_are_unique_but_registry_dedups() {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Symbol");
        let for_fn = method_fn(&heap, ctor, "for");
        let key_for_fn = method_fn(&heap, ctor, "keyFor");

        let x = str_val(&mut heap, "x");
        let a = call_ctor(&mut heap, ctor, &[x]).unwrap();
        let b = call_ctor(&mut heap, ctor, &[x]).unwrap();
        assert!(matches!(a, Value::Symbol(_)));
        assert_ne!(a, b, "Symbol() must be unique even with equal descriptions");

        let k = str_val(&mut heap, "k");
        let r1 = for_fn(&mut heap, Value::Undefined, &[k]).unwrap();
        let r2 = for_fn(&mut heap, Value::Undefined, &[k]).unwrap();
        assert_eq!(r1, r2, "Symbol.for must return the registry entry");
        assert_ne!(r1, a, "registry symbols differ from fresh Symbol()");

        let key = key_for_fn(&mut heap, Value::Undefined, &[r1]).unwrap();
        assert!(matches!(key, Value::Str(s) if s.as_wtf8() == Wtf8::from_str("k")));
        // Fresh symbols are not in the registry.
        assert_eq!(
            key_for_fn(&mut heap, Value::Undefined, &[a]).unwrap(),
            Value::Undefined
        );
        // keyFor requires a symbol.
        assert!(matches!(
            key_for_fn(&mut heap, Value::Undefined, &[Value::Int(1)]),
            Err(NativeError::TypeError(_))
        ));
        // typeof still works through the value layer.
        assert_eq!(a.type_of(), "symbol");
    }

    #[test]
    fn well_known_symbols_exist_and_skip_registry() {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Symbol");
        let key_for_fn = method_fn(&heap, ctor, "keyFor");
        for (i, name) in WELL_KNOWN_SYMBOLS.iter().enumerate() {
            let v = heap
                .get_property(ctor, Wtf8::from_str(name))
                .expect("well-known missing")
                .value;
            let sym = match v {
                Value::Symbol(s) => s,
                _ => panic!("{name} should be a symbol"),
            };
            // The heap roots the same symbol the constructor exposes, so
            // the abstract operations consult real symbol keys.
            assert_eq!(
                heap.well_known[i],
                Some(sym),
                "heap well-known table must root {name}"
            );
            // Well-known symbols are unique, not fresh `Symbol("Symbol.x")`.
            let desc = str_val(&mut heap, &format!("Symbol.{name}"));
            let fresh = call_ctor(&mut heap, ctor, &[desc]).unwrap();
            assert_ne!(v, fresh);
            // ... and absent from the `Symbol.for` registry.
            assert_eq!(
                key_for_fn(&mut heap, Value::Undefined, &[v]).unwrap(),
                Value::Undefined,
                "{name} must not be in the registry"
            );
        }
    }

    #[test]
    fn dispose_symbols_are_well_known() {
        // `Symbol.dispose` / `Symbol.asyncDispose` (Explicit Resource
        // Management) are installed as well-known symbols; the generic
        // loop above already covers them, this pins their descriptions.
        let r = eval_str("Symbol.dispose.toString() + '|' + Symbol.asyncDispose.toString()");
        assert_eq!(r, "Symbol(Symbol.dispose)|Symbol(Symbol.asyncDispose)");
        assert_eq!(eval_str("typeof Symbol.dispose"), "symbol");
        assert_eq!(eval_str("Symbol.keyFor(Symbol.dispose)"), "undefined");
    }

    #[test]
    fn collection_keeps_intrinsics_and_live_collections() {
        let (mut heap, global) = installed();
        // SAFETY: single-threaded test; the scope is dropped before `heap`.
        // `addr_of_mut!` (not a `&mut` reborrow) per `HandleScope::new`'s
        // contract, so later `&mut heap` uses do not invalidate the pointer.
        let scope = unsafe { HandleScope::new(std::ptr::addr_of_mut!(heap)) };
        let global_h = scope.add(global);

        // A map with entries, everything reachable from the scope.
        let ctor = global_ctor(&heap, global_h.get(), "Map");
        let map = match call_ctor(&mut heap, ctor, &[]).unwrap() {
            Value::Object(o) => o,
            _ => panic!("not an object"),
        };
        let map_h = scope.add(map);
        let proto = ctor_proto(&heap, ctor);
        // Long enough to be heap-allocated: the point is a rooted *heap*
        // string key surviving the collection below.
        let k = heap.alloc_str(&"k".repeat(32)).unwrap();
        let k_h = k.heap_handle().expect("long key is heap-backed");
        scope.add_value(Value::Str(Str::heap(k_h)));
        let sym_ctor = global_ctor(&heap, global_h.get(), "Symbol");
        let s_desc = str_val(&mut heap, "s");
        let sym = match call_ctor(&mut heap, sym_ctor, &[s_desc]).unwrap() {
            Value::Symbol(s) => s,
            _ => panic!("not a symbol"),
        };
        let sym_h = scope.add(sym);

        let map_this = Value::Object(map_h.get());
        let key = Value::Str(Str::heap(k_h));
        method_fn(&heap, proto, "set")(&mut heap, map_this, &[key, Value::Int(1)]).unwrap();
        // symbol keys work too (identity comparison).
        method_fn(&heap, proto, "set")(
            &mut heap,
            map_this,
            &[Value::Symbol(sym_h.get()), Value::Int(2)],
        )
        .unwrap();

        heap.collect();

        // Intrinsics survive as heap roots with no other roots held.
        assert!(heap.map_prototype.is_some());
        assert!(heap.set_prototype.is_some());
        assert!(heap.weakmap_prototype.is_some());
        assert!(heap.weakset_prototype.is_some());
        assert!(heap.symbol_prototype.is_some());
        assert!(heap.symbol_registry.is_some());
        // Methods are reachable through the rooted prototype.
        let get = method_fn(&heap, heap.map_prototype.unwrap(), "get");
        assert_eq!(get(&mut heap, map_this, &[key]).unwrap(), Value::Int(1));
        assert_eq!(
            get(&mut heap, map_this, &[Value::Symbol(sym_h.get())]).unwrap(),
            Value::Int(2)
        );
        // The registry entry survives via the rooted registry object.
        let key_for = method_fn(&heap, sym_ctor, "keyFor");
        let rk = str_val(&mut heap, "rk");
        let r = method_fn(&heap, sym_ctor, "for")(&mut heap, Value::Undefined, &[rk]).unwrap();
        let found = key_for(&mut heap, Value::Undefined, &[r]).unwrap();
        assert!(matches!(found, Value::Str(s) if s.as_wtf8() == Wtf8::from_str("rk")));
        heap.collect();
        let found = key_for(&mut heap, Value::Undefined, &[r]).unwrap();
        assert!(matches!(found, Value::Str(s) if s.as_wtf8() == Wtf8::from_str("rk")));
    }

    #[test]
    fn call_roots_holds_handle_across_later_mut_uses() {
        let (mut heap, _global) = installed();
        // Long enough to be heap-allocated: the point is rooting a *heap*
        // string across later allocations and explicit collections.
        let s = heap.alloc_str(&"pinned".repeat(6)).unwrap();
        // Mimic a native call: the `CallRoots` scope is created from the
        // `&mut` heap place, then the heap is used through fresh `&mut`
        // reborrows (allocations, explicit collections) before the scope is
        // dropped. That sequence was Stacked Borrows UB with the old
        // reborrow-derived raw pointer, and must keep the rooted handle
        // alive with the `addr_of_mut!` fix.
        {
            let mut roots = CallRoots::new(&mut heap);
            roots.root_value(Value::Str(s));
            for i in 0..256u32 {
                // Long enough to actually allocate: short strings inline.
                let _ = heap
                    .alloc_str(&format!("filler-{i}-{}", "x".repeat(24)))
                    .unwrap();
                heap.collect();
            }
            assert_eq!(s.as_wtf8(), Wtf8::from_str(&"pinned".repeat(6)));
        }
        // The scope is dropped: nothing else roots `s`, so a collection
        // frees it. The live-allocation count must shrink, proving the
        // rooted value above was really pinned by the scope and that the
        // scope no longer touches the heap after drop.
        let before = heap.alloc_count();
        heap.collect();
        let after = heap.alloc_count();
        assert!(
            after < before,
            "dropping the scope should release the rooted string"
        );
    }

    #[test]
    fn symbol_prototype_to_string_and_value_of() {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Symbol");
        let proto = ctor_proto(&heap, ctor);
        let desc = str_val(&mut heap, "desc");
        let s = call_ctor(&mut heap, ctor, &[desc]).unwrap();
        let text = call_method(&mut heap, proto, "toString", s, &[]).unwrap();
        assert!(matches!(text, Value::Str(g) if g.as_wtf8() == Wtf8::from_str("Symbol(desc)")));
        let anon = call_ctor(&mut heap, ctor, &[]).unwrap();
        let text = call_method(&mut heap, proto, "toString", anon, &[]).unwrap();
        assert!(matches!(text, Value::Str(g) if g.as_wtf8() == Wtf8::from_str("Symbol()")));
        assert_eq!(call_method(&mut heap, proto, "valueOf", s, &[]).unwrap(), s);
        // Incompatible receivers throw.
        assert!(matches!(
            call_method(&mut heap, proto, "toString", Value::Int(1), &[]),
            Err(NativeError::TypeError(_))
        ));
    }

    #[test]
    fn symbol_prototype_description_getter() {
        // Named and anonymous symbols.
        assert_eq!(eval_str("Symbol('d').description"), "d");
        assert_eq!(eval_str("Symbol().description === undefined"), "true");
        // Empty-string description is preserved (not undefined).
        assert_eq!(eval_str("Symbol('').description === ''"), "true");
        // Incompatible receivers throw.
        assert_eq!(
            eval_str(
                "var r = 'no-throw'; \
                 try { Object.getOwnPropertyDescriptor(Symbol.prototype, 'description') \
                         .get.call(1); } \
                 catch (e) { r = 'threw'; } r"
            ),
            "threw"
        );
        // Accessor attributes: non-enumerable, configurable, getter-only.
        assert_eq!(
            eval_str(
                "var d = Object.getOwnPropertyDescriptor(Symbol.prototype, 'description'); \
                 (!d.enumerable) + ',' + d.configurable + ',' + (d.set === undefined)"
            ),
            "true,true,true"
        );
        // Listed by getOwnPropertyNames (all string keys) but not by
        // Object.keys (non-enumerable).
        assert_eq!(
            eval_str("Object.getOwnPropertyNames(Symbol.prototype).indexOf('description') !== -1"),
            "true"
        );
        assert_eq!(
            eval_str("Object.keys(Symbol.prototype).indexOf('description')"),
            "-1"
        );
    }

    #[test]
    fn symbol_prototype_to_string_tag() {
        assert_eq!(eval_str("Symbol.prototype[Symbol.toStringTag]"), "Symbol");
        // Object.prototype.toString honors it.
        assert_eq!(
            eval_str("Object.prototype.toString.call(Symbol.prototype)"),
            "[object Symbol]"
        );
        // Attributes: non-writable, non-enumerable, configurable.
        assert_eq!(
            eval_str(
                "var d = Object.getOwnPropertyDescriptor(Symbol.prototype, Symbol.toStringTag); \
                 d.value + ',' + (!d.writable) + ',' + (!d.enumerable) + ',' + d.configurable"
            ),
            "Symbol,true,true,true"
        );
    }

    #[test]
    fn builtin_prototypes_to_string_tag() {
        // Every collection prototype (and Array/ArrayBuffer) carries its
        // @@toStringTag; Object.prototype.toString honors them.
        assert_eq!(
            eval_str(
                "var ts = Object.prototype.toString; \
                 [ts.call(Map.prototype), ts.call(Set.prototype), \
                  ts.call(WeakMap.prototype), ts.call(WeakSet.prototype), \
                  ts.call(Array.prototype), ts.call(new ArrayBuffer(8))].join(',')"
            ),
            "[object Map],[object Set],[object WeakMap],[object WeakSet],[object Array],[object ArrayBuffer]"
        );
        // The tags live under the real symbol key, in spec order after
        // string keys.
        assert_eq!(
            eval_str(
                "var keys = Reflect.ownKeys(Map.prototype); \
                 keys[keys.length - 1] === Symbol.toStringTag"
            ),
            "true"
        );
    }

    #[test]
    fn symbol_key_for_registry_edges() {
        // Round-trip through the registry.
        assert_eq!(eval_str("Symbol.keyFor(Symbol.for('x'))"), "x");
        // Unregistered symbols (even with descriptions) are undefined.
        assert_eq!(eval_str("Symbol.keyFor(Symbol('x')) === undefined"), "true");
        assert_eq!(eval_str("Symbol.keyFor(Symbol()) === undefined"), "true");
        // Well-known symbols are never in the registry.
        assert_eq!(
            eval_str("Symbol.keyFor(Symbol.iterator) === undefined"),
            "true"
        );
        assert_eq!(
            eval_str("Symbol.keyFor(Symbol.toStringTag) === undefined"),
            "true"
        );
        // Non-symbol argument throws.
        assert_eq!(
            eval_str(
                "var r = 'no-throw'; try { Symbol.keyFor('x'); } catch (e) { r = 'threw'; } r"
            ),
            "threw"
        );
        // Registry identity: Symbol.for returns the same symbol twice, and
        // keyFor is stable across descriptions that collide as strings.
        assert_eq!(
            eval_str(
                "var a = Symbol.for('dup'), b = Symbol.for('dup'); \
                 (a === b) + ',' + Symbol.keyFor(a)"
            ),
            "true,dup"
        );
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
    fn map_foreach_accepts_bytecode_callback() {
        // Note: `forEach` is invoked on a variable; method calls on `new`
        // temporaries are a separate VM gap.
        let r = eval_str(
            "let m = new Map([[1, 'a'], [2, 'b']]); let s = ''; m.forEach((v, k) => { s += k + ':' + v + ';'; }); s",
        );
        assert_eq!(r, "1:a;2:b;");
    }

    #[test]
    fn set_foreach_accepts_bytecode_callback() {
        let r =
            eval_str("let st = new Set([1, 2, 3]); let t = 0; st.forEach((v) => { t += v; }); t");
        assert_eq!(r, "6");
    }

    #[test]
    fn foreach_propagates_bytecode_throw() {
        let mut vm = crate::vm::Vm::new();
        let err = vm
            .eval("let m = new Map([[1, 2]]); m.forEach(() => { throw 'boom'; })")
            .unwrap_err();
        assert!(
            matches!(err, crate::vm::VmError::Thrown(ref s) if s == "boom"),
            "unexpected error: {err}"
        );
    }

    // -- Benchmarks ----------------------------------------------------
    //
    // Timing harnesses: they print per-operation throughput and assert
    // nothing about time. Run single-threaded with output visible:
    //   cargo test -p bowser_js bench_ -- --ignored --nocapture --test-threads=1

    fn report(label: &str, n: usize, dt: std::time::Duration) {
        eprintln!(
            "bench {label}: n={n} {dt:?} ({:.1} ns/op, {:.0} ops/s)",
            dt.as_nanos() as f64 / n as f64,
            n as f64 / dt.as_secs_f64()
        );
    }

    fn bench_map_with(n: usize) -> (Heap, Gc<Object>, Gc<Object>) {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Map");
        let proto = ctor_proto(&heap, ctor);
        let map = match call_ctor(&mut heap, ctor, &[]).unwrap() {
            Value::Object(o) => o,
            _ => panic!("not an object"),
        };
        let set = method_fn(&heap, proto, "set");
        let this = Value::Object(map);
        for i in 0..n {
            set(
                &mut heap,
                this,
                &[Value::Int(i as i32), Value::Int(i as i32)],
            )
            .unwrap();
        }
        (heap, map, proto)
    }

    fn bench_set_with(n: usize) -> (Heap, Gc<Object>, Gc<Object>) {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Set");
        let proto = ctor_proto(&heap, ctor);
        let set = match call_ctor(&mut heap, ctor, &[]).unwrap() {
            Value::Object(o) => o,
            _ => panic!("not an object"),
        };
        let add = method_fn(&heap, proto, "add");
        let this = Value::Object(set);
        for i in 0..n {
            add(&mut heap, this, &[Value::Int(i as i32)]).unwrap();
        }
        (heap, set, proto)
    }

    #[test]
    #[ignore]
    fn bench_map_set() {
        let n = 400;
        let t = std::time::Instant::now();
        let _ = bench_map_with(n);
        report("map.set", n, t.elapsed());
    }

    #[test]
    #[ignore]
    fn bench_map_get() {
        let n = 400;
        let (mut heap, map, proto) = bench_map_with(n);
        let get = method_fn(&heap, proto, "get");
        let this = Value::Object(map);
        // Reverse order: worst case for the linear scan.
        let t = std::time::Instant::now();
        for i in (0..n).rev() {
            assert_eq!(
                get(&mut heap, this, &[Value::Int(i as i32)]).unwrap(),
                Value::Int(i as i32)
            );
        }
        report("map.get (reverse)", n, t.elapsed());
    }

    #[test]
    #[ignore]
    fn bench_map_has() {
        let n = 400;
        let (mut heap, map, proto) = bench_map_with(n);
        let has = method_fn(&heap, proto, "has");
        let this = Value::Object(map);
        let t = std::time::Instant::now();
        for i in 0..n {
            assert_eq!(
                has(&mut heap, this, &[Value::Int(i as i32)]).unwrap(),
                Value::Bool(true)
            );
        }
        report("map.has", n, t.elapsed());
    }

    #[test]
    #[ignore]
    fn bench_map_iteration() {
        let n = 400;
        let (mut heap, map, proto) = bench_map_with(n);
        let this = Value::Object(map);
        for name in ["keys", "values", "entries"] {
            let t = std::time::Instant::now();
            let out = call_method(&mut heap, proto, name, this, &[]).unwrap();
            let dt = t.elapsed();
            match out {
                Value::Object(o) => assert_eq!(dense_len(&heap, o) as usize, n),
                _ => panic!("{name} did not return an array"),
            }
            report(&format!("map.{name}"), 1, dt);
        }
    }

    #[test]
    #[ignore]
    fn bench_map_for_each() {
        FOREACH_CALLS.store(0, Ordering::SeqCst);
        let n = 400;
        let (mut heap, map, proto) = bench_map_with(n);
        let cb = heap
            .alloc_object(Object::new(ObjectKind::NativeFunction(counting_cb)))
            .unwrap();
        let t = std::time::Instant::now();
        call_method(
            &mut heap,
            proto,
            "forEach",
            Value::Object(map),
            &[Value::Object(cb)],
        )
        .unwrap();
        let dt = t.elapsed();
        assert_eq!(FOREACH_CALLS.load(Ordering::SeqCst), n);
        report("map.forEach", n, dt);
    }

    #[test]
    #[ignore]
    fn bench_map_delete() {
        let n = 400;
        let (mut heap, map, proto) = bench_map_with(n);
        let delete = method_fn(&heap, proto, "delete");
        let this = Value::Object(map);
        // Delete from the front: worst case for the shift-down.
        let t = std::time::Instant::now();
        for i in 0..n {
            assert_eq!(
                delete(&mut heap, this, &[Value::Int(i as i32)]).unwrap(),
                Value::Bool(true)
            );
        }
        report("map.delete (front)", n, t.elapsed());
        assert_eq!(size_of(&heap, map), Value::Int(0));
    }

    #[test]
    #[ignore]
    fn bench_map_constructor() {
        let n = 400;
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Map");
        let pair_objs: Vec<Value> = (0..n)
            .map(|i| {
                let p = heap
                    .alloc_object(Object::new(ObjectKind::Ordinary))
                    .unwrap();
                heap.define_property_str(p, "0", Property::data(Value::Int(i as i32)))
                    .unwrap();
                heap.define_property_str(p, "1", Property::data(Value::Int(i as i32)))
                    .unwrap();
                heap.define_property_str(
                    p,
                    "length",
                    Property {
                        value: Value::Int(2),
                        writable: true,
                        enumerable: false,
                        configurable: false,
                    },
                )
                .unwrap();
                Value::Object(p)
            })
            .collect();
        let src = array_like(&mut heap, &pair_objs);
        let t = std::time::Instant::now();
        let map = call_ctor(&mut heap, ctor, &[Value::Object(src)]).unwrap();
        let dt = t.elapsed();
        match map {
            Value::Object(o) => assert_eq!(size_of(&heap, o), Value::Int(n as i32)),
            _ => panic!("not an object"),
        }
        report("new Map(pairs)", n, dt);
    }

    #[test]
    #[ignore]
    fn bench_set_add_has() {
        let n = 400;
        let t = std::time::Instant::now();
        let (mut heap, set, proto) = bench_set_with(n);
        let dt_add = t.elapsed();
        report("set.add", n, dt_add);
        let has = method_fn(&heap, proto, "has");
        let this = Value::Object(set);
        let t = std::time::Instant::now();
        for i in (0..n).rev() {
            assert_eq!(
                has(&mut heap, this, &[Value::Int(i as i32)]).unwrap(),
                Value::Bool(true)
            );
        }
        report("set.has (reverse)", n, t.elapsed());
    }

    #[test]
    #[ignore]
    fn bench_set_iteration() {
        let n = 400;
        let (mut heap, set, proto) = bench_set_with(n);
        let this = Value::Object(set);
        for name in ["keys", "values", "entries"] {
            let t = std::time::Instant::now();
            let out = call_method(&mut heap, proto, name, this, &[]).unwrap();
            let dt = t.elapsed();
            match out {
                Value::Object(o) => assert_eq!(dense_len(&heap, o) as usize, n),
                _ => panic!("{name} did not return an array"),
            }
            report(&format!("set.{name}"), 1, dt);
        }
    }

    /// Regression: `dense_from_values` held interned keys in an untraced
    /// local `Vec` across one `heap.intern` per element with no rooting at
    /// all. Interned strings are weak roots, so a collection triggered
    /// mid-loop freed (and evicted from the intern table) the keys pushed
    /// earlier, leaving dangling handles in `props` — use-after-free on
    /// later property access. The fix roots every key in `roots` as it is
    /// interned.
    ///
    /// Forces a collection on nearly every allocation (tiny GC threshold)
    /// Builds a 2000-element array with the GC threshold pinned low so
    /// collections fire mid-build, then checks that every stored key
    /// round-trips by content (index keys and `"length"` all pack inline,
    /// so there is no heap string to be freed mid-build) and that every
    /// element reads back correctly. `roots` (which also holds the
    /// finished array) stays alive across verification.
    #[test]
    fn dense_from_values_keys_survive_mid_loop_gc() {
        let mut heap = Heap::new();
        heap.set_gc_threshold(64);
        const N: usize = 2000;
        let values: Vec<Value> = (0..N as i32).map(Value::Int).collect();
        let mut roots = CallRoots::new(&mut heap);
        let arr = dense_from_values(&mut heap, &mut roots, &values).unwrap();
        heap.collect();
        let mut buf = [0u8; 20];
        let (length_key, length_prop) = &heap.get(arr).properties[0];
        assert_eq!(length_key.as_wtf8(), Wtf8::from_str("length"));
        assert!(length_key.is_inline());
        assert_eq!(length_prop.value, num(N as f64));
        for (i, &v) in values.iter().enumerate() {
            let (stored, prop) = &heap.get(arr).properties[i + 1]; // "length" is first
            assert!(stored.is_inline(), "index key {i} should pack inline");
            assert_eq!(
                stored.as_wtf8(),
                Wtf8::from_str(fmt_index(i as u64, &mut buf)),
                "key {i} corrupted mid-build"
            );
            assert_eq!(prop.value, v, "element {i} corrupted");
        }
    }

    #[test]
    #[ignore]
    fn bench_symbol_for() {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Symbol");
        let for_fn = method_fn(&heap, ctor, "for");
        let n = 2000;
        let t = std::time::Instant::now();
        for i in 0..n {
            let key = str_val(&mut heap, &format!("bench-key-{i}"));
            for_fn(&mut heap, Value::Undefined, &[key]).unwrap();
        }
        report("Symbol.for (fresh keys)", n, t.elapsed());
        let t = std::time::Instant::now();
        for i in 0..n {
            let key = str_val(&mut heap, &format!("bench-key-{i}"));
            for_fn(&mut heap, Value::Undefined, &[key]).unwrap();
        }
        report("Symbol.for (cached keys)", n, t.elapsed());
    }

    // -- Map/Set @@iterator -------------------------------------------------

    /// Drive an iterator object through `next()` until `done`, collecting
    /// the values.
    fn drive_iter(heap: &mut Heap, iter: Gc<Object>) -> Vec<Value> {
        let next = method_fn(heap, iter, "next");
        let mut out = Vec::new();
        loop {
            let r = match next(heap, Value::Object(iter), &[]).unwrap() {
                Value::Object(o) => o,
                v => panic!("next() did not return an object: {v:?}"),
            };
            let done = match heap
                .get_property(r, Wtf8::from_str("done"))
                .expect("done missing")
                .value
            {
                Value::Bool(b) => b,
                v => panic!("done is not a bool: {v:?}"),
            };
            if done {
                break;
            }
            out.push(
                heap.get_property(r, Wtf8::from_str("value"))
                    .expect("value missing")
                    .value,
            );
        }
        out
    }

    #[test]
    fn map_iterator_yields_entries_in_insertion_order() {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Map");
        let map = match call_ctor(&mut heap, ctor, &[]).unwrap() {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        let proto = ctor_proto(&heap, ctor);
        for (k, v) in [("a", 1), ("b", 2), ("c", 3)] {
            let kv = str_val(&mut heap, k);
            call_method(
                &mut heap,
                proto,
                "set",
                Value::Object(map),
                &[kv, Value::Int(v)],
            )
            .unwrap();
        }
        // `Symbol.iterator` is installed on the prototype under the real
        // symbol key — and the `"@@iterator"` string key does NOT alias
        // it.
        let sym = iterator_sym(&mut heap);
        assert!(
            heap.get_property_key(proto, PropKey::Symbol(sym)).is_some(),
            "Map.prototype[Symbol.iterator] missing"
        );
        assert!(
            heap.get_property(proto, Wtf8::from_str("@@iterator"))
                .is_none(),
            "\"@@iterator\" string key must not alias Symbol.iterator"
        );
        let iter_fn = method_fn_key(&heap, proto, PropKey::Symbol(sym));
        let it = match iter_fn(&mut heap, Value::Object(map), &[]).unwrap() {
            Value::Object(o) => o,
            v => panic!("Symbol.iterator did not return an object: {v:?}"),
        };
        // Iterators are iterable (`Symbol.iterator` returns `this`).
        let self_fn = method_fn_key(&heap, it, PropKey::Symbol(sym));
        let self_iter = match self_fn(&mut heap, Value::Object(it), &[]).unwrap() {
            Value::Object(o) => o,
            v => panic!("iterator Symbol.iterator did not return an object: {v:?}"),
        };
        assert_eq!(self_iter, it);
        let vals = drive_iter(&mut heap, it);
        assert_eq!(vals.len(), 3);
        for (i, (k, v)) in [("a", 1), ("b", 2), ("c", 3)].iter().enumerate() {
            let pair = dense_to_vec(
                &heap,
                match vals[i] {
                    Value::Object(o) => o,
                    v => panic!("entry {i} is not an object: {v:?}"),
                },
            );
            assert_eq!(pair.len(), 2);
            assert!(matches!(&pair[0], Value::Str(s) if s.as_wtf8() == Wtf8::from_str(k)));
            assert_eq!(pair[1], Value::Int(*v));
        }
    }

    #[test]
    fn set_iterator_yields_values_in_insertion_order() {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Set");
        let set = match call_ctor(&mut heap, ctor, &[]).unwrap() {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        let proto = ctor_proto(&heap, ctor);
        for v in [10, 20, 30] {
            call_method(
                &mut heap,
                proto,
                "add",
                Value::Object(set),
                &[Value::Int(v)],
            )
            .unwrap();
        }
        let sym = iterator_sym(&mut heap);
        assert!(
            heap.get_property_key(proto, PropKey::Symbol(sym)).is_some(),
            "Set.prototype[Symbol.iterator] missing"
        );
        assert!(
            heap.get_property(proto, Wtf8::from_str("@@iterator"))
                .is_none(),
            "\"@@iterator\" string key must not alias Symbol.iterator"
        );
        let iter_fn = method_fn_key(&heap, proto, PropKey::Symbol(sym));
        let it = match iter_fn(&mut heap, Value::Object(set), &[]).unwrap() {
            Value::Object(o) => o,
            v => panic!("Symbol.iterator did not return an object: {v:?}"),
        };
        let vals = drive_iter(&mut heap, it);
        assert_eq!(vals, vec![Value::Int(10), Value::Int(20), Value::Int(30)]);
    }

    #[test]
    fn iterator_next_on_incompatible_receiver_throws() {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Map");
        let proto = ctor_proto(&heap, ctor);
        let map = match call_ctor(&mut heap, ctor, &[]).unwrap() {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        let sym = iterator_sym(&mut heap);
        let it = match call_method_key(
            &mut heap,
            proto,
            PropKey::Symbol(sym),
            Value::Object(map),
            &[],
        )
        .unwrap()
        {
            Value::Object(o) => o,
            v => panic!("Symbol.iterator did not return an object: {v:?}"),
        };
        let next = method_fn(&heap, it, "next");
        assert!(
            next(&mut heap, Value::Undefined, &[]).is_err(),
            "next() with undefined receiver should throw"
        );
    }

    #[test]
    fn iterator_sees_live_delete_and_clear() {
        let (mut heap, global) = installed();
        let ctor = global_ctor(&heap, global, "Set");
        let set = match call_ctor(&mut heap, ctor, &[]).unwrap() {
            Value::Object(o) => o,
            v => panic!("not an object: {v:?}"),
        };
        let proto = ctor_proto(&heap, ctor);
        for v in [1, 2, 3] {
            call_method(
                &mut heap,
                proto,
                "add",
                Value::Object(set),
                &[Value::Int(v)],
            )
            .unwrap();
        }
        let sym = iterator_sym(&mut heap);
        let it = match call_method_key(
            &mut heap,
            proto,
            PropKey::Symbol(sym),
            Value::Object(set),
            &[],
        )
        .unwrap()
        {
            Value::Object(o) => o,
            v => panic!("Symbol.iterator did not return an object: {v:?}"),
        };
        let next = method_fn(&heap, it, "next");
        // First value.
        let r1 = match next(&mut heap, Value::Object(it), &[]).unwrap() {
            Value::Object(o) => o,
            v => panic!("next() did not return an object: {v:?}"),
        };
        assert_eq!(
            heap.get_property(r1, Wtf8::from_str("value"))
                .unwrap()
                .value,
            Value::Int(1)
        );
        // Delete the pending value 2: the iterator must skip it.
        call_method(
            &mut heap,
            proto,
            "delete",
            Value::Object(set),
            &[Value::Int(2)],
        )
        .unwrap();
        let rest = drive_iter(&mut heap, it);
        assert_eq!(rest, vec![Value::Int(3)]);
    }
}
