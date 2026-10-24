//! `Error` family and `Function` builtins.
//!
//! Installs `Error` (plus `EvalError`, `RangeError`, `ReferenceError`,
//! `SyntaxError`, `TypeError`, `URIError`) and `Function` on the global
//! object via [`install`]. The seven error prototypes and
//! `%Function.prototype%` are stashed in [`Heap`] fields (traced as roots),
//! following the sibling `builtin_object_array` convention: native
//! constructors look their prototype up on the heap because [`NativeFn`]
//! receives no callee and no `new.target`.
//!
//! `JSON` and `Math` are intentionally *not* here: they already live in
//! `crate::builtin_string_number`.
//!
//! ## v1 gaps (for the vm worker)
//!
//! - **No VM re-entry.** `Function(...)` cannot compile its body argument,
//!   so the constructor returns a function object whose invocation throws
//!   `TypeError` ("dynamic code compilation is a v1 gap"). When the vm
//!   worker provides compile-and-run re-entry, rewire
//!   [`function_constructor`] to produce a real [`ObjectKind::Function`].
//! - **`bind` builds an invocable bound function.** The bound target /
//!   this / args live in hidden slots ([`BOUND_TARGET`], [`BOUND_THIS`],
//!   [`BOUND_ARGS`]); the VM's dispatch intercepts [`Object::INVOKE_BOUND`]
//!   before reaching the native entry, and the sibling
//!   [`call_function`](crate::builtin_object_array::call_function) choke
//!   point mirrors that interception, so bound functions work through
//!   `call` / `apply`, direct invocation, and builtins taking callbacks
//!   (e.g. `arr.map(boundFn)`). [`bound_function_invoke`] itself is now an
//!   unreachable fallback that throws `TypeError` if dispatch ever reaches
//!   it. Chains are flattened at bind time (one hidden-slot hop no matter
//!   how many times `.bind()` was chained), and the bound arguments use
//!   dense array storage so each bind is O(bound args). The bound `name`
//!   is derived lazily — a prefix count plus the ultimate base name in
//!   hidden slots, materialized only when `name` is read — so a deep
//!   `.bind()` chain costs O(1) name bytes per link instead of O(depth).
//! - **`call`/`apply` invoke native functions, bytecode functions, and
//!   bound functions.** All three go through the sibling
//!   [`call_function`](crate::builtin_object_array::call_function) choke
//!   point (bytecode callees re-enter the VM via `vm::reenter_call`;
//!   bound callees resolve their hidden slots and recurse).
//! - **No `stack` traces.** Error instances carry an own `message` property
//!   only when constructed with one, and inherit `name` from their
//!   prototype; no call-stack is captured.
//! - **No `new.target` distinction.** `Error("x")` and `new Error("x")`
//!   both take the construct path.
//! - **Install order.** Links to `%Object.prototype%` are made
//!   opportunistically: if the object worker has not installed yet, the new
//!   prototypes temporarily chain to `null`.
//!
//! ## Rooting
//!
//! Follows the `builtin_object_array` v1 decision: automatic collection is
//! disabled by default, but an embedder may enable it before installing.
//! [`install`] therefore roots `global` for the whole install, stashes
//! each prototype in its [`Heap`] field as soon as it is allocated, and
//! [`install_error`] holds its fresh prototype across the allocations that
//! precede the constructor wiring. [`error_construct`] holds the fresh
//! error object across the message-string allocation.

use crate::builtin_object_array::{
    call_function, get_arg, get_length, get_own, get_prop, is_callable, num, to_length,
    to_string_js, to_string_js_str, walk_proto_chain,
};
use crate::builtin_reflect_proxy::{get_prop_chain, new_native_constructor};
use crate::builtin_string_number::{define_native, new_native_function};
use crate::gc::{Gc, Heap};
use crate::value::{
    AccessorProp, NativeError, NativeFn, Object, ObjectKind, Property, ShapeId, Value,
};
use crate::vm::active_new_target;
use crate::wtf8::{Wtf8, Wtf8Buf};

// ---------------------------------------------------------------------------
// Small helpers (module-private; copied pattern from sibling builtins)
// ---------------------------------------------------------------------------

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

/// Link `proto`'s `[[Prototype]]` to `%Object.prototype%` when it is
/// already installed (install order between builtin workers is not fixed).
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

/// A global binding: writable, non-enumerable, configurable (spec 18).
fn define_global(
    heap: &mut Heap,
    global: Gc<Object>,
    name: &str,
    value: Value,
) -> Result<(), NativeError> {
    heap.define_property_str(
        global,
        name,
        Property {
            value,
            writable: true,
            enumerable: false,
            configurable: true,
        },
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Error constructors
// ---------------------------------------------------------------------------

/// Shared `%Error%`-family constructor body: builds an ordinary object with
/// `proto` as its `[[Prototype]]`. An own `message` property is installed
/// only when the message argument is present and not `undefined` (spec
/// 20.5.8.1), so `new Error()` has no own `message`.
fn error_construct(heap: &mut Heap, args: &[Value], proto: Value) -> Result<Value, NativeError> {
    let message = match get_arg(args, 0) {
        Value::Undefined => None,
        v => Some(to_string_js_str(heap, v)?),
    };
    // `to_string_js_str` may hand back an unrooted `Str::Heap`: a `toString`
    // result whose temporary roots `call_function` already released, or a
    // fresh `alloc_str` handle. Root it across the allocations below, whose
    // collections would otherwise free it before it is stored on the error.
    // Released on both paths after the closure (`release_temp` truncates
    // everything above the base, so the error root pushed inside is dropped
    // too).
    let msg_base = message.as_ref().map(|&s| heap.protect(Value::Str(s)));
    let result = (|| {
        let err = heap.alloc_object(ordinary_object(proto))?;
        // `err` is not reachable from any root yet; hold it across the
        // property definition, whose key interning may allocate and collect.
        let err_base = heap.protect(Value::Object(err));
        let inner = (|| {
            if let Some(s) = message {
                heap.define_property_str(err, "message", Property::data(Value::Str(s)))?;
            }
            Ok(Value::Object(err))
        })();
        heap.release_temp(err_base);
        inner
    })();
    if let Some(b) = msg_base {
        heap.release_temp(b);
    }
    result
}

/// Generate one constructor `fn` per error kind. Each reads its prototype
/// from the matching [`Heap`] field because [`NativeFn`] cannot see which
/// constructor object was invoked.
macro_rules! error_constructor {
    ($fn_name:ident, $proto_field:ident) => {
        fn $fn_name(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
            let proto = heap.$proto_field.map(Value::Object).unwrap_or(Value::Null);
            error_construct(heap, args, proto)
        }
    };
}

error_constructor!(error_constructor, error_prototype);
error_constructor!(eval_error_constructor, eval_error_prototype);
error_constructor!(range_error_constructor, range_error_prototype);
error_constructor!(reference_error_constructor, reference_error_prototype);
error_constructor!(syntax_error_constructor, syntax_error_prototype);
error_constructor!(type_error_constructor, type_error_prototype);
error_constructor!(uri_error_constructor, uri_error_prototype);

/// `SuppressedError(error, suppressed, message?)` (explicit resource
/// management): an error carrying the original `error` and the
/// `suppressed` disposal error.
fn suppressed_error_constructor(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let error = get_arg(args, 0);
    let suppressed = get_arg(args, 1);
    let message = match get_arg(args, 2) {
        Value::Undefined => None,
        v => Some(to_string_js_str(heap, v)?),
    };
    let proto = heap
        .suppressed_error_prototype
        .map(Value::Object)
        .unwrap_or(Value::Null);
    // The message handle from `to_string_js_str` may be an unrooted
    // `Str::Heap` (see `error_construct`): root it before `alloc_object`,
    // whose collection would otherwise free it before it is stored.
    let msg_base = message.as_ref().map(|&s| heap.protect(Value::Str(s)));
    let err = heap.alloc_object(ordinary_object(proto))?;
    let base = heap.protect(Value::Object(err));
    let result = (|| {
        heap.define_property_str(err, "error", Property::data(error))?;
        heap.define_property_str(err, "suppressed", Property::data(suppressed))?;
        if let Some(s) = message {
            heap.define_property_str(err, "message", Property::data(Value::Str(s)))?;
        }
        Ok(Value::Object(err))
    })();
    // `base` was pushed after `msg_base`, so releasing the earlier base
    // drops both roots; with no message the error root is the only one.
    heap.release_temp(msg_base.unwrap_or(base));
    result
}

/// `Error.prototype.toString()` (spec 20.5.3.4): `name: message`, where a
/// missing/`undefined` `name` becomes `"Error"` and a missing/`undefined`
/// `message` becomes `""`.
fn error_prototype_to_string(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let obj = match this {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "Error.prototype.toString called on a non-object receiver".into(),
            ));
        }
    };
    let name = match get_prop(heap, obj, Wtf8::from_str("name")) {
        None | Some(Value::Undefined) => Wtf8Buf::from("Error"),
        Some(v) => to_string_js(heap, v)?,
    };
    let message = match get_prop(heap, obj, Wtf8::from_str("message")) {
        None | Some(Value::Undefined) => Wtf8Buf::new(),
        Some(v) => to_string_js(heap, v)?,
    };
    let out = if name.is_empty() {
        message
    } else if message.is_empty() {
        name
    } else {
        let mut out = name;
        out.push_str(": ");
        out.push_wtf8(&message);
        out
    };
    Ok(Value::Str(heap.alloc_str_wtf8(out.as_wtf8())?))
}

/// Install one error kind: a prototype object (`name`/`message` defaults,
/// chained to `parent_proto`), a constructor wired to it (the constructor's
/// `[[Prototype]]` is `ctor_parent` — `%Error%` for the NativeError
/// subtypes, `%Function.prototype%` for `%Error%` itself), and the
/// constructor on `global`. Returns `(proto, ctor)`; the caller stashes
/// the prototype in the matching [`Heap`] field.
///
/// Note the constructor's `[[Prototype]]` is *not* the function-prototype
/// link: `%TypeError%.__proto__ === %Error%` (the constructor object),
/// while `%TypeError.prototype%.__proto__ === %Error.prototype%`
/// (spec 20.5.7.1). [`new_native_function`](crate::builtin_string_number::new_native_function)
/// would default it to `%Function.prototype%`, which is correct only for
/// `%Error%` itself (spec 20.5.6.1).
fn install_error(
    heap: &mut Heap,
    global: Gc<Object>,
    name: &str,
    ctor: NativeFn,
    parent_proto: Value,
    ctor_parent: Value,
) -> Result<(Gc<Object>, Gc<Object>), NativeError> {
    let proto = heap.alloc_object(ordinary_object(parent_proto))?;
    // `proto` is not reachable from any root until it is wired to the
    // constructor and stashed in the `Heap` field by the caller; hold it
    // across the allocations below, any of which may trigger a collection.
    // (The closure keeps the temporary root released on the error path.)
    let base = heap.protect(Value::Object(proto));
    let result = (|| {
        let name_str = heap.alloc_str(name)?;
        heap.define_property_str(
            proto,
            "name",
            Property {
                value: Value::Str(name_str),
                writable: true,
                enumerable: false,
                configurable: true,
            },
        )?;
        let empty = heap.alloc_str("")?;
        heap.define_property_str(
            proto,
            "message",
            Property {
                value: Value::Str(empty),
                writable: true,
                enumerable: false,
                configurable: true,
            },
        )?;
        let ctor_obj = new_native_constructor(heap, name, ctor, 1)?;
        // `set_prototype` only writes the field: no allocation, so the
        // temporary root above still covers `proto` and the constructor
        // is reachable via `global` after `define_global` below.
        heap.set_prototype(ctor_obj, ctor_parent);
        wire_ctor_proto(heap, ctor_obj, proto)?;
        define_global(heap, global, name, Value::Object(ctor_obj))?;
        Ok((proto, ctor_obj))
    })();
    heap.release_temp(base);
    result
}

// ---------------------------------------------------------------------------
// Function constructor + Function.prototype
// ---------------------------------------------------------------------------

/// `%Function.prototype%` is itself callable and returns `undefined`
/// (spec 20.2.3).
fn function_prototype_call(
    _heap: &mut Heap,
    _this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    Ok(Value::Undefined)
}

/// Invocation target for objects built by [`function_constructor`].
/// v1 cannot compile the body argument, so calling one always throws.
fn dynamic_function_throw(
    _heap: &mut Heap,
    _this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    Err(NativeError::TypeError(
        "Function constructor: dynamic code compilation is a v1 gap".into(),
    ))
}

/// `Function(...)`: builds a callable object whose invocation throws (see
/// [`dynamic_function_throw`]). Spec-shaped `name` (`"anonymous"`) and
/// `length` (`1`), prototype `%Function.prototype%`.
fn function_constructor(
    heap: &mut Heap,
    _this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    // OrdinaryCreateFromConstructor(newTarget, "%Function.prototype%"):
    // a derived `class D extends Function` calling `super()` gets
    // D.prototype as the instance's prototype (see object_constructor in
    // builtin_object_array.rs).
    let proto = match active_new_target() {
        Some(Value::Object(nt)) => match get_prop_chain(heap, nt, Wtf8::from_str("prototype")) {
            Some(Value::Object(p)) => Value::Object(p),
            _ => heap
                .function_prototype
                .map(Value::Object)
                .unwrap_or(Value::Null),
        },
        _ => heap
            .function_prototype
            .map(Value::Object)
            .unwrap_or(Value::Null),
    };
    let f = heap.alloc_object(Object {
        kind: ObjectKind::NativeFunction(dynamic_function_throw),
        prototype: proto,
        ..Default::default()
    })?;
    let name = heap.alloc_str("anonymous")?;
    heap.define_property_str(
        f,
        "name",
        Property {
            value: Value::Str(name),
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )?;
    heap.define_property_str(
        f,
        "length",
        Property {
            value: Value::Int(1),
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )?;
    Ok(Value::Object(f))
}

/// `Function.prototype.call(thisArg, ...args)`: invokes the receiver via
/// the [`call_function`] choke point.
fn function_call(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    if !is_callable(heap, this) {
        return Err(NativeError::TypeError(
            "Function.prototype.call called on a non-callable receiver".into(),
        ));
    }
    let this_arg = get_arg(args, 0);
    let rest = args.get(1..).unwrap_or(&[]);
    call_function(heap, this, this_arg, rest)
}

/// `Function.prototype.apply(thisArg, argArray)`: `argArray` is read as an
/// array-like (v1: no iterable protocol); `null`/`undefined` means no args.
fn function_apply(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    if !is_callable(heap, this) {
        return Err(NativeError::TypeError(
            "Function.prototype.apply called on a non-callable receiver".into(),
        ));
    }
    let this_arg = get_arg(args, 0);
    let mut call_args: Vec<Value> = Vec::new();
    match get_arg(args, 1) {
        Value::Undefined | Value::Null => {}
        Value::Object(arr) => {
            let len = get_length(heap, arr)?;
            for i in 0..len {
                call_args.push(
                    get_own(heap, arr, Wtf8::from_str(&i.to_string())).unwrap_or(Value::Undefined),
                );
            }
        }
        _ => {
            return Err(NativeError::TypeError(
                "Function.prototype.apply: argument list must be an object".into(),
            ));
        }
    }
    call_function(heap, this, this_arg, &call_args)
}

/// Hidden slots on bound function objects (non-enumerable, following the
/// sibling `builtin_collections` convention).
const BOUND_TARGET: &str = "\u{0}bowser:bound-target";
const BOUND_THIS: &str = "\u{0}bowser:bound-this";
const BOUND_ARGS: &str = "\u{0}bowser:bound-args";
/// Compact lazy-name representation: the number of `"bound "` prefixes and
/// the ultimate target's base name. The full `"bound ".repeat(n) + base`
/// string is materialized only when the `name` accessor is read (see
/// [`bound_name_getter`]), so a deep `.bind()` chain costs O(1) name bytes
/// per link instead of O(depth) — the old eager
/// `format!("bound {target_name}")` made a 100k-deep chain retain ~GBs of
/// name bytes across dead-but-uncollected bound objects (GC is opt-in).
const BOUND_NAME_COUNT: &str = "\u{0}bowser:bound-name-count";
const BOUND_NAME_BASE: &str = "\u{0}bowser:bound-name-base";

/// The callable installed on every bound function object. Unreachable in
/// normal dispatch: the VM's `invoke` and the shared [`call_function`]
/// choke point both intercept [`Object::INVOKE_BOUND`] and resolve the
/// hidden slots before the native entry is ever called. It throws
/// `TypeError` if reached, so a missed interception fails loudly.
fn bound_function_invoke(
    _heap: &mut Heap,
    _this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    Err(NativeError::TypeError(
        "bound function invocation is a v1 gap: NativeFn cannot observe its callee".into(),
    ))
}

/// Getter backing the lazy `name` accessor installed on every bound
/// function object. Materializes `"bound ".repeat(count) + base` from the
/// [`BOUND_NAME_COUNT`] / [`BOUND_NAME_BASE`] hidden slots, then folds the
/// accessor back into the spec-shaped data property (non-writable,
/// non-enumerable, configurable) so repeated reads are O(1) and the
/// descriptor matches `Function.prototype.bind`'s contract — unless the
/// bound function is non-extensible (frozen / `preventExtensions`-ed), in
/// which case the accessor is left in place and only the materialized
/// name is returned (installing anything there would break the
/// frozen/non-extensible invariants). The VM copies the accessor
/// descriptor before invoking the getter, so redefining the property
/// inside the getter is safe.
///
/// The receiver is usually the bound function itself, but the lazy
/// accessor is inherited when the bound function is used as a prototype
/// (`Object.create(b).name` invokes this getter with the derived object
/// as receiver). Per spec the read then still yields the bound function's
/// name, so the receiver's prototype chain is walked (cycle-guarded) for
/// the bound function that owns the lazy `name` slots; slot reads and the
/// fold always target it, never the receiver. A receiver whose chain holds
/// no bound function throws `TypeError`, mirroring how other built-in
/// getters reject incompatible receivers.
///
/// Known deviation, accepted cost of laziness: before the first read,
/// `Object.getOwnPropertyDescriptor(b, 'name')` reports an accessor
/// descriptor where ECMA-262 19.2.3.2 specifies a data descriptor. The
/// first read folds it into the spec-shaped data property, so the
/// deviation is observable only pre-read.
fn bound_name_getter(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let receiver = match this {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "bound function name getter called on a non-object".into(),
            ));
        }
    };
    // Resolve the bound function that owns the lazy `name` slots: the
    // receiver itself in the common case, or the nearest bound function
    // up its prototype chain when the accessor was inherited (issue 3b).
    // Slot reads and the fold below target the owner, never the receiver.
    let owner = walk_proto_chain(heap, receiver, |o| {
        (heap.get(o).invoke_flags & Object::INVOKE_BOUND != 0).then_some(o)
    });
    let obj = match owner {
        Some(o) => o,
        None => {
            return Err(NativeError::TypeError(
                "bound function name getter called on an incompatible receiver".into(),
            ));
        }
    };
    let count = match get_own(heap, obj, Wtf8::from_str(BOUND_NAME_COUNT)) {
        Some(Value::Int(i)) if i >= 0 => i as usize,
        Some(Value::Float(f)) if f >= 0.0 && f.is_finite() => f as usize,
        _ => {
            return Err(NativeError::TypeError(
                "bound function is missing its name slots".into(),
            ));
        }
    };
    let base = match get_own(heap, obj, Wtf8::from_str(BOUND_NAME_BASE)) {
        Some(Value::Str(g)) => g,
        _ => {
            return Err(NativeError::TypeError(
                "bound function is missing its name slots".into(),
            ));
        }
    };
    let base_str = base.as_wtf8();
    // Function names are identifiers (always well-formed).
    let base_str = base_str.as_utf8().unwrap_or("");
    let mut name = String::with_capacity(count.saturating_mul(6).saturating_add(base_str.len()));
    for _ in 0..count {
        name.push_str("bound ");
    }
    name.push_str(base_str);
    // `obj` may be unreachable from any GC root (e.g. the getter was
    // borrowed off the descriptor while the object itself was dropped);
    // hold it across `alloc_str`, which can collect. `base` stays live
    // through `obj`: it is reachable via the BOUND_NAME_BASE slot.
    let root = heap.protect(Value::Object(obj));
    let name_val = heap.alloc_str(&name).map(Value::Str);
    heap.release_temp(root);
    let name_val = name_val?;
    // Cache the materialized string as the spec-shaped data property;
    // defining a data property evicts the accessor under the same key.
    // Only while the owner is still extensible: on a frozen or
    // `preventExtensions`-ed bound function the unvalidated define would
    // evict a non-configurable accessor and install a configurable data
    // property on a non-extensible object, flipping `Object.isFrozen`
    // true->false from a mere read. A non-extensible owner keeps the
    // accessor; reads stay correct either way.
    if heap.get(obj).extensible {
        heap.define_property_str(
            obj,
            "name",
            Property {
                value: name_val,
                writable: false,
                enumerable: false,
                configurable: true,
            },
        )?;
    }
    Ok(name_val)
}

/// `Function.prototype.bind(thisArg, ...args)`: builds a spec-shaped bound
/// function (hidden slots, lazily-derived `"bound <name>"`, adjusted
/// `length`) that dispatches to its target through the
/// [`Object::INVOKE_BOUND`] interception in the VM and in
/// [`call_function`].
///
/// Bind chains are flattened at bind time: binding an already-bound
/// function merges into a single layer (ultimate target, innermost
/// `thisArg`, concatenated bound arguments) instead of nesting, so
/// invocation does one hidden-slot hop and builds one argument vector no
/// matter how many times `.bind()` was chained. The `"bound "` prefix
/// count is likewise flattened into the [`BOUND_NAME_COUNT`] hidden slot,
/// so each link costs O(1) name bytes and the full
/// `"bound ".repeat(n) + base` string is materialized only when `name` is
/// read (see [`bound_name_getter`]).
fn function_bind(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let target = match this {
        Value::Object(o) if is_callable(heap, this) => o,
        _ => {
            return Err(NativeError::TypeError(
                "Function.prototype.bind called on a non-callable receiver".into(),
            ));
        }
    };
    // Flatten: an outer `thisArg` is discarded per spec (a bound
    // function's `this` is fixed at bind time), while bound arguments
    // accumulate inner-first. `name`/`length` below still derive from the
    // immediate target, preserving the observable `"bound bound f"`
    // naming and length adjustment of the unflattened shape.
    let (flat_target, flat_this, flat_args): (Gc<Object>, Value, Vec<Value>) =
        match crate::vm::bound_parts(heap, target) {
            Some((Value::Object(inner), inner_this, mut inner_args)) => {
                inner_args.extend(args.iter().skip(1).copied());
                (inner, inner_this, inner_args)
            }
            _ => (
                target,
                get_arg(args, 0),
                args.iter().skip(1).copied().collect(),
            ),
        };
    // Dense storage for the bound arguments: `bound_parts` reads them back
    // through `array_to_vec`, which hits the dense O(1) index fast path in
    // `Heap::get_property`, so each bind is O(bound args) instead of
    // O(bound args²) (no per-element property interning or linear scan).
    // The array stays dense in its hidden slot — never user-visible, and
    // every reader goes through `get_property` — so it is not materialized.
    let bound_args = heap.new_dense_array()?;
    // `bound_args` is unreachable from any root until it is installed in
    // `bound`'s BOUND_ARGS slot below; hold it across the allocations that
    // follow (`alloc_object` in particular can collect). A collection
    // there would otherwise sweep the array and leave a dangling handle
    // in the slot — the same use-after-free class as the lazy-name
    // `base_name` below.
    let args_root = heap.protect(Value::Object(bound_args));
    for a in flat_args.iter() {
        heap.dense_push(bound_args, *a)?;
    }

    let proto = heap
        .function_prototype
        .map(Value::Object)
        .unwrap_or(Value::Null);
    let bound = heap.alloc_object(Object {
        kind: ObjectKind::NativeFunction(bound_function_invoke),
        prototype: proto,
        invoke_flags: Object::INVOKE_BOUND,
        ..Default::default()
    })?;
    let hidden = Property {
        value: Value::Null,
        writable: false,
        enumerable: false,
        configurable: false,
    };
    heap.define_property_str(
        bound,
        BOUND_TARGET,
        Property {
            value: Value::Object(flat_target),
            ..hidden
        },
    )?;
    heap.define_property_str(
        bound,
        BOUND_THIS,
        Property {
            value: flat_this,
            ..hidden
        },
    )?;
    heap.define_property_str(
        bound,
        BOUND_ARGS,
        Property {
            value: Value::Object(bound_args),
            ..hidden
        },
    )?;
    // Installed: `bound`'s own temporary root now keeps the array alive.
    heap.release_temp(args_root);
    let target_name = match get_own(heap, target, Wtf8::from_str("name")) {
        // A string own `"name"`: either a plain target's name, or a
        // user-redefined name on an already-bound target (its cached,
        // materialized name also lands here). One more `"bound "` prefix,
        // exactly as before.
        Some(Value::Str(g)) => (1u32, g.as_wtf8().to_owned()),
        // A bound target still carrying the lazy representation: bump the
        // prefix count and keep sharing the ultimate base name. O(1), no
        // string is built — this is what keeps deep `.bind()` chains from
        // retaining O(depth) name bytes per link.
        _ if heap.get(target).invoke_flags & Object::INVOKE_BOUND != 0 => {
            let inner = match get_own(heap, target, Wtf8::from_str(BOUND_NAME_COUNT)) {
                Some(Value::Int(i)) if i >= 0 => i as u32,
                Some(Value::Float(f)) if f >= 0.0 && f.is_finite() => f as u32,
                _ => 0,
            };
            let base = match get_own(heap, target, Wtf8::from_str(BOUND_NAME_BASE)) {
                Some(Value::Str(g)) => g.as_wtf8().to_owned(),
                _ => Wtf8Buf::new(),
            };
            (inner.saturating_add(1), base)
        }
        // Declared functions carry no own `"name"` property (the name
        // lives in `FunctionData`); fall back to it so bound functions
        // of declared targets still get `"bound <name>"`.
        _ => match &heap.get(target).kind {
            ObjectKind::Function(data) => (1, data.name.as_wtf8().to_owned()),
            _ => (1, Wtf8Buf::new()),
        },
    };
    // Root `bound` across the allocations below (`new_native_function`,
    // `alloc_str`): it is not yet reachable from any GC root, and an
    // embedder-enabled collection must not free it mid-build.
    let root = heap.protect(Value::Object(bound));
    // The COUNT define runs BEFORE `alloc_str` on purpose. `base_name` is
    // a heap `Str` for target names longer than `STR_INLINE_MAX` bytes,
    // and it is unreachable from any root until the BASE define below
    // installs it — so it must not be live across any collecting call
    // before then. `define_property_ordinary` roots `prop.value` across
    // the key `intern_str`, which makes the BASE define (the first
    // collecting call `base_name` is exposed to) safe; the COUNT define's
    // own `intern_str` collection would sweep an already-allocated
    // `base_name`, so allocating it first is use-after-free. Do not move
    // the `alloc_str` above the COUNT define.
    heap.define_property_str(
        bound,
        BOUND_NAME_COUNT,
        Property {
            value: num(f64::from(target_name.0)),
            ..hidden
        },
    )?;
    let base_name = heap
        .alloc_str_wtf8(target_name.1.as_wtf8())
        .map(Value::Str)?;
    heap.define_property_str(
        bound,
        BOUND_NAME_BASE,
        Property {
            value: base_name,
            ..hidden
        },
    )?;
    // The `name` is a getter-only accessor (non-enumerable, configurable)
    // until first read; [`bound_name_getter`] materializes
    // `"bound ".repeat(n) + base` and folds it into the spec-shaped data
    // property. Reads therefore stay observationally `"bound <name>"`
    // while binds stay O(1) in name bytes.
    let name_getter = Value::Object(new_native_function(heap, "get name", bound_name_getter, 0)?);
    heap.define_accessor(
        bound,
        Wtf8::from_str("name"),
        AccessorProp {
            get: name_getter,
            set: Value::Undefined,
            enumerable: false,
            configurable: true,
        },
    )?;
    heap.release_temp(root);
    let target_len = match get_own(heap, target, Wtf8::from_str("length")) {
        Some(v) => to_length(v).unwrap_or(0),
        None => 0,
    };
    let bound_count = args.len().saturating_sub(1) as u64;
    heap.define_property_str(
        bound,
        "length",
        Property {
            value: num(target_len.saturating_sub(bound_count) as f64),
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )?;
    Ok(Value::Object(bound))
}

// ---------------------------------------------------------------------------
// install
// ---------------------------------------------------------------------------

/// Install `Error` (+ the six standard subtypes) and `Function` (+
/// `Function.prototype` with `call`/`apply`/`bind`) on `global`, stashing
/// the prototypes in [`Heap`] fields.
///
/// Call once per heap; not idempotent. Called by the VM bootstrap
/// ([`Vm::new`](crate::vm::Vm::new)) when it installs the builtin modules
/// on the fresh global object.
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
    // ---- Error family ---------------------------------------------------
    // Constructor `[[Prototype]]` links: each NativeError constructor's is
    // `%Error%` (spec 20.5.7.1); `%Error%`'s own is `%Function.prototype%`
    // (spec 20.5.6.1), linked after the Function install below because the
    // Function intrinsics do not exist yet. The subtype links are made
    // eagerly here, which also protects them from
    // `Heap::fixup_native_function_prototypes`: it only repairs
    // null-prototype native functions, so an explicit link is never
    // clobbered.
    let (error_proto, error_ctor) = install_error(
        heap,
        global,
        "Error",
        error_constructor,
        Value::Null,
        Value::Null,
    )?;
    link_to_object_prototype(heap, error_proto);
    // `Error.prototype.toString` (spec 20.5.3.4); the six subtypes inherit
    // it through the prototype chain.
    define_native(heap, error_proto, "toString", error_prototype_to_string, 0)?;
    heap.error_prototype = Some(error_proto);
    let parent = Value::Object(error_proto);
    let error_ctor_val = Value::Object(error_ctor);
    heap.eval_error_prototype = Some(
        install_error(
            heap,
            global,
            "EvalError",
            eval_error_constructor,
            parent,
            error_ctor_val,
        )?
        .0,
    );
    heap.range_error_prototype = Some(
        install_error(
            heap,
            global,
            "RangeError",
            range_error_constructor,
            parent,
            error_ctor_val,
        )?
        .0,
    );
    heap.reference_error_prototype = Some(
        install_error(
            heap,
            global,
            "ReferenceError",
            reference_error_constructor,
            parent,
            error_ctor_val,
        )?
        .0,
    );
    heap.syntax_error_prototype = Some(
        install_error(
            heap,
            global,
            "SyntaxError",
            syntax_error_constructor,
            parent,
            error_ctor_val,
        )?
        .0,
    );
    heap.type_error_prototype = Some(
        install_error(
            heap,
            global,
            "TypeError",
            type_error_constructor,
            parent,
            error_ctor_val,
        )?
        .0,
    );
    heap.uri_error_prototype = Some(
        install_error(
            heap,
            global,
            "URIError",
            uri_error_constructor,
            parent,
            error_ctor_val,
        )?
        .0,
    );
    heap.suppressed_error_prototype = Some(
        install_error(
            heap,
            global,
            "SuppressedError",
            suppressed_error_constructor,
            parent,
            error_ctor_val,
        )?
        .0,
    );

    // ---- Function -------------------------------------------------------
    let function_proto = heap.alloc_object(Object {
        kind: ObjectKind::NativeFunction(function_prototype_call),
        prototype: Value::Null,
        ..Default::default()
    })?;
    // Rooted as a heap intrinsic before any further allocation below.
    heap.function_prototype = Some(function_proto);
    link_to_object_prototype(heap, function_proto);
    let empty_name = heap.alloc_str("")?;
    heap.define_property_str(
        function_proto,
        "name",
        Property {
            value: Value::Str(empty_name),
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )?;
    heap.define_property_str(
        function_proto,
        "length",
        Property {
            value: Value::Int(0),
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )?;
    define_native(heap, function_proto, "call", function_call, 1)?;
    define_native(heap, function_proto, "apply", function_apply, 2)?;
    define_native(heap, function_proto, "bind", function_bind, 1)?;
    let function_ctor = new_native_constructor(heap, "Function", function_constructor, 1)?;
    wire_ctor_proto(heap, function_ctor, function_proto)?;
    define_global(heap, global, "Function", Value::Object(function_ctor))?;
    // `%Error%.[[Prototype]]` is `%Function.prototype%` (spec 20.5.6.1).
    // The link could not be made during the Error-family install above
    // because the Function intrinsics did not exist yet; making it now
    // (rather than relying on `Heap::fixup_native_function_prototypes`)
    // keeps the module-level unit tests deterministic.
    heap.set_prototype(error_ctor, Value::Object(function_proto));
    // `%Function%.[[Prototype]]` is `%Function.prototype%` (spec
    // 20.2.3.1): already the case, since `new_native_function` linked the
    // freshly installed `%Function.prototype%` at construction time.
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin_object_array as oa;
    use crate::builtin_string_number as sn;

    /// Full builtin surface: Object/Array, String/Number/BigInt/Math/JSON,
    /// and Error/Function.
    fn setup() -> (Heap, Gc<Object>) {
        let mut heap = Heap::new();
        let global = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        oa::install(&mut heap, global).unwrap();
        sn::install(&mut heap, global).unwrap();
        install(&mut heap, global).unwrap();
        (heap, global)
    }

    fn global_native(heap: &Heap, global: Gc<Object>, name: &str) -> (Gc<Object>, NativeFn) {
        match heap
            .get_property(global, Wtf8::from_str(name))
            .unwrap()
            .value
        {
            Value::Object(o) => match heap.get(o).kind {
                ObjectKind::NativeFunction(f) => (o, f),
                _ => panic!("{name} is not a native function"),
            },
            v => panic!("no {name} on global: {v:?}"),
        }
    }

    fn method_native(heap: &Heap, obj: Gc<Object>, name: &str) -> NativeFn {
        match oa::get_prop(heap, obj, Wtf8::from_str(name)) {
            Some(Value::Object(o)) => match heap.get(o).kind {
                ObjectKind::NativeFunction(f) => f,
                _ => panic!("{name} is not a native function"),
            },
            v => panic!("no method {name}: {v:?}"),
        }
    }

    fn s(heap: &mut Heap, text: &str) -> Value {
        Value::Str(heap.alloc_str(text).unwrap())
    }

    fn as_num(v: Value) -> f64 {
        match v {
            Value::Int(i) => i as f64,
            Value::Float(f) => f,
            v => panic!("expected number, got {v:?}"),
        }
    }

    fn as_str(_heap: &Heap, v: Value) -> String {
        match v {
            Value::Str(g) => g.as_wtf8().to_string(),
            v => panic!("expected string, got {v:?}"),
        }
    }

    #[test]
    fn install_exposes_all_requested_globals() {
        let (heap, global) = setup();
        for name in [
            "Object",
            "Array",
            "Error",
            "EvalError",
            "RangeError",
            "ReferenceError",
            "SyntaxError",
            "TypeError",
            "URIError",
            "Function",
            "JSON",
            "Math",
        ] {
            let p = heap
                .get_property(global, Wtf8::from_str(name))
                .unwrap_or_else(|| panic!("{name} not installed"));
            assert!(
                matches!(p.value, Value::Object(_)),
                "{name} is not an object"
            );
        }
    }

    #[test]
    fn error_message_and_prototype_chain() {
        let (mut heap, global) = setup();
        let (_, type_error) = global_native(&heap, global, "TypeError");
        let msg = s(&mut heap, "boom");
        let err = match type_error(&mut heap, Value::Undefined, &[msg]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected object, got {v:?}"),
        };
        assert_eq!(
            as_str(
                &heap,
                oa::get_own(&heap, err, Wtf8::from_str("message")).unwrap()
            ),
            "boom"
        );
        // err -> %TypeError.prototype% -> %Error.prototype% -> %Object.prototype%
        assert_eq!(
            heap.get(err).prototype,
            Value::Object(heap.type_error_prototype.unwrap())
        );
        assert_eq!(
            heap.get(heap.type_error_prototype.unwrap()).prototype,
            Value::Object(heap.error_prototype.unwrap())
        );
        assert_eq!(
            heap.get(heap.error_prototype.unwrap()).prototype,
            Value::Object(heap.object_prototype.unwrap())
        );
        // `name` is inherited from the prototype.
        assert_eq!(
            as_str(
                &heap,
                oa::get_prop(&heap, err, Wtf8::from_str("name")).unwrap()
            ),
            "TypeError"
        );
        // Missing message: no own `message` property (spec 20.5.8.1).
        let (_, error) = global_native(&heap, global, "Error");
        let bare = match error(&mut heap, Value::Undefined, &[]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected object, got {v:?}"),
        };
        assert!(
            oa::get_own(&heap, bare, Wtf8::from_str("message")).is_none(),
            "bare Error must not have an own message property"
        );
        // `undefined` counts as missing too.
        let undef = match error(&mut heap, Value::Undefined, &[Value::Undefined]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected object, got {v:?}"),
        };
        assert!(
            oa::get_own(&heap, undef, Wtf8::from_str("message")).is_none(),
            "Error(undefined) must not have an own message property"
        );
    }

    #[test]
    fn error_prototype_to_string_spec() {
        fn eval_str(src: &str) -> String {
            let mut vm = crate::Vm::new();
            match vm.eval(src) {
                Ok(Value::Str(g)) => g.as_wtf8().to_string(),
                Ok(v) => panic!("expected string for {src:?}, got {v:?}"),
                Err(e) => panic!("eval failed for {src:?}: {e:?}"),
            }
        }
        // Spec 20.5.3.4: `name: message`.
        assert_eq!(
            eval_str("String(new TypeError(\"bad\"));"),
            "TypeError: bad"
        );
        assert_eq!(eval_str("new Error(\"boom\").toString();"), "Error: boom");
        // Subtypes inherit it through the prototype chain.
        assert_eq!(
            eval_str("String(new RangeError(\"nope\"));"),
            "RangeError: nope"
        );
        // Missing `message` renders as just the name; missing `name`
        // defaults to `"Error"`.
        assert_eq!(eval_str("new Error().toString();"), "Error");
    }

    #[test]
    fn function_call_and_apply() {
        fn add(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
            let total: f64 = args
                .iter()
                .map(|&a| oa::to_number(a))
                .collect::<Result<Vec<_>, _>>()?
                .iter()
                .sum();
            Ok(oa::num(total))
        }
        let (mut heap, global) = setup();
        let add_obj = sn::new_native_function(&mut heap, "add", add, 2).unwrap();
        let add_fn = Value::Object(add_obj);
        heap.set_prototype(add_obj, Value::Object(heap.function_prototype.unwrap()));
        let (function_ctor_obj, _) = global_native(&heap, global, "Function");
        let proto = match heap
            .get_property(function_ctor_obj, Wtf8::from_str("prototype"))
            .unwrap()
            .value
        {
            Value::Object(o) => o,
            v => panic!("Function.prototype not an object: {v:?}"),
        };
        assert_eq!(proto, heap.function_prototype.unwrap());

        let call = method_native(&heap, proto, "call");
        let r = call(
            &mut heap,
            add_fn,
            &[Value::Int(0), Value::Int(2), Value::Int(3)],
        )
        .unwrap();
        assert_eq!(as_num(r), 5.0);

        let apply = method_native(&heap, proto, "apply");
        let arg_array = oa::new_array(&mut heap).unwrap();
        oa::set_idx(&mut heap, arg_array, 0, Value::Int(4)).unwrap();
        oa::set_idx(&mut heap, arg_array, 1, Value::Int(6)).unwrap();
        let r = apply(
            &mut heap,
            add_fn,
            &[Value::Int(0), Value::Object(arg_array)],
        )
        .unwrap();
        assert_eq!(as_num(r), 10.0);

        // Non-callable receivers throw.
        assert!(matches!(
            call(&mut heap, Value::Int(1), &[]),
            Err(NativeError::TypeError(_))
        ));
    }

    #[test]
    fn function_bind_shape_and_documented_gap() {
        fn target(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
            Ok(oa::get_arg(args, 0))
        }
        let (mut heap, global) = setup();
        let target_fn =
            Value::Object(sn::new_native_function(&mut heap, "target", target, 2).unwrap());
        let (function_ctor_obj, _) = global_native(&heap, global, "Function");
        let proto = match heap
            .get_property(function_ctor_obj, Wtf8::from_str("prototype"))
            .unwrap()
            .value
        {
            Value::Object(o) => o,
            v => panic!("Function.prototype not an object: {v:?}"),
        };
        let bind = method_native(&heap, proto, "bind");
        let bound = match bind(&mut heap, target_fn, &[Value::Int(9), Value::Int(1)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected object, got {v:?}"),
        };
        let name_val = read_bound_name(&mut heap, bound);
        assert_eq!(as_str(&heap, name_val), "bound target");
        // length = max(0, target.length - bound args) = 2 - 1.
        assert_eq!(
            as_num(oa::get_own(&heap, bound, Wtf8::from_str("length")).unwrap()),
            1.0
        );
        assert_eq!(
            heap.get(bound).prototype,
            Value::Object(heap.function_prototype.unwrap())
        );
        // Invocation is the documented v1 gap: NativeFn sees no callee.
        let f = match heap.get(bound).kind {
            ObjectKind::NativeFunction(f) => f,
            _ => panic!("bound is not a native function"),
        };
        assert!(matches!(
            f(&mut heap, Value::Undefined, &[]),
            Err(NativeError::TypeError(_))
        ));
    }

    #[test]
    fn function_constructor_builds_throwing_function() {
        let (mut heap, global) = setup();
        let (_, function_ctor) = global_native(&heap, global, "Function");
        let body = s(&mut heap, "return 1");
        let f = match function_ctor(&mut heap, Value::Undefined, &[body]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected object, got {v:?}"),
        };
        assert_eq!(Value::Object(f).type_of(), "function");
        assert_eq!(
            heap.get(f).prototype,
            Value::Object(heap.function_prototype.unwrap())
        );
        let inner = match heap.get(f).kind {
            ObjectKind::NativeFunction(f) => f,
            _ => panic!("not a native function"),
        };
        assert!(matches!(
            inner(&mut heap, Value::Undefined, &[]),
            Err(NativeError::TypeError(_))
        ));
    }

    #[test]
    fn json_round_trip() {
        let (mut heap, global) = setup();
        let json_obj = match heap
            .get_property(global, Wtf8::from_str("JSON"))
            .unwrap()
            .value
        {
            Value::Object(o) => o,
            v => panic!("JSON not an object: {v:?}"),
        };
        let stringify = method_native(&heap, json_obj, "stringify");
        let parse = method_native(&heap, json_obj, "parse");
        let obj = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.define_property_str(obj, "a", Property::data(Value::Int(1)))
            .unwrap();
        let x = s(&mut heap, "x");
        heap.define_property_str(obj, "b", Property::data(x))
            .unwrap();
        let json = stringify(&mut heap, Value::Undefined, &[Value::Object(obj)]).unwrap();
        assert_eq!(as_str(&heap, json), r#"{"a":1,"b":"x"}"#);
        let back = match parse(&mut heap, Value::Undefined, &[json]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected object, got {v:?}"),
        };
        assert_eq!(
            oa::get_own(&heap, back, Wtf8::from_str("a")).unwrap(),
            Value::Int(1)
        );
        assert_eq!(
            as_str(
                &heap,
                oa::get_own(&heap, back, Wtf8::from_str("b")).unwrap()
            ),
            "x"
        );
    }

    #[test]
    fn math_floor_and_pi() {
        let (mut heap, global) = setup();
        let math = match heap
            .get_property(global, Wtf8::from_str("Math"))
            .unwrap()
            .value
        {
            Value::Object(o) => o,
            v => panic!("Math not an object: {v:?}"),
        };
        let floor = method_native(&heap, math, "floor");
        assert_eq!(
            floor(&mut heap, Value::Undefined, &[oa::num(3.7)]).unwrap(),
            Value::Int(3)
        );
        assert_eq!(
            as_num(oa::get_prop(&heap, math, Wtf8::from_str("PI")).unwrap()),
            std::f64::consts::PI
        );
    }

    #[test]
    fn install_under_gc_pressure_roots_intrinsics() {
        let mut heap = Heap::new();
        // Enable automatic collection BEFORE any builtin allocation: the
        // installers must root `global` and every intrinsic across the
        // allocations that can trigger a collection.
        heap.set_gc_threshold(16);
        let global = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        oa::install(&mut heap, global).unwrap();
        sn::install(&mut heap, global).unwrap();
        install(&mut heap, global).unwrap();
        // `global` is not a heap-intrinsic root (the way a VM would trace
        // it); hold it across the final collection like an embedder would.
        let base = heap.protect(Value::Object(global));
        heap.collect();
        for name in [
            "Object",
            "Array",
            "String",
            "Number",
            "Error",
            "TypeError",
            "Function",
        ] {
            assert!(
                heap.get_property(global, Wtf8::from_str(name)).is_some(),
                "{name} missing after install under GC pressure"
            );
        }
        assert!(heap.error_prototype.is_some());
        assert!(heap.type_error_prototype.is_some());
        assert!(heap.function_prototype.is_some());
        assert!(heap.object_prototype.is_some());
        assert!(heap.array_prototype.is_some());
        heap.release_temp(base);
    }

    #[test]
    fn intrinsics_survive_collection() {
        let (mut heap, global) = setup();
        heap.set_gc_threshold(16);
        // Churn with the intrinsics unrooted by any scope: the Heap fields
        // must keep them alive.
        for _ in 0..200 {
            heap.alloc_str("trash").unwrap();
        }
        heap.collect();
        assert!(heap.error_prototype.is_some());
        assert!(heap.type_error_prototype.is_some());
        assert!(heap.function_prototype.is_some());
        assert!(heap.object_prototype.is_some());
        assert!(heap.array_prototype.is_some());
        // Globals still resolve through the (rooted) global object.
        let _ = global;
    }

    #[test]
    fn error_message_own_property_only_when_provided() {
        // Spec 20.5.8.1: the own `message` property is installed only when
        // the message argument is present and not `undefined`.
        let mut vm = crate::Vm::new();
        for src in [
            "Object.hasOwn(new Error(), 'message') === false;",
            "Object.hasOwn(new Error(undefined), 'message') === false;",
            "Object.hasOwn(new Error('x'), 'message') === true;",
            "new Error('x').message === 'x';",
            // Rendering still treats a missing message as "".
            "String(new Error()) === 'Error';",
            "String(new Error('boom')) === 'Error: boom';",
        ] {
            assert_eq!(vm.eval(src).unwrap(), Value::Bool(true), "{src}");
        }
    }

    #[test]
    fn error_construct_roots_message_across_alloc_object() {
        // Rooting test for the `to_string_js_str` migration in
        // `error_construct`: the message arrives as an unrooted `Str`
        // handle that must survive the `alloc_object` below before it is
        // stored on the error. The message is already a heap string, so
        // conversion allocates nothing and the threshold set here can only
        // fire inside `alloc_object` — the exact gap the temporary root
        // must cover. (A plain `set_gc_threshold(1)` is normally too blunt
        // — it ratchets to twice the live set after firing — but placed
        // immediately before the call with a non-allocating conversion it
        // is precise.)
        let mut heap = Heap::new();
        // Long enough to heap-allocate, deliberately unrooted: stands in
        // for the unrooted handle `to_string_js_str` hands back, the way a
        // `Value` in a native call's `args` slice is invisible to the
        // collector.
        let long = heap.alloc_str(&"m".repeat(64)).unwrap();
        heap.set_gc_threshold(1);
        let err = error_construct(&mut heap, &[Value::Str(long)], Value::Null).unwrap();
        let err = match err {
            Value::Object(o) => o,
            other => panic!("error_construct did not return an object: {other:?}"),
        };
        // Hold the error across the churn below; without the message root
        // the collection inside `alloc_object` frees the message block and
        // the garbage allocs reuse it observably.
        let ebase = heap.protect(Value::Object(err));
        // Same 64-byte size class as the message: a freed message block is
        // observably reused here.
        for _ in 0..200 {
            let _ = heap.alloc_str(&"GARBAGE-".repeat(8)).unwrap();
        }
        let msg = heap
            .get_property(err, Wtf8::from_str("message"))
            .unwrap()
            .value;
        assert!(
            matches!(msg, Value::Str(s) if s.as_wtf8() == Wtf8::from_str(&"m".repeat(64))),
            "message corrupted by collection: {msg:?}"
        );
        heap.release_temp(ebase);
    }

    /// Evaluate JS in a full VM and stringify the completion value.
    fn eval_str(source: &str) -> String {
        let mut vm = crate::Vm::new();
        let v = vm
            .eval(source)
            .unwrap_or_else(|e| panic!("eval failed: {e}"));
        oa::to_string_js(vm.heap_mut(), v)
            .map(|w| w.to_string())
            .unwrap_or_else(|e| panic!("to_string failed: {e:?}"))
    }

    // -- Regression tests: bound functions through call/apply/callbacks ----
    //
    // `Function.prototype.call` / `apply` (and builtins taking callbacks)
    // used to throw "bound function invocation is a v1 gap" because the
    // shared `call_function` choke point never checked `INVOKE_BOUND`.
    // Direct invocation already worked through the VM's own dispatch.

    #[test]
    fn bound_function_call_and_apply() {
        assert_eq!(
            eval_str(
                "var f = function(a, b) { return a + b; }; var g = f.bind(null, 1); g.call(null, 2);"
            ),
            "3"
        );
        assert_eq!(
            eval_str(
                "var f = function(a, b) { return a + b; }; var g = f.bind(null, 1); g.apply(null, [2]);"
            ),
            "3"
        );
        // The bound `this` wins over the call-time receiver.
        assert_eq!(
            eval_str(
                "var f = function(a) { return this.x + a; }; var g = f.bind({x: 10}); g.call({x: 99}, 5);"
            ),
            "15"
        );
        // Direct invocation still works.
        assert_eq!(
            eval_str("var f = function(a, b) { return a * b; }; var g = f.bind(null, 3); g(4);"),
            "12"
        );
    }

    #[test]
    fn bound_function_as_callback() {
        // `arr.map(boundFn)`: the callback goes through `call_function`.
        assert_eq!(
            eval_str("[1, 2, 3].map(function(x) { return x * 2; }.bind(null)).join(',');"),
            "2,4,6"
        );
        // Bound args are prepended ahead of the callback's own args.
        assert_eq!(
            eval_str("[1, 2].map(function(pre, x) { return pre + x; }.bind(null, 10)).join(',');"),
            "11,12"
        );
    }

    #[test]
    fn chained_bind_flattens_and_calls() {
        // Chains flatten at bind time: one hop, concatenated bound args.
        assert_eq!(
            eval_str(
                "var f = function(a, b, c) { return a + b + c; }; var g = f.bind(null, 'x').bind(null, 'y'); g('z');"
            ),
            "xyz"
        );
        assert_eq!(
            eval_str(
                "var f = function(a, b, c) { return a + b + c; }; var g = f.bind(null, 'x').bind(null, 'y'); g.call(null, 'z');"
            ),
            "xyz"
        );
        assert_eq!(
            eval_str(
                "var f = function(a, b, c) { return a + b + c; }; var g = f.bind(null, 'x').bind(null, 'y'); g.apply(null, ['z']);"
            ),
            "xyz"
        );
        // `length` still adjusts per link of the chain.
        assert_eq!(
            eval_str(
                "var f = function(a, b, c) { return 0; }; f.bind(null, 'x').bind(null, 'y').length;"
            ),
            "1"
        );
        // The bound name nests per link.
        assert_eq!(
            eval_str("var f = function foo() {}; f.bind(null).bind(null).name;"),
            "bound bound foo"
        );
    }

    /// Read a bound function's `name` through its lazy accessor, the way
    /// the VM's property read does (invoking the getter with the bound
    /// object as receiver).
    fn read_bound_name(heap: &mut Heap, bound: Gc<Object>) -> Value {
        let acc = heap
            .get_accessor(bound, Wtf8::from_str("name"))
            .expect("bound function should carry a lazy name accessor");
        assert!(!acc.enumerable, "bound name must stay non-enumerable");
        assert!(acc.configurable, "bound name must stay configurable");
        let get = match acc.get {
            Value::Object(o) => match heap.get(o).kind {
                ObjectKind::NativeFunction(f) => f,
                _ => panic!("bound name getter is not a native function"),
            },
            v => panic!("unexpected bound name getter value: {v:?}"),
        };
        get(heap, Value::Object(bound), &[]).expect("bound name getter failed")
    }

    #[test]
    fn bound_name_is_lazy_at_depth() {
        // Regression test for the bound-name OOM: at depth 100k the old
        // eager `format!("bound {target_name}")` retained ~GBs of name
        // bytes across dead-but-uncollected bound objects (GC is opt-in).
        // The lazy representation keeps each link at O(1) name bytes, so
        // a 10k-deep chain must build without materializing any name
        // string, and the final name must still read as
        // `"bound ".repeat(n) + base`.
        fn target(_heap: &mut Heap, _this: Value, _args: &[Value]) -> Result<Value, NativeError> {
            Ok(Value::Undefined)
        }
        let (mut heap, global) = setup();
        let target_fn =
            Value::Object(sn::new_native_function(&mut heap, "target", target, 0).unwrap());
        let (function_ctor_obj, _) = global_native(&heap, global, "Function");
        let proto = match heap
            .get_property(function_ctor_obj, Wtf8::from_str("prototype"))
            .unwrap()
            .value
        {
            Value::Object(o) => o,
            v => panic!("Function.prototype not an object: {v:?}"),
        };
        let bind = method_native(&heap, proto, "bind");
        const DEPTH: usize = 10_000;
        let mut cur = target_fn;
        for _ in 0..DEPTH {
            cur = bind(&mut heap, cur, &[Value::Null]).expect("bind failed");
        }
        let bound = match cur {
            Value::Object(o) => o,
            v => panic!("expected object, got {v:?}"),
        };
        // Laziness: no eager `"name"` data property was materialized on
        // any link, and the prefix count flattened instead of nesting.
        assert!(
            oa::get_own(&heap, bound, Wtf8::from_str("name")).is_none(),
            "bound name must stay lazy until read"
        );
        assert_eq!(
            as_num(oa::get_own(&heap, bound, Wtf8::from_str(BOUND_NAME_COUNT)).unwrap()),
            DEPTH as f64
        );
        assert_eq!(
            as_str(
                &heap,
                oa::get_own(&heap, bound, Wtf8::from_str(BOUND_NAME_BASE)).unwrap()
            ),
            "target"
        );
        // The materialized name is exactly `"bound " * DEPTH + "target"`.
        let expected = "bound ".repeat(DEPTH) + "target";
        let name_val = read_bound_name(&mut heap, bound);
        assert_eq!(as_str(&heap, name_val), expected);
        // First read folds the accessor into the spec-shaped data
        // property: non-writable, non-enumerable, configurable.
        let prop = heap
            .get_property(bound, Wtf8::from_str("name"))
            .expect("name should be cached now");
        assert_eq!(as_str(&heap, prop.value), expected);
        assert!(!prop.writable && !prop.enumerable && prop.configurable);
        assert!(
            heap.get_accessor(bound, Wtf8::from_str("name")).is_none(),
            "accessor should be evicted after caching"
        );
    }

    #[test]
    fn bound_name_edge_cases_via_eval() {
        // Anonymous functions are named "#anon" in this engine; empty base
        // names still accumulate prefixes (see the Function.prototype case
        // below for a truly empty base).
        assert_eq!(
            eval_str("var f = function() {}; var g = f.bind(null).bind(null); g.name;"),
            "bound bound #anon"
        );
        // `Function.prototype.name` is "": binding it yields one prefix.
        assert_eq!(eval_str("Function.prototype.bind(null).name;"), "bound ");
        // A deeper chain built in a loop, read once at the end. (String
        // `.length` is a v1 gap on primitives, so compare the full name.)
        assert_eq!(
            eval_str(
                "var f = function foo() {}; var g = f; \
                 for (var i = 0; i < 500; i++) { g = g.bind(null); } \
                 g.name;"
            ),
            "bound ".repeat(500) + "foo"
        );
        // Reading twice is stable (cached data property path).
        assert_eq!(
            eval_str(
                "var f = function foo() {}; var g = f.bind(null); \
                 var a = g.name; var b = g.name; a + '|' + b;"
            ),
            "bound foo|bound foo"
        );
    }

    #[test]
    fn bound_name_read_keeps_frozen_invariants() {
        // The lazy fold must not install anything on a non-extensible
        // bound function: the old unvalidated define evicted the
        // non-configurable accessor and installed a configurable data
        // property, flipping `Object.isFrozen(b)` true->false from a mere
        // read. Reads stay correct; the accessor is left in place.
        assert_eq!(
            eval_str(
                "var b = function foo(){}.bind(null); \
                 Object.freeze(b); \
                 var n = b.name; \
                 (n === 'bound foo') + '|' + Object.isFrozen(b);"
            ),
            "true|true"
        );
        // Repeated reads stay stable on a frozen bound function.
        assert_eq!(
            eval_str(
                "var b = function foo(){}.bind(null); \
                 Object.freeze(b); \
                 var a = b.name, c = b.name; \
                 (a === 'bound foo' && c === 'bound foo') + '|' + Object.isFrozen(b);"
            ),
            "true|true"
        );
        // Same for Reflect.preventExtensions (non-extensible, not frozen).
        assert_eq!(
            eval_str(
                "var b = function foo(){}.bind(null); \
                 Reflect.preventExtensions(b); \
                 var n = b.name; \
                 (n === 'bound foo') + '|' + Reflect.isExtensible(b);"
            ),
            "true|false"
        );
        // An extensible bound function still folds on first read (the
        // fast path is unchanged).
        assert_eq!(
            eval_str(
                "var b = function foo(){}.bind(null); \
                 var n = b.name; \
                 (n === 'bound foo') + '|' + \
                 (Object.getOwnPropertyDescriptor(b, 'name').value === 'bound foo');"
            ),
            "true|true"
        );
    }

    #[test]
    fn bound_name_read_through_prototype_chain() {
        // `Object.create(b).name` invokes the inherited lazy getter with
        // the derived object as receiver; per spec the read yields the
        // bound function's name instead of throwing.
        assert_eq!(
            eval_str(
                "var b = function foo(){}.bind(null); \
                 Object.create(b).name;"
            ),
            "bound foo"
        );
        // Deeper chains and chained binds resolve to the nearest bound
        // function up the chain.
        assert_eq!(
            eval_str(
                "var b = function foo(){}.bind(null).bind(null); \
                 Object.create(Object.create(b)).name;"
            ),
            "bound bound foo"
        );
        // The fold targets the bound function itself, never the receiver:
        // after an inherited read, `b` carries the folded data property
        // while the derived object still has none of its own.
        assert_eq!(
            eval_str(
                "var b = function foo(){}.bind(null); \
                 var d = Object.create(b); \
                 var n = d.name; \
                 (n === 'bound foo') + '|' + \
                 (Object.getOwnPropertyDescriptor(b, 'name').value === 'bound foo') + '|' + \
                 Object.hasOwn(d, 'name');"
            ),
            "true|true|false"
        );
    }

    #[test]
    fn bound_name_getter_rejects_incompatible_receiver() {
        // A receiver whose prototype chain holds no bound function still
        // throws `TypeError` (e.g. the getter borrowed onto a plain
        // object).
        fn target(_heap: &mut Heap, _this: Value, _args: &[Value]) -> Result<Value, NativeError> {
            Ok(Value::Undefined)
        }
        let (mut heap, global) = setup();
        let target_fn =
            Value::Object(sn::new_native_function(&mut heap, "target", target, 0).unwrap());
        let (function_ctor_obj, _) = global_native(&heap, global, "Function");
        let proto = match heap
            .get_property(function_ctor_obj, Wtf8::from_str("prototype"))
            .unwrap()
            .value
        {
            Value::Object(o) => o,
            v => panic!("Function.prototype not an object: {v:?}"),
        };
        let bind = method_native(&heap, proto, "bind");
        let bound = match bind(&mut heap, target_fn, &[Value::Null]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected object, got {v:?}"),
        };
        let acc = heap
            .get_accessor(bound, Wtf8::from_str("name"))
            .expect("bound function should carry a lazy name accessor");
        let get = match acc.get {
            Value::Object(o) => match heap.get(o).kind {
                ObjectKind::NativeFunction(f) => f,
                _ => panic!("bound name getter is not a native function"),
            },
            v => panic!("unexpected bound name getter value: {v:?}"),
        };
        let plain = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        assert!(matches!(
            get(&mut heap, Value::Object(plain), &[]),
            Err(NativeError::TypeError(_))
        ));
    }

    #[test]
    fn bind_long_target_name_survives_collection_at_every_alloc() {
        // Issue 1 regression, deterministic: `function_bind` used to
        // allocate the lazy base-name `Str` (a heap allocation for names
        // longer than `STR_INLINE_MAX` bytes) and hold it unrooted across
        // the `BOUND_NAME_COUNT` define, whose key interning can collect
        // and sweep it — use-after-free once the dangling handle was
        // installed as `BOUND_NAME_BASE` and read by `bound_name_getter`.
        // The fix orders the COUNT define before the `alloc_str`, so the
        // new string's first collecting call is the BASE define, which
        // roots `prop.value` across interning (see the comment at the fix
        // site).
        //
        // A single low threshold cannot place the collection
        // deterministically — every collection advances an enabled
        // threshold to 2x the live set — so this test sweeps the
        // collection point across every allocation `bind` performs: a
        // fresh heap per step (cold intern table, so key interning really
        // allocates), threshold armed to fire exactly on the k-th
        // allocation. Every position must leave the lazy name slots
        // intact and materialize the exact expected name; the old
        // ordering fails at the COUNT-key interning step, where the
        // unrooted base name was swept from under it.
        fn target(_heap: &mut Heap, _this: Value, _args: &[Value]) -> Result<Value, NativeError> {
            Ok(Value::Undefined)
        }
        // >15 bytes: forces a heap `Str` for the base name (the inline
        // path needs no rooting and never exhibited the bug).
        let long_name = "averylongfunctionname";
        assert!(long_name.len() > crate::value::STR_INLINE_MAX);
        // Generous upper bound on `bind`'s allocation count (~a dozen);
        // steps past the last allocation simply never fire.
        for k in 1..=40usize {
            let (mut heap, global) = setup();
            let target_fn =
                Value::Object(sn::new_native_function(&mut heap, long_name, target, 0).unwrap());
            // In a unit test there is no VM interpreter tracing the call
            // inputs, so root the target the way the embedder contract
            // requires; the args carry no heap values.
            let target_root = heap.protect(target_fn);
            let (function_ctor_obj, _) = global_native(&heap, global, "Function");
            let proto = match heap
                .get_property(function_ctor_obj, Wtf8::from_str("prototype"))
                .unwrap()
                .value
            {
                Value::Object(o) => o,
                v => panic!("Function.prototype not an object: {v:?}"),
            };
            let bind = method_native(&heap, proto, "bind");
            // Arm the threshold to fire exactly on the k-th allocation
            // inside `bind`; nothing allocates between here and the call.
            heap.set_gc_threshold(heap.alloc_count() + k);
            let bound = match bind(&mut heap, target_fn, &[Value::Null]).unwrap() {
                Value::Object(o) => o,
                v => panic!("expected object, got {v:?}"),
            };
            heap.set_gc_threshold(usize::MAX);
            heap.release_temp(target_root);
            let bound_root = heap.protect(Value::Object(bound));
            let expected = format!("bound {long_name}");
            let name_val = read_bound_name(&mut heap, bound);
            assert_eq!(
                as_str(&heap, name_val),
                expected,
                "wrong name with collection at allocation step {k}"
            );
            // The BASE slot itself still holds the live, exact base name.
            assert_eq!(
                as_str(
                    &heap,
                    oa::get_own(&heap, bound, Wtf8::from_str(BOUND_NAME_BASE)).unwrap()
                ),
                long_name,
                "corrupt base slot with collection at allocation step {k}"
            );
            heap.release_temp(bound_root);
        }
    }

    // -- Regression tests: error constructor [[Prototype]] links ------------
    //
    // `install_error` never set the error constructors' `[[Prototype]]`,
    // so `Object.getPrototypeOf(TypeError)` was `%Function.prototype%`
    // (or null) instead of `%Error%` — its prototype had no `name`,
    // breaking testharness.js's `assert_throws_js`, which walks the
    // constructor's proto chain looking for a function named `"Error"`.

    const ERROR_SUBTYPES: [&str; 7] = [
        "EvalError",
        "RangeError",
        "ReferenceError",
        "SyntaxError",
        "TypeError",
        "URIError",
        "SuppressedError",
    ];

    #[test]
    fn error_constructor_proto_chain() {
        // Each NativeError constructor's `[[Prototype]]` is `%Error%`
        // (spec 20.5.7.1); `SuppressedError` is an Error-family
        // constructor too.
        for subtype in ERROR_SUBTYPES {
            assert_eq!(
                eval_str(&format!("Object.getPrototypeOf({subtype}) === Error;")),
                "true",
                "{subtype}'s [[Prototype]] must be %Error%"
            );
            // Walking the constructor's proto chain finds the function
            // named "Error" (the `assert_throws_js` shape).
            assert_eq!(
                eval_str(&format!("Object.getPrototypeOf({subtype}).name;")),
                "Error",
                "{subtype}'s proto-chain walk must find a function named Error"
            );
            // The instance side of the chain is unchanged.
            assert_eq!(
                eval_str(&format!(
                    "Object.getPrototypeOf(new {subtype}('x')) === {subtype}.prototype;"
                )),
                "true",
                "{subtype} instances must still link to {subtype}.prototype"
            );
        }
        // `%Error%.__proto__` and `%Function%.__proto__` are
        // `%Function.prototype%` (spec 20.5.6.1, 20.2.3.1).
        assert_eq!(
            eval_str("Object.getPrototypeOf(Error) === Function.prototype;"),
            "true"
        );
        assert_eq!(
            eval_str("Object.getPrototypeOf(Function) === Function.prototype;"),
            "true"
        );
        // `instanceof` still works across the family.
        assert_eq!(
            eval_str("var e = new TypeError('x'); e instanceof TypeError && e instanceof Error;"),
            "true"
        );
        assert_eq!(
            eval_str("var e = new RangeError('x'); e instanceof RangeError && e instanceof Error;"),
            "true"
        );
    }

    #[test]
    fn error_constructor_proto_chain_heap_level() {
        // Same assertions as above, at the heap level, so they run
        // without the VM's post-install fixup pass (`install` must link
        // these explicitly, not rely on `fixup_native_function_prototypes`).
        let (heap, global) = setup();
        let error_ctor = global_native(&heap, global, "Error").0;
        assert_eq!(
            heap.get(error_ctor).prototype,
            Value::Object(heap.function_prototype.unwrap()),
            "%Error%.__proto__ must be %Function.prototype%"
        );
        for subtype in ERROR_SUBTYPES {
            let ctor = global_native(&heap, global, subtype).0;
            assert_eq!(
                heap.get(ctor).prototype,
                Value::Object(error_ctor),
                "{subtype}.__proto__ must be %Error%"
            );
        }
        let function_ctor = global_native(&heap, global, "Function").0;
        assert_eq!(
            heap.get(function_ctor).prototype,
            Value::Object(heap.function_prototype.unwrap()),
            "%Function%.__proto__ must be %Function.prototype%"
        );
    }
}
