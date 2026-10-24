//! Wave-49 regression tests for species-constructor handling in
//! `builtin_reflect_proxy.rs`.
//!
//! 1. `TypedArray.prototype.subarray` must run `TypedArraySpeciesCreate`
//!    (ECMA-262 22.2.3.27): a non-object `O.constructor` throws
//!    `TypeError` (`SpeciesConstructor` step 4), a non-constructor
//!    `@@species` throws `TypeError`, and the species constructor is
//!    genuinely invoked as `new S(buffer, byteOffset, length)` with its
//!    return value as the result.
//! 2. `TypedArray.prototype.toSorted` / `toReversed` are
//!    change-array-by-copy methods: they use `TypedArrayCreateSameType`,
//!    ignoring species entirely — a hostile `.constructor` must never
//!    throw and is never consulted.
//!
//! These drive the public [`bowser_js::Vm`] API (parse -> compile -> run).

use bowser_js::{Value, Vm};

fn eval_ok(src: &str) -> Value {
    Vm::new()
        .eval(src)
        .unwrap_or_else(|e| panic!("eval failed for {src:?}: {e:?}"))
}

/// Assert that `src` throws a `TypeError` (uncaught).
fn assert_type_error(src: &str) {
    let v = eval_ok(&format!(
        "var caught = null; try {{ {src} }} catch (e) {{ caught = e; }} \
         caught instanceof TypeError;"
    ));
    assert!(
        matches!(v, Value::Bool(true)),
        "expected TypeError for {src:?}, got {v:?}"
    );
}

fn eval_bool(src: &str) -> bool {
    match eval_ok(src) {
        Value::Bool(b) => b,
        v => panic!("expected bool for {src:?}, got {v:?}"),
    }
}

// -- subarray: SpeciesConstructor error typing -------------------------------

#[test]
fn subarray_nonobject_constructor_throws() {
    // test262: subarray/speciesctor-get-ctor-returns-throws.js
    assert_type_error("var ta = new Uint8Array([1,2,3]); ta.constructor = 42; ta.subarray(0, 1);");
    assert_type_error(
        "var ta = new Uint8Array([1,2,3]); \
         Object.defineProperty(ta, 'constructor', { value: 'str' }); ta.subarray();",
    );
}

#[test]
fn subarray_bigint_nonobject_constructor_throws() {
    assert_type_error(
        "var ta = new BigInt64Array([1n,2n]); ta.constructor = 42; ta.subarray(0, 1);",
    );
    assert_type_error(
        "var ta = new BigUint64Array([1n,2n]); ta.constructor = null; ta.subarray();",
    );
}

#[test]
fn subarray_nonconstructor_species_throws() {
    // test262: subarray/speciesctor-get-species-returns-throws.js
    assert_type_error(
        "var ta = new Uint8Array([1,2,3]); \
         ta.constructor = { [Symbol.species]: 42 }; ta.subarray(0, 1);",
    );
    assert_type_error(
        "var ta = new Uint8Array([1,2,3]); \
         ta.constructor = { [Symbol.species]: {} }; ta.subarray();",
    );
}

#[test]
fn subarray_throwing_constructor_getter_propagates() {
    let v = eval_ok(
        "var ta = new Uint8Array([1,2,3]); var boom = new Error('boom'); \
         Object.defineProperty(ta, 'constructor', { get: function() { throw boom; } }); \
         var caught = null; try { ta.subarray(0, 1); } catch (e) { caught = e; } \
         caught === boom;",
    );
    assert!(
        matches!(v, Value::Bool(true)),
        "getter throw must propagate, got {v:?}"
    );
}

#[test]
fn subarray_null_species_uses_default() {
    assert!(eval_bool(
        "var ta = new Uint8Array([1,2,3,4]); \
         ta.constructor = { [Symbol.species]: null }; \
         var s = ta.subarray(1, 3); \
         s.length === 2 && s[0] === 2 && s.buffer === ta.buffer && \
         Object.getPrototypeOf(s) === Uint8Array.prototype;"
    ));
}

// -- subarray: the species constructor is genuinely used ---------------------

#[test]
fn subarray_species_ctor_receives_buffer_offset_length() {
    assert!(eval_bool(
        "var ta = new Uint8Array([1,2,3,4]); var seen = null; \
         ta.constructor = { [Symbol.species]: function(b, o, l) { \
           seen = [b === ta.buffer, o, l]; return new Uint8Array(b, o, l); } }; \
         var s = ta.subarray(1, 3); \
         seen[0] === true && seen[1] === 1 && seen[2] === 2 && \
         s.length === 2 && s[0] === 2 && s.buffer === ta.buffer;"
    ));
}

#[test]
fn subarray_species_return_value_is_result() {
    // A species constructor returning a view on another buffer: the
    // returned object is used as-is.
    assert!(eval_bool(
        "var ta = new Uint8Array([1,2,3,4]); \
         var other = new Uint8Array([9,9,9]); \
         ta.constructor = { [Symbol.species]: function() { return new Uint8Array(other.buffer); } }; \
         var s = ta.subarray(1, 3); \
         s.buffer === other.buffer && s.length === 3 && s[0] === 9;"
    ));
}

#[test]
fn subarray_subclass_instanceof_and_shares() {
    assert!(eval_bool(
        "class MyTA extends Uint8Array {} \
         var ta = new MyTA([1,2,3,4]); var s = ta.subarray(1, 3); \
         s instanceof MyTA && s.buffer === ta.buffer && s.length === 2 && s[1] === 3;"
    ));
}

#[test]
fn subarray_hostile_species_plain_object_throws() {
    assert_type_error(
        "var ta = new Uint8Array([1,2,3]); \
         ta.constructor = { [Symbol.species]: function() { return { length: 2 }; } }; \
         ta.subarray(0, 1);",
    );
}

#[test]
fn subarray_species_returning_detached_view_throws() {
    assert_type_error(
        "var ta = new Uint8Array([1,2,3,4]); \
         ta.constructor = { [Symbol.species]: function() { \
           var b = new ArrayBuffer(8); var v = new Uint8Array(b); \
           b.transfer(); return v; } }; \
         ta.subarray(0, 2);",
    );
}

// -- subarray: normal behavior unchanged -------------------------------------

#[test]
fn subarray_normal_values_and_clamping() {
    assert!(eval_bool(
        "var ta = new Uint8Array([1,2,3,4,5]); \
         var a = ta.subarray(1, 3); var b = ta.subarray(-3); var c = ta.subarray(3, 1); \
         var d = ta.subarray(0, 99); \
         a.length === 2 && a[0] === 2 && a[1] === 3 && \
         b.length === 3 && b[0] === 3 && c.length === 0 && d.length === 5 && \
         a.buffer === ta.buffer;"
    ));
    // Writes through the sub-view hit the same buffer.
    assert!(eval_bool(
        "var ta = new Uint8Array([1,2,3,4]); var s = ta.subarray(1, 3); \
         s[0] = 99; ta[1] === 99;"
    ));
}

#[test]
fn subarray_detached_buffer_throws_type_error() {
    // test262: subarray/detached-buffer.js — the begin/end conversions
    // still run first (a throwing valueOf propagates before the
    // detached TypeError).
    assert_type_error(
        "var ta = new Uint8Array([1,2,3,4]); ta.buffer.transfer(); ta.subarray(0, 2);",
    );
    let v = eval_ok(
        "var ta = new Uint8Array([1,2,3,4]); ta.buffer.transfer(); \
         var evil = { valueOf: function() { throw new RangeError('boom'); } }; \
         var caught = null; try { ta.subarray(evil, 2); } catch (e) { caught = e; } \
         caught instanceof RangeError;",
    );
    assert!(
        matches!(v, Value::Bool(true)),
        "conversion throw must win, got {v:?}"
    );
}

#[test]
fn subarray_detached_custom_species_ignoring_args_no_throw() {
    // A custom species that ignores its arguments never observes the
    // detached buffer: no TypeError (matches engines).
    assert!(eval_bool(
        "var ta = new Uint8Array([1,2,3,4]); ta.buffer.transfer(); \
         ta.constructor = { [Symbol.species]: function() { return new Uint8Array(4); } }; \
         var s = ta.subarray(0, 2); s instanceof Uint8Array && s.length === 4;"
    ));
}

#[test]
fn subarray_length_tracking_result() {
    assert!(eval_bool(
        "var rab = new ArrayBuffer(16, { maxByteLength: 32 }); \
         var ta = new Uint8Array(rab); \
         var s = ta.subarray(4); \
         s.length === 12 && (rab.resize(32), s.length === 28) && (rab.resize(8), s.length === 4);",
    ));
    // Explicit end: fixed-length result.
    assert!(eval_bool(
        "var rab = new ArrayBuffer(16, { maxByteLength: 32 }); \
         var ta = new Uint8Array(rab); var s = ta.subarray(4, 8); \
         rab.resize(32); s.length === 4;",
    ));
}

// -- toSorted / toReversed ignore species ------------------------------------

#[test]
fn to_sorted_ignores_species() {
    // test262: toSorted/ignores-species.js semantics — hostile
    // constructors never throw and are never consulted.
    assert!(eval_bool(
        "var ta = new Uint8Array([3,1,2]); ta.constructor = 42; \
         var r = ta.toSorted(); \
         r.length === 3 && r[0] === 1 && r[1] === 2 && r[2] === 3 && \
         Object.getPrototypeOf(r) === Uint8Array.prototype && ta[0] === 3;"
    ));
    assert!(eval_bool(
        "var ta = new Uint8Array([3,1,2]); ta.constructor = {}; \
         var r = ta.toSorted(); r[0] === 1 && r[2] === 3;",
    ));
    let v = eval_ok(
        "var ta = new Uint8Array([3,1,2]); var boom = new Error('boom'); \
         Object.defineProperty(ta, 'constructor', { get: function() { throw boom; } }); \
         var r = ta.toSorted(); r[0] === 1 && r[1] === 2 && r[2] === 3;",
    );
    assert!(
        matches!(v, Value::Bool(true)),
        "throwing ctor getter must be ignored, got {v:?}"
    );
}

#[test]
fn to_reversed_ignores_species() {
    // test262: toReversed/ignores-species.js semantics.
    assert!(eval_bool(
        "var ta = new Uint8Array([1,2,3]); ta.constructor = 42; \
         var r = ta.toReversed(); \
         r.length === 3 && r[0] === 3 && r[1] === 2 && r[2] === 1 && \
         Object.getPrototypeOf(r) === Uint8Array.prototype && ta[0] === 1;"
    ));
    let v = eval_ok(
        "var ta = new Uint8Array([1,2,3]); var boom = new Error('boom'); \
         Object.defineProperty(ta, 'constructor', { get: function() { throw boom; } }); \
         var r = ta.toReversed(); r[0] === 3 && r[2] === 1;",
    );
    assert!(
        matches!(v, Value::Bool(true)),
        "throwing ctor getter must be ignored, got {v:?}"
    );
}

#[test]
fn to_sorted_to_reversed_normal_behavior() {
    assert!(eval_bool(
        "var ta = new Uint8Array([3,1,2]); var s = ta.toSorted(); \
         s.buffer !== ta.buffer && s[0] === 1 && ta[0] === 3;",
    ));
    assert!(eval_bool(
        "var ta = new BigInt64Array([3n,1n,2n]); var s = ta.toSorted(); \
         s[0] === 1n && s[2] === 3n && Object.getPrototypeOf(s) === BigInt64Array.prototype;",
    ));
    assert!(eval_bool(
        "var ta = new Uint8Array([1,2,3]); var r = ta.toReversed(); \
         r.buffer !== ta.buffer && r[0] === 3 && ta[2] === 3;",
    ));
    // Custom comparefn still honored by toSorted.
    assert!(eval_bool(
        "var ta = new Uint8Array([1,2,3]); var s = ta.toSorted(function(x, y) { return y - x; }); \
         s[0] === 3 && s[2] === 1;",
    ));
}
