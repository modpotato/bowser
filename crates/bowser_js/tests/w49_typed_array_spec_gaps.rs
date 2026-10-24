//! Wave-49 regression tests for two typed-array spec gaps in
//! `builtin_reflect_proxy.rs`.
//!
//! 1. `TypedArray.prototype.set(source, offset)` must convert `offset` with
//!    `ToIntegerOrInfinity` (ES2027 §23.2.3.26 step 4), so an object offset's
//!    `valueOf`/`toString` hooks run. The primitive-only `to_index` mapped
//!    objects to `NaN`→`0` and swallowed throwing hooks.
//! 2. `InitializeTypedArrayFromArrayBuffer` (ES2027 §23.2.5.1.3 step 9.1):
//!    when `length` is omitted AND the buffer is fixed-length,
//!    `bufferByteLength mod elementSize ≠ 0` throws `RangeError`. An explicit
//!    `length` never throws here; length-tracking views over resizable
//!    buffers take the step-8 auto path.
//!
//! These drive the public [`bowser_js::Vm`] API (parse -> compile -> run).

use bowser_js::{Value, Vm, VmError};

fn eval(src: &str) -> Value {
    Vm::new()
        .eval(src)
        .unwrap_or_else(|e| panic!("eval failed for {src:?}: {e:?}"))
}

fn eval_int(src: &str) -> i32 {
    match eval(src) {
        Value::Int(i) => i,
        v => panic!("expected int for {src:?}, got {v:?}"),
    }
}

fn assert_range_error(src: &str) {
    match Vm::new().eval(src) {
        Err(VmError::Thrown(s)) if s.contains("RangeError") => {}
        r => panic!("expected RangeError for {src:?}, got {r:?}"),
    }
}

// -- 1. `set` offset runs object hooks ----------------------------------------

#[test]
fn set_offset_object_valueof_selects_index() {
    // `valueOf` returning 1 must write at index 1, not silently at 0.
    let x = eval_int(
        "var ta = new Uint8Array(4); \
         ta.set([9], { valueOf: function() { return 1; } }); \
         ta[0] * 100 + ta[1];",
    );
    assert_eq!(x, 9);
}

#[test]
fn set_offset_throwing_valueof_propagates() {
    let r = Vm::new().eval(
        "var ta = new Uint8Array(4); \
         ta.set([9], { valueOf: function() { throw new Error('boom'); } });",
    );
    assert!(r.is_err(), "expected throw, got {r:?}");
}

#[test]
fn set_offset_primitives_unchanged() {
    // `undefined` -> 0 (missing-arg default).
    let x = eval_int("var ta = new Uint8Array(2); ta.set([7]); ta[0];");
    assert_eq!(x, 7);
    // Numeric offsets still work.
    let x = eval_int("var ta = new Uint8Array(4); ta.set([5], 2); ta[2];");
    assert_eq!(x, 5);
    // Negative offset -> RangeError.
    assert_range_error("var ta = new Uint8Array(2); ta.set([7], -1);");
    // Negative object offset -> RangeError too (hooks ran, result is -1).
    assert_range_error(
        "var ta = new Uint8Array(2); \
         ta.set([7], { valueOf: function() { return -1; } });",
    );
}

// -- 2. constructor omitted-length remainder check ----------------------------

#[test]
fn constructor_unaligned_fixed_buffer_throws_range_error() {
    assert_range_error("new Uint16Array(new ArrayBuffer(3));");
    assert_range_error("new Uint32Array(new ArrayBuffer(6));");
}

#[test]
fn constructor_aligned_fixed_buffer_ok() {
    let x = eval_int("new Uint16Array(new ArrayBuffer(4)).length;");
    assert_eq!(x, 2);
    let x = eval_int("new Uint8Array(new ArrayBuffer(3)).length;");
    assert_eq!(x, 3);
}

#[test]
fn constructor_explicit_length_never_throws_here() {
    // `new Uint16Array(new ArrayBuffer(3), 0, 1)` is legal: the remainder
    // check applies only to the omitted-length branch.
    let x = eval_int("new Uint16Array(new ArrayBuffer(3), 0, 1).length;");
    assert_eq!(x, 1);
}

#[test]
fn constructor_length_tracking_resizable_buffer_unaffected() {
    // Resizable buffer => length-tracking view (step-8 auto path); the
    // remainder check is skipped, no throw, truncated length.
    let x = eval_int(
        "var b = new ArrayBuffer(3, { maxByteLength: 8 }); \
         new Uint16Array(b).length;",
    );
    assert_eq!(x, 1);
}
