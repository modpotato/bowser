//! Wave-46 regression tests for the `[...spread]` getter skip in
//! `op_arr_append_spread` (`vm.rs`).
//!
//! 1. Spreading an array with an accessor (getter) element must invoke the
//!    getter with the array as receiver — spec `Get()` per element —
//!    instead of yielding `undefined`.
//! 2. A custom `@@iterator` on an array instance (own or inherited) must
//!    route the whole spread through the iterator protocol.
//! 3. The dense-data fast path and the string spread path are unchanged.
//!
//! These drive the public [`bowser_js::Vm`] API (parse -> compile -> run).

use bowser_js::{Value, Vm};

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

fn eval_bool(src: &str) -> bool {
    match eval(src) {
        Value::Bool(b) => b,
        v => panic!("expected bool for {src:?}, got {v:?}"),
    }
}

// -- 1. accessor elements: spread must invoke getters -------------------------

#[test]
fn spread_invokes_index_getter() {
    let x = eval_int(
        "var a = [1, 2]; \
         Object.defineProperty(a, '1', { get: () => 42 }); \
         var [x, y] = [...a]; x * 100 + y;",
    );
    assert_eq!(x, 142);
}

#[test]
fn spread_invokes_getter_with_array_as_receiver() {
    let ok = eval_bool(
        "var a = [0]; \
         Object.defineProperty(a, '0', { get: function () { return this === a; } }); \
         var [x] = [...a]; x === true;",
    );
    assert!(ok);
}

#[test]
fn spread_invokes_inherited_index_getter() {
    // `new Array(1)` has length 1 with a real hole at 0, so the
    // prototype getter is consulted.
    let x = eval_int(
        "var a = new Array(1); \
         Object.defineProperty(Array.prototype, '0', { get: () => 7, configurable: true }); \
         var [x] = [...a]; \
         delete Array.prototype['0']; x;",
    );
    assert_eq!(x, 7);
}

#[test]
fn spread_getter_runs_per_element_and_counts_calls() {
    let n = eval_int(
        "var calls = 0; \
         var a = []; \
         Object.defineProperty(a, '0', { get: () => { calls++; return 1; } }); \
         Object.defineProperty(a, '1', { get: () => { calls++; return 2; } }); \
         a.length = 2; \
         var [p, q] = [...a]; p + q * 10 + calls * 100;",
    );
    assert_eq!(n, 221);
}

// -- 2. custom @@iterator on array instance ------------------------------------

#[test]
fn spread_uses_custom_array_iterator() {
    let y = eval_int(
        "var b = [1, 2, 3]; \
         b[Symbol.iterator] = function* () { yield 9; }; \
         var [y] = [...b]; y;",
    );
    assert_eq!(y, 9);
}

#[test]
fn spread_uses_inherited_custom_iterator() {
    let y = eval_int(
        "Array.prototype[Symbol.iterator] = function* () { yield 5; yield 6; }; \
         var [y, z] = [...[1, 2]]; \
         delete Array.prototype[Symbol.iterator]; y * 10 + z;",
    );
    assert_eq!(y, 56);
}

#[test]
fn spread_custom_iterator_mixed_with_elements() {
    let r = eval_int(
        "var b = [1]; \
         b[Symbol.iterator] = function* () { yield 9; yield 10; }; \
         var r = 0; for (const v of [0, ...b]) { r = r * 100 + v; } r;",
    );
    assert_eq!(r, 910);
}

// -- 3. fast paths unchanged ----------------------------------------------------

#[test]
fn spread_plain_array_fast_path_unchanged() {
    let ok = eval_bool(
        "var [a, b, c] = [...[1, 2, 3]]; \
         a === 1 && b === 2 && c === 3;",
    );
    assert!(ok);
}

#[test]
fn spread_hole_yields_undefined() {
    let ok = eval_bool(
        "var a = new Array(2); \
         var [x, y] = [...a]; x === undefined && y === undefined;",
    );
    assert!(ok);
}

#[test]
fn spread_string_path_unchanged() {
    let ok = eval_bool("var [a, b] = [...'xy']; a === 'x' && b === 'y';");
    assert!(ok);
}

#[test]
fn spread_appends_after_existing_elements() {
    let ok = eval_bool(
        "var [a, b] = [0, ...[1, 2]]; a === 0 && b === 1 && [0, ...[1, 2]].length === 3;",
    );
    assert!(ok);
}
