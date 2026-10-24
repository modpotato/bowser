//! Wave-46 regression tests for the `iter_index` fixes in `vm.rs`.
//!
//! 1. Array destructuring / `for`-`of` must invoke accessor (getter)
//!    properties at indices: the spec Array iterator performs `Get()`.
//! 2. A non-default `@@iterator` on an array (own or inherited) must take
//!    the iterator-protocol path instead of the index fast path.
//! 3. String iteration must decode full code points, not lone UTF-16
//!    surrogates.
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

// -- 1. accessor elements: spec Get() ----------------------------------------

#[test]
fn destructuring_invokes_index_getter() {
    let x = eval_int(
        "var a = []; \
         Object.defineProperty(a, '0', { get: () => 42 }); \
         var [x] = a; x;",
    );
    assert_eq!(x, 42);
}

#[test]
fn destructuring_invokes_inherited_index_getter() {
    // `new Array(1)` has length 1 with a real hole at 0, so the
    // prototype getter is consulted (an empty array visits no indices
    // at all; note `[,]` is *not* a hole in this engine — separate
    // pre-existing literal quirk, out of scope here).
    let x = eval_int(
        "var a = new Array(1); \
         Object.defineProperty(Array.prototype, '0', { get: () => 7, configurable: true }); \
         var [x] = a; \
         delete Array.prototype['0']; x;",
    );
    assert_eq!(x, 7);
}

#[test]
fn for_of_invokes_index_getter() {
    let r = eval_int(
        "var a = []; \
         Object.defineProperty(a, '0', { get: () => 7 }); \
         Object.defineProperty(a, '1', { get: () => 8 }); \
         var r = 0; for (const v of a) { r = r * 10 + v; } r;",
    );
    assert_eq!(r, 78);
}

#[test]
fn destructuring_getter_receives_array_as_receiver() {
    let ok = eval_bool(
        "var a = []; \
         Object.defineProperty(a, '0', { get: function () { return this === a; } }); \
         var [x] = a; x === true;",
    );
    assert!(ok);
}

#[test]
fn plain_array_fast_path_unchanged() {
    let s = eval_int("var [m, n] = [1, 2]; m + n;");
    assert_eq!(s, 3);
    let hole = eval_bool("var [h] = [,]; h === undefined;");
    assert!(hole);
    let beyond = eval_bool("var [b] = [1]; b === 1;");
    assert!(beyond);
}

// -- 2. custom @@iterator -----------------------------------------------------

#[test]
fn destructuring_uses_custom_array_iterator() {
    let y = eval_int(
        "var b = [1, 2, 3]; \
         b[Symbol.iterator] = function* () { yield 9; }; \
         var [y] = b; y;",
    );
    assert_eq!(y, 9);
}

#[test]
fn destructuring_uses_inherited_custom_iterator() {
    let y = eval_int(
        "Array.prototype[Symbol.iterator] = function* () { yield 5; }; \
         var [y] = [1, 2]; \
         delete Array.prototype[Symbol.iterator]; y;",
    );
    assert_eq!(y, 5);
}

#[test]
fn for_of_uses_custom_array_iterator() {
    let r = eval_int(
        "var b = [1]; \
         b[Symbol.iterator] = function* () { yield 9; yield 10; }; \
         var r = 0; for (const v of b) { r = r + v; } r;",
    );
    assert_eq!(r, 19);
}

#[test]
fn plain_array_without_custom_iterator_still_fast() {
    let r = eval_int("var r = 0; for (const v of [1, 2, 3]) { r += v; } r;");
    assert_eq!(r, 6);
}

// -- 3. astral code points ----------------------------------------------------

#[test]
fn destructuring_decodes_astral_code_point() {
    let ok = eval_bool("var [z] = '\\uD83D\\uDE00'; z === '\\uD83D\\uDE00';");
    assert!(ok);
}

#[test]
fn destructuring_advances_past_surrogate_pair() {
    let ok = eval_bool(
        "var [p, q, r] = 'a\\uD83D\\uDE00b'; \
         p === 'a' && q === '\\uD83D\\uDE00' && r === 'b';",
    );
    assert!(ok);
}

#[test]
fn for_of_decodes_astral_code_points() {
    let ok = eval_bool(
        "var s = ''; var n = 0; \
         for (const c of 'a\\uD83D\\uDE00b') { s += c; n++; } \
         s === 'a\\uD83D\\uDE00b' && n === 3;",
    );
    assert!(ok);
}

#[test]
fn lone_surrogate_still_yields_as_itself() {
    let ok = eval_bool("var [l] = '\\uD83D'; l === '\\uD83D';");
    assert!(ok);
}

#[test]
fn trailing_lone_surrogate_yields_as_itself() {
    let ok = eval_bool("var [p, q] = 'x\\uDE00'; p === 'x' && q === '\\uDE00';");
    assert!(ok);
}
