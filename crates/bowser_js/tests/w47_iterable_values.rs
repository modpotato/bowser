//! Wave-47 G3 regression tests: `iterable_values` (in
//! `builtin_reflect_proxy.rs`) now returns the outer `protect` base, so
//! both callers (`new TA(iterable)` and `%TypedArray%.from`) release the
//! iterator object together with the collected values via a single
//! `heap.release_temp(base)`. Before the fix the returned base was the
//! slot *after* the iterator's protect, leaving the iterator temp-rooted
//! for the rest of the native call (stale slot, no safety impact).
//!
//! A stale temp root is not observable from script, so these tests pin the
//! functional behavior of both call sites plus the error path (which
//! releases to `base` itself).
//!
//! These drive the public [`bowser_js::Vm`] API (parse -> compile -> run).

use bowser_js::{Value, Vm};

fn eval(src: &str) -> Value {
    Vm::new()
        .eval(src)
        .unwrap_or_else(|e| panic!("eval failed for {src:?}: {e:?}"))
}

fn eval_str(src: &str) -> String {
    match eval(src) {
        Value::Str(s) => s.as_wtf8().to_string(),
        v => panic!("expected string for {src:?}, got {v:?}"),
    }
}

/// An iterable object whose `next` returns boxed results, keyed by the
/// real well-known `Symbol.iterator` (what `get_method` looks up).
const MK: &str = r#"
function mkiter(vals) {
    var i = 0;
    return {
        next: function() {
            if (i < vals.length) return {value: vals[i++], done: false};
            return {value: undefined, done: true};
        }
    };
}
function mkobj(vals) {
    var o = {};
    o[Symbol.iterator] = function() { return mkiter(vals); };
    return o;
}
"#;

fn run(src: &str) -> String {
    eval_str(&format!("{MK}\n{src}"))
}

// -- constructor path (`new TA(iterable)`) ----------------------------------

#[test]
fn typed_array_new_from_iterable() {
    assert_eq!(
        run(
            "var a = new Uint8Array(mkobj([1, 2, 300])); a.length + ':' + a[0] + ',' + a[1] + ',' + a[2];"
        ),
        "3:1,2,44"
    );
}

#[test]
fn typed_array_new_from_empty_iterable() {
    assert_eq!(
        run("var a = new Uint8Array(mkobj([])); '' + a.length;"),
        "0"
    );
}

// -- `%TypedArray%.from` path -------------------------------------------------

#[test]
fn typed_array_from_iterable() {
    assert_eq!(
        run(
            "var a = Uint8Array.from(mkobj([4, 5, 6])); a.length + ':' + a[0] + ',' + a[1] + ',' + a[2];"
        ),
        "3:4,5,6"
    );
}

#[test]
fn typed_array_from_iterable_with_mapfn() {
    assert_eq!(
        run(
            "var a = Uint8Array.from(mkobj([1, 2, 3]), function(v, i) { return v * 10 + i; }); a.length + ':' + a[0] + ',' + a[1] + ',' + a[2];"
        ),
        "3:10,21,32"
    );
}

// -- error path (release to `base` on failure) --------------------------------

const THROWING: &str = r#"
var o = {};
o[Symbol.iterator] = function() {
    return { next: function() { throw new Error('boom'); } };
};
"#;

#[test]
fn iterable_next_throw_propagates() {
    let src = format!("{MK}\n{THROWING}\nnew Uint8Array(o);");
    assert!(Vm::new().eval(&src).is_err());
}

#[test]
fn iterable_from_next_throw_propagates() {
    let src = format!("{MK}\n{THROWING}\nUint8Array.from(o);");
    assert!(Vm::new().eval(&src).is_err());
}
