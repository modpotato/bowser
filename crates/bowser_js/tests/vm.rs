//! VM integration tests: arithmetic, closures/upvalues, exceptions, TDZ.
//!
//! These drive the public [`bowser_js::Vm`] API (parse -> compile -> run)
//! and assert on the resulting [`bowser_js::Value`].

use bowser_js::{Value, Vm, Wtf8};

fn eval(src: &str) -> Result<Value, String> {
    Vm::new().eval(src).map_err(|e| format!("{e:?}"))
}

fn eval_ok(src: &str) -> Value {
    eval(src).unwrap_or_else(|e| panic!("eval failed for {src:?}: {e}"))
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

// -- arithmetic -----------------------------------------------------------

#[test]
fn arithmetic_precedence() {
    assert!(matches!(eval_ok("1 + 2 * 3;"), Value::Int(7)));
    assert!(matches!(eval_ok("(1 + 2) * 3;"), Value::Int(9)));
    assert!(matches!(eval_ok("10 - 4 / 2;"), Value::Int(8)));
    assert!(matches!(eval_ok("7 % 3;"), Value::Int(1)));
}

#[test]
fn arithmetic_float_and_string_coercion() {
    // Integral results canonicalize to Int; the value is what matters.
    assert!(matches!(eval_ok("1.5 + 1.5;"), Value::Int(3)));
    assert!(matches!(eval_ok("1.5 + 1.6;"), Value::Float(f) if (f - 3.1).abs() < 1e-12));
    // `+` with a string concatenates.
    assert!(matches!(eval_ok("'a' + 'b';"), Value::Str(_)));
    assert!(matches!(eval_ok("1 + '2';"), Value::Str(_)));
}

#[test]
fn unary_and_update() {
    assert!(matches!(eval_ok("var x = 5; x++; x;"), Value::Int(6)));
    assert!(matches!(eval_ok("var x = 5; ++x;"), Value::Int(6)));
    assert!(matches!(eval_ok("-3;"), Value::Int(-3)));
    assert!(matches!(eval_ok("!true;"), Value::Bool(false)));
}

// -- closures / upvalues ---------------------------------------------------

#[test]
fn closure_counter() {
    let v = eval_ok(
        "function make() { var n = 0; return function() { n = n + 1; return n; }; } \
         var c = make(); c(); c(); c();",
    );
    assert!(matches!(v, Value::Int(3)), "unexpected {v:?}");
}

#[test]
fn closure_captures_parameter() {
    let v = eval_ok(
        "function add(a) { return function(b) { return a + b; }; } \
         var add5 = add(5); add5(10);",
    );
    assert!(matches!(v, Value::Int(15)), "unexpected {v:?}");
}

#[test]
fn nested_closures_share_upvalue() {
    let v = eval_ok(
        "function outer() { var x = 1; \
           function mid() { x = x + 1; return function() { return x * 10; }; } \
           var inner = mid(); return inner(); } \
         outer();",
    );
    assert!(matches!(v, Value::Int(20)), "unexpected {v:?}");
}

// -- exceptions ------------------------------------------------------------

#[test]
fn try_catch_catches_throw() {
    let v = eval_ok("var out = 0; try { throw 1; } catch (e) { out = e + 10; } out;");
    assert!(matches!(v, Value::Int(11)), "unexpected {v:?}");
}

#[test]
fn try_catch_binds_error_object() {
    let v = eval_ok("var m = ''; try { throw new Error('boom'); } catch (e) { m = e.message; } m;");
    assert!(matches!(v, Value::Str(_)), "unexpected {v:?}");
}

#[test]
fn try_finally_runs_on_throw() {
    let v = eval_ok(
        "var fin = 0; \
         try { try { throw 9; } finally { fin = fin + 1; } } catch (e) { fin = fin + 100; } \
         fin;",
    );
    assert!(matches!(v, Value::Int(101)), "unexpected {v:?}");
}

#[test]
fn try_finally_runs_on_return() {
    let v = eval_ok(
        "var fin = 0; \
         function f() { try { return 5; } finally { fin = 7; } } \
         var r = f(); r + fin;",
    );
    assert!(matches!(v, Value::Int(12)), "unexpected {v:?}");
}

#[test]
fn uncaught_throw_propagates() {
    let r = eval("throw 'nope';");
    assert!(r.is_err(), "expected throw, got {r:?}");
}

#[test]
fn new_error_constructs() {
    let v = eval_ok("var e = new Error('x'); e instanceof Error;");
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

// -- TDZ --------------------------------------------------------------------

#[test]
fn tdz_read_before_init_throws() {
    let r = eval("x; let x = 1;");
    assert!(r.is_err(), "expected TDZ throw, got {r:?}");
}

#[test]
fn let_init_then_read_ok() {
    let v = eval_ok("let x = 41; x + 1;");
    assert!(matches!(v, Value::Int(42)), "unexpected {v:?}");
}

#[test]
fn const_reassignment_throws() {
    let r = eval("const x = 1; x = 2;");
    assert!(r.is_err(), "expected const throw, got {r:?}");
}

// -- thrown-value fidelity ----------------------------------------------------

#[test]
fn thrown_string_matching_error_text_stays_string() {
    // A user-thrown string that merely looks like a compiler error message
    // must stay a string: only compiler-emitted throws are retagged.
    let v = eval_ok(
        "var caught = null; \
         try { throw \"TypeError: assignment to constant variable\"; } catch (e) { caught = e; } \
         typeof caught === \"string\" && caught === \"TypeError: assignment to constant variable\";",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

#[test]
fn thrown_non_string_values_preserved() {
    let v = eval_ok(
        "var r = []; \
         try { throw 42; } catch (e) { r.push(e === 42); } \
         try { throw null; } catch (e) { r.push(e === null); } \
         try { throw {a: 1}; } catch (e) { r.push(e.a === 1); } \
         r[0] && r[1] && r[2];",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

#[test]
fn compiler_tdz_throw_stays_reference_error() {
    let v = eval_ok(
        "var caught = null; \
         { try { q; } catch (e) { caught = e; } let q; } \
         caught instanceof ReferenceError;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

#[test]
fn compiler_const_throw_stays_type_error() {
    let v = eval_ok(
        "var caught = null; const c = 1; \
         try { c = 2; } catch (e) { caught = e; } \
         caught instanceof TypeError;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

// -- construction -------------------------------------------------------------

#[test]
fn arrow_function_is_not_a_constructor() {
    let r = eval("new (() => {})();");
    assert!(r.is_err(), "expected TypeError, got {r:?}");
}

#[test]
fn generator_and_async_are_not_constructors() {
    let r = eval("new (function*() {})();");
    assert!(r.is_err(), "expected TypeError, got {r:?}");
    let r = eval("new (async function() {})();");
    assert!(r.is_err(), "expected TypeError, got {r:?}");
}

#[test]
fn ordinary_function_still_constructs() {
    let v = eval_ok("function C() { this.x = 7; } var o = new C(); o.x;");
    assert!(matches!(v, Value::Int(7)), "unexpected {v:?}");
}

// -- well-known symbols --------------------------------------------------------

#[test]
fn symbol_to_primitive_hook_is_consulted() {
    // `o[Symbol.toPrimitive] = f` registers the hook the VM's ToPrimitive
    // consults (the `+` operator uses the default hint).
    let v = eval_ok(
        "var o = {}; var seen = []; \
         o[Symbol.toPrimitive] = function(h) { seen.push(h); return 42; }; \
         var r = o + 1; \
         r === 43 && seen.length === 1 && seen[0] === \"default\";",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
    let v = eval_ok(
        "var o = {}; \
         o[Symbol.toPrimitive] = function(h) { return \"S!\"; }; \
         (\"\" + o) === \"S!\";",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

#[test]
fn symbol_iterator_key_addresses_iterator_protocol() {
    let v = eval_ok(
        "var o = {}; \
         o[Symbol.iterator] = function() { \
           var i = 0; \
           return { next: function() { \
             i = i + 1; \
             return i <= 2 ? {value: i * 10, done: false} : {done: true}; \
           } }; \
         }; \
         var out = []; for (var x of o) { out.push(x); } out[0] + out[1];",
    );
    assert!(matches!(v, Value::Int(30)), "unexpected {v:?}");
}

#[test]
fn non_well_known_symbol_keyed_property() {
    // User-created symbols are valid property keys; a write followed by
    // read-back through the same symbol works.
    let v = eval_ok("var s = Symbol(\"x\"); var o = {}; o[s] = 1; o[s];");
    assert!(matches!(v, Value::Int(1)), "unexpected {v:?}");
}

// -- gc stress: fresh allocations must survive mid-operation collections -----

fn eval_with_gc(src: &str) -> Value {
    let mut vm = Vm::new();
    // Collect on (almost) every allocation: any value the interpreter keeps
    // only in a Rust local across an allocating call is swept.
    vm.heap_mut().set_gc_threshold(1);
    vm.eval(src)
        .unwrap_or_else(|e| panic!("eval failed for {src:?}: {e:?}"))
}

#[test]
fn gc_stress_array_literal_keeps_elements() {
    let v = eval_with_gc("var a = [{x:1}, {x:2}, {x:3}]; a[0].x + a[1].x + a[2].x;");
    assert!(matches!(v, Value::Int(6)), "unexpected {v:?}");
}

#[test]
fn gc_stress_for_in_keys() {
    let v = eval_with_gc("var ks = []; for (var k in {a:1, b:2, c:3}) { ks.push(k); } ks.length;");
    assert!(matches!(v, Value::Int(3)), "unexpected {v:?}");
}

#[test]
fn gc_stress_spread_string() {
    let v = eval_with_gc("var a = [...\"abcd\"]; a[0] + a[1] + a[2] + a[3];");
    assert!(matches!(v, Value::Str(_)), "unexpected {v:?}");
    let v = eval_with_gc("var a = [...\"abcd\"]; a.length;");
    assert!(matches!(v, Value::Int(4)), "unexpected {v:?}");
}

#[test]
fn gc_stress_function_name_survives() {
    let v = eval_with_gc("function hello() { return 1; } hello();");
    assert!(matches!(v, Value::Int(1)), "unexpected {v:?}");
}

/// Install/sweep stress: with a collection on (almost) every allocation,
/// every builtin family installed at `Vm::new` time (intrinsics, hidden
/// builtins, prototypes) must survive repeated sweeps while a mixed
/// workload churns the heap. Any live object swept here is a soundness
/// bug in the install or sweep paths.
#[test]
fn gc_stress_builtin_families_survive_repeated_collections() {
    let mut vm = Vm::new();
    // Collect on (almost) every allocation: any value the interpreter
    // keeps only in a Rust local across an allocating call is swept.
    vm.heap_mut().set_gc_threshold(1);
    let v = vm
        .eval(
            r#"
            var out = [];
            var m = new Map(); m.set("a", 1); m.set("b", 2);
            out.push(m.get("a") + m.get("b"));
            var s = new Set([1, 2, 3]); out.push(s.size);
            var wm = new WeakMap(); var key = {}; wm.set(key, 9);
            out.push(wm.get(key));
            var pv = 0; Promise.resolve(41).then(function (v) { pv = v + 1; });
            out.push(pv);
            out.push(/a+/.test("aaab") ? 1 : 0);
            out.push("hello".toUpperCase());
            class C { #x = 5; get() { return this.#x; } }
            out.push(new C().get());
            function* g() { yield 1; yield 2; }
            var gi = g(); out.push(gi.next().value + gi.next().value);
            out.push(JSON.parse('{"n":7}').n);
            out.push([3, 1, 2].sort()[0]);
            var f = function (x) { return x * 2; };
            for (var i = 0; i < 50; i++) { f = f.bind(null); }
            out.push([21].map(f)[0]);
            out.push((new TypeError("x")).name === "TypeError" ? 1 : 0);
            out.push(new Date(0).getTime());
            var acc = "";
            for (var j = 0; j < 200; j++) { acc += "x"; }
            out.push(acc.slice(0, 3));
            // The promise callback runs as a microtask, after this script
            // returns: re-read `pv` through a stashed closure once the
            // microtasks have drained.
            globalThis.__readOut = function () { out[3] = pv; return out.join(","); };
            "installed";
            "#,
        )
        .unwrap_or_else(|e| panic!("eval failed: {e:?}"));
    assert!(matches!(v, Value::Str(_)), "unexpected {v:?}");
    vm.drain_microtasks();
    let v = vm
        .eval("globalThis.__readOut()")
        .unwrap_or_else(|e| panic!("eval failed: {e:?}"));
    let s = match v {
        Value::Str(s) => s.as_wtf8().to_string(),
        other => panic!("unexpected {other:?}"),
    };
    assert_eq!(
        s, "3,3,9,42,1,HELLO,5,3,7,1,42,1,0,xxx",
        "builtin family mismatch under gc stress"
    );
}

// -- nested re-entry: native -> JS -> native -> JS ---------------------------
//
// Regression tests for the re-entry soundness fix (`with_active` in
// vm.rs). The promise constructor (native) runs its executor (JS), which
// may call back into natives (`Array.prototype.map`) with JS callbacks.
// The old code derived a second `&mut Interp` from the `ACTIVE` raw
// pointer while the first was still live and used it afterward (undefined
// behavior under stacked/tree borrows). These tests pin the exact values
// each layer must produce, so any aliasing or misordering shows up as a
// wrong value, not just a crash.

#[test]
fn promise_executor_nested_native_callback_values() {
    // native (Promise ctor) -> JS (executor) -> native (Array.map) ->
    // JS (callback): the accumulator proves the callback ran for every
    // element with the right argument.
    let v = eval_ok(
        "var total = 0; \
         var p = new Promise(function(resolve) { \
           [1, 2, 3, 4].map(function(x) { total = total + x * x; return 0; }); \
           resolve(total); \
         }); \
         total;",
    );
    assert!(matches!(v, Value::Int(30)), "unexpected {v:?}");
}

#[test]
fn nested_reentry_construction_order() {
    // Three layers deep — native -> JS -> native -> JS -> native -> JS —
    // each appending its marker digit. The exact order proves every
    // layer ran exactly once, innermost last.
    let v = eval_ok(
        "var order = 0; \
         function inner() { \
           return new Promise(function(res2) { order = order * 10 + 3; res2(99); }); \
         } \
         var p = new Promise(function(resolve) { \
           order = order * 10 + 1; \
           [5].map(function(x) { order = order * 10 + 2; inner(); return x; }); \
           resolve(1); \
         }); \
         order;",
    );
    assert!(matches!(v, Value::Int(123)), "unexpected {v:?}");
}

// -- promise combinators --------------------------------------------------
//
// `Promise.all` / `race` / `any` / `allSettled` and
// `Promise.prototype.finally`. Settlements are async (promise jobs), so
// these run the script, drain microtasks, then read the `result` global.

// Runs `src`, drains all promise jobs, then reads the `result` global as
// a Rust string. The clone happens while the `Vm` (and its heap) is still
// alive: a `Value`'s `Gc` handles dangle once the `Vm` drops.
fn eval_str(src: &str) -> String {
    let mut vm = Vm::new();
    vm.eval(src)
        .unwrap_or_else(|e| panic!("eval failed for {src:?}: {e:?}"));
    vm.drain_microtasks();
    // `var` bindings live in a per-script declarative environment; the
    // combinators stash observations on `globalThis`, which `get_global`
    // reads from the shared global object.
    match vm.get_global("result").unwrap_or(Value::Undefined) {
        Value::Str(s) => s.as_wtf8().to_string(),
        Value::Undefined => panic!("expected string result, got undefined"),
        Value::Null => panic!("expected string result, got null"),
        Value::Int(i) => i.to_string(),
        _ => panic!("expected string result, got non-string"),
    }
}

#[test]
fn promise_all_fulfills_in_order() {
    let s = eval_str(
        "Promise.all([Promise.resolve(1), 2, Promise.resolve(3)]) \
           .then(function(v) { globalThis.result = v.join(','); });",
    );
    assert_eq!(s, "1,2,3");
}

#[test]
fn promise_all_empty_resolves_empty() {
    let s = eval_str(
        "Promise.all([]).then(function(v) { globalThis.result = v.length + ':' + (v instanceof Array); });",
    );
    assert_eq!(s, "0:true");
}

#[test]
fn promise_all_rejects_on_first_rejection() {
    let s = eval_str(
        "Promise.all([Promise.resolve(1), Promise.reject('boom')]) \
           .then(function() { globalThis.result = 'fulfilled'; }, \
                 function(e) { globalThis.result = 'rejected:' + e; });",
    );
    assert_eq!(s, "rejected:boom");
}

#[test]
fn promise_all_non_iterable_throws_synchronously() {
    let r = eval("Promise.all(123);");
    assert!(r.is_err(), "expected a synchronous TypeError, got {r:?}");
}

#[test]
fn promise_all_returns_a_promise() {
    let s = eval_str("globalThis.result = typeof Promise.all([]).then;");
    assert_eq!(s, "function");
}

#[test]
fn promise_all_iterates_strings() {
    let s = eval_str("Promise.all('ab').then(function(v) { globalThis.result = v.join(','); });");
    assert_eq!(s, "a,b");
}

#[test]
fn promise_race_first_settled_wins() {
    let s = eval_str(
        "Promise.race([Promise.resolve('a'), Promise.resolve('b')]) \
           .then(function(v) { globalThis.result = v; });",
    );
    assert_eq!(s, "a");
}

#[test]
fn promise_race_rejection_wins() {
    let s = eval_str(
        "Promise.race([Promise.reject('x'), Promise.resolve('y')]) \
           .then(function() { globalThis.result = 'fulfilled'; }, \
                 function(e) { globalThis.result = 'rejected:' + e; });",
    );
    assert_eq!(s, "rejected:x");
}

#[test]
fn promise_race_plain_values() {
    let s = eval_str("Promise.race([1, 2]).then(function(v) { globalThis.result = v; });");
    assert_eq!(s, "1");
}

#[test]
fn promise_any_first_fulfilled_wins() {
    let s = eval_str(
        "Promise.any([Promise.reject('e1'), Promise.resolve('win')]) \
           .then(function(v) { globalThis.result = v; });",
    );
    assert_eq!(s, "win");
}

#[test]
fn promise_any_all_rejected_collects_errors() {
    let s = eval_str(
        "Promise.any([Promise.reject('a'), Promise.reject('b')]) \
           .then(function() { globalThis.result = 'fulfilled'; }, \
                 function(e) { globalThis.result = e.message + '|' + e.errors.join(','); });",
    );
    assert_eq!(s, "All promises were rejected|a,b");
}

#[test]
fn promise_any_empty_rejects() {
    let s = eval_str(
        "Promise.any([]) \
           .then(function() { globalThis.result = 'fulfilled'; }, \
                 function(e) { globalThis.result = e.message + ':' + e.errors.length; });",
    );
    assert_eq!(s, "All promises were rejected:0");
}

#[test]
fn promise_all_settled_records() {
    let s = eval_str(
        "Promise.allSettled([Promise.resolve(1), Promise.reject('e')]) \
           .then(function(rs) { \
             globalThis.result = rs.map(function(r) { \
               return r.status + ':' + (r.status === 'fulfilled' ? r.value : r.reason); \
             }).join(';'); \
           });",
    );
    assert_eq!(s, "fulfilled:1;rejected:e");
}

#[test]
fn promise_all_settled_empty() {
    let s =
        eval_str("Promise.allSettled([]).then(function(rs) { globalThis.result = rs.length; });");
    assert_eq!(s, "0");
}

#[test]
fn promise_finally_runs_and_passes_value_through() {
    let s = eval_str(
        "var seen = 0; \
         Promise.resolve(5).finally(function() { seen = 1; }) \
           .then(function(v) { globalThis.result = seen + ':' + v; });",
    );
    assert_eq!(s, "1:5");
}

#[test]
fn promise_finally_preserves_rejection() {
    let s = eval_str(
        "var seen = 0; \
         Promise.reject('oops').finally(function() { seen = 1; }) \
           .then(function() { globalThis.result = 'fulfilled'; }, \
                 function(e) { globalThis.result = seen + ':' + e; });",
    );
    assert_eq!(s, "1:oops");
}

#[test]
fn promise_finally_non_function_is_passthrough() {
    let s =
        eval_str("Promise.resolve(7).finally(null).then(function(v) { globalThis.result = v; });");
    assert_eq!(s, "7");
}

#[test]
fn promise_finally_waits_for_returned_promise() {
    // The fulfillment value is delayed until the promise returned from
    // `onFinally` settles; the exact order proves the chaining.
    let s = eval_str(
        "var result; var order = ''; \
         Promise.resolve(1).finally(function() { \
           order += 'f'; \
           return Promise.resolve().then(function() { order += 'i'; }); \
         }).then(function(v) { order += 't' + v; globalThis.result = order; });",
    );
    assert_eq!(s, "fit1");
}

#[test]
fn promise_finally_throw_replaces_outcome() {
    let s = eval_str(
        "Promise.resolve(1).finally(function() { throw 'fb'; }) \
           .then(function() { globalThis.result = 'fulfilled'; }, \
                 function(e) { globalThis.result = 'rejected:' + e; });",
    );
    assert_eq!(s, "rejected:fb");
}

// -- optional chaining ------------------------------------------------------

#[test]
fn optional_chain_success_yields_value() {
    // Regression: the success path must not fall through into the shared
    // nullish tail that pushes `undefined`.
    assert!(matches!(eval_ok("var a = {b: 1}; a?.b;"), Value::Int(1)));
    assert!(matches!(
        eval_ok("var a = {b: {c: 2}}; a?.b.c;"),
        Value::Int(2)
    ));
    assert!(matches!(
        eval_ok("var a = {b: function() { return 3; }}; a?.b();"),
        Value::Int(3)
    ));
}

#[test]
fn optional_chain_nullish_yields_undefined() {
    assert!(matches!(eval_ok("var a = null; a?.b;"), Value::Undefined));
    assert!(matches!(
        eval_ok("var a = {b: null}; a?.b?.c;"),
        Value::Undefined
    ));
    assert!(matches!(
        eval_ok("var a = undefined; a?.b();"),
        Value::Undefined
    ));
}

#[test]
fn delete_optional_chain() {
    // `delete a?.b` short-circuits to `true` on a nullish base.
    assert!(matches!(
        eval_ok("var a = null; delete a?.b;"),
        Value::Bool(true)
    ));
    assert!(matches!(
        eval_ok("var a = {b: 1}; delete a?.b; a.b;"),
        Value::Undefined
    ));
    // A `?.` deeper in the base short-circuits the whole delete.
    assert!(matches!(
        eval_ok("var a = null; delete a?.b.c;"),
        Value::Bool(true)
    ));
    assert!(matches!(
        eval_ok("var a = {b: null}; delete a.b?.c;"),
        Value::Bool(true)
    ));
    // But a nullish value reaching a non-optional link throws instead of
    // short-circuiting, like an ordinary property access.
    assert_type_error("var a = {b: null}; delete a?.b.c;");
    assert_type_error("var a = {b: undefined}; delete a?.b.c;");
    // Sanity: a plain nullish delete base throws too.
    assert_type_error("delete null.c;");
}

// -- truthiness -------------------------------------------------------------

#[test]
fn truthiness_empty_string() {
    // Regression: empty string is falsy.
    assert!(matches!(eval_ok("!\"\";"), Value::Bool(true)));
    assert!(matches!(
        eval_ok("var x; if (\"\") { x = 1; } else { x = 2; } x;"),
        Value::Int(2)
    ));
    assert!(matches!(eval_ok("\"\" ? 1 : 2;"), Value::Int(2)));
    assert!(matches!(eval_ok("\"x\" ? 1 : 2;"), Value::Int(1)));
}

// -- Array.prototype higher-order methods -------------------------------------

#[test]
fn array_callback_third_arg() {
    // map/filter (and friends) pass (value, index, array).
    assert!(matches!(
        eval_ok("[10, 20].map(function(v, i, a) { return a[i] + v; }).join(',');"),
        Value::Str(_)
    ));
    // Filter sees the array too.
    assert!(matches!(
        eval_ok("[1, 2, 3].filter(function(v, i, a) { return a[i] === v; }).length;"),
        Value::Int(3)
    ));
}

#[test]
fn array_for_each() {
    assert!(matches!(
        eval_ok("var s = 0; [1, 2, 3].forEach(function(v) { s += v; }); s;"),
        Value::Int(6)
    ));
    assert!(matches!(
        eval_ok("[1, 2, 3].forEach(function(v) { return v; });"),
        Value::Undefined
    ));
}

#[test]
fn array_reduce() {
    assert!(matches!(
        eval_ok("[1, 2, 3, 4].reduce(function(a, b) { return a + b; }, 0);"),
        Value::Int(10)
    ));
    // No initial value: first element seeds.
    assert!(matches!(
        eval_ok("[1, 2, 3].reduce(function(a, b) { return a + b; });"),
        Value::Int(6)
    ));
    // Empty with no initial value throws.
    assert!(eval("[].reduce(function(a, b) { return a + b; });").is_err());
    // Index and array args.
    assert!(matches!(
        eval_ok("[5].reduce(function(a, v, i, arr) { return a + v + i + arr.length; }, 0);"),
        Value::Int(6)
    ));
}

#[test]
fn array_reduce_right() {
    assert!(matches!(
        eval_ok("['a', 'b', 'c'].reduceRight(function(a, b) { return a + b; }, '');"),
        Value::Str(_)
    ));
    assert!(matches!(
        eval_ok("[1, 2, 3].reduceRight(function(a, b) { return a - b; });"),
        Value::Int(0) // 3 - 2 - 1
    ));
}

#[test]
fn array_find_find_index() {
    assert!(matches!(
        eval_ok("[1, 2, 3, 4].find(function(v) { return v > 2; });"),
        Value::Int(3)
    ));
    assert!(matches!(
        eval_ok("[1, 2, 3].find(function(v) { return v > 9; });"),
        Value::Undefined
    ));
    assert!(matches!(
        eval_ok("[1, 2, 3, 4].findIndex(function(v) { return v > 2; });"),
        Value::Int(2)
    ));
    assert!(matches!(
        eval_ok("[1, 2, 3].findIndex(function(v) { return v > 9; });"),
        Value::Int(-1)
    ));
}

#[test]
fn array_some_every() {
    assert!(matches!(
        eval_ok("[1, 2, 3].some(function(v) { return v > 2; });"),
        Value::Bool(true)
    ));
    assert!(matches!(
        eval_ok("[1, 2, 3].some(function(v) { return v > 9; });"),
        Value::Bool(false)
    ));
    assert!(matches!(
        eval_ok("[1, 2, 3].every(function(v) { return v > 0; });"),
        Value::Bool(true)
    ));
    assert!(matches!(
        eval_ok("[1, 2, 3].every(function(v) { return v > 2; });"),
        Value::Bool(false)
    ));
}

// -- empty-string truthiness --------------------------------------------------

#[test]
fn empty_string_truthiness() {
    // `!""` must be `true`; `""` is falsy everywhere `ToBoolean` applies.
    assert!(matches!(eval_ok(r#"!"";"#), Value::Bool(true)));
    assert!(matches!(eval_ok(r#"!"" === true;"#), Value::Bool(true)));
    assert!(matches!(eval_ok(r#"!!"";"#), Value::Bool(false)));
    assert!(matches!(
        eval_ok(r#"var s = ""; s ? 1 : 2;"#),
        Value::Int(2)
    ));
    assert!(matches!(
        eval_ok(r#"var r = "" && 1; r === "";"#),
        Value::Bool(true)
    ));
    assert!(matches!(
        eval_ok(r#"var r = "" || 1; r === 1;"#),
        Value::Bool(true)
    ));
    assert!(matches!(
        eval_ok(r#"var r = 0; if ("") { r = 1; } r;"#),
        Value::Int(0)
    ));
    assert!(matches!(
        eval_ok(r#"var r = 0; while ("") { r = 1; break; } r;"#),
        Value::Int(0)
    ));
    // Non-empty strings stay truthy.
    assert!(matches!(eval_ok(r#"!"x";"#), Value::Bool(false)));
    assert!(matches!(
        eval_ok(r#"var s = "x"; s ? 1 : 2;"#),
        Value::Int(1)
    ));
}

// -- optional chaining (extra shapes) -----------------------------------------

#[test]
fn optional_chain_computed_and_bare_call() {
    // `a?.[k]` on a nullish base short-circuits.
    assert!(matches!(
        eval_ok(r#"var a = null; var k = "b"; a?.[k];"#),
        Value::Undefined
    ));
    assert!(matches!(
        eval_ok(r#"var a = {b: 7}; var k = "b"; a?.[k];"#),
        Value::Int(7)
    ));
    // Bare `f?.()` on a nullish callee short-circuits; otherwise calls
    // with `undefined` as the receiver.
    assert!(matches!(
        eval_ok(r#"var f = null; f?.();"#),
        Value::Undefined
    ));
    assert!(matches!(
        eval_ok(r#"var f = function() { return 42; }; f?.();"#),
        Value::Int(42)
    ));
    // Arguments of a short-circuited call are never evaluated.
    assert!(matches!(
        eval_ok(
            r#"var seen = false; var f = null;
               f?.(seen = true); seen;"#
        ),
        Value::Bool(false)
    ));
    // Long chains: the first nullish `?.` skips everything after it.
    assert!(matches!(
        eval_ok(r#"var a = null; a?.b.c.d.e;"#),
        Value::Undefined
    ));
    assert!(matches!(
        eval_ok(r#"var a = {b: {c: {d: 9}}}; a?.b.c.d;"#),
        Value::Int(9)
    ));
    // A nullish value in the middle of the chain (no `?.` on the later
    // links) does NOT short-circuit: the non-optional link throws.
    assert_type_error(r#"var a = {b: null}; a?.b.c.d;"#);
    assert_type_error(r#"var a = {b: undefined}; a?.b.c;"#);
}

#[test]
fn optional_call_preserves_this() {
    // `a?.b()` invokes `b` with `a` as the receiver, exactly like `a.b()`.
    assert!(matches!(
        eval_ok("var a = { m: function() { return this === a; } }; a?.m();"),
        Value::Bool(true)
    ));
    // The receiver is the object the method was read from, deeper in a chain.
    assert!(matches!(
        eval_ok(
            "var inner = { m: function() { return this === inner; } }; \
             var a = { b: inner }; a?.b.m();"
        ),
        Value::Bool(true)
    ));
    // A nullish method with a non-optional call throws instead of
    // short-circuiting; only `?.()` short-circuits.
    assert_type_error("var a = {}; a?.b();");
    assert_type_error("var a = {b: null}; a?.b();");
    assert!(matches!(eval_ok("var a = {}; a?.b?.();"), Value::Undefined));
    assert!(matches!(
        eval_ok("var a = {b: null}; a?.b?.();"),
        Value::Undefined
    ));
}

#[test]
fn optional_chain_skips_side_effects() {
    // Short-circuiting skips computed-key evaluation and the rest of the chain.
    assert!(matches!(
        eval_ok(
            "var hits = 0; function key() { hits++; return 'x'; } \
             var a = null; var r = a?.[key()].y; \
             (r === undefined) && (hits === 0);"
        ),
        Value::Bool(true)
    ));
    // A `?.` link short-circuits later calls and their arguments are
    // never evaluated.
    assert!(matches!(
        eval_ok(
            "var hits = 0; function arg() { hits++; return 1; } \
             var a = { b: null }; var r = a?.b?.c(arg()); \
             (r === undefined) && (hits === 0);"
        ),
        Value::Bool(true)
    ));
    // Without the second `?.` the nullish mid-chain value throws before
    // the later call's arguments are evaluated.
    assert!(matches!(
        eval_ok(
            "var hits = 0; function arg() { hits++; return 1; } \
             var caught = null; var a = { b: null }; \
             try { a?.b.c(arg()); } catch (e) { caught = e; } \
             (caught instanceof TypeError) && (hits === 0);"
        ),
        Value::Bool(true)
    ));
    // Live chains evaluate every link exactly once, in source order.
    assert!(matches!(
        eval_ok(
            "var order = ''; function k(s) { order += s; return s; } \
             var a = { x: { y: 5 } }; var r = a?.[k('x')][k('y')]; \
             (r === 5) && (order === 'xy');"
        ),
        Value::Bool(true)
    ));
}

#[test]
fn optional_chain_only_optional_links_short_circuit() {
    // Only a link explicitly marked `?.` short-circuits. A nullish value
    // flowing into a later non-optional link throws a TypeError.
    assert_type_error("var a = {b: null}; a?.b.c;");
    assert_type_error("var a = {b: undefined}; a?.b.c;");
    assert_type_error("var a = {b: null}; a?.b.c.d;");
    // Computed access on a nullish mid-chain value throws as well.
    assert_type_error(r#"var a = {b: null}; var k = "c"; a?.b[k];"#);
    assert_type_error("var a = {b: null}; a?.b[0];");
    // The fully-optional spellings still short-circuit to `undefined`.
    assert!(matches!(
        eval_ok("var a = {b: null}; a?.b?.c;"),
        Value::Undefined
    ));
    assert!(matches!(
        eval_ok("var a = {b: null}; a?.b?.c?.d;"),
        Value::Undefined
    ));
    assert!(matches!(
        eval_ok(r#"var a = {b: null}; var k = "c"; a?.b?.[k];"#),
        Value::Undefined
    ));
    // A non-optional call after `?.` throws on a nullish callee...
    assert_type_error("var a = {}; a?.b();");
    assert_type_error("var a = {b: null}; a?.b();");
    // ...while `?.()` short-circuits.
    assert!(matches!(
        eval_ok("var a = {b: null}; a?.b?.();"),
        Value::Undefined
    ));
    // Non-nullish values flow through non-optional links normally.
    assert!(matches!(
        eval_ok("var a = {b: {c: 5}}; a?.b.c;"),
        Value::Int(5)
    ));
    assert!(matches!(
        eval_ok("var a = {b: {c: {d: 9}}}; a?.b.c.d;"),
        Value::Int(9)
    ));
    // Short-circuiting skips side effects in later links, including the
    // keys of non-optional links that are never reached.
    assert!(matches!(
        eval_ok(
            "var hits = 0; function k() { hits++; return 'c'; } \
             var a = null; var r = a?.b[k()]; \
             (r === undefined) && (hits === 0);"
        ),
        Value::Bool(true)
    ));
    // A `?.` before a non-optional link still short-circuits when its own
    // base is nullish, even with more links after it.
    assert!(matches!(
        eval_ok("var a = null; a?.b.c.d;"),
        Value::Undefined
    ));
}

#[test]
fn optional_chain_with_templates() {
    // `?.` inside a template substitution is an ordinary expression.
    assert!(matches!(eval_ok("var a = null; `${a?.b}`;"), Value::Str(_)));
    // A computed key may be a template literal.
    assert!(matches!(
        eval_ok("var a = null; var s = 'x'; a?.[`${s}y`];"),
        Value::Undefined
    ));
}

// -- Array.prototype: sort / reverse / fill / copyWithin ----------------------

#[test]
fn array_sort() {
    // Default order is by UTF-16 code units of `ToString(v)`.
    assert_eq!(
        eval_str(r#"var a = [3, 1, 2]; a.sort(); globalThis.result = a.join(",");"#),
        "1,2,3"
    );
    assert_eq!(
        eval_str(r#"var a = [10, 9, 80]; a.sort(); globalThis.result = a.join(",");"#),
        "10,80,9"
    );
    assert_eq!(
        eval_str(r#"var a = ["b", "a", "c"]; a.sort(); globalThis.result = a.join(",");"#),
        "a,b,c"
    );
    // `undefined` sorts after every defined value; holes move to the end.
    assert_eq!(
        eval_str(r#"var a = [3, undefined, 1]; a.sort(); globalThis.result = a.join(",");"#),
        "1,3,"
    );
    assert_eq!(
        eval_str(r#"var a = [3, 1, 2]; delete a[1]; a.sort(); globalThis.result = a.join(",");"#),
        "2,3,"
    );
    // Comparator form; sorts in place and returns `this`.
    assert_eq!(
        eval_str(
            r#"var a = [3, 1, 2]; a.sort(function(x, y) { return y - x; });
               globalThis.result = a.join(",");"#
        ),
        "3,2,1"
    );
    assert!(matches!(
        eval_ok(r#"var a = [2, 1]; a.sort() === a;"#),
        Value::Bool(true)
    ));
    // Non-callable comparator throws.
    assert!(eval(r#"[1, 2].sort(42);"#).is_err());
}

#[test]
fn array_to_sorted() {
    assert_eq!(
        eval_str(
            r#"var a = [3, 1, 2]; var b = a.toSorted();
               globalThis.result = b.join(",") + "|" + a.join(",");"#
        ),
        "1,2,3|3,1,2"
    );
    assert_eq!(
        eval_str(
            r#"var b = [3, 1, 2].toSorted(function(x, y) { return y - x; });
               globalThis.result = b.join(",");"#
        ),
        "3,2,1"
    );
}

#[test]
fn array_reverse() {
    assert_eq!(
        eval_str(r#"var a = [1, 2, 3]; a.reverse(); globalThis.result = a.join(",");"#),
        "3,2,1"
    );
    // Holes mirror; reversed in place, returns `this`.
    assert_eq!(
        eval_str(r#"var a = [1, , 3]; a.reverse(); globalThis.result = a.join(",");"#),
        "3,,1"
    );
    assert!(matches!(
        eval_ok(r#"var a = [1]; a.reverse() === a;"#),
        Value::Bool(true)
    ));
}

#[test]
fn array_to_reversed() {
    assert_eq!(
        eval_str(
            r#"var a = [1, 2, 3]; var b = a.toReversed();
               globalThis.result = b.join(",") + "|" + a.join(",");"#
        ),
        "3,2,1|1,2,3"
    );
    // Holes are preserved in the copy.
    assert_eq!(
        eval_str(r#"globalThis.result = [1, , 3].toReversed().join(",");"#),
        "3,,1"
    );
}

#[test]
fn array_fill() {
    assert_eq!(
        eval_str(r#"var a = [1, 2, 3, 4]; a.fill(0, 1, 3); globalThis.result = a.join(",");"#),
        "1,0,0,4"
    );
    assert_eq!(
        eval_str(r#"var a = [1, 2, 3]; a.fill(9); globalThis.result = a.join(",");"#),
        "9,9,9"
    );
    assert_eq!(
        eval_str(r#"var a = [1, 2, 3, 4]; a.fill(0, -2); globalThis.result = a.join(",");"#),
        "1,2,0,0"
    );
    assert!(matches!(
        eval_ok(r#"var a = [1]; a.fill(2) === a;"#),
        Value::Bool(true)
    ));
}

#[test]
fn array_copy_within() {
    assert_eq!(
        eval_str(
            r#"var a = [1, 2, 3, 4, 5]; a.copyWithin(0, 3); globalThis.result = a.join(",");"#
        ),
        "4,5,3,4,5"
    );
    assert_eq!(
        eval_str(
            r#"var a = [1, 2, 3, 4, 5]; a.copyWithin(0, 3, 4); globalThis.result = a.join(",");"#
        ),
        "4,2,3,4,5"
    );
    assert_eq!(
        eval_str(
            r#"var a = [1, 2, 3, 4, 5]; a.copyWithin(-2, -3, -1);
               globalThis.result = a.join(",");"#
        ),
        "1,2,3,3,4"
    );
    assert!(matches!(
        eval_ok(r#"var a = [1]; a.copyWithin(0, 0) === a;"#),
        Value::Bool(true)
    ));
}

// -- Array.prototype: flat / flatMap / findLast -------------------------------

#[test]
fn array_flat() {
    assert!(matches!(
        eval_ok(r#"[1, [2, 3], [4, [5]]].flat().length;"#),
        Value::Int(5)
    ));
    assert_eq!(
        eval_str(r#"globalThis.result = [1, [2, 3], [4, [5]]].flat(2).join(",");"#),
        "1,2,3,4,5"
    );
    assert_eq!(
        eval_str(r#"globalThis.result = [1, [2, [3, [4]]]].flat(Infinity).join(",");"#),
        "1,2,3,4"
    );
    // Depth 0: shallow copy, no flattening.
    assert!(matches!(
        eval_ok(r#"[1, [2]].flat(0).length;"#),
        Value::Int(2)
    ));
    // Holes are dropped (`delete` makes a real hole; elisions currently
    // compile to explicit `undefined`).
    assert!(matches!(
        eval_ok(r#"var a = [1, 2, 3]; delete a[1]; a.flat().length;"#),
        Value::Int(2)
    ));
}

#[test]
fn array_flat_map() {
    assert_eq!(
        eval_str(
            r#"globalThis.result = [1, 2, 3].flatMap(function(v) { return [v, v * 2]; }).join(",");"#
        ),
        "1,2,2,4,3,6"
    );
    assert_eq!(
        eval_str(
            r#"globalThis.result = [1, 2].flatMap(function(v) { return v * 10; }).join(",");"#
        ),
        "10,20"
    );
}

#[test]
fn array_find_last() {
    assert!(matches!(
        eval_ok(r#"[1, 2, 3, 4].findLast(function(v) { return v % 2 === 0; });"#),
        Value::Int(4)
    ));
    assert!(matches!(
        eval_ok(r#"[1, 2, 3].findLast(function(v) { return v > 10; });"#),
        Value::Undefined
    ));
}

// -- Array.prototype: keys / values / entries / with --------------------------

#[test]
fn array_keys_values_entries() {
    assert_eq!(
        eval_str(r#"globalThis.result = ["a", "b"].keys().join(",");"#),
        "0,1"
    );
    assert_eq!(
        eval_str(r#"globalThis.result = ["a", "b"].values().join(",");"#),
        "a,b"
    );
    assert_eq!(
        eval_str(
            r#"globalThis.result = ["a", "b"].entries()
                 .map(function(p) { return p[0] + ":" + p[1]; }).join(",");"#
        ),
        "0:a,1:b"
    );
}

#[test]
fn array_with() {
    assert_eq!(
        eval_str(
            r#"var a = [1, 2, 3]; var b = a.with(1, 9);
               globalThis.result = b.join(",") + "|" + a.join(",");"#
        ),
        "1,9,3|1,2,3"
    );
    assert_eq!(
        eval_str(r#"globalThis.result = [1, 2, 3].with(-1, 9).join(",");"#),
        "1,2,9"
    );
    // Out-of-range index throws.
    assert!(eval(r#"[1, 2, 3].with(5, 9);"#).is_err());
    assert!(eval(r#"[1, 2, 3].with(-4, 9);"#).is_err());
}

// -- `this` / new.target / constructor semantics --------------------------------

/// Assert that `src` throws an uncaught `ReferenceError`.
fn assert_reference_error(src: &str) {
    match eval(src) {
        Ok(v) => panic!("expected ReferenceError, got value {v:?}"),
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                msg.contains("ReferenceError"),
                "expected ReferenceError, got {msg}"
            );
        }
    }
}

#[test]
fn sloppy_this_primitive_boxing_preserves_value() {
    // `f.call(42)` in sloppy mode boxes `this` into a Number wrapper whose
    // hidden [[PrimitiveValue]] slot is 42.
    let v = eval_ok("function f() { return this.valueOf(); } f.call(42);");
    assert!(matches!(v, Value::Int(42)), "unexpected {v:?}");
    // `eval_str` keeps the `Vm` alive while cloning the string out: a
    // `Value::Str` handle dangles once the `Vm` drops.
    let st = eval_str("function f() { return this.valueOf(); } globalThis.result = f.call('ab');");
    assert_eq!(st, "ab");
    let v = eval_ok("function f() { return this.valueOf(); } f.call(Symbol('s'));");
    assert!(matches!(v, Value::Symbol(_)), "unexpected {v:?}");
    // `typeof` still sees the wrapper object ...
    let st = eval_str("function f() { return typeof this; } globalThis.result = f.call(42);");
    assert_eq!(st, "object");
    // ... while a strict function sees the raw primitive.
    let st = eval_str(
        "function f() { 'use strict'; return typeof this; } globalThis.result = f.call(42);",
    );
    assert_eq!(st, "number");
}

#[test]
fn derived_constructor_this_tdz() {
    // `this` before `super()` throws ReferenceError.
    let v = eval_ok(
        "class B {} \
         var caught = null; \
         class D extends B { constructor() { try { this.x; } catch (e) { caught = e; } super(); } } \
         new D(); \
         caught instanceof ReferenceError;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
    // An arrow capturing `this` before `super()` throws too.
    let v = eval_ok(
        "class B {} \
         var caught = null; \
         class D extends B { constructor() { var f = () => this; try { f(); } catch (e) { caught = e; } super(); } } \
         new D(); \
         caught instanceof ReferenceError;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
    // Falling off the end without super() throws ReferenceError.
    let v = eval_ok(
        "class B {} \
         var caught = null; \
         class D extends B { constructor() {} } \
         try { new D(); } catch (e) { caught = e; } \
         caught instanceof ReferenceError;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
    // Uncaught variants propagate as ReferenceError.
    assert_reference_error(
        "class B {} class D extends B { constructor() { this.x; super(); } } new D();",
    );
    assert_reference_error("class B {} class D extends B { constructor() {} } new D();");
    // Returning an object without super() is allowed.
    let v = eval_ok(
        "class B {} \
         class D extends B { constructor() { return { ok: 42 }; } } \
         new D().ok;",
    );
    assert!(matches!(v, Value::Int(42)), "unexpected {v:?}");
    // The normal super() path is unaffected.
    let v = eval_ok(
        "class B { constructor() { this.x = 1; } } \
         class D extends B { constructor() { super(); this.y = 2; } } \
         var d = new D(); d.x + d.y;",
    );
    assert!(matches!(v, Value::Int(3)), "unexpected {v:?}");
}

#[test]
fn construct_uses_new_target_prototype() {
    // Reflect.construct with an explicit newTarget: the instance's
    // prototype is newTarget.prototype.
    let v = eval_ok(
        "function C() {} \
         function NT() {} \
         NT.prototype.marker = 9; \
         var o = Reflect.construct(C, [], NT); \
         o.marker;",
    );
    assert!(matches!(v, Value::Int(9)), "unexpected {v:?}");
    let v = eval_ok(
        "function C() {} \
         function NT() {} \
         Object.getPrototypeOf(Reflect.construct(C, [], NT)) === NT.prototype;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
    // newTarget.prototype not an object: fall back to Object.prototype.
    let v = eval_ok(
        "function C() {} \
         function NT() {} \
         NT.prototype = 1; \
         Object.getPrototypeOf(Reflect.construct(C, [], NT)) === Object.prototype;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
    // `new D()` on a derived class: the super() call creates the instance
    // with D.prototype (outer new.target), not B.prototype.
    let st = eval_str(
        "class B {} \
         class D extends B { constructor() { super(); } } \
         D.prototype.tag = 'd'; \
         var d = new D(); \
         globalThis.result = d.tag;",
    );
    assert_eq!(st, "d");
}

#[test]
fn object_literal_method_strictness() {
    // A "use strict" directive inside an object-literal method makes the
    // method strict: `this` is not boxed.
    let st = eval_str(
        "var o = { m() { 'use strict'; return typeof this; } }; globalThis.result = o.m.call(42);",
    );
    assert_eq!(st, "number");
    // Without the directive the method stays sloppy: `this` is boxed.
    let st = eval_str("var o = { m() { return typeof this; } }; globalThis.result = o.m.call(42);");
    assert_eq!(st, "object");
    // Strict semantics: assignment to an undeclared variable throws.
    let v = eval_ok(
        "var o = { m() { 'use strict'; undeclared_xyz = 1; } }; \
         var caught = null; \
         try { o.m(); } catch (e) { caught = e; } \
         caught instanceof ReferenceError;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
    // The sloppy method instead defines a global.
    let st = eval_str(
        "var o = { m() { undeclared_abc = 1; } }; o.m(); globalThis.result = typeof undeclared_abc;",
    );
    assert_eq!(st, "number");
    // Getters and setters inherit strictness too.
    let v = eval_ok(
        "var caught = null; \
         var o = { get x() { 'use strict'; undeclared_g = 1; return 1; } }; \
         try { o.x; } catch (e) { caught = e; } \
         caught instanceof ReferenceError;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
    let v = eval_ok(
        "var caught = null; \
         var o = { set x(v) { 'use strict'; undeclared_s = v; } }; \
         try { o.x = 1; } catch (e) { caught = e; } \
         caught instanceof ReferenceError;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
    // Inherited strictness: a method in a strict script is strict.
    let st = eval_str(
        "\"use strict\"; var o = { m() { return typeof this; } }; globalThis.result = o.m.call(42);",
    );
    assert_eq!(st, "number");
}

#[test]
fn object_literal_method_strictness_inherited() {
    // The strict flag propagates from the enclosing parse context into
    // object-literal method/getter/setter function objects: a method in a
    // strict script throws on assignment to an undeclared variable instead
    // of silently creating a global.
    assert_reference_error("\"use strict\"; var o = { m() { undeclared_ih1 = 1; } }; o.m();");
    assert_reference_error(
        "\"use strict\"; var o = { get g() { undeclared_ih2 = 1; return 1; } }; o.g;",
    );
    assert_reference_error(
        "\"use strict\"; var o = { set s(v) { undeclared_ih3 = v; } }; o.s = 1;",
    );
    // A "use strict" directive at the top of a method body makes the method
    // strict even in a sloppy script.
    assert_reference_error("var o = { m() { \"use strict\"; undeclared_ih4 = 1; } }; o.m();");
    // Strictness propagates to functions nested inside the method.
    assert_reference_error(
        "var o = { m() { \"use strict\"; function f() { undeclared_ih5 = 1; } f(); } }; o.m();",
    );
    assert_reference_error(
        "var o = { m() { \"use strict\"; var f = () => { undeclared_ih6 = 1; }; f(); } }; o.m();",
    );
    // A directive that is not in the prologue does not enable strict mode.
    let st = eval_str(
        "var o = { m() { 0; \"use strict\"; undeclared_ih7 = 1; } }; \
         o.m(); globalThis.result = typeof undeclared_ih7;",
    );
    assert_eq!(st, "number");
}

// -- stack guards --------------------------------------------------------------
// Regression tests for the post-test262 SIGABRT: before the call
// trampoline, every JS call nested a fresh native `Interp::run` frame
// (tens of KiB in debug), so ~150-200 nested calls overflowed the native
// stack and aborted the process on an `'<unknown>'` thread. The
// trampoline (`dispatch_call`) pushes sync call frames onto the dispatch
// loop's frame stack instead, so deep JS recursion no longer consumes
// native stack per call: it runs to `max_frames` (10_000) and then trips
// a catchable stack-overflow resource error. The byte-budget guard
// (`native_stack_exhausted`) still covers paths that recurse natively
// without pushing JS frames.

/// Deep JS recursion must surface a resource error, never abort.
/// (Recursion goes through a `var`-bound function: this engine does not
/// bind a function declaration's own name inside its body.)
#[test]
fn deep_recursion_is_a_clean_stack_error() {
    // Past max_frames: trips the frame cap with a clean resource error.
    let err = eval("var f; f = function (n) { return n <= 0 ? 0 : 1 + f(n - 1); }; f(20000);")
        .expect_err("20000-deep recursion must trip the frame cap");
    assert!(
        err.contains("stack overflow"),
        "expected stack-overflow resource error, got {err}"
    );
    // Depths that used to trip the native-stack guard (500) now run
    // cleanly on the trampoline.
    assert!(matches!(
        eval_ok("var f; f = function (n) { return n <= 0 ? 0 : 1 + f(n - 1); }; f(500);"),
        Value::Int(500)
    ));
    // Shallow recursion is unaffected.
    assert!(matches!(
        eval_ok("var f; f = function (n) { return n <= 0 ? 0 : 1 + f(n - 1); }; f(10);"),
        Value::Int(10)
    ));
}

/// The trampoline keeps a single dispatch loop across deep call chains:
/// return values flow back correctly through thousands of frames, and
/// mutual recursion stays balanced.
#[test]
fn trampolined_deep_call_chain_returns_values() {
    let v = eval_ok(
        "var even; var odd; \
         even = function (n) { return n === 0 ? true : odd(n - 1); }; \
         odd = function (n) { return n === 0 ? false : even(n - 1); }; \
         even(5000);",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
    // Mutual recursion right up against the 10k frame cap: each call
    // pushes exactly one frame on the trampoline, so 9000-deep mutual
    // recursion fits and still unwinds with the right values.
    let v = eval_ok(
        "var even; var odd; \
         even = function (n) { return n === 0 ? true : odd(n - 1); }; \
         odd = function (n) { return n === 0 ? false : even(n - 1); }; \
         even(9000);",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
    // Deep `new` chains: each frame's construction protocol resolves as
    // frames unwind in the one loop.
    let v = eval_ok(
        "function C(n) { this.n = n; } \
         var o = new C(1); \
         for (var i = 2; i <= 2000; i++) { o = new C(i); } \
         o.n;",
    );
    assert!(matches!(v, Value::Int(2000)), "unexpected {v:?}");
}

/// Deep recursion through a bound function: every recursive step hops
/// through the bound-function dispatch (resolved iteratively, no Rust
/// recursion) and then trampolines the target frame. Values must
/// accumulate correctly across thousands of bound-hop + trampoline
/// round trips.
#[test]
fn trampolined_deep_recursion_through_bound_function() {
    let v = eval_ok(
        "var f = function (n) { return n <= 0 ? 0 : 1 + f(n - 1); }.bind(null); \
         f(5000);",
    );
    assert!(matches!(v, Value::Int(5000)), "unexpected {v:?}");
}

/// `new` inside deep recursion: each level pushes a construct frame on
/// the trampoline and resolves the construction protocol as it
/// unwinds. The sum proves every level's instance was built and
/// discarded correctly.
#[test]
fn trampolined_construct_inside_deep_recursion() {
    let v = eval_ok(
        "function C(n) { this.n = n; } \
         var f = function (n) { return n <= 0 ? 0 : (new C(n)).n + f(n - 1); }; \
         f(2000);",
    );
    assert!(matches!(v, Value::Int(2001000)), "unexpected {v:?}");
}

/// An exception thrown 5000 trampolined frames deep must unwind the
/// whole interpreter-frame stack in the one dispatch loop and land in
/// the top-level handler with the thrown value intact — no nested
/// `run()` boundaries to confuse the unwinder.
#[test]
fn trampolined_throw_unwinds_thousands_of_frames() {
    let v = eval_ok(
        "var f = function (n) { if (n <= 0) { throw { tag: 9 }; } return 1 + f(n - 1); }; \
         var caught = null; \
         try { f(5000); } catch (e) { caught = e; } \
         caught.tag;",
    );
    assert!(matches!(v, Value::Int(9)), "unexpected {v:?}");
}

/// `new.target` through three levels of derived constructors: the
/// trampolined construct frames propagate the original `new.target`
/// (`C`) down the `super()` chain into the base constructor's frame.
#[test]
fn trampolined_new_target_through_derived_chain() {
    let v = eval_ok(
        "var seen = []; \
         class A { constructor() { seen.push(new.target === C); } } \
         class B extends A {} \
         class C extends B {} \
         new C(); \
         seen.length === 1 && seen[0];",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

/// Deep `ToPrimitive` recursion through `+` rides the site-call
/// trampoline (Phase B2): each `@@toPrimitive` hook becomes a frame on
/// the one dispatch loop instead of a nested native `run()`.
#[test]
fn trampolined_toprimitive_deep_recursion() {
    let v = eval_ok(
        "var n = 0; \
         function mk(depth) { \
           var o = {}; \
           Object.defineProperty(o, Symbol.toPrimitive, { value: function (h) { \
             n++; \
             if (depth > 0) return (mk(depth - 1) + 0); \
             return 1; \
           } }); \
           return o; \
         } \
         (mk(3000) + 0) + n;",
    );
    assert!(matches!(v, Value::Int(3002)), "unexpected {v:?}");
}

/// Deep `valueOf` recursion through `-` rides the site-call trampoline
/// (Phase B2 remainder: `Sub`/`Mul`/`Div`/`Mod`/`Pow` now stage
/// `ToPrimitive` hook calls like `Add` instead of nesting a native
/// `run()` per hook). Previously the nested `invoke_value` path raised a
/// catchable stack error at ~150 levels.
#[test]
fn trampolined_toprimitive_deep_recursion_sub() {
    let v = eval_ok(
        "var n = 0; \
         var o = { valueOf: function () { n++; if (n < 3000) return o - 0; return n; } }; \
         o - 0;",
    );
    assert!(matches!(v, Value::Int(3000)), "unexpected {v:?}");
}

/// Deep `valueOf` recursion through `*` rides the site-call trampoline.
#[test]
fn trampolined_toprimitive_deep_recursion_mul() {
    let v = eval_ok(
        "var n = 0; \
         var o = { valueOf: function () { n++; if (n < 3000) return o * 1; return n; } }; \
         o * 1;",
    );
    assert!(matches!(v, Value::Int(3000)), "unexpected {v:?}");
}

/// Deep `valueOf` recursion through `/` rides the site-call trampoline.
#[test]
fn trampolined_toprimitive_deep_recursion_div() {
    let v = eval_ok(
        "var n = 0; \
         var o = { valueOf: function () { n++; if (n < 3000) return o / 1; return n; } }; \
         o / 1;",
    );
    assert!(matches!(v, Value::Int(3000)), "unexpected {v:?}");
}

/// Deep `valueOf` recursion through `%` rides the site-call trampoline.
#[test]
fn trampolined_toprimitive_deep_recursion_mod() {
    let v = eval_ok(
        "var n = 0; \
         var o = { valueOf: function () { n++; if (n < 3000) return o % 100000; return n; } }; \
         o % 100000;",
    );
    assert!(matches!(v, Value::Int(3000)), "unexpected {v:?}");
}

/// Deep `valueOf` recursion through `**` rides the site-call trampoline.
#[test]
fn trampolined_toprimitive_deep_recursion_pow() {
    let v = eval_ok(
        "var n = 0; \
         var o = { valueOf: function () { n++; if (n < 3000) return o ** 1; return n; } }; \
         o ** 1;",
    );
    assert!(matches!(v, Value::Int(3000)), "unexpected {v:?}");
}

/// The trampolined non-`Add` arithmetic ops convert with the number
/// hint: `@@toPrimitive` sees `"number"` for `-` (and `"default"` for
/// `+`), and the arithmetic still applies.
#[test]
fn trampolined_toprimitive_arith_number_hint() {
    let v = eval_ok(
        "var seen = []; \
         var o = {}; \
         Object.defineProperty(o, Symbol.toPrimitive, { value: function (h) { seen.push(h); return 7; } }); \
         var r = (o - 0) * 10 + o * 2 + (o + 0); \
         seen.join(',') + '=' + r;",
    );
    assert!(
        matches!(v, Value::Str(s) if s.as_wtf8() == "number,number,default=91"),
        "unexpected {v:?}"
    );
}

/// A throwing `valueOf` staged from `*` unwinds through the trampoline
/// and is catchable with the right message.
#[test]
fn trampolined_toprimitive_arith_throw_caught() {
    let v = eval_ok(
        "var o = { valueOf: function () { throw new Error('boom'); } }; \
         var c = 'none'; \
         try { o * 2; } catch (e) { c = e.message; } \
         c;",
    );
    assert!(
        matches!(v, Value::Str(s) if s.as_wtf8() == "boom"),
        "unexpected {v:?}"
    );
}

/// Chained staging for `-`: both operands need hooks, so the
/// continuation stages the rhs hook after the lhs hook completes.
#[test]
fn trampolined_toprimitive_arith_both_operands_hook() {
    let v = eval_ok(
        "var a = { valueOf: function () { return 6; } }; \
         var b = { valueOf: function () { return 7; } }; \
         (a - b) * 100 + (b - a);",
    );
    assert!(matches!(v, Value::Int(-99)), "unexpected {v:?}");
}

/// A `valueOf` returning an object falls through to `toString` under the
/// number hint on the trampolined path (mirrors the nested path).
#[test]
fn trampolined_toprimitive_arith_method_fallthrough() {
    let v = eval_ok(
        "var o = { valueOf: function () { return {}; }, toString: function () { return '5'; } }; \
         o - 0;",
    );
    assert!(matches!(v, Value::Int(5)), "unexpected {v:?}");
}

/// BigInt arithmetic through a trampolined `*` with a `valueOf` hook.
#[test]
fn trampolined_toprimitive_arith_bigint() {
    let v = eval_ok(
        "var o = { valueOf: function () { return 10n; } }; \
         (o * 2n) === 20n && (o ** 2n) === 100n;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

/// Deep `for`-`of` recursion through a custom iterator's `next()` rides
/// the site-call trampoline (Phase B2).
#[test]
fn trampolined_iterator_next_deep_recursion() {
    let v = eval_ok(
        "var n = 0; \
         function mkIter(depth) { \
           var i = 0; \
           return { \
             next: function () { \
               if (i === 0 && depth > 0) { n++; for (var x of mkIter(depth - 1)) {} } \
               i++; \
               if (i <= 2) return { value: i, done: false }; \
               return { value: undefined, done: true }; \
             }, \
             [Symbol.iterator]: function () { return this; } \
           }; \
         } \
         var total = 0; \
         for (var v of mkIter(3000)) { total += v; } \
         total + n;",
    );
    assert!(matches!(v, Value::Int(3003)), "unexpected {v:?}");
}

/// Deep `yield*` delegation chain (5000 generators): every hop rides the
/// `yield*` trampoline (`yield_star_swap` drives the inner frame on the
/// current dispatch loop instead of nesting `resume_generator` → `run()`
/// per hop). The innermost return value must propagate back up through
/// every level.
#[test]
fn trampolined_yield_star_deep_delegation_chain() {
    let v = eval_ok(
        "function* mk(n) { if (n <= 0) return 12345; return yield* mk(n - 1); } \
         var r = mk(5000).next(); \
         r.done && r.value === 12345;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

/// `yield*` chain with multiple resumes: each `.next()` must descend the
/// delegation to the innermost yielding generator — intermediate frames
/// are suspended past their `YieldStar` and are never re-driven (driving
/// one would run it past its `yield*` as if the delegation had
/// completed). The sum proves every yielded value crossed all 5000
/// levels on each resume.
#[test]
fn trampolined_yield_star_chain_multiple_resumes() {
    let v = eval_ok(
        "function* mk(n) { if (n <= 0) { yield 7; yield 8; } else { yield* mk(n - 1); } } \
         var g = mk(5000); var sum = 0; var r; \
         while (!(r = g.next()).done) { sum += r.value; } \
         sum;",
    );
    assert!(matches!(v, Value::Int(15)), "unexpected {v:?}");
}

/// `new` through the trampoline: the base/derived construction protocol
/// (fresh instance vs. `super()`-installed `this`, `new.target`) is
/// resolved by `do_return` as each construct frame unwinds.
#[test]
fn trampolined_construct_protocol() {
    // Base constructor returning a primitive: the fresh instance wins.
    let v = eval_ok("function C() { this.x = 7; return 42; } var o = new C(); o.x;");
    assert!(matches!(v, Value::Int(7)), "unexpected {v:?}");
    // Base constructor returning an object: the object wins.
    let v = eval_ok("function C() { this.x = 7; return { y: 8 }; } var o = new C(); o.y;");
    assert!(matches!(v, Value::Int(8)), "unexpected {v:?}");
    // new.target is the constructor the `new` named (observed via a
    // global: a constructor's primitive return is discarded in favor of
    // the fresh instance, per the construction protocol).
    let v = eval_ok("function C() { globalThis.ntSeen = new.target; } new C(); ntSeen === C;");
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
    // Derived constructor: super() installs this, fields land on it.
    let v = eval_ok(
        "class B { constructor() { this.tag = 3; } } \
         class D extends B { constructor() { super(); this.own = 4; } } \
         var d = new D(); d.tag + d.own;",
    );
    assert!(matches!(v, Value::Int(7)), "unexpected {v:?}");
    // Derived constructor that never calls super() and returns a
    // primitive: ReferenceError, catchable in the caller.
    let v = eval_ok(
        "class B {} \
         class D extends B { constructor() { return 1; } } \
         var caught = null; \
         try { new D(); } catch (e) { caught = e; } \
         caught instanceof ReferenceError;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

/// Deep `.bind()` chains flatten at bind time (see `function_bind`): a
/// 3000-deep chain is a single bound function, not 3000 nested calls, so
/// it calls through instead of tripping the native-recursion cap.
#[test]
fn deep_bind_chain_flattens_at_bind_time() {
    assert!(matches!(
        eval_ok(
            "var f = function () { return 42; }; \
             for (var i = 0; i < 3000; i++) { f = f.bind(null); } \
             f();"
        ),
        Value::Int(42)
    ));
    // A short chain still calls through.
    assert!(matches!(
        eval_ok(
            "var f = function (a) { return a + 1; }; \
             var g = f.bind(null); var h = g.bind(null); h(41);"
        ),
        Value::Int(42)
    ));
}

/// `==` over a deep `valueOf` chain rides the site-call trampoline
/// (Phase B2 remainder): each hook becomes a frame on the one dispatch
/// loop instead of a nested native `run()`, so the 1100-deep chain that
/// used to trip the native-recursion cap now completes.
///
/// Note: each `valueOf` performs `==` on the next object (rather than
/// returning it directly) because `to_primitive` falls back to `toString`
/// when `valueOf` returns an object, which would break the chain with a
/// `"[object Object]"` string instead of recursing.
#[test]
fn loose_eq_valueof_chain_rides_trampoline() {
    // 1100 objects, each `valueOf` comparing the next with `==`: the
    // nested comparisons stage hook frames instead of nesting native
    // `run()` calls. The alternating true/false resolves to `true`.
    let v = eval_ok(
        "function wrap(next) { return { valueOf: function() { return next == 0; } }; } \
         var x = 0; \
         for (var i = 0; i < 1100; i++) { x = wrap(x); } \
         x == 0;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

/// `Array.prototype.flat(Infinity)` is iterative: a 5k-deep nesting
/// flattens without touching the native stack budget.
#[test]
fn deep_flat_infinity_does_not_overflow() {
    assert!(matches!(
        eval_ok(
            "var a = [7]; for (var i = 0; i < 5000; i++) { a = [a]; } \
             var r = a.flat(Infinity); r.length === 1 && r[0] === 7;"
        ),
        Value::Bool(true)
    ));
}

// -- class heritage / computed keys are strict mode code (spec 10.2.1) ----

#[test]
fn sloppy_class_heritage_is_strict() {
    // The heritage is strict mode code even when the enclosing scope is
    // sloppy: assignment to an undeclared name must throw ReferenceError
    // rather than creating a global.
    assert_reference_error("class C extends (sloppyHeritageProbe = 1, Object) {}");
    // No global leaked as a side effect, and the error was a ReferenceError.
    let v = eval_ok(
        "var isRefErr = false; \
         try { class C extends (sloppyHeritageProbe = 1, Object) {} } \
         catch (e) { isRefErr = e instanceof ReferenceError; } \
         isRefErr && typeof sloppyHeritageProbe === 'undefined';",
    );
    assert!(
        matches!(v, Value::Bool(true)),
        "heritage leaked a global or did not throw ReferenceError: {v:?}"
    );
}

#[test]
fn sloppy_class_computed_method_key_is_strict() {
    assert_reference_error("class C { [sloppyKeyProbe = 2]() {} }");
    // No global leaked as a side effect, and the error was a ReferenceError.
    let v = eval_ok(
        "var isRefErr = false; \
         try { class C { [sloppyKeyProbe = 2]() {} } } \
         catch (e) { isRefErr = e instanceof ReferenceError; } \
         isRefErr && typeof sloppyKeyProbe === 'undefined';",
    );
    assert!(
        matches!(v, Value::Bool(true)),
        "computed key leaked a global or did not throw ReferenceError: {v:?}"
    );
}

#[test]
fn sloppy_class_computed_accessor_key_is_strict() {
    assert_reference_error("class C { get [sloppyAccProbe = 3]() { return 1; } }");
}

#[test]
fn strict_class_heritage_behavior_unchanged() {
    // Heritage still evaluates and links the prototype chain.
    assert!(matches!(
        eval_ok("'use strict'; class B {} class C extends B {} var c = new C(); c instanceof B;"),
        Value::Bool(true)
    ));
    // Undeclared assignment in a strict heritage still throws.
    assert_reference_error("'use strict'; class C extends (strictHeritageProbe = 1, Object) {}");
    // A heritage with side effects evaluates exactly once, in order.
    assert!(matches!(
        eval_ok(
            "var log = []; class B {} \
             class C extends (log.push('h'), B) { m() { return 7; } } \
             log.length === 1 && new C().m();"
        ),
        Value::Int(7)
    ));
}

#[test]
fn class_heritage_keeps_enclosing_this_and_scope() {
    // `this` in the heritage is the enclosing `this` (arrow wrapper
    // preserves it lexically).
    assert!(matches!(
        eval_ok(
            "function f() { return class extends (this.Klass || Object) {}; } \
             new (f())() instanceof Object;"
        ),
        Value::Bool(true)
    ));
    // Outer locals are visible to the heritage through the wrapper.
    assert!(matches!(
        eval_ok(
            "var Base2 = Object; class C extends Base2 {} var c = new C(); c instanceof Object;"
        ),
        Value::Bool(true)
    ));
    // `super` in a heritage inside a static method still resolves.
    assert!(matches!(
        eval_ok(
            "class P { static get base() { return Object; } } \
             class A extends P { static m() { return class extends super.base {}; } } \
             var K = A.m(); new K() instanceof Object;"
        ),
        Value::Bool(true)
    ));
}

// -- class computed field keys are strict mode code (spec 10.2.1) ---------

#[test]
fn sloppy_class_computed_instance_field_key_is_strict() {
    // Instance field keys evaluate in a strict wrapper even when the
    // enclosing scope is sloppy: assignment to an undeclared name must
    // throw ReferenceError rather than creating a global.
    assert_reference_error("class C { [sloppyFieldProbe = 1] = 2 } new C();");
    let v = eval_ok(
        "var isRefErr = false; \
         try { class C { [sloppyFieldProbe = 1] = 2 } new C(); } \
         catch (e) { isRefErr = e instanceof ReferenceError; } \
         isRefErr && typeof sloppyFieldProbe === 'undefined';",
    );
    assert!(
        matches!(v, Value::Bool(true)),
        "instance field key leaked a global or did not throw ReferenceError: {v:?}"
    );
}

#[test]
fn sloppy_class_computed_static_field_key_is_strict() {
    // Same for static field keys (evaluated at class definition time).
    assert_reference_error("class C { static [sloppyStaticFieldProbe = 1] = 2 }");
    let v = eval_ok(
        "var isRefErr = false; \
         try { class C { static [sloppyStaticFieldProbe = 1] = 2 } } \
         catch (e) { isRefErr = e instanceof ReferenceError; } \
         isRefErr && typeof sloppyStaticFieldProbe === 'undefined';",
    );
    assert!(
        matches!(v, Value::Bool(true)),
        "static field key leaked a global or did not throw ReferenceError: {v:?}"
    );
}

#[test]
fn class_computed_field_keys_evaluate_exactly_once() {
    // Computed field keys evaluate exactly once, at class definition
    // time (spec: ClassDefinitionEvaluation / DefineField): constructing
    // more instances must not re-evaluate them.
    assert!(matches!(
        eval_ok(
            "var n = 0; \
             class C { [(n++, 'k')] = 42; } \
             var c1 = new C(); \
             var c2 = new C(); \
             n === 1 && c1['k'] === 42 && c2['k'] === 42;",
        ),
        Value::Bool(true)
    ));
    // Static field keys likewise run once, at class definition time.
    assert!(matches!(
        eval_ok(
            "var m = 0; \
             class C { static [(m++, 's')] = 7; } \
             m === 1 && C['s'] === 7;",
        ),
        Value::Bool(true)
    ));
}

#[test]
fn class_computed_field_keys_keep_this_and_super() {
    // `this` in an instance field key is the *outer* `this` (the key is
    // evaluated in the definition scope, not on the instance): the
    // constructor-assigned `tag` is not visible to the key.
    assert!(matches!(
        eval_ok(
            "var ok = (function () { \
               this.tag = 'k'; \
               class C { constructor() { this.tag = 'WRONG'; } [this.tag] = 42; } \
               var c = new C(); \
               return c['k'] === 42 && c.tag === 'WRONG'; \
             }).call({}); \
             ok;",
        ),
        Value::Bool(true)
    ));
    // `this` in a static field key is likewise the outer `this`, not the
    // class: the class's own `tag` static is not visible to the key.
    assert!(matches!(
        eval_ok(
            "var ok = (function () { \
               this.tag = 's'; \
               class C { static tag = 'WRONG'; static [this.tag] = 7; } \
               return C['s'] === 7 && C.tag === 'WRONG'; \
             }).call({}); \
             ok;",
        ),
        Value::Bool(true)
    ));
    // `super` in a static field key resolves against the base constructor.
    assert!(matches!(
        eval_ok(
            "class B {} B.x = 'sk'; \
             class D extends B { static [B.x] = 5; } \
             D['sk'] === 5;",
        ),
        Value::Bool(true)
    ));
    // `super` in a field *initializer* (allowed by the parser, unlike in
    // keys) still resolves: the initializer region around the edited key
    // code is untouched.
    assert!(matches!(
        eval_ok(
            "class B {} B.prototype.x = 'iv'; \
             class D extends B { ['k'] = super.x; } \
             new D()['k'] === 'iv';",
        ),
        Value::Bool(true)
    ));
}

// -- for-in enumeration ------------------------------------------------------
// These exercise the reworked `intrinsic_enum_keys` (item 1): dense indices,
// holes, and accessors across the prototype chain, with duplicates visited
// once.

/// Evaluate `src` and copy a string result into an owned `String`. The
/// `Vm` must stay alive while the `Str` handle is read: copying after the
/// VM is dropped is use-after-free.
fn eval_str_on(vm: &mut Vm, src: &str) -> String {
    match vm
        .eval(src)
        .unwrap_or_else(|e| panic!("eval failed for {src:?}: {e:?}"))
    {
        Value::Str(s) => s.as_wtf8().to_string(),
        v => panic!("expected string for {src:?}, got {v:?}"),
    }
}

#[test]
fn for_in_sees_dense_indices_holes_and_accessors() {
    let mut vm = Vm::new();
    // Dense indices are visited; a hole punched by `delete` is skipped.
    // (Array-literal elisions materialize as `undefined` elements, not
    // holes, so the hole here comes from `delete`.)
    let keys = eval_str_on(
        &mut vm,
        "var a = [10, 20, 30]; delete a[1]; \
         var keys = []; for (var k in a) { keys.push(k); } keys.join(',');",
    );
    assert_eq!(keys, "0,2");
    // An enumerable accessor (getter) is visited like an ordinary key;
    // a non-enumerable one is not. (Data properties enumerate before
    // accessors in this engine's documented key order.)
    let keys = eval_str_on(
        &mut vm,
        "var o = {}; \
         Object.defineProperty(o, 'acc', \
           { get: function() { return 1; }, enumerable: true, configurable: true }); \
         Object.defineProperty(o, 'hidden', \
           { get: function() { return 2; }, enumerable: false, configurable: true }); \
         o.plain = 3; \
         var keys = []; for (var k in o) { keys.push(k); } keys.join(',');",
    );
    assert_eq!(keys, "plain,acc");
}

#[test]
fn for_in_dedupes_proto_keys() {
    let mut vm = Vm::new();
    // `x` lives on both the object and its prototype: it is visited once
    // (own key first, then the prototype's unshadowed keys).
    let keys = eval_str_on(
        &mut vm,
        "function B() {} B.prototype.x = 1; B.prototype.y = 2; \
         var o = new B(); o.x = 10; \
         var keys = []; for (var k in o) { keys.push(k); } keys.join(',');",
    );
    assert_eq!(keys, "x,y");
}

// -- deep call-chain compilation ----------------------------------------------

#[test]
fn deep_bind_chain_source_nesting_compiles_clean() {
    // 3000 nested `.bind()` calls: the parser builds member/call chains in
    // a loop (no depth-guard trip) and the compiler flattens them
    // iteratively, so this must compile without overflowing the host
    // stack; the bound chain must still call through to `f`.
    let src = format!(
        "function f() {{ return 42; }} var g = f{}; g();",
        ".bind(null)".repeat(3000)
    );
    let v = eval_ok(&src);
    assert!(matches!(v, Value::Int(42)), "unexpected {v:?}");
}

// -- install-path GC rooting ---------------------------------------------------

#[test]
fn install_paths_survive_threshold_1_collections() {
    // Reinstalls are not reachable from the public API, so this hammers the
    // builtins installed by the regexp/date/reflect-proxy/VM paths under
    // constant collection instead: with the threshold at 1, nearly every
    // allocation may collect, and any prototype, constructor, or hidden
    // intrinsic lost (or left unlinked) during installation shows up as a
    // crash or a wrong result.
    let mut vm = Vm::new();
    vm.heap_mut().set_gc_threshold(1);
    // A check that forces a collection first, then evaluates.
    let run = |vm: &mut Vm, src: &str| -> Value {
        vm.heap_mut().collect();
        vm.eval(src)
            .unwrap_or_else(|e| panic!("eval failed for {src:?}: {e:?}"))
    };
    // RegExp: constructor, prototype methods.
    assert!(matches!(
        run(&mut vm, "/ab+c/.test('abbbc')"),
        Value::Bool(true)
    ));
    assert_eq!(
        eval_str_on(&mut vm, "new RegExp('a(b)').exec('ab')[1]"),
        "b"
    );
    // Date: constructor and statics.
    assert!(matches!(
        run(&mut vm, "new Date(2020, 0, 2).getFullYear()"),
        Value::Int(2020)
    ));
    assert!(matches!(run(&mut vm, "Date.now() > 0"), Value::Bool(true)));
    // Reflect / Proxy (the proxy `get` trap is a v1 gap; assert the
    // installed constructor shape instead).
    assert!(matches!(
        run(&mut vm, "Reflect.get({a: 41}, 'a') + 1"),
        Value::Int(42)
    ));
    assert_eq!(eval_str_on(&mut vm, "typeof Proxy"), "function");
    assert_eq!(eval_str_on(&mut vm, "typeof Proxy.revocable"), "function");
    // ArrayBuffer / DataView / typed arrays. (DataView instances do not
    // yet link to `DataView.prototype` — a v1 gap — so assert the
    // installed constructor shapes and the working ArrayBuffer path.)
    assert!(matches!(
        run(&mut vm, "new ArrayBuffer(8).byteLength"),
        Value::Int(8)
    ));
    assert_eq!(eval_str_on(&mut vm, "typeof DataView"), "function");
    assert!(matches!(
        run(&mut vm, "new Int8Array(4).length"),
        Value::Int(4)
    ));
    // VM intrinsics: Promise, generators, hidden `#enum_keys` / spread.
    assert!(matches!(
        run(&mut vm, "Promise.resolve(1) instanceof Promise"),
        Value::Bool(true)
    ));
    assert!(matches!(
        run(
            &mut vm,
            "function* g() { yield 7; } var it = g(); it.next().value;",
        ),
        Value::Int(7)
    ));
    assert!(matches!(
        run(
            &mut vm,
            "var o = {a: 1}; var n = 0; for (var k in o) { n++; } n;",
        ),
        Value::Int(1)
    ));
    assert!(matches!(
        run(&mut vm, "var a = [...[1, 2], 3]; a.length;"),
        Value::Int(3)
    ));
}

// -- deep-chain compile performance -------------------------------------------
// Regression tests: compiling a deep member/call chain must be linear-time.
// `compile_chain` used to call `is_opt_chain` (an O(spine) walk) once per
// flatten iteration, making deep chains quadratic (a 3000-deep chain did
// ~4.5M redundant spine visits; at 200k deep the old code needs tens of
// billions). A single upfront scan keeps it linear, so these assert a
// generous wall-clock bound that a quadratic regression would blow past.

/// 200k-deep non-optional call chain: `id()()()…` where each call returns
/// the function itself, so the VM runs 200k sequential calls without
/// growing its call stack and the result is still `id`. (A named function
/// declaration's own name is not visible inside its body in this engine,
/// so the self-reference goes through `globalThis`.)
#[test]
fn deep_call_chain_compiles_in_linear_time() {
    let depth = 200_000;
    let mut calls = String::from("id");
    for _ in 0..depth {
        calls.push_str("()");
    }
    let src = format!(
        "globalThis.id = function() {{ return globalThis.id; }}; \
         globalThis.result = typeof ({calls});"
    );
    let start = std::time::Instant::now();
    let s = eval_str(&src);
    let elapsed = start.elapsed();
    assert_eq!(s, "function");
    assert!(
        elapsed < std::time::Duration::from_secs(60),
        "200k-deep call chain took {elapsed:?}; expected linear-time compile"
    );
}

/// Deep optional chain routing through `compile_optional_chain`.
/// `x` is an object, so the first `?.f` yields `undefined` and every later
/// link short-circuits; the result is `undefined` either way.
///
/// Depth is 1000, not 200k: each optional link emits a nullish check whose
/// jump offset is an `i16`, so bytecode size (not compile time) caps how
/// deep an optional chain can go — a pre-existing encoding limit, unrelated
/// to this fix. The quadratic-regression coverage lives in
/// `deep_call_chain_compiles_in_linear_time` above.
#[test]
fn deep_optional_chain_compiles_in_linear_time() {
    let depth = 1_000;
    let mut chain = String::from("x");
    for _ in 0..depth {
        chain.push_str("?.f");
    }
    let src = format!("var x = {{}}; globalThis.result = typeof ({chain});");
    let start = std::time::Instant::now();
    let s = eval_str(&src);
    let elapsed = start.elapsed();
    assert_eq!(s, "undefined");
    assert!(
        elapsed < std::time::Duration::from_secs(60),
        "1000-deep optional chain took {elapsed:?}; expected linear-time compile"
    );
}

/// The upfront optionality scan in `compile_chain` must route exactly like
/// the old per-link checks: optional and non-optional chains agree wherever
/// their semantics coincide.
#[test]
fn optional_chain_routing_parity_with_plain_chain() {
    // Defined bases: identical values.
    assert!(matches!(
        eval_ok("var o = {a: {b: 7}}; o.a.b;"),
        Value::Int(7)
    ));
    assert!(matches!(
        eval_ok("var o = {a: {b: 7}}; o?.a?.b;"),
        Value::Int(7)
    ));
    assert!(matches!(
        eval_ok("var o = {a: {b: 7}}; o?.a.b;"),
        Value::Int(7)
    ));
    assert!(matches!(
        eval_ok("var o = {f: function() { return 3; }}; o?.f();"),
        Value::Int(3)
    ));
    // Nullish bases still short-circuit.
    assert!(matches!(
        eval_ok("var n = null; n?.a?.b;"),
        Value::Undefined
    ));
    assert!(matches!(eval_ok("var f = null; f?.();"), Value::Undefined));
    // An optional link inside a `new` callee is not on the chain spine, so
    // it still takes the `New`-arm special case (compile the callee, then
    // apply `new`) rather than `compile_optional_chain` on the `New`.
    assert!(matches!(
        eval_ok("function K() { this.v = 9; } var o = {k: K}; var i = new (o?.k)(); i.v;"),
        Value::Int(9)
    ));
    assert!(matches!(
        eval_ok(
            "var o = {k: null}; var r = null; try { new (o?.k)(); } catch (e) { r = e instanceof TypeError; } r;"
        ),
        Value::Bool(true)
    ));
}

// -- function-declaration self-name visibility --------------------------------
//
// The function's own name must be declared in the enclosing scope *before*
// the body template is compiled, so self-references inside the body compile
// as upvalue captures of the outer binding instead of global lookups.

#[test]
fn function_self_reference_returns_itself() {
    assert!(matches!(
        eval_ok("function id() { return id; } id() === id;"),
        Value::Bool(true)
    ));
}

#[test]
fn nested_function_self_reference_captures_upvalue() {
    assert!(matches!(
        eval_ok(
            "function outer() { function inner() { return inner; } return inner; } var f = outer(); f() === f;"
        ),
        Value::Bool(true)
    ));
}

#[test]
fn function_self_reference_recursion() {
    assert!(matches!(
        eval_ok("function fact(n) { return n <= 1 ? 1 : n * fact(n - 1); } fact(5);"),
        Value::Int(120)
    ));
}

#[test]
fn function_hoisting_call_before_declaration() {
    assert!(matches!(
        eval_ok("var ok = false; foo(); function foo() { ok = true; } ok;"),
        Value::Bool(true)
    ));
}

#[test]
fn function_redeclaration_last_wins() {
    assert!(matches!(
        eval_ok("function f() { return 1; } function f() { return 2; } f();"),
        Value::Int(2)
    ));
}

#[test]
fn var_function_merge_function_wins() {
    assert!(matches!(
        eval_ok("var f; function f() { return 3; } f();"),
        Value::Int(3)
    ));
}

// -- named function expressions -------------------------------------------------
//
// Per ECMA-262 `NamedEvaluation`, `const f = function g() { ... }` binds
// `g` immutably in a fresh declarative environment whose binding is
// initialized to the function object itself, so `g` resolves inside the
// body, `g === f`, and the name does not leak to the enclosing scope.

fn eval_string(src: &str) -> String {
    match eval_ok(src) {
        Value::Str(s) => s.as_wtf8().to_string(),
        other => panic!("expected string for {src:?}, got {other:?}"),
    }
}

#[test]
fn named_function_expression_self_reference() {
    assert!(matches!(
        eval_ok("const f = function g() { return g; }; f() === f;"),
        Value::Bool(true)
    ));
}

#[test]
fn named_function_expression_recursion() {
    assert!(matches!(
        eval_ok("const fact = function f(n) { return n <= 1 ? 1 : n * f(n - 1); }; fact(5);"),
        Value::Int(120)
    ));
}

#[test]
fn named_function_expression_name_not_leaked() {
    assert_eq!(
        eval_string("const h = function q() {}; typeof q;"),
        "undefined"
    );
}

#[test]
fn named_function_expression_shadowing() {
    // The inner name resolves to the function; the outer `g` is untouched.
    assert_eq!(
        eval_string("var g = 1; const f3 = function g() { return typeof g; }; f3();"),
        "function"
    );
    assert!(matches!(
        eval_ok("var g = 1; const f3 = function g() {}; f3(); g;"),
        Value::Int(1)
    ));
}

#[test]
fn named_function_expression_param_shadows_name() {
    assert!(matches!(
        eval_ok("const f = function g(g) { return g; }; f(7);"),
        Value::Int(7)
    ));
}

#[test]
fn named_function_expression_var_shadows_name() {
    assert!(matches!(
        eval_ok("const f = function g() { var g = 5; return g; }; f();"),
        Value::Int(5)
    ));
}

#[test]
fn named_function_expression_inner_function_decl_shadows_name() {
    assert!(matches!(
        eval_ok("const f = function g() { function g() { return 1; } return g(); }; f();"),
        Value::Int(1)
    ));
}

#[test]
fn named_function_expression_nested_capture() {
    assert!(matches!(
        eval_ok(
            "const f = function g() { return function() { return g; }; }; const inner = f(); inner() === f;"
        ),
        Value::Bool(true)
    ));
}

#[test]
fn named_function_expression_per_evaluation_binding() {
    // Each evaluation gets its own name binding holding its own function
    // object (the slot lives in the callee's frame, not the enclosing one).
    assert!(matches!(
        eval_ok(
            "var fs = []; for (var i = 0; i < 3; i++) fs.push(function g() { return g; }); fs[0]() === fs[0] && fs[1]() === fs[1] && fs[2]() === fs[2];"
        ),
        Value::Bool(true)
    ));
}

#[test]
fn named_function_expression_sloppy_reassignment_noop() {
    assert!(matches!(
        eval_ok("const f = function g() { g = 42; return g; }; f() === f;"),
        Value::Bool(true)
    ));
}

#[test]
fn named_function_expression_sloppy_compound_reassignment_noop() {
    assert_eq!(
        eval_string("const f = function g() { g += 'x'; return typeof g; }; f();"),
        "function"
    );
    assert_eq!(
        eval_string("const f = function g() { g++; return typeof g; }; f();"),
        "function"
    );
}

#[test]
fn named_function_expression_sloppy_upvalue_reassignment_noop() {
    assert!(matches!(
        eval_ok(
            "var out; const f = function g() { (function() { g = 9; })(); out = g; }; f(); out === f;"
        ),
        Value::Bool(true)
    ));
}

#[test]
fn named_function_expression_strict_reassignment_throws() {
    assert_type_error("const f = function g() { 'use strict'; g = 42; }; f();");
}

#[test]
fn named_function_expression_strict_upvalue_reassignment_throws() {
    assert_type_error("const f = function g() { (function() { 'use strict'; g = 9; })(); }; f();");
}

#[test]
fn named_function_expression_async_generator() {
    assert!(matches!(
        eval_ok("const f = async function g() { return g; }; f() instanceof Promise;"),
        Value::Bool(true)
    ));
    assert!(matches!(
        eval_ok("const f = function* g() { yield g; }; const it = f(); it.next().value === f;"),
        Value::Bool(true)
    ));
}
// -- global declaration instantiation: no user code while seeding (F1) --------

/// The F1 probe: seeding the value cells for top-level `var`
/// declarations must never invoke user code. `b` is an accessor whose
/// getter redefines `a` and allocates heavily; with a collection on
/// (almost) every allocation, invoking it while `a`'s cell sat unrooted
/// would be a use-after-free. The declaration must not call the getter
/// at all, and `a`'s seeded object must survive intact.
#[test]
fn gc_stress_global_var_seeding_never_invokes_accessors() {
    let mut vm = Vm::new();
    // Collect on (almost) every allocation: any value the interpreter
    // keeps only in a Rust local across an allocating call is swept.
    vm.heap_mut().set_gc_threshold(1);
    vm.eval(
        r#"
        globalThis.a = {x: 12345};
        var getterCalls = 0;
        Object.defineProperty(globalThis, "b", {
            get: function() {
                getterCalls++;
                globalThis.a = {x: -1};
                var garbage = [];
                for (var i = 0; i < 50; i++) { garbage.push({y: i}); }
                return 7;
            },
            configurable: true
        });
        "#,
    )
    .expect("setup eval failed");
    // `a` is an own data global (value cell seeded from the descriptor),
    // `b` an accessor global (passthrough alias). Declaring over them
    // must not invoke the getter.
    vm.eval("var a; var b;").expect("probe eval failed");
    let x = vm.eval("a.x").expect("read eval failed");
    assert!(
        matches!(x, Value::Int(12345)),
        "seeded global clobbered or freed: {x:?}"
    );
    let calls = vm.eval("getterCalls").expect("counter eval failed");
    assert!(
        matches!(calls, Value::Int(0)),
        "getter invoked during declaration: {calls:?}"
    );
}

// -- direct eval vs named function expression self-name (F3) -------------------

#[test]
fn direct_eval_assign_to_self_name_sloppy_is_noop() {
    // Slot target: the eval runs in the function expression's own frame,
    // where the self-name is a compiled binding. The comparison runs
    // inside the VM so the assertion only touches a `Bool` (a returned
    // `Value::Str`'s GC handle dangles once the `Vm` drops).
    let v = eval_ok(
        "var f = function me() { eval('me = 99;'); return typeof me === 'function'; }; f();",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

#[test]
fn direct_eval_assign_to_self_name_strict_throws_type_error() {
    assert_type_error("var f = function me() { eval('\"use strict\"; me = 99;'); }; f();");
}

#[test]
fn direct_eval_assign_to_captured_self_name_sloppy_is_noop() {
    // Cell target: the nested closure captures the self-name as an
    // upvalue; the eval runs in the nested frame. Comparisons run inside
    // the VM so the assertions only touch `Bool`s (a returned
    // `Value::Str`'s GC handle dangles once the `Vm` drops).
    let v = eval_ok(
        "var f = function me() { \
           return function() { eval('me = 99;'); return typeof me === 'function'; }; \
         }; \
         f()();",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
    // The outer binding still names the function too.
    let w = eval_ok(
        "var g = function me() { \
           return function() { eval('me = 99;'); }; \
         }; \
         g()(); typeof g === 'function';",
    );
    assert!(matches!(w, Value::Bool(true)), "unexpected {w:?}");
}

#[test]
fn direct_eval_assign_to_captured_self_name_strict_throws_type_error() {
    assert_type_error(
        "var f = function me() { \
           return function() { eval('\"use strict\"; me = 99;'); }; \
         }; \
         f()();",
    );
}

// -- direct eval cell writes sync classic-script globals (F5) ------------------

#[test]
fn direct_eval_cell_write_syncs_global_mid_run() {
    // `g` is captured by `f`'s closure as an upvalue cell carrying the
    // global name; the eval's store must sync `globalThis.g` immediately,
    // not just at post-run install.
    let v = eval_ok("var g = 1; function f() { return eval('g = 2; globalThis.g'); } f();");
    assert!(matches!(v, Value::Int(2)), "unexpected {v:?}");
    let w = eval_ok("var g = 1; function f() { eval('g = 2;'); } f(); globalThis.g;");
    assert!(matches!(w, Value::Int(2)), "unexpected {w:?}");
}

#[test]
fn direct_eval_cell_write_to_accessor_global_fires_setter_once() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        var setterCalls = 0;
        Object.defineProperty(globalThis, "cacc", {
            get: function() { return 1; },
            set: function(v) { setterCalls++; },
            configurable: true
        });
        function f() { return eval("cacc = 9; setterCalls;"); }
        "#,
    )
    .expect("setup eval failed");
    // The captured accessor global is a passthrough: the eval store goes
    // straight through the global [[Set]], firing the setter exactly once
    // (the dummy cell is never written).
    let v = vm.eval("f();").expect("eval failed");
    assert!(matches!(v, Value::Int(1)), "unexpected {v:?}");
}

// -- native function prototypes (spec 10.2.4) -------------------------------

#[test]
fn native_functions_inherit_function_prototype() {
    // Every builtin function object has `%Function.prototype%` as its
    // prototype, so `.call`/`.apply`/`.bind` resolve on them.
    assert!(matches!(
        eval_ok("Object.getPrototypeOf(Array.prototype.every) === Function.prototype;"),
        Value::Bool(true)
    ));
    assert!(matches!(
        eval_ok("Object.getPrototypeOf(Object.keys) === Function.prototype;"),
        Value::Bool(true)
    ));
    let t = eval_ok("typeof Array.prototype.every.call;");
    assert!(
        matches!(t, Value::Str(ref st) if st.as_wtf8() == Wtf8::from_str("function")),
        "expected function, got {t:?}"
    );
    assert!(matches!(
        eval_ok("Array.prototype.every.call([1, 2], x => x > 0);"),
        Value::Bool(true)
    ));
}

// -- private member access in accessors --------------------------------------

#[test]
fn private_access_in_getter_and_setter() {
    assert!(matches!(
        eval_ok("class C { #m() { return 42; } get m() { return this.#m(); } } new C().m;"),
        Value::Int(42)
    ));
    assert!(matches!(
        eval_ok(
            "class C { #x = 1; get x() { return this.#x; } set x(v) { this.#x = v; } } const c = new C(); c.x = 7; c.x;"
        ),
        Value::Int(7)
    ));
    assert!(matches!(
        eval_ok("class C { static #s() { return 3; } static get s() { return this.#s(); } } C.s;"),
        Value::Int(3)
    ));
}

// -- Boolean constructor and globals ------------------------------------------

#[test]
fn boolean_constructor_and_prototype() {
    assert!(matches!(eval_ok("Boolean(1);"), Value::Bool(true)));
    assert!(matches!(eval_ok("Boolean(0);"), Value::Bool(false)));
    assert!(matches!(eval_ok("Boolean('');"), Value::Bool(false)));
    let t = eval_ok("typeof Boolean;");
    assert!(
        matches!(t, Value::Str(ref st) if st.as_wtf8() == Wtf8::from_str("function")),
        "expected function, got {t:?}"
    );
    assert!(matches!(
        eval_ok("Object.getPrototypeOf(Boolean) === Function.prototype;"),
        Value::Bool(true)
    ));
    // `new Boolean(x)` boxes.
    let t = eval_ok("typeof new Boolean(true);");
    assert!(
        matches!(t, Value::Str(ref st) if st.as_wtf8() == Wtf8::from_str("object")),
        "expected object, got {t:?}"
    );
    assert!(matches!(
        eval_ok("new Boolean(0).valueOf();"),
        Value::Bool(false)
    ));
    let t = eval_ok("new Boolean(1).toString();");
    assert!(
        matches!(t, Value::Str(ref st) if st.as_wtf8() == Wtf8::from_str("true")),
        "expected 'true', got {t:?}"
    );
    assert!(matches!(
        eval_ok("Object.getPrototypeOf(new Boolean(true)) === Boolean.prototype;"),
        Value::Bool(true)
    ));
}

#[test]
fn global_isnan_and_isfinite_coerce() {
    assert!(matches!(eval_ok("isNaN(NaN);"), Value::Bool(true)));
    assert!(matches!(eval_ok("isNaN('foo');"), Value::Bool(true)));
    assert!(matches!(eval_ok("isNaN(5);"), Value::Bool(false)));
    assert!(matches!(eval_ok("isFinite(5);"), Value::Bool(true)));
    assert!(matches!(eval_ok("isFinite('5');"), Value::Bool(true)));
    assert!(matches!(eval_ok("isFinite(Infinity);"), Value::Bool(false)));
    // The `Number.*` statics do NOT coerce.
    assert!(matches!(
        eval_ok("Number.isNaN('foo');"),
        Value::Bool(false)
    ));
    assert!(matches!(
        eval_ok("Number.isFinite('5');"),
        Value::Bool(false)
    ));
}

// -- RegExp replace: replacer cannot corrupt the match scan ------------------

#[test]
fn replace_collects_matches_before_running_replacer() {
    // A replacer that mutates `lastIndex` must not corrupt (or panic) the
    // scan: matches are collected first, per spec.
    let t =
        eval_ok("const rx = /a/g; \"aa\".replace(rx, () => { rx.lastIndex = 0; return \"x\"; });");
    assert!(
        matches!(t, Value::Str(ref s) if s.as_wtf8() == Wtf8::from_str("xx")),
        "expected 'xx', got {t:?}"
    );
}

// -- RegExp literals: invalid patterns are compile-time SyntaxErrors ------

#[test]
fn invalid_regex_literal_is_compile_time_syntax_error() {
    // Spec early error: an invalid pattern must fail before evaluation,
    // so `negative: {phase: parse}` tests see it at parse time.
    let err = eval("/\\P{ASCII=F}/u").expect_err("expected SyntaxError for invalid regex");
    assert!(
        err.contains("invalid regular expression"),
        "expected regex SyntaxError, got {err}"
    );
    // Valid patterns still compile and run.
    let t = eval_ok("/a+/g.test('aa')");
    assert!(matches!(t, Value::Bool(true)), "expected true, got {t:?}");
}

// -- lexical scoping across textual order (TDZ) ------------------------------
// A function defined textually BEFORE a `let`/`const` in the same scope
// must still resolve the binding (with TDZ enforced at call time), not
// fall back to a global lookup. Regression tests for the wave-39 WPT
// smoke failure (`get_stack is not defined` from testharness.js).

#[test]
fn fn_before_const_resolves_after_init() {
    // The exact repro shape: F is defined before the const, constructed
    // after the initializer ran (mirrors testharness.js, whose
    // AssertionError constructor calls a later-declared `const
    // get_stack`). The constructor records through a side channel because
    // `new` discards a primitive return value.
    let v = eval_ok(
        "var seen = null; \
         function F() { seen = get_stack(); } \
         const get_stack = function() { return 42; }; \
         new F(); seen === 42;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

#[test]
fn fn_before_let_resolves_after_init() {
    let v = eval_ok("function F() { return x; } let x = 7; F() === 7;");
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

#[test]
fn fn_before_const_called_before_init_still_tdz() {
    // Calling before the initializer ran must still throw a genuine TDZ
    // ReferenceError ("cannot access ... before initialization"), not the
    // old global-miss "x is not defined".
    let v = eval_ok(
        "function F() { return x; } \
         var caught = null; \
         try { F(); } catch (e) { caught = e; } \
         const x = 1; \
         caught instanceof ReferenceError && \
         caught.message.indexOf('before initialization') !== -1;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

#[test]
fn fn_before_class_resolves_after_init() {
    // Class declarations are lexical too: same pre-declaration applies.
    let v = eval_ok(
        "function F() { return new C().n; } \
         class C { constructor() { this.n = 11; } } \
         F() === 11;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

#[test]
fn fn_before_const_in_block_scope() {
    let v = eval_ok("var r; { function F() { return x; } const x = 3; r = F(); } r === 3;");
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

#[test]
fn nested_fn_before_const_in_enclosing_fn() {
    let v = eval_ok(
        "function outer() { function F() { return x; } const x = 9; return F(); } \
         outer() === 9;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

#[test]
fn fn_after_const_still_works() {
    let v = eval_ok("const x = 5; function F() { return x; } F() === 5;");
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

#[test]
fn closure_sees_later_const_assignment() {
    // The captured cell (not a snapshot): a later write is visible.
    let v = eval_ok("function F() { return x; } let x = 1; x = 2; F() === 2;");
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

// -- wave-44 trampoline fixes ------------------------------------------------

#[test]
fn yield_star_stale_delegate_outer_direct_next() {
    // HIGH: outer `yield*`s inner (generator), inner `yield*`s a custom
    // iterable (slow path). After parking on the slow path, driving inner
    // directly must complete inner normally; a stale `delegate_outer` must
    // not resume the outer generator (its return value leaked into
    // `inner.next()`'s result).
    let v = eval_ok(
        "var innerGen; \
         var custom = { \
           n: 0, \
           next: function () { \
             this.n++; \
             if (this.n === 1) return { value: 'y1', done: false }; \
             return { value: 'custom-ret', done: true }; \
           }, \
           [Symbol.iterator]: function () { return this; }, \
         }; \
         function* inner() { var cr = yield* custom; return 'inner-ret:' + cr; } \
         function* outer() { yield* (innerGen = inner()); return 'outer-ret'; } \
         var outerGen = outer(); \
         var r1 = outerGen.next(); \
         var r2 = innerGen.next(); \
         r1.value === 'y1' && r1.done === false \
           && r2.value === 'inner-ret:custom-ret' && r2.done === true;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

#[test]
fn yield_star_stale_delegate_outer_direct_throw() {
    // HIGH: same parking as above, but the parked generator is driven
    // with `.throw()`: the throw must propagate to the direct caller, not
    // resume (and be caught by) the outer generator via a stale
    // `delegate_outer`.
    let v = eval_ok(
        "var innerGen; \
         var outerCaught = false; \
         var custom = { \
           next: function () { return { value: 'y1', done: false }; }, \
           [Symbol.iterator]: function () { return this; }, \
         }; \
         function* inner() { yield* custom; return 'inner-ret'; } \
         function* outer() { \
           try { yield* (innerGen = inner()); } catch (e) { outerCaught = true; } \
           return 'outer-ret'; \
         } \
         var outerGen = outer(); \
         var r1 = outerGen.next(); \
         var caughtIsTypeError = false; \
         try { innerGen.throw('boom'); } catch (e) { caughtIsTypeError = e instanceof TypeError; } \
         r1.value === 'y1' && caughtIsTypeError && !outerCaught;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

#[test]
fn yield_star_completed_delegate_hops_reset() {
    // MEDIUM: each `yield* <completed>` walked one delegation hop without
    // resetting the budget on the immediate-done path, so enough
    // sequential delegations raised a bogus `RangeError: stack overflow`.
    let v = eval_ok(
        "function* d() {} var done = d(); done.next(); \
         function* g() { for (var i = 0; i < 20000; i++) yield* done; return 'ok'; } \
         var r = g().next(); r.done === true && r.value === 'ok';",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

#[test]
fn toprimitive_async_hook_body_runs_nested() {
    // MEDIUM: an async `@@toPrimitive` hook used to be staged as a frame
    // whose body never ran — the op faulted on the result promise first
    // and dropped it. Now the hook drives nested (like `YieldStarNext` /
    // `Assimilate` async callees): the body runs to first await, then the
    // promise is rejected as a non-primitive result.
    let v = eval_ok(
        "var log = []; var o = {}; \
         Object.defineProperty(o, Symbol.toPrimitive, \
           { value: async function (h) { log.push('body'); return 1; } }); \
         try { o + 1; } catch (e) { log.push(e instanceof TypeError ? 'typeerror' : 'other'); } \
         log.join(',');",
    );
    assert!(
        matches!(v, Value::Str(s) if s.as_wtf8() == Wtf8::from_str("body,typeerror")),
        "unexpected {v:?}"
    );
}

#[test]
fn proxy_construct_trap_returning_primitive_throws() {
    // LOW: `SiteKind::ProxyConstruct` — a construct trap returning a
    // primitive violates ECMA-262 10.5.15 step 13.
    assert_type_error(
        "var p = new Proxy(function () {}, \
           { construct: function (t, args) { return 42; } }); \
         new p();",
    );
}

#[test]
fn proxy_construct_deep_chain_forwards() {
    // LOW: a construct trap through two proxy layers still enforces the
    // invariant and delivers the trap's object.
    let v = eval_ok(
        "var inner = new Proxy(function () {}, \
           { construct: function (t, args) { return { marker: 'trap' }; } }); \
         var outer = new Proxy(inner, {}); \
         var r = new outer(1, 2); \
         r.marker;",
    );
    assert!(
        matches!(v, Value::Str(s) if s.as_wtf8() == Wtf8::from_str("trap")),
        "unexpected {v:?}"
    );
}

#[test]
fn eval_trampolined_throw_propagates() {
    // LOW: `SiteKind::Eval` — a throwing trampolined eval frame runs the
    // completion tail and propagates the thrown value.
    let v = eval_ok(
        "var caught = null; \
         try { eval('throw 99;'); } catch (e) { caught = e; } \
         caught === 99;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

#[test]
fn eval_throwing_indirect_sloppy_keeps_fresh_globals() {
    // LOW: a throwing indirect sloppy eval still installs the globals it
    // defined before throwing (fresh globals are defined up front).
    let v = eval_ok(
        "try { (0, eval)('var partialInstall = 123; throw \"boom\";'); } catch (e) {} \
         typeof partialInstall === 'number' && partialInstall === 123;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

#[test]
fn newregex_overridden_global_staged() {
    // LOW: `SiteKind::NewRegex` — a user-overridden bytecode `RegExp` goes
    // through the site-call trampoline; the placeholder is swapped for the
    // override's return value.
    let v = eval_ok(
        "var seen = []; \
         RegExp = function (p, f) { seen.push(p + '/' + f); return 'custom:' + p; }; \
         var r = /ab/i; \
         r + '|' + seen.join(',');",
    );
    assert!(
        matches!(v, Value::Str(s) if s.as_wtf8() == Wtf8::from_str("custom:ab|ab/i")),
        "unexpected {v:?}"
    );
}

// -- destructuring assignment (ECMA-262 13.15.5) ---------------------------
// `[a, b] = rhs` / `({a, b} = rhs)`: the parser keeps the pattern as an
// `Expr::Array`/`Expr::Object` LHS; the compiler converts it back to a
// `Pat` and reuses the declaration-side binder in assignment mode.

/// Assert `src` evaluates to boolean `true`.
fn eval_true(src: &str) {
    let v = eval_ok(src);
    assert!(
        matches!(v, Value::Bool(true)),
        "expected true, got {v:?} for {src:?}"
    );
}

#[test]
fn destructuring_assign_array_basic() {
    eval_true("var a, b; [a, b] = [1, 2]; a === 1 && b === 2;");
}

#[test]
fn destructuring_assign_array_elision() {
    eval_true("var a, b; [a, , b] = [1, 2, 3]; a === 1 && b === 3;");
}

#[test]
fn destructuring_assign_array_defaults() {
    // NOTE: the parser currently rejects a bare `[a = 1] = rhs`
    // ("invalid initializer in object/array pattern" — a `pending_cover`
    // bug in the array-literal cover path, parser.rs); the parenthesized
    // form parses and exercises the same compiler path.
    eval_true("var a, b; ([a = 10, b = 20] = [1]); a === 1 && b === 20;");
    // Defaults trigger only on `undefined`, not on `null`.
    eval_true("var a; ([a = 10] = [null]); a === null;");
}

#[test]
fn destructuring_assign_array_rest() {
    eval_true(
        "var a, r; [a, ...r] = [1, 2, 3]; \
         a === 1 && r.length === 2 && r[0] === 2 && r[1] === 3;",
    );
    eval_true("var r; [...r] = [1]; r.length === 1 && r[0] === 1;");
}

#[test]
fn destructuring_assign_array_nested() {
    eval_true("var a, b; [[a], {x: b}] = [[1], {x: 2}]; a === 1 && b === 2;");
    eval_true("var a; [[[a]]] = [[[7]]]; a === 7;");
}

#[test]
fn destructuring_assign_array_result_value() {
    // The assignment expression evaluates to the RHS value.
    eval_true("var a; var r = ([a] = [5]); r[0] === 5 && a === 5;");
    eval_true("var a, b; [a] = [b] = [3]; a === 3 && b === 3;");
}

#[test]
fn destructuring_assign_array_evaluation_order() {
    // The RHS evaluates exactly once, before any pattern work.
    eval_true("var n = 0, a; [a] = (n++, [42]); n === 1 && a === 42;");
    // Elements extract left to right; a default runs only when its
    // element is `undefined`, after the element value is known.
    eval_true(
        "var log = []; \
         function t(n, v) { log.push(n); return v; } \
         var a, b; ([a = t('dflt', 9), b] = [t('e0', undefined), t('e1', 7)]); \
         a === 9 && b === 7 && log.join(',') === 'e0,e1,dflt';",
    );
    // Iterator steps run in element order (ECMA-262 13.15.5: array
    // patterns consume the iterator protocol, not indexed reads).
    eval_true(
        "var log = []; var a, b; \
         function* g() { log.push('g0'); yield 1; log.push('g1'); yield 2; } \
         [a, b] = g(); \
         a === 1 && b === 2 && log.join(',') === 'g0,g1';",
    );
    // Array-likes without [Symbol.iterator] are not valid RHS values.
    assert_type_error("var q; [q] = {0: 1, length: 1};");
}

#[test]
fn destructuring_assign_object_basic() {
    eval_true("var a, b; ({a, b} = {a: 1, b: 2}); a === 1 && b === 2;");
    eval_true("var c; ({a: c} = {a: 42}); c === 42;");
}

#[test]
fn destructuring_assign_object_defaults() {
    eval_true("var a, b; ({a = 1, b = 2} = {}); a === 1 && b === 2;");
    eval_true("var a; ({a = 1} = {a: undefined}); a === 1;");
    eval_true("var a; ({a = 1} = {a: null}); a === null;");
}

#[test]
fn destructuring_assign_object_rest() {
    eval_true(
        "var a, rest; ({a, ...rest} = {a: 1, b: 2, c: 3}); \
         a === 1 && rest.b === 2 && rest.c === 3 && !('a' in rest);",
    );
    // Rest collects own enumerable keys, including renamed and
    // defaulted properties' keys.
    eval_true(
        "var rest; ({x: a, y: b = 2, ...rest} = {x: 1, z: 3}); \
         a === 1 && b === 2 && rest.z === 3 && \
         !('x' in rest) && !('y' in rest);",
    );
    // Numeric keys are excluded too.
    eval_true(
        "var rest; ({0: z, ...rest} = {0: 'zero', 1: 'one'}); \
         z === 'zero' && rest[1] === 'one' && !('0' in rest);",
    );
    // NOTE: getters are not invoked by the copy — `#obj_spread`
    // (which the rest lowering reuses) copies data properties only.
    // That is a pre-existing spread limitation, not a rest gap:
    // `...rest` and `{...o}` agree.
    eval_true(
        "var o = {a: 1, get b() { return 2; }}; \
         var r1, r2; ({a, ...r1} = o); var c = {...o}; \
         ('b' in r1) === ('b' in c);",
    );
}

#[test]
fn destructuring_declaration_object_rest() {
    // The shared pattern binder lowers `...rest` via `#obj_spread` +
    // deletes, so declarations benefit too (`#obj_rest` was never
    // implemented in the VM).
    eval_true("var {a, ...rest} = {a: 1, b: 2}; a === 1 && rest.b === 2 && !('a' in rest);");
}

#[test]
fn destructuring_assign_object_computed_keys() {
    eval_true("var k = 'x', v; ({[k]: v} = {x: 9}); v === 9;");
    // The computed key evaluates exactly once, before the property read.
    eval_true("var n = 0, v; ({[(n++, 'x')]: v} = {x: 5}); n === 1 && v === 5;");
    eval_true("var v; ({'a-b': v} = {'a-b': 4}); v === 4;");
    eval_true("var v; ({0: v} = {0: 5}); v === 5;");
}

#[test]
fn destructuring_assign_object_nested() {
    eval_true("var x, y; ({p: {x}, q: [y]} = {p: {x: 1}, q: [2]}); x === 1 && y === 2;");
}

#[test]
fn destructuring_assign_object_result_value() {
    eval_true("var o; var r = ({a: o} = {a: 3}); r.a === 3 && o === 3;");
}

#[test]
fn destructuring_assign_null_rhs_throws_type_error() {
    assert_type_error("[a] = null;");
    assert_type_error("[a] = undefined;");
    assert_type_error("({a} = null);");
    assert_type_error("({a} = undefined);");
}

#[test]
fn destructuring_assign_const_target_throws() {
    assert_type_error("const a = 1; [a] = [2];");
    assert_type_error("const a = 1; ({a} = {a: 2});");
    assert_type_error("const a = 1; ([a = 2] = []);");
}

#[test]
fn destructuring_assign_compound_op_is_syntax_error() {
    // Patterns are not valid simple assignment targets: `[a] += b`
    // stays a SyntaxError ("invalid left-hand side").
    let err = eval("[a] += [1];").expect_err("expected SyntaxError");
    assert!(err.contains("invalid left-hand side"), "unexpected {err}");
    let err = eval("({a} -= {a: 1});").expect_err("expected SyntaxError");
    assert!(err.contains("invalid left-hand side"), "unexpected {err}");
}

#[test]
fn destructuring_assign_with_object() {
    // Assignment-mode stores are with-aware, like plain `a = v`.
    eval_true("var o = {a: 0}; with (o) { [a] = [7]; } o.a === 7;");
    eval_true("var o = {b: 0}; with (o) { ({b} = {b: 8}); } o.b === 8;");
}

// -- destructuring: member targets (ECMA-262 13.15.5) ------------------------
// `MemberExpression` is a valid destructuring assignment target.

#[test]
fn destructuring_member_target_array() {
    eval_true("var o = {}; [o.x] = [1]; o.x === 1;");
    eval_true("var o = {}; var k = 'y'; [o[k]] = [2]; o.y === 2;");
    eval_true("var o = {}; [[o.a]] = [[3]]; o.a === 3;");
    // The assignment expression still evaluates to the RHS value.
    eval_true("var o = {}; var r = ([o.x] = [4]); r[0] === 4 && o.x === 4;");
    // Multiple member targets bind left to right.
    eval_true("var o = {}, p = {}; [o.a, p.b] = [5, 6]; o.a === 5 && p.b === 6;");
}

#[test]
fn destructuring_member_target_object() {
    eval_true("var o = {}; ({p: o.q} = {p: 7}); o.q === 7;");
    eval_true("var o = {}; ({p: o.q = 8} = {}); o.q === 8;");
    eval_true("var o = {}; var k = 'z'; ({[k]: o.w} = {z: 9}); o.w === 9;");
}

#[test]
fn destructuring_member_target_for_of() {
    eval_true("var o = {}; for ([o.z] of [[10]]) {} o.z === 10;");
    eval_true(
        "var o = {}, n = 0; for ([o['k' + (n++)]] of [[11], [12]]) {} \
         o.k0 === 11 && o.k1 === 12 && n === 2;",
    );
}

#[test]
fn destructuring_member_target_evaluation_order() {
    // The RHS evaluates before each element's member reference resolves
    // (ECMA-262 13.15.5 IteratorDestructuringAssignmentEvaluation /
    // KeyedDestructuringAssignmentEvaluation).
    eval_true(
        "var log = []; var target = {}; \
         var holder = { get o() { log.push('ref'); return target; } }; \
         [holder.o.x] = (log.push('rhs'), [1]); \
         target.x === 1 && log.join(',') === 'rhs,ref';",
    );
    eval_true(
        "var log = []; var target = {}; \
         var holder = { get o() { log.push('ref'); return target; } }; \
         ({p: holder.o.x} = (log.push('rhs'), {p: 2})); \
         target.x === 2 && log.join(',') === 'rhs,ref';",
    );
    // Bare member assignment keeps reference-before-RHS.
    eval_true(
        "var log = []; var target = {}; \
         var holder = { get o() { log.push('ref'); return target; } }; \
         holder.o.x = (log.push('rhs'), 3); \
         target.x === 3 && log.join(',') === 'ref,rhs';",
    );
}

#[test]
fn destructuring_member_target_rejects_optional_chain() {
    let err = eval("[a?.b] = [1];").expect_err("expected SyntaxError");
    assert!(err.contains("optional chain"), "unexpected {err}");
    let err = eval("({p: a?.b} = {});").expect_err("expected SyntaxError");
    assert!(err.contains("optional chain"), "unexpected {err}");
}

// -- destructuring: iterator protocol (ECMA-262 13.15.5) --------------------
// Array patterns consume the iterator protocol (`GetIterator` +
// `IteratorStep` + `IteratorValue`), not indexed property reads.

#[test]
fn destructuring_array_iterator_generator() {
    eval_true(
        "function* g() { yield 1; yield 2; } \
         var [x, y] = g(); x === 1 && y === 2;",
    );
    eval_true(
        "function* g() { yield 1; yield 2; } \
         const [a] = g(); a === 1;",
    );
    // An exhausted iterator yields `undefined` for the rest.
    eval_true(
        "function* g() { yield 1; } \
         var [a, b] = g(); a === 1 && b === undefined;",
    );
}

#[test]
fn destructuring_array_iterator_set_map_parity() {
    // `Set.prototype[Symbol.iterator]` / `Map.prototype[Symbol.iterator]`
    // are installed (wave 47): array destructuring goes through the same
    // iterator machinery as `for-of`, so it agrees with it exactly — both
    // consume Sets and Maps now.
    eval_true("var [a, ...r] = new Set([1, 2, 3]); a === 1 && r.join(',') === '2,3';");
    eval_true("var [[k, v]] = new Map([['a', 1]]); k === 'a' && v === 1;");
    eval_true(
        "var seen = []; for (var x of new Set([1])) { seen.push(x); } \
         seen.length === 1 && seen[0] === 1;",
    );
}

#[test]
fn destructuring_array_iterator_custom_iterable() {
    eval_true(
        "var it = { \
           [Symbol.iterator]() { \
             var n = 0; \
             return { next() { \
               n++; \
               return n <= 2 ? {value: n * 10, done: false} \
                             : {value: undefined, done: true}; \
             } }; \
           } \
         }; \
         var [a, b] = it; a === 10 && b === 20;",
    );
}

#[test]
fn destructuring_array_iterator_nested() {
    // Nested array patterns each run the iterator protocol.
    eval_true("var [[k, v]] = [['a', 1]]; k === 'a' && v === 1;");
    eval_true(
        "function* g() { yield [1, 2]; yield [3, 4]; } \
         var [[a, b], [c]] = g(); a === 1 && b === 2 && c === 3;",
    );
}

#[test]
fn destructuring_array_iterator_elision_and_defaults() {
    eval_true(
        "function* g() { yield 1; yield 2; yield 3; } \
         var [, b] = g(); b === 2;",
    );
    eval_true("function* g() {} var [a = 1] = g(); a === 1;");
    eval_true(
        "function* g() { yield undefined; } \
         var [a = 1] = g(); a === 1;",
    );
}

#[test]
fn destructuring_array_iterator_rest() {
    eval_true("function* g() {} var [...r] = g(); r.length === 0;");
    eval_true(
        "function* g() { yield 1; yield 2; } \
         var [a, ...r] = g(); a === 1 && r.length === 1 && r[0] === 2;",
    );
    // Rest keeps stepping until the iterator is done.
    eval_true(
        "var n = 0; \
         function* g() { n++; yield 1; n++; yield 2; n++; } \
         var [...r] = g(); r.length === 2 && n === 3;",
    );
}

#[test]
fn destructuring_array_iterator_string() {
    eval_true("var [a, b] = 'xy'; a === 'x' && b === 'y';");
}

#[test]
fn destructuring_array_non_iterable_throws() {
    assert_type_error("var [a] = 5;");
    assert_type_error("var [a] = {0: 1, length: 1};");
    assert_type_error("var [a] = null;");
    assert_type_error("var [a, ...r] = {};");
    assert_type_error("var a; [a] = 5;");
}

// -- loose equality: object vs object ---------------------------------------
// Per ECMA-262 §7.2.14 step 1: when both operands are Object, abstract
// equality reduces to strict equality (identity), never ToPrimitive.

#[test]
fn loose_eq_distinct_objects_are_not_equal() {
    assert!(matches!(eval_ok("({} == {});"), Value::Bool(false)));
    assert!(matches!(
        eval_ok("var a = {}; var b = {}; a == b;"),
        Value::Bool(false)
    ));
    assert!(matches!(
        eval_ok("var a = {}; var b = {}; a != b;"),
        Value::Bool(true)
    ));
    // Distinct arrays are objects too.
    assert!(matches!(
        eval_ok("var a = [1]; var b = [1]; a == b;"),
        Value::Bool(false)
    ));
}

#[test]
fn loose_eq_same_object_reference_is_equal() {
    assert!(matches!(eval_ok("var o = {}; o == o;"), Value::Bool(true)));
    assert!(matches!(
        eval_ok("var a = {}; var b = a; a == b;"),
        Value::Bool(true)
    ));
    assert!(matches!(eval_ok("var a = {}; a != a;"), Value::Bool(false)));
}

#[test]
fn loose_eq_object_vs_primitive_still_to_primitives() {
    assert!(matches!(eval_ok("[1] == '1';"), Value::Bool(true)));
    assert!(matches!(
        eval_ok("({ valueOf: function() { return 'x'; } }) == 'x';"),
        Value::Bool(true)
    ));
    assert!(matches!(
        eval_ok("({ toString: function() { return '42'; } }) == '42';"),
        Value::Bool(true)
    ));
    assert!(matches!(eval_ok("[] == '';"), Value::Bool(true)));
}

#[test]
fn loose_eq_nullproto_objects_do_not_throw() {
    // A wrapper-style object with no valueOf/toString on its chain must not
    // throw TypeError when compared to another object; distinct refs are
    // simply unequal.
    assert!(matches!(
        eval_ok("var a = Object.create(null); var b = Object.create(null); a == b;"),
        Value::Bool(false)
    ));
    assert!(matches!(
        eval_ok("var a = Object.create(null); a == a;"),
        Value::Bool(true)
    ));
}
