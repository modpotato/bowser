//! Wave-36 JS conformance smalls: for-in/of LHS forms, `with`, BigInt
//! `ToPrimitive`, `RegExp(regexp)` identity, per-type typed-array
//! prototypes, `RegExp.prototype.toString`.

use bowser_js::{Value, Vm};

fn eval(src: &str) -> Result<Value, String> {
    Vm::new().eval(src).map_err(|e| format!("{e:?}"))
}

fn eval_ok(src: &str) -> Value {
    eval(src).unwrap_or_else(|e| panic!("eval failed for {src:?}: {e}"))
}

fn eval_bool(src: &str) -> bool {
    match eval_ok(src) {
        Value::Bool(b) => b,
        v => panic!("expected bool for {src:?}, got {v:?}"),
    }
}

fn eval_str(src: &str) -> String {
    match eval_ok(&format!("String({src})")) {
        Value::Str(s) => s.as_wtf8().to_string(),
        v => panic!("expected string for {src:?}, got {v:?}"),
    }
}

// -- 1. for-in/of LHS forms in both loop-compile paths -----------------------
// The extracted (per-iteration-body) path triggers when the body holds a
// closure plus a lexical declaration.

#[test]
fn for_of_member_lhs_extracted() {
    assert!(eval_bool(
        "var o={}, fns=[]; \
         for (o.p of [1,2]) { let z = o.p; fns.push(() => z); } \
         fns[0]() === 1 && fns[1]() === 2 && o.p === 2;"
    ));
}

#[test]
fn for_of_member_lhs_plain() {
    assert!(eval_bool("var o={}; for (o.p of [5]) {} o.p === 5;"));
}

#[test]
fn for_in_member_lhs_extracted() {
    assert!(eval_bool(
        "var o={}, ks=[]; \
         for (o.k in {x:1}) { let z = o.k; ks.push(() => z); } \
         ks[0]() === 'x' && o.k === 'x';"
    ));
}

#[test]
fn for_of_var_destructuring_lhs_extracted() {
    assert!(eval_bool(
        "var fns=[]; \
         for (var {a} of [{a:1},{a:2}]) { let z = a; fns.push(() => z); } \
         fns[0]() === 1 && fns[1]() === 2;"
    ));
}

#[test]
fn for_of_bare_destructuring_lhs_extracted() {
    assert!(eval_bool(
        "var fns=[], b; \
         for ([b] of [[3],[4]]) { let z = b; fns.push(() => z); } \
         fns[0]() === 3 && fns[1]() === 4 && b === 4;"
    ));
}

#[test]
fn for_of_var_destructuring_lhs_plain() {
    assert!(eval_bool(
        "var r=0; for (var {a} of [{a:7}]) { r = a; } r === 7;"
    ));
}

#[test]
fn for_of_let_destructuring_lhs_extracted() {
    assert!(eval_bool(
        "var fns=[]; \
         for (const {a} of [{a:9}]) { fns.push(() => a); } \
         fns[0]() === 9;"
    ));
}

// -- 2. `with` statement (sloppy mode) ----------------------------------------

#[test]
fn with_reads_object_properties() {
    assert!(eval_bool(
        "var o={x:42}; var r=0; with (o) { r = x; } r === 42;"
    ));
}

#[test]
fn with_write_goes_to_object() {
    assert!(eval_bool("var o={x:1}; with (o) { x = 99; } o.x === 99;"));
}

#[test]
fn with_missing_name_falls_through_to_global() {
    assert!(eval_bool("var o={}; with (o) { y = 7; } y === 7;"));
}

#[test]
fn with_object_shadows_outer_var() {
    assert!(eval_bool(
        "var x=1; var o={x:2}; var r=0; with (o) { r = x; } r === 2 && x === 1;"
    ));
}

#[test]
fn with_inner_lexical_shadows_object() {
    assert!(eval_bool(
        "var o={x:2}; var r=0; with (o) { let x=1; r = x; } r === 1;"
    ));
}

#[test]
fn with_null_throws_typeerror() {
    assert!(eval_bool(
        "var ok=false; try { with (null) { 1; } } catch (e) { ok = e instanceof TypeError; } ok;"
    ));
}

#[test]
fn with_undefined_throws_typeerror() {
    assert!(eval_bool(
        "var ok=false; try { with (undefined) { 1; } } catch (e) { ok = e instanceof TypeError; } ok;"
    ));
}

#[test]
fn with_typeof_probes_object() {
    assert!(eval_bool(
        "var o={x:1}; var a, b; \
         with (o) { a = typeof x; b = typeof zzz; } \
         a === 'number' && b === 'undefined';"
    ));
}

#[test]
fn with_nested() {
    assert!(eval_bool(
        "var a={x:1}, b={y:2}; var r=0; \
         with (a) { with (b) { r = x + y; } } r === 3;"
    ));
}

#[test]
fn with_compound_and_update() {
    assert!(eval_bool(
        "var o={n:1}; with (o) { n += 1; n++; } o.n === 3;"
    ));
}

#[test]
fn with_prototype_chain_property() {
    assert!(eval_bool(
        "var base={inherited:5}; var o=Object.create(base); var r=0; \
         with (o) { r = inherited; } r === 5;"
    ));
}

#[test]
fn with_strict_rejected() {
    assert!(eval("\"use strict\"; with ({}) {}").is_err());
}

#[test]
fn with_method_call_this_is_object() {
    // `with (o) { f() }` calls `f` with `this === o`.
    assert!(eval_bool(
        "var o={x:5, get(){return this.x}}; var r=0; with (o) { r = get(); } r === 5;"
    ));
}

#[test]
fn with_method_call_this_falls_through() {
    // Miss: ordinary global call, `this` is the global object (sloppy).
    assert!(eval_bool(
        "var x=9; function get(){return this.x} var r=0; \
         with ({}) { r = get(); } r === 9;"
    ));
}

// -- 3. BigInt(object) via ToPrimitive ------------------------------------------

#[test]
fn bigint_from_object_valueof() {
    assert_eq!(eval_str("BigInt({valueOf(){return 7}})"), "7");
}

#[test]
fn bigint_from_object_to_string() {
    assert_eq!(eval_str("BigInt({toString(){return '42'}})"), "42");
}

#[test]
fn bigint_from_object_to_primitive_hook() {
    assert_eq!(eval_str("BigInt({[Symbol.toPrimitive](){return 9}})"), "9");
}

#[test]
fn bigint_from_plain_object_throws() {
    // `ToPrimitive({})` yields `"[object Object]"`, which is not a valid
    // BigInt literal: SyntaxError.
    assert!(eval_bool(
        "var n=''; try { BigInt({}); } catch (e) { n = e.name; } n === 'SyntaxError';"
    ));
}

// -- 4. RegExp(regexp) identity short-circuit -----------------------------------

#[test]
fn regexp_call_form_returns_same_object() {
    assert!(eval_bool("var r=/ab/g; RegExp(r) === r;"));
}

#[test]
fn regexp_construct_form_copies() {
    assert!(eval_bool("var r=/ab/g; new RegExp(r) !== r;"));
}

#[test]
fn regexp_call_form_with_flags_copies() {
    assert!(eval_bool(
        "var r=/ab/g; var r2=RegExp(r,'i'); \
         r2 !== r && r2.source === 'ab' && r2.flags === 'i';"
    ));
}

#[test]
fn regexp_call_form_overridden_constructor_copies() {
    assert!(eval_bool(
        "var r=/ab/; r.constructor=String; RegExp(r) !== r;"
    ));
}

// -- 6. RegExp.prototype.toString ------------------------------------------------

#[test]
fn regexp_tostring_on_regexp() {
    assert_eq!(eval_str("/ab/g.toString()"), "/ab/g");
    assert_eq!(eval_str("new RegExp('b','m').toString()"), "/b/m");
}

#[test]
fn regexp_tostring_on_plain_object() {
    assert_eq!(
        eval_str("RegExp.prototype.toString.call({source:'x',flags:'gi'})"),
        "/x/gi"
    );
}

#[test]
fn regexp_tostring_defaults() {
    // Absent/`undefined` source/flags go through `ToString`
    // (spec 22.2.5.15 steps 3-4): no `(?:)` default here.
    assert_eq!(
        eval_str("RegExp.prototype.toString.call({})"),
        "/undefined/undefined"
    );
    assert_eq!(
        eval_str("RegExp.prototype.toString.call({source:'a'})"),
        "/a/undefined"
    );
    assert_eq!(
        eval_str("RegExp.prototype.toString.call({flags:'m'})"),
        "/undefined/m"
    );
    assert_eq!(
        eval_str("RegExp.prototype.toString.call({source:undefined,flags:undefined})"),
        "/undefined/undefined"
    );
}

#[test]
fn regexp_tostring_empty_source_kept() {
    // A present-but-empty source stays empty (not defaulted to `(?:)`);
    // the absent flags still stringify to `"undefined"`.
    assert_eq!(
        eval_str("RegExp.prototype.toString.call({source:''})"),
        "//undefined"
    );
}

// -- 7. `with` + for-in/of assignment (all four LHS forms) --------------------
// A `var`/bare pattern or identifier LHS assigns per iteration (spec
// `PutValue`), so an enclosing `with` object's properties are updated.
// `let`/`const` declare fresh per-iteration bindings and skip probing.

#[test]
fn with_for_of_bare_destructuring_assigns_to_object() {
    assert!(eval_bool(
        "var o={a:0}; with (o) { for ({a} of [{a:5}]) {} } o.a === 5;"
    ));
}

#[test]
fn with_for_of_var_destructuring_assigns_to_object() {
    assert!(eval_bool(
        "var o={a:0}; with (o) { for (var {a} of [{a:5}]) {} } o.a === 5;"
    ));
}

#[test]
fn with_for_of_bare_array_pattern_assigns_to_object() {
    assert!(eval_bool(
        "var o={b:0}; with (o) { for ([b] of [[7]]) {} } o.b === 7;"
    ));
}

#[test]
fn with_for_of_nested_pattern_assigns_to_object() {
    assert!(eval_bool(
        "var o={x:0,y:0}; with (o) { for ({x, y: y} of [{x:1,y:2}]) {} } o.x === 1 && o.y === 2;"
    ));
}

#[test]
fn with_for_in_bare_ident_assigns_to_object() {
    assert!(eval_bool(
        "var o={k:''}; with (o) { for (k in {x:1}) {} } o.k === 'x';"
    ));
}

#[test]
fn with_for_of_bare_destructuring_extracted() {
    // Extracted (closure in body) path stores wrapper-side, where the
    // `with` stack is intact.
    assert!(eval_bool(
        "var o={a:0}, fns=[]; \
         with (o) { for ({a} of [{a:5},{a:6}]) { let z = a; fns.push(() => z); } } \
         o.a === 6 && fns[0]() === 5 && fns[1]() === 6;"
    ));
}

#[test]
fn with_for_of_destructuring_miss_falls_through_to_var() {
    // Miss on the with-object: falls back to the declared `var` binding.
    assert!(eval_bool(
        "var o={}, a=0; with (o) { for ({a} of [{a:9}]) {} } a === 9 && !('a' in o);"
    ));
}

#[test]
fn with_for_of_let_destructuring_skips_object() {
    // `let`/`const` keep declaration mode: fresh binding, object untouched.
    assert!(eval_bool(
        "var o={a:0}, r=0; with (o) { for (let {a} of [{a:5}]) { r = a; } } r === 5 && o.a === 0;"
    ));
    assert!(eval_bool(
        "var o={a:0}, fns=[]; with (o) { for (const {a} of [{a:5}]) { fns.push(() => a); } } fns[0]() === 5 && o.a === 0;"
    ));
}

// -- 8. `with` + extracted loop bodies ---------------------------------------
// The synthetic `#for_body` arrow captures enclosing with-objects as
// upvalues and keeps probing (user closures still do not observe them).

#[test]
fn with_extracted_for_of_body_sees_object() {
    assert!(eval_bool(
        "var o={x:42}, fns=[]; \
         with (o) { for (let z of [1]) { let q = x; fns.push(() => q); } } \
         fns[0]() === 42;"
    ));
}

#[test]
fn with_extracted_for_of_body_store_hits_object() {
    // Store and load inside the extracted body both probe the
    // with-object; the closure captures the probed read (user closures
    // themselves still do not observe the with-object).
    assert!(eval_bool(
        "var o={x:0}, fns=[]; \
         with (o) { for (let z of [1]) { x = 7; let q = x; fns.push(() => q); } } \
         o.x === 7 && fns[0]() === 7;"
    ));
}

#[test]
fn with_extracted_for_of_inner_lexical_shadows_object() {
    assert!(eval_bool(
        "var o={x:1}, fns=[]; \
         with (o) { for (let z of [1]) { let x = 2; fns.push(() => x); } } \
         fns[0]() === 2 && o.x === 1;"
    ));
}

#[test]
fn with_extracted_for_of_nested_with_probes_innermost_first() {
    assert!(eval_bool(
        "var a={x:1}, b={x:2}, fns=[]; \
         with (a) { with (b) { for (let z of [1]) { let q = x; fns.push(() => q); } } } \
         fns[0]() === 2;"
    ));
}

#[test]
fn with_extracted_regular_for_body_sees_object() {
    assert!(eval_bool(
        "var o={x:5}, fns=[]; \
         with (o) { for (let i = 0; i < 1; i++) { let q = x; fns.push(() => q); } } \
         fns[0]() === 5;"
    ));
}

#[test]
fn with_extracted_for_in_body_sees_object() {
    assert!(eval_bool(
        "var o={v:3}, fns=[]; \
         with (o) { for (let k in {a:1}) { let q = v; fns.push(() => q); } } \
         fns[0]() === 3;"
    ));
}

// -- 9. `with` + `delete` ------------------------------------------------------

#[test]
fn with_delete_ident_removes_object_property() {
    assert!(eval_bool(
        "var o={x:1}, r=false; with (o) { r = delete x; } r === true && !('x' in o);"
    ));
}

#[test]
fn with_delete_ident_missing_is_true_noop() {
    assert!(eval_bool(
        "var o={}, r=false; with (o) { r = delete zzz; } r === true;"
    ));
}

#[test]
fn delete_ident_outside_with_still_noop_true() {
    // Pre-existing behavior: `delete x` outside `with` is a no-op `true`.
    assert!(eval_bool(
        "var x=1; var r = delete x; r === true && x === 1;"
    ));
}

// -- 10. `with` operand conversion ---------------------------------------------

#[test]
fn with_primitive_operand_throws_typeerror() {
    // Primitives are not boxed (v1 gap); the conversion never consults
    // the lexical `Object` binding.
    assert!(eval_bool(
        "var n=''; try { with (5) { 1; } } catch (e) { n = e.name; } n === 'TypeError';"
    ));
}

#[test]
fn with_shadowed_object_binding_not_consulted() {
    assert!(eval_bool(
        "var ok=false; { let Object = () => 42; try { with (5) { 1; } } catch (e) { ok = e instanceof TypeError; } } ok;"
    ));
}

#[test]
fn with_object_operand_ignores_shadowed_object_ctor() {
    assert!(eval_bool(
        "var o={x:9}, r=0; { let Object = () => 42; with (o) { r = x; } } r === 9;"
    ));
}

// -- 11. `RegExp.prototype.toString` receivers ----------------------------------

#[test]
fn regexp_tostring_primitive_receiver_throws() {
    // Spec 22.2.5.15 step 2: non-Object receivers throw `TypeError`.
    for recv in ["5", "'x'", "true"] {
        assert!(
            eval_bool(&format!(
                "var n=''; try {{ RegExp.prototype.toString.call({recv}); }} catch (e) {{ n = e.name; }} n === 'TypeError';"
            )),
            "receiver {recv} should throw TypeError"
        );
    }
    assert!(eval_bool(
        "var n=''; try { RegExp.prototype.toString.call(null); } catch (e) { n = e.name; } n === 'TypeError';"
    ));
    assert!(eval_bool(
        "var n=''; try { RegExp.prototype.toString.call(undefined); } catch (e) { n = e.name; } n === 'TypeError';"
    ));
}

#[test]
fn regexp_tostring_getters_on_receiver() {
    // `source`/`flags` go through `Get` (prototype chain included).
    assert_eq!(
        eval_str("RegExp.prototype.toString.call(Object.create({source:'p',flags:'g'}))"),
        "/p/g"
    );
}
