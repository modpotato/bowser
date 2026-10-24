//! Wave-46: trampoline the `ToPrimitive` conversions in the
//! bitwise/shift/unary/relational/equality ops (Phase B2 remainder).
//!
//! Wave 45 staged `ToPrimitive` hook calls for the six arithmetic ops;
//! this wave extends the same [`SiteKind::ToPrimitive`] machinery (via the
//! generalized [`ToPrimKind`] continuation) to `&`/`|`/`^`, `<<`/`>>`/`>>>`,
//! unary `-`/`+`/`~`, `<`/`<=`/`>`/`>=`, and `==`/`!=`. A hook with side
//! effects inside any of these must no longer nest a native `run()` frame
//! per call: deep hook recursion that previously raised a catchable stack
//! error at ~150 levels now rides the single dispatch loop.

use bowser_js::{Value, Vm};

fn eval(src: &str) -> Result<Value, String> {
    Vm::new().eval(src).map_err(|e| format!("{e:?}"))
}

fn eval_ok(src: &str) -> Value {
    eval(src).unwrap_or_else(|e| panic!("eval failed for {src:?}: {e}"))
}

fn eval_str(src: &str) -> String {
    match eval_ok(src) {
        Value::Str(s) => s.as_wtf8().to_string(),
        v => panic!("expected string for {src:?}, got {v:?}"),
    }
}

// -- deep hook recursion rides the trampoline -------------------------------

/// Deep `valueOf` recursion through `&` rides the site-call trampoline:
/// each hook becomes a frame on the one dispatch loop instead of a nested
/// native `run()`.
#[test]
fn trampolined_toprimitive_deep_recursion_bitand() {
    let v = eval_ok(
        "var n = 0; \
         var o = { valueOf: function () { n++; if (n < 3000) return o & 0xFFFF; return n; } }; \
         o & 0xFFFF;",
    );
    assert!(matches!(v, Value::Int(3000)), "unexpected {v:?}");
}

/// Deep `valueOf` recursion through `|`.
#[test]
fn trampolined_toprimitive_deep_recursion_bitor() {
    let v = eval_ok(
        "var n = 0; \
         var o = { valueOf: function () { n++; if (n < 3000) return o | 0; return n; } }; \
         o | 0;",
    );
    assert!(matches!(v, Value::Int(3000)), "unexpected {v:?}");
}

/// Deep `valueOf` recursion through `^`.
#[test]
fn trampolined_toprimitive_deep_recursion_bitxor() {
    let v = eval_ok(
        "var n = 0; \
         var o = { valueOf: function () { n++; if (n < 3000) return o ^ 0; return n; } }; \
         o ^ 0;",
    );
    assert!(matches!(v, Value::Int(3000)), "unexpected {v:?}");
}

/// Deep `valueOf` recursion through `<<`.
#[test]
fn trampolined_toprimitive_deep_recursion_shl() {
    let v = eval_ok(
        "var n = 0; \
         var o = { valueOf: function () { n++; if (n < 3000) return o << 0; return n; } }; \
         o << 0;",
    );
    assert!(matches!(v, Value::Int(3000)), "unexpected {v:?}");
}

/// Deep `valueOf` recursion through `>>`.
#[test]
fn trampolined_toprimitive_deep_recursion_shr() {
    let v = eval_ok(
        "var n = 0; \
         var o = { valueOf: function () { n++; if (n < 3000) return o >> 0; return n; } }; \
         o >> 0;",
    );
    assert!(matches!(v, Value::Int(3000)), "unexpected {v:?}");
}

/// Deep `valueOf` recursion through `>>>`.
#[test]
fn trampolined_toprimitive_deep_recursion_shru() {
    let v = eval_ok(
        "var n = 0; \
         var o = { valueOf: function () { n++; if (n < 3000) return o >>> 0; return n; } }; \
         o >>> 0;",
    );
    assert!(matches!(v, Value::Int(3000)), "unexpected {v:?}");
}

/// Deep `@@toPrimitive` recursion through unary `-`: the nested op's value
/// is discarded and each hook returns a constant, so the assertion is on
/// the hook count plus the constant — the point is the 3000-deep staging
/// completes at all.
#[test]
fn trampolined_toprimitive_deep_recursion_unary_neg() {
    let v = eval_ok(
        "var n = 0; \
         function mk(depth) { \
           var o = {}; \
           Object.defineProperty(o, Symbol.toPrimitive, { value: function (h) { \
             n++; \
             if (depth > 0) { -(mk(depth - 1)); } \
             return 7; \
           } }); \
           return o; \
         } \
         -(mk(3000)) + n;",
    );
    assert!(matches!(v, Value::Int(2994)), "unexpected {v:?}");
}

/// Deep `valueOf` recursion through unary `+`.
#[test]
fn trampolined_toprimitive_deep_recursion_unary_pos() {
    let v = eval_ok(
        "var n = 0; \
         var o = { valueOf: function () { n++; if (n < 3000) return +o; return n; } }; \
         +o;",
    );
    assert!(matches!(v, Value::Int(3000)), "unexpected {v:?}");
}

/// Deep `@@toPrimitive` recursion through unary `~`.
#[test]
fn trampolined_toprimitive_deep_recursion_unary_bitnot() {
    let v = eval_ok(
        "var n = 0; \
         function mk(depth) { \
           var o = {}; \
           Object.defineProperty(o, Symbol.toPrimitive, { value: function (h) { \
             n++; \
             if (depth > 0) { ~(mk(depth - 1)); } \
             return 7; \
           } }); \
           return o; \
         } \
         ~(mk(3000)) + n;",
    );
    // ~7 === -8; 3001 hook invocations.
    assert!(matches!(v, Value::Int(2993)), "unexpected {v:?}");
}

/// Deep `valueOf` recursion through `<`.
#[test]
fn trampolined_toprimitive_deep_recursion_rel_lt() {
    let v = eval_ok(
        "var n = 0; \
         var o = { valueOf: function () { n++; if (n < 3000) return (o < 1000000); return n; } }; \
         o < 1000000;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

/// Deep `valueOf` recursion through `==`: the equality loop re-enters
/// from the staged continuation instead of nesting `invoke_value`.
#[test]
fn trampolined_toprimitive_deep_recursion_eq() {
    let v = eval_ok(
        "var n = 0; \
         var o = { valueOf: function () { n++; if (n < 3000) return (o == 1); return 1; } }; \
         o == 1;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

/// Deep `@@toPrimitive` recursion through `!=`.
#[test]
fn trampolined_toprimitive_deep_recursion_ne() {
    let v = eval_ok(
        "var n = 0; \
         function mk(depth) { \
           var o = {}; \
           Object.defineProperty(o, Symbol.toPrimitive, { value: function (h) { \
             n++; \
             if (depth > 0) { mk(depth - 1) != 1; } \
             return 5; \
           } }); \
           return o; \
         } \
         (mk(3000) != 1) ? n : -1;",
    );
    assert!(matches!(v, Value::Int(3001)), "unexpected {v:?}");
}

// -- hints, staging protocol, error paths ----------------------------------

/// Hint selection across the newly trampolined ops: number for
/// bitwise/shift/unary/relational, default for equality.
#[test]
fn trampolined_toprimitive_ops_hint_selection() {
    let s = eval_str(
        "var seen = []; \
         var o = {}; \
         Object.defineProperty(o, Symbol.toPrimitive, { value: function (h) { seen.push(h); return 3; } }); \
         var r = (o & 1) + ',' + (o << 1) + ',' + (-o) + ',' + (o < 10) + ',' + (o == 3); \
         r + '=' + seen.join(',');",
    );
    assert_eq!(s, "1,6,-3,true,true=number,number,number,number,default");
}

/// `valueOf` returning an object falls through to a *staged* `toString`:
/// the chained staging completes and the placeholder never leaks into the
/// result.
#[test]
fn trampolined_toprimitive_bitand_method_fallthrough_staged() {
    let s = eval_str(
        "var calls = []; \
         var o = { valueOf: function () { calls.push('v'); return {}; }, \
                   toString: function () { calls.push('s'); return '6'; } }; \
         (o & 1) + ',' + calls.join('');",
    );
    assert_eq!(s, "0,vs");
}

/// Same fallthrough through the equality loop: the converted primitive
/// feeds back into `loose_eq` and the comparison completes.
#[test]
fn trampolined_toprimitive_eq_method_fallthrough_staged() {
    let v = eval_ok(
        "var o = { valueOf: function () { return {}; }, \
                   toString: function () { return '6'; } }; \
         o == '6';",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

/// A staged `@@toPrimitive` returning an object is a catchable `TypeError`
/// (the staged-path validation in the continuation).
#[test]
fn trampolined_toprimitive_bitand_hook_object_throws() {
    let s = eval_str(
        "var o = {}; \
         Object.defineProperty(o, Symbol.toPrimitive, { value: function (h) { return {}; } }); \
         var c = 'none'; \
         try { o & 1; } catch (e) { c = (e instanceof TypeError) ? 'type' : 'other'; } \
         c;",
    );
    assert_eq!(s, "type");
}

/// Same validation through the equality loop.
#[test]
fn trampolined_toprimitive_eq_hook_object_throws() {
    let s = eval_str(
        "var o = {}; \
         Object.defineProperty(o, Symbol.toPrimitive, { value: function (h) { return {}; } }); \
         var c = 'none'; \
         try { o == 1; } catch (e) { c = (e instanceof TypeError) ? 'type' : 'other'; } \
         c;",
    );
    assert_eq!(s, "type");
}

/// Chained staging for `&`: both operands need hooks, so the continuation
/// stages the rhs hook after the lhs hook completes.
#[test]
fn trampolined_toprimitive_bitand_both_operands_hook() {
    let s = eval_str(
        "var a = { valueOf: function () { return 6; } }; \
         var b = { valueOf: function () { return 7; } }; \
         (a & b) + ',' + (b | a);",
    );
    assert_eq!(s, "6,7");
}

/// Chained staging for `<`: both operands need hooks.
#[test]
fn trampolined_toprimitive_rel_both_operands_hook() {
    let v = eval_ok(
        "var a = { valueOf: function () { return 6; } }; \
         var b = { valueOf: function () { return 7; } }; \
         (a < b) && (b > a) && (a <= 6) && (b >= 7);",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

/// A throwing `valueOf` staged from `<<` unwinds through the trampoline
/// and is catchable.
#[test]
fn trampolined_toprimitive_shift_throw_caught() {
    let s = eval_str(
        "var o = { valueOf: function () { throw new Error('boom'); } }; \
         var c = 'none'; \
         try { o << 1; } catch (e) { c = e.message; } \
         c;",
    );
    assert_eq!(s, "boom");
}

/// A throwing `valueOf` staged from `==` unwinds and is catchable.
#[test]
fn trampolined_toprimitive_eq_throw_caught() {
    let s = eval_str(
        "var o = { valueOf: function () { throw new Error('boom'); } }; \
         var c = 'none'; \
         try { o == 1; } catch (e) { c = e.message; } \
         c;",
    );
    assert_eq!(s, "boom");
}

// -- semantics parity --------------------------------------------------------

/// Equality of two objects never calls valueOf/toString: per ECMA-262 §7.2.14
/// step 1, Object-vs-Object reduces to strict (reference) equality, so no
/// ToPrimitive hook runs on either side.
#[test]
fn loose_eq_object_sides_skip_toprimitive() {
    let s = eval_str(
        "var log = []; \
         var a = { valueOf: function () { log.push('a'); return 1; } }; \
         var b = { valueOf: function () { log.push('b'); return 2; } }; \
         (a == b) + ':' + log.join('');",
    );
    assert_eq!(s, "false:");
}

/// Object-vs-Object `==`/`!=` never runs ToPrimitive hooks: ECMA-262
/// §7.2.14 step 1 reduces to strict (reference) equality, so two distinct
/// plain objects are unequal and `!=` is their negation with no hook
/// invocations.
#[test]
fn loose_eq_object_sides_never_toprimitive() {
    let s = eval_str("var a = {}; var b = {}; (a == b) + ':' + (a != b);");
    assert_eq!(s, "false:true");
}

/// `~` on a hook-returned string: the primitive tail applies.
#[test]
fn unary_bitnot_hook_string() {
    let v = eval_ok("var o = { toString: function () { return '5'; } }; ~o;");
    assert!(matches!(v, Value::Int(-6)), "unexpected {v:?}");
}

/// `>>>` on a hook-returned `-1`: zero-fill shift of the primitive tail.
#[test]
fn shift_shru_hook_negative_one() {
    let v = eval_ok("var o = { valueOf: function () { return -1; } }; o >>> 0;");
    assert!(
        matches!(v, Value::Float(f) if f == 4294967295.0),
        "unexpected {v:?}"
    );
}

/// Relational string comparison through hooks: both convert, then the
/// borrowed-view string path applies.
#[test]
fn rel_lt_hook_strings() {
    let v = eval_ok(
        "var a = { toString: function () { return 'a'; } }; \
         var b = { toString: function () { return 'b'; } }; \
         (a < b) && (b > a) && !(a > b);",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

/// `!=` negation applies on the synchronous path too.
#[test]
fn ne_negation_sync_path() {
    let v = eval_ok(
        "var o = { valueOf: function () { return 1; } }; \
         (o != 2) && !(o != 1) && (o == 1);",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

/// Hook evaluation order: lhs converts before rhs for `&`.
#[test]
fn bitand_hook_evaluation_order() {
    let s = eval_str(
        "var log = []; \
         var a = { valueOf: function () { log.push('a'); return 1; } }; \
         var b = { valueOf: function () { log.push('b'); return 2; } }; \
         (a & b) + ':' + log.join('');",
    );
    assert_eq!(s, "0:ab");
}

/// Unary `-` preserves the `-0` rule from the old path.
#[test]
fn unary_neg_negative_zero() {
    let v = eval_ok("var o = { valueOf: function () { return -0; } }; -o;");
    assert!(
        matches!(v, Value::Float(f) if f == 0.0 && !f.is_sign_negative()),
        "unexpected {v:?}"
    );
}
