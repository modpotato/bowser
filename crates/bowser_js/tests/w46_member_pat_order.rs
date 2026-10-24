//! Wave-46 regression tests: ECMA-262 13.15.5.3
//! IteratorDestructuringAssignmentEvaluation order for member-expression
//! targets in array patterns.
//!
//! For a non-literal target, `lref = Evaluation(DestructuringAssignmentTarget)`
//! (base/key reference) must run *before* `IteratorValue` (the `next()`
//! call). V8-verified: `[base()[key()]] = iter` orders
//! `@@iterator,base,key,next`.

use bowser_js::{Value, Vm};

fn eval(src: &str) -> Result<Value, String> {
    Vm::new().eval(src).map_err(|e| format!("{e:?}"))
}

fn eval_true(src: &str) {
    let v = eval(src).unwrap_or_else(|e| panic!("eval failed for {src:?}: {e}"));
    assert!(
        matches!(v, Value::Bool(true)),
        "expected true, got {v:?} for {src:?}"
    );
}

/// `[base()[key()]] = iter` must evaluate `@@iterator`, then `base`,
/// then `key`, then `next()` — the member reference is resolved before
/// the iterator step.
#[test]
fn array_pattern_member_target_ref_before_next() {
    eval_true(
        "var log = []; var target = {}; \
         function base() { log.push('base'); return target; } \
         function key() { log.push('key'); return 'p'; } \
         var iter = {}; \
         iter[Symbol.iterator] = function() { \
           log.push('@@iterator'); \
           return { next: function() { log.push('next'); return {value: 7, done: false}; } }; \
         }; \
         [base()[key()]] = iter; \
         target.p === 7 && log.join(',') === '@@iterator,base,key,next';",
    );
}

/// Nested patterns: `[[base()[key()]]] = outer` — the inner member
/// reference is resolved before the *inner* iterator step.
#[test]
fn nested_array_pattern_member_target_ref_before_inner_next() {
    eval_true(
        "var log = []; var target = {}; \
         function base() { log.push('base'); return target; } \
         function key() { log.push('key'); return 'q'; } \
         var inner = {}; \
         inner[Symbol.iterator] = function() { \
           log.push('@@iterator'); \
           return { next: function() { log.push('next'); return {value: 42, done: false}; } }; \
         }; \
         var outer = {}; \
         outer[Symbol.iterator] = function() { \
           log.push('@@iterator'); \
           var n = 0; \
           return { next: function() { log.push('next'); n = n + 1; \
             return n === 1 ? {value: inner, done: false} : {done: true}; } }; \
         }; \
         [[base()[key()]]] = outer; \
         target.q === 42 \
           && log.join(',') === '@@iterator,next,@@iterator,base,key,next';",
    );
}

/// Member target with a default: `[base()[key()] = dflt()] = iter` —
/// reference before `next()`, default only after an `undefined` value.
#[test]
fn array_pattern_member_target_default_after_next() {
    eval_true(
        "var log = []; var target = {}; \
         function base() { log.push('base'); return target; } \
         function key() { log.push('key'); return 'r'; } \
         function dflt() { log.push('dflt'); return 99; } \
         var iter = {}; \
         iter[Symbol.iterator] = function() { \
           log.push('@@it'); \
           return { next: function() { log.push('next'); return {done: true}; } }; \
         }; \
         [base()[key()] = dflt()] = iter; \
         target.r === 99 && log.join(',') === '@@it,base,key,next,dflt';",
    );
}

/// `for ([base()[key()]] of outer)` — per-iteration member-target bind
/// keeps the same reference-before-`next()` order.
#[test]
fn for_of_array_pattern_member_target_ref_before_next() {
    eval_true(
        "var log = []; var target = {}; \
         function base() { log.push('base'); return target; } \
         function key() { log.push('key'); return 'k'; } \
         var inner = {}; \
         inner[Symbol.iterator] = function() { \
           log.push('inner-@@iterator'); \
           return { next: function() { log.push('inner-next'); return {value: 5, done: false}; } }; \
         }; \
         var outer = {}; \
         outer[Symbol.iterator] = function() { \
           log.push('@@iterator'); \
           var n = 0; \
           return { next: function() { log.push('next'); n = n + 1; \
             return n === 1 ? {value: inner, done: false} : {done: true}; } }; \
         }; \
         for ([base()[key()]] of outer) { } \
         target.k === 5 \
           && log.join(',') === '@@iterator,next,inner-@@iterator,base,key,inner-next,next';",
    );
}

/// A closure in a member target's base still compiles and evaluates
/// (scan_closure_pat must descend into member base/key expressions).
#[test]
fn member_target_base_may_contain_closure() {
    eval_true(
        "var target = {}; \
         [(function() { return target; })().p] = [11]; \
         target.p === 11;",
    );
}
