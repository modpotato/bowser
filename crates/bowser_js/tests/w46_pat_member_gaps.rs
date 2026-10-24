//! Wave-46 regression tests: member-expression destructuring targets in
//! object patterns and array rest elements (ECMA-262 13.15.5.2/13.15.5.3),
//! plus eager `@@iterator()` result validation.
//!
//! Gap A: in an object pattern, `lref = Evaluation(
//! DestructuringAssignmentTarget)` runs *before* `GetV(obj, key)` (and the
//! `PropertyName` evaluation, including `ToPropertyKey`, runs before
//! `lref`). V8-verified: `({[sourceKey()]: target()[targetKey()]} =
//! source())` orders `source,source-key,source-key-tostring,target,
//! target-key,get`.
//!
//! Gap B: the eager iterator acquisition (first array element is a member
//! target) must validate the `@@iterator()` result is an object, throwing
//! `TypeError` otherwise, exactly like the normal `IterNext` path.
//!
//! Gap C: for an array rest member target (`[...base()[key()]] = iter`)
//! the reference evaluates before the rest-collection loop's `next()`
//! calls: `@@iterator,base,key,next,next,next`.

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

/// Assert that `src` throws a `TypeError` (uncaught).
fn assert_type_error(src: &str) {
    let v = eval(&format!(
        "var caught = null; try {{ {src} }} catch (e) {{ caught = e; }} \
         caught instanceof TypeError;"
    ))
    .unwrap_or_else(|e| panic!("eval failed for {src:?}: {e}"));
    assert!(
        matches!(v, Value::Bool(true)),
        "expected TypeError for {src:?}, got {v:?}"
    );
}

// -- Gap A: object-pattern member targets -----------------------------------

/// `({[sourceKey()]: target()[targetKey()]} = source())`: the key (with
/// `ToPropertyKey`) evaluates before the member reference, which
/// evaluates before the property read.
#[test]
fn object_pattern_member_target_ref_before_get() {
    eval_true(
        "var log = []; var target = {}; \
         function source() { log.push('source'); return src; } \
         function sourceKey() { log.push('source-key'); \
           return { toString: function() { log.push('source-key-tostring'); return 'sk'; } }; } \
         function targetFn() { log.push('target'); return target; } \
         function targetKey() { log.push('target-key'); return 'tk'; } \
         var src = {}; \
         Object.defineProperty(src, 'sk', { enumerable: true, configurable: true, \
           get: function() { log.push('get'); return 42; } }); \
         ({[sourceKey()]: targetFn()[targetKey()]} = source()); \
         target.tk === 42 \
           && log.join(',') === 'source,source-key,source-key-tostring,target,target-key,get';",
    );
}

/// Nested object patterns: the inner member reference resolves before the
/// inner `GetV`, while the outer (pattern-valued) read keeps its
/// GetV-first order.
#[test]
fn nested_object_pattern_member_target_ref_before_inner_get() {
    eval_true(
        "var log = []; var target = {}; \
         function src() { log.push('src'); \
           return { get k1() { log.push('get1'); \
             return { get k2() { log.push('get2'); return 9; } }; } }; } \
         function k1() { log.push('k1'); \
           return { toString: function() { log.push('k1ts'); return 'k1'; } }; } \
         function k2() { log.push('k2'); \
           return { toString: function() { log.push('k2ts'); return 'k2'; } }; } \
         function targetFn() { log.push('target'); return target; } \
         function targetKey() { log.push('target-key'); return 'tk'; } \
         ({[k1()]: {[k2()]: targetFn()[targetKey()]}} = src()); \
         target.tk === 9 \
           && log.join(',') === 'src,k1,k1ts,get1,k2,k2ts,target,target-key,get2';",
    );
}

/// Member target with a default in an object pattern: the default still
/// runs after the read, only when the value is `undefined`.
#[test]
fn object_pattern_member_target_default_after_get() {
    eval_true(
        "var log = []; var target = {}; \
         function k() { log.push('k'); return 'q'; } \
         function targetFn() { log.push('target'); return target; } \
         function targetKey() { log.push('target-key'); return 'tk'; } \
         function dflt() { log.push('dflt'); return 77; } \
         var src = {}; \
         ({[k()]: targetFn()[targetKey()] = dflt()} = src); \
         target.tk === 77 \
           && log.join(',') === 'k,target,target-key,dflt';",
    );
}

// -- Gap B: eager @@iterator() result validation ----------------------------

/// `a[Symbol.iterator] = () => 5; [o.p] = a` must throw `TypeError`,
/// mirroring the normal path's GetIterator validation.
#[test]
fn eager_iterator_non_object_result_throws() {
    assert_type_error(
        "var a = [1,2,3]; a[Symbol.iterator] = function() { return 5; }; \
         var o = {}; [o.p] = a;",
    );
}

/// `null` is not an object either.
#[test]
fn eager_iterator_null_result_throws() {
    assert_type_error(
        "var a = [1,2,3]; a[Symbol.iterator] = function() { return null; }; \
         var o = {}; [o.p] = a;",
    );
}

/// Nested shape `[[o.p]] = [a]`: the inner pattern's eager acquisition
/// hits the same validation.
#[test]
fn nested_eager_iterator_non_object_result_throws() {
    assert_type_error(
        "var a = [1,2,3]; a[Symbol.iterator] = function() { return 'nope'; }; \
         var o = {}; [[o.p]] = [a];",
    );
}

/// `for ([o.p] of [a])`: per-iteration binding shares the eager path.
#[test]
fn for_of_eager_iterator_non_object_result_throws() {
    assert_type_error(
        "var a = [1,2,3]; a[Symbol.iterator] = function() { return 5; }; \
         var o = {}; for ([o.p] of [a]) { }",
    );
}

/// Sanity: a well-behaved `@@iterator` still binds through the eager
/// path (the validation must not reject real iterators).
#[test]
fn eager_iterator_object_result_still_binds() {
    eval_true(
        "var a = [1,2,3]; \
         a[Symbol.iterator] = function() { \
           var n = 0; \
           return { next: function() { n = n + 1; \
             return n === 1 ? {value: 11, done: false} : {done: true}; } }; \
         }; \
         var o = {}; [o.p] = a; o.p === 11;",
    );
}

// -- Gap C: array rest with member target ------------------------------------

/// `[...base()[key()]] = iter`: the member reference resolves before the
/// rest-collection loop's `next()` calls.
#[test]
fn array_rest_member_target_ref_before_nexts() {
    eval_true(
        "var log = []; var target = {}; \
         function base() { log.push('base'); return target; } \
         function key() { log.push('key'); return 'p'; } \
         var iter = {}; \
         iter[Symbol.iterator] = function() { \
           log.push('@@iterator'); \
           var n = 0; \
           return { next: function() { log.push('next'); n = n + 1; \
             return n <= 2 ? {value: n, done: false} : {done: true}; } }; \
         }; \
         [...base()[key()]] = iter; \
         target.p.length === 2 && target.p[0] === 1 && target.p[1] === 2 \
           && log.join(',') === '@@iterator,base,key,next,next,next';",
    );
}

/// Rest member target after a leading element: the reference resolves
/// after the first element's `next()` but before the rest loop's.
#[test]
fn array_rest_member_target_after_leading_element() {
    eval_true(
        "var log = []; var target = {}; \
         function base() { log.push('base'); return target; } \
         function key() { log.push('key'); return 'p'; } \
         var iter = {}; \
         iter[Symbol.iterator] = function() { \
           log.push('@@iterator'); \
           var n = 0; \
           return { next: function() { log.push('next'); n = n + 1; \
             return n <= 3 ? {value: n, done: false} : {done: true}; } }; \
         }; \
         var first; [first, ...base()[key()]] = iter; \
         first === 1 && target.p.length === 2 \
           && log.join(',') === '@@iterator,next,base,key,next,next,next';",
    );
}

/// Object rest member target `{...base()[key()]}`: reference before the
/// property copy (unobservable in this engine, kept for spec shape).
#[test]
fn object_rest_member_target_ref_before_copy() {
    eval_true(
        "var log = []; var target = {}; \
         function base() { log.push('base'); return target; } \
         function key() { log.push('key'); return 'p'; } \
         function src() { log.push('src'); return {a: 1, b: 2}; } \
         ({...base()[key()]} = src()); \
         target.p.a === 1 && target.p.b === 2 \
           && log.join(',') === 'src,base,key';",
    );
}
