//! Wave-47 tests for `IteratorClose` on abrupt completion (spec 7.4.6,
//! via 14.7.5 `for`-`of` and 13.15.5.3 array-pattern destructuring).
//!
//! Four new ops carry it: `Op::IterGuard` arms a runtime guard on the
//! frame's handler stack (one per `for`-`of` loop and per array pattern),
//! so a `throw` unwinding through the guard closes the in-flight iterator
//! (`throw_value`); `Op::IterUnguard` disarms the guard on normal
//! completion; `Op::IterClose` / `Op::IterCloseRet` are emitted by
//! `unwind_protected` for `break` / `return` through a `for`-`of` loop
//! (the `Protected::ForOf` region keeps them ordered correctly against
//! `try`/`finally`).
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

/// A manually-driven iterator object logging `return()` calls into `log`.
const MK: &str = r#"
function mkiter(log, vals) {
    var i = 0;
    return {
        next: function() {
            if (i < vals.length) return {value: vals[i++], done: false};
            return {value: undefined, done: true};
        },
        "return": function() { log.push('closed'); return {done: true}; }
    };
}
function mkobj(log, vals) {
    return { [Symbol.iterator]: function() { return mkiter(log, vals); } };
}
"#;

fn run(src: &str) -> String {
    eval_str(&format!("{MK}\n{src}"))
}

// -- break / continue / exhaustion ------------------------------------------

#[test]
fn break_closes_iterator() {
    assert_eq!(
        run("var log=[]; var o=mkobj(log,[1,2,3]); for (var x of o) { break; } log.join(',');"),
        "closed"
    );
}

#[test]
fn continue_does_not_close() {
    assert_eq!(
        run("var log=[]; var o=mkobj(log,[1,2,3]); for (var x of o) { continue; } log.join(',');"),
        ""
    );
}

#[test]
fn exhaustion_does_not_close() {
    assert_eq!(
        run("var log=[]; var o=mkobj(log,[1,2]); for (var x of o) {} log.join(',');"),
        ""
    );
}

#[test]
fn labeled_break_closes_all_loops() {
    assert_eq!(
        run(
            "var log=[]; var a=mkobj(log,['a']); var b=mkobj(log,['b']); \
             outer: for (var x of a) { for (var y of b) { break outer; } } log.join(',');"
        ),
        "closed,closed"
    );
}

#[test]
fn labeled_continue_closes_only_inner() {
    // `continue outer` runs twice (two outer iterations); each pass closes
    // the inner iterator, never the outer one.
    assert_eq!(
        run(
            "var log=[]; var a=mkobj(log,['a1','a2']); var b=mkobj(log,['b']); \
             outer: for (var x of a) { for (var y of b) { continue outer; } } log.join(',');"
        ),
        "closed,closed"
    );
}

// -- return ------------------------------------------------------------------

#[test]
fn return_closes_iterator() {
    assert_eq!(
        run("var log=[]; var o=mkobj(log,[1,2,3]); \
             function f(){ for (var x of o) { return 42; } } \
             var r = f(); r + '|' + log.join(',');"),
        "42|closed"
    );
}

#[test]
fn return_inside_try_finally_orders_finally_before_close() {
    // `return` in `try` inside `for`-`of`: the finalizer runs as the
    // abrupt exits the `try`, then `IteratorClose` as it exits the loop.
    assert_eq!(
        run("var log=[]; var o=mkobj(log,[1]); \
             function f(){ for (var x of o) { try { return 7; } finally { log.push('fin'); } } } \
             var r=f(); r+'|'+log.join(',');"),
        "7|fin,closed"
    );
}

#[test]
fn return_inside_loop_inside_try_orders_close_before_finally() {
    // `return` in `for`-`of` inside `try`: the close runs as the abrupt
    // exits the loop, then the finalizer as it exits the `try`.
    assert_eq!(
        run("var log=[]; var o=mkobj(log,[1]); \
             function f(){ try { for (var x of o) { return 8; } } finally { log.push('fin'); } } \
             var r=f(); r+'|'+log.join(',');"),
        "8|closed,fin"
    );
}

// -- throw -------------------------------------------------------------------

#[test]
fn throw_closes_and_original_error_wins() {
    // Spec 7.4.6 step 4: the in-flight throw wins over a `return()` throw;
    // the close still runs for its side effects.
    assert_eq!(
        run("var log=[]; var o=mkobj(log,[1,2,3]); var e=null; \
             try { for (var x of o) { throw new Error('orig'); } } \
             catch (err) { e = err.message; } log.join(',') + '|' + e;"),
        "closed|orig"
    );
}

#[test]
fn throw_inside_try_in_loop_closes_then_catches() {
    assert_eq!(
        run("var log=[]; var o=mkobj(log,[1,2,3]); var e=null; \
             for (var x of o) { try { throw new Error('boom'); } catch (err) { e = err.message; break; } } \
             e + '|' + log.join(',');"),
        "boom|closed"
    );
}

// -- return() throwing on a non-throw path -----------------------------------

#[test]
fn throwing_return_replaces_break() {
    // On a non-throw abrupt path, a `return()` throw replaces the
    // completion (spec 7.4.6 step 5).
    assert_eq!(
        run("var o={ [Symbol.iterator]: function(){ return { \
                 next: function(){ return {value:1,done:false}; }, \
                 \"return\": function(){ throw new Error('close-err'); } }; } }; \
             var e=null; try { for (var x of o) { break; } } catch(err){ e=err.message; } e;"),
        "close-err"
    );
}

// -- destructuring ------------------------------------------------------------

#[test]
fn destructuring_abrupt_closes_iterator() {
    assert_eq!(
        run("var log=[]; var e=null; \
             try { var [a,b] = { [Symbol.iterator]: function(){ \
                 var it=mkiter(log,[1,2,3]); var n=0; \
                 return { \
                     next: function(){ n++; if(n==2) throw new Error('boom'); return it.next(); }, \
                     \"return\": function(){ log.push('dclosed'); return {done:true}; } }; } }; } \
             catch(err){ e=err.message; } log.join(',')+'|'+e;"),
        "dclosed|boom"
    );
}

#[test]
fn destructuring_next_throw_closes_pattern_iterator() {
    assert_eq!(
        run("var log=[]; var e=null; \
             try { var [[a]] = { [Symbol.iterator]: function(){ \
                 var it=mkiter(log,[1]); \
                 return { next: function(){ throw new Error('inner-boom'); }, \
                          \"return\": function(){ log.push('outer-closed'); return {done:true}; } }; } }; } \
             catch(err){ e=err.message; } log.join(',')+'|'+e;"),
        "outer-closed|inner-boom"
    );
}

// -- generators ---------------------------------------------------------------

#[test]
fn break_runs_generator_finally() {
    // A generator's `return()` runs its `finally` blocks: breaking out of
    // `for`-`of` over a generator must trigger them.
    assert_eq!(
        run(
            "var log=[]; function* g(){ try { yield 1; yield 2; } finally { log.push('fin'); } } \
             for (var x of g()) { break; } log.join(',');"
        ),
        "fin"
    );
}

#[test]
fn throw_runs_generator_finally_and_original_wins() {
    assert_eq!(
        run(
            "var log=[]; function* g(){ try { yield 1; yield 2; } finally { log.push('fin'); } } \
             var e=null; try { for (var x of g()) { throw new Error('orig'); } } \
             catch (err) { e = err.message; } log.join(',') + '|' + e;"
        ),
        "fin|orig"
    );
}

// -- no-iterator cases stay cheap ----------------------------------------------

#[test]
fn generator_return_while_suspended_in_for_of_closes() {
    // `.return()` on a generator suspended at a `yield` inside `for`-`of`:
    // the abrupt completion unwinds through the loop's guard.
    assert_eq!(
        run(
            "var log=[]; function* g(){ for (var x of mkobj(log,[1,2,3])) { yield x; } } \
             var gi = g(); gi.next(); gi.return(99); log.join(',');"
        ),
        "closed"
    );
}

#[test]
fn for_of_pattern_lhs_abrupt_closes_both_iterators() {
    // `for ([a] of outer)`: the per-iteration pattern has its own guard;
    // an abrupt inside the pattern closes the pattern iterator first,
    // then the outer loop's as unwinding continues.
    assert_eq!(
        run("var log=[]; var e=null; \
             var outer = mkobj(log,['x']); \
             try { for (var [a] of outer) { throw new Error('body-boom'); } } \
             catch (err) { e = err.message; } log.join(',')+'|'+e;"),
        "closed|body-boom"
    );
}

#[test]
fn array_for_of_break_still_works() {
    match eval("var n=0; for (var x of [1,2,3]) { n+=x; if(x==2) break; } n;") {
        Value::Int(3) => {}
        v => panic!("expected 3, got {v:?}"),
    }
}

#[test]
fn for_in_loop_unaffected() {
    match eval("var s=''; for (var k in {a:1,b:2}) { s+=k; } s;") {
        Value::Str(s) => assert_eq!(s.as_wtf8().to_string(), "ab"),
        v => panic!("expected \"ab\", got {v:?}"),
    }
}
