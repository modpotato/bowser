//! Classic-script global bindings: top-level `function` and `var`
//! declarations of a script run through [`Vm::eval`] must become own
//! properties of the global object (ECMA-262 GlobalDeclarationInstantiation).
//!
//! These drive the public [`bowser_js::Vm`] API across *multiple* evals on
//! one `Vm`, since the bug was that each eval's frame-local bindings were
//! discarded when the eval returned.

use bowser_js::{Value, Vm};

fn eval_on(vm: &mut Vm, src: &str) -> Value {
    vm.eval(src)
        .unwrap_or_else(|e| panic!("eval failed for {src:?}: {e:?}"))
}

fn eval_str(vm: &mut Vm, src: &str) -> String {
    match eval_on(vm, src) {
        Value::Str(s) => s.as_wtf8().to_string(),
        v => panic!("expected string for {src:?}, got {v:?}"),
    }
}

fn eval_int(vm: &mut Vm, src: &str) -> i32 {
    match eval_on(vm, src) {
        Value::Int(i) => i,
        v => panic!("expected int for {src:?}, got {v:?}"),
    }
}

// -- sloppy function declarations ------------------------------------------

#[test]
fn sloppy_function_decl_becomes_global() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "function foo(){return 42}");
    // Visible to later scripts by bare name ...
    assert_eq!(eval_int(&mut vm, "foo()"), 42);
    assert_eq!(eval_str(&mut vm, "typeof foo"), "function");
    // ... and as a `globalThis` property.
    assert_eq!(eval_str(&mut vm, "typeof globalThis.foo"), "function");
    assert_eq!(eval_int(&mut vm, "globalThis.foo()"), 42);
}

#[test]
fn function_decl_visible_on_global_this_within_same_eval() {
    let mut vm = Vm::new();
    let t = eval_str(&mut vm, "function bar(){}; typeof globalThis.bar");
    assert_eq!(t, "function");
}

// -- sloppy var declarations ------------------------------------------------

#[test]
fn sloppy_var_becomes_global() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "var x = 1");
    assert_eq!(eval_int(&mut vm, "x"), 1);
    assert_eq!(eval_int(&mut vm, "globalThis.x"), 1);
}

#[test]
fn var_redeclaration_across_evals_updates_global() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "var z = 1");
    eval_on(&mut vm, "var z = 2");
    assert_eq!(eval_int(&mut vm, "z"), 2);
    assert_eq!(eval_int(&mut vm, "globalThis.z"), 2);
}

#[test]
fn var_without_initializer_keeps_existing_global_value() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "var w = 5");
    // A bare redeclaration must not clobber the existing global.
    eval_on(&mut vm, "var w;");
    assert_eq!(eval_int(&mut vm, "w"), 5);
}

// -- strict mode ------------------------------------------------------------

#[test]
fn strict_var_becomes_global() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "'use strict'; var y = 2");
    // ECMA-262 GlobalDeclarationInstantiation has no strictness branch:
    // strict top-level `var`s are globals too.
    assert_eq!(eval_str(&mut vm, "typeof globalThis.y"), "number");
    // ... and the name resolves in a later script.
    assert_eq!(eval_int(&mut vm, "y"), 2);
    assert_eq!(eval_int(&mut vm, "globalThis.y"), 2);
}

#[test]
fn strict_var_visible_across_scripts() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "'use strict'; var sy = 21;");
    // A later sloppy script sees the strict script's global ...
    assert_eq!(eval_int(&mut vm, "sy * 2"), 42);
    // ... and vice versa: a later strict script sees a sloppy global.
    eval_on(&mut vm, "var sl = 10;");
    assert_eq!(eval_int(&mut vm, "'use strict'; sl + sy"), 31);
}

#[test]
fn strict_function_decl_becomes_global() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "'use strict'; function baz(){return 7}");
    assert_eq!(eval_str(&mut vm, "typeof baz"), "function");
    assert_eq!(eval_str(&mut vm, "typeof globalThis.baz"), "function");
    assert_eq!(eval_int(&mut vm, "baz()"), 7);
}

// -- write-through during the run -------------------------------------------

#[test]
fn in_eval_global_this_reads_after_compound_assignment() {
    let mut vm = Vm::new();
    let s = eval_str(&mut vm, "var s = 'a'; s += 'b'; globalThis.s");
    assert_eq!(s, "ab");
}

#[test]
fn nested_direct_eval_assignment_visible_on_global_this() {
    let mut vm = Vm::new();
    let x = eval_int(&mut vm, "var x = 1; eval('x = 2'); globalThis.x");
    assert_eq!(x, 2);
}

#[test]
fn closure_assignment_syncs_on_post_run_install() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "var c = 1; function bump(){ c = 41; }");
    eval_on(&mut vm, "bump()");
    // The closure wrote through the shared cell; the post-run install
    // copies the final value out to the global object.
    assert_eq!(eval_int(&mut vm, "globalThis.c"), 41);
    assert_eq!(eval_int(&mut vm, "c"), 41);
}

// -- interaction between declarations ---------------------------------------

#[test]
fn function_and_var_sharing_a_name_install_once() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "var f; function f(){return 9}");
    // The function declaration wins the shared binding.
    assert_eq!(eval_str(&mut vm, "typeof f"), "function");
    assert_eq!(eval_int(&mut vm, "f()"), 9);
    assert_eq!(eval_str(&mut vm, "typeof globalThis.f"), "function");
}
// -- accessor globals are passthrough bindings --------------------------------

/// Define an accessor global `acc` plus call counters, in one setup script.
fn setup_accessor_global(vm: &mut Vm) {
    eval_on(
        vm,
        r#"
        var getterCalls = 0;
        var setterCalls = 0;
        var lastSet = null;
        Object.defineProperty(globalThis, "acc", {
            get: function() { getterCalls++; return 10; },
            set: function(v) { setterCalls++; lastSet = v; },
            configurable: true
        });
        "#,
    );
}

#[test]
fn accessor_global_declaration_does_not_invoke_getter() {
    let mut vm = Vm::new();
    setup_accessor_global(&mut vm);
    // A bare `var` over an accessor global is a passthrough alias: the
    // getter must not run during declaration.
    eval_on(&mut vm, "var acc;");
    assert_eq!(eval_int(&mut vm, "getterCalls"), 0);
}

#[test]
fn accessor_global_function_declaration_fires_setter_once() {
    let mut vm = Vm::new();
    setup_accessor_global(&mut vm);
    // A `function` declaration over an accessor global is a passthrough
    // alias: the declaration stores through the global [[Set]], firing
    // the setter exactly once, without invoking the getter.
    eval_on(&mut vm, "function acc() {}");
    assert_eq!(eval_int(&mut vm, "setterCalls"), 1);
    assert_eq!(eval_int(&mut vm, "getterCalls"), 0);
}

#[test]
fn accessor_global_read_does_fresh_get_per_read() {
    let mut vm = Vm::new();
    setup_accessor_global(&mut vm);
    eval_on(&mut vm, "var acc;");
    // Each read is a fresh global [[Get]]: the getter fires per read.
    assert_eq!(eval_int(&mut vm, "acc"), 10);
    assert_eq!(eval_int(&mut vm, "acc + acc"), 20);
    assert_eq!(eval_int(&mut vm, "getterCalls"), 3);
}

#[test]
fn accessor_global_write_fires_setter_once_per_write() {
    let mut vm = Vm::new();
    setup_accessor_global(&mut vm);
    eval_on(&mut vm, "var acc; acc = 5;");
    assert_eq!(eval_int(&mut vm, "setterCalls"), 1);
    assert_eq!(eval_int(&mut vm, "lastSet"), 5);
    // The getter still reports its own value; the setter never ran at
    // declaration time.
    assert_eq!(eval_int(&mut vm, "acc"), 10);
    assert_eq!(eval_int(&mut vm, "getterCalls"), 1);
}

#[test]
fn accessor_global_compound_assignment_gets_then_sets_once() {
    let mut vm = Vm::new();
    setup_accessor_global(&mut vm);
    eval_on(&mut vm, "var acc; acc += 5;");
    // One [[Get]] for the base, one [[Set]] for the result.
    assert_eq!(eval_int(&mut vm, "getterCalls"), 1);
    assert_eq!(eval_int(&mut vm, "setterCalls"), 1);
    assert_eq!(eval_int(&mut vm, "lastSet"), 15);
}

#[test]
fn accessor_global_throwing_getter_propagates_on_read() {
    let mut vm = Vm::new();
    eval_on(
        &mut vm,
        r#"
        Object.defineProperty(globalThis, "boom", {
            get: function() { throw new Error("bang"); },
            configurable: true
        });
        "#,
    );
    // Declaring over it is fine (no getter runs) ...
    eval_on(&mut vm, "var boom;");
    // ... but reading invokes the throwing getter.
    let v = eval_on(
        &mut vm,
        "var caught = null; try { boom; } catch (e) { caught = e; } \
         caught instanceof Error;",
    );
    assert!(matches!(v, Value::Bool(true)), "unexpected {v:?}");
}

// -- post-run install only installs fresh names ------------------------------

#[test]
fn post_run_install_does_not_resettle_existing_accessor_global() {
    let mut vm = Vm::new();
    setup_accessor_global(&mut vm);
    // Two in-run writes: the setter fires exactly twice. A post-run
    // install that re-set every declared name would fire it a third time.
    eval_on(&mut vm, "var acc; acc = 5; acc = 6;");
    assert_eq!(eval_int(&mut vm, "setterCalls"), 2);
    assert_eq!(eval_int(&mut vm, "lastSet"), 6);
}

#[test]
fn post_run_install_does_not_clobber_existing_data_global() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "var w = 5;");
    // A bare redeclaration plus an in-run write: write-through keeps the
    // global in sync; the post-run install must not touch it again (it
    // only installs fresh names).
    eval_on(&mut vm, "var w; w = 7;");
    assert_eq!(eval_int(&mut vm, "w"), 7);
    assert_eq!(eval_int(&mut vm, "globalThis.w"), 7);
}

// -- inherited globals ---------------------------------------------------------

#[test]
fn inherited_global_var_reads_through_prototype_chain() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "Object.prototype.inhData = 42;");
    // A bare `var` over an inherited data property binds without creating
    // an own property; reads resolve through the prototype chain.
    eval_on(&mut vm, "var inhData;");
    assert_eq!(eval_int(&mut vm, "inhData"), 42);
    assert_eq!(eval_str(&mut vm, "typeof inhData"), "number");
    // No own property was created by the bare `var`.
    let owned = eval_on(
        &mut vm,
        "Object.getOwnPropertyDescriptor(globalThis, 'inhData') === undefined",
    );
    assert!(
        matches!(owned, Value::Bool(true)),
        "bare var created an own property"
    );
}

#[test]
fn inherited_global_write_shadows_own_property() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "Object.prototype.inhData = 42;");
    eval_on(&mut vm, "var inhData; inhData = 99;");
    // Ordinary [[Set]] semantics: the write shadows an own property onto
    // the global object; the prototype's value is untouched.
    assert_eq!(eval_int(&mut vm, "inhData"), 99);
    assert_eq!(eval_int(&mut vm, "globalThis.inhData"), 99);
    // The write shadowed an own data property onto the global object.
    let owned = eval_on(
        &mut vm,
        "Object.getOwnPropertyDescriptor(globalThis, 'inhData') !== undefined",
    );
    assert!(
        matches!(owned, Value::Bool(true)),
        "write did not shadow an own property"
    );
    assert_eq!(eval_int(&mut vm, "Object.prototype.inhData"), 42);
}

#[test]
fn builtin_inherited_global_has_no_own_property_after_var() {
    let mut vm = Vm::new();
    // `toString` is inherited from Object.prototype in every realm.
    eval_on(&mut vm, "var toString;");
    assert_eq!(eval_str(&mut vm, "typeof toString"), "function");
    // No own property was created by the bare `var`.
    let owned = eval_on(
        &mut vm,
        "Object.getOwnPropertyDescriptor(globalThis, 'toString') === undefined",
    );
    assert!(
        matches!(owned, Value::Bool(true)),
        "bare var created an own property"
    );
}

// -- global declarative record: top-level let/const/class ----------------------
// The realm's shared global declarative environment record (ECMA-262
// 9.1.1.4): top-level `let`/`const`/`class` of a classic script persist
// in the `Vm` and later scripts resolve them by name — before the
// global object, and never as own properties of it.

fn eval_err(vm: &mut Vm, src: &str) -> String {
    match vm.eval(src) {
        Ok(v) => panic!("expected error for {src:?}, got {v:?}"),
        Err(e) => e.to_string(),
    }
}

fn eval_bool(vm: &mut Vm, src: &str) -> bool {
    match eval_on(vm, src) {
        Value::Bool(b) => b,
        v => panic!("expected bool for {src:?}, got {v:?}"),
    }
}

#[test]
fn top_level_let_visible_to_later_script() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "let sharedLet = 41;");
    // A later script resolves it by bare name ...
    assert_eq!(eval_int(&mut vm, "sharedLet + 1"), 42);
    // ... but it is not a property of the global object.
    assert_eq!(
        eval_str(&mut vm, "typeof globalThis.sharedLet"),
        "undefined"
    );
}

#[test]
fn top_level_const_callable_from_later_script() {
    let mut vm = Vm::new();
    // The WPT shape: `/common/sab.js` declares `const createBuffer`.
    eval_on(&mut vm, "const createBuffer = (n) => n * 2;");
    assert_eq!(eval_int(&mut vm, "createBuffer(21)"), 42);
    assert_eq!(eval_str(&mut vm, "typeof createBuffer"), "function");
}

#[test]
fn top_level_let_assignment_from_later_script() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "let counter = 1;");
    eval_on(&mut vm, "counter = 41;");
    assert_eq!(eval_int(&mut vm, "counter + 1"), 42);
    // Compound assignment goes through the same record.
    eval_on(&mut vm, "counter += 1;");
    assert_eq!(eval_int(&mut vm, "counter"), 42);
}

#[test]
fn top_level_const_assignment_from_later_script_throws() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "const frozen = 1;");
    let err = eval_err(&mut vm, "frozen = 2;");
    assert!(
        err.contains("TypeError") && err.contains("assignment to constant"),
        "unexpected error: {err}"
    );
    // The value is unchanged.
    assert_eq!(eval_int(&mut vm, "frozen"), 1);
}

#[test]
fn top_level_let_redeclaration_across_scripts_throws() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "let dup = 1;");
    let err = eval_err(&mut vm, "let dup = 2;");
    assert!(
        err.contains("SyntaxError") && err.contains("already been declared"),
        "unexpected error: {err}"
    );
    // The original binding survives the failed redeclaration.
    assert_eq!(eval_int(&mut vm, "dup"), 1);
}

#[test]
fn top_level_const_redeclaration_across_scripts_throws() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "const dupC = 1;");
    let err = eval_err(&mut vm, "const dupC = 2;");
    assert!(
        err.contains("SyntaxError") && err.contains("already been declared"),
        "unexpected error: {err}"
    );
}

#[test]
fn let_after_var_across_scripts_throws() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "var lv = 1;");
    let err = eval_err(&mut vm, "let lv = 2;");
    assert!(
        err.contains("SyntaxError") && err.contains("already been declared"),
        "unexpected error: {err}"
    );
    assert_eq!(eval_int(&mut vm, "lv"), 1);
}

#[test]
fn var_after_let_across_scripts_throws() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "let vl = 1;");
    let err = eval_err(&mut vm, "var vl = 2;");
    assert!(
        err.contains("SyntaxError") && err.contains("already been declared"),
        "unexpected error: {err}"
    );
    assert_eq!(eval_int(&mut vm, "vl"), 1);
}

#[test]
fn function_after_let_across_scripts_throws() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "let fl = 1;");
    let err = eval_err(&mut vm, "function fl() { return 2; }");
    assert!(
        err.contains("SyntaxError") && err.contains("already been declared"),
        "unexpected error: {err}"
    );
    assert_eq!(eval_int(&mut vm, "fl"), 1);
}

#[test]
fn var_redeclaration_still_allowed() {
    // `var` keeps its sloppy redeclaration semantics across scripts:
    // no SyntaxError, the later declaration wins.
    let mut vm = Vm::new();
    eval_on(&mut vm, "var rv = 1;");
    eval_on(&mut vm, "var rv = 2;");
    assert_eq!(eval_int(&mut vm, "rv"), 2);
    assert_eq!(eval_int(&mut vm, "globalThis.rv"), 2);
}

#[test]
fn block_scoped_let_does_not_leak_across_scripts() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "{ let hidden = 1; }");
    assert_eq!(eval_str(&mut vm, "typeof hidden"), "undefined");
}

#[test]
fn top_level_class_visible_to_later_script() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "class Klass { constructor() { this.n = 42; } }");
    assert_eq!(eval_int(&mut vm, "new Klass().n"), 42);
    assert_eq!(eval_str(&mut vm, "typeof Klass"), "function");
}

#[test]
fn closure_capturing_top_level_let_shared_across_scripts() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "let cap = 1; function bump() { cap++; }");
    // A later script calls the closure: it observes and mutates the same
    // cell the global record holds.
    eval_on(&mut vm, "bump(); bump();");
    assert_eq!(eval_int(&mut vm, "cap"), 3);
}

#[test]
fn uninitialized_top_level_let_is_tdz_in_later_script() {
    let mut vm = Vm::new();
    // The first script throws before the initializer runs: the binding
    // exists but is uninitialized. A later script must see a TDZ
    // ReferenceError, not "not defined".
    let err = eval_err(
        &mut vm,
        "function readQ() { return q; } readQ(); let q = 1;",
    );
    assert!(
        err.contains("before initialization"),
        "unexpected error: {err}"
    );
    let err = eval_err(&mut vm, "q;");
    assert!(
        err.contains("ReferenceError") && err.contains("before initialization"),
        "unexpected error: {err}"
    );
    // `typeof` of a TDZ binding throws too.
    let err = eval_err(&mut vm, "typeof q;");
    assert!(
        err.contains("ReferenceError") && err.contains("before initialization"),
        "unexpected error: {err}"
    );
}

#[test]
fn typeof_top_level_let_in_later_script() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "let typedLet = 5;");
    assert_eq!(eval_str(&mut vm, "typeof typedLet"), "number");
}

#[test]
fn let_shadows_sloppy_global_property() {
    let mut vm = Vm::new();
    // A sloppy assignment is a plain global property, not a var binding,
    // so a later `let` may shadow it (it is configurable).
    eval_on(&mut vm, "sg = 10;");
    eval_on(&mut vm, "let sg = 20;");
    assert_eq!(eval_int(&mut vm, "sg"), 20);
    assert_eq!(eval_int(&mut vm, "globalThis.sg"), 10);
}

#[test]
fn indirect_eval_top_level_const_visible_to_later_script() {
    let mut vm = Vm::new();
    // Indirect eval runs as a classic script: its top-level `const`
    // lands in the shared declarative record.
    eval_on(&mut vm, "(0, eval)('const indirectConst = 42;');");
    assert_eq!(eval_int(&mut vm, "indirectConst"), 42);
}

#[test]
fn indirect_eval_top_level_let_redeclaration_throws() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "let ie = 1;");
    let err = eval_err(&mut vm, "(0, eval)('let ie = 2;');");
    assert!(
        err.contains("SyntaxError") && err.contains("already been declared"),
        "unexpected error: {err}"
    );
}

#[test]
fn direct_eval_sees_global_lexical() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "let directSeen = 42;");
    // A direct eval whose caller does not declare the name falls through
    // to the global declarative record.
    assert_eq!(eval_int(&mut vm, "eval('directSeen')"), 42);
    assert!(eval_bool(&mut vm, "eval('typeof directSeen') === 'number'"));
}

#[test]
fn rerunning_same_compiled_handle_with_let_throws() {
    let mut vm = Vm::new();
    let handle = vm.compile_script("let rerun = 1;").unwrap();
    vm.run_compiled(handle).unwrap();
    assert_eq!(eval_int(&mut vm, "rerun"), 1);
    // Like eval'ing the same source twice: 16.1.6 redeclaration error.
    let err = vm
        .run_compiled(handle)
        .map(|_| "<ok>".to_string())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("SyntaxError") && err.contains("already been declared"),
        "unexpected error: {err}"
    );
}

#[test]
fn strict_top_level_let_shared_across_scripts() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "'use strict'; let strictLet = 42;");
    // A later sloppy script sees the strict script's lexical, and vice versa.
    assert_eq!(eval_int(&mut vm, "strictLet"), 42);
    assert_eq!(eval_int(&mut vm, "'use strict'; strictLet"), 42);
}

#[test]
fn reset_global_lexical_clears_record() {
    let mut vm = Vm::new();
    eval_on(&mut vm, "let doomed = 1;");
    vm.reset_global_lexical();
    assert_eq!(eval_str(&mut vm, "typeof doomed"), "undefined");
    // And the name is declarable again.
    eval_on(&mut vm, "let doomed = 2;");
    assert_eq!(eval_int(&mut vm, "doomed"), 2);
}
