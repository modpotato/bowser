//! V8-parity regression pins for class-body strict mode.
//!
//! Class bodies are always strict (spec 15.7.1): `parse_class_common`
//! forces strict mode for the whole class parse. These tests pin that the
//! forced strictness applies to every class-body position: the heritage,
//! computed keys, field initializers, the class name, and method bodies.
//!
//! Run with: `cargo test -p bowser_js --test class_strict`

use bowser_js::{Value, Vm, VmError};

/// Assert `src` parses. Anything but a `Parse` error counts: runtime
/// failures prove the source got past the parser.
fn ok(src: &str) {
    let mut vm = Vm::new();
    if let Err(VmError::Parse(e)) = vm.eval(src) {
        panic!("parse failed for {src:?}: {e}");
    }
}

/// Assert `src` fails to parse; return the diagnostic for substring checks.
fn err(src: &str) -> String {
    let mut vm = Vm::new();
    match vm.eval(src) {
        Err(VmError::Parse(e)) => e,
        other => panic!("expected parse error for {src:?}, got {other:?}"),
    }
}

fn eval(src: &str) -> Result<Value, String> {
    Vm::new().eval(src).map_err(|e| format!("{e:?}"))
}

fn eval_ok(src: &str) -> Value {
    eval(src).unwrap_or_else(|e| panic!("eval failed for {src:?}: {e}"))
}

/// Assert that `src` throws a `ReferenceError` (uncaught).
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

// ---------------------------------------------------------------------------
// Parse-time: octal escapes are early errors in every strict class position
// ---------------------------------------------------------------------------

#[test]
fn class_heritage_octal_escape_is_early_error() {
    // The heritage is parsed with strict mode forced, so a legacy octal
    // escape in it is a syntax error even in a sloppy script.
    assert!(err("class C extends '\\1' {}").contains("octal"));
    assert!(err("class C extends '\\01' {}").contains("octal"));
    // `\0` alone is not an octal escape and stays legal.
    ok("class C extends '\\0' {}");
}

#[test]
fn class_computed_key_octal_escape_is_early_error() {
    assert!(err("class C { ['\\1']() {} }").contains("octal"));
    assert!(err("class C { static ['\\8'] = 1; }").contains("octal"));
    ok("class C { ['\\0']() {} }");
}

#[test]
fn class_field_initializer_octal_escape_is_early_error() {
    // Field initializers are strict-mode code (the class forces strict),
    // so octal escapes in them are early errors.
    assert!(err("class C { x = '\\1'; }").contains("octal"));
    assert!(err("class C { static x = '\\01'; }").contains("octal"));
    ok("class C { x = '\\0'; }");
}

// ---------------------------------------------------------------------------
// Parse-time: strict identifier rules inside class bodies
// ---------------------------------------------------------------------------

#[test]
fn class_name_eval_or_arguments_is_early_error() {
    // `eval`/`arguments` may not be a class binding name in strict mode;
    // the class-body strictness makes this an early error in sloppy code.
    assert!(err("class eval {}").contains("eval"));
    assert!(err("class arguments {}").contains("arguments"));
    // Class expressions with those names error too.
    assert!(err("(class eval {});").contains("eval"));
    // Elsewhere in sloppy code `eval` is still a legal binding name.
    ok("var eval = 1;");
}

#[test]
fn class_method_with_statement_is_early_error() {
    // Method bodies are strict (the class forces strict), so `with` is a
    // syntax error even in a sloppy script.
    assert!(err("class C { m() { with (o) {} } }").contains("with"));
    assert!(err("class C { static m() { with (o) {} } }").contains("with"));
    assert!(err("class C { static { with (o) {} } }").contains("with"));
    // Sloppy functions outside a class still allow `with`.
    ok("function f() { with (o) {} }");
}

// ---------------------------------------------------------------------------
// Runtime: strictness of heritage / computed keys (spec 10.2.1)
// ---------------------------------------------------------------------------

#[test]
fn class_heritage_undeclared_assignment_is_reference_error() {
    // The heritage is strict-mode code even in a sloppy script: assignment
    // to an undeclared name throws ReferenceError instead of creating a
    // global.
    assert_reference_error("class C extends (undeclaredHeritageProbe = 1, Object) {}");
    // No global leaked as a side effect.
    let v = eval_ok(
        "var isRefErr = false; \
         try { class C extends (undeclaredHeritageProbe = 1, Object) {} } \
         catch (e) { isRefErr = e instanceof ReferenceError; } \
         isRefErr && typeof undeclaredHeritageProbe === 'undefined';",
    );
    assert!(
        matches!(v, Value::Bool(true)),
        "heritage leaked a global or did not throw ReferenceError: {v:?}"
    );
}

#[test]
fn class_computed_key_undeclared_assignment_is_reference_error() {
    // Computed keys are strict-mode code even in a sloppy script.
    assert_reference_error("class C { [undeclaredKeyProbe = 2]() {} }");
    let v = eval_ok(
        "var isRefErr = false; \
         try { class C { [undeclaredKeyProbe = 2]() {} } } \
         catch (e) { isRefErr = e instanceof ReferenceError; } \
         isRefErr && typeof undeclaredKeyProbe === 'undefined';",
    );
    assert!(
        matches!(v, Value::Bool(true)),
        "computed key leaked a global or did not throw ReferenceError: {v:?}"
    );
}
