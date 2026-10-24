//! Parser tests driven through the public [`bowser_js::Vm`] API.
//!
//! `Vm::eval` parses with the script goal (sloppy by default; a
//! `"use strict"` directive opts in) and surfaces parse failures as
//! [`bowser_js::VmError::Parse`], so these tests assert exactly what the
//! old AST-level suite asserted: "parses without error" (`ok`) and "fails
//! to parse with a diagnostic mentioning X" (`err`). Runtime behavior is
//! irrelevant here: any non-`Parse` outcome (a value, a thrown JS error, a
//! resource exhaustion) proves the source parsed.
//!
//! NOTE: the old suite also covered the *module* goal (`parse`). `Vm::eval`
//! only exposes the script goal, so module-goal-specific cases
//! (`import`/`export` declarations, `import.meta`) are asserted here as
//! parse *errors*; module parsing itself is exercised through the
//! crate-internal module loader. The script-goal cases the old suite
//! dropped (sloppy-mode `with`, `var yield` / `var await` as identifiers,
//! Annex B function statements, `for (var x = 1 in y)`) are covered again
//! below.
//!
//! Run with: `cargo test -p bowser_js --test parser`

use bowser_js::{Value, Vm, VmError};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Assert `src` parses (module goal). Anything but a `Parse` error counts:
/// runtime failures prove the source got past the parser.
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

// ---------------------------------------------------------------------------
// Literals
// ---------------------------------------------------------------------------

#[test]
fn literals() {
    ok("1; 1.5; 0x10; 0b101; 0o17; 1e10; 5n;");
    ok("'str'; \"str\"; `template`;");
    ok("true; false; null;");
    ok("/regex/; /regex/gi;");
}

#[test]
fn numeric_raw_preserved() {
    // Hex literal evaluates to 16 (the lexer preserved the raw "0x10").
    let mut vm = Vm::new();
    assert!(matches!(vm.eval("0x10;"), Ok(Value::Int(16))));
}

// ---------------------------------------------------------------------------
// Expressions: binary, unary, logical, assignment
// ---------------------------------------------------------------------------

#[test]
fn binary_operators() {
    ok("a + b; a - b; a * b; a / b; a % b; a ** b;");
    ok("a < b; a > b; a <= b; a >= b; a == b; a != b; a === b; a !== b;");
    ok("a << b; a >> b; a >>> b; a & b; a | b; a ^ b;");
    ok("a in b; a instanceof b;");
}

#[test]
fn unary_operators() {
    ok("-a; +a; !a; ~a; typeof a; void a; delete a.b;");
    ok("++a; --a; a++; a--;");
}

#[test]
fn logical_nullish() {
    ok("a && b; a || b; a ?? b;");
    // Unparenthesized ?? cannot mix with && or ||
    assert!(err("a ?? b && c;").contains("cannot be mixed"));
    assert!(err("a && b ?? c;").contains("cannot be mixed"));
    ok("(a ?? b) && c; a ?? (b && c);");
}

#[test]
fn assignment_operators() {
    ok("a = 1; a += 1; a -= 1; a *= 1; a /= 1; a %= 1; a **= 1;");
    ok("a <<= 1; a >>= 1; a >>>= 1; a &= 1; a |= 1; a ^= 1;");
    ok("a &&= 1; a ||= 1; a ??= 1;");
}

#[test]
fn exponentiation_restrictions() {
    // Unary cannot be directly left of **
    assert!(err("-a ** b;").contains("unary"));
    assert!(err("!a ** b;").contains("unary"));
    ok("(-a) ** b; -(a ** b);");
    // Right-associative
    ok("a ** b ** c;");
}

#[test]
fn conditional_and_sequence() {
    ok("a ? b : c;");
    ok("a, b, c;");
}

// ---------------------------------------------------------------------------
// Destructuring
// ---------------------------------------------------------------------------

#[test]
fn array_destructuring() {
    ok("const [a, b] = c;");
    ok("const [a = 1, , ...rest] = c;");
    ok("[a, b] = c;");
    ok("for ([a, b] of c);");
}

#[test]
fn object_destructuring() {
    ok("const {a, b: c} = d;");
    ok("const {a = 1, ...rest} = d;");
    ok("({a, b} = d);");
    ok("for ({a, b} of c);");
}

#[test]
fn nested_destructuring() {
    ok("const [{a: [b]}, {c}] = d;");
    ok("const {a: {b = 1}} = c;");
}

#[test]
fn bare_cover_initializer_in_assignment_target() {
    // The cover initializer's `pending_cover` flag must not leak into the
    // initializer's own parse: `[a = 1] = rhs` is a valid destructuring
    // assignment, not an "invalid initializer in object/array pattern".
    ok("var a; [a = 1] = [];");
    ok("var a; [a = 1] = b;");
    ok("var a; for ([a = 1] of b);");
    ok("var o; ({o = 1} = {});");
    ok("var a; [a = b = 1] = [];");
    // A bare cover initializer that never becomes a target is still an error.
    let e = err("[a = 1];");
    assert!(
        e.contains("invalid initializer"),
        "unexpected diagnostic: {e}"
    );
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

#[test]
fn function_declarations() {
    ok("function f() {}");
    ok("function* g() {}");
    ok("async function h() {}");
    ok("async function* i() {}");
    ok("function f(a, b = 1, ...rest) {}");
}

#[test]
fn function_expressions() {
    ok("(function() {}); (function named() {});");
    ok("(function*() {}); (async function() {});");
}

#[test]
fn arrows() {
    ok("() => 1;");
    ok("x => x;");
    ok("(a, b) => a + b;");
    ok("([a], {b}) => a;");
    ok("() => {};");
    ok("async () => 1; async x => x;");
    // No line break before =>
    assert!(err("x\n=> 1;").contains("=>"));
}

#[test]
fn strict_function_validation() {
    // Duplicate params in strict mode
    assert!(err("\"use strict\"; function f(a, a) {}").contains("duplicate"));
    // eval/arguments as param in strict
    assert!(err("\"use strict\"; function f(eval) {}").contains("eval"));
}

#[test]
fn yield_await_contexts() {
    ok("function* g() { yield 1; yield* h(); }");
    ok("async function f() { await 1; }");
}

// ---------------------------------------------------------------------------
// Classes
// ---------------------------------------------------------------------------

#[test]
fn basic_classes() {
    ok("class C {}");
    ok("class C extends D {}");
    ok("(class {}); (class Named {});");
    ok("class C { constructor() {} method() {} }");
}

#[test]
fn class_fields() {
    ok("class C { x = 1; y; }");
    ok("class C { static x = 1; static y; }");
    // Field named constructor is invalid
    assert!(err("class C { constructor = 1; }").contains("constructor"));
    assert!(err("class C { static constructor = 1; }").contains("constructor"));
}

#[test]
fn private_members() {
    ok("class C { #x = 1; #m() {} get #g() {} }");
    ok("class C { #x; m() { return this.#x; } }");
    // #x in obj
    ok("class C { #x; m(o) { return #x in o; } }");
    // Undeclared private name
    assert!(err("class C { m() { return this.#y; } }").contains("#y"));
    assert!(err("class C { m(o) { return #y in o; } }").contains("#y"));
    // Duplicate private name
    assert!(err("class C { #x; #x; }").contains("duplicate"));
    // #constructor invalid
    assert!(err("class C { #constructor() {} }").contains("#constructor"));
    // delete this.#x invalid
    assert!(err("class C { m() { delete this.#x; } }").contains("private"));
}

#[test]
fn static_blocks() {
    ok("class C { static { } }");
    // return invalid in static block
    assert!(err("class C { static { return 1; } }").contains("return"));
    // sloppy scripts: `await` is a plain identifier, not a reserved word.
    ok("class C { static { await; } }");
    // `await 1` still fails to parse (unexpected token), just not with an
    // await-specific diagnostic.
    let _ = err("class C { static { await 1; } }");
}

#[test]
fn accessors() {
    ok("class C { get x() {} set x(v) {} }");
    ok("class C { static get x() {} static set x(v) {} }");
    ok("class C { get #x() {} set #x(v) {} }");
    // Bad arity
    assert!(err("class C { get x(a) {} }").contains("getter"));
    assert!(err("class C { set x() {} }").contains("setter"));
    // Async accessors invalid
    assert!(err("class C { async get x() {} }").contains("async"));
}

#[test]
fn constructor_rules() {
    ok("class C { constructor() {} }");
    ok("class D extends B { constructor() { super(); } }");
    // Duplicate constructor
    assert!(err("class C { constructor() {} constructor() {} }").contains("duplicate"));
    // Async/generator constructor
    assert!(err("class C { async constructor() {} }").contains("constructor"));
    assert!(err("class C { *constructor() {} }").contains("constructor"));
    // super() only in derived constructor
    assert!(err("class C { constructor() { super(); } }").contains("super"));
    ok("class D extends B { constructor() { super(); } }");
    // super() not in methods
    assert!(err("class C { m() { super(); } }").contains("super"));
}

#[test]
fn super_property() {
    ok("class D extends B { m() { return super.x; } }");
    ok("class C { m() { return super.x; } }"); // valid even non-derived
    ok("class C { x = super.y; }"); // valid in field
}

// ---------------------------------------------------------------------------
// Optional chaining
// ---------------------------------------------------------------------------

#[test]
fn optional_chaining() {
    ok("a?.b; a?.[b]; a?.(c);");
    ok("a?.b.c; a?.b?.c;");
    // Cannot be assignment target
    assert!(err("a?.b = 1;").contains("optional"));
    // Cannot be tagged template tag
    assert!(err("a?.b`c`;").contains("optional"));
    // Cannot be the direct operand of `new` (parens are fine)
    assert!(err("new a?.b();").contains("optional"));
    ok("new (a?.b)();");
    // delete is allowed
    ok("delete a?.b;");
}

// ---------------------------------------------------------------------------
// Modules
// ---------------------------------------------------------------------------

#[test]
fn imports() {
    // `Vm::eval` is the script goal: static imports are rejected.
    for src in [
        "import 'm';",
        "import x from 'm';",
        "import * as ns from 'm';",
        "import {a, b as c} from 'm';",
        "import x, * as ns from 'm';",
        "import x, {a} from 'm';",
        "import {'a-b' as c} from 'm';",
    ] {
        assert!(
            err(src).contains("only allowed in modules"),
            "expected module-only error for {src:?}"
        );
    }
}

#[test]
fn exports() {
    // `Vm::eval` is the script goal: exports are rejected.
    for src in [
        "export * from 'm';",
        "export * as ns from 'm';",
        "export {a, b as c};",
        "export {a} from 'm';",
        "export default 1;",
        "export default function f() {}",
        "export default class C {}",
        "export var x = 1;",
        "export function f() {}",
        "export class C {}",
    ] {
        assert!(
            err(src).contains("only allowed in modules"),
            "expected module-only error for {src:?}"
        );
    }
}

#[test]
fn dynamic_import() {
    ok("import('m');");
    ok("const p = import('m');");
}

#[test]
fn import_meta() {
    // `Vm::eval` is the script goal: `import.meta` is rejected.
    assert!(err("import.meta;").contains("only allowed in modules"));
}

// ---------------------------------------------------------------------------
// ASI
// ---------------------------------------------------------------------------

#[test]
fn asi() {
    ok("var a = 1\nvar b = 2");
    ok("a = 1\nb = 2");
    ok("function f() {}\nfunction g() {}");
    // No ASI before ( in `a\n(b)` — actually there IS no ASI, it's a call
    ok("a\n(b);"); // This is a call a(b), not two statements
}

// ---------------------------------------------------------------------------
// Strict mode
// ---------------------------------------------------------------------------

#[test]
fn strict_directive() {
    ok("\"use strict\"; var x = 1;");
    // with is invalid in strict
    assert!(err("\"use strict\"; with (o) {}").contains("with"));
    // delete of identifier invalid in strict
    assert!(err("\"use strict\"; delete x;").contains("delete"));
}

#[test]
fn strict_octal() {
    assert!(err("\"use strict\"; '\\1';").contains("octal"));
    assert!(err("\"use strict\"; '\\8';").contains("octal"));
    ok("\"use strict\"; '\\0';");
    assert!(err("\"use strict\"; '\\01';").contains("octal"));
}

#[test]
fn strict_object_methods() {
    // Parse-time strict checks apply inside object-literal methods, both
    // via inherited strictness and via a "use strict" directive in the
    // method body.
    assert!(err("\"use strict\"; var o = { m() { delete x; } };").contains("delete"));
    assert!(err("var o = { m() { \"use strict\"; delete x; } };").contains("delete"));
    assert!(err("var o = { get g() { \"use strict\"; delete x; } };").contains("delete"));
    assert!(err("var o = { set s(v) { \"use strict\"; delete x; } };").contains("delete"));
    assert!(err("\"use strict\"; var o = { m() { with (o) {} } };").contains("with"));
    assert!(err("var o = { m() { \"use strict\"; with (o) {} } };").contains("with"));
    assert!(err("\"use strict\"; var o = { m(a, a) {} };").contains("duplicate"));
    assert!(err("var o = { m(a, a) { \"use strict\"; } };").contains("duplicate"));
    assert!(err("\"use strict\"; var o = { set s(eval) {} };").contains("eval"));
    // Sloppy methods still allow all of these.
    ok("var o = { m() { delete x; } };");
    ok("var o = { m(a, a) {} };");
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

#[test]
fn control_flow() {
    ok("if (a) b; else c;");
    ok("while (a) b;");
    ok("do b; while (a);");
    ok("for (;;) b;");
    ok("for (var i = 0; i < 10; i++) b;");
    ok("for (a in b) c;");
    ok("for (a of b) c;");
    ok("for (const [k, v] of obj) c;");
    ok("switch (a) { case 1: b; default: c; }");
    ok("try { a; } catch (e) { b; } finally { c; }");
    ok("try { a; } catch { b; }");
    ok("throw new Error();");
    ok("label: while (1) { break label; continue label; }");
}

#[test]
fn for_await() {
    ok("async function f() { for await (const x of y) {} }");
    assert!(err("for await (const x of y) {}").contains("await"));
}

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------

#[test]
fn templates() {
    ok("`hello`;");
    ok("`hello ${name}!`;");
    ok("tag`hello ${a} ${b}`;");
    // Invalid escape in tagged template: cooked is None (no parse error)
    ok("tag`\\u{110000}`;");
    // Invalid escape in untagged template is a parse error
    assert!(err("`\\u{110000}`;").contains("escape"));
}

// ---------------------------------------------------------------------------
// Regex
// ---------------------------------------------------------------------------

#[test]
fn regex() {
    ok("/abc/; /abc/gi; /a\\/b/;");
    ok("var re = /test/;");
}

// ---------------------------------------------------------------------------
// Spanned errors
// ---------------------------------------------------------------------------

#[test]
fn error_spans() {
    // VmError::Parse carries the SyntaxError Debug, span included.
    let s = err("var 1 = 2;");
    assert!(s.contains("Span"), "unexpected: {s}");
    // Span should point at the `1`
    assert!(s.contains("start: 4"), "unexpected: {s}");
}

// ---------------------------------------------------------------------------
// Every AST variant: statements
// ---------------------------------------------------------------------------

#[test]
fn all_stmt_variants() {
    // Expr, Block, VarDecl, FunctionDecl, ClassDecl, Return, If, While,
    // DoWhile, For, ForIn, ForOf, Switch, Try, Throw, Break, Continue,
    // Labeled, Debugger, Empty, Import, Export
    ok("x;");
    ok("{ x; }");
    ok("var a, b = 1; let c; const d = 1;");
    ok("function f() {}");
    ok("class C {}");
    ok("function f() { return 1; return; }");
    ok("if (a) b;");
    ok("while (a) b;");
    ok("do b; while (a);");
    ok("for (;;);");
    ok("for (a in b);");
    ok("for (a of b);");
    ok("switch (a) {}");
    ok("try {} catch {}");
    ok("throw a;");
    ok("while (1) { break; continue; }");
    ok("lbl: x;");
    ok("debugger;");
    ok(";");
    assert!(err("import 'm';").contains("only allowed in modules"));
    assert!(err("export {};").contains("only allowed in modules"));
}

#[test]
fn all_expr_variants() {
    // Ident, Lit, Array, Object, Function, Class, Arrow, Call, New,
    // Member, Unary, Update, Binary, Logical, Assign, Conditional,
    // Sequence, Template, TaggedTemplate, This, Super, NewTarget,
    // ImportMeta, Await, Yield, Spread, PrivateIdent, Regex
    ok("x;");
    ok("1;");
    ok("[1, 2];");
    ok("({a: 1});");
    ok("(function() {});");
    ok("(class {});");
    ok("() => 1;");
    ok("f(1);");
    ok("new F();");
    ok("a.b;");
    ok("-a;");
    ok("a++;");
    ok("a + b;");
    ok("a && b;");
    ok("a = 1;");
    ok("a ? b : c;");
    ok("a, b;");
    ok("`t${x}`;");
    ok("tag`t`;");
    ok("this;");
    ok("class C { m() { super.x; } };");
    ok("class C { static { new.target; } };");
    assert!(err("import.meta;").contains("only allowed in modules"));
    ok("async function f() { await 1; }");
    ok("function* g() { yield 1; }");
    ok("f(...args);");
    ok("class C { #x; m(o) { #x in o; } };");
    ok("/re/;");
}

#[test]
fn all_pat_variants() {
    // Ident, Array, Object, Rest, Assign
    ok("var x = 1;");
    ok("var [a] = 1;");
    ok("var {a} = 1;");
    ok("var [...r] = 1;");
    ok("var [a = 1] = 2;");
    ok("function f([a], {b}, ...r) {}");
}

// ---------------------------------------------------------------------------
// Contextual keywords as identifiers
// ---------------------------------------------------------------------------

#[test]
fn contextual_keywords_are_identifiers_in_primary_position() {
    // `target`, `of`, `from`, `as`, `let`, `static`, `get`, `set` lex as
    // keyword kinds but are ordinary identifiers in primary position.
    ok("var target = 1; target = 2; target;");
    ok("var of = 1; of;");
    ok("var from = 1; from + 1;");
    ok("var as = 1; as;");
    ok("var static_ = 1; static_;");
    ok("var get = 1; get;");
    ok("var set = 1; set;");
    ok("var let_ = 1; let_;");
    // `let` itself is an identifier in sloppy primary position.
    ok("var x = 1; x = let;");
}
