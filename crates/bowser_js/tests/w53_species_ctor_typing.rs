//! Wave-53 regression tests: `IsConstructor` error typing on the
//! species-constructor paths.
//!
//! ECMA-262 `SpeciesConstructor` step 6 requires `IsConstructor(S)` —
//! not mere callability. A callable-but-not-constructible *native*
//! (e.g. `Array.isArray`, `Math.max`, `parseInt`) as `@@species` (or as
//! a `Reflect.construct` target / `newTarget`, or as the `this` of
//! `%TypedArray%.of`/`from`) must throw `TypeError` instead of being
//! silently "constructed". `ArrayBuffer.prototype.slice` runs the full
//! `SpeciesConstructor(O, %ArrayBuffer%)` + `Construct` sequence, so
//! hostile constructors error and subclass species are honored.
//!
//! These drive the public [`bowser_js::Vm`] API (parse -> compile -> run).

use bowser_js::{Value, Vm};

fn eval_ok(src: &str) -> Value {
    Vm::new()
        .eval(src)
        .unwrap_or_else(|e| panic!("eval failed for {src:?}: {e:?}"))
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

fn eval_bool(src: &str) -> bool {
    match eval_ok(src) {
        Value::Bool(b) => b,
        v => panic!("expected bool for {src:?}, got {v:?}"),
    }
}

// -- TypedArray species: native non-constructors ----------------------------

#[test]
fn species_native_non_constructor_throws() {
    // `IsConstructor(Array.isArray)` is false: `SpeciesConstructor`
    // step 6 throws `TypeError` instead of "constructing" it.
    for method in ["slice", "subarray"] {
        assert_type_error(&format!(
            "var ta = new Uint8Array([1,2,3]); \
             ta.constructor = {{ [Symbol.species]: Array.isArray }}; ta.{method}();"
        ));
    }
    assert_type_error(
        "var ta = new Uint8Array([1,2,3]); \
         ta.constructor = { [Symbol.species]: Array.isArray }; ta.map(x => x);",
    );
    assert_type_error(
        "var ta = new Uint8Array([1,2,3]); \
         ta.constructor = { [Symbol.species]: parseInt }; ta.filter(x => true);",
    );
    assert_type_error(
        "var ta = new Uint8Array([1,2,3]); \
         ta.constructor = { [Symbol.species]: Math.max }; ta.slice(0, 1);",
    );
    // BigInt-kind views go through the same gate.
    assert_type_error(
        "var ta = new BigInt64Array([1n,2n]); \
         ta.constructor = { [Symbol.species]: Array.isArray }; ta.slice();",
    );
}

#[test]
fn species_legit_constructors_still_work() {
    // Subclass species: the species constructor is genuinely invoked.
    assert!(eval_bool(
        "class M extends Uint8Array {}; \
         var r = new M([1,2,3]).map(x => x * 2); \
         r instanceof M && r.length === 3 && r[0] === 2 && r[2] === 6;"
    ));
    // `@@species: undefined` falls back to the intrinsic default.
    assert!(eval_bool(
        "var ta = new Uint8Array([1,2,3]); \
         ta.constructor = { [Symbol.species]: undefined }; \
         var r = ta.slice(1); r instanceof Uint8Array && r.length === 2 && r[0] === 2;"
    ));
    // Proxy of a constructor is a constructor (proxy-transparent
    // `IsConstructor`); a proxy of a non-constructor throws.
    assert_type_error(
        "var ta = new Uint8Array([1,2,3]); \
         ta.constructor = { [Symbol.species]: new Proxy(Array.isArray, {}) }; ta.slice();",
    );
    // Non-object / non-constructor-object species still throw.
    assert_type_error(
        "var ta = new Uint8Array([1,2,3]); \
         ta.constructor = { [Symbol.species]: 42 }; ta.slice();",
    );
    assert_type_error(
        "var ta = new Uint8Array([1,2,3]); \
         ta.constructor = { [Symbol.species]: {} }; ta.slice();",
    );
}

// -- Reflect.construct --------------------------------------------------------

#[test]
fn reflect_construct_rejects_non_constructor_natives() {
    assert_type_error("Reflect.construct(Array.isArray, []);");
    assert_type_error("Reflect.construct(Math.max, []);");
    assert_type_error("Reflect.construct(parseInt, []);");
    // `newTarget` gets the same `IsConstructor` gate.
    assert_type_error("Reflect.construct(Array, [], Array.isArray);");
}

#[test]
fn reflect_construct_accepts_real_constructors() {
    assert!(eval_bool(
        "var a = Reflect.construct(Array, [3]); a instanceof Array && a.length === 3;"
    ));
    assert!(eval_bool(
        "var t = Reflect.construct(Uint8Array, [4]); t instanceof Uint8Array && t.length === 4;"
    ));
    assert!(eval_bool(
        "var b = Reflect.construct(ArrayBuffer, [8]); b instanceof ArrayBuffer && b.byteLength === 8;"
    ));
    assert!(eval_bool(
        "var m = Reflect.construct(Map, []); m instanceof Map;"
    ));
    // Bytecode classes still construct.
    assert!(eval_bool(
        "class C { constructor(x) { this.x = x; } } \
         Reflect.construct(C, [42]).x === 42;"
    ));
    // Bytecode non-constructors throw `TypeError` (from the VM path).
    assert_type_error("Reflect.construct(() => {}, []);");
}

// -- TypedArray.of / from ------------------------------------------------------

#[test]
fn typed_array_of_from_reject_non_constructor_this() {
    assert_type_error("Uint8Array.of.call(Array.isArray, 1, 2);");
    assert_type_error("Uint8Array.from.call(Math.max, [1, 2]);");
}

#[test]
fn typed_array_of_from_accept_real_constructors() {
    assert!(eval_bool(
        "var t = Uint8Array.of(1, 2, 3); t instanceof Uint8Array && t[2] === 3;"
    ));
    assert!(eval_bool(
        "class M extends Uint8Array {}; \
         var t = M.of(1, 2); t instanceof M && t.length === 2;"
    ));
    assert!(eval_bool(
        "var t = Uint8Array.from([4, 5]); t instanceof Uint8Array && t[0] === 4;"
    ));
}

// -- ArrayBuffer.prototype.slice species ---------------------------------------

#[test]
fn arraybuffer_slice_species_errors() {
    // Non-object `constructor` → `TypeError` (`SpeciesConstructor`).
    assert_type_error("var b = new ArrayBuffer(8); b.constructor = 42; b.slice(0, 4);");
    // Non-constructor `@@species` → `TypeError`.
    assert_type_error(
        "var b = new ArrayBuffer(8); \
         b.constructor = { [Symbol.species]: {} }; b.slice(0, 4);",
    );
    assert_type_error(
        "var b = new ArrayBuffer(8); \
         b.constructor = { [Symbol.species]: Array.isArray }; b.slice(0, 4);",
    );
    // Species returning a non-buffer → `TypeError`
    // (`RequireInternalSlot(new, [[ArrayBufferData]])`).
    assert_type_error(
        "var b = new ArrayBuffer(8); \
         b.constructor = { [Symbol.species]: function() { return {}; } }; b.slice(0, 4);",
    );
    // Species returning a short buffer → `TypeError`.
    assert_type_error(
        "var b = new ArrayBuffer(8); \
         b.constructor = { [Symbol.species]: function() { return new ArrayBuffer(1); } }; \
         b.slice(0, 4);",
    );
}

#[test]
fn arraybuffer_slice_species_subclass_honored() {
    // An explicit `@@species` is honored: the result is built by the
    // species constructor and carries its prototype, with the sliced
    // bytes copied in.
    assert!(eval_bool(
        "class M extends ArrayBuffer {} \
         var b = new M(8); \
         new Uint8Array(b).set([10, 20, 30, 40, 50, 60, 70, 80]); \
         b.constructor = { [Symbol.species]: M }; \
         var c = b.slice(2, 6); \
         c instanceof M && c instanceof ArrayBuffer && c.byteLength === 4 && \
         new Uint8Array(c)[0] === 30 && new Uint8Array(c)[3] === 60;"
    ));
    // Without an explicit species the default is `%ArrayBuffer%`
    // (`SpeciesConstructor` step 5): subclass instances slice to plain
    // `ArrayBuffer`s.
    assert!(eval_bool(
        "class M extends ArrayBuffer {} \
         var b = new M(4); new Uint8Array(b).set([1, 2, 3, 4]); \
         var c = b.slice(1, 3); \
         c instanceof ArrayBuffer && !(c instanceof M) && c.byteLength === 2 && \
         new Uint8Array(c)[0] === 2 && new Uint8Array(c)[1] === 3 && \
         (new Uint8Array(c).set([9, 9]), new Uint8Array(b)[1] === 2);"
    ));
    // `@@species: null` falls back to `%ArrayBuffer%` too.
    assert!(eval_bool(
        "class M extends ArrayBuffer {} \
         var b = new M(4); b.constructor = { [Symbol.species]: null }; \
         var c = b.slice(0, 2); \
         c instanceof ArrayBuffer && !(c instanceof M) && c.byteLength === 2;"
    ));
    // `newTarget` blindness is fixed as a drive-by: subclass
    // construction links the subclass prototype.
    assert!(eval_bool(
        "class M extends ArrayBuffer {} \
         var b = new M(8); b instanceof M && b instanceof ArrayBuffer;"
    ));
}

// -- Native constructor registry (w53 H1) -----------------------------------
// `is_constructor` gates `Reflect.construct` and the species paths
// through the install-time registry (`mark_native_constructor`). Three
// genuinely-constructible natives were never marked, so
// `Reflect.construct(URL, …)` threw `TypeError: not a constructor`
// while `new URL(…)` worked. These drive the full page web-API install.

/// A `Vm` with the production page web-API surface installed
/// (`URL`, `URLSearchParams`, `SnapEvent`, …).
fn web_vm() -> Vm {
    let mut vm = Vm::new();
    let (doc, _) = bowser_dom::Document::parse_html(b"<html><head></head><body></body></html>");
    let doc = std::rc::Rc::new(std::cell::RefCell::new(doc));
    let base = bowser_url::parse("http://example.com/", None).unwrap();
    vm.install_page_web_apis(doc, base, "BowserTest/1.0")
        .expect("production install");
    vm
}

fn web_eval_bool(vm: &mut Vm, src: &str) -> bool {
    match vm
        .eval(src)
        .unwrap_or_else(|e| panic!("eval failed for {src:?}: {e:?}"))
    {
        Value::Bool(b) => b,
        v => panic!("expected bool for {src:?}, got {v:?}"),
    }
}

#[test]
fn reflect_construct_accepts_marked_native_constructors() {
    let mut vm = web_vm();
    assert!(web_eval_bool(
        &mut vm,
        "var u = Reflect.construct(URL, ['https://x/']); \
         u instanceof URL && u.href === 'https://x/';"
    ));
    assert!(web_eval_bool(
        &mut vm,
        "var p = Reflect.construct(URLSearchParams, ['a=1']); \
         p instanceof URLSearchParams && p.get('a') === '1';"
    ));
    assert!(web_eval_bool(
        &mut vm,
        "var e = Reflect.construct(SnapEvent, ['snapchange']); \
         e instanceof SnapEvent && e.type === 'snapchange';"
    ));
}

#[test]
fn native_species_path_uses_marked_constructor() {
    // `TypedArray.prototype.map` species-constructs through
    // `SpeciesConstructor`, which consults the same registry. Before the
    // mark this threw `TypeError: @@species is not a constructor`; now
    // the hostile (non-view) species result is returned with the mapped
    // values stored as plain properties.
    let mut vm = web_vm();
    assert!(web_eval_bool(
        &mut vm,
        "var ta = new Uint8Array([1, 2, 3]); \
         ta.constructor = { [Symbol.species]: URLSearchParams }; \
         var r = ta.map(x => x * 2); \
         r instanceof URLSearchParams && r[0] === 2 && r[1] === 4 && r[2] === 6;"
    ));
}

// -- Proxy `P_TARGET` hardening (cycle guard, forgery, repointing) ----------

use std::time::Duration;

/// Evaluate `src` on a worker thread with a hard wall-clock timeout.
///
/// The `P_TARGET`-following loops in `is_constructor`/`is_callable_deep`
/// run in native code with no VM fuel check: if the cycle guard ever
/// regresses, a plain `eval` would wedge the test runner forever instead
/// of failing. The timeout converts that wedge into a test failure.
///
/// `src` must assign its boolean outcome to the `__ok` global (function
/// calls in this engine do not propagate a completion value, so the
/// outcome cannot be `return`ed).
fn eval_guarded(src: &str, timeout: Duration) -> Result<(), String> {
    // Only plain data crosses the thread boundary.
    let script = format!(
        "var __ok = false; \
         try {{ {src} }} \
         catch (e) {{ __ok = 'threw:' + (e && e.constructor ? e.constructor.name : typeof e); }} \
         __ok === true;"
    );
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        let out = match Vm::new().eval(&script) {
            Ok(v) => format!("{v:?}"),
            Err(e) => format!("eval error: {e}"),
        };
        let _ = tx.send(out);
    });
    match rx.recv_timeout(timeout) {
        Ok(s) => {
            if s == "Bool(true)" {
                Ok(())
            } else {
                Err(format!("script outcome was not `true`: {s}"))
            }
        }
        Err(_) => Err(format!("eval timed out after {timeout:?}: probable hang")),
    }
}

/// The hidden slot keys, spelled without a literal NUL in the Rust source.
const PROXY_KEYS: &str = r#"
    var K = String.fromCharCode(0) + "bowser:proxy-target";
    var H = String.fromCharCode(0) + "bowser:proxy-handler";
"#;

#[test]
fn proxy_target_self_store_ignored_sloppy() {
    // `p[K] = p` must be rejected: the proxy's `P_TARGET` slot is
    // non-writable. Before the fix the VM's store path overwrote the
    // slot directly, forging a self-cycle that wedged `is_constructor`
    // with no fuel check. The proxy keeps working afterwards.
    let src = format!(
        "{PROXY_KEYS} \
         var t = function() {{ this.m = 1; }}; \
         var p = new Proxy(t, {{}}); \
         p[K] = p; \
         var r = Reflect.construct(p, []); \
         __ok = (r.m === 1);"
    );
    eval_guarded(&src, Duration::from_secs(10))
        .unwrap_or_else(|e| panic!("cycle guard failed: {e}"));
}

#[test]
fn proxy_target_self_store_rejected_strict() {
    // Strict mode turns the same rejected write into a `TypeError`;
    // the proxy's real target link is untouched either way.
    let src = format!(
        "var __inner = false; \
         (function() {{ \
             \"use strict\"; \
             {PROXY_KEYS} \
             var t = function() {{ this.m = 1; }}; \
             var p = new Proxy(t, {{}}); \
             var threw = false; \
             try {{ p[K] = p; }} catch (e) {{ threw = e instanceof TypeError; }} \
             var r = Reflect.construct(p, []); \
             __inner = (threw && r.m === 1); \
         }})(); \
         __ok = __inner;"
    );
    eval_guarded(&src, Duration::from_secs(10))
        .unwrap_or_else(|e| panic!("strict rejection failed: {e}"));
}

#[test]
fn proxy_target_forged_cycle_capped_at_max_hops() {
    // A 300-link forged `P_TARGET` chain ending in a self-loop, with
    // both slots present on every link (defeats the fail-closed parity
    // check, so only the hop cap can stop it). Must throw `TypeError`,
    // not hang.
    let src = format!(
        "{PROXY_KEYS} \
         var prev = {{}}; \
         for (var i = 0; i < 300; i++) {{ \
             var o = {{}}; o[K] = prev; o[H] = {{}}; prev = o; \
         }} \
         prev[K] = prev; \
         var threw = false; \
         try {{ Reflect.construct(prev, []); }} catch (e) {{ threw = e instanceof TypeError; }} \
         __ok = threw;"
    );
    eval_guarded(&src, Duration::from_secs(10)).unwrap_or_else(|e| panic!("hop cap failed: {e}"));
}

#[test]
fn proxy_target_receiver_set_back_edge_fails_closed() {
    // `Reflect.set` follows the spec `[[Set]]` receiver dance, so the
    // forged slot lands on the proxy's *target* (`t[K] = p`). The
    // `P_TARGET` walk must still terminate: `t` carries only the forged
    // target slot (no handler slot), so the fail-closed parity check
    // stops the walk, and the proxy — genuinely of constructor `t` —
    // keeps constructing.
    let src = format!(
        "{PROXY_KEYS} \
         var t = function() {{ this.m = 1; }}; \
         var p = new Proxy(t, {{}}); \
         var ok = Reflect.set(p, K, p); \
         var d = Reflect.getOwnPropertyDescriptor(p, K); \
         var r = Reflect.construct(p, []); \
         __ok = (ok && d !== undefined && r.m === 1);"
    );
    eval_guarded(&src, Duration::from_secs(10))
        .unwrap_or_else(|e| panic!("back-edge guard failed: {e}"));
}

#[test]
fn forged_single_slot_proxy_target_fails_closed() {
    // A lone forged `P_TARGET` (no handler slot) is not a proxy:
    // `IsConstructor` must not follow it to `Array`.
    assert_type_error(&format!(
        "{PROXY_KEYS} \
         var o = {{}}; o[K] = Array; \
         Reflect.construct(o, []);"
    ));
    // Same for the species gate.
    assert_type_error(&format!(
        "{PROXY_KEYS} \
         var o = {{}}; o[K] = Array; \
         var ta = new Uint8Array([1,2,3]); \
         ta.constructor = {{ [Symbol.species]: o }}; ta.slice();"
    ));
}

#[test]
fn proxy_target_slot_is_not_repointable() {
    // Assigning the hidden key on a *proxy of* a proxy used to rewrite
    // the inner proxy's real `P_TARGET` slot via the receiver-based
    // `[[Set]]`. The slot is non-writable now: the write fails (silently
    // in sloppy mode) and the inner proxy still constructs its original
    // target — had the repoint succeeded, `p1` would construct `Array`
    // and `marker` would be `undefined`.
    assert!(eval_bool(&format!(
        "{PROXY_KEYS} \
         var t = function() {{ this.marker = 42; }}; \
         var p1 = new Proxy(t, {{}}); \
         var p2 = new Proxy(p1, {{}}); \
         p2[K] = Array; \
         var r = Reflect.construct(p1, []); \
         r.marker === 42;"
    )));
}

// -- Reflect.construct newTarget strictness ---------------------------------

#[test]
fn reflect_construct_rejects_non_constructible_new_targets() {
    // ECMA-262 `Reflect.construct` step 3: `IsConstructor(newTarget)`.
    // The native-target arm never consults the VM, so an arrow/method/
    // async/generator `newTarget` must be rejected by the gate itself.
    assert_type_error("Reflect.construct(Array, [], () => {});");
    assert_type_error("Reflect.construct(Array, [], async () => {});");
    assert_type_error("Reflect.construct(Array, [], function* () {});");
    assert_type_error("var o = { m() {} }; Reflect.construct(Array, [], o.m);");
    // A proxy of an arrow is still an arrow for `IsConstructor`.
    assert_type_error("Reflect.construct(Array, [], new Proxy(() => {}, {}));");
    // And on the bytecode-target arm, where the VM *does* see the call.
    assert_type_error("Reflect.construct(function() {}, [], () => {});");
    assert_type_error("class C {} Reflect.construct(C, [], () => {});");
}

#[test]
fn reflect_construct_accepts_constructible_new_targets() {
    // A plain function `newTarget` still works; the instance takes its
    // prototype from `newTarget.prototype`.
    assert!(eval_bool(
        "function NT() {} NT.prototype.marker = 7; \
         var a = Reflect.construct(Array, [3], NT); \
         a instanceof NT && a.marker === 7 && a.length === 3;"
    ));
    // The default (`newTarget` === target) is unaffected.
    assert!(eval_bool(
        "var a = Reflect.construct(Array, [3]); a instanceof Array && a.length === 3;"
    ));
}

// -- Rooted construction paths ------------------------------------------------

#[test]
fn arraybuffer_construct_new_target_links_prototype() {
    // Exercises the rooted `AllocateArrayBuffer(newTarget)` path in
    // `arraybuffer_constructor`: the fresh buffer is held across the
    // `newTarget.prototype` lookup and `set_prototype`.
    assert!(eval_bool(
        "class D extends ArrayBuffer {} \
         var b = Reflect.construct(ArrayBuffer, [8], D); \
         b instanceof D && b instanceof ArrayBuffer && b.byteLength === 8 && \
         Object.getPrototypeOf(b) === D.prototype;"
    ));
}

#[test]
fn array_construct_new_target_links_prototype() {
    // Same for the rooted path in `array_constructor`.
    assert!(eval_bool(
        "class D extends Array {} \
         var a = Reflect.construct(Array, [1, 2, 3], D); \
         a instanceof D && a instanceof Array && a.length === 3 && \
         Object.getPrototypeOf(a) === D.prototype;"
    ));
}
