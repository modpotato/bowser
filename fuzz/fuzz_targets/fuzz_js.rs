//! Fuzz the JavaScript engine (`bowser_js`): lexer, parser, compiler,
//! bytecode VM, and the GC's raw-pointer surface (`gc.rs` `HandleScope`
//! plumbing, `vm.rs` `with_active` reentrancy).
//!
//! The input bytes are decoded as lossy UTF-8 and run as a classic script
//! in a fresh `Vm` per iteration (`Vm::new` + `Vm::eval`: parse, compile,
//! execute). A fresh interpreter per input keeps GC / interpreter state
//! from leaking across iterations, so a crash reproduces from the input
//! alone.
//!
//! Runaway scripts are bounded by the engine itself: every evaluation
//! starts with a fixed fuel budget (100M ops in `Interp`), and fuel
//! exhaustion surfaces as an ordinary `RangeError`, not a hang. Anything
//! past that is libfuzzer's `-timeout`, which turns a true hang into a
//! finding.
//!
//! Expected: a completion value or an ordinary `VmError` (parse /
//! compile / thrown). Panic / OOM / hang = bug.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Bound the source size so each iteration stays cheap; real scripts
    // past this are a corpus concern, not a parser/VM stress concern.
    let data = &data[..data.len().min(1 << 20)];
    let source = String::from_utf8_lossy(data);

    let mut vm = bowser_js::Vm::new();
    let _ = vm.eval(&source);
});
