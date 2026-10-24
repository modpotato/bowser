//! Benchmark harness for the JS VM opcode dispatch loop in
//! `crates/bowser_js/src/vm.rs` (`Interp::run` / `exec_op`). Runs
//! dispatch-heavy JS snippets end-to-end through `Vm::eval` and reports
//! ns/op. Not part of the shipped product; used for optimization work only.

use bowser_js::Vm;
use std::time::Instant;

fn bench(name: &str, iters: u64, prog: &str) {
    // Warm up once so install/compile costs don't pollute the measurement.
    let mut vm = Vm::new();
    vm.eval(prog)
        .unwrap_or_else(|e| panic!("bench {name} failed: {e:?}"));

    let mut best = u64::MAX;
    for _ in 0..5 {
        let mut vm = Vm::new();
        let start = Instant::now();
        vm.eval(prog)
            .unwrap_or_else(|e| panic!("bench {name} failed: {e:?}"));
        let ns = start.elapsed().as_nanos() as u64;
        best = best.min(ns);
    }
    let per = best as f64 / iters as f64;
    println!("{name:<28} {per:12.1} ns/op  ({iters} iters)");
}

fn main() {
    // Pure dispatch: integer arithmetic loop. Per iteration the loop body
    // dispatches ~10 ops (loads, add, compare, conditional jump).
    bench(
        "arith-loop",
        2_000_000,
        "let s = 0; for (let i = 0; i < 2000000; i++) { s = s + i; } s;",
    );
    // Branchy loop: forces JumpIfTrue/JumpIfFalse plus arithmetic.
    bench(
        "branch-loop",
        1_000_000,
        "let s = 0; for (let i = 0; i < 1000000; i++) { if (i % 2 === 0) { s = s + i; } else { s = s - i; } } s;",
    );
    // Call dispatch: one Call/Return pair per iteration.
    bench(
        "call-loop",
        200_000,
        "function f(x) { return x + 1; } let s = 0; for (let i = 0; i < 200000; i++) { s = f(s); } s;",
    );
}
