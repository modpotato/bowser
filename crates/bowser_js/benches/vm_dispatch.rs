//! VM op-dispatch throughput benches (std only, hand-rolled harness).
//!
//! Each case defines its workload once on a shared [`Vm`], then times
//! repeated `eval("<fn>(INNER)")` calls. The per-call parse+compile of the
//! tiny call wrapper is constant overhead; the measured time is dominated
//! by VM dispatch of the inner loop body. Reports ns per inner-loop
//! iteration (best of 3 trials, each targeting ~0.25 s of work).
//!
//! Run: `cargo bench -p bowser_js --bench vm_dispatch`

use bowser_js::Vm;
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Time `f` adaptively: scale iterations until a trial takes >= 0.25 s,
/// then take the best ns/iter of 3 trials.
fn bench(mut f: impl FnMut()) -> f64 {
    // Warm up.
    f();
    let mut iters = 1u64;
    loop {
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        let elapsed = start.elapsed();
        if elapsed >= Duration::from_millis(250) || iters >= 1_000 {
            let mut best = elapsed.as_secs_f64() / iters as f64;
            for _ in 0..2 {
                let start = Instant::now();
                for _ in 0..iters {
                    f();
                }
                let ns = start.elapsed().as_secs_f64() / iters as f64;
                if ns < best {
                    best = ns;
                }
            }
            return best * 1e9;
        }
        iters *= 2;
    }
}

fn main() {
    // Inner loop trip count per measured eval call: large enough that the
    // wrapper `eval("...()")` parse+compile is noise.
    const INNER: u32 = 20_000;

    let mut vm = Vm::new();
    // NB: top-level `function` declarations live in a per-script scope in
    // this engine, so the workloads are installed on `globalThis` to
    // survive across the repeated eval calls below.
    vm.eval(
        "globalThis.__arith = function(n){ var s = 0; \
         for (var i = 0; i < n; i++) { s = s + i * 2 - (i >> 1) + (i ^ 3); } \
         return s; };",
    )
    .unwrap();
    vm.eval(
        "globalThis.__obj = { x: 1, y: 2 }; \
         globalThis.__prop = function(n){ var s = 0; \
         for (var i = 0; i < n; i++) { globalThis.__obj.x = globalThis.__obj.x + 1; s = s + globalThis.__obj.x + globalThis.__obj.y; } \
         return s; };",
    )
    .unwrap();
    vm.eval(
        "globalThis.__add = function(a, b){ return a + b; }; \
         globalThis.__call = function(n){ var s = 0; \
         for (var i = 0; i < n; i++) { s = globalThis.__add(s, i); } \
         return s; };",
    )
    .unwrap();

    println!("=== VM dispatch (inner iters/call: {INNER}) ===");

    let call = format!("globalThis.__arith({INNER})");
    let ns = bench(|| {
        black_box(vm.eval(black_box(&call)).unwrap());
    });
    println!(
        "{:<28} {:>12.1} ns/loop-iter  ({:.2} Miter/s)",
        "arith_loop",
        ns / INNER as f64,
        INNER as f64 / ns * 1e3
    );

    let call = format!("globalThis.__prop({INNER})");
    let ns = bench(|| {
        black_box(vm.eval(black_box(&call)).unwrap());
    });
    println!(
        "{:<28} {:>12.1} ns/loop-iter  ({:.2} Miter/s)",
        "property_loop",
        ns / INNER as f64,
        INNER as f64 / ns * 1e3
    );

    let call = format!("globalThis.__call({INNER})");
    let ns = bench(|| {
        black_box(vm.eval(black_box(&call)).unwrap());
    });
    println!(
        "{:<28} {:>12.1} ns/loop-iter  ({:.2} Miter/s)",
        "call_loop",
        ns / INNER as f64,
        INNER as f64 / ns * 1e3
    );
}
