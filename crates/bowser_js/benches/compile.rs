//! Parse+compile scaling benches (std only, hand-rolled harness).
//!
//! The compiler pipeline is crate-private, so this measures it through the
//! public surface: `eval` of a script that defines (but never calls) one
//! generated function. Script runtime is just the function-object creation,
//! so the measured time is lexer + parser + compiler (+ template intern).
//! Reports total ns per eval and ns per source byte vs. source size
//! (best of 3 trials, each targeting ~0.25 s of work).
//!
//! Run: `cargo bench -p bowser_js --bench compile`

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

/// Generate a function with `n` simple statements in its body.
fn gen_source(n: usize) -> String {
    let mut src = String::with_capacity(n * 16 + 64);
    src.push_str("function __big() {\n  var x = 0;\n");
    for i in 0..n {
        src.push_str(&format!("  x = x + {i};\n"));
    }
    src.push_str("  return x;\n}\n");
    src
}

fn case(vm: &mut Vm, name: &str, stmts: usize) {
    let src = gen_source(stmts);
    let bytes = src.len();
    let ns = bench(|| {
        black_box(vm.eval(black_box(&src)).unwrap());
    });
    println!(
        "{name:<28} {ns:>14.1} ns/eval  ({bytes:>7} bytes, {:>8.1} ns/byte)",
        ns / bytes as f64
    );
}

fn main() {
    let mut vm = Vm::new();
    println!("=== parse+compile vs source size ===");
    case(&mut vm, "small_100_stmts", 100);
    case(&mut vm, "medium_2000_stmts", 2_000);
    case(&mut vm, "large_20000_stmts", 20_000);
}
