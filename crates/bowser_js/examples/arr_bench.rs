//! Benchmark harness for the array/object builtins in
//! `builtin_object_array.rs`. Runs JS snippets end-to-end through `Vm::eval`
//! and reports ns/op. Not part of the shipped product; used for optimization
//! work only.

use bowser_js::Vm;
use std::time::Instant;

fn bench(name: &str, iters: u64, setup: &str, body: &str) {
    // Warm up once so install/compile costs don't pollute the measurement.
    let mut vm = Vm::new();
    let prog =
        format!("{setup}; let __t0 = 0; for (let __i = 0; __i < {iters}; __i++) {{ {body} }}");
    vm.eval(&prog)
        .unwrap_or_else(|e| panic!("bench {name} failed: {e:?}"));

    let mut best = u64::MAX;
    for _ in 0..5 {
        let mut vm = Vm::new();
        let start = Instant::now();
        vm.eval(&prog)
            .unwrap_or_else(|e| panic!("bench {name} failed: {e:?}"));
        let ns = start.elapsed().as_nanos() as u64;
        best = best.min(ns);
    }
    let per = best as f64 / iters as f64;
    println!("{name:<28} {per:12.1} ns/op  ({iters} iters)");
}

fn main() {
    // Array constructor + push: the hottest path (per-push property scans).
    bench(
        "push-1000",
        20,
        "",
        "let a = []; for (let j = 0; j < 1000; j++) a.push(j); __t0 += a.length;",
    );
    // Iteration builtins over a dense 1000-element array.
    bench(
        "map-1000",
        20,
        "let m = []; for (let j = 0; j < 1000; j++) m.push(j); function dbl(x) { return x * 2; }",
        "__t0 += m.map(dbl).length;",
    );
    bench(
        "filter-1000",
        20,
        "let f = []; for (let j = 0; j < 1000; j++) f.push(j); function ev(x) { return x % 2 == 0; }",
        "__t0 += f.filter(ev).length;",
    );
    bench(
        "slice-1000",
        50,
        "let s = []; for (let j = 0; j < 1000; j++) s.push(j);",
        "__t0 += s.slice(100, 900).length;",
    );
    bench(
        "join-1000",
        50,
        "let jn = []; for (let j = 0; j < 1000; j++) jn.push(j);",
        "__t0 += jn.join(',').length;",
    );
    bench(
        "indexOf-1000",
        50,
        "let io = []; for (let j = 0; j < 1000; j++) io.push(j);",
        "__t0 += io.indexOf(999);",
    );
    bench(
        "includes-1000",
        50,
        "let ic = []; for (let j = 0; j < 1000; j++) ic.push(j);",
        "if (ic.includes(999)) __t0++;",
    );
    bench(
        "splice-100",
        100,
        "",
        "let sp = []; for (let j = 0; j < 100; j++) sp.push(j); sp.splice(50, 10, 1, 2, 3); __t0 += sp.length;",
    );
    bench(
        "shift-200",
        30,
        "",
        "let sh = []; for (let j = 0; j < 200; j++) sh.push(j); while (sh.length) sh.shift(); __t0++;",
    );
    bench(
        "concat-200",
        50,
        "let c1 = []; for (let j = 0; j < 200; j++) c1.push(j); let c2 = []; for (let j = 0; j < 200; j++) c2.push(j);",
        "__t0 += c1.concat(c2).length;",
    );
    // Number -> string: js_number_to_string path via join of floats.
    bench(
        "join-floats-500",
        50,
        "let jf = []; for (let j = 0; j < 500; j++) jf.push(j + 0.5);",
        "__t0 += jf.join(',').length;",
    );
    // Object builtins.
    bench(
        "object-keys-200",
        50,
        "",
        "let o = {}; for (let j = 0; j < 200; j++) o['k' + j] = j; __t0 += Object.keys(o).length;",
    );
    bench(
        "object-assign-200",
        50,
        "let sa = {}; for (let j = 0; j < 200; j++) sa['k' + j] = j;",
        "__t0 += Object.assign({}, sa).k199;",
    );
    bench(
        "array-from-500",
        50,
        "let al = { length: 500 }; for (let j = 0; j < 500; j++) al[j] = j;",
        "__t0 += Array.from(al).length;",
    );
    // Sparse / hole handling.
    bench(
        "sparse-slice-1000",
        50,
        "let sp2 = new Array(1000); sp2[0] = 1; sp2[999] = 2;",
        "__t0 += sp2.slice(0, 1000).length;",
    );
}
