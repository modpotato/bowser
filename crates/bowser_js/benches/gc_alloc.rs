//! GC allocation-throughput baseline (std only, hand-rolled harness).
//!
//! The collector (`crates/bowser_js/src/gc.rs`) allocates heap strings
//! from per-class chunk arenas as coalesced header+payload blocks, with a
//! malloc fallback for oversize blocks; objects come from the object
//! pages. Each case isolates one cost and keeps every other cost out:
//!
//! * `str_heap_alloc`: `Heap::alloc_str` on 60-byte payloads (longer than
//!   `STR_INLINE_MAX`, so they always take the heap path; `alloc_str_heap`
//!   itself is crate-private and is exactly what `alloc_str` calls here).
//! * `str_owned_alloc`: `Heap::alloc_str_owned` on a 60-byte `String`
//!   (arena-copy path: the payload is copied into the string block; the
//!   earlier `realloc` buffer-adopt fast path was reverted as a measured
//!   tie on glibc).
//! * `intern_hit`: `Heap::intern` on an already-canonical long string
//!   (pure intern-table lookup, no allocation).
//! * `intern_miss`: `Heap::intern` on a fresh string every call (allocates
//!   one `GcStr` per op, like `alloc_str` plus the table insert).
//! * `object_alloc`: object allocation driven through `eval`
//!   (`Heap::alloc_object` takes the crate-private `Object`, so the direct
//!   call is unreachable from outside the crate; each iteration allocates
//!   one fresh object literal plus two properties, kept alive by
//!   overwriting a single global so the per-iteration store is O(1) —
//!   pushing into an array would add an O(n) push to the measurement).
//! * `collect_cold`: full `Heap::collect()` pause on a synthetic heap with
//!   a known live set (20k linked objects, each holding a heap string)
//!   plus 20k unreachable garbage objects: mark + sweep.
//! * `collect_steady`: same live set with no fresh garbage (mark-only
//!   pause).
//!
//! `collect_cold_mem` / `collect_steady_mem` print the heap's live-byte
//! accounting (`Heap::live_bytes()` plus the per-arena breakdown) after the
//! final collection of each case, so memory work can be compared in bytes
//! across runs: run the bench before/after and diff those lines.
//!
//! `concat_str` writes its operands straight into the arena block
//! instead of staging through a buffer (one copy of the operands, kept);
//! the old coalescing trade-off (one fewer allocation — no separate
//! payload buffer — for one extra copy, measured +13% on
//! `str_owned_alloc`) is not recovered: a `realloc`-adopt fast path for
//! `alloc_wtf8_owned` was benchmarked and reverted as a measured tie
//! with the arena-copy path on glibc — the realloc's arena lock costs
//! more than the lock-free tcache free it replaced, so the unsafe
//! complexity buys nothing.
//!
//! The GC's automatic threshold is disabled for the whole run
//! (`set_gc_threshold(usize::MAX)`), so collections happen only where a
//! case explicitly calls `collect()`. Each case gets one warmup call, then
//! the best ns/op of 3 trials targeting ~0.25 s of work; per-case cleanup
//! (freeing that trial's garbage, rebuilding the next trial's garbage)
//! runs between trials, never inside the timed region.
//!
//! Run: `cargo bench -p bowser_js --bench gc_alloc`

use bowser_js::Vm;
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Time `f` adaptively: scale iterations until a trial takes >= 0.25 s or
/// `max_iters` is hit, then take the best ns/iter of 3 trials. `between`
/// runs once after warmup and once after every trial (cleanup / rebuild),
/// never inside the timed region.
fn bench_with(
    vm: &mut Vm,
    mut f: impl FnMut(&mut Vm),
    mut between: impl FnMut(&mut Vm),
    max_iters: u64,
) -> f64 {
    f(vm); // warm up: caches, shape tables, allocator paths
    between(vm);
    let mut iters = 1u64;
    loop {
        let start = Instant::now();
        for _ in 0..iters {
            f(vm);
        }
        let elapsed = start.elapsed();
        between(vm);
        if elapsed >= Duration::from_millis(250) || iters >= max_iters {
            let mut best = elapsed.as_secs_f64() / iters as f64;
            for _ in 0..2 {
                let start = Instant::now();
                for _ in 0..iters {
                    f(vm);
                }
                let ns = start.elapsed().as_secs_f64() / iters as f64;
                between(vm);
                if ns < best {
                    best = ns;
                }
            }
            return best * 1e9;
        }
        iters *= 2;
    }
}

/// `n` distinct strings, each 60 bytes (well above `STR_INLINE_MAX = 15`),
/// so every `alloc_str` / `intern` miss takes the heap path. Sized above
/// the largest iteration cap so the `intern_miss` case never wraps around
/// to an already-interned string within a trial.
fn payload_strings(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| format!("gc-bench-payload-{i:06}-abcdefghijklmnopqrstuvwxyz0123456789"))
        .collect()
}

fn mop(ns_per_op: f64) -> f64 {
    1e3 / ns_per_op
}

/// Print the heap's live-byte accounting from the most recent `collect()`:
/// total plus the per-arena breakdown (objects, strings, symbols, bigints).
/// Lets a wave's perf work be measured in bytes, not just wall-clock:
/// run the bench before/after and diff these lines.
fn print_live_bytes(vm: &Vm, label: &str) {
    let b = vm.heap().arena_bytes();
    println!(
        "{label:<28} live bytes: {:>10} total  (objects {:>9}, strings {:>9}, symbols {:>6}, bigints {:>6})",
        b.total(),
        b.objects,
        b.strings,
        b.symbols,
        b.bigints
    );
}

fn main() {
    let mut vm = Vm::new();
    // No automatic collections: every collect in this run is explicit.
    vm.heap_mut().set_gc_threshold(usize::MAX);

    println!("=== GC alloc throughput (auto-GC disabled) ===");

    // --- alloc_str: fresh heap string per op (the alloc_str_heap path) ---
    let strs = payload_strings(8192);
    let mut idx = 0usize;
    let ns = bench_with(
        &mut vm,
        |vm| {
            let s = &strs[idx % strs.len()];
            idx += 1;
            black_box(vm.heap_mut().alloc_str(black_box(s)).unwrap());
        },
        |vm| vm.heap_mut().collect(), // free the trial's garbage strings
        5_000,
    );
    println!(
        "{:<28} {:>12.1} ns/op  ({:.2} Mop/s)",
        "str_heap_alloc",
        ns,
        mop(ns)
    );

    // --- alloc_str_owned: move a String buffer into the heap ---
    let mut idx = 0usize;
    let ns = bench_with(
        &mut vm,
        |vm| {
            let s = strs[idx % strs.len()].clone();
            idx += 1;
            black_box(vm.heap_mut().alloc_str_owned(black_box(s)).unwrap());
        },
        |vm| vm.heap_mut().collect(),
        5_000,
    );
    println!(
        "{:<28} {:>12.1} ns/op  ({:.2} Mop/s)",
        "str_owned_alloc",
        ns,
        mop(ns)
    );

    // --- intern hit: canonical string already in the table ---
    let canon = strs[0].clone();
    vm.heap_mut().intern(&canon).unwrap();
    let ns = bench_with(
        &mut vm,
        |vm| {
            black_box(vm.heap_mut().intern(black_box(&canon)).unwrap());
        },
        |_| {},
        50_000, // no allocation on a hit: safe to run many iters
    );
    println!(
        "{:<28} {:>12.1} ns/op  ({:.2} Mop/s)",
        "intern_hit",
        ns,
        mop(ns)
    );

    // --- intern miss: fresh string every call ---
    // NB: the intern table is weak but entries are only evicted by a
    // collection, and auto-GC is disabled, so the payload pool must hold
    // more distinct strings than the iteration cap — otherwise the pool
    // would wrap around and later iterations would (incorrectly) hit.
    let mut idx = 0usize;
    let ns = bench_with(
        &mut vm,
        |vm| {
            let s = &strs[idx % strs.len()];
            idx += 1;
            black_box(vm.heap_mut().intern(black_box(s)).unwrap());
        },
        |vm| vm.heap_mut().collect(), // evict the dead weak-table entries
        5_000,
    );
    println!(
        "{:<28} {:>12.1} ns/op  ({:.2} Mop/s)",
        "intern_miss",
        ns,
        mop(ns)
    );

    // --- object allocation, driven through eval (alloc_object is
    // --- crate-private, so this goes through object literals). The sink is
    // --- a single overwritten global: O(1) per iteration, so the measured
    // --- cost is the object literal + two property definitions, not the
    // --- sink.
    const OBJ_N: u32 = 1_000;
    vm.eval(
        "globalThis.__mkObj = function(n){ \
             for (var i = 0; i < n; i++) { globalThis.__sink = {a: i, b: i + 1}; } \
         };",
    )
    .unwrap();
    let call = format!("globalThis.__mkObj({OBJ_N})");
    let ns = bench_with(
        &mut vm,
        |vm| {
            black_box(vm.eval(black_box(&call)).unwrap());
        },
        |vm| vm.heap_mut().collect(), // free the trial's objects
        200,
    );
    let ns_per_obj = ns / f64::from(OBJ_N);
    println!(
        "{:<28} {:>12.1} ns/object  ({:.2} Mobj/s, {OBJ_N} objects/eval)",
        "object_alloc",
        ns_per_obj,
        mop(ns_per_obj)
    );

    // --- synthetic heap: known live set + garbage, then collect() ---
    const LIVE: u32 = 20_000;
    const GARBAGE: u32 = 20_000;
    vm.eval(&format!(
        "globalThis.__live = []; \
         for (var i = 0; i < {LIVE}; i++) {{ \
             globalThis.__live.push({{idx: i, tag: 'live-object-tag-padding-' + i, next: null}}); \
         }} \
         for (var i = 0; i < {LIVE} - 1; i++) {{ \
             globalThis.__live[i].next = globalThis.__live[i + 1]; \
         }} \
         globalThis.__mkGarbage = function(n) {{ \
             for (var i = 0; i < n; i++) {{ \
                 var t = {{idx: i, tag: 'garbage-object-tag-padding-' + i, kids: [{{a: 1}}, {{b: 2}}]}}; \
                 if (t.idx < -1) {{ globalThis.__never = t; }} \
             }} \
         }};"
    ))
    .unwrap();
    let live_count = vm.heap().alloc_count();
    println!("live set: {LIVE} linked objects ({live_count} total allocations incl. builtins)");

    // Cold collect: fresh garbage before every trial (mark + sweep).
    let garbage_src = format!("globalThis.__mkGarbage({GARBAGE})");
    let ns = bench_with(
        &mut vm,
        |vm| {
            vm.heap_mut().collect();
        },
        |vm| {
            black_box(vm.eval(black_box(&garbage_src)).unwrap());
        },
        30,
    );
    let after_build = vm.heap().alloc_count();
    // One last collect so the final count below proves the sweep reclaimed
    // exactly the garbage (the last trial's `between` rebuilt garbage that
    // no timed collect has freed yet).
    vm.heap_mut().collect();
    let after = vm.heap().alloc_count();
    println!(
        "{:<28} {:>12.1} ns/collect  ({LIVE} live + {GARBAGE} garbage objects; \
         {:>8.1} ns/live-obj, heap {after_build} -> {after} allocs)",
        "collect_cold",
        ns,
        ns / f64::from(LIVE)
    );
    print_live_bytes(&vm, "collect_cold_mem");

    // Steady collect: no fresh garbage (mark-only pause on the live set).
    let ns = bench_with(
        &mut vm,
        |vm| {
            vm.heap_mut().collect();
        },
        |_| {},
        30,
    );
    println!(
        "{:<28} {:>12.1} ns/collect  ({LIVE} live objects, no garbage; {:>8.1} ns/live-obj)",
        "collect_steady",
        ns,
        ns / f64::from(LIVE)
    );
    print_live_bytes(&vm, "collect_steady_mem");
}
