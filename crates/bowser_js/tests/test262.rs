//! test262 conformance runner — scoreboard for `bowser_js`.
//!
//! The scoreboard test scans ~54k files, so it is gated behind `TEST262_RUN`;
//! without `TEST262_RUN=1` it prints a one-line skip notice and returns Ok.
//!
//! # Environment knobs
//!
//! - `TEST262_RUN=1`: run the scoreboard (default: skip with a notice).
//! - `TEST262_FILTER`: path substring; run only matching files, e.g.
//!   `TEST262_FILTER=test/language/statements/try`.
//! - `TEST262_EXCLUDE`: comma-separated path substrings to skip, e.g.
//!   `TEST262_EXCLUDE=tco-non-eval-global` (applied after `TEST262_FILTER`).
//! - `TEST262_JOBS`: worker-thread count (default: available parallelism).
//! - `TEST262_CACHE`: path of the incremental outcome-cache file
//!   (default: under the crate target dir). Delete it to force a full
//!   re-run. The cache is checkpointed atomically after every chunk, so a
//!   killed/restarted run resumes from the last chunk, not from scratch.
//! - `TEST262_DEBUG_RESET`: enable `eprintln` diagnostics from
//!   [`Evaluator::reset`] when a prototype scrub is incomplete.
//! - `TEST262_FORCE_DEBUG=1`: silence the debug-build warning (see below)
//!   while still running the suite in debug.
//! - `TEST262_SMOKE=1`: run the curated smoke subset instead of the full
//!   scan (see "Smoke subset"). Applies before `TEST262_FILTER` /
//!   `TEST262_EXCLUDE`, which further refine it.
//! - `TEST262_TIMING=1`: instrument per-test wall time into phases
//!   (read, frontmatter, reset, eval sloppy, eval strict, drain) and print
//!   aggregate phase totals plus the top-20 slowest tests in the report.
//! - `TEST262_QUICK=1`: halve `PER_TEST_TIMEOUT` and `WATCHDOG_TIMEOUT`.
//!   Pathological slow tests may flip from pass/fail to `timeout` under
//!   this profile; do not compare its scores against full-timeout runs.
//!   The timeout profile is hashed into the outcome-cache fingerprint, so
//!   quick runs neither read nor pollute the full-profile cache entries.
//!
//! # Recommended dev loop
//!
//! Iterate in release mode on the smoke subset with a tight filter, e.g.
//!
//! ```sh
//! TEST262_RUN=1 TEST262_SMOKE=1 TEST262_FILTER=test/language/statements/try \
//!   cargo test --release -p bowser_js --test test262 -- --nocapture
//! ```
//!
//! (`--nocapture` shows the scoreboard; without it cargo swallows stdout.)
//!
//! For a full-suite run use release mode; the debug interpreter is roughly
//! two orders of magnitude slower per op and the suite is unusable in
//! debug. Running under `cfg!(debug_assertions)` prints a prominent
//! warning; set `TEST262_FORCE_DEBUG=1` to silence it (the run still
//! proceeds).
//!
//! ```sh
//! TEST262_RUN=1 cargo test --release -p bowser_js --test test262 -- --nocapture
//! ```
//!
//! # Smoke subset (`TEST262_SMOKE=1`)
//!
//! Deterministic, ~2k files, sampled from the sorted file list so the same
//! subset is selected on every run:
//!
//! - `test/language`: every 16th file (`SMOKE_LANGUAGE_STRIDE`), ~1.5k
//!   files — broad language-feature coverage without the 24k-file scan.
//! - `test/built-ins`: every 40th file (`SMOKE_BUILTINS_STRIDE`), ~0.6k
//!   files.
//! - `test/annexB`, `test/intl402`, `test/staging`, `test/harness`:
//!   excluded from the smoke subset; reach them with `TEST262_FILTER`.
//!
//! # Architecture
//!
//! ```text
//! test262/test/**/*.js
//!   -> collect_tests        (recursive walk, *.js only)
//!   -> HarnessCache         (harness/*.js preloaded once, shared)
//!   -> thread pool          (std::thread::scope, one worker per core)
//!   -> Evaluator thread     (one Vm per worker, reused across tests)
//!   -> scoreboard report    (merged; same format as the single-threaded runner)
//! ```
//!
//! # Why one Vm per worker (and why reuse is sound here)
//!
//! Constructing a [`bowser_js::Vm`] installs every builtin (~33ms in
//! debug); doing that per test-mode dominated the old runner. Each worker
//! thread owns one long-lived [`Evaluator`]: a single `Vm` reused across
//! its tests, with [`Evaluator::reset`] scrubbing global state between
//! tests.
//!
//! Reuse is sound because of how this engine scopes `eval`: top-level
//! `var`/`let`/`const`/`function`/`class` declarations in an `eval`'d
//! script live in a per-eval environment, never on the shared global
//! object (a later `eval` cannot even name them). The only things that can
//! leak across evals are:
//!
//! - sloppy-mode assignments to undeclared names (`x = 1`,
//!   `globalThis.y = 2`): enumerable own properties of `globalThis`,
//!   removed by [`Evaluator::reset`] via the native key scan;
//! - overwrites of builtin globals (`Array = 5`): detected by
//!   [`Evaluator::reset`] against a baseline snapshot and handled by
//!   discarding the `Vm` for a fresh one (restoring via `set_global`
//!   would panic on a global the test made non-configurable);
//! - additions to builtin prototypes / namespace objects
//!   (`Array.prototype.foo = 1`, `Math.bar = 2`): added keys are deleted
//!   and overwritten data properties restored by the native prototype
//!   scrub in reset; a replaced or reparented prototype object rebuilds
//!   the `Vm`;
//! - the module registry (`"\0bowser_module_registry"`): deleted in reset;
//! - the `$DONE_*` protocol globals: re-seeded per async test, deleted by
//!   reset;
//! - stashed event-loop task errors: drained in reset;
//! - microtasks queued but never yielded to: drained at the end of each
//!   test, so they cannot run during the next test's event-loop drain.
//!
//! What reset deliberately does NOT try to undo (documented limits):
//! pending far-future timers. Non-enumerable pollution *is* scrubbed: the
//! native [`Vm::own_property_keys`] reads the heap property lists
//! directly, so a test that clobbered `Reflect.ownKeys` cannot blind the
//! scan — no generated JS runs during reset at all. It drives both the
//! global key-set check and the prototype-scrub added-key scan, so
//! non-enumerable additions to builtin prototypes / namespace objects are
//! deleted and overwrites of non-enumerable builtin properties are
//! restored from baseline values. A property the scrub must delete or
//! restore that a test made non-configurable forces a `Vm` rebuild
//! instead.
//!
//! Two documented divergences from the old generated-JS reset: the native
//! value read never invokes getters (accessor values are not compared),
//! and get/set identity is presence-only (a getter/setter swap is
//! invisible). See `native_reset.rs`.
//!
//! # Watchdog
//!
//! Sync tests have no in-engine timeout besides fuel exhaustion (100M
//! ops; ~90s in debug for a tight infinite loop). Each worker therefore
//! talks to its evaluator thread over channels and waits with
//! [`WATCHDOG_TIMEOUT`]. If the evaluator does not answer in time, the
//! worker abandons it (the thread finishes on its own once fuel runs out
//! and then exits; its join handle is retained and joined at shutdown so
//! it is never dropped detached), spawns a fresh evaluator, and records
//! a `timeout`.
//! Async tests additionally keep their own [`PER_TEST_TIMEOUT`] drain
//! budget inside the evaluator, so a never-settling `$DONE` does not burn
//! an evaluator thread. The drain is adaptive: once the event loop is
//! quiescent (no pending timers/microtasks/macrotasks) with `$DONE`
//! uncalled, the test fails immediately instead of burning the budget —
//! no future JS can run, so `$DONE` can never arrive. The full budget is
//! kept only while work is actually pending.
//!
//! # Module tests
//!
//! Module-goal tests run through
//! `bowser_js::modules::{ModuleLoader, ModuleError}`: fresh loader per
//! test, network fetching disabled, synthetic `https://test262.invalid/`
//! base URL, relative imports pre-registered from disk, resolution-phase
//! errors mapped to the negative-phase verdict logic, module goal run
//! once because modules are intrinsically strict. The harness preamble
//! (assert.js, includes — skipped for `raw` tests — and the async
//! `$DONE` shim) is prepended to the module source so it is in scope for
//! the module body.
//!
//! # Skips
//!
//! Tests whose frontmatter `features` name a capability in
//! [`UNSUPPORTED_FEATURES`] are skipped without running. The engine has
//! no tail-call optimization and no JS call-depth guard, so the TCO
//! tests (100k-deep recursion expecting proper tail calls) would
//! overflow the evaluator thread's native stack and abort the whole
//! test binary instead of failing. Skips are reported by reason in the
//! scoreboard like any other outcome.
//!
//! # Memory
//!
//! The engine's automatic GC is opt-in (disabled by default), so a
//! `Vm` reused for thousands of tests accumulates garbage without
//! bound. The evaluator therefore rebuilds its `Vm` when its live heap
//! grows by [`VM_REBUILD_HEAP_BUDGET`] allocations (or every
//! [`VM_REBUILD_EVERY`] tests as a backstop), capping heap growth.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, mpsc};
use std::time::{Duration, Instant};

use bowser_js::{
    OwnDescriptor, ScriptHandle, Value, Vm, VmError, Wtf8, install_test262_helpers,
    modules::{ModuleError, ModuleLoader},
};

/// Native Phase-B reset (prototype scrub + global key scrub). The old
/// generated-JS reset evals are gone; see the module docs for the two
/// documented divergences (getter skipping, presence-only get/set).
#[path = "native_reset/mod.rs"]
mod native_reset;

/// How many individual failures to print in the report.
const MAX_FAILURES_SHOWN: usize = 25;
/// Wall-clock guard per test for the async `$DONE` drain (enforced inside
/// the evaluator thread; the evaluator survives). The drain fails fast
/// once the event loop is quiescent with `$DONE` uncalled, so the guard
/// only ever binds tests with work actually pending.
const PER_TEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Watchdog: wall-clock budget for one evaluator round-trip, covering sync
/// tests too. On expiry the worker abandons the evaluator thread and
/// records a `timeout`.
const WATCHDOG_TIMEOUT: Duration = Duration::from_secs(60);
/// Grace for joining watchdog-abandoned evaluator threads in
/// [`EvalHandle::shutdown`]. Abandoned evaluators terminate on their own
/// via fuel exhaustion (bounded work), so this is generous; a thread
/// still alive past it is stuck in a fuel-unbounded loop (e.g.
/// `run_until_idle` on self-rescheduling timers), which is a VM bug — it
/// is left detached rather than hanging the harness.
const ABANDONED_JOIN_GRACE: Duration = Duration::from_secs(30);
/// Harness files every test gets, in order, before its own `includes`.
const DEFAULT_INCLUDES: &[&str] = &["assert.js", "sta.js"];
/// test262 features that require engine capabilities Bowser does not
/// implement. Tests carrying one are skipped, never run:
/// `tail-call-optimization` tests recurse `$MAX_ITERATIONS` (100_000)
/// deep, which overflows the evaluator thread's native stack — an
/// uncatchable process abort, not a catchable test failure — so running
/// them could never produce a verdict.
const UNSUPPORTED_FEATURES: &[&str] = &["tail-call-optimization"];
/// Native stack for evaluator threads. JS call depth is bounded by the
/// engine (`Interp::max_frames`), and native recursion that bypasses it
/// (bound-function forwarding, structural `==`, `Array.prototype.flat`)
/// is capped by `MAX_NATIVE_RECURSION`-style guards that raise a
/// catchable `RangeError` instead of overflowing. 8 MiB matches the
/// parser/compiler depth-guard calibration and gives deep-but-finite
/// recursion headroom; the genuinely unbounded cases (the
/// [`UNSUPPORTED_FEATURES`] skips above) need infinitely more and
/// are skipped instead. Larger stacks backfire: deep tests commit the
/// whole stack, and watchdog-abandoned threads then leak it all.
const EVALUATOR_STACK_SIZE: usize = 8 * 1024 * 1024;
/// Rebuild the evaluator's `Vm` after this many jobs. The engine's
/// automatic GC is opt-in (disabled by default); without it the heap
/// grows without bound because one `Vm` is reused for thousands of
/// tests, and the full suite OOMs. This count backstop also bounds
/// retained compiled templates (the body cache flushes at
/// [`BODY_CACHE_MAX_ENTRIES`], but dropped templates stay interned in
/// the `Vm` until rebuild). Raised from 200: measured heap growth is
/// ~50 live allocations per job, so 1000 jobs is ~50k allocations
/// (~a few MB) — well within budget — while 5x fewer rebuilds saves the
/// builtin reinstall + preamble recompiles each rebuild costs.
/// [`VM_REBUILD_HEAP_BUDGET`] usually fires first at the same point;
/// this is the backstop for low-allocation stretches.
const VM_REBUILD_EVERY: u64 = 1000;
/// Live-heap allocation budget per evaluator lifetime (see
/// [`Heap::alloc_count`]). The engine's GC is opt-in/disabled, so the
/// heap only grows while a `Vm` is reused; when a `Vm` has accumulated
/// this many allocations beyond its post-build baseline it is rebuilt.
/// Set to ~1000 jobs' worth at the measured ~50 allocs/job, so it
/// agrees with [`VM_REBUILD_EVERY`] at normal growth and fires first on
/// allocation spikes. `TEST262_HEAP_STATS=1` logs growth at each rebuild
/// for re-tuning.
const VM_REBUILD_HEAP_BUDGET: usize = 50_000;
/// Every this many fast-path resets, re-run the epoch-covered scans
/// (baseline values + prototype targets, check-only) to validate the
/// heap's mutation-epoch bookkeeping: a miss means a write escaped the
/// epoch and the fast path skipped a needed scrub — a cross-test
/// contamination bug. Misses are counted in the scoreboard; the slow
/// path then repairs/rebuilds as usual, so a miss degrades to one slow
/// reset, never to silent pollution. Kept well below the rebuild
/// interval so it actually fires within one evaluator lifetime.
const RESET_VALIDATE_EVERY: u64 = 100;
/// Max entries in [`Evaluator::body_cache`]. The `Vm` rebuild every
/// [`VM_REBUILD_EVERY`] tests replaces the whole `Evaluator` (and its
/// cache) anyway, and a 1000-job epoch holds at most ~2000 distinct
/// (body, mode) assemblies, so this is a backstop, not a tuned limit:
/// on overflow the cache is flushed instead of growing without bound.
/// Each entry pins one compiled template in the current `Vm` plus one
/// shared `Arc<str>` (no new allocation: the key aliases the job's
/// body), so the bound caps template retention too.
const BODY_CACHE_MAX_ENTRIES: usize = 1024;
/// `TEST262_SMOKE=1` sampling strides (see "Smoke subset" in the header
/// docs). Deterministic: applied to the sorted file list, so the subset is
/// identical across runs.
const SMOKE_LANGUAGE_STRIDE: usize = 16;
const SMOKE_BUILTINS_STRIDE: usize = 40;
/// How many of the slowest tests [`print_report`] lists under
/// `TEST262_TIMING=1`.
const TOP_SLOWEST_SHOWN: usize = 20;

/// `TEST262_TIMING=1` enables per-phase wall-clock instrumentation.
/// Read once (lazily) so the common path pays a single atomic load per
/// test instead of an `env::var` per phase.
static TIMING_ENABLED: LazyLock<bool> =
    LazyLock::new(|| env::var("TEST262_TIMING").is_ok_and(|v| v == "1"));
/// `TEST262_QUICK=1` selects the quick timeout profile (halved budgets).
static QUICK: LazyLock<bool> = LazyLock::new(|| env::var("TEST262_QUICK").is_ok_and(|v| v == "1"));
/// `TEST262_HEAP_STATS=1` logs live-heap growth at each evaluator
/// rebuild, for tuning [`VM_REBUILD_HEAP_BUDGET`].
static HEAP_STATS: LazyLock<bool> =
    LazyLock::new(|| env::var("TEST262_HEAP_STATS").is_ok_and(|v| v == "1"));

/// Wall-clock guard per test for the async `$DONE` drain, effective value
/// after the `TEST262_QUICK` profile. Halved under quick: pathological
/// slow tests may flip to `timeout` there.
fn per_test_timeout() -> Duration {
    if *QUICK {
        PER_TEST_TIMEOUT / 2
    } else {
        PER_TEST_TIMEOUT
    }
}

/// Effective watchdog budget for one evaluator round-trip, after the
/// `TEST262_QUICK` profile. Halved under quick.
fn watchdog_timeout() -> Duration {
    if *QUICK {
        WATCHDOG_TIMEOUT / 2
    } else {
        WATCHDOG_TIMEOUT
    }
}

/// Per-phase wall-clock buckets (nanoseconds), one per executed test when
/// `TEST262_TIMING=1`, else all zeros. `read` and `frontmatter` are filled
/// on the worker thread in [`run_one`]; the rest are filled by the
/// evaluator thread in [`Evaluator::run_job`] and carried back in
/// [`JobResult`].
#[derive(Clone, Copy, Debug, Default)]
struct PhaseTimes {
    read: u64,
    frontmatter: u64,
    reset: u64,
    eval_sloppy: u64,
    eval_strict: u64,
    drain: u64,
}

impl PhaseTimes {
    fn add(&mut self, other: &PhaseTimes) {
        self.read += other.read;
        self.frontmatter += other.frontmatter;
        self.reset += other.reset;
        self.eval_sloppy += other.eval_sloppy;
        self.eval_strict += other.eval_strict;
        self.drain += other.drain;
    }

    fn total(&self) -> u64 {
        self.read + self.frontmatter + self.reset + self.eval_sloppy + self.eval_strict + self.drain
    }
}

/// Well-known global names whose values are snapshotted at baseline.
/// The global key set itself is enumerated with the native
/// [`Vm::own_property_keys`], which reads the heap property lists and sees
/// non-enumerable globals too, so key *additions* are caught by the key-set
/// check; this list exists to catch *overwrites* of non-enumerable builtins
/// (`baseline_values` covers the enumerable ones). Any change (or deletion)
/// rebuilds the `Vm`; the list is a superset of what the engine installs,
/// and unknown names are simply skipped.
const KNOWN_GLOBALS: &[&str] = &[
    "globalThis",
    "Infinity",
    "NaN",
    "undefined",
    "Array",
    "ArrayBuffer",
    "Atomics",
    "BigInt",
    "BigInt64Array",
    "BigUint64Array",
    "Boolean",
    "DataView",
    "Date",
    "Error",
    "EvalError",
    "FinalizationRegistry",
    "Float32Array",
    "Float64Array",
    "Function",
    "Int8Array",
    "Int16Array",
    "Int32Array",
    "Map",
    "Math",
    "Number",
    "Object",
    "Promise",
    "Proxy",
    "RangeError",
    "ReferenceError",
    "Reflect",
    "RegExp",
    "Set",
    "SharedArrayBuffer",
    "String",
    "Symbol",
    "SyntaxError",
    "TypeError",
    "URIError",
    "Uint8Array",
    "Uint8ClampedArray",
    "Uint16Array",
    "Uint32Array",
    "WeakMap",
    "WeakSet",
    "WeakRef",
    "AggregateError",
    "WebAssembly",
    "decodeURI",
    "decodeURIComponent",
    "encodeURI",
    "encodeURIComponent",
    "escape",
    "unescape",
    "eval",
    "isFinite",
    "isNaN",
    "parseFloat",
    "parseInt",
    "queueMicrotask",
    "structuredClone",
    "setTimeout",
    "clearTimeout",
    "setInterval",
    "clearInterval",
    "console",
    "fetch",
    "navigator",
    "process",
];

/// `$DONE` shim for async tests: the per-test token is baked into the
/// compiled shim as an integer literal, so a stale `$DONE` closure from
/// a previous test's leftover async work can never satisfy a later
/// test's guard — the literal differs even though the protocol globals
/// were re-seeded for the new test. `$DONE_token` still travels per
/// test as a global so a test that reassigns it no-ops its own `$DONE`
/// calls instead of completing (the old `$DONE_expected` twin global is
/// gone: there is nothing left for it to disagree with).
///
/// Compiled variants are cached per token in a tiny bounded cache (see
/// [`Evaluator::done_shim`]), so the ~20x precompile win is kept: the
/// sequential-token pattern compiles one shim variant and then hits the
/// cache for the rest of the evaluator's life. Only the `format!` runs
/// per async test — no per-test lex, parse, or compile.
fn done_shim_source(token: u32) -> String {
    format!(
        r#"
// ===== bowser test262 runner: $DONE protocol shim =====
// Real driver: run the source, then drain the event loop
// (EventLoop::run_until_idle) until $DONE_called is true or the
// per-test timeout elapses. Pass requires $DONE called with no error.
function $DONE(error) {{
  if ({token} !== $DONE_token) {{ return; }}
  if ($DONE_called) {{ throw new Test262Error("$DONE called more than once"); }}
  $DONE_called = true;
  $DONE_error = error;
}}
"#
    )
}

/// The `$DONE` token for an async job, or a job failure when it is
/// missing. `run_job` always mints a token for async jobs, so `None` here
/// is an unreachable harness bug — but a panic would abort the whole
/// test262 run, so the one job fails instead (`debug_assert!` still
/// trips in debug builds).
fn done_token_or_bail(done_token: Option<u32>) -> Result<u32, EngineError> {
    match done_token {
        Some(token) => Ok(token),
        None => {
            debug_assert!(false, "async job must carry a $DONE token");
            Err(EngineError::Thrown(
                "harness bug: async job missing its $DONE token".to_string(),
            ))
        }
    }
}

/// Maximum compiled `$DONE` shim variants kept in
/// [`Evaluator::done_shim_cache`]. Tokens are minted sequentially, so in
/// practice the cache holds the current window and almost never evicts;
/// the bound exists so a pathological token pattern cannot grow it
/// without limit.
const DONE_SHIM_CACHE_MAX: usize = 8;

// ---------------------------------------------------------------------------
// Frontmatter
// ---------------------------------------------------------------------------

/// Parsed test262 YAML frontmatter (hand-rolled subset parser; no serde).
#[derive(Debug, Default, Clone)]
struct FrontMatter {
    description: String,
    flags: Vec<String>,
    includes: Vec<String>,
    features: Vec<String>,
    negative_phase: Option<String>,
    negative_type: Option<String>,
    only_strict: bool,
    no_strict: bool,
    is_async: bool,
    is_module: bool,
    is_raw: bool,
    can_block_is_false: bool,
}

fn unquote(s: &str) -> &str {
    let b = s.as_bytes();
    if b.len() >= 2
        && ((b[0] == b'\'' && b[b.len() - 1] == b'\'') || (b[0] == b'"' && b[b.len() - 1] == b'"'))
    {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

/// Split a `key: value` line at the first colon.
fn split_kv(line: &str) -> Option<(&str, &str)> {
    let i = line.find(':')?;
    let (k, rest) = line.split_at(i);
    Some((k.trim(), rest[1..].trim()))
}

/// Parse `[a, b, c]` flow lists; `None` if not a flow list.
fn parse_flow_list(value: &str) -> Option<Vec<String>> {
    let inner = value.trim().strip_prefix('[')?.strip_suffix(']')?;
    if inner.trim().is_empty() {
        return Some(Vec::new());
    }
    Some(
        inner
            .split(',')
            .map(|s| unquote(s.trim()).to_string())
            .collect(),
    )
}

fn push_list(fm: &mut FrontMatter, key: &str, item: &str) {
    let item = unquote(item.trim()).to_string();
    match key {
        "flags" => fm.flags.push(item),
        "includes" => fm.includes.push(item),
        "features" => fm.features.push(item),
        _ => {}
    }
}

/// Parse the `/*--- ... ---*/` frontmatter block. Returns defaults when the
/// block is absent or malformed.
fn parse_frontmatter(src: &str) -> FrontMatter {
    let mut fm = FrontMatter::default();
    let Some(open) = src.find("/*---") else {
        return fm;
    };
    let after = &src[open + 5..];
    let Some(close) = after.find("---*/") else {
        return fm;
    };
    let body = &after[..close];

    // Keys whose `- item` children we collect.
    let mut list_key: Option<&str> = None;
    // Inside `negative:` nested map.
    let mut negative = false;
    let mut negative_indent = 0usize;
    // Lines deeper than this indent belong to a block scalar / unknown
    // nested map and are skipped.
    let mut skip_indent: Option<usize> = None;

    for line in body.lines() {
        let indent = line.len() - line.trim_start_matches(' ').len();
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if let Some(si) = skip_indent {
            if indent > si {
                continue;
            }
            skip_indent = None;
        }
        if negative {
            if indent > negative_indent {
                if let Some((k, v)) = split_kv(t) {
                    match k {
                        "phase" => fm.negative_phase = Some(unquote(v).to_string()),
                        "type" => fm.negative_type = Some(unquote(v).to_string()),
                        _ => {}
                    }
                }
                continue;
            }
            negative = false;
        }
        if let Some(item) = t.strip_prefix("- ") {
            if let Some(key) = list_key {
                push_list(&mut fm, key, item);
            }
            continue;
        }
        // A new key line ends any pending block list.
        list_key = None;
        let Some((k, v)) = split_kv(t) else { continue };
        match k {
            "flags" | "includes" | "features" => {
                if v.is_empty() {
                    list_key = Some(k); // block list follows
                } else if let Some(items) = parse_flow_list(v) {
                    for item in items {
                        push_list(&mut fm, k, &item);
                    }
                } else if v == "|" || v == ">" {
                    skip_indent = Some(indent);
                }
                // Any other scalar after these keys is ignored.
            }
            "negative" if v.is_empty() => {
                negative = true;
                negative_indent = indent;
            }
            "description" => {
                fm.description = unquote(v.trim_end_matches('|').trim()).to_string();
            }
            _ => {
                // Unknown key with a nested/block value: skip its children.
                if v.is_empty() || v == "|" || v == ">" {
                    skip_indent = Some(indent);
                }
            }
        }
    }

    for flag in &fm.flags {
        match flag.as_str() {
            "onlyStrict" => fm.only_strict = true,
            "noStrict" => fm.no_strict = true,
            "async" => fm.is_async = true,
            "module" => fm.is_module = true,
            "raw" => fm.is_raw = true,
            "CanBlockIsFalse" => fm.can_block_is_false = true,
            _ => {}
        }
    }
    fm
}

// ---------------------------------------------------------------------------
// Pipeline stage availability
// ---------------------------------------------------------------------------

/// Pipeline stages in order, with the on-disk source file each would live in.
const STAGES: &[(&str, &str)] = &[
    ("lexer", "lexer.rs"),
    ("parser", "parser.rs"),
    ("compiler", "compiler.rs"),
    ("vm", "vm.rs"),
    ("event_loop", "event_loop.rs"),
];

/// How reachable a pipeline stage is from this integration test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StageStatus {
    /// Declared `pub` in `lib.rs`: callable from the harness.
    Public,
    /// Declared in `lib.rs` but not `pub`: compiled, unreachable from here.
    Private,
    /// Source file exists in `src/` but is not declared: not compiled.
    Unwired,
    /// No source file at all.
    Missing,
}

impl StageStatus {
    fn label(self) -> &'static str {
        match self {
            StageStatus::Public => "public (harness-usable)",
            StageStatus::Private => "compiled but private",
            StageStatus::Unwired => "on disk, NOT wired into crate",
            StageStatus::Missing => "missing",
        }
    }
}

fn stage_status(lib_rs: &str, src_dir: &Path, stage: &str, file: &str) -> StageStatus {
    let public = lib_rs.contains(&format!("pub mod {stage}"))
        || lib_rs.contains(&format!("pub use {stage}::"));
    if public {
        return StageStatus::Public;
    }
    let declared =
        lib_rs.contains(&format!("mod {stage}")) || lib_rs.contains(&format!("use {stage}::"));
    if declared {
        return StageStatus::Private;
    }
    if src_dir.join(file).is_file() {
        return StageStatus::Unwired;
    }
    StageStatus::Missing
}

/// Probe the crate on disk: for each [`STAGES`] entry, how reachable is it?
/// Returns `None` when the crate sources cannot be found.
fn stage_report() -> Option<Vec<(&'static str, StageStatus)>> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").ok()?;
    let src_dir = PathBuf::from(manifest_dir).join("src");
    let lib_rs = fs::read_to_string(src_dir.join("lib.rs")).ok()?;
    Some(
        STAGES
            .iter()
            .map(|(stage, file)| (*stage, stage_status(&lib_rs, &src_dir, stage, file)))
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Shared harness state
// ---------------------------------------------------------------------------

/// Harness files preloaded once and shared by all workers. `Vm` reuse
/// means the per-test disk reads of the old runner become one load.
#[derive(Debug, Default)]
struct HarnessCache {
    files: HashMap<String, Arc<str>>,
}

impl HarnessCache {
    fn load(harness_dir: &Path) -> Result<Self, String> {
        let mut files = HashMap::new();
        let entries = fs::read_dir(harness_dir)
            .map_err(|e| format!("cannot read harness dir {}: {e}", harness_dir.display()))?;
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().is_some_and(|e| e == "js")
                && let Some(name) = p.file_name().and_then(|n| n.to_str())
            {
                let src = fs::read_to_string(&p)
                    .map_err(|e| format!("cannot read harness file {name}: {e}"))?;
                files.insert(name.to_string(), Arc::from(src));
            }
        }
        Ok(HarnessCache { files })
    }

    fn get(&self, name: &str) -> Result<Arc<str>, String> {
        self.files
            .get(name)
            .cloned()
            .ok_or_else(|| format!("unknown harness file {name}"))
    }
}

/// Read-only state shared by every worker thread.
#[derive(Debug)]
struct Shared {
    test_root: PathBuf,
    cache: HarnessCache,
    outcome_cache: OutcomeCache,
}

// ---------------------------------------------------------------------------
// Engine driver
// ---------------------------------------------------------------------------

/// Engine failure, structured by pipeline stage so `negative` frontmatter
/// can be matched against the phase where the failure surfaced.
#[derive(Debug)]
enum EngineError {
    /// The parser rejected the source.
    Parse(String),
    /// The compiler rejected the AST.
    Compile(String),
    /// The script threw a JS value (stringified).
    Thrown(String),
    /// Fuel or native-stack budget exhausted. This is an engine
    /// limitation, never a JS throw: it must not satisfy a `negative`
    /// test's expected error (that would be a false pass for the wrong
    /// reason), so [`verdict`] always fails it.
    ResourceExhausted(String),
    /// Module resolution failed (unresolvable specifier, fetch failure).
    Resolution(String),
    /// The async `$DONE` protocol: `$DONE` was called with an error.
    DoneError(String),
    /// Wall-clock guard tripped (runaway loop or never-settling async).
    Timeout,
    /// The evaluator thread died unexpectedly (not a test verdict).
    EvaluatorGone,
}

/// One unit of work for the evaluator thread: the cached harness sources,
/// the test body, and how to run it.
#[derive(Debug)]
struct ExecJob {
    /// Harness sources (from the shared cache), in order.
    preamble: Vec<Arc<str>>,
    /// Deduped harness file names, in order — the cache key for the
    /// precompiled preamble (same names ⇒ same sources ⇒ same preamble).
    preamble_names: Vec<String>,
    /// Test body.
    body: Arc<str>,
    strict: bool,
    is_async: bool,
    /// Module-goal tests: the test's relative posix path under `test/`,
    /// evaluated as a module at `https://test262.invalid/<rel>`.
    /// `None` means a classic script.
    module: Option<Arc<str>>,
    /// Negative `phase: parse` shortcut: parse the body once on the
    /// evaluator with no reset, preamble, eval, or drains. `strict`
    /// selects the strict (iff `only_strict`) or sloppy body assembly;
    /// `preamble_names` is the would-be include set (its emptiness marks
    /// raw tests, mirroring `preamble.is_empty()` in the normal build).
    parse_only: bool,
}

/// Value equality for reset purposes: structural equality, except that a
/// `Float` NaN equals itself (derived `PartialEq` says `NaN != NaN`,
/// which would flag the `NaN` global as changed on every reset).
fn reset_eq(a: Value, b: Value) -> bool {
    match (a, b) {
        (Value::Float(x), Value::Float(y)) => x == y || (x.is_nan() && y.is_nan()),
        _ => a == b,
    }
}

/// Sorted own-property names of `globalThis` (enumerable or not), via the
/// native [`Vm::own_property_keys`] (Phase B): it reads the heap property
/// lists directly, so a test that clobbered the live `Reflect.ownKeys`
/// cannot blind the scan — no generated JS runs at all. Used for the
/// baseline snapshot and reset.
fn global_keys(vm: &Vm) -> Vec<String> {
    let global = vm.get_global("globalThis").expect("globalThis missing");
    let mut keys: Vec<String> = vm
        .own_property_keys(&global)
        .iter()
        .map(|k| k.as_wtf8().to_string())
        .collect();
    keys.sort();
    keys
}

fn map_vm_error(e: VmError) -> EngineError {
    match e {
        VmError::Parse(m) => EngineError::Parse(m),
        VmError::Compile(m) => EngineError::Compile(m),
        VmError::Thrown(m) => EngineError::Thrown(m),
        // Resource exhaustion is NOT a catchable JS throw: mapping it to
        // `Thrown` let `negative` tests pass for the wrong reason.
        VmError::Resource(m) => EngineError::ResourceExhausted(m),
    }
}

// ---------------------------------------------------------------------------
// Module-goal tests
// ---------------------------------------------------------------------------

/// Synthetic base URL that module specifiers resolve against. Uses the
/// `.invalid` TLD so it can never hit the network (fetching is disabled
/// anyway); the loader only uses it as a registry key.
const MODULE_URL_BASE: &str = "https://test262.invalid/";

/// Scan `src` for static `from '...'` / `import '...'` specifiers.
/// Skips line/block comments and string/template literals, so `import`
/// inside a comment or a string no longer produces a bogus specifier.
/// (Template `${...}` substitutions are skipped wholesale: a static
/// import hidden inside one would be missed, but every hit is just a
/// pre-registration candidate, so a miss costs nothing beyond a failed
/// load later.) Regex literals are not special-cased — only comments and
/// strings/templates are skipped, per the scanner's contract.
fn import_specifiers(src: &str) -> Vec<String> {
    let bytes = src.as_bytes();
    let mut specs = Vec::new();
    let mut i = 0;
    // Advance `i` past a quoted string starting at the opening quote.
    // Handles backslash escapes; an unterminated string runs to EOF.
    let skip_quoted = |bytes: &[u8], mut i: usize| {
        let q = bytes[i];
        i += 1;
        while i < bytes.len() {
            if bytes[i] == b'\\' {
                i += 2;
                continue;
            }
            if bytes[i] == q {
                return i + 1;
            }
            i += 1;
        }
        i
    };
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            // Line comment: skip to the newline.
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            // Block comment: skip to the closer.
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
            continue;
        }
        if b == b'\'' || b == b'"' || b == b'`' {
            i = skip_quoted(bytes, i);
            continue;
        }
        // Ordinary code: only ASCII bytes can start a keyword, and `i`
        // is always a char boundary here (every skip above lands on
        // ASCII or EOF), so `src[i..]` is safe to slice.
        let (kw_len, is_kw) = if b == b'i' && src[i..].starts_with("import") {
            (6, true)
        } else if b == b'f' && src[i..].starts_with("from") {
            (4, true)
        } else {
            (0, false)
        };
        if !is_kw {
            // Skip one whole char so a mid-char index never slices.
            i += src[i..].chars().next().map_or(1, |c| c.len_utf8());
            continue;
        }
        let boundary = |idx: usize| {
            !(bytes[idx].is_ascii_alphanumeric() || bytes[idx] == b'_' || bytes[idx] == b'$')
        };
        let left_ok = i == 0 || boundary(i - 1);
        let right_ok = i + kw_len >= bytes.len() || boundary(i + kw_len);
        i += kw_len;
        if !(left_ok && right_ok) {
            continue;
        }
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if i < bytes.len() && (bytes[i] == b'\'' || bytes[i] == b'"') {
            let q = bytes[i];
            let after = skip_quoted(bytes, i);
            // `after` is one past the closing quote, or `bytes.len()`
            // when the string is unterminated.
            if after > i && bytes[after - 1] == q {
                specs.push(src[i + 1..after - 1].to_string());
            }
            i = after;
        }
    }
    specs
}

/// `ModuleError::Eval` is a formatted string; detect the cases where it
/// wraps engine resource exhaustion so they map to
/// [`EngineError::ResourceExhausted`] instead of a catchable JS throw.
/// `VmError::Resource` displays as `resource exhausted: ...`, prefixed
/// here by the loader with `module {url}: `; the loader's own allocation
/// failures are matched exactly.
fn is_module_resource_error(url: &str, msg: &str) -> bool {
    msg.starts_with(&format!("module {url}: resource exhausted:"))
        || msg == "out of memory allocating module registry"
        || msg == "out of memory allocating module namespace"
        || msg == "out of memory registering module namespace"
}

/// A harness preamble installed once per [`Evaluator`] and reinstalled
/// per job without re-running any JS: the identical `assert.js` +
/// `sta.js` + includes text would otherwise execute its compiled
/// top-level (function-object creation, ~17 global installs) twice per
/// classic test (sloppy + strict variants).
///
/// Snapshot semantics (exact): each compiled variant is run ONCE per
/// evaluator on the clean post-reset global, and every global the run
/// installs is captured as a `(name, value)` pair in installation
/// order. A snapshot is only taken when every installed global is a
/// fresh, non-baseline own data property with exactly the attributes
/// [`Vm::set_global`] installs (`writable + enumerable +
/// configurable`, the engine's own fresh-global attributes), no
/// baseline global was overwritten, and the heap mutation epoch is
/// unchanged by the run (so the preamble touched nothing but the
/// global object: no prototype pollution, no builtin writes). The
/// per-job reinstall is then `set_global` per entry — indistinguishable
/// from running the compiled preamble again. Anything else (a throwing
/// preamble run, accessor properties, non-default attributes,
/// baseline-name collisions, epoch drift) falls back to
/// [`PreambleKind::Split`], preserving today's behavior exactly.
///
/// Known divergence (documented): the installed [`Value`]s — in
/// particular the preamble's function objects — are shared across all
/// jobs of the evaluator's lifetime instead of being freshly created
/// per job. A test that mutates a *direct property* of a shared object
/// (e.g. `assert.foo = 1`) is caught: the snapshot entries are
/// reset-tracked at capture, so the next [`Evaluator::reset`] demotes
/// the include set back to [`PreambleKind::Split`] (see
/// [`Evaluator::demote_dirtied_snapshots`]) and later jobs get fresh
/// objects again. Mutations *nested* below an entry (e.g.
/// `byteConversionValues.values.x = 1`, where `values` is itself an
/// object) are invisible to every reset layer — only the entry object
/// itself is tracked, not the graph reachable from it — and still leak
/// until the evaluator rebuilds. Same class as the documented reset
/// divergences (shared identity invisible to the scrub), and
/// unobservable to any test that does not mutate the harness namespace
/// itself.
#[derive(Debug, Clone)]
struct PreambleSnapshot {
    /// `(name, value)` in `[[OwnPropertyKeys]]` installation order.
    entries: Vec<(String, Value)>,
}

/// A harness preamble compiled once per [`Evaluator`]: the identical
/// `assert.js` + `sta.js` + includes text is otherwise lexed, parsed, and
/// compiled twice per classic test (sloppy + strict variants).
#[derive(Debug, Clone)]
enum PreambleKind {
    /// The preamble declares no top-level lexical bindings, so it can run
    /// as its own classic script: run the precompiled template, then
    /// compile+run only the (much smaller) body per job. Module-goal jobs
    /// use the strict variant (modules are intrinsically strict).
    Split {
        sloppy: ScriptHandle,
        strict: ScriptHandle,
    },
    /// The preamble was snapshotted (see [`PreambleSnapshot`]): reinstall
    /// its globals natively per job instead of re-running the compiled
    /// template. The strict/sloppy variant split is preserved (one
    /// snapshot per variant). Module-goal jobs use the strict snapshot.
    /// The compiled handles are retained so a demoted snapshot (see
    /// [`Evaluator::demote_dirtied_snapshots`]) can fall back to the
    /// exact `Split` behavior without recompiling.
    Snapshot {
        sloppy: Rc<PreambleSnapshot>,
        strict: Rc<PreambleSnapshot>,
        sloppy_h: ScriptHandle,
        strict_h: ScriptHandle,
    },
    /// The preamble declares top-level `let`/`const`/`class` (or failed to
    /// compile standalone): fall back to assembling preamble+body per job,
    /// exactly as before.
    Inline,
}

/// Outcome of [`Evaluator::capture_preamble_snapshot`].
/// [`SnapshotCapture::Dirty`] is distinct from
/// [`SnapshotCapture::Ineligible`]: the latter leaves the `Vm` in the
/// clean post-reset state (the `Split` fallback is safe), while the
/// former means a cleanup delete failed and residue survived.
#[derive(Debug)]
enum SnapshotCapture {
    /// The preamble was snapshotted; the `Vm` is clean again.
    Snapshot(Rc<PreambleSnapshot>),
    /// Not snapshot-safe, but the `Vm` is clean: `Split` is safe.
    Ineligible,
    /// A cleanup delete failed (e.g. a non-configurable install): the
    /// `Vm` still holds residue — rebuild, do not fall back.
    Dirty,
}

/// Per-job reset accounting, carried back to the worker thread in
/// [`JobResult`] and merged into the [`Scoreboard`].
#[derive(Debug, Default, Clone, Copy)]
struct ResetStats {
    /// A reset ran for this job (false when the job skipped it because a
    /// rebuild was due, or for parse-only jobs).
    ran: bool,
    /// The reset took the slow path (epoch changed or fast path refused).
    slow: bool,
    /// The periodic epoch-fast-path validation ran for this job.
    validated: bool,
    /// ... and the epoch's "nothing was touched" claim was violated.
    validation_miss: bool,
    /// The reset found the `Vm` poisoned beyond cheap repair and the job
    /// ran on a freshly rebuilt evaluator instead.
    poison: bool,
}

/// The per-worker evaluator: owns one [`Vm`], reused across tests, with a
/// global-state reset between tests (see module docs for why reuse is
/// sound).
/// [`Evaluator::body_cache`] key: an `Arc<str>` hashed and compared by
/// pointer identity instead of by contents. The cache owns the `Arc`, so
/// the pointer is stable and cannot dangle or be recycled while the entry
/// lives. Both modes of one test share the same `Arc` (hoisted per test
/// in [`run_one`] and cloned per mode), so the cross-mode hit is
/// preserved while hashing costs O(1) instead of O(body bytes).
#[derive(Clone, Debug)]
struct BodyKey(Arc<str>);

impl PartialEq for BodyKey {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for BodyKey {}

impl std::hash::Hash for BodyKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Thin data address; `eq` also compares length via `ptr_eq`, so
        // equal keys hash equal. Two distinct live `Arc<str>`s never share
        // a data address while the cache holds them alive.
        (Arc::as_ptr(&self.0) as *const u8 as usize).hash(state);
    }
}

struct Evaluator {
    vm: Vm,
    /// The global object, for the native key scans in [`Evaluator::reset`].
    global: Value,
    /// Sorted `globalThis` own-property names right after `Vm::new()`.
    baseline_keys: Vec<String>,
    /// Baseline values of every baseline global own key (enumerable or
    /// not), to detect overwrites. This covers the well-known builtins
    /// too: any existing `KNOWN_GLOBALS` entry is a baseline own key, so
    /// no separate builtin scan is needed.
    baseline_values: Vec<(String, Value)>,
    /// Prototype / namespace-object scrub state (see
    /// [`native_reset::ProtoScrub`]). Snapshotted natively, so it stashes
    /// nothing on `globalThis` and the baselines below need no
    /// re-snapshot.
    scrub: native_reset::ProtoScrub,
    /// Monotonic id minted into each async test's `$DONE` shim.
    done_seq: u32,
    /// Tests run since the last `Vm` rebuild (see [`VM_REBUILD_EVERY`]).
    tests_run: u64,
    /// Precompiled harness preambles, one entry per distinct include set.
    preamble_cache: HashMap<Vec<String>, PreambleKind>,
    /// Compiled test bodies for the split classic pipeline, keyed by
    /// (body identity, strict, preamble-empty). The async `$DONE` shim is
    /// NOT part of the cached source: its per-test token is baked into a
    /// literal in the cached shim ([`Evaluator::done_shim_cache`]).
    /// Bodies are keyed by
    /// pointer identity ([`BodyKey`]): both modes of one test share the
    /// same `Arc` (hoisted per test, cloned per mode), so the second mode
    /// pays no lex/parse/compile and no hashing walks the body bytes.
    /// Cleared on every `Vm` rebuild (the whole `Evaluator` is replaced,
    /// and `ScriptHandle`s are tied to one `Vm` via its nonce), and
    /// flushed at [`BODY_CACHE_MAX_ENTRIES`].
    body_cache: HashMap<(BodyKey, bool, bool), ScriptHandle>,
    /// Compiled `$DONE` shim variants, one per distinct async token, in
    /// insertion order (LRU-ish, tiny). Each variant has its token baked
    /// in as a literal ([`done_shim_source`]), which closes the
    /// protocol-global clobber: a stale `$DONE` from a previous test's
    /// leftover async work carries the old literal and can never pass a
    /// later test's guard. Per async test the runner only *runs* the
    /// cached handle — no per-test lex/parse/compile, so the precompile
    /// win is kept. Bounded at [`DONE_SHIM_CACHE_MAX`]; the oldest
    /// variant is evicted on overflow. Cleared on every `Vm` rebuild
    /// (the whole `Evaluator` is replaced, and `ScriptHandle`s are tied
    /// to one `Vm` via its nonce).
    done_shim_cache: Vec<(u32, ScriptHandle)>,
    /// Memoized module-fixture sources, keyed by module URL: the
    /// transitive `_FIXTURE.js` files imported by module tests are
    /// shared across many tests (e.g. sixty tests in one directory
    /// importing the same `dep_FIXTURE.js`), and [`register_imports`]
    /// would otherwise re-read each one from disk per importing test.
    /// This is purely an I/O memoization keyed by source identity.
    /// Every test still gets a fresh [`ModuleLoader`] which evaluates
    /// each fixture anew (reusing a loader across tests would skip
    /// re-evaluation, which is unsound for stateful fixtures such as
    /// `setup_FIXTURE.js`); the parse+compile half is memoized
    /// separately by the loader's process-wide compile cache (see
    /// `ModuleCompileCache` in `module_loader.rs`). Naturally bounded by
    /// the fixture count (a few hundred); cleared on `Vm` rebuild with
    /// the rest of the `Evaluator`.
    fixture_cache: HashMap<String, Arc<str>>,
    /// Heap mutation epoch as of the last successful [`Evaluator::reset`];
    /// the fast path skips the scrub scans while it is unchanged.
    synced_epoch: u64,
    /// Fast-path resets since the evaluator was built; drives the
    /// periodic [`RESET_VALIDATE_EVERY`] validation.
    resets_fast: u64,
    /// Per-job reset accounting for the scoreboard (see [`ResetStats`]).
    last_reset: ResetStats,
    /// Live heap allocations right after `Vm::new()`; the rebuild
    /// backstop compares against this (see [`VM_REBUILD_HEAP_BUDGET`]).
    heap_base: usize,
}

impl Evaluator {
    fn new() -> Self {
        let mut vm = Vm::new();
        // The evaluator thread runs on an 8 MiB stack
        // ([`EVALUATOR_STACK_SIZE`]); let deep-but-finite recursion use
        // 7 MiB of it, keeping 1 MiB of headroom to unwind (unwinding
        // from the guard allocates nothing significant). Without this
        // the 1 MiB default would trip far below what the thread allows,
        // and without any byte budget a deep recursion aborts the whole
        // test binary with `fatal runtime error: stack overflow` instead
        // of recording a stack-overflow failure.
        vm.set_max_native_stack(7 * 1024 * 1024);
        // The test262 `$262` helper object (`$262.detachArrayBuffer`,
        // backing the `detachArrayBuffer.js` harness include): installed
        // before the baseline snapshot so the per-test scrub treats
        // `$262` as a builtin binding rather than test debris. The real
        // test262 driver provides the same object on every realm.
        install_test262_helpers(&mut vm).expect("install $262 test helper");
        let global = vm.get_global("globalThis").expect("globalThis missing");
        // Prototype scrub state: snapshotted natively, so it stashes
        // nothing on `globalThis` and the baselines below need no
        // re-snapshot.
        let scrub = native_reset::build_proto_scrub(&vm);
        let baseline_keys = global_keys(&vm);
        let baseline_values = baseline_keys
            .iter()
            .map(|k| (k.clone(), vm.get_global(k).unwrap_or(Value::Undefined)))
            .collect();
        // Note: no separate well-known-builtin baseline. Any existing
        // `KNOWN_GLOBALS` entry is a baseline own key of `globalThis`
        // (nothing runs between the snapshots above), so
        // `baseline_values` already covers it; a second scan would
        // re-check the same bindings.
        // Reset fast path: every scrub target and its global binding bumps
        // the heap mutation epoch on any write; baseline global keys do
        // too. Non-baseline global additions (the per-test preamble/body
        // globals) are recorded for the fast key scrub instead of bumping,
        // so the common clean test leaves the epoch unchanged and `reset`
        // skips the scans. The global object is its own scrub target
        // (`globalThis` is in `KNOWN_GLOBALS`) but is *not* tracked: the
        // global registration below covers it with the finer granularity
        // above — tracking it would bump on every test's globals and the
        // fast path would never fire. (Its scrub checks — added-key
        // repair, baseline missing/attrs/value findings — are all subsumed:
        // non-baseline additions land in the added list, baseline writes
        // bump.)
        for (target, binding) in scrub.tracked_objects() {
            if target != global {
                vm.track_reset_object(&target);
            }
            if binding != global {
                vm.track_reset_object(&binding);
            }
        }
        vm.set_reset_global_baseline(&global, &baseline_keys);
        let synced_epoch = vm.mutation_epoch();
        let heap_base = vm.heap().alloc_count();
        Evaluator {
            vm,
            global,
            baseline_keys,
            baseline_values,
            scrub,
            done_seq: 0,
            tests_run: 0,
            preamble_cache: HashMap::new(),
            body_cache: HashMap::new(),
            done_shim_cache: Vec::new(),
            fixture_cache: HashMap::new(),
            synced_epoch,
            resets_fast: 0,
            last_reset: ResetStats::default(),
            heap_base,
        }
    }

    /// Scrub global state left by the previous test. Returns `false` when
    /// the `Vm` is poisoned beyond cheap repair; the caller then rebuilds
    /// the evaluator.
    ///
    /// Soundness note: top-level `var`/`let`/`const`/`function`/`class`
    /// declarations in an `eval`'d script never reach the shared global
    /// object in this engine (they live in a per-eval environment), so the
    /// common cross-test leak is sloppy-mode assignments to undeclared
    /// names, which are enumerable and deletable. Overwritten builtins are
    /// rarer and handled by rebuilding: redefining a global the test made
    /// non-configurable would panic inside `set_global`, so no restore is
    /// attempted.
    ///
    /// Ordering: the builtin-binding checks run before the prototype
    /// scrub because the scrub assumes the baseline builtin bindings
    /// (e.g. `Array.prototype`) are intact; a clobbered binding rebuilds
    /// before the scrub can misbehave.
    fn reset(&mut self) -> bool {
        // Stashed event-loop task errors from the previous test.
        self.vm.take_uncaught_task_errors();
        // The realm's global declarative record (top-level `let`/`const`
        // of classic scripts) does not live on the global object, so the
        // key scrub below cannot see it: clear it explicitly, or one
        // test's top-level lexicals would leak into — and collide with —
        // the next test's.
        self.vm.reset_global_lexical();
        self.last_reset = ResetStats::default();
        self.last_reset.ran = true;
        // Snapshot demotion (see [`PreambleSnapshot`]): when the heap
        // mutation epoch moved, some reset-tracked object was written —
        // drain the dirtied list and demote any snapshotted include set
        // whose shared objects were among them, so later jobs get fresh
        // preamble objects instead of the mutated shared ones. Draining
        // here consumes the list `native_reset::dirtied_targets` would
        // use for its dirtied-subset scrub; the slow path below
        // compensates with the check-only `prototypes_clean`, which arms
        // the forced full scan on any drift (exactly its documented
        // purpose). When the epoch is unchanged no tracked object could
        // have been written (writes bump the epoch and dirty together),
        // so the list is necessarily empty and the drain is skipped —
        // the fast path below is untouched.
        let dirtied: Vec<Value> = if self.vm.mutation_epoch() != self.synced_epoch {
            let dirtied = self.vm.take_dirtied_reset_objects();
            self.demote_dirtied_snapshots(&dirtied);
            dirtied
        } else {
            Vec::new()
        };
        // Fast path: the heap mutation epoch is unchanged since the last
        // reset, so no scrub target, binding, or baseline global could
        // have been touched — the value scans and the prototype scrub
        // would find nothing. The only possible residue is non-baseline
        // global additions (the previous test's preamble/body globals),
        // recorded by key: delete exactly those. Any refusal (a test made
        // one non-configurable) falls back to the full scrub below, which
        // re-covers the drained list via its own key scan.
        if self.vm.mutation_epoch() == self.synced_epoch && self.reset_fast_path() {
            return true;
        }
        self.last_reset.slow = true;
        // Any overwritten baseline global means the Vm is poisoned:
        // rebuild instead of attempting a restore. (This also guards the
        // scrub's stash globals: a test tampering with them rebuilds here
        // before the scrub can act on garbage.) This single scan covers
        // every baseline own key, enumerable or not — including the
        // well-known builtins, which are baseline globals too.
        for (name, value) in &self.baseline_values {
            if self
                .vm
                .get_global(name)
                .is_none_or(|v| !reset_eq(v, *value))
            {
                if std::env::var("TEST262_DEBUG_RESET").is_ok() {
                    eprintln!("reset: baseline global changed: {name}");
                }
                return false;
            }
        }
        // Scrub prototype / namespace-object pollution (added properties
        // deleted, overwritten properties restored — enumerable or not,
        // via the native `own_property_keys`); unrecoverable shapes rebuild
        // the Vm.
        //
        // When the demote drain above consumed a non-empty dirtied list,
        // the scrub's dirtied-subset optimization would see an empty set
        // and check only the global target. Re-verify every target with
        // the check-only `prototypes_clean` first: on drift it arms the
        // forced full scan, so the repair pass below stays exact (this
        // is the scenario `prototypes_clean` documents). When the drain
        // was skipped (empty list) the subset behaves exactly as before.
        if !dirtied.is_empty() {
            let debug = std::env::var("TEST262_DEBUG_RESET").is_ok();
            // The return value is intentionally ignored: drift arms the
            // full scan internally, and the repair happens below either
            // way.
            let _ = native_reset::prototypes_clean(&mut self.vm, &self.scrub, debug);
        }
        if !self.scrub_prototypes() {
            if std::env::var("TEST262_DEBUG_RESET").is_ok() {
                eprintln!("reset: prototype scrub failed");
            }
            return false;
        }
        // Delete every global (enumerable or not) not present at baseline
        // and verify the key set, natively: `own_property_keys` reads the
        // heap lists directly, so a clobbered `Reflect.ownKeys` cannot
        // blind the scan, and no generated JS runs at all.
        let debug = std::env::var("TEST262_DEBUG_RESET").is_ok();
        if !native_reset::scrub_global_keys(&mut self.vm, &self.global, &self.baseline_keys, debug)
        {
            return false;
        }
        // Reset complete: drain any recorded additions (the key scan just
        // handled them) and snapshot the epoch for the next fast path.
        let _ = self.vm.take_reset_added_keys();
        self.synced_epoch = self.vm.mutation_epoch();
        true
    }

    /// The epoch fast path of [`Evaluator::reset`]: the mutation epoch is
    /// unchanged, so no scrub target, binding, or baseline global was
    /// written. Deletes the recorded non-baseline global additions.
    /// Returns `false` when a delete is refused — the caller falls
    /// through to the slow path, whose key scan re-covers the drained
    /// list — or when the periodic validation trips.
    fn reset_fast_path(&mut self) -> bool {
        self.resets_fast += 1;
        for key in self.vm.take_reset_added_keys() {
            if !self.vm.delete_property(&self.global, &key) {
                return false;
            }
        }
        if self.resets_fast.is_multiple_of(RESET_VALIDATE_EVERY) {
            // Periodic validation of the epoch bookkeeping (see
            // [`RESET_VALIDATE_EVERY`]): re-run the epoch-covered scans
            // and assert the "nothing was touched" claim. Runs after the
            // recorded additions are deleted above: those are expected
            // on the fast path (the `globalThis` scrub target would
            // otherwise flag them as drift).
            self.last_reset.validated = true;
            if !self.epoch_claim_holds() {
                self.last_reset.validation_miss = true;
                if std::env::var("TEST262_DEBUG_RESET").is_ok() {
                    eprintln!("reset: fast-path validation found drift the epoch missed");
                }
                return false;
            }
        }
        true
    }

    /// Demote every [`PreambleKind::Snapshot`] whose shared objects appear
    /// in `dirtied` back to [`PreambleKind::Split`]. A test that writes a
    /// direct property of a snapshot entry (the entries are reset-tracked
    /// at capture) would otherwise leak the mutation to every later job
    /// until the evaluator rebuilds: the global key scans cannot see it
    /// (the binding is deleted and reinstalled each job) and the value is
    /// the same shared object. The demoted include set re-runs the
    /// compiled preamble per job, producing fresh objects. Nested objects
    /// reachable from an entry are not tracked — only direct properties
    /// are covered (see [`PreambleSnapshot`]).
    fn demote_dirtied_snapshots(&mut self, dirtied: &[Value]) {
        if dirtied.is_empty() {
            return;
        }
        for kind in self.preamble_cache.values_mut() {
            let (sloppy_h, strict_h, hit) = match kind {
                PreambleKind::Snapshot {
                    sloppy,
                    strict,
                    sloppy_h,
                    strict_h,
                } => {
                    let hit = dirtied.iter().any(|d| {
                        sloppy.entries.iter().any(|(_, v)| v == d)
                            || strict.entries.iter().any(|(_, v)| v == d)
                    });
                    (*sloppy_h, *strict_h, hit)
                }
                _ => continue,
            };
            if hit {
                *kind = PreambleKind::Split {
                    sloppy: sloppy_h,
                    strict: strict_h,
                };
            }
        }
    }

    /// Check-only re-verification of the epoch's "nothing was touched"
    /// claim: baseline global values plus the prototype scrub targets.
    /// True when the epoch-covered state is pristine.
    fn epoch_claim_holds(&mut self) -> bool {
        let debug = std::env::var("TEST262_DEBUG_RESET").is_ok();
        for (name, value) in &self.baseline_values {
            if self
                .vm
                .get_global(name)
                .is_none_or(|v| !reset_eq(v, *value))
            {
                if debug {
                    eprintln!("reset: validation: baseline global drifted: {name}");
                }
                return false;
            }
        }
        native_reset::prototypes_clean(&mut self.vm, &self.scrub, debug)
    }

    /// Run the native prototype / namespace-object scrub (see
    /// [`native_reset::scrub_prototypes`]): added properties deleted,
    /// overwritten data values restored, attribute flips re-defined —
    /// enumerable or not, with no generated JS. Returns `false` when a
    /// target is poisoned beyond cheap repair (replaced/reparented
    /// prototype, non-configurable interference, accessor attribute
    /// changes); the caller rebuilds the `Vm`.
    fn scrub_prototypes(&mut self) -> bool {
        let debug = std::env::var("TEST262_DEBUG_RESET").is_ok();
        if native_reset::scrub_prototypes(&mut self.vm, &self.scrub, debug) {
            return true;
        }
        if debug {
            eprintln!("reset: prototype scrub failed");
        }
        false
    }

    /// Stringify a global for error reporting (via the VM's own `String`).
    fn global_string(&mut self, name: &str) -> String {
        match self.vm.eval(&format!("String({name})")) {
            Ok(Value::Str(s)) => s.as_wtf8().to_string(),
            Ok(v) => format!("{v:?}"),
            Err(e) => format!("<unreadable: {e}>"),
        }
    }

    /// Whether the evaluator should be rebuilt before the next job:
    /// the count backstop ([`VM_REBUILD_EVERY`]) or the heap budget
    /// ([`VM_REBUILD_HEAP_BUDGET`]) was reached. Rebuilding caps heap
    /// growth — the engine's GC is opt-in/disabled, so a long-lived `Vm`
    /// accumulates garbage without bound. A poisoned reset rebuilds
    /// separately in [`Evaluator::run_job`].
    fn rebuild_due(&self) -> bool {
        self.tests_run >= VM_REBUILD_EVERY
            || self.vm.heap().alloc_count().saturating_sub(self.heap_base) >= VM_REBUILD_HEAP_BUDGET
    }

    /// `TEST262_HEAP_STATS=1` diagnostic: log live-heap growth at each
    /// rebuild, for tuning [`VM_REBUILD_HEAP_BUDGET`].
    fn report_heap_stats(&self, why: &str) {
        if *HEAP_STATS {
            eprintln!(
                "test262: heap-stats rebuild ({why}): {} jobs, {} live allocs (+{} since build)",
                self.tests_run,
                self.vm.heap().alloc_count(),
                self.vm.heap().alloc_count().saturating_sub(self.heap_base),
            );
        }
    }

    /// Run one job: reset, assemble, evaluate, and (for async tests) drain
    /// the event loop under the `$DONE` protocol. Module-goal jobs run
    /// the body through a fresh [`ModuleLoader`] instead of `Vm::eval`.
    ///
    /// Microtasks the test queued but never yielded to are drained at the
    /// end (even on failure: a throwing test may have queued jobs before
    /// throwing), so they cannot run during the *next* test's event-loop
    /// drain and pollute it.
    fn run_job(&mut self, job: &ExecJob, shared: &Shared) -> JobResult {
        let timing = *TIMING_ENABLED;
        let mut phases = PhaseTimes::default();
        if job.parse_only {
            // Negative `phase: parse` shortcut: one parse decides both
            // modes (see `run_one`), so skip the reset, the preamble,
            // the eval, and both drains entirely.
            return self.run_parse_only(job, timing);
        }
        self.tests_run += 1;
        let t_reset = if timing { Some(Instant::now()) } else { None };
        let reset_stats = if self.rebuild_due() {
            // Count/heap backstop: the fresh Evaluator needs no reset,
            // so skip it (and report no reset for this job).
            self.report_heap_stats("backstop");
            *self = Evaluator::new();
            ResetStats::default()
        } else {
            let ok = self.reset();
            let mut stats = self.last_reset;
            if !ok {
                // Poisoned reset rebuilds here; either way the fresh
                // Evaluator resets the counter.
                stats.poison = true;
                self.report_heap_stats("poison");
                *self = Evaluator::new();
            }
            stats
        };
        if let Some(t) = t_reset {
            phases.reset = t.elapsed().as_nanos() as u64;
        }

        // Warm the preamble cache here — on the clean post-reset global,
        // before the $DONE protocol globals below. The snapshot capture
        // (see [`PreambleSnapshot`]) must observe exactly the baseline
        // keys: the $DONE globals are non-baseline additions and would
        // otherwise be captured into (and reinstalled from) the snapshot.
        // After the first job per include set this is a HashMap lookup +
        // clone; the one capture per include set per evaluator is timed
        // separately (see [`JobResult::preamble_capture_ns`]) and reported
        // as a synthetic slow-list entry, so it pollutes no test's phase
        // buckets.
        let t_capture = if timing { Some(Instant::now()) } else { None };
        let _ = self.preamble_kind(job);
        let preamble_capture_ns = t_capture
            .map(|t| t.elapsed().as_nanos() as u64)
            .unwrap_or(0);

        // Mint the async token and seed the $DONE protocol globals before
        // assembling either pipeline. The shim is compiled once per
        // distinct token ([`Evaluator::done_shim`]) with the token baked
        // in as a literal, so no per-test lex/parse/compile — and a stale
        // `$DONE` from a previous test's leftover async work can never
        // pass this test's guard.
        let done_token = if job.is_async {
            // Masked to 31 bits: `$DONE_token` is stored as
            // `Value::Int(token as i32)` while the shim bakes the `u32`
            // decimal literal, so a token >= 2^31 would wrap negative and
            // never match its guard literal — `$DONE` could never complete
            // and the test would time out. (`strict_eq` treats `Int` and
            // `Float` as different types, so storing a float is not an
            // option.) Unreachable in practice (`done_seq` resets per
            // `Evaluator::new`, `VM_REBUILD_EVERY = 1000`), but a latent
            // landmine without the mask.
            let token = self.done_seq & 0x7FFF_FFFF;
            self.done_seq = self.done_seq.wrapping_add(1);
            self.vm.set_global("$DONE_called", Value::Bool(false));
            self.vm.set_global("$DONE_error", Value::Undefined);
            self.vm.set_global("$DONE_token", Value::Int(token as i32));
            Some(token)
        } else {
            None
        };

        let inner = self.run_job_inner(job, shared, done_token);
        if timing {
            // Module tests are intrinsically strict; classic tests split
            // by mode.
            if job.module.is_some() || job.strict {
                phases.eval_strict = inner.eval_ns;
            } else {
                phases.eval_sloppy = inner.eval_ns;
            }
            phases.drain = inner.drain_ns;
        }
        self.vm.drain_microtasks();
        JobResult {
            outcome: inner.result,
            phases,
            reset_stats,
            preamble_capture_ns,
        }
    }

    /// The body of [`run_job`](Self::run_job): assemble, evaluate, and
    /// (for async tests) drain under the `$DONE` protocol. Returns the
    /// engine result plus per-phase wall time (zeros when `TEST262_TIMING`
    /// is unset).
    fn run_job_inner(
        &mut self,
        job: &ExecJob,
        shared: &Shared,
        done_token: Option<u32>,
    ) -> InnerResult {
        let timing = *TIMING_ENABLED;
        let t_eval = if timing { Some(Instant::now()) } else { None };

        let eval_result: Result<(), EngineError> = if let Some(rel) = &job.module {
            self.run_module_job(job, shared, rel, done_token)
        } else {
            self.run_classic_job(job, done_token)
        };
        let eval_ns = t_eval.map(|t| t.elapsed().as_nanos() as u64).unwrap_or(0);
        if let Err(e) = eval_result {
            return InnerResult {
                result: Err(e),
                eval_ns,
                drain_ns: 0,
            };
        }

        let t_drain = if timing { Some(Instant::now()) } else { None };
        if job.is_async {
            // Drain the event loop until the test calls $DONE or the
            // per-test guard trips. Sleep briefly between idle drains so
            // wall-clock timers can become due.
            let t0 = Instant::now();
            // Fail fast on a quiescent event loop: no pending macrotasks,
            // microtasks, or timers means no future JS can ever run, so
            // $DONE can never be called. A dead async test — e.g. an async
            // function suspended forever on a never-settling promise, like
            // a module waiting on a dynamic import the engine rejects
            // outright — then fails in microseconds instead of burning the
            // full per-test timeout. The full timeout applies only while
            // work is actually pending (timers due later still count), so
            // this cannot flip a pass to a fail: $DONE is only ever called
            // by JS, and JS only ever runs as an event-loop task.
            let timed_out = || InnerResult {
                result: Err(EngineError::Timeout),
                eval_ns,
                drain_ns: t_drain
                    .as_ref()
                    .map(|t| t.elapsed().as_nanos() as u64)
                    .unwrap_or(0),
            };
            loop {
                self.vm.run_until_idle();
                if matches!(self.vm.get_global("$DONE_called"), Some(Value::Bool(true))) {
                    break;
                }
                if t0.elapsed() > per_test_timeout() || !self.vm.event_loop().has_pending_work() {
                    return timed_out();
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            if !matches!(
                self.vm.get_global("$DONE_error"),
                Some(Value::Undefined) | None
            ) {
                let msg = self.global_string("$DONE_error");
                return InnerResult {
                    result: Err(EngineError::DoneError(msg)),
                    eval_ns,
                    drain_ns: t_drain.map(|t| t.elapsed().as_nanos() as u64).unwrap_or(0),
                };
            }
        }
        InnerResult {
            result: Ok(()),
            eval_ns,
            drain_ns: t_drain.map(|t| t.elapsed().as_nanos() as u64).unwrap_or(0),
        }
    }

    /// Classic-script job: install the preamble (snapshot reinstall or
    /// precompiled-template run), then the cached `$DONE` shim variant
    /// for this test's token and the cached body template.
    /// Include sets with top-level lexical bindings fall back to the old
    /// assemble-everything path ([`PreambleKind::Inline`]).
    fn run_classic_job(
        &mut self,
        job: &ExecJob,
        done_token: Option<u32>,
    ) -> Result<(), EngineError> {
        match self.preamble_kind(job) {
            PreambleKind::Split { sloppy, strict } => {
                let handle = if job.strict { strict } else { sloppy };
                self.vm
                    .run_compiled(handle)
                    .map(|_| ())
                    .map_err(map_vm_error)?;
            }
            PreambleKind::Snapshot { sloppy, strict, .. } => {
                // Native reinstall of the snapshotted globals (see
                // [`PreambleSnapshot`]): no JS runs at all.
                // `set_global` installs exactly the attributes the
                // engine's own fresh-global define uses, and a
                // successful reset guarantees the names are absent (the
                // slow path verifies the key set equals the baseline; a
                // leftover key poisons the Vm into a rebuild instead), so
                // this is indistinguishable from the compiled run.
                let snap = if job.strict { strict } else { sloppy };
                for (name, value) in &snap.entries {
                    self.vm.set_global(name, *value);
                }
            }
            PreambleKind::Inline => {
                let shim = if job.is_async {
                    done_shim_source(done_token_or_bail(done_token)?)
                } else {
                    String::new()
                };
                let source = assemble_classic_source(job.strict, &job.preamble, &shim, &job.body);
                return self.vm.eval(&source).map(|_| ()).map_err(map_vm_error);
            }
        }
        // The cached `$DONE` shim variant for this test's token:
        // running it installs a fresh `$DONE` binding whose guard
        // literal matches this test only, with no per-test
        // lex/parse/compile.
        if job.is_async {
            let token = done_token_or_bail(done_token)?;
            let shim = self.done_shim(token)?;
            self.vm
                .run_compiled(shim)
                .map(|_| ())
                .map_err(map_vm_error)?;
        }
        let body = self.body_handle(job.strict, job.preamble.is_empty(), &job.body)?;
        self.vm.run_compiled(body).map(|_| ()).map_err(map_vm_error)
    }

    /// The compiled form of this job's body source for the split classic
    /// pipeline, compiling and caching on first use per (body identity,
    /// strict, preamble-empty) triple. The cached source is exactly
    /// [`assemble_body_source`] minus the `$DONE` shim (precompiled
    /// separately); both modes of one test share the same `Arc` (hoisted
    /// per test in [`run_one`], cloned per mode), so the second mode hits
    /// the cache by pointer identity and pays no lex/parse/compile and no
    /// body hashing at all.
    fn body_handle(
        &mut self,
        strict: bool,
        preamble_empty: bool,
        body: &Arc<str>,
    ) -> Result<ScriptHandle, EngineError> {
        let key = (BodyKey(Arc::clone(body)), strict, preamble_empty);
        if let Some(&handle) = self.body_cache.get(&key) {
            return Ok(handle);
        }
        let source = assemble_body_source(strict, preamble_empty, "", body);
        let handle = self.vm.compile_script(&source).map_err(map_vm_error)?;
        if self.body_cache.len() >= BODY_CACHE_MAX_ENTRIES {
            // Backstop (see [`BODY_CACHE_MAX_ENTRIES`]): flush rather
            // than grow without bound. The interned templates of dropped
            // entries stay in the `Vm` until its next rebuild, which is
            // still bounded by [`VM_REBUILD_EVERY`].
            self.body_cache.clear();
        }
        self.body_cache.insert(key, handle);
        Ok(handle)
    }

    /// The compiled `$DONE` shim for `token`, compiling and caching on
    /// first use per token. The token is baked into the shim source as a
    /// literal ([`done_shim_source`]), so each distinct token compiles
    /// exactly once per evaluator and the sequential-token pattern is a
    /// cache hit for the rest of the evaluator's life — no per-test
    /// lex/parse/compile. The cache is tiny and bounded
    /// ([`DONE_SHIM_CACHE_MAX`]); on overflow the oldest variant is
    /// evicted.
    fn done_shim(&mut self, token: u32) -> Result<ScriptHandle, EngineError> {
        if let Some(&(_, handle)) = self.done_shim_cache.iter().find(|&&(t, _)| t == token) {
            return Ok(handle);
        }
        let handle = self
            .vm
            .compile_script(&done_shim_source(token))
            .map_err(map_vm_error)?;
        self.done_shim_cache.push((token, handle));
        if self.done_shim_cache.len() > DONE_SHIM_CACHE_MAX {
            self.done_shim_cache.remove(0);
        }
        Ok(handle)
    }

    /// Negative `phase: parse` job: parse the body once, with no reset,
    /// no preamble, no eval, and no drains. `compile_script` parses and
    /// compiles without running; both [`EngineError::Parse`] and
    /// [`EngineError::Compile`] satisfy a parse-phase negative in
    /// [`verdict`], and a clean compile is the expected-error-missing
    /// case. The parsed source mirrors the classic body assembly minus
    /// the `$DONE` shim, which always parses.
    fn run_parse_only(&mut self, job: &ExecJob, timing: bool) -> JobResult {
        let mut phases = PhaseTimes::default();
        // A parse-only job still interns one template, so it counts
        // toward the rebuild bound like any other job.
        self.tests_run += 1;
        if self.rebuild_due() {
            self.report_heap_stats("backstop");
            *self = Evaluator::new();
        }
        let t_eval = timing.then(Instant::now);
        let src = assemble_body_source(job.strict, job.preamble_names.is_empty(), "", &job.body);
        let outcome = self
            .vm
            .compile_script(&src)
            .map(|_| ())
            .map_err(map_vm_error);
        if timing {
            // Bucket with eval: this is the parse/compile work the old
            // pipeline paid once per mode.
            phases.eval_sloppy = t_eval.map(|t| t.elapsed().as_nanos() as u64).unwrap_or(0);
        }
        JobResult {
            outcome,
            phases,
            reset_stats: ResetStats::default(),
            preamble_capture_ns: 0,
        }
    }

    /// Module-goal job: run the precompiled strict preamble as a classic
    /// script (module goal is intrinsically strict), then evaluate only
    /// the body through the module loader — the preamble text is no
    /// longer re-parsed per test. The async `$DONE` shim is not part of
    /// the module source: it runs beforehand as the precompiled
    /// per-token variant ([`Evaluator::done_shim`]), exactly like
    /// [`run_classic_job`], so its text never parses per test. Besides
    /// the compile saving, this also fixes `asyncTest` (harness
    /// `asyncHelpers.js`) for module tests: it requires `$DONE` to be a
    /// `globalThis` own property, which a module-scoped shim never is.
    fn run_module_job(
        &mut self,
        job: &ExecJob,
        shared: &Shared,
        rel: &str,
        done_token: Option<u32>,
    ) -> Result<(), EngineError> {
        let mut module_src = String::new();
        match self.preamble_kind(job) {
            PreambleKind::Split { strict, .. } => {
                self.vm
                    .run_compiled(strict)
                    .map(|_| ())
                    .map_err(map_vm_error)?;
            }
            PreambleKind::Snapshot { strict, .. } => {
                // Same native reinstall as the classic path (see
                // [`PreambleSnapshot`]); modules are intrinsically strict.
                for (name, value) in &strict.entries {
                    self.vm.set_global(name, *value);
                }
            }
            PreambleKind::Inline => {
                // Lexical preamble: keep the old behavior (preamble text
                // prepended into the module source so its declarations stay
                // in scope for the module body).
                for p in &job.preamble {
                    module_src.push_str(p);
                    if !p.ends_with('\n') {
                        module_src.push('\n');
                    }
                }
            }
        }
        // The cached `$DONE` shim variant for this test's token:
        // running it installs a global `$DONE` whose guard literal
        // matches this test only, with no per-test lex/parse/compile.
        // The module body resolves `$DONE` through the global scope.
        // (No module test declares a top-level `$DONE`, and the harness
        // preamble never references it, so the global/module scope
        // difference is unobservable.)
        if job.is_async {
            let token = done_token_or_bail(done_token)?;
            let shim = self.done_shim(token)?;
            self.vm
                .run_compiled(shim)
                .map(|_| ())
                .map_err(map_vm_error)?;
        }
        module_src.push_str(&job.body);
        if !job.body.ends_with('\n') {
            module_src.push('\n');
        }
        self.exec_module(shared, &module_src, rel)
    }

    /// Pre-register the transitive closure of relative imports so the
    /// loader never touches the network. `entry_url`/`entry_src` are the
    /// test module itself. Fixture file contents are memoized in
    /// [`Evaluator::fixture_cache`]: the same `_FIXTURE.js` is imported
    /// by dozens of tests, so without the cache each one re-reads it
    /// from disk here.
    fn register_imports(
        &mut self,
        loader: &mut ModuleLoader,
        entry_url: &str,
        entry_src: &str,
        test_root: &Path,
    ) {
        let mut seen = HashSet::new();
        // The stack shares the fixture text with `fixture_cache` as
        // `Arc<str>`: pushing a second owned copy per import is pure
        // waste on the hot import path.
        let mut stack: Vec<(String, Arc<str>)> =
            vec![(entry_url.to_string(), Arc::from(entry_src))];
        seen.insert(entry_url.to_string());
        while let Some((referrer, src)) = stack.pop() {
            for spec in import_specifiers(&src) {
                if !(spec.starts_with("./") || spec.starts_with("../")) {
                    continue;
                }
                let url = match loader.resolve(&spec, &referrer) {
                    Ok(u) => u,
                    Err(_) => continue,
                };
                if !seen.insert(url.clone()) {
                    continue;
                }
                let rel = match url.strip_prefix(MODULE_URL_BASE) {
                    Some(r) => r,
                    None => continue,
                };
                let content: Arc<str> = if let Some(hit) = self.fixture_cache.get(&url) {
                    Arc::clone(hit)
                } else {
                    match fs::read_to_string(test_root.join(rel)) {
                        Ok(text) => {
                            let content: Arc<str> = Arc::from(text);
                            self.fixture_cache.insert(url.clone(), Arc::clone(&content));
                            content
                        }
                        Err(_) => continue,
                    }
                };
                let src: &str = &content;
                loader.add_source(url.clone(), src);
                stack.push((url, content));
            }
        }
    }

    /// Evaluate the assembled source as a module: fresh loader, network
    /// fetching disabled, relative imports pre-registered from disk
    /// (with fixture contents memoized in
    /// [`Evaluator::fixture_cache`]).
    fn exec_module(&mut self, shared: &Shared, source: &str, rel: &str) -> Result<(), EngineError> {
        let mut loader = ModuleLoader::new();
        loader.set_allow_net(false);
        let url = format!("{MODULE_URL_BASE}{rel}");
        self.register_imports(&mut loader, &url, source, &shared.test_root);
        match loader.load_source(&mut self.vm, &url, source) {
            Ok(_) => Ok(()),
            Err(ModuleError::Parse(m)) => Err(EngineError::Parse(m)),
            Err(ModuleError::Resolve(m) | ModuleError::Fetch(m)) => Err(EngineError::Resolution(m)),
            Err(ModuleError::Eval(m)) => Err(if is_module_resource_error(&url, &m) {
                EngineError::ResourceExhausted(m)
            } else {
                EngineError::Thrown(m)
            }),
        }
    }

    /// The compiled form of this job's harness preamble, compiling and
    /// caching on first use per include set. [`PreambleKind::Split`]
    /// results are additionally snapshotted (see [`PreambleSnapshot`])
    /// when eligible.
    ///
    /// The first call per include set per evaluator must happen on the
    /// clean post-reset global with no pending reset-added-keys (see
    /// [`capture_preamble_snapshot`](Self::capture_preamble_snapshot));
    /// [`run_job`](Self::run_job) hoists this call to just after the
    /// reset and before the `$DONE` protocol globals for exactly that
    /// reason. Later calls are cache hits.
    fn preamble_kind(&mut self, job: &ExecJob) -> PreambleKind {
        if let Some(kind) = self.preamble_cache.get(&job.preamble_names) {
            return kind.clone();
        }
        let kind = Self::compile_preamble(&mut self.vm, &job.preamble);
        let kind = self.snapshot_preamble(job, kind);
        self.preamble_cache
            .insert(job.preamble_names.clone(), kind.clone());
        kind
    }

    /// Try to convert a [`PreambleKind::Split`] into a
    /// [`PreambleKind::Snapshot`]: run each compiled variant once on the
    /// clean post-reset global and capture its installed globals (see
    /// [`capture_preamble_snapshot`](Self::capture_preamble_snapshot)).
    /// Anything ineligible keeps the `Split` behavior exactly. A capture
    /// that leaves undeletable residue rebuilds the evaluator and takes
    /// the plain `Split` path on the fresh `Vm` instead.
    fn snapshot_preamble(&mut self, job: &ExecJob, kind: PreambleKind) -> PreambleKind {
        let (sloppy_h, strict_h) = match kind {
            PreambleKind::Split { sloppy, strict } => (sloppy, strict),
            other => return other,
        };
        let sloppy = match self.capture_preamble_snapshot(sloppy_h) {
            SnapshotCapture::Snapshot(s) => Some(s),
            SnapshotCapture::Ineligible => None,
            // The compiled handles die with the old `Vm`; recompile the
            // plain `Split` on the fresh evaluator instead of capturing
            // again (the same preamble would leave the same residue).
            SnapshotCapture::Dirty => return self.rebuild_after_dirty_capture(job),
        };
        let strict = match self.capture_preamble_snapshot(strict_h) {
            SnapshotCapture::Snapshot(s) => Some(s),
            SnapshotCapture::Ineligible => None,
            SnapshotCapture::Dirty => return self.rebuild_after_dirty_capture(job),
        };
        match (sloppy, strict) {
            (Some(sloppy), Some(strict)) => PreambleKind::Snapshot {
                sloppy,
                strict,
                sloppy_h,
                strict_h,
            },
            _ => PreambleKind::Split {
                sloppy: sloppy_h,
                strict: strict_h,
            },
        }
    }

    /// Rebuild the evaluator after a snapshot capture left undeletable
    /// residue ([`SnapshotCapture::Dirty`]) and take the plain `Split`
    /// path on the fresh `Vm`: the `Split` fallback must not run on the
    /// dirty `Vm`, where the surviving key would corrupt its per-job
    /// runs.
    fn rebuild_after_dirty_capture(&mut self, job: &ExecJob) -> PreambleKind {
        *self = Evaluator::new();
        Self::compile_preamble(&mut self.vm, &job.preamble)
    }

    /// Run one compiled preamble variant once and capture the globals it
    /// installed as a [`PreambleSnapshot`], then remove them again so the
    /// `Vm` is back in the post-reset clean state.
    ///
    /// Returns [`SnapshotCapture::Ineligible`] (leaving the `Vm` clean
    /// either way) when the preamble is not snapshot-safe: the variant
    /// threw, a baseline global was overwritten or deleted, the heap
    /// mutation epoch moved (the run touched something besides fresh
    /// globals — a prototype, a builtin, a reset-tracked object), or an
    /// installed global is not a fresh non-baseline data property with
    /// exactly the [`Vm::set_global`] attributes. Returns
    /// [`SnapshotCapture::Dirty`] when a cleanup delete failed and
    /// residue survived: the caller must rebuild the evaluator instead
    /// of falling back to `Split` on the dirty `Vm`.
    ///
    /// Precondition: the global holds exactly the baseline keys and the
    /// reset-added-keys list is empty (true just after
    /// [`reset`](Self::reset) or a fresh [`Evaluator::new`], before the
    /// `$DONE` protocol globals are installed).
    fn capture_preamble_snapshot(&mut self, handle: ScriptHandle) -> SnapshotCapture {
        let epoch_before = self.vm.mutation_epoch();
        let run_ok = self.vm.run_compiled(handle).is_ok();
        // Installed globals, in `[[OwnPropertyKeys]]` (installation)
        // order. The precondition guarantees every non-baseline key was
        // installed by this run.
        let installed: Vec<String> = self
            .vm
            .own_property_keys(&self.global)
            .iter()
            .map(|k| k.as_wtf8().to_string())
            .filter(|k| self.baseline_keys.binary_search(k).is_err())
            .collect();
        let mut eligible =
            run_ok && self.vm.mutation_epoch() == epoch_before && self.baseline_intact();
        let mut entries = Vec::with_capacity(installed.len());
        if eligible {
            for name in &installed {
                match self.vm.get_own_descriptor(&self.global, name) {
                    Some(OwnDescriptor {
                        value: Some(v),
                        writable: Some(true),
                        enumerable: Some(true),
                        configurable: Some(true),
                        has_getter: false,
                        has_setter: false,
                    }) => {
                        // The snapshot reinstall runs after the `$DONE`
                        // protocol globals are seeded (see `run_job`), so
                        // a snapshot entry under the `$DONE` namespace
                        // would be silently overwritten. The hoist
                        // guarantees the capture sees only baseline keys
                        // before seeding, so this can only fire for a
                        // preamble that defines such a name itself.
                        debug_assert!(
                            !name.starts_with("$DONE"),
                            "snapshot captured a $DONE-protocol global: {name}"
                        );
                        entries.push((name.clone(), v))
                    }
                    // Accessor properties, non-default attributes, or a
                    // key that vanished mid-run: not snapshot-safe.
                    _ => {
                        eligible = false;
                        break;
                    }
                }
            }
        }
        // Restore the post-reset clean state whatever happened above: the
        // `Split` fallback's per-job run must start from the same state as
        // today's first run, and the snapshot path reinstalls from
        // scratch. Deletes of non-baseline keys neither bump the epoch
        // nor stay recorded; the drain below removes the run's
        // added-key records (the precondition guarantees the list holds
        // nothing else). A failed delete leaves residue behind (a
        // non-configurable install): report it so the caller rebuilds
        // instead of falling back to `Split` on the dirty `Vm`.
        let mut cleanup_ok = true;
        for name in &installed {
            if !self.vm.delete_property(&self.global, name) {
                cleanup_ok = false;
            }
        }
        let _ = self.vm.take_reset_added_keys();
        if !cleanup_ok {
            return SnapshotCapture::Dirty;
        }
        if eligible {
            // Register the shared snapshot objects as reset-tracked: a
            // later job that writes a direct property of one (e.g.
            // `assert.foo = 1`) bumps the epoch and dirties the object,
            // so the next `reset()` demotes the include set back to
            // `Split` (see [`Evaluator::demote_dirtied_snapshots`]).
            // Non-objects are ignored by `track_reset_object`.
            for (_, value) in &entries {
                self.vm.track_reset_object(value);
            }
            SnapshotCapture::Snapshot(Rc::new(PreambleSnapshot { entries }))
        } else {
            SnapshotCapture::Ineligible
        }
    }

    /// Whether every baseline global still holds its baseline value
    /// (present and [`reset_eq`]-equal): the snapshot capture's run must
    /// not have overwritten or deleted one.
    fn baseline_intact(&self) -> bool {
        self.baseline_values.iter().all(|(name, value)| {
            self.vm
                .get_global(name)
                .is_some_and(|v| reset_eq(v, *value))
        })
    }

    /// Compile one preamble text (sloppy + strict variants) for
    /// split-script reuse. Falls back to [`PreambleKind::Inline`] when the
    /// preamble declares top-level lexical bindings or cannot compile
    /// standalone.
    fn compile_preamble(vm: &mut Vm, preamble: &[Arc<str>]) -> PreambleKind {
        let mut src = String::new();
        for p in preamble {
            src.push_str(p);
            if !p.ends_with('\n') {
                src.push('\n');
            }
        }
        let sloppy = match vm.compile_script(&src) {
            Ok(handle) => handle,
            Err(_) => return PreambleKind::Inline,
        };
        if sloppy.has_top_level_lexical_bindings() {
            return PreambleKind::Inline;
        }
        let mut strict_src = String::with_capacity(src.len() + 16);
        strict_src.push_str("\"use strict\";\n");
        strict_src.push_str(&src);
        match vm.compile_script(&strict_src) {
            Ok(strict) => PreambleKind::Split { sloppy, strict },
            Err(_) => PreambleKind::Inline,
        }
    }
}

/// What [`Evaluator::run_job`] hands back to the worker thread over the
/// channel: the engine verdict plus per-phase wall-clock buckets (zeros
/// unless `TEST262_TIMING=1`).
struct JobResult {
    outcome: Result<(), EngineError>,
    phases: PhaseTimes,
    reset_stats: ResetStats,
    /// Nanoseconds spent warming the preamble cache in [`Evaluator::run_job`]
    /// (the once-per-include-set snapshot capture), zeros unless
    /// `TEST262_TIMING=1`. Attributed to a synthetic slow-list entry so the
    /// capture cost does not pollute any test's phase buckets.
    preamble_capture_ns: u64,
}

/// Phases internal to [`Evaluator::run_job_inner`]: evaluate, then the
/// async `$DONE` drain.
struct InnerResult {
    result: Result<(), EngineError>,
    eval_ns: u64,
    drain_ns: u64,
}

// ---------------------------------------------------------------------------
// Watchdog: worker thread + evaluator thread
// ---------------------------------------------------------------------------

/// Handle to a worker's evaluator thread. The worker sends jobs and waits
/// with [`WATCHDOG_TIMEOUT`]; on expiry the evaluator is abandoned (it
/// finishes on its own once fuel runs out) and a fresh one is spawned.
/// Abandoned threads are never dropped detached: their join handles are
/// retained and joined at [`EvalHandle::shutdown`], so no runaway thread
/// can outlive the harness and abort the process with a stack overflow
/// after the tests finish.
struct EvalHandle {
    tx: mpsc::Sender<ExecJob>,
    rx: mpsc::Receiver<JobResult>,
    join: Option<std::thread::JoinHandle<()>>,
    /// Join handles of watchdog-abandoned evaluator threads, kept so
    /// `shutdown` can join them. Dropping a `JoinHandle` detaches the
    /// thread; a detached evaluator doing deep recursion would eventually
    /// overflow its stack and abort the whole test binary.
    abandoned: Vec<std::thread::JoinHandle<()>>,
    /// Retained so watchdog respawns can rebuild the evaluator thread.
    shared: Arc<Shared>,
    /// Per-phase wall time of the most recent job (zeros unless
    /// `TEST262_TIMING=1`). Stashed by [`EvalHandle::run`] for the worker
    /// to read via [`EvalHandle::last_phases`].
    last_phases: PhaseTimes,
    /// Reset accounting of the most recent job. Stashed by
    /// [`EvalHandle::run`] for the worker to read via
    /// [`EvalHandle::last_reset_stats`].
    last_reset: ResetStats,
    /// Preamble-cache warm cost (ns) of the most recent job (zeros unless
    /// `TEST262_TIMING=1`). Stashed by [`EvalHandle::run`] for the worker
    /// to read via [`EvalHandle::last_capture_ns`].
    last_capture_ns: u64,
}

impl EvalHandle {
    fn spawn(shared: &Arc<Shared>) -> Self {
        let (job_tx, job_rx) = mpsc::channel::<ExecJob>();
        let (res_tx, res_rx) = mpsc::channel::<JobResult>();
        let shared = Arc::clone(shared);
        let thread_shared = Arc::clone(&shared);
        let join = std::thread::Builder::new()
            .name("test262-evaluator".to_string())
            // Deep JS recursion nests several large native frames per
            // call (see EVALUATOR_STACK_SIZE); the default 2 MiB stack
            // would abort the process on modest nesting.
            .stack_size(EVALUATOR_STACK_SIZE)
            .spawn(move || {
                let mut ev = Evaluator::new();
                for job in job_rx {
                    let r = ev.run_job(&job, &thread_shared);
                    if res_tx.send(r).is_err() {
                        break;
                    }
                }
            })
            .expect("evaluator thread spawn failed");
        EvalHandle {
            tx: job_tx,
            rx: res_rx,
            join: Some(join),
            abandoned: Vec::new(),
            shared,
            last_phases: PhaseTimes::default(),
            last_reset: ResetStats::default(),
            last_capture_ns: 0,
        }
    }

    /// Abandon the current evaluator and spawn a fresh one. The old
    /// thread is expected to exit on its own via fuel exhaustion; its
    /// join handle is retained in `abandoned` and joined at [`shutdown`]
    /// so the thread is never dropped detached.
    fn abandon(&mut self) {
        let mut abandoned = std::mem::take(&mut self.abandoned);
        if let Some(join) = self.join.take() {
            abandoned.push(join);
        }
        *self = EvalHandle::spawn(&self.shared);
        self.abandoned = abandoned;
    }

    /// Run one job, enforcing the watchdog. Never touches the `Vm`
    /// directly; all engine interaction happens on the evaluator thread.
    /// The job's per-phase wall time is stashed for
    /// [`EvalHandle::last_phases`], and its reset accounting for
    /// [`EvalHandle::last_reset_stats`].
    fn run(&mut self, job: ExecJob) -> Result<(), EngineError> {
        if self.tx.send(job).is_err() {
            // Evaluator died; replace it.
            self.abandon();
            return Err(EngineError::EvaluatorGone);
        }
        match self.rx.recv_timeout(watchdog_timeout()) {
            Ok(r) => {
                self.last_phases = r.phases;
                self.last_reset = r.reset_stats;
                self.last_capture_ns = r.preamble_capture_ns;
                r.outcome
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Watchdog: abandon the stuck evaluator (it exits on its
                // own after fuel exhaustion) and continue with a fresh one.
                self.abandon();
                Err(EngineError::Timeout)
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.abandon();
                Err(EngineError::EvaluatorGone)
            }
        }
    }

    /// Per-phase wall time (ns) of the most recent [`EvalHandle::run`]
    /// call; all zeros unless `TEST262_TIMING=1`.
    fn last_phases(&self) -> PhaseTimes {
        self.last_phases
    }

    /// Reset accounting of the most recent [`EvalHandle::run`] call.
    fn last_reset_stats(&self) -> ResetStats {
        self.last_reset
    }

    /// Preamble-cache warm cost (ns) of the most recent
    /// [`EvalHandle::run`] call; all zeros unless `TEST262_TIMING=1`.
    fn last_capture_ns(&self) -> u64 {
        self.last_capture_ns
    }

    /// Clean shutdown at the end of a worker's chunk: the evaluator is
    /// idle in `recv`, so dropping the sender lets it exit promptly.
    /// Watchdog-abandoned evaluators are joined too: they are already
    /// running down their remaining fuel, so the join returns promptly.
    /// The join is bounded by [`ABANDONED_JOIN_GRACE`]: a thread still
    /// alive past it is stuck in a fuel-unbounded loop (e.g.
    /// `run_until_idle` on self-rescheduling timers), which is a VM bug —
    /// it is left detached rather than hanging shutdown.
    fn shutdown(mut self) {
        drop(self.tx);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
        let deadline = Instant::now() + ABANDONED_JOIN_GRACE;
        for handle in self.abandoned.drain(..) {
            while !handle.is_finished() {
                if Instant::now() >= deadline {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            // Dropping `handle` here detaches an unfinished thread; see
            // the doc comment above for why that only happens on a VM bug.
        }
    }
}

// ---------------------------------------------------------------------------
// Scoreboard
// ---------------------------------------------------------------------------

/// Decided outcome of one test file across its strict/sloppy modes.
enum Outcome {
    Pass,
    Fail {
        category: &'static str,
        detail: String,
    },
    /// Skipped without running: the test requires an engine capability
    /// Bowser does not implement (see [`UNSUPPORTED_FEATURES`]). Running
    /// it could not pass, and the deep-recursion ones would overflow the
    /// evaluator's native stack — an uncatchable process abort, not a
    /// test failure — so the harness skips them up front.
    Skip {
        reason: &'static str,
        // Kept for debugging skips; not read by the scoreboard.
        _detail: String,
    },
}

#[derive(Debug, Default)]
struct DirStats {
    total: usize,
    passed: usize,
    failed: usize,
    skipped: usize,
}

#[derive(Debug)]
struct Failure {
    path: String,
    category: &'static str,
    detail: String,
}

#[derive(Debug, Default)]
struct Scoreboard {
    total: usize,
    passed: usize,
    failure_count: usize,
    skipped: usize,
    /// Module-goal tests (flag `module`): total collected and passed.
    /// All of them run now (never skipped); tracked separately so the
    /// newly-wired module suite has its own pass rate in the report.
    module_total: usize,
    module_passed: usize,
    /// Passes that came from an early (parse/compile) error on a
    /// `negative.phase: parse` test.
    negative_early_passes: usize,
    failures: Vec<Failure>,
    by_dir: BTreeMap<String, DirStats>,
    fail_categories: BTreeMap<&'static str, usize>,
    skip_reasons: BTreeMap<&'static str, usize>,
    flag_counts: BTreeMap<String, usize>,
    feature_counts: BTreeMap<String, usize>,
    /// Aggregated per-phase wall time (ns), populated only when
    /// `TEST262_TIMING=1`.
    phase_totals: PhaseTimes,
    /// (total phase time in ns, relative path) of the slowest tests,
    /// sorted descending, capped at [`TOP_SLOWEST_SHOWN`]. Populated only
    /// when `TEST262_TIMING=1`.
    slow_tests: Vec<(u64, String)>,
    /// Reset-path accounting (see [`ResetStats`]): jobs that ran a reset,
    /// and of those how many took the fast path, the slow path, ran the
    /// periodic fast-path validation, and saw the validation miss.
    resets: u64,
    resets_fast: u64,
    resets_slow: u64,
    resets_validated: u64,
    reset_validation_misses: u64,
    /// Jobs whose reset found the `Vm` poisoned beyond cheap repair and
    /// rebuilt the evaluator instead.
    poison_rebuilds: u64,
}

fn record_outcome(sb: &mut Scoreboard, dir: &str, rel: &str, outcome: Outcome) {
    let stats = sb.by_dir.entry(dir.to_string()).or_default();
    stats.total += 1;
    match outcome {
        Outcome::Pass => {
            sb.passed += 1;
            stats.passed += 1;
        }
        Outcome::Fail { category, detail } => {
            sb.failure_count += 1;
            stats.failed += 1;
            *sb.fail_categories.entry(category).or_default() += 1;
            if sb.failures.len() < MAX_FAILURES_SHOWN {
                let mut detail = detail;
                if detail.len() > 300 {
                    detail.truncate(300);
                }
                sb.failures.push(Failure {
                    path: rel.to_string(),
                    category,
                    detail,
                });
            }
        }
        Outcome::Skip { reason, .. } => {
            sb.skipped += 1;
            stats.skipped += 1;
            *sb.skip_reasons.entry(reason).or_default() += 1;
        }
    }
}

/// Merge a worker's scoreboard into the accumulator (failures keep worker
/// order, truncated to [`MAX_FAILURES_SHOWN`]).
fn merge_scoreboard(acc: &mut Scoreboard, mut other: Scoreboard) {
    acc.total += other.total;
    acc.passed += other.passed;
    acc.failure_count += other.failure_count;
    acc.skipped += other.skipped;
    acc.module_total += other.module_total;
    acc.module_passed += other.module_passed;
    acc.negative_early_passes += other.negative_early_passes;
    acc.resets += other.resets;
    acc.resets_fast += other.resets_fast;
    acc.resets_slow += other.resets_slow;
    acc.resets_validated += other.resets_validated;
    acc.reset_validation_misses += other.reset_validation_misses;
    acc.poison_rebuilds += other.poison_rebuilds;
    acc.failures.append(&mut other.failures);
    if acc.failures.len() > MAX_FAILURES_SHOWN {
        acc.failures.truncate(MAX_FAILURES_SHOWN);
    }
    for (dir, s) in other.by_dir {
        let e = acc.by_dir.entry(dir).or_default();
        e.total += s.total;
        e.passed += s.passed;
        e.failed += s.failed;
        e.skipped += s.skipped;
    }
    for (k, v) in other.fail_categories {
        *acc.fail_categories.entry(k).or_default() += v;
    }
    for (k, v) in other.skip_reasons {
        *acc.skip_reasons.entry(k).or_default() += v;
    }
    for (k, v) in other.flag_counts {
        *acc.flag_counts.entry(k).or_default() += v;
    }
    for (k, v) in other.feature_counts {
        *acc.feature_counts.entry(k).or_default() += v;
    }
    acc.phase_totals.add(&other.phase_totals);
    acc.slow_tests.extend(other.slow_tests);
    acc.slow_tests.sort_by_key(|a| std::cmp::Reverse(a.0));
    acc.slow_tests.truncate(TOP_SLOWEST_SHOWN);
}

fn collect_tests(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|e| e == "js") {
                out.push(p);
            }
        }
    }
}

/// `TEST262_SMOKE=1` subset: deterministic sampling of the sorted file
/// list. `test/language` keeps every [`SMOKE_LANGUAGE_STRIDE`]-th file
/// (~1.5k) and `test/built-ins` every [`SMOKE_BUILTINS_STRIDE`]-th file
/// (~0.6k); every other top-level directory is excluded. Applied before
/// `TEST262_FILTER` / `TEST262_EXCLUDE`, which further refine the subset.
fn smoke_filter(files: Vec<PathBuf>, test_root: &Path) -> Vec<PathBuf> {
    let mut language_n = 0usize;
    let mut builtins_n = 0usize;
    files
        .into_iter()
        .filter(|p| {
            let rel = p
                .strip_prefix(test_root)
                .map(|r| r.to_string_lossy().replace('\\', "/"))
                .unwrap_or_default();
            match rel.split('/').next().unwrap_or("") {
                "language" => {
                    let keep = language_n.is_multiple_of(SMOKE_LANGUAGE_STRIDE);
                    language_n += 1;
                    keep
                }
                "built-ins" => {
                    let keep = builtins_n.is_multiple_of(SMOKE_BUILTINS_STRIDE);
                    builtins_n += 1;
                    keep
                }
                _ => false,
            }
        })
        .collect()
}

/// Assemble a classic-script test source: the `"use strict";` directive
/// FIRST, then the harness preamble, the async `$DONE` shim, and the test
/// body. The directive must be the very first statement: a directive
/// prologue only takes effect at the start of the script, so emitting it
/// after the preamble silently ran every "strict" variant in sloppy mode.
fn assemble_classic_source(strict: bool, preamble: &[Arc<str>], shim: &str, body: &str) -> String {
    let mut source = String::new();
    if strict {
        source.push_str("\"use strict\";\n");
    }
    for p in preamble {
        source.push_str(p);
        if !p.ends_with('\n') {
            source.push('\n');
        }
    }
    if !shim.is_empty() {
        source.push_str(shim);
        source.push('\n');
    }
    source.push_str(body);
    if !body.ends_with('\n') {
        source.push('\n');
    }
    source
}

/// Assemble a classic-script test body for the split pipeline: the
/// `"use strict";` directive first for strict jobs, the async `$DONE`
/// shim, and the test body. The harness preamble runs separately (see
/// [`PreambleKind::Split`]), so for sloppy jobs *with* a preamble a
/// leading `;` neuters a `"use strict";` directive of the body itself —
/// with the preamble inline the body was never script-initial, so its
/// own directive never took effect. Raw tests have no preamble and keep
/// their exact current text.
fn assemble_body_source(strict: bool, preamble_empty: bool, shim: &str, body: &str) -> String {
    let mut source = String::new();
    if strict {
        source.push_str("\"use strict\";\n");
    } else if !preamble_empty {
        source.push_str(";\n");
    }
    if !shim.is_empty() {
        source.push_str(shim);
        source.push('\n');
    }
    source.push_str(body);
    if !body.ends_with('\n') {
        source.push('\n');
    }
    source
}

/// Build one [`ExecJob`] for a test file + mode. Harness sources come from
/// the in-memory cache; a missing include is a harness error (no
/// evaluator round-trip needed). Tests with the `raw` flag run verbatim:
/// no harness preamble is injected. `body` is the test's source, hoisted
/// to one `Arc` per test by the caller and cloned (not re-copied) per
/// mode.
fn build_job(
    shared: &Shared,
    fm: &FrontMatter,
    body: &Arc<str>,
    strict: bool,
    module: Option<Arc<str>>,
) -> Result<ExecJob, String> {
    let mut preamble = Vec::new();
    let mut seen = Vec::new();
    if !fm.is_raw {
        for name in DEFAULT_INCLUDES
            .iter()
            .map(|s| (*s).to_string())
            .chain(fm.includes.iter().cloned())
        {
            if seen.contains(&name) {
                continue;
            }
            seen.push(name.clone());
            preamble.push(shared.cache.get(&name)?);
        }
    }
    Ok(ExecJob {
        preamble,
        preamble_names: seen,
        body: Arc::clone(body),
        strict,
        is_async: fm.is_async,
        module,
        parse_only: false,
    })
}

/// Map an engine result to the test262 verdict, honoring `negative`
/// frontmatter. Mirrors the old single-threaded runner's rules.
fn verdict(result: Result<(), EngineError>, fm: &FrontMatter) -> (Outcome, bool) {
    let mut neg_early_pass = false;
    let outcome = match result {
        Ok(()) => {
            if let Some(phase) = fm.negative_phase.as_deref() {
                // A negative test that ran clean: the expected failure
                // never surfaced.
                Outcome::Fail {
                    category: "negative-no-error",
                    detail: format!("expected {phase}-phase failure, test completed"),
                }
            } else {
                Outcome::Pass
            }
        }
        Err(EngineError::Parse(msg) | EngineError::Compile(msg)) => {
            match fm.negative_phase.as_deref() {
                // The earliest stages; an early failure on a
                // parse-phase negative is an observed early failure.
                Some("parse") => {
                    neg_early_pass = true;
                    Outcome::Pass
                }
                Some(other) => Outcome::Fail {
                    category: "negative-wrong-phase",
                    detail: format!("early error before expected {other} phase: {msg}"),
                },
                None => Outcome::Fail {
                    category: "early-error",
                    detail: msg,
                },
            }
        }
        Err(EngineError::Resolution(msg)) => match fm.negative_phase.as_deref() {
            Some("resolution") => Outcome::Pass,
            Some(other) => Outcome::Fail {
                category: "negative-wrong-phase",
                detail: format!("resolution failure on {other}-phase negative: {msg}"),
            },
            None => Outcome::Fail {
                category: "resolution-error",
                detail: msg,
            },
        },
        Err(EngineError::Thrown(msg)) => match fm.negative_phase.as_deref() {
            Some("runtime") => Outcome::Pass,
            Some(other) => Outcome::Fail {
                category: "negative-wrong-phase",
                detail: format!("runtime throw on {other}-phase negative: {msg}"),
            },
            None => Outcome::Fail {
                category: "runtime-throw",
                detail: msg,
            },
        },
        Err(EngineError::ResourceExhausted(msg)) => Outcome::Fail {
            // Never a pass, even for `negative` tests: the engine ran
            // out of fuel/stack, which proves nothing about the test's
            // expected error. A negative test "passing" here would be a
            // false pass for the wrong reason.
            category: "resource-exhausted",
            detail: msg,
        },
        Err(EngineError::DoneError(msg)) => Outcome::Fail {
            category: "async-done-error",
            detail: msg,
        },
        Err(EngineError::Timeout) => Outcome::Fail {
            category: "timeout",
            detail: format!("exceeded {}s wall clock", watchdog_timeout().as_secs()),
        },
        Err(EngineError::EvaluatorGone) => Outcome::Fail {
            category: "harness-error",
            detail: "evaluator thread died".to_string(),
        },
    };
    (outcome, neg_early_pass)
}

/// The first [`UNSUPPORTED_FEATURES`] entry in the frontmatter, if any.
/// [`run_one`] skips such tests instead of running them.
fn unsupported_feature(fm: &FrontMatter) -> Option<&str> {
    fm.features
        .iter()
        .find(|f| UNSUPPORTED_FEATURES.contains(&f.as_str()))
        .map(String::as_str)
}

/// Conservative scan for a `"use strict"` directive in the script's
/// directive prologue (ECMA-262: the longest leading sequence of
/// ExpressionStatements each consisting of exactly one StringLiteral).
/// Returns `true` only when a directive with the exact source text
/// `"use strict"` or `'use strict'` is found; any ambiguity — escape
/// sequences in the literal (whose strictness effect is engine-defined),
/// a missing `;` (ASI subtleties), unterminated input, or a non-directive
/// leading statement — returns `false`.
///
/// Used to skip the redundant sloppy-mode run for `raw` tests: with no
/// preamble the sloppy assembly is the body verbatim, so a `"use strict"`
/// directive in the body's prologue is script-initial and the sloppy run
/// is already strict — identical to the strict run. Non-`raw` tests are
/// never eligible: the preamble (or the `;` neuter in
/// [`assemble_body_source`]) precedes the body, so its directive never
/// takes effect in the sloppy assembly. Async tests are never eligible
/// either: the `$DONE` shim precedes the body.
fn body_has_use_strict_directive(src: &str) -> bool {
    let b = src.as_bytes();
    let n = b.len();
    let mut i = 0;
    // Skip whitespace, comments, and a leading hashbang line.
    let skip_trivia = |mut i: usize| -> usize {
        loop {
            while i < n && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c) {
                i += 1;
            }
            if i + 1 < n && b[i] == b'/' && b[i + 1] == b'/' {
                while i < n && b[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            if i + 1 < n && b[i] == b'/' && b[i + 1] == b'*' {
                let mut j = i + 2;
                while j + 1 < n && !(b[j] == b'*' && b[j + 1] == b'/') {
                    j += 1;
                }
                i = (j + 2).min(n);
                continue;
            }
            break;
        }
        i
    };
    i = skip_trivia(i);
    // A hashbang is only legal as the very first line; tolerate it here
    // so a `#`-led file simply fails closed below (returns false) unless
    // the hashbang is genuinely first.
    if i + 1 < n && b[i] == b'#' && b[i + 1] == b'!' {
        while i < n && b[i] != b'\n' {
            i += 1;
        }
        i = skip_trivia(i);
    }
    loop {
        i = skip_trivia(i);
        if i >= n {
            return false;
        }
        let q = b[i];
        if q != b'"' && q != b'\'' {
            return false; // first non-directive statement: prologue ends
        }
        // Scan the literal; any escape bails conservative (a `\u`-escaped
        // "use strict" is not the exact directive source text).
        let mut j = i + 1;
        let mut escaped = false;
        let mut closed = false;
        while j < n {
            let c = b[j];
            if c == q {
                closed = true;
                break;
            }
            if c == b'\n' {
                return false; // unterminated: bail
            }
            if c == b'\\' {
                escaped = true;
                j += 1; // skip the backslash; the increment below skips the escaped char
                if j >= n {
                    return false;
                }
            }
            j += 1;
        }
        if !closed {
            return false; // unterminated: bail
        }
        if escaped {
            return false;
        }
        let lit = &src[i + 1..j];
        j += 1; // past the closing quote
        // The literal must terminate the statement with `;` (ASI cases
        // bail conservative).
        let k = skip_trivia(j);
        if k >= n || b[k] != b';' {
            return false;
        }
        if lit == "use strict" {
            return true;
        }
        i = k + 1;
    }
}

/// Run one test file through the driver on this worker's evaluator and
/// record the outcome.
///
/// Classic tests run in sloppy and strict mode unless a flag restricts
/// them; module-goal tests run once through the module loader (module
/// goal is intrinsically strict).
fn run_one(sb: &mut Scoreboard, shared: &Arc<Shared>, eval: &mut EvalHandle, path: &Path) {
    sb.total += 1;
    let timing = *TIMING_ENABLED;
    let rel = path
        .strip_prefix(&shared.test_root)
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| path.to_string_lossy().into_owned());
    let dir = rel.split('/').next().unwrap_or("?").to_string();
    let mut phases = PhaseTimes::default();

    // Incremental outcome cache: on a fingerprint hit, replay the
    // recorded outcome (counts, module/negative-early tallies, and the
    // pass/fail/skip itself) without touching the engine.
    if let Some(entry) = shared.outcome_cache.lookup(&rel, path) {
        for flag in &entry.flags {
            *sb.flag_counts.entry(flag.clone()).or_default() += 1;
        }
        for feature in &entry.features {
            *sb.feature_counts.entry(feature.clone()).or_default() += 1;
        }
        if entry.neg_early && matches!(entry.outcome, CachedOutcome::Pass) {
            sb.negative_early_passes += 1;
        }
        if entry.is_module {
            sb.module_total += 1;
            if matches!(entry.outcome, CachedOutcome::Pass) {
                sb.module_passed += 1;
            }
        }
        record_outcome(sb, &dir, &rel, cached_to_outcome(&entry.outcome));
        return;
    }

    let t_read = if timing { Some(Instant::now()) } else { None };
    // One `Arc` per test, cloned (not re-copied) per mode by
    // `build_job`: the body-cache key is pointer identity, so both modes
    // share the cache entry, and no per-mode memcpy or full-body hash
    // runs.
    let body: Arc<str> = match fs::read_to_string(path) {
        Ok(s) => Arc::from(s.as_str()),
        Err(e) => {
            let outcome = Outcome::Fail {
                category: "io-error",
                detail: format!("cannot read test file: {e}"),
            };
            record_and_cache(
                sb,
                shared,
                &dir,
                &rel,
                path,
                outcome,
                &[],
                &[],
                false,
                false,
            );
            return;
        }
    };
    if let Some(t) = t_read {
        phases.read = t.elapsed().as_nanos() as u64;
    }
    let t_fm = if timing { Some(Instant::now()) } else { None };
    let fm = parse_frontmatter(&body);
    if let Some(t) = t_fm {
        phases.frontmatter = t.elapsed().as_nanos() as u64;
    }

    // Static analysis for the blocker projection, independent of the run.
    for flag in &fm.flags {
        *sb.flag_counts.entry(flag.clone()).or_default() += 1;
    }
    for feature in &fm.features {
        *sb.feature_counts.entry(feature.clone()).or_default() += 1;
    }

    // Skip tests that require unimplemented engine features. The
    // tail-call-optimization tests in particular must not run: they
    // recurse 100k deep expecting TCO, and without it they overflow the
    // evaluator's native stack, aborting the whole test binary instead
    // of failing.
    if let Some(feature) = unsupported_feature(&fm) {
        let outcome = Outcome::Skip {
            reason: "unsupported-feature",
            _detail: format!("requires unimplemented feature: {feature}"),
        };
        record_and_cache(
            sb,
            shared,
            &dir,
            &rel,
            path,
            outcome,
            &fm.flags,
            &fm.features,
            fm.is_module,
            false,
        );
        return;
    }

    // Module-goal tests run once, always strict; classic tests run in
    // sloppy and strict mode unless a flag restricts them. A `raw`
    // (preamble-free), non-async classic test whose directive prologue
    // already contains `"use strict"` runs strict in *both* modes — the
    // sloppy assembly is the body verbatim, so its directive is
    // script-initial — and the second run is redundant: skip it. (Every
    // other case keeps both runs: with a preamble the body's directive
    // never takes effect in the sloppy assembly, and the async `$DONE`
    // shim precedes the body.)
    let module_rel: Option<Arc<str>> = fm.is_module.then(|| Arc::from(rel.clone()));
    let modes: &[bool] = if fm.is_module {
        &[false]
    } else if fm.only_strict {
        &[true]
    } else if fm.no_strict {
        &[false]
    } else if fm.is_raw && !fm.is_async && body_has_use_strict_directive(&body) {
        &[true]
    } else {
        &[false, true]
    };

    let mut outcome = Outcome::Pass;
    let mut neg_early_pass = false;

    // Negative `phase: parse` shortcut (classic tests only): one parse
    // decides both modes, so skip the reset + eval + drains entirely.
    // Strict mode only adds early errors, so a sloppy parse failure
    // implies a strict one, and a sloppy success fails the negative
    // test either way; `only_strict` tests parse the strict assembly
    // instead. Module-goal tests keep the old path: their parse phase
    // runs through the module loader's resolution, which a script parse
    // would not reproduce.
    if fm.negative_phase.as_deref() == Some("parse") && module_rel.is_none() {
        let mut preamble_names = Vec::new();
        if !fm.is_raw {
            for name in DEFAULT_INCLUDES
                .iter()
                .map(|s| (*s).to_string())
                .chain(fm.includes.iter().cloned())
            {
                if !preamble_names.contains(&name) {
                    preamble_names.push(name);
                }
            }
        }
        let job = ExecJob {
            preamble: Vec::new(),
            preamble_names,
            body: Arc::clone(&body),
            strict: fm.only_strict,
            is_async: false,
            module: None,
            parse_only: true,
        };
        let (mode_outcome, early) = verdict(eval.run(job), &fm);
        if timing {
            phases.add(&eval.last_phases());
        }
        let rs = eval.last_reset_stats();
        if rs.ran {
            sb.resets += 1;
            if rs.slow {
                sb.resets_slow += 1;
            } else {
                sb.resets_fast += 1;
            }
            if rs.validated {
                sb.resets_validated += 1;
            }
            if rs.validation_miss {
                sb.reset_validation_misses += 1;
            }
        }
        if rs.poison {
            sb.poison_rebuilds += 1;
        }
        neg_early_pass |= early;
        outcome = mode_outcome;
        record_and_cache(
            sb,
            shared,
            &dir,
            &rel,
            path,
            outcome,
            &fm.flags,
            &fm.features,
            fm.is_module,
            neg_early_pass,
        );
        if timing {
            let total = phases.total();
            sb.phase_totals.add(&phases);
            sb.slow_tests.push((total, rel));
            if sb.slow_tests.len() > TOP_SLOWEST_SHOWN * 4 {
                sb.slow_tests.sort_by_key(|a| std::cmp::Reverse(a.0));
                sb.slow_tests.truncate(TOP_SLOWEST_SHOWN);
            }
        }
        return;
    }

    for &strict in modes {
        let job = match build_job(shared, &fm, &body, strict, module_rel.clone()) {
            Ok(j) => j,
            Err(e) => {
                outcome = Outcome::Fail {
                    category: "harness-error",
                    detail: e,
                };
                break;
            }
        };
        let (mode_outcome, early) = verdict(eval.run(job), &fm);
        if timing {
            phases.add(&eval.last_phases());
        }
        let rs = eval.last_reset_stats();
        if rs.ran {
            sb.resets += 1;
            if rs.slow {
                sb.resets_slow += 1;
            } else {
                sb.resets_fast += 1;
            }
            if rs.validated {
                sb.resets_validated += 1;
            }
            if rs.validation_miss {
                sb.reset_validation_misses += 1;
            }
        }
        if rs.poison {
            sb.poison_rebuilds += 1;
        }
        neg_early_pass |= early;
        if !matches!(mode_outcome, Outcome::Pass) {
            outcome = mode_outcome;
            break;
        }
    }
    record_and_cache(
        sb,
        shared,
        &dir,
        &rel,
        path,
        outcome,
        &fm.flags,
        &fm.features,
        fm.is_module,
        neg_early_pass,
    );
    if timing {
        let total = phases.total();
        sb.phase_totals.add(&phases);
        sb.slow_tests.push((total, rel));
        // The preamble snapshot capture (once per include set per
        // evaluator) is attributed to a synthetic entry instead of the
        // test whose turn it happened to be.
        let capture_ns = eval.last_capture_ns();
        if capture_ns > 0 {
            sb.slow_tests
                .push((capture_ns, "<preamble snapshot capture>".to_string()));
        }
        // Bound worker-local growth; the merge keeps the global top-20.
        if sb.slow_tests.len() > TOP_SLOWEST_SHOWN * 4 {
            sb.slow_tests.sort_by_key(|a| std::cmp::Reverse(a.0));
            sb.slow_tests.truncate(TOP_SLOWEST_SHOWN);
        }
    }
}

/// Record an outcome in the scoreboard and store it in the outcome cache
/// for future runs. The module / negative-early tallies live here so
/// cached replays and fresh runs share the logic.
#[allow(clippy::too_many_arguments)]
fn record_and_cache(
    sb: &mut Scoreboard,
    shared: &Arc<Shared>,
    dir: &str,
    rel: &str,
    path: &Path,
    outcome: Outcome,
    flags: &[String],
    features: &[String],
    is_module: bool,
    neg_early: bool,
) {
    if neg_early && matches!(outcome, Outcome::Pass) {
        sb.negative_early_passes += 1;
    }
    if is_module {
        sb.module_total += 1;
        if matches!(outcome, Outcome::Pass) {
            sb.module_passed += 1;
        }
    }
    let entry =
        OutcomeCache::entry_from_outcome(path, &outcome, flags, features, is_module, neg_early);
    record_outcome(sb, dir, rel, outcome);
    if let Some(entry) = entry {
        shared.outcome_cache.insert(rel.to_string(), entry);
    }
}

/// Worker main: process a chunk of test files on one evaluator thread.
fn run_chunk(files: &[PathBuf], shared: &Arc<Shared>) -> Scoreboard {
    let mut sb = Scoreboard::default();
    let mut eval = EvalHandle::spawn(shared);
    for path in files {
        run_one(&mut sb, shared, &mut eval, path);
    }
    eval.shutdown();
    sb
}

fn print_report(
    sb: &Scoreboard,
    engine_name: &str,
    stages: Option<&[(&'static str, StageStatus)]>,
) {
    let decided = sb.passed + sb.failure_count;
    let rate = if decided == 0 {
        0.0
    } else {
        100.0 * sb.passed as f64 / decided as f64
    };
    println!("================ test262 scoreboard — bowser_js ================");
    println!("engine : {engine_name}");
    println!(
        "tests  : {} total | {} passed | {} failed | {} skipped | pass rate {rate:.2}% (of decided)",
        sb.total, sb.passed, sb.failure_count, sb.skipped,
    );
    if sb.module_total > 0 {
        let mrate = 100.0 * sb.module_passed as f64 / sb.module_total as f64;
        println!(
            "module : {} total | {} passed | pass rate {mrate:.2}% (module-goal tests)",
            sb.module_total, sb.module_passed,
        );
    }
    if sb.negative_early_passes > 0 {
        println!(
            "         ({} passes are negative/parse tests that failed before runtime)",
            sb.negative_early_passes
        );
    }
    if let Ok(f) = env::var("TEST262_FILTER") {
        println!("filter : TEST262_FILTER={f}");
    }
    if let Ok(x) = env::var("TEST262_EXCLUDE")
        && !x.trim().is_empty()
    {
        println!("exclusions : TEST262_EXCLUDE={x}");
    }
    if env::var("TEST262_SMOKE").is_ok_and(|v| v == "1") {
        println!("smoke      : TEST262_SMOKE=1 (curated subset; see header docs)");
    }
    if *QUICK {
        println!(
            "quick      : TEST262_QUICK=1 (per-test timeout {}s, watchdog {}s)",
            per_test_timeout().as_secs(),
            watchdog_timeout().as_secs()
        );
    }
    println!();
    println!("pipeline stage availability (probed from crates/bowser_js/src):");
    match stages {
        Some(rows) => {
            for (stage, status) in rows {
                println!("  {stage:<12} {}", status.label());
            }
        }
        None => println!("  (could not read crate sources)"),
    }
    println!();
    println!("per top-level directory:");
    println!(
        "  {:<12} {:>8} {:>8} {:>8} {:>8}",
        "dir", "total", "passed", "failed", "skipped"
    );
    for (dir, s) in &sb.by_dir {
        println!(
            "  {dir:<12} {:>8} {:>8} {:>8} {:>8}",
            s.total, s.passed, s.failed, s.skipped
        );
    }
    println!();
    println!("skip reasons:");
    let mut skips: Vec<(&&str, &usize)> = sb.skip_reasons.iter().collect();
    skips.sort_by(|a, b| b.1.cmp(a.1));
    for (reason, n) in skips {
        println!("  {reason:<22} {n:>8}");
    }
    println!();
    println!("failure categories (from runs):");
    let mut cats: Vec<(&&str, &usize)> = sb.fail_categories.iter().collect();
    cats.sort_by(|a, b| b.1.cmp(a.1));
    for (cat, n) in cats {
        println!("  {cat:<22} {n:>8}");
    }
    println!();
    println!("static flag counts (blocker projection):");
    let mut flags: Vec<(&String, &usize)> = sb.flag_counts.iter().collect();
    flags.sort_by(|a, b| b.1.cmp(a.1));
    for (flag, n) in flags.iter().take(12) {
        println!("  {flag:<22} {n:>8}");
    }
    println!();
    println!("static feature counts, top 15 (blocker projection):");
    let mut feats: Vec<(&String, &usize)> = sb.feature_counts.iter().collect();
    feats.sort_by(|a, b| b.1.cmp(a.1));
    for (feat, n) in feats.iter().take(15) {
        println!("  {feat:<40} {n:>8}");
    }
    println!();
    println!("first {} failures:", sb.failures.len());
    for f in &sb.failures {
        println!("  [{}] {}\n      {}", f.category, f.path, f.detail);
    }
    println!();
    println!("reset paths:");
    println!("  {:<22} {:>8}", "resets", sb.resets);
    println!("  {:<22} {:>8}", "fast path", sb.resets_fast);
    println!("  {:<22} {:>8}", "slow path", sb.resets_slow);
    println!("  {:<22} {:>8}", "fast-path validated", sb.resets_validated);
    println!(
        "  {:<22} {:>8}",
        "validation misses", sb.reset_validation_misses
    );
    println!("  {:<22} {:>8}", "poison rebuilds", sb.poison_rebuilds);
    if *TIMING_ENABLED {
        println!();
        println!("per-phase wall time (executed tests only; cached replays add 0):");
        let p = &sb.phase_totals;
        let secs = |ns: u64| format!("{:.1}s", ns as f64 / 1e9);
        println!("  {:<12} {:>10}", "phase", "total");
        println!("  {:<12} {:>10}", "read", secs(p.read));
        println!("  {:<12} {:>10}", "frontmatter", secs(p.frontmatter));
        println!("  {:<12} {:>10}", "reset", secs(p.reset));
        println!("  {:<12} {:>10}", "eval sloppy", secs(p.eval_sloppy));
        println!("  {:<12} {:>10}", "eval strict", secs(p.eval_strict));
        println!("  {:<12} {:>10}", "drain", secs(p.drain));
        println!();
        let mut slow = sb.slow_tests.clone();
        slow.sort_by_key(|a| std::cmp::Reverse(a.0));
        println!("top {} slowest tests (sum of phases):", slow.len());
        for (ns, path) in slow.iter().take(TOP_SLOWEST_SHOWN) {
            println!("  {:>8.1}s  {}", *ns as f64 / 1e9, path);
        }
    }
    println!("================================================================");
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn worker_count() -> usize {
    if let Ok(j) = env::var("TEST262_JOBS")
        && let Ok(n) = j.parse::<usize>()
    {
        return n.max(1);
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(1)
}

// ---------------------------------------------------------------------------
// Incremental outcome cache
// ---------------------------------------------------------------------------

/// Magic bytes for the outcome cache file. Bumped from `T262CH01` when
/// the fingerprint switched from hashing file metadata (paths, mtimes,
/// sizes) to hashing file contents: metadata-only caches would otherwise
/// replay stale outcomes after same-size/same-mtime edits.
const OUTCOME_CACHE_MAGIC: &[u8; 8] = b"T262CH02";
/// Cache format version; bump when [`CacheEntry`] changes.
const OUTCOME_CACHE_VERSION: u32 = 1;

/// A cached per-test result, replayed without touching the engine when
/// the test file's fingerprint (mtime + size) is unchanged and the
/// engine/harness fingerprint matches.
#[derive(Clone, Debug)]
struct CacheEntry {
    mtime_secs: u64,
    mtime_nanos: u32,
    size: u64,
    outcome: CachedOutcome,
    flags: Vec<String>,
    features: Vec<String>,
    is_module: bool,
    neg_early: bool,
}

#[derive(Clone, Debug)]
enum CachedOutcome {
    Pass,
    Fail { category: String, detail: String },
    Skip { reason: String },
}

/// On-disk cache of per-test outcomes. A full scoreboard run is dominated
/// by engine execution; when neither the engine, the harness, nor the test
/// files changed, replaying cached outcomes makes a repeat run two orders
/// of magnitude faster (collection + lookup only).
///
/// Threading: shared across worker threads via [`Shared`]; the entry map
/// is mutex-guarded (lookups are short). Entries are merged on save, so a
/// filtered run updates the cache instead of wiping it. The save is
/// atomic (temp file + rename).
#[derive(Debug)]
struct OutcomeCache {
    path: PathBuf,
    /// Engine + harness fingerprint the file was written with.
    fingerprint: u64,
    /// Whether to consult/store entries at all.
    enabled: bool,
    entries: std::sync::Mutex<std::collections::HashMap<String, CacheEntry>>,
    /// Set by [`OutcomeCache::insert`], cleared by [`OutcomeCache::save`]
    /// under the entries lock. Lets a save with no new outcomes since the
    /// last write short-circuit: the periodic per-chunk checkpoints make
    /// the final save a no-op in the common case.
    dirty: AtomicBool,
}

impl OutcomeCache {
    /// Disabled cache (unit tests).
    fn disabled() -> Self {
        OutcomeCache {
            path: PathBuf::new(),
            fingerprint: 0,
            enabled: false,
            entries: std::sync::Mutex::new(std::collections::HashMap::new()),
            dirty: AtomicBool::new(false),
        }
    }

    /// Load the cache for a scoreboard run. `TEST262_CACHE` overrides the
    /// path (`off` disables); the default sits next to the test binary's
    /// target dir so each build profile gets its own.
    fn load(manifest_dir: &Path) -> Self {
        let (path, enabled) = match std::env::var("TEST262_CACHE") {
            Ok(v) if v == "off" => (PathBuf::new(), false),
            Ok(v) if !v.is_empty() => (PathBuf::from(v), true),
            _ => (default_cache_path(), true),
        };
        let fingerprint = engine_harness_fingerprint(manifest_dir);
        let mut cache = OutcomeCache {
            path,
            fingerprint,
            enabled,
            entries: std::sync::Mutex::new(std::collections::HashMap::new()),
            dirty: AtomicBool::new(false),
        };
        if !enabled {
            return cache;
        }
        if let Ok(data) = fs::read(&cache.path)
            && let Some(entries) = parse_outcome_cache(&data, fingerprint)
        {
            cache.entries = std::sync::Mutex::new(entries);
        }
        cache
    }

    /// Look up a cached entry, verifying the test file's fingerprint.
    /// Returns `None` on any mismatch (treated as a cache miss).
    fn lookup(&self, rel: &str, path: &Path) -> Option<CacheEntry> {
        if !self.enabled {
            return None;
        }
        let entry = self.entries.lock().ok()?.get(rel)?.clone();
        let md = fs::metadata(path).ok()?;
        let mt = md.modified().ok()?;
        let d = mt.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
        if entry.mtime_secs == d.as_secs()
            && entry.mtime_nanos == d.subsec_nanos()
            && entry.size == md.len()
        {
            Some(entry)
        } else {
            None
        }
    }

    /// Store an entry (called on cache misses after running the test).
    /// Marks the cache dirty so the next periodic checkpoint persists it.
    fn insert(&self, rel: String, entry: CacheEntry) {
        if !self.enabled {
            return;
        }
        if let Ok(mut entries) = self.entries.lock() {
            entries.insert(rel, entry);
            self.dirty.store(true, Ordering::Relaxed);
        }
    }

    /// Persist the cache (merging with whatever was loaded), in the same
    /// binary format as every write. Atomic (temp file + rename).
    /// Returns `true` when a write actually happened.
    ///
    /// The dirty flag is cleared under the entries lock, so a save issued
    /// with no new outcomes since the last write is a no-op. The
    /// scoreboard calls this after every chunk (so a daemon restart
    /// mid-run loses at most one chunk's worth of outcomes) and once more
    /// at the end (normally the no-op case).
    fn save(&self) -> bool {
        if !self.enabled {
            return false;
        }
        let mut buf = Vec::new();
        {
            let entries = match self.entries.lock() {
                Ok(e) => e,
                Err(_) => return false,
            };
            if !self.dirty.swap(false, Ordering::Relaxed) {
                return false;
            }
            buf.extend_from_slice(OUTCOME_CACHE_MAGIC);
            buf.extend_from_slice(&OUTCOME_CACHE_VERSION.to_le_bytes());
            buf.extend_from_slice(&self.fingerprint.to_le_bytes());
            buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
            let mut keys: Vec<&String> = entries.keys().collect();
            keys.sort();
            for k in keys {
                let e = &entries[k];
                write_lp_str(&mut buf, k);
                buf.extend_from_slice(&e.mtime_secs.to_le_bytes());
                buf.extend_from_slice(&e.mtime_nanos.to_le_bytes());
                buf.extend_from_slice(&e.size.to_le_bytes());
                match &e.outcome {
                    CachedOutcome::Pass => buf.push(0),
                    CachedOutcome::Fail { category, detail } => {
                        buf.push(1);
                        write_lp_str(&mut buf, category);
                        write_lp_str(&mut buf, detail);
                    }
                    CachedOutcome::Skip { reason } => {
                        buf.push(2);
                        write_lp_str(&mut buf, reason);
                    }
                }
                buf.extend_from_slice(&(e.flags.len() as u32).to_le_bytes());
                for f in &e.flags {
                    write_lp_str(&mut buf, f);
                }
                buf.extend_from_slice(&(e.features.len() as u32).to_le_bytes());
                for f in &e.features {
                    write_lp_str(&mut buf, f);
                }
                buf.push(e.is_module as u8);
                buf.push(e.neg_early as u8);
            }
        }
        let tmp = self.path.with_extension("tmp");
        if fs::write(&tmp, &buf).is_ok() && fs::rename(&tmp, &self.path).is_ok() {
            true
        } else {
            // Keep the outcomes marked for a retry at the next checkpoint.
            self.dirty.store(true, Ordering::Relaxed);
            false
        }
    }

    /// Build a [`CacheEntry`] from a freshly computed outcome. Returns
    /// `None` when the file metadata is unreadable (the result is still
    /// recorded in the scoreboard; it just is not cached).
    fn entry_from_outcome(
        path: &Path,
        outcome: &Outcome,
        flags: &[String],
        features: &[String],
        is_module: bool,
        neg_early: bool,
    ) -> Option<CacheEntry> {
        let md = fs::metadata(path).ok()?;
        let mt = md.modified().ok()?;
        let d = mt.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
        let outcome = match outcome {
            Outcome::Pass => CachedOutcome::Pass,
            Outcome::Fail { category, detail } => CachedOutcome::Fail {
                category: category.to_string(),
                // Match `record_outcome`'s truncation so replays agree.
                detail: detail.chars().take(300).collect(),
            },
            Outcome::Skip { reason, .. } => CachedOutcome::Skip {
                reason: reason.to_string(),
            },
        };
        Some(CacheEntry {
            mtime_secs: d.as_secs(),
            mtime_nanos: d.subsec_nanos(),
            size: md.len(),
            outcome,
            flags: flags.to_vec(),
            features: features.to_vec(),
            is_module,
            neg_early,
        })
    }
}

/// Default cache path: next to the test binary's target dir, so each
/// build profile (and each `CARGO_TARGET_DIR`) gets its own cache.
fn default_cache_path() -> PathBuf {
    if let Ok(exe) = std::env::current_exe()
        && let Some(deps) = exe.parent()
        && let Some(profile) = deps.parent()
    {
        // <target>/<profile>/deps/test262-<hash> -> <target>.
        return profile.join("test262-outcome-cache.bin");
    }
    std::env::temp_dir().join("test262-outcome-cache.bin")
}

/// 64-bit FNV-1a.
fn fnv1a64(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Collect `.rs` files under `dir`, recursively.
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_rs_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

/// Collect every regular file under `dir`, recursively (used for the
/// vendored test262 harness: `.js` helpers plus `features.yml` and the
/// `sm/` extras). Missing directories contribute nothing.
fn collect_dir_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_dir_files(&p, out);
        } else if p.is_file() {
            out.push(p);
        }
    }
}

/// Parse `path = "../sibling"` entries out of a `Cargo.toml` and resolve
/// them against the manifest's directory. Deliberately naive (one
/// dependency per line, double-quoted path); this is a cache-invalidation
/// heuristic, and missing a dependency only risks a stale replay, not a
/// wrong one. The crate's own sources are always included regardless.
fn path_dependencies(cargo_toml: &Path) -> Vec<PathBuf> {
    let base = cargo_toml.parent().unwrap_or_else(|| Path::new("."));
    let Ok(text) = fs::read_to_string(cargo_toml) else {
        return Vec::new();
    };
    let mut deps = Vec::new();
    for line in text.lines() {
        let line = line.trim_start();
        if line.starts_with('#') || !line.contains('=') {
            continue;
        }
        // Over-inclusion is safe here: hashing an extra crate's sources
        // only risks an unnecessary cache invalidation, never a stale
        // replay.
        let Some(eq) = line.find('=') else {
            continue;
        };
        let rhs = &line[eq + 1..];
        let Some(p) = rhs.find("path") else {
            continue;
        };
        let rhs = &rhs[p + "path".len()..];
        let Some(q1) = rhs.find('"') else {
            continue;
        };
        let rest = &rhs[q1 + 1..];
        let Some(q2) = rest.find('"') else {
            continue;
        };
        deps.push(base.join(&rest[..q2]));
    }
    deps
}

/// The crate itself plus every transitive workspace path dependency of it
/// (e.g. `bowser_dom`, `bowser_css`, `bowser_url`, `bowser_encoding`,
/// `bowser_html`, `bowser_layout`, `bowser_paint`, `bowser_net`), resolved
/// from each crate's `Cargo.toml`. Changes in any of them can change test
/// outcomes, so they all participate in the fingerprint.
fn fingerprint_crate_dirs(manifest_dir: &Path) -> Vec<PathBuf> {
    let mut seen: HashSet<PathBuf> = HashSet::new();
    let mut stack = vec![manifest_dir.to_path_buf()];
    let mut crates = Vec::new();
    while let Some(dir) = stack.pop() {
        let Ok(canon) = dir.canonicalize() else {
            continue;
        };
        if !seen.insert(canon.clone()) {
            continue;
        }
        crates.push(canon.clone());
        stack.extend(path_dependencies(&canon.join("Cargo.toml")));
    }
    crates.sort();
    crates
}

/// Fingerprint the engine + harness sources: any change to the contents of
/// `crates/bowser_js/src`, the harness test files under `tests/`, the
/// vendored test262 harness JS, the crate manifests, any transitive
/// workspace path dependency, or the effective timeout profile
/// invalidates the outcome cache.
///
/// Hashes file *contents* (FNV-1a), not just paths/mtimes/sizes, so a
/// same-size edit with an unchanged mtime still invalidates the cache.
/// The timeout profile participates because `TEST262_QUICK=1` halves the
/// per-test and watchdog budgets: a `timeout` verdict under the quick
/// profile is not the same evidence as under the full profile, so the
/// two profiles must not share outcome-cache entries.
fn engine_harness_fingerprint(manifest_dir: &Path) -> u64 {
    engine_harness_fingerprint_with_timeouts(manifest_dir, per_test_timeout(), watchdog_timeout())
}

/// [`engine_harness_fingerprint`] with the timeout profile passed
/// explicitly, so unit tests can compare profiles without flipping the
/// process-global `TEST262_QUICK` switch.
fn engine_harness_fingerprint_with_timeouts(
    manifest_dir: &Path,
    per_test: Duration,
    watchdog: Duration,
) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for f in &fingerprint_source_files(manifest_dir) {
        h = fnv1a64(h, f.to_string_lossy().as_bytes());
        if let Ok(bytes) = fs::read(f) {
            h = fnv1a64(h, &bytes);
        }
    }
    h = fnv1a64(h, &per_test.as_nanos().to_le_bytes());
    h = fnv1a64(h, &watchdog.as_nanos().to_le_bytes());
    h
}

/// Every file whose contents participate in
/// [`engine_harness_fingerprint`], sorted and deduplicated: each crate's
/// `src/` plus `Cargo.toml` (transitive path dependencies included), the
/// harness test files under `tests/` (including
/// `tests/native_reset/mod.rs`), and the vendored test262 harness JS.
fn fingerprint_source_files(manifest_dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = Vec::new();
    for dir in fingerprint_crate_dirs(manifest_dir) {
        collect_rs_files(&dir.join("src"), &mut files);
        files.push(dir.join("Cargo.toml"));
    }
    files.push(manifest_dir.join("tests/test262.rs"));
    collect_rs_files(&manifest_dir.join("tests"), &mut files);
    collect_dir_files(&test262_harness_dir(manifest_dir), &mut files);
    files.sort();
    files.dedup();
    files
}

/// The vendored test262 harness directory (`harness/`, with `assert.js`,
/// `sta.js`, and the `includes/` helpers whose globals the preamble
/// snapshots): edits there change test outcomes, so it participates in
/// the fingerprint. Resolved relative to the crate dir the same way the
/// suite entry point does.
fn test262_harness_dir(manifest_dir: &Path) -> PathBuf {
    manifest_dir.join("../../vendor-sources/test262/harness")
}

fn write_lp_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

struct CacheReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> CacheReader<'a> {
    fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let b = self.data.get(self.pos..end)?;
        self.pos = end;
        Some(b)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(*self.bytes(1)?.first()?)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.bytes(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.bytes(8)?.try_into().ok()?))
    }
    fn lp_str(&mut self) -> Option<String> {
        let n = self.u32()? as usize;
        std::str::from_utf8(self.bytes(n)?)
            .ok()
            .map(|s| s.to_string())
    }
}

/// Parse a cache file, validating magic, version, and fingerprint.
/// Returns `None` (start empty) on any mismatch or corruption.
fn parse_outcome_cache(
    data: &[u8],
    fingerprint: u64,
) -> Option<std::collections::HashMap<String, CacheEntry>> {
    let mut r = CacheReader { data, pos: 0 };
    if r.bytes(8)? != OUTCOME_CACHE_MAGIC {
        return None;
    }
    if r.u32()? != OUTCOME_CACHE_VERSION {
        return None;
    }
    if r.u64()? != fingerprint {
        return None;
    }
    let count = r.u32()? as usize;
    let mut entries = std::collections::HashMap::with_capacity(count.min(65536));
    for _ in 0..count {
        let path = r.lp_str()?;
        let mtime_secs = r.u64()?;
        let mtime_nanos = r.u32()?;
        let size = r.u64()?;
        let outcome = match r.u8()? {
            0 => CachedOutcome::Pass,
            1 => CachedOutcome::Fail {
                category: r.lp_str()?,
                detail: r.lp_str()?,
            },
            2 => CachedOutcome::Skip {
                reason: r.lp_str()?,
            },
            _ => return None,
        };
        let nflags = r.u32()? as usize;
        let mut flags = Vec::with_capacity(nflags.min(32));
        for _ in 0..nflags {
            flags.push(r.lp_str()?);
        }
        let nfeatures = r.u32()? as usize;
        let mut features = Vec::with_capacity(nfeatures.min(32));
        for _ in 0..nfeatures {
            features.push(r.lp_str()?);
        }
        let is_module = r.u8()? != 0;
        let neg_early = r.u8()? != 0;
        entries.insert(
            path,
            CacheEntry {
                mtime_secs,
                mtime_nanos,
                size,
                outcome,
                flags,
                features,
                is_module,
                neg_early,
            },
        );
    }
    Some(entries)
}

/// Map a cached category string back to its `&'static str`. The set is
/// closed (see `verdict`); anything unknown conservatively becomes a
/// harness error.
fn cached_category(s: &str) -> &'static str {
    match s {
        "negative-no-error" => "negative-no-error",
        "negative-wrong-phase" => "negative-wrong-phase",
        "early-error" => "early-error",
        "resolution-error" => "resolution-error",
        "runtime-throw" => "runtime-throw",
        "resource-exhausted" => "resource-exhausted",
        "async-done-error" => "async-done-error",
        "timeout" => "timeout",
        "io-error" => "io-error",
        "harness-error" => "harness-error",
        _ => "harness-error",
    }
}

/// Map a cached skip reason back to its `&'static str`.
fn cached_reason(s: &str) -> &'static str {
    match s {
        "unsupported-feature" => "unsupported-feature",
        _ => "unsupported-feature",
    }
}

/// Rebuild an [`Outcome`] from its cached form.
fn cached_to_outcome(outcome: &CachedOutcome) -> Outcome {
    match outcome {
        CachedOutcome::Pass => Outcome::Pass,
        CachedOutcome::Fail { category, detail } => Outcome::Fail {
            category: cached_category(category),
            detail: detail.clone(),
        },
        CachedOutcome::Skip { reason } => Outcome::Skip {
            reason: cached_reason(reason),
            _detail: String::new(),
        },
    }
}

#[test]
fn test262_conformance_scoreboard() {
    // Gate: the full suite is not part of a normal `cargo test` run (it scans
    // ~54k files and is unusable in debug). Opt in explicitly with
    // TEST262_RUN=1; otherwise skip with a notice instead of scanning.
    if !env::var("TEST262_RUN").is_ok_and(|v| v == "1") {
        println!("test262: skipped (set TEST262_RUN=1 to run the full conformance suite)");
        return;
    }
    let manifest_dir =
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by cargo");
    if cfg!(debug_assertions) && !env::var("TEST262_FORCE_DEBUG").is_ok_and(|v| v == "1") {
        eprintln!(
            "================================================================\n\
             test262: WARNING — running a DEBUG build. The interpreter is\n\
             ~10-100x slower per op in debug; the full suite is effectively\n\
             unusable. Use release instead:\n\
               cargo test --release -p bowser_js --test test262 -- --nocapture\n\
             (set TEST262_FORCE_DEBUG=1 to silence this warning; the run\n\
             still proceeds)\n\
             ================================================================"
        );
    }
    let root = PathBuf::from(manifest_dir.clone()).join("../../vendor-sources/test262");
    assert!(
        root.is_dir(),
        "test262 checkout not found at {}; clone it with:\n  \
         git clone --depth 1 https://github.com/tc39/test262 {}",
        root.display(),
        root.display(),
    );
    let test_root = root.join("test");
    let harness_dir = root.join("harness");

    let filter = env::var("TEST262_FILTER").ok().filter(|f| !f.is_empty());
    let excludes: Vec<String> = env::var("TEST262_EXCLUDE")
        .ok()
        .map(|v| {
            v.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let mut files = Vec::new();
    collect_tests(&test_root, &mut files);
    files.sort();
    assert!(
        !files.is_empty(),
        "no test files found under {}",
        test_root.display()
    );
    let smoke = env::var("TEST262_SMOKE").is_ok_and(|v| v == "1");
    if smoke {
        files = smoke_filter(files, &test_root);
        eprintln!(
            "test262: SMOKE subset selected: {} files \
             (every {SMOKE_LANGUAGE_STRIDE}-th of test/language, every \
             {SMOKE_BUILTINS_STRIDE}-th of test/built-ins)",
            files.len()
        );
    }
    if *QUICK {
        eprintln!(
            "test262: QUICK profile: per-test timeout {}s, watchdog {}s \
             (halved; pathological slow tests may flip to timeout)",
            per_test_timeout().as_secs(),
            watchdog_timeout().as_secs()
        );
    }
    if *TIMING_ENABLED {
        eprintln!("test262: per-phase timing enabled (TEST262_TIMING=1)");
    }
    let files: Vec<PathBuf> = files
        .into_iter()
        .filter(|p| {
            let path = p.to_string_lossy().replace('\\', "/");
            if let Some(f) = &filter
                && !path.contains(f.as_str())
            {
                return false;
            }
            !excludes.iter().any(|e| path.contains(e.as_str()))
        })
        .collect();

    let cache = HarnessCache::load(&harness_dir).expect("harness cache load failed");
    let stages = stage_report();

    let jobs = worker_count().min(files.len().max(1));
    eprintln!("test262: running {} tests on {jobs} workers", files.len());

    let shared = Arc::new(Shared {
        test_root,
        cache,
        outcome_cache: OutcomeCache::load(Path::new(&manifest_dir)),
    });
    let suite_t0 = Instant::now();
    let mut sb = Scoreboard::default();
    std::thread::scope(|s| {
        let chunk_len = files.len().div_ceil(jobs).max(1);
        let nchunks = files.len().div_ceil(chunk_len);
        let mut handles = Vec::new();
        for chunk in files.chunks(chunk_len) {
            let shared = Arc::clone(&shared);
            handles.push(s.spawn(move || run_chunk(chunk, &shared)));
        }
        for (i, h) in handles.into_iter().enumerate() {
            merge_scoreboard(&mut sb, h.join().expect("worker panicked"));
            // Checkpoint the incremental outcome cache after every chunk:
            // the cache is shared (mutex), so everything this and earlier
            // chunks recorded is now on disk. A daemon restart mid-run
            // resumes from here instead of losing the whole run.
            let ckpt_t0 = Instant::now();
            let wrote = shared.outcome_cache.save();
            // `save` returns false when the cache is disabled, when no
            // new outcomes arrived since the last write, or when the
            // lock was poisoned — "clean (no write)" hid all three, so
            // say which one it was.
            let cache_msg = if !shared.outcome_cache.enabled {
                "disabled"
            } else if wrote {
                "checkpointed"
            } else {
                "clean (no new outcomes)"
            };
            eprintln!(
                "test262: [{}/{}] {} tests ({} passed, {} failed, {} skipped) in {:.1}s — cache {} in {:.1}s",
                i + 1,
                nchunks,
                sb.total,
                sb.passed,
                sb.failure_count,
                sb.skipped,
                suite_t0.elapsed().as_secs_f64(),
                cache_msg,
                ckpt_t0.elapsed().as_secs_f64(),
            );
        }
    });
    let elapsed = suite_t0.elapsed();

    // Final save: normally a no-op — the last chunk's checkpoint already
    // persisted everything, and the dirty flag short-circuits the write.
    // (Only writes if an outcome slipped in after the final checkpoint.)
    let cache_t0 = Instant::now();
    if shared.outcome_cache.save() {
        eprintln!(
            "test262: outcome cache saved in {:.1}s",
            cache_t0.elapsed().as_secs_f64()
        );
    } else {
        eprintln!("test262: outcome cache unchanged since last checkpoint (final save skipped)");
    }

    print_report(
        &sb,
        &format!(
            "vm driver (parser -> compiler -> Vm + event loop) [{jobs} workers, Vm reuse + watchdog]"
        ),
        stages.as_deref(),
    );
    println!(
        "ran {} tests in {:.1}s on {jobs} workers",
        sb.total,
        elapsed.as_secs_f64()
    );
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[test]
fn frontmatter_parses_flow_lists_and_scalars() {
    let src = "/*---\ndescription: some test\nesid: sec-foo\n\
         flags: [onlyStrict, async]\nincludes: [assert.js, compareArray.js]\n\
         features: [arrow-function]\n---*/\nassert.sameValue(1, 1);\n";
    let fm = parse_frontmatter(src);
    assert_eq!(fm.description, "some test");
    assert_eq!(
        fm.flags,
        vec!["onlyStrict".to_string(), "async".to_string()]
    );
    assert_eq!(
        fm.includes,
        vec!["assert.js".to_string(), "compareArray.js".to_string()]
    );
    assert_eq!(fm.features, vec!["arrow-function".to_string()]);
    assert!(fm.only_strict);
    assert!(fm.is_async);
    assert!(!fm.no_strict);
    assert!(!fm.is_module);
    assert!(fm.negative_phase.is_none());
}

#[test]
fn frontmatter_parses_block_lists_and_negative_map() {
    let src = "/*---\ndescription: negative parse test\nflags:\n  - noStrict\n  - module\n\
         includes:\n  - propertyHelper.js\nnegative:\n  phase: parse\n  type: SyntaxError\n---*/\n\
         $DONOTEVALUATE();\n";
    let fm = parse_frontmatter(src);
    assert_eq!(fm.flags, vec!["noStrict".to_string(), "module".to_string()]);
    assert_eq!(fm.includes, vec!["propertyHelper.js".to_string()]);
    assert_eq!(fm.negative_phase.as_deref(), Some("parse"));
    assert_eq!(fm.negative_type.as_deref(), Some("SyntaxError"));
    assert!(fm.no_strict);
    assert!(fm.is_module);
    assert!(!fm.only_strict);
}

#[test]
fn frontmatter_skips_block_scalars_and_unknown_nested_maps() {
    let src = "/*---\ndescription: |\n  multi\n  line description\ninfo: |\n  CallExpression :\n    CoverCallExpressionAndAsyncArrowHead\n\
         defines:\n  - assert\nflags: [onlyStrict]\n---*/\n'use strict';\n";
    let fm = parse_frontmatter(src);
    // Indented block-scalar lines must not leak into flags or description.
    assert_eq!(fm.flags, vec!["onlyStrict".to_string()]);
    assert!(fm.only_strict);
    assert!(fm.includes.is_empty());
}

#[test]
fn frontmatter_handles_quoted_items_and_empty_lists() {
    let src = "/*---\nflags: []\nfeatures: ['BigInt', \"class\"]\n---*/\n";
    let fm = parse_frontmatter(src);
    assert!(fm.flags.is_empty());
    assert_eq!(fm.features, vec!["BigInt".to_string(), "class".to_string()]);
}

#[test]
fn frontmatter_absent_yields_defaults() {
    let fm = parse_frontmatter("assert.sameValue(1, 1);\n");
    assert!(fm.flags.is_empty());
    assert!(fm.includes.is_empty());
    assert!(fm.features.is_empty());
    assert!(!fm.only_strict && !fm.is_async && !fm.is_module);
    assert!(fm.negative_phase.is_none());
}

#[test]
fn frontmatter_ignores_can_block_and_raw_flags() {
    let src = "/*---\nflags: [CanBlockIsFalse, raw]\n---*/\n";
    let fm = parse_frontmatter(src);
    assert!(fm.can_block_is_false);
    assert!(fm.is_raw);
}

#[test]
fn done_shim_source_bakes_token_literal() {
    let src = done_shim_source(42);
    assert!(src.contains("function $DONE("));
    assert!(src.contains("$DONE_called"));
    assert!(src.contains("more than once"));
    // The token is a substituted literal, guarded against the per-test
    // `$DONE_token` global; the old `$DONE_expected` twin global is gone.
    assert!(src.contains("42 !== $DONE_token"));
    assert!(!src.contains("$DONE_expected"));
    assert!(!src.contains("{token}"));
    // Distinct tokens produce distinct literals.
    assert!(done_shim_source(7).contains("7 !== $DONE_token"));
    assert!(!done_shim_source(7).contains("42 !=="));
}

#[test]
fn done_shim_cached_once_per_token() {
    let mut eval = Evaluator::new();
    // Simulate `run_job`'s async setup: seed the protocol globals, run
    // the cached shim variant for the token, call `$DONE`.
    eval.vm.set_global("$DONE_called", Value::Bool(false));
    eval.vm.set_global("$DONE_error", Value::Undefined);
    eval.vm.set_global("$DONE_token", Value::Int(7));
    let shim = eval.done_shim(7).expect("compile shim for token 7");
    eval.vm.run_compiled(shim).expect("run cached shim");
    // Same token twice: one compile, same handle both times.
    let again = eval.done_shim(7).expect("cache hit for token 7");
    assert_eq!(shim, again);
    assert_eq!(eval.done_shim_cache.len(), 1);
    eval.vm.eval("$DONE();").expect("call $DONE");
    assert!(
        matches!(eval.vm.get_global("$DONE_called"), Some(Value::Bool(true))),
        "cached $DONE must set $DONE_called"
    );
    // A second call throws the "more than once" error.
    assert!(eval.vm.eval("$DONE();").is_err());
    // Token mismatch: the call is a no-op, not a completion.
    eval.vm.set_global("$DONE_called", Value::Bool(false));
    eval.vm.set_global("$DONE_token", Value::Int(8));
    eval.vm
        .eval("$DONE();")
        .expect("mismatched call must not throw");
    assert!(matches!(
        eval.vm.get_global("$DONE_called"),
        Some(Value::Bool(false))
    ));
    // A new token compiles a second variant; the cache stays bounded.
    for token in 8..8 + DONE_SHIM_CACHE_MAX as u32 + 4 {
        eval.done_shim(token).expect("compile shim variant");
    }
    assert!(eval.done_shim_cache.len() <= DONE_SHIM_CACHE_MAX);
}

/// The protocol-global clobber this design closes: a stale `$DONE`
/// closure from a previous test's leftover async work must not complete
/// the next test, even though both protocol globals were re-seeded for
/// it. The old `$DONE_token !== $DONE_expected` check could not tell the
/// stale call apart; the baked-in literal can.
#[test]
fn stale_done_closure_from_previous_test_is_dropped() {
    let mut eval = Evaluator::new();
    // Test A (token 7): run its shim variant, stash its `$DONE`.
    eval.vm.set_global("$DONE_called", Value::Bool(false));
    eval.vm.set_global("$DONE_error", Value::Undefined);
    eval.vm.set_global("$DONE_token", Value::Int(7));
    let shim_a = eval.done_shim(7).expect("compile shim for token 7");
    eval.vm.run_compiled(shim_a).expect("run shim A");
    // Stash A's `$DONE` on the shared global object (a bare sloppy-mode
    // assignment, not `var`: top-level `var` in an `eval`'d script lives
    // in a per-eval environment and would not survive to the next eval).
    eval.vm.eval("staleA = $DONE;").expect("stash A's $DONE");
    // Test B (token 8): its own shim variant, re-seeded globals.
    eval.vm.set_global("$DONE_called", Value::Bool(false));
    eval.vm.set_global("$DONE_error", Value::Undefined);
    eval.vm.set_global("$DONE_token", Value::Int(8));
    let shim_b = eval.done_shim(8).expect("compile shim for token 8");
    eval.vm.run_compiled(shim_b).expect("run shim B");
    // The stale closure from A must not complete B — no throw (the
    // guard drops it), no state change.
    eval.vm
        .eval("staleA(new Error('stale'));")
        .expect("stale call must not throw");
    assert!(
        matches!(eval.vm.get_global("$DONE_called"), Some(Value::Bool(false))),
        "stale $DONE from the previous test must not set $DONE_called"
    );
    assert!(
        matches!(eval.vm.get_global("$DONE_error"), Some(Value::Undefined)),
        "stale $DONE from the previous test must not set $DONE_error"
    );
    // B's own `$DONE` still completes B.
    eval.vm.eval("$DONE();").expect("B's own $DONE");
    assert!(matches!(
        eval.vm.get_global("$DONE_called"),
        Some(Value::Bool(true))
    ));
}

#[test]
fn strict_directive_scan_finds_prologue_directive() {
    // Exact directive source text, both quote styles.
    assert!(body_has_use_strict_directive(
        "\"use strict\";\nvar x = 1;\n"
    ));
    assert!(body_has_use_strict_directive("'use strict';\n"));
    // Frontmatter comment and other comments/blank lines are trivia.
    assert!(body_has_use_strict_directive(
        "/*---\nflags: [raw]\n---*/\n// comment\n\n\"use strict\";\n"
    ));
    // Later in the prologue, after other directives.
    assert!(body_has_use_strict_directive(
        "\"use asm\";\n'use strict';\nfoo();\n"
    ));
    // Whitespace and comments between the literal and `;`.
    assert!(body_has_use_strict_directive("\"use strict\" /* c */ ;\n"));
    // Hashbang first: the directive after it is still found (the sloppy
    // assembly is strict there too).
    assert!(body_has_use_strict_directive(
        "#!/usr/bin/env node\n\"use strict\";\n"
    ));
}

#[test]
fn strict_directive_scan_fails_closed() {
    // No directive at all.
    assert!(!body_has_use_strict_directive("var x = 1;\n"));
    assert!(!body_has_use_strict_directive(""));
    // Directive not in the prologue (after a real statement).
    assert!(!body_has_use_strict_directive(
        "var x = 1;\n\"use strict\";\n"
    ));
    // Escaped literal: not the exact directive source text.
    assert!(!body_has_use_strict_directive("\"use\\u0020strict\";\n"));
    assert!(!body_has_use_strict_directive("'use\\x20strict';\n"));
    // Missing `;` (ASI would apply, but the scan stays conservative).
    assert!(!body_has_use_strict_directive(
        "\"use strict\"\nvar x = 1;\n"
    ));
    // String used as a value, not a directive.
    assert!(!body_has_use_strict_directive("var x = \"use strict\";\n"));
    // Unterminated literal.
    assert!(!body_has_use_strict_directive("\"use strict;\n"));
}

fn unit_cache() -> HarnessCache {
    let mut files = HashMap::new();
    files.insert("assert.js".to_string(), Arc::from("function assert() {}\n"));
    files.insert(
        "sta.js".to_string(),
        Arc::from("function Test262Error() {}\n"),
    );
    files.insert("extra.js".to_string(), Arc::from("var extra = 1;\n"));
    HarnessCache { files }
}

fn unit_shared() -> Arc<Shared> {
    Arc::new(Shared {
        test_root: PathBuf::from("/nonexistent"),
        cache: unit_cache(),
        outcome_cache: OutcomeCache::disabled(),
    })
}

#[test]
fn build_job_assembles_preamble_from_cache() {
    let shared = unit_shared();
    // assert.js listed explicitly must not be duplicated.
    let fm = FrontMatter {
        includes: vec!["assert.js".to_string(), "extra.js".to_string()],
        ..Default::default()
    };
    let job = build_job(
        &shared,
        &fm,
        &Arc::from("assert.sameValue(1, 1);"),
        true,
        None,
    )
    .unwrap();
    assert_eq!(job.preamble.len(), 3); // assert.js, sta.js, extra.js
    assert!(job.preamble[0].contains("function assert() {}"));
    assert!(job.preamble[1].contains("function Test262Error() {}"));
    assert!(job.preamble[2].contains("var extra = 1;"));
    assert!(job.strict);
    assert!(!job.is_async);

    // Missing include -> harness error, no evaluator involved.
    let fm2 = FrontMatter {
        includes: vec!["nope.js".to_string()],
        ..Default::default()
    };
    let err = build_job(&shared, &fm2, &Arc::from("1;"), false, None).unwrap_err();
    assert!(err.contains("nope.js"), "unexpected error: {err}");
}

#[test]
fn evaluator_runs_source_and_reports_pipeline_errors() {
    let shared = unit_shared();
    let mut eval = EvalHandle::spawn(&shared);
    let job = |body: &str| ExecJob {
        preamble: vec![],
        preamble_names: vec![],
        body: Arc::from(body),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    assert!(eval.run(job("var x = 1 + 2;")).is_ok());
    // Lexer failure surfaces as a parse-stage error.
    assert!(matches!(
        eval.run(job("var x = 'unterminated;")),
        Err(EngineError::Parse(_))
    ));
    // Runtime throw.
    assert!(matches!(
        eval.run(job("throw new Error('boom');")),
        Err(EngineError::Thrown(_))
    ));
    eval.shutdown();
}

/// The watchdog's abandon path retains the stuck thread's join handle and
/// `shutdown` joins it: no detached threads leak, and shutdown does not
/// hang on an abandoned thread that finishes its fuel-bounded job.
#[test]
fn watchdog_abandoned_evaluator_is_joined_at_shutdown() {
    let shared = unit_shared();
    let mut eval = EvalHandle::spawn(&shared);
    let job = |body: &str| ExecJob {
        preamble: vec![],
        preamble_names: vec![],
        body: Arc::from(body),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    // Occupy the evaluator with a fuel-bounded job, then abandon it
    // mid-flight exactly the way the watchdog does on timeout.
    eval.tx
        .send(job(
            "var s = 0; for (var i = 0; i < 300000; i++) { s += i; }",
        ))
        .unwrap();
    std::thread::sleep(Duration::from_millis(100));
    eval.abandon();
    assert_eq!(
        eval.abandoned.len(),
        1,
        "abandoned handle must be retained, not detached"
    );
    // The replacement evaluator still works.
    assert!(eval.run(job("1 + 1;")).is_ok());
    // shutdown joins the abandoned thread (it finishes its fuel-bounded
    // job on its own), so this returns promptly instead of hanging.
    let t0 = Instant::now();
    eval.shutdown();
    assert!(
        t0.elapsed() < ABANDONED_JOIN_GRACE,
        "shutdown must not hang on abandoned threads"
    );
}

#[test]
fn evaluator_resets_sloppy_globals_between_tests() {
    let shared = unit_shared();
    let mut eval = EvalHandle::spawn(&shared);
    let job = |body: &str| ExecJob {
        preamble: vec![],
        preamble_names: vec![],
        body: Arc::from(body),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    // Sloppy assignment lands on the shared global object...
    assert!(eval.run(job("leakedGlobal123 = 42;")).is_ok());
    // ...but the next test must not see it. The probe throws on a leak:
    // run() discards the completion value, so a bare comparison
    // expression would prove nothing.
    assert!(
        eval.run(job(
            "if (typeof leakedGlobal123 !== 'undefined') throw new Error('leaked');"
        ))
        .is_ok()
    );
    // Overwriting a (non-enumerable) builtin poisons the Vm: reset
    // detects the change and rebuilds, so the next test sees a
    // working Array.
    assert!(eval.run(job("Array = 5;")).is_ok());
    assert!(
        eval.run(job(
            "if (!Array.isArray([])) throw new Error('Array not restored');"
        ))
        .is_ok()
    );
    eval.shutdown();
}

/// A preamble with top-level lexical bindings must take the
/// [`PreambleKind::Inline`] fallback in `run_classic_job`: one classic
/// script cannot observe another script's `let`/`const`/`class`, so the
/// split-script fast path is unsound for it.
#[test]
fn preamble_lexical_bindings_take_inline_path() {
    let shared = unit_shared();
    let mut eval = Evaluator::new();
    let mk = |preamble: &str, body: &str| ExecJob {
        preamble: vec![Arc::from(preamble)],
        preamble_names: vec![format!("preamble-{}.js", preamble.len())],
        body: Arc::from(body),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    let lex = mk(
        "const inlineLex = 40 + 2;\n",
        "if (inlineLex !== 42) throw new Error('lexical preamble not visible');",
    );
    assert!(matches!(eval.preamble_kind(&lex), PreambleKind::Inline));
    // End to end: only the Inline assembly (preamble text prepended into
    // the same script) lets the body observe the preamble's `const`.
    let r = eval.run_job(&lex, &shared);
    assert!(r.outcome.is_ok(), "unexpected: {:?}", r.outcome);
    // var-only preambles take the snapshot fast path (see
    // [`PreambleSnapshot`]): no JS runs per job.
    let var_only = mk(
        "var splitVar = 40 + 2;\n",
        "if (splitVar !== 42) throw new Error('split preamble not visible');",
    );
    assert!(matches!(
        eval.preamble_kind(&var_only),
        PreambleKind::Snapshot { .. }
    ));
    let r = eval.run_job(&var_only, &shared);
    assert!(r.outcome.is_ok(), "unexpected: {:?}", r.outcome);
}

/// The module job's lexical-preamble branch (`run_module_job`): the
/// preamble text is prepended into the module source so its declarations
/// stay in scope for the module body.
#[test]
fn preamble_lexical_bindings_take_inline_path_in_module() {
    let shared = unit_shared();
    let mut eval = Evaluator::new();
    let job = ExecJob {
        preamble: vec![Arc::from("const inlineLexM = 6 * 7;\n")],
        preamble_names: vec!["lex-inline-m.js".to_string()],
        body: Arc::from(
            "if (inlineLexM !== 42) throw new Error('lexical preamble not visible in module');",
        ),
        strict: false,
        is_async: false,
        module: Some(Arc::from("unit/probe.js")),
        parse_only: false,
    };
    assert!(matches!(eval.preamble_kind(&job), PreambleKind::Inline));
    let r = eval.run_job(&job, &shared);
    assert!(r.outcome.is_ok(), "unexpected: {:?}", r.outcome);
}

/// The snapshot reinstall is indistinguishable from running the compiled
/// preamble: the body observes the same globals with the same values and
/// the engine's own fresh-global attributes, in installation order, for
/// both the sloppy and the strict variant.
#[test]
fn preamble_snapshot_reinstall_matches_compiled_run() {
    let shared = unit_shared();
    let mut eval = Evaluator::new();
    let mk = |body: &str, strict: bool| ExecJob {
        preamble: vec![Arc::from(
            "function snapFn() { return 41 + 1; }\nvar snapVar = 6 * 7;\n",
        )],
        preamble_names: vec!["snap-probe.js".to_string()],
        body: Arc::from(body),
        strict,
        is_async: false,
        module: None,
        parse_only: false,
    };
    assert!(matches!(
        eval.preamble_kind(&mk("", false)),
        PreambleKind::Snapshot { .. }
    ));
    for strict in [false, true] {
        let r = eval.run_job(
            &mk(
                "if (snapFn() !== 42) throw new Error('snapFn wrong');\
                 if (snapVar !== 42) throw new Error('snapVar wrong');",
                strict,
            ),
            &shared,
        );
        assert!(r.outcome.is_ok(), "unexpected: {:?}", r.outcome);
    }
    // After the last job the snapshot was reinstalled (not re-run): the
    // globals are present with the engine's fresh-global attributes.
    for name in ["snapFn", "snapVar"] {
        let d = eval
            .vm
            .get_own_descriptor(&eval.global, name)
            .expect("snapshot global missing");
        assert_eq!(
            (d.writable, d.enumerable, d.configurable),
            (Some(true), Some(true), Some(true)),
            "wrong attributes for {name}"
        );
    }
    // Overwriting a snapshotted global does not leak: the reset deletes
    // it and the reinstall restores the captured value.
    let r = eval.run_job(&mk("snapFn = 5;", false), &shared);
    assert!(r.outcome.is_ok(), "unexpected: {:?}", r.outcome);
    let r = eval.run_job(
        &mk(
            "if (typeof snapFn !== 'function' || snapFn() !== 42) throw new Error('not restored');",
            false,
        ),
        &shared,
    );
    assert!(r.outcome.is_ok(), "unexpected: {:?}", r.outcome);
}

/// Snapshot eligibility is conservative: a preamble that installs an
/// accessor property keeps the [`PreambleKind::Split`] behavior exactly.
#[test]
fn preamble_snapshot_falls_back_for_accessor() {
    let shared = unit_shared();
    let mut eval = Evaluator::new();
    let job = ExecJob {
        preamble: vec![Arc::from(
            "Object.defineProperty(globalThis, 'snapAcc', { get() { return 42; }, configurable: true });\n",
        )],
        preamble_names: vec!["snap-acc.js".to_string()],
        body: Arc::from("if (snapAcc !== 42) throw new Error('accessor not visible');"),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    assert!(matches!(
        eval.preamble_kind(&job),
        PreambleKind::Split { .. }
    ));
    let r = eval.run_job(&job, &shared);
    assert!(r.outcome.is_ok(), "unexpected: {:?}", r.outcome);
}

/// A preamble that throws cannot be snapshotted: the `Split` fallback
/// preserves the per-job throw.
#[test]
fn preamble_snapshot_falls_back_for_throwing_preamble() {
    let shared = unit_shared();
    let mut eval = Evaluator::new();
    let job = ExecJob {
        preamble: vec![Arc::from("throw new Error('preamble boom');\n")],
        preamble_names: vec!["snap-throw.js".to_string()],
        body: Arc::from(""),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    assert!(matches!(
        eval.preamble_kind(&job),
        PreambleKind::Split { .. }
    ));
    let r = eval.run_job(&job, &shared);
    assert!(
        matches!(r.outcome, Err(EngineError::Thrown(_))),
        "unexpected: {:?}",
        r.outcome
    );
}

/// A preamble that overwrites a baseline global is not snapshot-safe
/// (the install would be a `[[Set]]` on an existing binding, not a fresh
/// define): it keeps the `Split` behavior exactly.
#[test]
fn preamble_snapshot_falls_back_for_baseline_overwrite() {
    let mut eval = Evaluator::new();
    let job = ExecJob {
        preamble: vec![Arc::from("var Array = 42;\n")],
        preamble_names: vec!["snap-baseline.js".to_string()],
        body: Arc::from(""),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    assert!(matches!(
        eval.preamble_kind(&job),
        PreambleKind::Split { .. }
    ));
    // The evaluator is poisoned by the capture run's overwrite (same as
    // today's first Split run); drop it without further use.
}

/// A preamble that pollutes a prototype bumps the heap mutation epoch, so
/// it is not snapshot-safe and keeps the `Split` behavior exactly.
#[test]
fn preamble_snapshot_falls_back_for_prototype_pollution() {
    let shared = unit_shared();
    let mut eval = Evaluator::new();
    let job = ExecJob {
        preamble: vec![Arc::from("Array.prototype.snapPolluted = true;\n")],
        preamble_names: vec!["snap-proto.js".to_string()],
        body: Arc::from("if (![].snapPolluted) throw new Error('pollution not visible');"),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    assert!(matches!(
        eval.preamble_kind(&job),
        PreambleKind::Split { .. }
    ));
    let r = eval.run_job(&job, &shared);
    assert!(r.outcome.is_ok(), "unexpected: {:?}", r.outcome);
}

/// A test that mutates a snapshotted harness object no longer leaks the
/// mutation: the snapshot entries are reset-tracked at capture, so the
/// next `reset()` demotes the include set back to [`PreambleKind::Split`]
/// and later jobs get fresh objects (see [`PreambleSnapshot`]).
#[test]
fn preamble_snapshot_mutation_demotes_to_split() {
    let shared = unit_shared();
    let mut eval = Evaluator::new();
    let mk = |body: &str| ExecJob {
        preamble: vec![Arc::from("function snapShared() { return 1; }\n")],
        preamble_names: vec!["snap-shared.js".to_string()],
        body: Arc::from(body),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    assert!(matches!(
        eval.preamble_kind(&mk("")),
        PreambleKind::Snapshot { .. }
    ));
    // Mutate a direct property of the shared function object.
    let r = eval.run_job(&mk("snapShared.extra = 42;"), &shared);
    assert!(r.outcome.is_ok(), "unexpected: {:?}", r.outcome);
    // The next job's reset demoted the include set: the compiled preamble
    // re-runs, the function object is fresh, and the mutation is gone.
    let r = eval.run_job(
        &mk("if (snapShared.extra !== undefined) throw new Error('mutation leaked');"),
        &shared,
    );
    assert!(r.outcome.is_ok(), "unexpected: {:?}", r.outcome);
    assert!(
        matches!(
            eval.preamble_cache.get(&vec!["snap-shared.js".to_string()]),
            Some(PreambleKind::Split { .. })
        ),
        "dirty snapshot must demote to Split"
    );
}

/// A test that only *calls* the snapshotted harness functions never
/// dirties them: the include set stays on the [`PreambleKind::Snapshot`]
/// fast path across jobs.
#[test]
fn preamble_snapshot_survives_clean_jobs() {
    let shared = unit_shared();
    let mut eval = Evaluator::new();
    let mk = |body: &str| ExecJob {
        preamble: vec![Arc::from("function snapClean() { return 41 + 1; }\n")],
        preamble_names: vec!["snap-clean.js".to_string()],
        body: Arc::from(body),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    for _ in 0..3 {
        let r = eval.run_job(
            &mk("if (snapClean() !== 42) throw new Error('wrong');"),
            &shared,
        );
        assert!(r.outcome.is_ok(), "unexpected: {:?}", r.outcome);
    }
    assert!(
        matches!(
            eval.preamble_cache.get(&vec!["snap-clean.js".to_string()]),
            Some(PreambleKind::Snapshot { .. })
        ),
        "clean jobs must not demote the snapshot"
    );
}

/// The `$DONE` protocol globals are installed after the snapshot capture
/// (see the hoist in `run_job`), so they are never captured into the
/// snapshot: two sequential async jobs with a snapshotted preamble each
/// complete under their own token.
#[test]
fn preamble_snapshot_does_not_capture_done_token() {
    let shared = unit_shared();
    let mut eval = Evaluator::new();
    let mk = || ExecJob {
        preamble: vec![Arc::from("function snapAsync() { return 42; }\n")],
        preamble_names: vec!["snap-async.js".to_string()],
        body: Arc::from("if (snapAsync() !== 42) throw new Error('preamble missing'); $DONE();"),
        strict: false,
        is_async: true,
        module: None,
        parse_only: false,
    };
    assert!(matches!(
        eval.preamble_kind(&mk()),
        PreambleKind::Snapshot { .. }
    ));
    // A stale captured token would make the second job's $DONE a no-op
    // and time out; both must pass promptly.
    for _ in 0..2 {
        let r = eval.run_job(&mk(), &shared);
        assert!(r.outcome.is_ok(), "unexpected: {:?}", r.outcome);
    }
}

/// A capture whose cleanup cannot delete an installed global (a
/// non-configurable install) rebuilds the evaluator instead of falling
/// back to `Split` on the dirty `Vm`: the surviving key would corrupt
/// the fallback's per-job runs.
#[test]
fn preamble_snapshot_rebuilds_on_undeletable_capture_residue() {
    let shared = unit_shared();
    let mut eval = Evaluator::new();
    // Prime `tests_run` so the rebuild is observable.
    let probe = ExecJob {
        preamble: vec![],
        preamble_names: vec![],
        body: Arc::from("1;"),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    let r = eval.run_job(&probe, &shared);
    assert!(r.outcome.is_ok(), "unexpected: {:?}", r.outcome);
    assert_eq!(eval.tests_run, 1);
    let job = ExecJob {
        preamble: vec![Arc::from(
            "Object.defineProperty(globalThis, 'snapNC', { value: 42, configurable: false });\n",
        )],
        preamble_names: vec!["snap-nc.js".to_string()],
        body: Arc::from("if (snapNC !== 42) throw new Error('missing');"),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    // Ineligible (non-default attributes), and the evaluator was rebuilt
    // rather than falling back on the dirty Vm.
    assert!(matches!(
        eval.preamble_kind(&job),
        PreambleKind::Split { .. }
    ));
    assert_eq!(
        eval.tests_run, 0,
        "dirty capture residue must rebuild the evaluator"
    );
    // The fresh evaluator still runs the job: the `Split` fallback
    // installs the non-configurable global on the clean Vm.
    let r = eval.run_job(&job, &shared);
    assert!(r.outcome.is_ok(), "unexpected: {:?}", r.outcome);
}

/// A preamble that installs a `$DONE`-namespaced global would collide
/// with the `$DONE` protocol seeding that runs after the snapshot
/// reinstall: the capture asserts instead of silently capturing it.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "$DONE")]
fn preamble_snapshot_rejects_done_namespace_capture() {
    let mut eval = Evaluator::new();
    let job = ExecJob {
        preamble: vec![Arc::from("var $DONE_probe = 1;\n")],
        preamble_names: vec!["snap-done.js".to_string()],
        body: Arc::from(""),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    let _ = eval.preamble_kind(&job);
}

/// A clean test leaves the heap mutation epoch unchanged, so `reset`
/// takes the fast path: the value scans and the prototype scrub are
/// skipped, and the epoch is neither bumped nor resynced.
#[test]
fn reset_fast_path_fires_on_unchanged_epoch() {
    let shared = unit_shared();
    let mut eval = Evaluator::new();
    let clean = ExecJob {
        preamble: vec![],
        preamble_names: vec![],
        body: Arc::from("var cleanProbeVar = 1;"),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    let r = eval.run_job(&clean, &shared);
    assert!(r.outcome.is_ok(), "unexpected: {:?}", r.outcome);
    let epoch = eval.vm.mutation_epoch();
    assert_eq!(eval.synced_epoch, epoch);
    assert!(eval.reset());
    assert_eq!(
        eval.vm.mutation_epoch(),
        epoch,
        "fast path must not move the epoch"
    );
    assert_eq!(eval.synced_epoch, epoch);
}

/// Polluting a scrub target bumps the epoch, so `reset` takes the slow
/// path: it runs the full scrub (the pollution is actually removed) and
/// resyncs the epoch.
#[test]
fn reset_full_scrub_runs_on_bumped_epoch() {
    let shared = unit_shared();
    let mut eval = Evaluator::new();
    let poison = ExecJob {
        preamble: vec![],
        preamble_names: vec![],
        body: Arc::from("Array.prototype.poisonProbe = 123;"),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    let r = eval.run_job(&poison, &shared);
    assert!(r.outcome.is_ok(), "unexpected: {:?}", r.outcome);
    assert_ne!(
        eval.vm.mutation_epoch(),
        eval.synced_epoch,
        "prototype write must bump the epoch"
    );
    assert!(eval.reset());
    assert_eq!(
        eval.vm.mutation_epoch(),
        eval.synced_epoch,
        "slow path must resync the epoch"
    );
    assert_eq!(
        eval.vm
            .eval("Array.prototype.poisonProbe === undefined")
            .unwrap(),
        Value::Bool(true)
    );
}

#[test]
fn evaluator_async_done_protocol_with_token() {
    let shared = unit_shared();
    let mut eval = EvalHandle::spawn(&shared);
    let mk = |body: &str| ExecJob {
        preamble: vec![],
        preamble_names: vec![],
        body: Arc::from(body),
        strict: false,
        is_async: true,
        module: None,
        parse_only: false,
    };
    // $DONE() with no error passes; the evaluator injects the $DONE
    // shim itself and mints a fresh token per test.
    assert!(eval.run(mk("$DONE();")).is_ok());
    // $DONE(err) surfaces the error.
    assert!(matches!(
        eval.run(mk("$DONE(new Error('nope'));")),
        Err(EngineError::DoneError(_))
    ));
    eval.shutdown();
}

#[test]
fn evaluator_runs_module_tests_through_loader() {
    let shared = unit_shared();
    let mut eval = EvalHandle::spawn(&shared);
    let mk = |body: &str| ExecJob {
        preamble: vec![],
        preamble_names: vec![],
        body: Arc::from(body),
        strict: false,
        is_async: false,
        module: Some(Arc::from("unit/probe.js")),
        parse_only: false,
    };
    assert!(eval.run(mk("export const x = 1;")).is_ok());
    assert!(matches!(
        eval.run(mk("export const = ;")),
        Err(EngineError::Parse(_))
    ));
    assert!(matches!(
        eval.run(mk("import './does-not-exist.js';")),
        Err(EngineError::Resolution(_))
    ));
    assert!(matches!(
        eval.run(mk("throw new Error('kablam');")),
        Err(EngineError::Thrown(_))
    ));
    eval.shutdown();
}

#[test]
fn evaluator_async_module_done_shim_is_global() {
    // Async module jobs run the precompiled per-token `$DONE` shim as a
    // classic script (like `run_classic_job`), not as text prepended to
    // the module source: the module body resolves `$DONE` through the
    // global scope, and `globalThis` owns the binding (harness
    // `asyncHelpers.js` requires it for `asyncTest`).
    let shared = unit_shared();
    let mut eval = EvalHandle::spawn(&shared);
    let mk = |body: &str| ExecJob {
        preamble: vec![],
        preamble_names: vec![],
        body: Arc::from(body),
        strict: false,
        is_async: true,
        module: Some(Arc::from("unit/async-probe.js")),
        parse_only: false,
    };
    assert!(eval.run(mk("$DONE();")).is_ok());
    assert!(
        eval.run(mk(
            "if (typeof globalThis.$DONE !== 'function') throw new Error('no global $DONE'); $DONE();"
        ))
        .is_ok()
    );
    assert!(matches!(
        eval.run(mk("$DONE(new Error('nope'));")),
        Err(EngineError::DoneError(_))
    ));
    eval.shutdown();
}

#[test]
fn fixture_cache_memoizes_fixture_sources() {
    // `register_imports` memoizes fixture file contents per evaluator:
    // the second import of the same fixture hits the cache instead of
    // re-reading from disk. The loader still parses, compiles, and
    // evaluates per test (fresh loader per `exec_module`); only the
    // I/O is memoized.
    let dir = std::env::temp_dir().join("bowser_fixture_cache_probe");
    let _ = std::fs::create_dir_all(dir.join("unit"));
    std::fs::write(dir.join("unit/dep_FIXTURE.js"), "export const x = 1;\n").unwrap();
    let mut eval = Evaluator::new();
    assert!(eval.fixture_cache.is_empty());
    let mut loader = ModuleLoader::new();
    loader.set_allow_net(false);
    eval.register_imports(
        &mut loader,
        "https://test262.invalid/unit/main.js",
        "import { x } from './dep_FIXTURE.js';\n",
        &dir,
    );
    assert_eq!(eval.fixture_cache.len(), 1);
    assert!(
        eval.fixture_cache
            .get("https://test262.invalid/unit/dep_FIXTURE.js")
            .is_some_and(|c| c.contains("export const x = 1;"))
    );
    // Second registration of the same fixture: cache hit, no re-read
    // (remove the file; a re-read would fail and skip the import).
    std::fs::remove_file(dir.join("unit/dep_FIXTURE.js")).unwrap();
    let mut loader2 = ModuleLoader::new();
    loader2.set_allow_net(false);
    eval.register_imports(
        &mut loader2,
        "https://test262.invalid/unit/main.js",
        "import { x } from './dep_FIXTURE.js';\n",
        &dir,
    );
    assert_eq!(eval.fixture_cache.len(), 1);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn import_specifier_scanner_finds_static_imports() {
    let src = r#"
import def, {a} from "./a.js";
import * as ns from '../b.js';
import "./side.js";
export {x} from "./c.js";
"#;
    let mut specs = import_specifiers(src);
    specs.sort();
    assert!(specs.contains(&"./a.js".to_string()), "{specs:?}");
    assert!(specs.contains(&"../b.js".to_string()), "{specs:?}");
    assert!(specs.contains(&"./side.js".to_string()), "{specs:?}");
    assert!(specs.contains(&"./c.js".to_string()), "{specs:?}");
}

#[test]
fn unsupported_features_are_detected_for_skip() {
    let tco = FrontMatter {
        features: vec!["tail-call-optimization".to_string()],
        ..Default::default()
    };
    assert_eq!(unsupported_feature(&tco), Some("tail-call-optimization"));
    // A supported feature alongside an unsupported one still skips.
    let mixed = FrontMatter {
        features: vec![
            "arrow-function".to_string(),
            "tail-call-optimization".to_string(),
        ],
        ..Default::default()
    };
    assert_eq!(unsupported_feature(&mixed), Some("tail-call-optimization"));
    let ok = FrontMatter {
        features: vec!["arrow-function".to_string()],
        ..Default::default()
    };
    assert_eq!(unsupported_feature(&ok), None);
    assert_eq!(unsupported_feature(&FrontMatter::default()), None);
}

#[test]
fn stage_report_detects_wired_and_unwired_modules() {
    // The pipeline is crate-private by design (lib.rs docs: the public
    // embedder surface is Vm/Value/VmError/NativeError only). The probe
    // must see every stage as wired, never unwired: vm is harness-usable
    // through the re-exported Vm/VmError, the rest are encapsulated.
    let stages = stage_report().expect("stage_report must find the crate sources");
    for want in ["lexer", "parser", "compiler", "event_loop"] {
        let row = stages
            .iter()
            .find(|(s, _)| *s == want)
            .unwrap_or_else(|| panic!("{want} row present"));
        assert_eq!(
            row.1,
            StageStatus::Private,
            "{want} should be wired but private"
        );
    }
    let vm_row = stages
        .iter()
        .find(|(s, _)| *s == "vm")
        .expect("vm row present");
    assert_eq!(
        vm_row.1,
        StageStatus::Public,
        "vm should be public via Vm/VmError"
    );
    assert_eq!(stages.len(), STAGES.len());
}

// ---------------------------------------------------------------------------
// Harness correctness regression tests (review issues, 2026-09-28)
// ---------------------------------------------------------------------------

#[test]
fn assemble_body_source_neutering_rules() {
    // Strict: the directive must be the very first statement.
    let src = assemble_body_source(true, false, "", "body();");
    assert!(src.starts_with("\"use strict\";\n"), "{src:?}");
    // Sloppy with a preamble: a leading `;` neuters a `"use strict";`
    // directive of the body itself (with the preamble inline the body was
    // never script-initial, so its directive never took effect).
    let src = assemble_body_source(false, false, "", "\"use strict\";\nbody();");
    assert!(src.starts_with(";\n"), "{src:?}");
    // Sloppy raw (no preamble): exact current text, no prefix.
    let src = assemble_body_source(false, true, "", "\"use strict\";\nbody();");
    assert!(src.starts_with("\"use strict\";"), "{src:?}");
    // Shim placement preserved in all modes.
    let src = assemble_body_source(false, false, "/*shim*/", "body();");
    assert!(src.starts_with(";\n/*shim*/\nbody();"), "{src:?}");
    let src = assemble_body_source(true, false, "/*shim*/", "body();");
    assert!(
        src.starts_with("\"use strict\";\n/*shim*/\nbody();"),
        "{src:?}"
    );
}

#[test]
fn assemble_classic_source_orders_directive_first() {
    let preamble = vec![Arc::from("var p = 1;\n")];
    // Strict: the directive must be the very first statement — a
    // directive prologue only takes effect at the start of the script.
    let src = assemble_classic_source(true, &preamble, "", "body();");
    assert!(src.starts_with("\"use strict\";\n"), "{src:?}");
    let ppos = src.find("var p = 1;").expect("preamble present");
    assert!(ppos >= "\"use strict\";\n".len());
    // Sloppy: no directive at all.
    let src2 = assemble_classic_source(false, &preamble, "", "body();");
    assert!(!src2.contains("use strict"));
    assert!(src2.starts_with("var p = 1;"));
    // Full order: directive, preamble, shim, body.
    let src3 = assemble_classic_source(true, &preamble, "/*shim*/", "body();");
    let mut last = 0;
    for part in ["\"use strict\";", "var p = 1;", "/*shim*/", "body();"] {
        let pos = src3.find(part).expect("part present");
        assert!(pos >= last, "out of order: {part}");
        last = pos;
    }
}

#[test]
fn evaluator_strict_runs_are_actually_strict() {
    let shared = unit_shared();
    let mut eval = EvalHandle::spawn(&shared);
    let job = |body: &str, strict: bool| ExecJob {
        preamble: vec![Arc::from("var preambleVar = 1;\n")],
        preamble_names: vec!["preamble.js".to_string()],
        body: Arc::from(body),
        strict,
        is_async: false,
        module: None,
        parse_only: false,
    };
    // Sloppy assignment to an undeclared name: legal in sloppy mode...
    assert!(eval.run(job("sloppyAssignProbe = 1;", false)).is_ok());
    // ...but a ReferenceError when the strict directive takes effect.
    // Before the fix the directive was injected after the preamble, so
    // the "strict" run was sloppy and this assignment wrongly succeeded.
    assert!(
        matches!(
            eval.run(job("strictAssignProbe = 1;", true)),
            Err(EngineError::Thrown(m)) if m.contains("ReferenceError")
        ),
        "strict run must reject undeclared assignment"
    );
    eval.shutdown();
}

#[test]
fn body_cache_compiles_once_per_mode() {
    let mut eval = Evaluator::new();
    let body = Arc::from("var bodyCacheProbe = 1;");
    let h1 = eval.body_handle(false, true, &body).expect("compile body");
    let h2 = eval.body_handle(false, true, &body).expect("compile body");
    assert_eq!(h1, h2, "second lookup must hit the cache");
    assert_eq!(eval.body_cache.len(), 1);
    // Strict is a different assembly (directive prefix): its own entry.
    let h3 = eval
        .body_handle(true, true, &body)
        .expect("compile strict body");
    assert_ne!(h1, h3, "strict body must be a separate template");
    assert_eq!(eval.body_cache.len(), 2);
    // The cached template actually runs and installs its globals.
    eval.vm.run_compiled(h1).expect("run cached body");
    assert!(matches!(
        eval.vm.get_global("bodyCacheProbe"),
        Some(Value::Int(1))
    ));
}

#[test]
fn body_cache_keyed_by_pointer_identity() {
    let mut eval = Evaluator::new();
    let body = Arc::from("var bodyKeyProbe = 1;");
    // Same Arc, as `run_one` clones per mode: hit by pointer identity.
    let h1 = eval.body_handle(false, true, &body).expect("compile body");
    let h2 = eval.body_handle(false, true, &body).expect("compile body");
    assert_eq!(h1, h2, "cloned Arc must hit the cache");
    assert_eq!(eval.body_cache.len(), 1);
    // A distinct Arc with identical contents is a different key: it
    // compiles separately. Correct (same resulting semantics), just not
    // shared — the price of O(1) pointer hashing.
    let body2 = Arc::from("var bodyKeyProbe = 1;");
    assert!(!Arc::ptr_eq(&body, &body2));
    let h3 = eval.body_handle(false, true, &body2).expect("compile body");
    assert_ne!(h1, h3, "distinct Arc must not alias the entry");
    assert_eq!(eval.body_cache.len(), 2);
}

#[test]
fn parse_only_job_parses_without_running() {
    let shared = unit_shared();
    let mut eval = Evaluator::new();
    let mk = |body: &str, strict: bool| ExecJob {
        preamble: vec![],
        preamble_names: vec![],
        body: Arc::from(body),
        strict,
        is_async: false,
        module: None,
        parse_only: true,
    };
    // A body that parses cleanly but would throw if evaluated: the
    // shortcut must not run it.
    let r = eval.run_job(&mk("throw new Error('must not run');", false), &shared);
    assert!(
        r.outcome.is_ok(),
        "parse-only must not evaluate: {:?}",
        r.outcome
    );
    // A syntax error surfaces as a parse-stage failure.
    let r = eval.run_job(&mk("var x = ;", false), &shared);
    assert!(
        matches!(r.outcome, Err(EngineError::Parse(_))),
        "expected parse error, got {:?}",
        r.outcome
    );
    // The job's strict flag selects the assembly: an octal literal is
    // legal sloppy but a strict-mode early error.
    let r = eval.run_job(&mk("var octalProbe = 010;", true), &shared);
    assert!(
        matches!(
            r.outcome,
            Err(EngineError::Parse(_)) | Err(EngineError::Compile(_))
        ),
        "expected strict early error, got {:?}",
        r.outcome
    );
    let r = eval.run_job(&mk("var octalProbe = 010;", false), &shared);
    assert!(
        r.outcome.is_ok(),
        "sloppy octal literal must parse: {:?}",
        r.outcome
    );
}

#[test]
fn build_job_raw_skips_preamble() {
    let shared = unit_shared();
    let fm = FrontMatter {
        is_raw: true,
        includes: vec!["extra.js".to_string()],
        ..Default::default()
    };
    let job = build_job(&shared, &fm, &Arc::from("1;"), true, None).unwrap();
    assert!(job.preamble.is_empty(), "raw tests run verbatim");
    assert!(job.strict);
    // Non-raw still gets the default includes plus extras.
    let fm2 = FrontMatter {
        includes: vec!["extra.js".to_string()],
        ..Default::default()
    };
    let job2 = build_job(&shared, &fm2, &Arc::from("1;"), false, None).unwrap();
    assert_eq!(job2.preamble.len(), 3);
}

#[test]
fn resource_exhaustion_is_not_a_js_throw() {
    // End-to-end: real engine stack exhaustion surfaces as
    // VmError::Resource (fast: a tiny native-stack budget trips the
    // guard in under a millisecond).
    let mut vm = Vm::new();
    vm.set_max_native_stack(512 * 1024);
    let vm_err = vm
        .eval("var f = function(n){ return f(n+1); }; f(0);")
        .unwrap_err();
    assert!(
        matches!(vm_err, VmError::Resource(_)),
        "expected Resource, got {vm_err:?}"
    );
    // ...and the harness maps it to ResourceExhausted, never Thrown.
    let engine_err = map_vm_error(vm_err);
    assert!(matches!(engine_err, EngineError::ResourceExhausted(_)));

    // A negative/runtime test expecting a throw must FAIL (not pass)
    // when the engine actually ran out of resources: the exhaustion
    // proves nothing about the test's expected error.
    let fm = FrontMatter {
        negative_phase: Some("runtime".to_string()),
        negative_type: Some("RangeError".to_string()),
        ..Default::default()
    };
    let (outcome, early) = verdict(
        Err(EngineError::ResourceExhausted("stack overflow".to_string())),
        &fm,
    );
    assert!(!early);
    match outcome {
        Outcome::Fail { category, .. } => assert_eq!(category, "resource-exhausted"),
        _ => panic!("resource exhaustion must fail a negative test, not pass it"),
    }
    // Same for a non-negative test.
    let (outcome2, _) = verdict(
        Err(EngineError::ResourceExhausted("fuel exhausted".to_string())),
        &FrontMatter::default(),
    );
    assert!(matches!(
        outcome2,
        Outcome::Fail {
            category: "resource-exhausted",
            ..
        }
    ));
}

#[test]
fn module_resource_errors_map_to_resource_exhausted() {
    let url = "https://test262.invalid/unit/probe.js";
    assert!(is_module_resource_error(
        url,
        "module https://test262.invalid/unit/probe.js: resource exhausted: fuel exhausted"
    ));
    assert!(is_module_resource_error(
        url,
        "out of memory allocating module registry"
    ));
    // A JS throw whose message merely mentions resource exhaustion is
    // still a real throw: the "uncaught: " prefix breaks the match.
    assert!(!is_module_resource_error(
        url,
        "module https://test262.invalid/unit/probe.js: uncaught: resource exhausted: boom"
    ));
    assert!(!is_module_resource_error(
        url,
        "module https://test262.invalid/unit/probe.js: uncaught: Error: x"
    ));
}

#[test]
fn evaluator_scrubs_prototype_and_namespace_pollution() {
    let shared = unit_shared();
    let mut eval = EvalHandle::spawn(&shared);
    let job = |body: &str| ExecJob {
        preamble: vec![],
        preamble_names: vec![],
        body: Arc::from(body),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    // Enumerable additions to a builtin prototype and a namespace
    // object...
    assert!(
        eval.run(job(
            "Array.prototype.scrubProbeA = 1; Math.scrubProbeM = 2;"
        ))
        .is_ok()
    );
    // ...must not leak into the next test.
    assert!(
        eval.run(job(
            "if ('scrubProbeA' in Array.prototype) throw new Error('prototype leak'); \
             if ('scrubProbeM' in Math) throw new Error('namespace leak');"
        ))
        .is_ok()
    );
    eval.shutdown();
}

#[test]
fn evaluator_scrubs_non_enumerable_prototype_pollution() {
    let shared = unit_shared();
    let mut eval = EvalHandle::spawn(&shared);
    let job = |body: &str| ExecJob {
        preamble: vec![],
        preamble_names: vec![],
        body: Arc::from(body),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    // Non-enumerable additions to a builtin prototype, a namespace
    // object, and globalThis itself. `Object.keys` cannot see any of
    // these; the native scrub reads the heap property lists directly.
    assert!(
        eval.run(job("Object.defineProperty(Array.prototype, 'neProto', \
               {value: 1, enumerable: false, configurable: true}); \
             Object.defineProperty(Math, 'neMath', \
               {value: 2, enumerable: false, configurable: true}); \
             Object.defineProperty(globalThis, 'neGlobal', \
               {value: 3, enumerable: false, configurable: true});"))
            .is_ok()
    );
    // ...none of them may leak into the next test.
    assert!(
        eval.run(job(
            "if ('neProto' in Array.prototype) throw new Error('prototype leak'); \
             if ('neMath' in Math) throw new Error('namespace leak'); \
             if ('neGlobal' in globalThis) throw new Error('global leak');"
        ))
        .is_ok()
    );
    eval.shutdown();
}

#[test]
fn evaluator_scrub_survives_clobbered_own_keys() {
    let shared = unit_shared();
    let mut eval = EvalHandle::spawn(&shared);
    let job = |body: &str| ExecJob {
        preamble: vec![],
        preamble_names: vec![],
        body: Arc::from(body),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    // Clobber the live `Reflect.ownKeys` to lie (return no keys) and
    // pollute a prototype in the same test. Reset reads the heap
    // property lists natively, so the added-key scan still sees the
    // pollution; the clobbered binding itself is restored from its
    // baseline value.
    assert!(
        eval.run(job("Reflect.ownKeys = function() { return []; }; \
             Array.prototype.ownKeysProbe = 1;"))
            .is_ok()
    );
    assert!(
        eval.run(job(
            "if (Reflect.ownKeys({a: 1}).length !== 1) throw new Error('ownKeys still clobbered'); \
             if ('ownKeysProbe' in Array.prototype) throw new Error('prototype leak');"
        ))
        .is_ok()
    );
    eval.shutdown();
}

#[test]
fn evaluator_restores_overwritten_enumerable_builtin_props() {
    let shared = unit_shared();
    let mut eval = EvalHandle::spawn(&shared);
    let job = |body: &str| ExecJob {
        preamble: vec![],
        preamble_names: vec![],
        body: Arc::from(body),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    // Reflect's methods are enumerable in this engine: overwrite one.
    // (Fails loudly if the engine ever makes it non-writable, instead of
    // passing vacuously.)
    assert!(
        eval.run(job(
            "Reflect.apply = 42; if (Reflect.apply !== 42) throw new Error('setup failed');"
        ))
        .is_ok()
    );
    // The next test sees the original back.
    assert!(
        eval.run(job(
            "if (typeof Reflect.apply !== 'function') throw new Error('not restored');"
        ))
        .is_ok()
    );
    eval.shutdown();
}

#[test]
fn evaluator_rebuilds_on_nonconfigurable_prototype_pollution() {
    let shared = unit_shared();
    let mut eval = EvalHandle::spawn(&shared);
    let job = |body: &str| ExecJob {
        preamble: vec![],
        preamble_names: vec![],
        body: Arc::from(body),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    // A non-configurable addition cannot be deleted: reset must rebuild
    // the Vm instead of leaking it.
    assert!(
        eval.run(job(
            "Object.defineProperty(Array.prototype, 'ncScrub', {value: 1, configurable: false});"
        ))
        .is_ok()
    );
    assert!(
        eval.run(job(
            "if ('ncScrub' in Array.prototype) throw new Error('leaked');"
        ))
        .is_ok()
    );
    eval.shutdown();
}

#[test]
fn evaluator_scrub_repairs_enumerable_flip() {
    let shared = unit_shared();
    let mut eval = EvalHandle::spawn(&shared);
    let job = |body: &str| ExecJob {
        preamble: vec![],
        preamble_names: vec![],
        body: Arc::from(body),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    // Flip a baseline attribute: `Array.prototype.map` becomes
    // enumerable. The value comparison and the added-key scan both miss
    // this; only the attribute check sees it. (Fails loudly if the
    // engine ever ships it enumerable, instead of passing vacuously.)
    // Note: `Object.defineProperty` merges absent descriptor fields into
    // the current property, so the partial flip preserves the baseline
    // value/`writable`/`configurable` and stays configurable — this
    // exercises the scrub's A-kind *repair* path (re-define with the
    // baseline descriptor), not a Vm rebuild. The rebuild path for
    // attribute flips is covered by
    // `evaluator_rebuilds_on_nonconfigurable_attribute_flip` below, and
    // the repair path directly by
    // `scrub_attr_check_detects_and_repairs_flips`.
    assert!(
        eval.run(job(
            "Object.defineProperty(Array.prototype, 'map', {enumerable: true}); \
             var d = Object.getOwnPropertyDescriptor(Array.prototype, 'map'); \
             if (d.enumerable !== true) throw new Error('setup failed');"
        ))
        .is_ok()
    );
    // The next test must see the pristine attributes back, with no
    // for-in leakage.
    assert!(
        eval.run(job(
            "var d = Object.getOwnPropertyDescriptor(Array.prototype, 'map'); \
             if (d.enumerable !== false) throw new Error('enumerable not repaired'); \
             if (d.writable !== true || d.configurable !== true) throw new Error('attrs changed'); \
             for (var k in []) { if (k === 'map') throw new Error('for-in leak'); }"
        ))
        .is_ok()
    );
    eval.shutdown();
}

#[test]
fn evaluator_scrub_repairs_writable_flip() {
    let shared = unit_shared();
    let mut eval = EvalHandle::spawn(&shared);
    let job = |body: &str| ExecJob {
        preamble: vec![],
        preamble_names: vec![],
        body: Arc::from(body),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    // Flip `writable` to false on a baseline builtin property. Plain
    // assignment cannot repair this — only re-definition can. (The
    // partial descriptor merges, so `configurable` survives and this
    // exercises the scrub's A-kind repair path; the unit test
    // `scrub_attr_check_detects_and_repairs_flips` drives the same path
    // directly.)
    assert!(
        eval.run(job(
            "Object.defineProperty(Array.prototype, 'map', {writable: false}); \
             var d = Object.getOwnPropertyDescriptor(Array.prototype, 'map'); \
             if (d.writable !== false) throw new Error('setup failed');"
        ))
        .is_ok()
    );
    // The next test must see the baseline writability back.
    assert!(
        eval.run(job(
            "var d = Object.getOwnPropertyDescriptor(Array.prototype, 'map'); \
             if (d.writable !== true) throw new Error('writable not repaired'); \
             if (d.enumerable !== false || d.configurable !== true) throw new Error('attrs changed');"
        ))
        .is_ok()
    );
    eval.shutdown();
}

#[test]
fn evaluator_rebuilds_on_nonconfigurable_attribute_flip() {
    let shared = unit_shared();
    let mut eval = EvalHandle::spawn(&shared);
    let job = |body: &str| ExecJob {
        preamble: vec![],
        preamble_names: vec![],
        body: Arc::from(body),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    // An attribute flip on a property the test also made
    // non-configurable cannot be re-defined: reset must rebuild the Vm
    // instead of leaking the flipped attributes.
    assert!(
        eval.run(job(
            "Object.defineProperty(Array.prototype, 'map', {enumerable: true, configurable: false}); \
             var d = Object.getOwnPropertyDescriptor(Array.prototype, 'map'); \
             if (d.configurable !== false || d.enumerable !== true) throw new Error('setup failed');"
        ))
        .is_ok()
    );
    assert!(
        eval.run(job(
            "var d = Object.getOwnPropertyDescriptor(Array.prototype, 'map'); \
             if (d.enumerable !== false) throw new Error('enumerable leaked'); \
             if (d.configurable !== true) throw new Error('configurable leaked'); \
             if (typeof Array.prototype.map !== 'function') throw new Error('map missing');"
        ))
        .is_ok()
    );
    eval.shutdown();
}

#[test]
fn scrub_detects_and_repairs_attr_flips_natively() {
    // Native equivalent of the old generated-check A-path test: flip
    // attributes on two baseline properties (invisible to both the
    // added-key scan and the value comparison), then drive the native
    // scrub and confirm the baseline triples come back.
    let mut e = Evaluator::new();
    e.vm.eval("Object.defineProperty(Array.prototype, 'map', {enumerable: true});")
        .unwrap();
    e.vm.eval(
        "Object.defineProperty(Array.prototype, 'push', \
             {enumerable: false, writable: false, configurable: true});",
    )
    .unwrap();
    assert!(
        e.scrub_prototypes(),
        "native scrub must repair attribute flips"
    );
    // Baseline triples are back, values untouched.
    for (name, attr, want) in [
        ("map", "enumerable", false),
        ("map", "writable", true),
        ("map", "configurable", true),
        ("push", "enumerable", false),
        ("push", "writable", true),
        ("push", "configurable", true),
    ] {
        let got =
            e.vm.eval(&format!(
                "Object.getOwnPropertyDescriptor(Array.prototype,'{name}').{attr}"
            ))
            .unwrap();
        assert_eq!(got, Value::Bool(want), "{name}.{attr}");
    }
    match e.vm.eval("typeof Array.prototype.map").unwrap() {
        Value::Str(s) => assert_eq!(s.as_wtf8(), Wtf8::from_str("function")),
        other => panic!("unexpected typeof result: {other:?}"),
    }
    assert!(e.reset(), "full reset clean after repair");
}

#[test]
fn scrub_getter_swap_is_invisible_presence_only() {
    // Documented divergence from the old generated-JS check (which
    // compared get/set identity and rebuilt on a swap): the native
    // snapshot is presence-only, so swapping a getter for another
    // getter is invisible to the scrub.
    let mut e = Evaluator::new();
    e.vm.eval(
        "Object.defineProperty(Array.prototype, 'zzzAcc', \
             {get: function(){ return 11; }, \
              set: function(v){ globalThis.__zzzAccSeen = v; }, \
              enumerable: false, configurable: true});",
    )
    .unwrap();
    e.scrub = native_reset::build_proto_scrub(&e.vm);
    assert_eq!(e.vm.eval("Array.prototype.zzzAcc").unwrap(), Value::Int(11));
    // Swap the getter with a partial descriptor: `set`/`enumerable`/
    // `configurable` survive, so even the attribute clause sees nothing.
    e.vm.eval("Object.defineProperty(Array.prototype, 'zzzAcc', {get: function(){ return 42; }});")
        .unwrap();
    assert_eq!(e.vm.eval("Array.prototype.zzzAcc").unwrap(), Value::Int(42));
    // Presence-only: the swap is not a finding; the scrub repairs
    // nothing and reports clean (the swapped getter stays installed —
    // this is the accepted divergence).
    assert!(
        e.scrub_prototypes(),
        "getter swap must be invisible to the presence-only check"
    );
    assert_eq!(e.vm.eval("Array.prototype.zzzAcc").unwrap(), Value::Int(42));
    // But an attribute flip on the same accessor IS a finding, and it is
    // unrecoverable (the get/set pair cannot be rebuilt): the scrub
    // gives up and the evaluator rebuilds.
    e.vm.eval("Object.defineProperty(Array.prototype, 'zzzAcc', {enumerable: true});")
        .unwrap();
    assert!(
        !e.scrub_prototypes(),
        "accessor attribute flip must be unrecoverable"
    );
}

#[test]
fn evaluator_drains_microtasks_at_end_of_test() {
    let shared = unit_shared();
    // Premise: without a drain, the queued microtask has not run yet.
    let mut bare = Vm::new();
    bare.eval("Promise.resolve().then(function(){ globalThis.__drainPremise = 1; });")
        .unwrap();
    assert!(bare.get_global("__drainPremise").is_none());
    assert!(bare.event_loop().has_pending_work());

    // Through the evaluator, the end-of-test drain runs the queued job
    // before the next test's reset (reset runs at the START of run_job),
    // so the effect is visible here and the queue is empty.
    let mut ev = Evaluator::new();
    let job = ExecJob {
        preamble: vec![],
        preamble_names: vec![],
        body: Arc::from("Promise.resolve().then(function(){ globalThis.__drainFlag = 7; });"),
        strict: false,
        is_async: false,
        module: None,
        parse_only: false,
    };
    ev.run_job(&job, &shared).outcome.unwrap();
    assert!(matches!(
        ev.vm.get_global("__drainFlag"),
        Some(Value::Int(7))
    ));
    assert!(!ev.vm.event_loop().has_pending_work());
}

#[test]
fn import_specifier_scanner_skips_comments_and_strings() {
    let src = r#"
// import def from "./line-comment.js";
/* import "./block-comment.js"; */
/* multi-line
   import x from "./block2.js"; */
var s = "import './str1.js';";
var t = 'from "./str2.js"';
var u = `import "./tpl.js";`;
var v = "escaped \" import './esc.js';";
import real from "./real.js";
import "./side-effect.js";
"#;
    let mut specs = import_specifiers(src);
    specs.sort();
    assert_eq!(
        specs,
        vec!["./real.js".to_string(), "./side-effect.js".to_string()]
    );
}

#[test]
fn import_specifier_scanner_handles_edge_cases() {
    // Unterminated string: no panic, no spec.
    assert!(import_specifiers("import x from './never-ends;").is_empty());
    // Non-ASCII identifiers: no panic, real imports still found.
    let specs = import_specifiers("var caf\u{e9} = 1;\nimport y from './uni.js';");
    assert_eq!(specs, vec!["./uni.js".to_string()]);
    // `import` as an identifier substring is not a keyword.
    assert!(import_specifiers("var important = 1;").is_empty());
    assert!(import_specifiers("var ximport = 1;").is_empty());
    // A comment opener inside a string does not start a comment.
    let specs2 = import_specifiers("var s = '// not a comment';\nimport z from './z.js';");
    assert_eq!(specs2, vec!["./z.js".to_string()]);
}

// ---------------------------------------------------------------------------
// Fingerprint unit tests
// ---------------------------------------------------------------------------

/// Build a fake two-crate workspace under a unique temp dir:
/// `root/<name>/` holds `Cargo.toml` (with one path dep) and `src/lib.rs`,
/// `root/<dep>/` holds `Cargo.toml` and `src/lib.rs`.
fn fake_workspace(name: &str) -> (PathBuf, PathBuf) {
    let root =
        std::env::temp_dir().join(format!("test262-fingerprint-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let main = root.join(name);
    let dep = root.join("fake_dep");
    fs::create_dir_all(main.join("src")).unwrap();
    fs::create_dir_all(dep.join("src")).unwrap();
    fs::write(
        main.join("Cargo.toml"),
        "[package]\nname = \"fake_main\"\n[dependencies]\nfake_dep = { path = \"../fake_dep\" }\n",
    )
    .unwrap();
    fs::write(main.join("src/lib.rs"), "pub fn main_fn() {}\n").unwrap();
    fs::write(dep.join("Cargo.toml"), "[package]\nname = \"fake_dep\"\n").unwrap();
    fs::write(dep.join("src/lib.rs"), "pub fn dep_fn() {}\n").unwrap();
    (root, main)
}

#[test]
fn engine_fingerprint_is_stable_for_identical_trees() {
    let (_root, main) = fake_workspace("stable");
    let a = engine_harness_fingerprint(&main);
    let b = engine_harness_fingerprint(&main);
    assert_eq!(a, b, "fingerprint of an unchanged tree must be stable");
}

#[test]
fn engine_fingerprint_catches_same_size_content_change() {
    let (_root, main) = fake_workspace("samesize");
    let before = engine_harness_fingerprint(&main);
    // Same byte length as the original ("pub fn main_fn() {}\n"), so the
    // old metadata-only hash (paths + mtimes + sizes) would not notice.
    fs::write(main.join("src/lib.rs"), "pub fn main_fX() {}\n").unwrap();
    let after = engine_harness_fingerprint(&main);
    assert_ne!(
        before, after,
        "a same-size content edit must invalidate the fingerprint"
    );
}

#[test]
fn engine_fingerprint_covers_path_dependencies() {
    let (_root, main) = fake_workspace("pathdep");
    let before = engine_harness_fingerprint(&main);
    // Same byte length as the original dep file, so only content hashing
    // (and dep inclusion) can catch this.
    fs::write(main.join("../fake_dep/src/lib.rs"), "pub fn dep_fX() {}\n").unwrap();
    let after = engine_harness_fingerprint(&main);
    assert_ne!(
        before, after,
        "a same-size edit in a path dependency must invalidate the fingerprint"
    );
}

/// The timeout profile participates in the fingerprint: `TEST262_QUICK=1`
/// halves the per-test and watchdog budgets, so a `timeout` verdict under
/// the quick profile must never be replayed by a full-profile run (or
/// vice versa) — the two profiles get disjoint outcome-cache entries.
#[test]
fn engine_fingerprint_includes_timeout_profile() {
    let (_root, main) = fake_workspace("timeoutprofile");
    let full = engine_harness_fingerprint_with_timeouts(
        &main,
        Duration::from_secs(10),
        Duration::from_secs(20),
    );
    let quick = engine_harness_fingerprint_with_timeouts(
        &main,
        Duration::from_secs(5),
        Duration::from_secs(10),
    );
    assert_ne!(
        full, quick,
        "QUICK and full timeout profiles must not share cache entries"
    );
    assert_eq!(
        full,
        engine_harness_fingerprint_with_timeouts(
            &main,
            Duration::from_secs(10),
            Duration::from_secs(20)
        ),
        "fingerprint must be stable for the same profile"
    );
}

/// The harness test files under `tests/` (including
/// `tests/native_reset/mod.rs`) participate in the fingerprint: editing
/// the reset helper without touching `tests/test262.rs` or `src/` must
/// still invalidate the outcome cache.
#[test]
fn engine_fingerprint_covers_tests_dir() {
    let (_root, main) = fake_workspace("testsdir");
    fs::create_dir_all(main.join("tests/native_reset")).unwrap();
    fs::write(main.join("tests/native_reset/mod.rs"), "// reset helper\n").unwrap();
    let before = engine_harness_fingerprint(&main);
    fs::write(main.join("tests/native_reset/mod.rs"), "// reset helper!\n").unwrap();
    let after = engine_harness_fingerprint(&main);
    assert_ne!(
        before, after,
        "edits under tests/ must invalidate the fingerprint"
    );
}

/// Build a fake crate whose layout mirrors the real one for the vendored
/// harness: `root/ws/crates/<name>` is the manifest dir, so
/// `../../vendor-sources/test262/harness` resolves under the temp root.
fn fake_workspace_with_harness(name: &str) -> (PathBuf, PathBuf) {
    let root =
        std::env::temp_dir().join(format!("test262-fingerprint-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let main = root.join("ws/crates").join(name);
    let harness = root.join("ws/vendor-sources/test262/harness");
    fs::create_dir_all(main.join("src")).unwrap();
    fs::create_dir_all(&harness).unwrap();
    fs::write(main.join("Cargo.toml"), "[package]\nname = \"fake_main\"\n").unwrap();
    fs::write(main.join("src/lib.rs"), "pub fn main_fn() {}\n").unwrap();
    fs::write(harness.join("assert.js"), "function assert() {}\n").unwrap();
    (root, main)
}

/// The vendored test262 harness JS participates in the fingerprint: the
/// preamble snapshots its globals, so a harness edit changes outcomes.
#[test]
fn engine_fingerprint_covers_vendored_harness_js() {
    let (_root, main) = fake_workspace_with_harness("harnessjs");
    let before = engine_harness_fingerprint(&main);
    fs::write(
        test262_harness_dir(&main).join("assert.js"),
        "function assert() { /* changed */ }\n",
    )
    .unwrap();
    let after = engine_harness_fingerprint(&main);
    assert_ne!(
        before, after,
        "harness JS edits must invalidate the fingerprint"
    );
}

/// The real crate layout wires every fingerprinted source family:
/// engine `src/`, the harness test files, and the vendored harness JS.
#[test]
fn fingerprint_source_files_cover_all_families() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let files = fingerprint_source_files(manifest_dir);
    let has = |suffix: &str| {
        files
            .iter()
            .any(|p| p.to_string_lossy().replace('\\', "/").ends_with(suffix))
    };
    assert!(
        has("tests/test262.rs"),
        "harness test file must be fingerprinted"
    );
    assert!(
        has("tests/native_reset/mod.rs"),
        "native reset helper must be fingerprinted"
    );
    assert!(
        has("src/vm.rs") || has("src/lib.rs"),
        "engine sources must be fingerprinted"
    );
    let harness_dir = test262_harness_dir(manifest_dir);
    if harness_dir.is_dir() {
        assert!(
            has("vendor-sources/test262/harness/assert.js"),
            "vendored harness JS must be fingerprinted"
        );
    }
}

// ---------------------------------------------------------------------------
// Outcome-cache checkpoint unit tests
// ---------------------------------------------------------------------------

/// Simulate a mid-run daemon restart: run a "chunk" (insert + checkpoint),
/// drop all state, rebuild the cache from disk the way
/// [`OutcomeCache::load`] does, and verify the checkpointed outcomes are
/// reused — never re-executed — with the existing fingerprint/mtime/size
/// validation still enforced.
#[test]
fn outcome_cache_checkpoint_survives_restart() {
    let root = std::env::temp_dir().join(format!("test262-cache-restart-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let cache_path = root.join("outcome-cache.bin");
    let test_file = root.join("some-test.js");
    fs::write(&test_file, "assert.sameValue(1, 1);\n").unwrap();
    // Fixed stand-in for `engine_harness_fingerprint()`; the point is that
    // the reloaded cache must validate against the SAME fingerprint.
    let fingerprint = 0x9e37_79b9_7f4a_7c15u64;
    let make_cache = || OutcomeCache {
        path: cache_path.clone(),
        fingerprint,
        enabled: true,
        entries: std::sync::Mutex::new(std::collections::HashMap::new()),
        dirty: AtomicBool::new(false),
    };

    // Chunk 1: record an outcome and checkpoint.
    let cache = make_cache();
    let entry = OutcomeCache::entry_from_outcome(
        &test_file,
        &Outcome::Pass,
        &["onlyStrict".to_string()],
        &["arrow-function".to_string()],
        true,
        false,
    )
    .expect("entry_from_outcome must read the temp test file");
    cache.insert("some-test.js".to_string(), entry);
    assert!(cache.save(), "checkpoint with new outcomes must write");
    assert!(
        !cache.save(),
        "checkpoint with no new outcomes must be a no-op"
    );

    // Simulate the daemon restart: drop all in-memory state, then rebuild
    // from the on-disk checkpoint exactly like `OutcomeCache::load` does
    // (read file, validate magic/version/fingerprint via
    // `parse_outcome_cache`).
    drop(cache);
    let data = fs::read(&cache_path).expect("checkpoint file must exist");
    let entries = parse_outcome_cache(&data, fingerprint)
        .expect("checkpoint must parse under the same fingerprint");
    assert!(
        parse_outcome_cache(&data, fingerprint ^ 1).is_none(),
        "fingerprint mismatch must invalidate the checkpoint"
    );
    let resumed = OutcomeCache {
        path: cache_path.clone(),
        fingerprint,
        enabled: true,
        entries: std::sync::Mutex::new(entries),
        dirty: AtomicBool::new(false),
    };

    // The checkpointed outcome is reused: same pass, same flags/features,
    // same module tally bits — the test must NOT be re-executed.
    let hit = resumed
        .lookup("some-test.js", &test_file)
        .expect("must be a cache hit after reload");
    assert!(matches!(hit.outcome, CachedOutcome::Pass));
    assert_eq!(hit.flags, vec!["onlyStrict".to_string()]);
    assert_eq!(hit.features, vec!["arrow-function".to_string()]);
    assert!(hit.is_module);
    // Nothing new inserted since the reload, so a save is a no-op.
    assert!(!resumed.save(), "clean reloaded cache must not rewrite");

    // Editing the test file invalidates the entry (mtime/size validation).
    fs::write(&test_file, "assert.sameValue(1, 1);\n// changed\n").unwrap();
    assert!(
        resumed.lookup("some-test.js", &test_file).is_none(),
        "edited test file must miss the cache"
    );

    let _ = fs::remove_dir_all(&root);
}
