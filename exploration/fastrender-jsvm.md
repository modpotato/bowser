# fastrender JS engine survey (vm-js)

**Status / provenance caveat:** The local clone at `~/workspace/bowser/vendor-sources/fastrender`
never materialized (empty `.git` dir after 20 minutes of retrying; zero commits, no checked-out
files). This survey is therefore based on fastrender's public design docs, the `instructions/js_engine.md`
workstream spec (highest-priority JS workstream), `docs/js_embedding.md`, the public conformance table,
and recent commit messages for `vendor/ecma-rs/vm-js` — **not** on reading the source. All verdicts
should be re-validated against the actual tree once the clone lands. Sibling explorers can corroborate.

**What fastrender is:** experimental Rust browser engine (wilsonzlin/fastrender), itself largely
agent-written ("research into collaborative parallel AI coding agents"). Its JS engine is **vm-js**,
part of the `vendor/ecma-rs/` workspace that fastrender *owns outright* (vendored, upstream changes
driven in-repo). Upstream sibling repo: wilsonzlin/ecma-rs.

## JS VM layout (per docs)

```
vendor/ecma-rs/
├── parse-js/            — JS/TS/JSX parser (lexer + recursive-descent; TS conformance fixtures,
│                          cancellable parsing, TSX/JSX disambiguation, import attributes)
├── hir-js/              — HIR lowering of the AST (NameInterner, fingerprinting)
├── semantic-js/         — semantic analysis
├── vm-js/               — the runtime
│   ├── src/lib.rs       — public API: Vm, VmOptions, Budget, JsRuntime
│   ├── src/vm.rs        — VM struct, execution entry points
│   ├── src/heap.rs      — GC heap
│   ├── src/value.rs     — JS values (Value, GcObject, GcString, …)
│   ├── src/object.rs    — JS objects
│   ├── src/exec.rs      — AST interpreter
│   ├── src/hir_exec.rs  — compiled-HIR executor
│   ├── src/builtins/    — built-in objects
│   ├── src/execution/   — execution contexts, call stack
│   └── tests/           — unit + integration tests
├── test262-semantic/    — test262 harness driving vm-js (executor + harness, async $DONE)
├── emit-js/, minify-js/, typecheck-ts/ — tooling crates (not the VM)
└── webidl-runtime/      — WebIDL DOM bindings adapter on top of vm-js
```

## Execution model

**Not bytecode. Not JIT.** Two interpreters:
- `exec.rs`: **tree-walking AST interpreter**.
- `hir_exec.rs`: **compiled-HIR executor** (executes the hir-js lowered IR directly; effectively a
  second interpreter over a normalized IR, with special handling for async/generator continuations).
- Commit messages (Jan 2026) reference ordering fixes applied in "the AST interpreter,
  async/generator continuations, and the compiled (HIR) executor", confirming both paths are live.
- The workstream doc lists as P2/future: inline caching, hidden classes/shapes, polymorphic ICs,
  and explicitly "(JIT compilation is longer-term)". So the perf story today is interpreter-class;
  documented goal is only "performance is acceptable for interactive use".
- **Async model:** generators and async/await with "async/generator continuations"; Promise job queue
  via host hook; async module evaluation incl. top-level await in scripts/modules/static blocks;
  minimal module execution + filesystem module loader in the test262 harness.

## ES language feature coverage (from the workstream scope + commits)

Broad ES2015–2024 surface is *claimed*: var/let/const with hoisting + TDZ; function declarations,
expressions, arrow functions, generators (incl. async generators, yield*, yield in patterns),
async/await incl. top-level await; classes (declaration/expression, inheritance, super,
getters/setters, private fields/methods, static blocks); destructuring and spread; iterators,
for...of/for-await, iterator protocol; template literals (raw+cooked preservation); optional
chaining, nullish coalescing; modules (import/export syntax, dynamic import, export *
aliases); direct and indirect eval; strict mode (incl. strict-mode new.target/await/yield
validation); Annex B sloppy-mode behaviors; RegExp with `/v` Unicode string properties
(generated emoji/property tables) and inline modifiers; Symbol, BigInt; typed arrays and
ArrayBuffer; Promise combinators (all/race/allSettled/any/withResolvers); Proxy and Reflect;
Map/Set/WeakMap/WeakSet. **Planned/lower-priority:** WeakRef, FinalizationRegistry, realms and
compartments (browser-security sandboxing). Actual conformance target: **95%+ of test262 "core"
tests passing** — an aspiration, not a reported number; no published pass rate found.

## DOM/WebAPI bindings approach

Clean host-embedding seam, which is the good part of the design:
- `VmHostHooks` trait on vm-js: `host_enqueue_promise_job` (Promise job queue) and
  `host_resolve_imported_module` (module loading) are injected by the host.
- `webidl-runtime` (crate `webidl-js-runtime`) is a WebIDL bindings adapter built on vm-js's
  heap/value APIs, with `HeapLimits`-aware construction (`VmJsRuntime::new/with_limits`).
- Script/HTML integration (script elements, module graphs, event loop with
  `EventLoop::run_until_idle`, run/queue budgets, wall-time deadlines) lives in fastrender's
  `docs/js_embedding.md` / `instructions/js_html_integration.md`, `js_dom.md`, `js_web_apis.md`.
- Conformance table self-reports JS as "Partial (event loop, DOM subset, growing Web API surface)".

## Garbage collection

- `heap.rs` implements a **GC heap** with explicit rooting (`Scope`, `RootId` handles) and
  **configurable limits** (`HeapLimits`: allocation caps, pressure-triggered collection, OOM as a
  catchable JS error — never a Rust panic).
- GC *algorithm* details (mark-sweep vs generational vs moving) were not visible in the docs I
  could reach; the handle-based `GcObject`/`GcString` design plus scope rooting is consistent with a
  tracing collector. Treat "what collector" as unconfirmed until source is read.
- **Safety budgets are the strongest part of the engine:** fuel ("tick") budgets, wall-clock
  deadlines (`check_time_every`), interrupts/cancellation, max stack depth, heap limits, and a
  "JS is hostile input" discipline (`timeout -k` enforced in the workstream).

## Approximate LOC and test coverage

- **Not measurable** — clone never completed. Indirect signals: ecma-rs is a workspace of ~10+
  crates (parse-js, hir-js, semantic-js, vm-js, emit-js, minify-js, typecheck-ts, test262-semantic,
  webidl-runtime, diagnostics, effect-model). For scale reference, QuickJS (a comparable
  scope interpreter) is ~60k LOC of C; vm-js plausibly lands in the **tens of thousands of lines of
  Rust** plus generated Unicode/RegExp tables. Verify by `tokei` once the tree exists.
- Tests: vm-js has `src` unit tests + `tests/` integration tests; generator/yield CI smoke tests;
  `test262-semantic` harness executes real test262 subsets (incl. async `$DONE`, async modules);
  parse-js has extensive conformance/regression fixtures (incl. vendored TypeScript baseline tests).
  test262 "done" criterion is 95%+ core — i.e. **not yet done by their own measure**.

## Verdicts for Bowser (std-Rust, minimal deps, WebGPU, extreme perf, 90% real-site loading)

| Component | Verdict | One-line reason |
|---|---|---|
| Execution model (AST interp + HIR executor) | **scrap** | Dual interpreter with no JIT and no inline caching cannot meet an "extreme performance" bar; hidden classes/ICs are only P2 wishes, JIT is "longer-term". |
| GC heap design (`Heap`, `HeapLimits`, `Scope`/`RootId` rooting) | **salvage-parts** | Budget/limit-aware heap with OOM-as-catchable-error is a good pattern, but the collector class/algorithm is unconfirmed and unlikely to be generational/moving. |
| Built-in objects (Array/String/RegExp/Date/JSON/…) | **salvage-parts** | Builtins are the most labor-intensive part of any engine and could be ported, but they are coupled to vm-js's `Value`/`GcObject` handle design, so porting means adaptation, not reuse. |
| parse-js (+ hir-js, semantic-js) | **steal** | Parser/HIR lowering is execution-model-agnostic and the most directly reusable piece; TS/JSX support and conformance fixtures are a bonus — if quality holds up on inspection. |
| DOM bindings approach (`VmHostHooks`, webidl-runtime, event-loop budgets) | **steal** | The host-hook seam (promise job queue, module resolution) plus fuel/deadline/interrupt budgeting is exactly the embedding shape Bowser wants; copy the architecture even if the code isn't lifted. |
| test262 harness (test262-semantic) | **steal** | A working harness that drains microtasks, handles async `$DONE`, and loads modules is genuinely useful infrastructure for aiming at Chromium/Firefox-style test suites. |
| RegExp `/v` Unicode tables + conformance fixtures | **salvage-parts** | Generated Unicode property tables and test fixtures are valuable data even apart from the engine. |

**Bottom line:** fastrender's engine is a spec-correctness-first, safety-budgeted *interpreter* —
interesting as a host-embedding and safety design reference, but its execution model is the opposite
of what Bowser's performance goals need. Steal the parser, the test262 harness, and the
`Budget`/host-hook architecture; do not adopt vm-js as Bowser's engine.

**Open items for parent:** (1) the `~/workspace/bowser/vendor-sources/fastrender` clone never
completed — someone should verify/fix the clone; (2) LOC numbers and GC algorithm need source
confirmation; (3) everything above is docs-based and the project is heavily agent-generated, so
treat claimed feature coverage with extra skepticism on first source contact.
