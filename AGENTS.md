# AGENTS.md — Bowser

Read this before writing any code. It overrides general habits.

## The non-negotiables

1. **Standard-library Rust first.** `std` + `core` + `alloc` cover nearly
   everything. A new entry in `[dependencies]` is a design failure until
   proven otherwise. If you believe a dependency is unavoidable (platform
   GPU, windowing, font rasterization), you must: (a) prove std cannot do
   it, (b) get orchestrator approval, (c) pick the smallest pure-Rust
   option, feature-gated to the absolute minimum. `cargo tree` must stay
   tiny. Vendoring *data* is preferred over depending on crates; depending
   on a crate to avoid writing a security-critical primitive (TLS) needs
   explicit sign-off.
2. **Performance is the product.** Smaller and faster than every modern
   browser. Zero-copy where possible, arena/bump allocation for DOM and
   layout trees, no allocation in hot paths, cache-friendly data layouts.
   Profile before optimizing; add `cargo bench` harnesses for hot paths.
3. **Memory cleanliness.** No leaks, no unbounded growth. The only garbage
   collector in this codebase is the one inside `bowser_js`, and it is an
   explicitly designed component, not a crutch. Everything else is
   ownership plus arenas with clear lifetimes.
4. **`unsafe` is a last resort.** Every `unsafe` block carries a
   `// SAFETY:` comment stating the exact invariant being upheld, and it
   gets a second reviewer before merge.
5. **Correctness is measured, not asserted.** Every subsystem ships with
   unit tests and drives toward web-platform-tests (and test262 for the JS
   engine). A feature without a test does not exist.

## Build rules

- Edition 2024. `cargo fmt --check` clean, `cargo clippy -- -D warnings` clean.
- Release profile: `lto = true`, `codegen-units = 1`, `opt-level = 3`,
  `strip = true`.
- One crate per subsystem under `crates/`. No dependency cycles between
  crates. Dependency order: `url` -> `encoding` -> `net` -> `html` -> `dom`
  -> `css` -> `layout` -> `paint` -> `js` -> `app` (`bowser_wpt` is a
  standalone harness with no internal deps; `bowser_app` depends on it).
- Public APIs are minimal and documented. Prefer concrete internals over
  clever abstraction.

## Workflow

- All work happens locally on master. Completed work syncs directly to the
  `develop` branch on GitHub; never push to master/main, never open a PR
  unless the maintainer explicitly approves one.
- Never commit broken code. If it does not compile and pass `cargo test`
  for its crate, it does not land.
- Commit messages: imperative and scoped, e.g.
  `html: handle foster parenting in table contexts`. Multi-crate wave
  commits use `wave X: <summary>` (e.g. `wave E: perf/hardening pass`).
- Test fixtures that are large or binary live under `tests/fixtures/`,
  never inline in source.
