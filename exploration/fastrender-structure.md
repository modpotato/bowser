# fastrender structural survey (project Bowser explorer report)

**Date:** 2026-09-25
**Source:** `~/workspace/bowser/vendor-sources/fastrender` (clone was still downloading after ~40 min; survey reconstructed from the repo's own published docs: README.md at commit `19bf103`, `docs/testing.md`, `docs/build_performance.md`, `docs/ecma_rs_ownership.md` on github.com/wilsonzlin/fastrender). All paths below are as documented upstream, not from a local checkout. LOC figures for the five largest modules come from the repo's own `docs/build_performance.md`; total-size claims come from the project's public description.

## Snapshot

- **Repo:** wilsonzlin/fastrender — "Experimental new browser engine", Rust, created 2025-11-16.
- **Scale:** ~29,858 commits, ~1,569 stars. Built in one week by ~2,000 concurrent AI coding agents (Cursor research project); public description claims **3M+ lines of Rust across thousands of files**.
- **Default branch:** `main`; latest indexed commit `19bf1036105d4eeb8bf3330678b7cb11c1490bdc`.
- **Build:** Rust 1.70+, Git submodules, CMake. `cargo run --release --features browser_ui --bin browser`.

## Top-level layout

```
fastrender/
├── src/                 # the mega-crate (nearly all engine code lives here)
├── crates/              # cargo workspace members (FastRender tooling only)
├── vendor/              # vendored deps: ecma-rs/ (JS engine, git submodule), taffy/ (layout fork)
├── tests/               # integration tests (two test binaries; see below)
├── benches/             # criterion benches
├── docs/                # internal documentation (large)
├── instructions/        # per-workstream agent guides
├── specs/               # spec notes
├── progress/pages/      # pageset accuracy scoreboards (*.json)
├── investigation/       # ad-hoc investigation notes
├── scripts/             # cargo_agent.sh, run_limited.sh, cargo_llvm.sh, ci checks, chrome-compare wrappers
├── xtask/               # cargo xtask automation (golden updates, chrome baselines, fixture diffs)
├── tools/               # misc tooling
├── assets/              # chrome CSS/SVG icons, etc.
├── fuzz/                # fuzzing targets
├── .cargo/ .github/     # cargo config, CI workflows
├── Cargo.toml / Cargo.lock
├── AGENTS.md, README.md, vitest.config.cjs
```

## Cargo workspace members (`crates/`)

Docs name two members; policy (`docs/ecma_rs_ownership.md`) says `crates/` is reserved for FastRender-specific tooling, never JS/WebIDL infra (that goes in `vendor/ecma-rs/`):

- `crates/fastrender-ipc/` — IPC message protocols (navigation, resource loading, frame transport; shared-memory frame buffers)
- `crates/js-wpt-dom-runner/` — JS WPT DOM test runner (`xtask js wpt-dom`)

**Note:** the main engine itself is a single mega-crate in `src/` (see "mega-crate" note under Assessment).

## `src/` module tree (with LOC where documented)

Figures from `docs/build_performance.md` "Dependency Analysis Results":

| Module | LOC (repo's own docs) | Contents |
|---|---|---|
| `layout/` | **~282K** | `contexts/block/`, `contexts/inline/` (native BFC/IFC), `contexts/flex.rs`, `contexts/grid.rs` (delegate to vendored taffy), `table.rs` (native auto+fixed, bidi constraint solving), `fragmentation.rs`, `absolute_positioning.rs`, multicol |
| `js/` | **~213K** | `event_loop.rs` (tasks/microtasks/rAF), `html_script_scheduler.rs`, `html_script_pipeline.rs`, `vmjs/` bindings, `webidl/` (FastRender DOM glue), `import_maps/`, `quickjs_url.rs`, `fetch.rs`, `url.rs`, `time.rs`, `legacy/` |
| `paint/` | **~208K** | `display_list.rs`, `display_list_renderer.rs`, `stacking.rs`, `painter.rs`, `rasterize.rs`, `blur.rs`, `svg_filter.rs`, `tests/` (paint regression goldens) |
| `style/` | **~157K** | `cascade.rs` (Cascade L4: layers, scope, origins), `properties.rs`, `values.rs` (calc(), colors), `var_resolution.rs`, media/container queries |
| `text/` | **~54K** | `pipeline.rs` (itemize/shape/layout), `bidi.rs` (UAX #9), `line_break.rs` (UAX #14, hyphenation), `color_fonts/` (COLR/sbix/SVG-in-OT) |
| `tree/` | n/a | `box_tree.rs`, `box_generation.rs`, `table_fixup.rs` (anonymous wrappers) |
| `css/` | n/a | CSS parsing, selectors (Selectors 4 claimed complete), values |
| `dom.rs` + `dom2/` | n/a | HTML5 parsing, DOM tree, shadow DOM; `dom2/` = live DOM for JS mutation |
| `ui/` | n/a | `browser_app.rs`, `browser_tab_controller.rs`, `browser_thread.rs`, `browser_worker.rs`, `tab_engine.rs`, `ui_worker.rs`, `worker.rs`, `worker_loop.rs`, `render_worker.rs`, `messages.rs` (UI↔worker protocol) |
| `resource/` | n/a | `disk_cache.rs`, `web_fetch/`, `data_url.rs` (fetch stack) |
| `sandbox/` | n/a | `linux_seccomp.rs`, `macos_seatbelt.rs`, `windows.rs` (AppContainer) — tested helpers; sandbox is target-arch, not yet enforced |
| `ipc/` | n/a | IPC message protocols (mirrors `crates/fastrender-ipc`) |
| `interaction/` | n/a | `engine.rs` (caret/selection, hit testing) |
| `web/` | n/a | browser Web APIs (timers, fetch, URL, crypto, blob, form-data, text-encoding) |
| `api.rs` | n/a | public API: `FastRender`, `FastRenderConfig`, `RenderOptions`, `BrowserTab` |
| `error`, `geometry` | n/a | shared error type; geometry primitives |
| `bin/` | n/a | 10 CLI binaries: `browser`, `fetch_and_render`, `fetch_pages`, `prefetch_assets`, `pageset_progress`, `render_fixtures`, `bundle_page`, `inspect_frag`, `dump_a11y`, `diff_renders` |

Dependency direction (import counts, same doc): `layout → style (1525), tree (482), geometry (129), text (69)`; `paint → style (432), text (196), tree (108), geometry (127)`; `style → css (304), dom (192)`; `js → dom (29), error (21), resource (19), web (12)`. **Known cycles:** layout↔tree (482 vs 3), paint↔text (196 vs 20).

## Dependency list (full `Cargo.toml` not retrievable; reconstructed from repo docs + fork README)

**Confirmed / strongly evidenced:**
- `scraper` — HTML parsing (wraps **html5ever**; html5ever is a transitive dep, not direct)
- `cssparser` — CSS tokenization/parsing
- `taffy` — **vendored fork** at `vendor/taffy/` (patched grid auto-placement); flexbox + grid layout delegate to it
- `tiny-skia` — 2D software rasterization (README: "software rasterizer"; 3D transforms flattened to 2D)
- `rustybuzz` — OpenType text shaping (GSUB/GPOS)
- `resvg` / `usvg` — SVG rendering (SVG-in-OT glyphs, filters)
- `fontdb` — font database/loading
- `image` — PNG encode/decode
- `vm-js`, `webidl`, `webidl-vm-js`, `webidl-runtime` — **path deps** into `vendor/ecma-rs/` (the JS engine, a git submodule with its own workspace: `parse-js`, `vm-js`, `native-js`, `optimize-js`, `runtime-native`, `knowledge-base`, `effect-js`, CLIs)
- `rquickjs` — **dev-dependency** (QuickJS URL/URLSearchParams bindings for offline tests)
- `criterion` — benches (`benches/paint_benches.rs`, `benches/chrome_frame_render.rs`, `harness = false`)

**Flags for Bowser relevance:**
- **html5ever: present but transitive** via `scraper`. NOT a direct dep.
- **servo-origin crates (stylo/selectors/etc.): none evidenced.** CSS cascade/selectors are hand-rolled (`src/css/`, `src/style/`); the repo claims Selectors 4 / Cascade 4 "complete" natively.
- **winit / wgpu: not confirmed.** Desktop shell builds (`browser_ui` feature, needs wayland/xkbcommon/vulkan/egl dev libs per README), but the rendering path documented is software (`tiny-skia` pixmaps); GPU process/compositor is explicitly "target architecture", not current. Treat as unknown — needs the actual `Cargo.toml`.
- Heavy bespoke subsystems instead of crates: bidi/line-break (own UAX #9/#14), cascade L4, table layout, multicol, fragmentation, seccomp/Seatbelt/AppContainer sandboxing.

## Test directory layout

Enforced 2-integration-binary architecture (`docs/testing.md`, `docs/test_architecture.md`; CI script bans new top-level `tests/*.rs` and `#[path]` shims):

- `tests/integration.rs` — the single integration-test binary; pulls modules via `mod` (e.g. `tests/regression/` incl. `pages.rs` pageset suite, `js_url_bindings.rs`; `tests/browser_integration/` incl. `ui_worker_tabs.rs`, `ui_worker_fragment_navigation.rs`, `ui_stage_heartbeat_forwarding.rs`, `media_playback_rate.rs`; `tests/fixtures/` runner)
- `tests/allocation_failure.rs` — second binary, custom `#[global_allocator]`
- Unit tests live in `src/` (`#[cfg(test)]`); paint regressions in `src/paint/tests/`
- `tests/wpt/` — Web Platform Tests subset; `tests/pages/fixtures/` — offline self-contained page captures with subresources inlined (incl. `media_playback/assets/`); `tests/pages/golden/` — committed golden PNGs; `tests/pages/overrides.toml`, `pageset_guardrails.json`, `pageset_timeouts.json` — accuracy budgets
- `tests/fixtures/html/*.html` + `tests/fixtures/golden/` — pixel golden fixture suite (auto-discovered; `FIXTURES_*` env knobs); `tests/fixtures/fonts/`, `tests/fixtures/media/` — deterministic font/media fixtures
- Chrome-vs-FastRender pixel-diff workflow via `xtask` (`fixture-chrome-diff`, `chrome-baseline-fixtures`); `progress/pages/*.json` — accuracy telemetry

## Docs / instructions layout

- `docs/`: `index.md`, `conformance.md` (spec matrix), `api.md`, `js_embedding.md`, `testing.md`, `test_architecture.md`, `build_performance.md`, `ecma_rs_ownership.md` (repo-shape policy), `runtime_stacks.md`, `live_rendering_loop.md`, `renderer_chrome_schemes.md`, `resource-limits.md`, ...
- `instructions/`: workstream guides for agent swarms — `js_web_apis.md`, `browser_chrome.md`, `browser_responsiveness.md`, `live_rendering.md`, `multiprocess_security.md`, `renderer_chrome.md`, `video_support.md` (browser_ux.md removed)
- `specs/`, `investigation/`, `progress/pages/` — spec notes, ad-hoc investigations, accuracy tracking

## Structural verdict

**Organized at the top, mega-crate underneath.** Module boundaries follow the classic browser pipeline (parse → style → box tree → layout → paint → text) and are documented unusually well for the size (README pipeline diagrams, per-module docs). The `vendor/` split is principled: JS engine (`ecma-rs` submodule) and layout lib (`taffy`) are vendored forks, `crates/` holds only FastRender tooling (`fastrender-ipc`, `js-wpt-dom-runner`), tests have a strict enforced architecture.

**Spaghetti risks (structural, not line-level):**
1. **Single mega-crate** `src/` — the repo's own docs call crate-splitting a bottleneck blocked by **circular deps** (layout↔tree, paint↔text). For Bowser, this means engine stages are not independently reusable crates.
2. **Import fan-in** — `layout` imports `style` 1525 times; `paint` imports `style` 432 / `text` 196 times. Style is a god-module hub.
3. **Duplicated live/static DOM** (`dom.rs` vs `dom2/`), duplicated IPC (`src/ipc/` + `crates/fastrender-ipc/`), two JS runtimes in play (`vm-js` production + `rquickjs` dev-only + `native-js` LLVM experiment) — signs of agent-swarm parallel evolution.
4. **Dependency posture vs Bowser goals:** std-lib Rust first is NOT how fastrender works — it leans on `scraper/cssparser/tiny-skia/rustybuzz/resvg/fontdb`, vendors its layout lib, and submodules its JS engine. Good news: **no servo-origin deps, no wgpu/winit confirmed** — its dependency footprint is narrower than Servo's, and its value for Bowser is mostly as a **reference implementation** (cascade L4, table auto-layout, bidi/shaping pipeline, test/golden infrastructure) rather than a dependency source. The `tests/pages/fixtures` + golden-diff methodology and `docs/build_performance.md`'s module-coupling analysis are the most directly reusable artifacts.

**Caveats:** no local checkout was available (clone still downloading ~300MB pack after 40+ min); LOC for tree/css/dom/ui/resource/sandbox/ipc/interaction/web not quantified (only the five modules above have published figures); full `Cargo.toml` dep list needs the finished clone — winit/wgpu and exact versions remain unconfirmed.
