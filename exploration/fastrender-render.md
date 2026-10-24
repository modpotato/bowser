# fastrender — rendering pipeline survey (project Bowser explorer report)

**Repo:** `github.com/wilsonzlin/fastrender` (experimental Rust browser engine, from Cursor's Jan-2026 "build a browser with AI agents" experiment; ~3M+ lines, ~2,800+ Rust files, thousands of files total)
**Survey date:** 2026-09-25
**Method:** docs-based. The git clone (`vendor-sources/fastrender`, `--depth 1`) was still downloading/indexing a 300MB+ pack after ~20 min, so per instructions the survey used the published docs, README, `docs/conformance.md` (a detailed self-reported support matrix), `docs/architecture.md`, `Cargo.toml`, and the GitHub API (file tree with blob sizes, commit history). No source lines were executed; LOC figures below are measured exactly for 7 flagship files fetched raw, and estimated for module totals at ~33 bytes/line (measured from those files). The GitHub tree API truncated, so module totals are lower bounds.

## Critical context

- **Last commit 2026-01-18 — repo is inactive (~8 months).** Development was an AI-agent sprint in Jan 2026; nothing has landed since.
- **AI-generated scale:** flagship files are 9K–41K lines each (e.g. `src/style/cascade.rs` = 41,374 lines, `src/layout/contexts/inline/mod.rs` = 32,791 lines). Sheer size makes audit/verification of the custom algorithms impractical; assume uneven quality.
- **Spec-first posture:** the docs claim spec-driven (not heuristic) design, with a maintained conformance matrix in `docs/conformance.md`, research notes under `docs/research/` (Servo layout architecture, table layout spec, CSS 2.1 VFM), and a "no pixel-nudging after layout" staged pipeline (parse → style → box → layout → paint).
- **Real-world accuracy signal is modest:** regression workflow (`progress/pages/*.json`) measures pixel similarity vs headless-Chrome baselines; a Jan-2026 commit reports techmeme.com at ~28–31%. So "renders simple websites quickly and largely correctly" (their own words), far from the 90%-of-real-websites bar.
- **Rasterizer is CPU-only (tiny-skia).** wgpu appears only in the UI shell (`src/ui/wgpu_pixmap_texture.rs`) to blit the finished CPU pixmap into an egui texture — not for rendering. This is the single biggest architectural mismatch with Bowser's WebGPU goal.

---

## 1. HTML parsing

- **Module path:** `src/dom.rs` (20,529 lines, measured), plus `src/dom/`, `src/html/` (20 files: `encoding.rs`, `viewport.rs`, `streaming_parser.rs`, `pausable_html5ever.rs`, `dom2_tree_sink.rs`, CSP, meta-refresh, iframe URLs, image attrs…), `src/dom2/` (67 files — a second "live DOM" for JS mutation).
- **Approach:** delegates the hard part to **html5ever 0.35 in spec mode** (Servo's HTML5 tree builder) with `TreeBuilderOpts::scripting_enabled` toggled for `<noscript>` semantics; encoding sniffing per HTML Living Standard (BOM → Content-Type → `<meta charset>`, Windows-1252 fallback) in `src/html/encoding.rs`; static `<template shadowroot>` attachment + slot distribution at parse time; streaming/pausable parser for the script-executing runtime.
- **Spec-compliance signals:** strong at the tokenizer/tree-builder level (inherited from html5ever, the same parser Servo uses). Their own additions (encoding, base URL, viewport meta, details/summary inertness) are documented against the spec in `docs/conformance.md`. Two competing DOMs (`DomNode` vs `dom2::Document` + snapshot) is a smell.
- **Approx LOC:** ~65K+ (dom.rs 20.5K + html/ ~12K + dom2/ ~33K, lower bound).
- **External deps:** `html5ever 0.35`, `markup5ever`, `markup5ever_rcdom`, `encoding_rs`.
- **Test coverage:** module tests in `src/dom.rs`/`src/html/*`; `tests/dom_integration/`, `tests/wpt_dom/`; accuracy-fixture pageset vs Chrome.
- **Verdict: STEAL (the dependency, not the code).** html5ever is the Servo-vetted, spec-compliant HTML5 parser; writing a tokenizer/tree-builder from scratch is exactly the kind of work that will sink the 90%-compat goal. Bowser should depend on html5ever directly and skip fastrender's 20K-line wrapper.

## 2. CSS parsing & cascade

- **Module path:** parsing in `src/css/` (`parser.rs` 9,376 lines measured, `selectors.rs` 2,923, `types.rs`, `loader.rs`, `properties.rs`, `supports.rs`); cascade/style in `src/style/` (267 files: `cascade.rs` **41,374 lines measured**, `properties.rs`, `values.rs`, `color.rs`, `custom_properties.rs`, `media.rs`, UA stylesheet in `src/user_agent.css`).
- **Approach:** CSS syntax via Servo's **`cssparser 0.35`** + **`selectors 0.32`** crates; a from-scratch cascade engine claiming CSS Cascade 4 (origins, `!important`, `@layer`, `@scope`, custom properties with fallback, `@import`, `@supports`, MQ4 media queries, container queries L1 + a style-query subset); selectors claimed at Selectors L4 (`:is/:where/:has`, nth-*, `:lang/:dir`, `::marker`/`::backdrop`, form pseudos). Strict value parsing; color via `csscolorparser` + own Lab/LCH/OKLab/OKLCH support.
- **Spec-compliance signals:** best-in-survey on paper — `@layer`, `@scope`, container queries, Selectors 4, cascade layers are all claimed with module tests plus WPT suites at `tests/wpt/tests/style/conditional/` and `.../container_queries/`. But the 41K-line cascade.rs is unauditable; breadth of claims + AI generation + stalled repo = treat as "broad but unverified."
- **Approx LOC:** ~185K+ (css/ ~33K + style/ ~150K, lower bound).
- **External deps:** `cssparser`, `selectors`, `csscolorparser`, `smallvec`, `im`.
- **Test coverage:** module tests in parser/cascade; `tests/wpt/` harness with `manifest.toml` + `expected/` (partial WPT import, style-focused); value-level tests (e.g. "invalid CSS values after var() resolution").
- **Verdict: SALVAGE-PARTS.** Steal the *dependency choices* (`cssparser` + `selectors` — same crates Servo uses, pure Rust, near-mandatory for compat); scrap the 41K-line hand-rolled cascade engine (too big to verify, dead upstream). Property/value *tables* and the UA stylesheet may be worth mining.

## 3. Layout

- **Module path:** `src/layout/` (506 files; flagships: `contexts/inline/mod.rs` 32,791 lines, `contexts/block/mod.rs` 18,607 lines measured; `table.rs`, `fragmentation.rs`, `pagination.rs`, `absolute_positioning.rs`, `float_context.rs`, `taffy_integration.rs`, `contexts/flex.rs`, `contexts/grid.rs`), box-tree generation in `src/tree/` (10 files: `box_generation.rs`, `box_tree.rs`, `fragment_tree.rs`, `table_fixup.rs`, `anonymous.rs`).
- **Approach:** two-tree architecture (box tree → fragment tree, per their Servo research notes), enum-dispatched formatting contexts: custom BFC/IFC with floats, margin collapsing, baseline alignment, ruby; **native table layout** (auto + fixed, both border models — explicitly no flex/grid shortcuts); multicol; fragmentation/pagination; absolute/fixed/sticky positioning with transform-aware containing blocks; **flexbox and grid delegated to vendored `taffy`** (`taffy = { path = "vendor/taffy" }`).
- **Spec-compliance signals:** strongest structural story — box/fragment split, native table algorithm with a dedicated research doc, WPT-backed style tests; flex/grid ride on taffy (Servo's layout lib). Inline layout at 32K lines in one file is the classic AI-bloat warning sign, though.
- **Approx LOC:** ~345K+ (layout/ ~312K + tree/ ~33K, lower bound) — the largest module by far.
- **External deps:** `taffy` (vendored fork).
- **Test coverage:** module tests in block/inline/table files; `tests/layout/` (19 files: float, grid, fragmentation…); `tests/render/`, `tests/regression/`, `tests/ref/`; pageset accuracy fixtures.
- **Verdict: SALVAGE-PARTS.** Steal the *architecture* (box tree → fragment tree, formatting-context dispatch) and the *taffy* decision for flex/grid — taffy is dependency-light, Servo-used, and the only sane way to get flex/grid without writing them. Scrap the 300K+ lines of custom block/inline/table code: unauditable, dead upstream, and inline layout is where browsers hide their worst bugs.

## 4. Paint

- **Module path:** `src/paint/` (402 files: `display_list.rs`, `display_list_builder.rs`, `display_list_renderer.rs`, `stacking.rs`, `painter.rs`, `rasterize.rs`, `gradient.rs`, `clip_path.rs`, `css_filter.rs`, `svg_filter.rs`, `text_rasterize.rs`, `text_shadow.rs`, `transform3d.rs`, `scene_tree.rs`, `paint_thread_pool.rs`, …).
- **Approach:** display list + stacking-context painter (stacking, isolation, blend modes, top layer), then **CPU rasterization via `tiny-skia 0.11`** into a `Pixmap`; SVG via `resvg`; 3D transforms "flattened for tiny-skia"; multithreaded paint pool (`paint_thread_pool.rs`, rayon). This is a software renderer with a GPU texture blit at the very end for the desktop shell.
- **Spec-compliance signals:** backgrounds/borders 3, masking, filters/backdrop-filter, transforms all claimed ✅ in the matrix with module + paint regression tests (`src/paint/tests/`, golden tests). But 3D is flattened and the whole backend assumes a CPU pixmap.
- **Approx LOC:** ~215K+ (lower bound).
- **External deps:** `tiny-skia 0.11`, `resvg`, `svgtypes`, `roxmltree`, `image`, `png`, `rayon`.
- **Test coverage:** module tests in display-list/stacking/renderer files; `src/paint/tests/` regression tests; `tests/paint/`; pageset pixel-accuracy fixtures.
- **Verdict: SCRAP.** The display-list/stacking *design* is textbook-correct, but the entire backend is built around a CPU rasterizer — fundamentally mismatched to Bowser's WebGPU goal, and porting it would mean rewriting every draw op against wgpu anyway. Keep `tiny-skia` in mind only as a CPU fallback/reference rasterizer, not as architecture.

## 5. Text shaping

- **Module path:** `src/text/` (98 files; flagship `pipeline.rs` 12,607 lines measured; `line_break.rs`, `hyphenation.rs`, `justify.rs`, `bidi_controls.rs`, `segmentation.rs`, `font_fallback.rs`, `font_loader.rs`, `font_resolver.rs`, `face_cache.rs`, `emoji*.rs`, `color_fonts/` with COLR v0/v1, CBDT/CBLC, sbix, SVG-in-OT + golden tests, `otvar/`).
- **Approach:** shaping via **`rustybuzz 0.20`** (pure-Rust HarfBuzz port) with OpenType features/variations; bidi per UAX #9 (`unicode-bidi`); line breaking (`unicode-linebreak` + `icu_segmenter` with compiled data); script itemization; font management via `fontdb` + `ttf-parser`/`skrifa`/`read-fonts`; color-emoji pipeline (bitmap, SVG, COLR v0 and v1 paint graphs).
- **Spec-compliance signals:** very credible dependency stack — rustybuzz *is* HarfBuzz shaping semantics, icu_segmenter is ICU line breaking. The pipeline orchestration (12.6K lines) is the questionable part, not the shaping itself.
- **Approx LOC:** ~51K+ (lower bound).
- **External deps:** `rustybuzz`, `fontdb`, `ttf-parser`, `skrifa`, `read-fonts`, `unicode-bidi`, `unicode-linebreak`, `unicode-segmentation`, `unicode-width`, `icu_segmenter`.
- **Test coverage:** module tests in pipeline/line_break/hyphenation/justify; golden tests for color fonts (`src/text/color_fonts/golden_tests.rs`); `tests/fonts/`.
- **Verdict: STEAL (the stack).** rustybuzz + fontdb + icu_segmenter + unicode-bidi is the correct, pure-Rust, system-dependency-free text stack — it is what Bowser should use verbatim. Salvage pipeline *concepts* (itemize → shape → break runs) but not the 12K-line implementation.

---

## Cross-cutting assessment for Bowser

1. **The "std-lib Rust first, minimal dependencies" goal collides with the "90% of real websites + Chromium/Firefox test suites" goal.** fastrender's experience is instructive: even with html5ever + cssparser + selectors + taffy + rustybuzz doing the heavy lifting, its measured real-site accuracy is ~30% on fixtures. The Servo-ecosystem crates (`html5ever`, `cssparser`, `selectors`, `taffy`, `rustybuzz`, `fontdb`, `icu_*`, `unicode-*`) are all pure Rust with minimal transitive deps — they are the pragmatic minimum, not bloat. Budget the dependency allowance here; minimal-deps purity anywhere else.
2. **Nothing in fastrender's paint stage transfers to a WebGPU renderer.** If Bowser wants WebGPU, the display list must be designed GPU-first from day one (record passes, batch by pipeline state, atlas glyphs). tiny-skia is at most a CPU reference.
3. **Do not import fastrender code wholesale.** The value is (a) its *dependency list*, (b) its *pipeline architecture* (parse → cascade → box tree → fragment tree → display list, staged with no paint-time layout overrides), and (c) its *docs* (`docs/conformance.md` as a spec checklist, `docs/research/*` as design references). The ~860K+ lines of custom pipeline code are unaudited AI output from a dead repo.
4. **Test strategy worth copying:** WPT harness with expected-manifest (`tests/wpt/`), Chrome-baseline pixel-accuracy fixtures (`progress/pages/`), module tests colocated in `src/`, and paint regression tests under `src/paint/tests/`.
5. **License check needed:** no license file was verified in this survey (README claims experimental/research); confirm before reusing any code or vendored taffy fork.
