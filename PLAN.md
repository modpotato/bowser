# Bowser — Build Plan (exploration synthesis, 2026-09-25; status updated 2026-09-29)

## Status: waves A–H complete; post-wave-H review fixes committed (6947f46)

Post-wave-H harsh re-review found and fixed: private-field side table (was
string-keyed `#x` properties, bypassable) with VM-global brand allocation;
eval scoping (direct eval nested-function locals, indirect sloppy global
instantiation, copy-out rooting, var/lexical conflicts, prologue clobber);
materialize_dense temp-root leak; `+=` fusion rhs-rebind; yield* delegated
resume stale send; hostile RepeatImage tile hardening; fractional
isolated-group GPU/CPU alignment. All gates re-verified green before commit.

Work proceeded in eight waves rather than the phase vocabulary in the old
exploration notes. All 11 crates exist; check, test, and clippy are green as
of wave H.

## Architecture: monorepo, 11 crates

| Crate | Subsystem | Landed in |
|---|---|---|
| `bowser_url` | WHATWG URL parser, IDNA, percent-encoding | wave A — foundations |
| `bowser_encoding` | WHATWG Encoding Standard: label table, BOM sniffing, legacy decoders, meta prescan hooks | wave A — foundations |
| `bowser_net` | hand-rolled TCP / TLS 1.2+1.3 / HTTP/1.1 / HTTP/2, fetch, cookies, compression; vendored wreq-util fingerprint profiles | wave A (TLS client), wave B (fetch) — networking |
| `bowser_html` | WHATWG tokenizer + tree construction, full error recovery | wave A — foundations |
| `bowser_dom` | DOM core, events, querySelector, geometry | wave B — static document |
| `bowser_css` | parsing, cascade, values, custom properties | wave B — static document |
| `bowser_layout` | block, inline, flexbox, flex-wrap, table, grid, positioned layout | wave B — static document; table/grid/abspos/flex-wrap landed in later waves |
| `bowser_paint` | display lists, WebGPU renderer (wgpu) + tiny-skia 0.11 CPU fallback backend | wave C — programmable document; CPU backend hardened in wave F |
| `bowser_js` | from-scratch bytecode-VM JS runtime, browser-specific, own GC, bindings, module execution | wave C/D — programmable document |
| `bowser_wpt` | web-platform-tests harness (scoreboard; pluggable driver) | wave A — foundations; real `BowserDriver` landed at `crates/bowser_app/src/wpt_driver.rs` (feature-gated `wpt-driver`) |
| `bowser_app` | browser binary, tabs, navigation UI, script plumbing | wave C/D/E |

The old phase mapping was: phases 0–1 (URL/encodings/TLS/HTTP/harness), phase 2
(HTML/DOM/CSS/layout/paint), phase 3 (JS runtime), phase 5 (app chrome), with a
phase 4 (workers, WebSockets, shadow DOM, canvas, media, IndexedDB) still
unbuilt. WPT + test262 remains the scoreboard.

## fastrender verdict: reference only, NO code copying

**License check 2026-09-25: wilsonzlin/fastrender has no LICENSE file and no
license metadata. No license = all rights reserved = we cannot copy its code.**
We may study its architecture and public design docs, but every line of Bowser
is written from scratch. (This also settles the "steal" question: the answer is
reference-only.)

- **Reference** (read, don't lift): CSS cascade L4 design, table auto-layout,
  bidi/shaping pipeline, golden-diff pixel test methodology, `VmHostHooks` +
  fuel/deadline/interrupt budget architecture, test262 harness design.
- **Do not adopt**: execution model (dual tree-walking interpreter, no JIT),
  mega-crate structure, software-only paint pipeline (tiny-skia; Bowser is
  WebGPU from day one).
- fastrender is inactive since Jan 2026 and community-verified not to compile.

## wreq verdict: vendor the data, hand-roll the stack

- 137 fingerprint profiles live in `wreq-util/src/emulate/` (~165 KiB Rust source).
  Transcribe constants into a compact std-only table (<100 KiB). Apache-2.0,
  keep attribution.
- Applying profiles needs ClientHello byte-level control (extension order, GREASE,
  permutation, ALPS, ML-KEM-768). rustls explicitly refuses this → **bowser_net
  hand-rolls TLS**. This is the single biggest security-sensitive work item;
  it gets dedicated reviewers.

## Repo split

Monorepo. No separate JS-runtime repo: nothing yet justifies the split, and the
brief says keep building in the monorepo meanwhile. Revisit if `bowser_js`
outgrows monorepo ergonomics.

## Dependency decisions (orchestrator-approved, feature-gated minimal)

- `wgpu` 30 — mandated by the WebGPU requirement.
- `tiny-skia` 0.11 (CPU fallback backend; `default-features = false`,
  `std`+`simd` only) — orchestrator-approved rationale: software
  rasterization is genuinely std-undoable, and it is needed for the
  headless/WPT render paths where no Vulkan device exists.
- Text/fonts/images (not std-doable): `swash` (+ `skrifa`/`read-fonts` via
  its own tree), `png`, `zune-jpeg`, `winit`, `pollster`.
- Rejected/never adopted (removed from the approved list): `vello`,
  `kurbo`, `peniko`, `parley`, `fontique`, `etagere`.
- Everything else: std only. New deps need orchestrator sign-off per AGENTS.md.
- Deferred: brotli (gzip/deflate first), HTTP/3/QUIC, service workers, WebGL,
  CSS Regions (flow-into/flow-from — not pursued, wave-31 decision; no stub
  exists, `break-inside: avoid-region` parsing aliases remain only as aliases).

## JS engine direction

From-scratch bytecode VM, browser-specific optimizations welcome (per explicit
user direction). Boa (pure Rust, ~94.6% test262) is the documented fallback if
the from-scratch engine stalls. fastrender's vm-js is not adopted (interpreter).

## Cutoff lines (keep the 90% bar)

Skip: quirks mode (always standards mode), XHTML-as-HTML, plugins/Flash/Java/
ActiveX/VBScript/XSLT, NTLM/Kerberos, pre-CSS relics as engines (parse-only),
HTTP/3 initially, P2 web APIs (WebRTC/WebTransport/service workers/WebGL).

## Worker allocation (unbounded, throughput-first)

Work ran as eight waves, each followed by review (whole-codebase +
per-section, harsh; one worker per issue found). Worker counts scaled to
need — the model is unbounded throughput, never token-frugal.

- Wave A — foundations: URL parser, encodings, TLS client, HTML parser (all reviewed and fixed)
- Wave B — static document: fetch, DOM, CSS, layout (all reviewed and fixed)
- Wave C — programmable document: app integration, JS runtime, paint pipeline; clippy gate green
- Wave D — wiring: JS bindings wired, module execution, paint/app integration
- Wave E — perf/hardening: JS VM dispatch, paint GPU/CPU paths, GC, dom/html single-pass, CSS cascade memoization, net TLS resumption/pool, app script plumbing; check/test/clippy green
- Wave F (`16d6c85`) — conformance + security hardening: HKDF panic surface, secure-cookie overwrite, schemeful SameSite, redirect anchoring, persistent default HSTS, TLS validation (progress/HRR/trailing-message/CCS), A-label/IDN hostname verification; CSS custom-property keywords, `revert`/`revert-layer`; layout BFC/flex fixes, shared font cache; JS builtins (Object/Array/JSON/promise combinators), ArrayBuffer rework, GC rooting/Handle fixes; app page web APIs installable per realm; WPT driver hardening (watchdog-bounded lookup, wedged-driver retirement, multi-reference reftests)
- Wave G (`9312ea0`) — harsh-review fix cascade across net/layout/css/paint/url/js: DNS cache generations + happy-eyeballs cap, gzip coding-depth caps, grid/flex/table clamping, abspos constraint equations, image-cache 64MiB bound, JS strictness/TDZ/native-stack-guard fixes, test262 harness hardening
- Wave H (2026-09-29) — conformance sweep + new features: HTTP/2 with std-only HPACK (h2 ALPN, fallback to H1, single-use sessions); HTML corpus deviations 16→4 (select in-scope boundary, PI/doctype fixes; remaining 4 need script execution); layout fragmentation (break-before/after/inside, widows/orphans, page slicing, fixed-box repetition); grid dense flow + auto-fill/auto-fit (+`grid-auto-flow` CSS registration); collapsed-border paint integration; sticky positioning (`layout_with_scroll`); CSS var() fallback edge cases; DOM API expansion (matches/closest/innerHTML/outerHTML/getElementsByClassName, live HtmlCollection); paint rounded-rect clips (CPU/GPU bit-exact coverage), `mix-blend-mode` ×6, `RepeatImage` display item, fractional rect clips; JS eval(), deferred string concat (18×), object inline cache (6.5×), dense arrays, regexp UTF-16 re-encode fix (79×, unhung the full suite); test262 scoreboard 4,326/12,844 (33.7%)

## Risks

1. Hand-rolled TLS: security minefield. Mitigation: dedicated reviewers, fuzzing,
   differential testing against rustls/BoringSSL test vectors.
2. JS engine is the long pole. Mitigation: largest worker allocation, test262 as
   the gate, Boa fallback documented.
3. wgpu quarterly breaking releases. Mitigation: pin version, budget upgrades.
4. No cargo on build machines initially → solved: rustup stable installed,
   workers use `~/.cargo/bin`.
