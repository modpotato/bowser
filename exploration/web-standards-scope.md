# Project Bowser — Web Standards Scope (Exploration)

*Research-only scoping of what "a real browser" must implement in 2026. Written 2026-09-25. No code, no changes.*

## 1. The target: what "90% of real websites" means

Loading 90% of real websites is mostly about the **long tail of legacy-plus-modern patterns**, not the newest specs. Reference points:

- **Servo** (the active Rust browser engine, Linux Foundation, ~36k stars) went from **48.2% → 61.6% WPT pass rate in 2025** (~1.5M passing subtests) and still "renders many real websites, but complex sites still break" — with float/table/flexbox CSS, SpiderMonkey JS, and HTTP/TLS stack in place. Bowser's 90%-of-sites bar is roughly Servo-today × the last mile of DOM/CSS/JS edge cases, workers, storage, and media.
- **Interop 2026** (Apple, Google, Mozilla, Microsoft, Igalia, Bocoup) measures interop via WPT across 20 focus areas. The list is a good "what developers actually hit" signal: anchor positioning, advanced `attr()`, container style queries, `contrast-color()`, CSS zoom, custom highlights, dialog/popover, fetch uploads+ranges, IndexedDB `getAllRecords()`, JSPI/Wasm, media pseudo-classes, Navigation API, scoped custom element registries, scroll-driven animations, scroll snap, `shape()`, view transitions, **web compat**, WebRTC, WebTransport. Most of these are post-MVP for Bowser; **web compat** (quirks real sites depend on) is the one that matters early.
- Site-breakage reality: the web compat focus exists because big sites depend on unspecified edge behavior (e.g. `document.write`, sync XHR during parse, UA-string sniffing, cookie quirks, form encoding edge cases). Aiming at Chromium/Firefox WPT scores is right, but the **90% sites bar is won in HTML parser error-recovery, DOM, CSS cascade, and fetch/cookie correctness**, not in anchor positioning.

## 2. Web-platform-tests: what it is and how big it is

- WPT (github.com/web-platform-tests/wpt) is the W3C-coordinated cross-browser suite; **Interop scores are WPT scores**.
- Scale: roughly **~2 million individual subtests grouped into ~60,000 test files** (per The New Stack, 2025). It grows every year; it is explicitly a moving target.
- Layout: **each top-level directory = one spec** (WHATWG specs named by subdomain, W3C by shortname), except `css/` which holds all CSSWG suites. The `css/` tree is the single largest block (tens of thousands of files; CSS 2.1 alone was ~10k).
- **WPT excludes ECMAScript** (that's test262, run against the JS engine separately) and WebGL (separate Khronos suite).
- Tests are mostly self-contained HTML pages (testharness.js for JS assertions, reftests for pixel comparison). Running them needs a test server (`wptserve`), webdriver-ish automation, and screenshot comparison — plan this harness early, it's the project's scoreboard.
- Practical implication: nobody passes 100%. Chromium/Firefox sit well above 90% on most areas; **Servo at 61.6% is the honest "usable-ish engine" benchmark**. Bowser's phased targets could be: 25% (page renders), 50% (Servo parity), 70% (daily-drivable for common sites).

## 3. Subsystem inventory (prioritized)

### P0 — nothing works without these
1. **URL parsing (WHATWG URL)** — every subsystem consumes URLs. IDNA/punycode needed (real international domains).
2. **Character encodings** — UTF-8 plus the legacy label table (windows-1252 etc.); encoding sniffing per HTML spec.
3. **Networking: TCP → TLS 1.2/1.3 → HTTP/1.1** — hand-rolled or via crates (see §7). Redirects, cookie jar (RFC 6265 + real-world quirks), `data:`/`blob:`/`file:` schemes, gzip/deflate/br decompression, HSTS, basic caching (Cache-Control/ETag), range requests.
4. **HTTP/2** — required early: most real sites negotiate h2 via ALPN. HTTP/3/QUIC can wait.
5. **Fetch** — the WHATWG fetch algorithm (CORS, credentials modes, redirect handling, streaming bodies). `fetch()`/`XMLHttpRequest` are the two network APIs real sites use.
6. **HTML parsing** — WHATWG tokenizer + tree construction with full error recovery (this *is* the 90% bar; tag-soup tolerance, adoption agency algorithm, foster parenting, `document.write` re-entrancy). `innerHTML`/`outerHTML` fragment parsing falls out of the same code.
7. **DOM Core + Events + DOM Standard** — Node/Element/Document, tree mutation, `querySelector(All)`, classList, dataset, DOM events (capture/bubble, `addEventListener`, `dispatchEvent`, `EventTarget`), MutationObserver, `getComputedStyle`, geometry (`getBoundingClientRect`, `offset*`, scroll metrics), ranges/selection (needed by editors and copy/paste).
8. **CSS: parsing, cascade, inheritance, values** — full grammar incl. error recovery, specificity, custom properties (`var()`), `calc()`/`min()`/`max()`/`clamp()`, @media/@supports/@font-face/@import, UA stylesheet, inheritance, `!important`.
9. **CSS layout, in dependency order** — block layout → inline layout + text (line breaking, bidi basics) → box model/margin collapsing → floats/clear → positioning (relative/absolute/fixed/sticky) → **tables** (still everywhere on real sites) → **flexbox** → **grid**. Then: overflow/scrolling, stacking contexts/z-index, transforms, opacity, border-radius, box-shadow, backgrounds/gradients, list markers, `::before`/`::after` generated content.
10. **JavaScript engine + bindings** — the single biggest subsystem. Options: embed **Boa** (pure Rust, ~94.6% test262 conformance as of Aug 2025; bytecode VM) or SpiderMonkey/V8 via FFI (violates minimal-deps/std-only). Writing a competitive JS engine from scratch is a multi-year project on its own; Boa is the only sane Rust-native path. Then: WebIDL-style DOM bindings (document/window/navigator/location/history), **event loop** (task queues + microtasks — correctness here breaks everything), timers (`setTimeout`/`setInterval`/`requestAnimationFrame`), `Promise` integration, `JSON`, `URL`/`URLSearchParams`, `TextEncoder/Decoder`, `Intl` (basic), `structuredClone`.
11. **Paint + compositing + text** — display lists, rasterization (software first), text shaping (rustybuzz-class; harfbuzz is C), font stack (system fonts + `@font-face` webfonts), images (PNG/JPEG/GIF/WebP/AVIF decoders + SVG), scrolling, hit testing, visible text selection.
12. **Resource pipeline** — `<script>` (classic + modules + import maps), `<link rel=stylesheet>`, `<img>`/`srcset`/`sizes`, preload/prefetch, image lazy-loading.

### P1 — needed for the 90% bar on modern sites
13. **WebSockets** (RFC 6455) — chat, live updates, devtools-adjacent tooling.
14. **Workers** — dedicated workers first (sites offload to them constantly), then shared workers; service workers later (P2 — big surface: lifecycle, Cache API).
15. **Storage** — cookies (P0), `localStorage`/`sessionStorage` (P1, trivial), **IndexedDB** (P1 — heavily used by frameworks), Cache API (with service workers, P2).
16. **HTML forms** — form submission encoding, validation API, `<input>` types, file upload.
17. **History/Navigation** — session history, `pushState`/`replaceState`, `popstate`/`hashchange`, Navigation API (P2).
18. **Canvas 2D** — charts, captchas, fingerprinting libs; software raster is fine initially.
19. **Media: `<audio>`/`<video>` + MSE** — decode pipeline is huge; MSE is what YouTube-likes need. WebCodecs/WebAudio are P2.
20. **Observers**: IntersectionObserver + ResizeObserver (lazy-load, infinite scroll, responsive components), MutationObserver (P0, frameworks depend on it).
21. **Shadow DOM + Custom Elements** — many component libs and sites (YouTube, GitHub) require at least basic support.
22. **Clipboard, Fullscreen, Notifications, Geolocation, File API** — small, high-leverage.
23. **WebCrypto** — subtle but blocking for auth flows (WebAuthn-adjacent crypto, SubtleCrypto).
24. **SVG** — inline SVG is P1 (icons everywhere); full SVG-as-image SMIL animation is P2/P3.
25. **Accessibility tree** — needed for automation/testing and OS integration; also feeds WPT-adjacent a11y tests.

### P2 — real but deferrable (needed for parity, not for 90%)
Service workers + Cache API + Push API, WebGL 1/2 (three.js sites), WebAudio, WebRTC, WebTransport, WebGPU, WebXR, WebNN, File System Access API, WebCodecs, WebHID/Bluetooth/USB/NFC/Serial, Payment Request, Credential Management/WebAuthn, view transitions, scroll-driven animations, anchor positioning, container queries (style queries), `@scope`, custom highlights, popover/dialog full interop, `shape()`, advanced `attr()`, MathML, printing, find-in-page, spellcheck, autofill/passwords, translations, reader mode.

### P3 — explicitly out of scope (do not build)
NPAPI/PPAPI plugins, Flash, Java applets, ActiveX, VBScript, XSLT (browsers keep it but no site needs it new), FTP/gopher protocols, **quirks mode** (see §5), RSS/Atom UI, browser extensions API, devtools protocol (CDP) — maybe much later for test automation.

## 4. Web APIs real sites depend on most (rough priority)

Tier A (site breaks without): `document`/`window` DOM, `querySelector(All)`, event listeners, `fetch`, `XMLHttpRequest`, `Promise`+microtasks, `setTimeout`/`setInterval`/`requestAnimationFrame`, `localStorage`, cookies, `JSON`, `URL`, `classList`/`dataset`, `innerHTML`, `getComputedStyle`, `getBoundingClientRect`, `addEventListener` on window/document/elements, History API, `postMessage`, `navigator.userAgent` (sniffing!), `matchMedia`, `IntersectionObserver`, `ResizeObserver`, `MutationObserver`, `WebSocket`, `FormData`, `Blob`/`File`/`FileReader`, canvas 2D basics, `Intl` (formatting), `structuredClone`.
Tier B (large minority of sites): IndexedDB, workers, geolocation, notifications, clipboard, fullscreen, drag-and-drop, `<video>`/MSE, WebGL, SVG DOM, custom elements/shadow DOM, `crypto.subtle`, `AbortController`, streams (`ReadableStream` — fetch bodies!), `TextEncoder/Decoder`, `BroadcastChannel`, `SharedArrayBuffer` (needs COOP/COEP), `OffscreenCanvas`.
Tier C (niche): everything in P2 above.

## 5. Cutoff lines: what to deliberately skip and still hit 90%

- **Quirks mode: skip it; always render in standards mode.** Every modern site ships a doctype. Implementing the full quirks/almost-standards matrix is pure legacy cost. (Risk: a thin tail of ancient pages renders slightly differently — acceptable.)
- **XML/XHTML mode: parse as HTML.** `application/xhtml+xml` documents can be fed through the HTML parser (like tag soup); true XML well-formedness errors can be ignored.
- **No plugins, no Java, no Flash, no ActiveX, no VBScript, no XSLT UI.**
- **HTTP/1.1 + HTTP/2 + TLS 1.2/1.3 only at first.** Skip HTTP/3/QUIC, ECH, OCSP stapling, client certs initially.
- **Skip pre-CSS-layout relics**: `<font>`, `<center>`, `<marquee>`, `<blink>` need only *parse-and-style* support (they're in the HTML parser's formatting-element list and UA stylesheet anyway — near-free), not special engines.
- **CSS: implement widely-used properties first** (the ~150 properties covering box model, typography, flex/grid, positioning, backgrounds, transforms); defer `shape()`, `contrast-color()`, advanced `attr()`, CSS nesting edge cases, `@property` full semantics, Houdini APIs, CSS modules.
- **JS: target ES2023-ish via Boa; skip** decorators-proposal churn, `SharedArrayBuffer` initially, WASM GC/JSPI (Interop 2026 items — not needed for 90%).
- **Media: MSE + basic `<video>` before WebCodecs/WebAudio.**
- **No enterprise/legacy auth**: NTLM/Kerberos/Negotiate, client certificates — defer.

## 6. Prioritized build order (dependency order — what blocks what)

```
Phase 0 — foundations (blocks everything)
  URL parser → encodings → base64/percent-encoding → MIME sniffing
  WPT harness + reftest screenshot infra (the scoreboard; build it FIRST)

Phase 1 — fetch a page
  TCP+TLS → HTTP/1.1 → redirects → cookies → compression → HTTP/2 → fetch algorithm
  (network stack unblocks all resource loading)

Phase 2 — static document
  HTML tokenizer/tree construction → DOM → CSS parse/cascade → style resolution
  → block+inline layout → tables/floats/positioning → flexbox → grid
  → paint/display lists → text+fonts+images → windowed renderer (screenshot first, window later)

Phase 3 — make it programmable
  JS engine (Boa) → WebIDL bindings → event loop (tasks/microtasks) → timers/rAF
  → DOM/HTML/BOM bindings → XHR/fetch bindings → storage → events end-to-end
  (this is where "90% of sites" starts becoming true)

Phase 4 — dynamic web
  workers → WebSockets → shadow DOM/custom elements → observers → canvas 2D
  → forms/validation → history/navigation → media/MSE → IndexedDB → SVG DOM

Phase 5 — parity + chrome
  service workers, WebGL, WebAudio, WebRTC/WebTransport (defer most)
  browser chrome: tabs, navigation, history UI, downloads, settings, profiles
  accessibility tree, automation hooks
```

Dependency notes: layout needs style; style needs DOM; DOM needs the HTML parser; script execution needs parser+DOM+event loop; paint needs layout+fonts+images; fetch needs network+cookies; workers need a second JS event loop + structured clone; service workers need fetch interception + Cache API + storage.

## 7. Notes on "std-only Rust, minimal deps" (tension to resolve)

- The goal says std-only + minimal deps + extreme performance. Strictly std-only means hand-rolling **TLS, HTTP/2 framing, HPACK, Brotli, image codecs, font shaping** — each a multi-month project and a security minefield (TLS especially).
- Pure-Rust crates exist that keep the *spirit* (no C/C++, memory-safe): **rustls** (TLS 1.2/1.3; note default provider `ring` contains C/asm — pure-Rust providers now exist), **h2** (HTTP/2), **webpki**, **flate2/miniz_oxide**, **brotli** (pure-Rust port exists), **image** (pure-Rust decoders), **rustybuzz** (pure-Rust shaping), **Boa** (pure-Rust JS).
- Recommendation for the orchestrator: define "minimal deps" as **pure-Rust, no-C crates only** rather than literal std-only; reserve hand-rolled implementations for the differentiators (HTML parser, CSS, layout, DOM — where Servo/others prove from-scratch is tractable). Reconsider only if the std-only constraint is absolute, in which case budget ~6-12 extra months for TLS+HTTP/2 alone.
- Reference architectures worth mining (read-only): **Servo** (Stylo CSS, html5ever parser — both originated as from-scratch Rust), **starfish-rs** (from-scratch Rust engine with a well-sequenced Epic 1→9 plan: static pipeline → coverage+images → networking → JS → grid/transforms → text → CSS coverage → JS web APIs → SVG), **Brimp** (Rust browser reusing Servo components + feature gates per subsystem).

## 8. Suggested WPT-driven milestones

1. `html/` + `dom/` parsing tests green → Phase 2 credible.
2. `css/css-box`, `css-flexbox`, `css-grid`, `css-tables` reftests → layout credible.
3. test262 via Boa ≥ 90% → JS credible.
4. `fetch/`, `xhr/`, `cookies/`, `webstorage/`, `workers/` → dynamic web credible.
5. Interop-style dashboard tracking per-directory pass rates; web-compat spot-checks against top-site list.
