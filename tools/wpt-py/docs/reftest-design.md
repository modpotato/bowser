# Reftest design

How Bowser will run WPT reftests (pixel-comparison tests) once the renderer
exists. Status: design only. The crate already ships the browser-independent
pieces: `bowser_wpt::reftest::extract_references` (finds `<link rel=match|mismatch>`
in a test file) and `bowser_wpt::reftest::compare` (screenshot diff with
per-channel and fraction tolerance).

## What a reftest is

A reftest is an HTML file plus one or more reference files linked from it:

```html
<link rel="match" href="reference.html">
<link rel="mismatch" href="not-this.html">
```

The test passes when its rendering is pixel-equal to every `match`
reference and pixel-different from every `mismatch` reference. References
are ordinary pages rendered by the same engine; they may also carry
`<meta name="fuzzy" content="...">` annotations declaring how much
difference is acceptable (see below).

## Pipeline

```
enumerate (manifest/walk)
  -> serve test + references over HTTP (wptserve-compatible)
  -> render each in the real viewport (800x600 default, per-test overrides)
  -> compare screenshots
  -> report pass/fail + diff metrics
```

### 1. Serving

Tests must be served over HTTP, not `file://`: many tests rely on the
`/resources/` path root, on `wptserve`'s pipe/sub features, and on same-origin
behavior. Bowser needs a small test HTTP server (or a `wptserve` reverse
proxy) that maps URL paths to the checkout. Reuse the real `wptserve` from
the WPT repo if possible; it already handles `.asis` files, range/pipes, and
WebSocket upgrades. This is the single biggest piece of future work here.

### 2. Rendering

The `BrowserDriver::render` implementation (lands with `bowser_paint`):

- Launch headless `bowser_app` (or an in-process harness hooking the
  paint pipeline directly; in-process is faster and avoids IPC flakiness).
- Navigate to the test URL, run to load-complete + `reftest-wait` handling:
  tests with class `reftest-wait` on `<html>` signal async work; the driver
  must wait until the class is removed or a timeout fires.
- Capture the viewport as RGBA at device-pixel ratio 1. Screenshots must be
  deterministic: disable animations, fix the clock (`?` query params that
  wptserve honors), and use one font set.

### 3. Comparison

`reftest::compare` with a `ReftestTolerance`:

- Default is exact (`per_channel: 0, max_diff_fraction: 0.0`).
- WPT `fuzzy` metadata (`<meta name="fuzzy" content="maxDifference=0-64;totalPixels=0-100">`)
  maps onto the tolerance: `maxDifference` is the per-channel allowance,
  `totalPixels` the allowed count of differing pixels. Parsing that meta
  format is future work in `reftest.rs`.
- Size mismatch is an error, not a failure: it means the driver or the
  setup is broken (wrong viewport, crash mid-render), not that the pixels
  differ.

### 4. Reporting

Reftest outcomes are per-reference checks (`RefCheck { reference, rel,
comparison, ok }`); the test passes when every check goes the way its `rel`
demands. On failure, persist both screenshots plus a diff image (red
highlight on differing pixels) under `target/wpt-artifacts/` for inspection.
Never fail the whole run on one bad render; keep going and summarize.

## Open questions

- In-process vs. subprocess driver: in-process is faster and gives exact
  failure attribution; subprocess is closer to "a real browser". Decide when
  `bowser_app` has a headless mode.
- Font determinism: WPT ships `resources/fonts/` (Ahem). The renderer must
  prefer those over system fonts during runs.
- Print reftests (`print-reftest`): paged-media rendering; defer until the
  layout engine does fragmentation.
- `testdriver.js` actions (used by some reftests for scrolling/clicking):
  needs the Action API in the driver; defer.
