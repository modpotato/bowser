# Fetching web-platform-tests for local runs

`bowser-wpt-run` needs a local checkout of
[web-platform-tests/wpt](https://github.com/web-platform-tests/wpt). This
note explains how to get one and how the harness consumes it.

## Clone

WPT is large (gigabytes with history). A shallow, blob-filtered clone is
enough for running tests:

```sh
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/web-platform-tests/wpt.git
cd wpt
# Only the directories you actually want to run; everything else stays
# un-downloaded until touched.
git sparse-checkout set dom html css/css-box
```

To run the full suite later, `git sparse-checkout disable` and let git
fetch blobs on demand. Re-run `git pull --depth 1` periodically; WPT is a
moving target and scores are only comparable against a pinned revision.
Record the revision you ran against (`git rev-parse HEAD`) alongside results.

Disk budget: a full checkout with blobs is several GB. Sparse is strongly
recommended on small machines.

## Generating MANIFEST.json (recommended)

The harness can walk the checkout directly (`--wpt-root`), but the accurate
path is the official manifest, which knows each test's real type
(testharness vs. reftest vs. manual vs. ...):

```sh
cd wpt
./wpt manifest --path MANIFEST.json
```

This needs Python 3; it is the same manifest `wpt run` uses. Regenerate it
after pulling.

### The v9 format bowser reads

`./wpt manifest` writes version 9, and bowser parses exactly that shape —
not an approximation of it:

```json
{"version": 9, "url_base": "/", "items": {
  "testharness": {"dom/a.html": [[null, {}]]},
  "reftest":     {"css/b.html": [[null, [["/css/b-ref.html", "=="]], {}]]}
}}
```

- `version` must be `9`. Anything else — including a missing version — is
  an error, not a guess: the manifest is stale or came from another tool,
  so regenerate it with `./wpt manifest`.
- Each path maps to a **list** of items, not a single item. One file can
  produce several tests (multi-global variants such as `?1=1`, `?2=2`).
- A testharness item is `[url, extras]`; a reftest item is
  `[url, references, extras]`. `url` is `null` when it equals the path —
  the common case — and the served URL is `url_base` + path then. Real
  manifests store urls without a leading slash.
- Entries whose shape bowser does not understand are skipped and counted
  (reported as "unknown shape" by `bowser-wpt-run`); they never abort the
  run. A manifest where *every* entry is skipped usually means the file is
  not actually v9 — check the `version` field.

## Running the harness

```sh
# List what would run:
bowser-wpt-run --manifest wpt/MANIFEST.json --list --kind testharness --include dom/

# Or walk the checkout without a manifest:
bowser-wpt-run --wpt-root wpt --list --include css/

# Actually run against the real browser driver (`BowserDriver` in
# `crates/bowser_app/src/wpt_driver.rs`, behind the `wpt-driver` cargo
# feature; exercised by `crates/bowser_app/tests/wpt_render.rs`). To
# exercise just the manifest/filter/report plumbing without a browser,
# `bowser-wpt-run` still defaults to the `StubDriver` (everything reports
# DRIVER-FAIL):
bowser-wpt-run --manifest wpt/MANIFEST.json --include html/ --timeout-secs 30
```

## What the harness does not do yet

- Serve tests over HTTP (`wptserve`). Needed before any real run: many
  tests assume the `/resources/` root and same-origin HTTP semantics.
- Screenshot comparison at scale (see `reftest-design.md`).
- Honor manifest `extras`. `timeout`, `pac`, `fuzzy`, `viewport_size` and
  friends are parsed but ignored: every testharness test gets the CLI
  `--timeout-secs`, every reftest the default exact comparison. In
  particular, `<meta name="fuzzy">` annotations are not mapped to
  comparison tolerances yet.
- Reftest references come from the manifest when present. The runner
  prefers the `[[ref_url, "=="|"="], ...]` lists parsed from
  `MANIFEST.json` at manifest time; it only scrapes `<link
  rel=match|mismatch>` from the file on disk as a fallback when the
  manifest entry carries no reference list (walk mode, hand-built
  manifests).
- The JSON parser is deliberately small and std-only; it rejects inputs
  nested beyond its depth limit instead of recursing forever.

## Test selection guidance

Start small and grow: `html/`, `dom/`, then `css/css-box`. Track
per-directory pass rates over time; that per-directory breakdown is the
scoreboard the project milestones are defined against.
