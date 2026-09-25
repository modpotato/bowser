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

## Running the harness

```sh
# List what would run:
bowser-wpt-run --manifest wpt/MANIFEST.json --list --kind testharness --include dom/

# Or walk the checkout without a manifest:
bowser-wpt-run --wpt-root wpt --list --include css/

# Actually run (today: stub driver, everything reports DRIVER-FAIL;
# useful for exercising the manifest/filter/report plumbing):
bowser-wpt-run --manifest wpt/MANIFEST.json --include html/ --timeout-secs 30
```

## What the harness does not do yet

- Serve tests over HTTP (`wptserve`). Needed before any real run: many
  tests assume the `/resources/` root and same-origin HTTP semantics.
- Drive a real browser. `BrowserDriver` is a trait with only a stub
  implementation until the renderer exists.
- Screenshot comparison at scale (see `reftest-design.md`).

## Test selection guidance

Start small and grow: `html/`, `dom/`, then `css/css-box`. Track
per-directory pass rates over time; that per-directory breakdown is the
scoreboard the project milestones are defined against.
