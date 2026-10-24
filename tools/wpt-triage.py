#!/usr/bin/env python3
"""wpt-triage.py: triage a bowser-wpt wptreport JSON against the expected-failure catalog.

Usage:
    python3 tools/wpt-triage.py <report.json> [--metadata-dir DIR]

Reads the wptreport JSON produced by `bowser-wpt --output-json` (run with
`--metadata-dir` so entries covered by the expected-failure catalog carry an
`"expected"` key) and prints plain, greppable lines:

    NEWF:     test file with a bad outcome and NO "expected" key (a new
              failure: the tail of the run).
    NEWF-SUB: one failing subtest of a NEWF file (subtest detail).
    FIXED:    test file WITH an "expected" key whose outcome is now green
              (catalog removal candidate).
    CHANGED:  test file WITH an "expected" key whose actual file status no
              longer matches the catalogued status (catalog maintenance).

Semantics mirror crates/bowser_wpt: a testharness file is green iff its
status is OK/PASS and every subtest passes. Bad file statuses are anything
else except SKIP (SKIP is neutral). Bad subtest statuses are FAIL, TIMEOUT,
NOTRUN. An "expected" key always marks catalog coverage, even when the
outcome is green.

A NEWF file whose message starts with "runner watchdog expired" is flagged
env=watchdog (runner-side kill, likely timing/environment); env=harness
otherwise (the page itself failed: likely a real bug).

Exit status: 0 when there are no new failures, 1 when there are.
"""

import json
import os
import re
import sys

GOOD_FILE = {"OK", "PASS"}
BAD_SUBTEST = {"FAIL", "TIMEOUT", "NOTRUN"}
WATCHDOG_PREFIX = "runner watchdog expired"


def is_green(entry):
    status = entry.get("status")
    if status not in GOOD_FILE:
        return False
    return all(s.get("status") == "PASS" for s in entry.get("subtests", []))


def is_skip(entry):
    return entry.get("status") == "SKIP"


def is_covered(entry, expected):
    """Mirror runner::is_expected_failure: does the actual outcome match the
    catalogued expectation? Only FAIL, TIMEOUT, ERROR exist in the catalog."""
    status = entry.get("status")
    if expected == "FAIL":
        # testharness: harness OK/FAIL and not green (incl. OK with failing
        # subtests); reftest/crashtest: status FAIL.
        return status in ("OK", "FAIL") and not is_green(entry)
    if expected == "TIMEOUT":
        # Both harness-reported TIMEOUT and the runner watchdog (DriverTimeout)
        # export as file status TIMEOUT; the message distinguishes them.
        return status == "TIMEOUT"
    if expected == "ERROR":
        return status == "ERROR"
    return False


def subtest_counts(entry):
    counts = {"pass": 0, "fail": 0, "timeout": 0, "notrun": 0, "other": 0}
    for s in entry.get("subtests", []):
        st = s.get("status", "?")
        if st == "PASS":
            counts["pass"] += 1
        elif st == "FAIL":
            counts["fail"] += 1
        elif st == "TIMEOUT":
            counts["timeout"] += 1
        elif st == "NOTRUN":
            counts["notrun"] += 1
        else:
            counts["other"] += 1
    return counts


def failing_subtests(entry):
    return [s for s in entry.get("subtests", []) if s.get("status") in BAD_SUBTEST]


def load_metadata_dir(d):
    """Minimal mirror of ExpectedMetadata::parse: section names with a
    leading '/' stripped, 'expected: STATUS' per section."""
    expected = {}
    pending = None
    sec_re = re.compile(r"^\[(.+)\]$")
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".ini"):
            continue
        with open(os.path.join(d, fn), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith(";"):
                    continue
                m = sec_re.match(line)
                if m:
                    name = m.group(1).strip().lstrip("/")
                    pending = name
                    continue
                if pending is None:
                    continue
                if ":" in line or "=" in line:
                    i = min((line.find(c) for c in ":=" if c in line))
                    key, val = line[:i].strip(), line[i + 1 :].strip()
                    if key == "expected" and val:
                        expected[pending] = val
    return expected


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__.strip().splitlines()[0])
    path = sys.argv[1]
    metadata = {}
    if "--metadata-dir" in sys.argv:
        md = sys.argv[sys.argv.index("--metadata-dir") + 1]
        metadata = load_metadata_dir(md)

    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    results = report.get("results", [])

    newf, fixed, changed, skipped, green, cataloged = [], [], [], [], [], []
    total_bad_subtests = 0
    total_pass_subtests = 0

    for entry in results:
        test = entry.get("test", "?")
        exp = entry.get("expected")
        if exp is None and metadata:
            exp = metadata.get(test.lstrip("/"))
        c = subtest_counts(entry)
        total_pass_subtests += c["pass"]
        total_bad_subtests += c["fail"] + c["timeout"] + c["notrun"]

        if is_skip(entry):
            skipped.append(entry)
            continue
        if exp is not None and is_green(entry):
            fixed.append((entry, exp))
            continue
        if exp is not None:
            cataloged.append(entry)
            if not is_covered(entry, exp):
                changed.append((entry, exp))
            continue
        # No catalog coverage: bad file status, or bad subtests in a good file.
        if entry.get("status") not in GOOD_FILE or failing_subtests(entry):
            newf.append(entry)
            continue
        green.append(entry)

    n_files = len(results)
    print(f"# wpt triage: {path}")
    print(
        f"# files: {n_files} total, {len(green)} green, {len(newf)} new-fail, "
        f"{len(cataloged)} expected-fail, {len(fixed)} fixed, "
        f"{len(changed)} changed-vs-catalog, {len(skipped)} skip"
    )
    print(
        f"# subtests: {total_pass_subtests} pass, {total_bad_subtests} not-pass"
    )

    for entry in newf:
        test = entry["test"]
        c = subtest_counts(entry)
        msg = entry.get("message") or ""
        env = "watchdog" if msg.startswith(WATCHDOG_PREFIX) else "harness"
        print(
            f"NEWF: {test} file={entry.get('status')} expected=none "
            f"pass={c['pass']} fail={c['fail']} timeout={c['timeout']} "
            f"notrun={c['notrun']} env={env}"
        )
        for s in failing_subtests(entry):
            print(f"NEWF-SUB: {test} :: {s.get('name')} status={s.get('status')}")

    for entry, exp in fixed:
        c = subtest_counts(entry)
        nsub = sum(c.values())
        print(
            f"FIXED: {entry['test']} expected={exp} now={entry.get('status')} "
            f"subtests={nsub} all-pass"
        )

    for entry, exp in changed:
        c = subtest_counts(entry)
        print(
            f"CHANGED: {entry['test']} expected={exp} actual={entry.get('status')} "
            f"pass={c['pass']} fail={c['fail']} timeout={c['timeout']} "
            f"notrun={c['notrun']}"
        )

    sys.exit(1 if newf else 0)


if __name__ == "__main__":
    main()
