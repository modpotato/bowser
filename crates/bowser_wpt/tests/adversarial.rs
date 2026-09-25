//! Adversarial regression tests for `bowser_wpt`.
//!
//! Each test pins the *correct* behavior for a tricky path found by review:
//! malformed JSON that must error instead of abort, the real v9
//! `MANIFEST.json` shape, walk classification edge cases, reference-link
//! scanning subtleties, and screenshot comparison soundness.
//!
//! These tests fail on the unfixed code and pass once the corresponding fix
//! lands. They are integration tests on purpose: they exercise only the
//! public API, exactly the way the CLI and future drivers will.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use bowser_wpt::driver::{BrowserDriver, DriverError, Screenshot, Viewport};
use bowser_wpt::manifest::{Manifest, TestItem, TestKind};
use bowser_wpt::reftest::{self, RefRel, ReftestTolerance};
use bowser_wpt::results::{
    HarnessResult, HarnessStatus, SubtestResult, SubtestStatus, parse_wptreport,
};
use bowser_wpt::runner::{RunConfig, Runner, TestOutcome};

static TREE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A fresh scratch directory for one test. Removed by the test when done.
fn scratch_dir(tag: &str) -> PathBuf {
    let n = TREE_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("bowser_wpt_adv_{tag}_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_files(dir: &Path, files: &[(&str, &str)]) {
    for (rel, content) in files {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, content).unwrap();
    }
}

fn write_file(dir: &Path, name: &str, content: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, content).unwrap();
    path
}

// ---------------------------------------------------------------------------
// JSON edge cases, exercised through the public manifest parser.
// ---------------------------------------------------------------------------

#[test]
fn lone_surrogates_are_rejected() {
    // A lone high surrogate, a lone low surrogate, and a high surrogate
    // followed by something that is not a low surrogate must all fail.
    // They must never silently become U+FFFD and never panic.
    for escape in ["\\uD83D", "\\uDE00", "\\uD83D\\u0041"] {
        let json = format!(
            "{{\"version\": 9, \"items\": {{\"testharness\": {{\"{escape}.html\": [[null, {{}}]]}}}}}}"
        );
        assert!(
            Manifest::from_json(&json).is_err(),
            "lone surrogate {escape} must be rejected"
        );
    }
}

#[test]
fn valid_surrogate_pair_still_parses() {
    let json = "{\"version\": 9, \"items\": {\"testharness\": \
                {\"a\\uD83D\\uDE00b.html\": [[null, {}]]}}}";
    let m = Manifest::from_json(json).unwrap();
    assert_eq!(m.items()[0].id, "a\u{1F600}b.html");
}

#[test]
fn deep_nesting_returns_err_instead_of_aborting() {
    // 200k nested arrays would need hundreds of megabytes of stack with a
    // naive recursive parser. The parser must return Err (depth limit),
    // never abort the process with a stack overflow.
    let depth = 200_000;
    let mut json = "[".repeat(depth);
    json.push_str(&"]".repeat(depth));
    assert!(Manifest::from_json(&json).is_err());
}

// ---------------------------------------------------------------------------
// Real v9 MANIFEST.json handling.
// ---------------------------------------------------------------------------

#[test]
fn v9_null_url_defaults_to_path() {
    // In real manifests the item url is null when it equals the path.
    let m = Manifest::from_json(
        r#"{"version": 9, "url_base": "/", "items": {
            "testharness": {"dom/a.html": [[null, {}]]}
        }}"#,
    )
    .unwrap();
    assert_eq!(m.len(), 1);
    assert_eq!(m.skipped, 0);
    let t = &m.items()[0];
    assert_eq!(t.id, "dom/a.html");
    assert_eq!(t.url, "/dom/a.html");
    assert_eq!(t.kind, TestKind::Testharness);
}

#[test]
fn v9_explicit_url_is_used_verbatim() {
    // Variant urls (multi-global `?1=1` / `?2=2`) are stored explicitly.
    let m = Manifest::from_json(
        r#"{"version": 9, "url_base": "/", "items": {
            "testharness": {"dom/a.html": [["dom/a.html?1=1", {}]]}
        }}"#,
    )
    .unwrap();
    assert_eq!(m.len(), 1);
    assert_eq!(m.items()[0].url, "/dom/a.html?1=1");
}

#[test]
fn v9_null_url_respects_url_base() {
    let m = Manifest::from_json(
        r#"{"version": 9, "url_base": "/_wpt_/", "items": {
            "testharness": {"a.html": [[null, {}]]}
        }}"#,
    )
    .unwrap();
    assert_eq!(m.items()[0].url, "/_wpt_/a.html");
}

#[test]
fn v9_multi_item_paths_yield_one_test_per_item() {
    // One file can produce several tests; the value per path is a list.
    // A malformed entry inside the list is skipped, not fatal.
    let m = Manifest::from_json(
        r#"{"version": 9, "url_base": "/", "items": {
            "testharness": {"css/a.html": [
                [null, {}],
                ["css/a.html?1=1", {}],
                ["css/a.html?2=2", {}],
                "bogus"
            ]}
        }}"#,
    )
    .unwrap();
    assert_eq!(m.len(), 3);
    assert_eq!(m.skipped, 1);
    let urls: Vec<&str> = m.items().iter().map(|t| t.url.as_str()).collect();
    assert_eq!(urls, ["/css/a.html", "/css/a.html?1=1", "/css/a.html?2=2"]);
}

#[test]
fn v9_reftest_item_with_references() {
    // Reftest items are [url, references, extras]; references must not break
    // enumeration.
    let m = Manifest::from_json(
        r#"{"version": 9, "url_base": "/", "items": {
            "reftest": {"css/b.html": [[null, [["/css/b-ref.html", "=="],
                                                      ["/css/b-other.html", "!="]],
                                             {}]]}
        }}"#,
    )
    .unwrap();
    assert_eq!(m.len(), 1);
    assert_eq!(m.skipped, 0);
    let t = &m.items()[0];
    assert_eq!(t.kind, TestKind::Reftest);
    assert_eq!(t.url, "/css/b.html");
}

#[test]
fn v9_all_item_kinds_map() {
    let m = Manifest::from_json(
        r#"{"version": 9, "items": {
            "testharness": {"a.html": [[null, {}]]},
            "crashtest": {"b.html": [[null, {}]]},
            "manual": {"c.html": [[null, {}]]},
            "print-reftest": {"d.html": [[null, [["/d-ref.html", "=="]], {}]]},
            "wdspec": {"e.py": [[null, {}]]}
        }}"#,
    )
    .unwrap();
    assert_eq!(m.len(), 5);
    assert_eq!(m.by_kind(&TestKind::Testharness).len(), 1);
    assert_eq!(m.by_kind(&TestKind::Crashtest).len(), 1);
    assert_eq!(m.by_kind(&TestKind::Manual).len(), 1);
    assert_eq!(m.by_kind(&TestKind::PrintReftest).len(), 1);
    assert_eq!(m.by_kind(&TestKind::Other("wdspec".into())).len(), 1);
}

#[test]
fn v9_version_mismatch_is_rejected() {
    // Matches wpt itself: anything but version 9 is a version mismatch,
    // including a missing version. Regenerate the manifest.
    for version in ["8", "10"] {
        let json = format!("{{\"version\": {version}, \"items\": {{}}}}");
        assert!(
            Manifest::from_json(&json).is_err(),
            "version {version} must be rejected"
        );
    }
    assert!(Manifest::from_json(r#"{"items": {}}"#).is_err());
}

// ---------------------------------------------------------------------------
// Walk classification edge cases.
// ---------------------------------------------------------------------------

const TESTHARNESS_HTML: &str = "<!doctype html><script src=\"/resources/testharness.js\"></script>";

#[test]
fn walk_excludes_ref_files() {
    // `*-ref.html` files are reftest references, not tests, even when their
    // content looks like a test.
    let dir = scratch_dir("ref-exclusion");
    write_files(
        &dir,
        &[
            ("css/t.html", TESTHARNESS_HTML),
            ("css/t-ref.html", TESTHARNESS_HTML),
            ("css/other-ref.html", "<!doctype html><title>ref</title>"),
        ],
    );
    let m = Manifest::walk(&dir).unwrap();
    let ids: Vec<&str> = m.items().iter().map(|t| t.id.as_str()).collect();
    assert_eq!(ids, ["css/t.html"]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn walk_classifies_xht_files() {
    // WPT ships XHTML tests as .xht; they must be enumerated and classified.
    let dir = scratch_dir("xht");
    write_files(
        &dir,
        &[
            ("a/t.xht", TESTHARNESS_HTML),
            ("a/u.XHT", TESTHARNESS_HTML),
            ("a/v.xhtml", TESTHARNESS_HTML),
        ],
    );
    let m = Manifest::walk(&dir).unwrap();
    assert_eq!(m.len(), 3);
    assert!(
        m.items().iter().all(|t| t.kind == TestKind::Testharness),
        "every .xht/.xhtml file must classify as testharness"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Reference-link scanning edge cases.
// ---------------------------------------------------------------------------

#[test]
fn unquoted_href_with_slash_is_kept_whole() {
    // Unquoted attribute values end at whitespace or `>`, not at `/`:
    // `<link rel=match href=sub/dir/ref.html>` has href "sub/dir/ref.html".
    let dir = scratch_dir("href-slash");
    let p = write_file(&dir, "t.html", "<link rel=match href=sub/dir/ref.html>");
    let refs = reftest::extract_references(&p).unwrap();
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].rel, RefRel::Match);
    assert_eq!(refs[0].href, "sub/dir/ref.html");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn relx_does_not_count_as_rel() {
    // `relx` is a different attribute; it must not classify as a reftest.
    let dir = scratch_dir("relx-negative");
    let p = write_file(&dir, "t.html", "<link relx=match href=\"x.html\">");
    assert!(reftest::extract_references(&p).unwrap().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn real_rel_is_found_after_relx_prefixed_attribute() {
    // A bogus `relx` attribute earlier in the tag must not hide a real
    // `rel` attribute later in the same tag.
    let dir = scratch_dir("relx-positive");
    let p = write_file(
        &dir,
        "t.html",
        "<link relx=\"mismatch\" rel=match href=\"x.html\">",
    );
    let refs = reftest::extract_references(&p).unwrap();
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].rel, RefRel::Match);
    assert_eq!(refs[0].href, "x.html");
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Screenshot comparison soundness.
// ---------------------------------------------------------------------------

#[test]
fn compare_rejects_mismatched_buffers() {
    // A buffer that cannot hold width*height RGBA pixels is broken input:
    // an error, never a silent pass (or a panic on the short read).
    let short = Screenshot {
        width: 2,
        height: 2,
        rgba: vec![0u8; 15],
    };
    let long = Screenshot {
        width: 2,
        height: 2,
        rgba: vec![0u8; 17],
    };
    let full = Screenshot::solid(2, 2, 0, 0, 0, 255);
    assert!(reftest::compare(&short, &full, ReftestTolerance::EXACT).is_err());
    assert!(reftest::compare(&full, &short, ReftestTolerance::EXACT).is_err());
    assert!(reftest::compare(&long, &full, ReftestTolerance::EXACT).is_err());
    assert!(reftest::compare(&short, &short, ReftestTolerance::EXACT).is_err());
}

// ---------------------------------------------------------------------------
// wptreport import.
// ---------------------------------------------------------------------------

#[test]
fn wptreport_pass_and_fail_subtests() {
    let doc = r#"{"results": [
        {"test": "/a.html", "status": "OK", "subtests": [
            {"name": "one", "status": "PASS"},
            {"name": "two", "status": "FAIL", "message": "nope"}
        ]},
        {"test": "/b.html", "status": "OK", "subtests": [
            {"name": "one", "status": "PASS"}
        ]}
    ]}"#;
    let (results, skipped) = parse_wptreport(doc).unwrap();
    assert_eq!(skipped, 0);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].subtests[0].status, SubtestStatus::Pass);
    assert_eq!(results[0].subtests[1].status, SubtestStatus::Fail);
    assert_eq!(results[0].subtests[1].message.as_deref(), Some("nope"));
    assert!(!results[0].is_green());
    assert_eq!(results[0].passed(), 1);
    assert_eq!(results[0].failed(), 1);
    assert!(results[1].is_green());
}

// ---------------------------------------------------------------------------
// Runner routing.
// ---------------------------------------------------------------------------

/// A driver with canned answers, so routing is testable without a browser.
struct ScriptedDriver;

impl BrowserDriver for ScriptedDriver {
    fn run_testharness(
        &self,
        test: &TestItem,
        _timeout: Duration,
    ) -> Result<HarnessResult, DriverError> {
        Ok(HarnessResult {
            test: test.id.clone(),
            status: HarnessStatus::Ok,
            duration_ms: Some(1),
            subtests: vec![SubtestResult {
                name: "t".into(),
                status: SubtestStatus::Pass,
                message: None,
            }],
        })
    }

    fn load_without_crash(&self, _url: &str) -> Result<(), DriverError> {
        Ok(())
    }

    fn render(&self, _url: &str, _viewport: Viewport) -> Result<Screenshot, DriverError> {
        Ok(Screenshot::solid(1, 1, 0, 0, 0, 255))
    }

    fn name(&self) -> &str {
        "scripted"
    }
}

#[test]
fn crashtest_routes_through_load_without_crash() {
    // A crashtest passes when the page loads without killing the browser;
    // it never waits on a testharness completion callback, so the outcome
    // is Crashtest (not Harness) even though this driver's run_testharness
    // would also succeed.
    let m = Manifest::from_json(
        r#"{"version": 9, "items": {
            "crashtest": {"a/crash.html": [[null, {}]]}
        }}"#,
    )
    .unwrap();
    let mut r = Runner::new(ScriptedDriver, RunConfig::default());
    let report = r.run(&m);
    assert_eq!(report.summary.total, 1);
    assert_eq!(report.summary.crashtest_pass, 1);
    assert_eq!(report.summary.crashtest_fail, 0);
    assert_eq!(report.summary.harness_pass, 0);
    assert!(
        matches!(
            report.reports[0].outcome,
            TestOutcome::Crashtest { passed: true }
        ),
        "crashtest must produce a crashtest outcome, got {:?}",
        report.reports[0].outcome
    );
    assert!(report.reports[0].outcome.passed());
}
