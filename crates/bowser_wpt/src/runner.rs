//! Running a manifest against a [`BrowserDriver`](crate::driver::BrowserDriver).
//!
//! [`Runner`] filters a [`Manifest`](crate::manifest::Manifest), hands each
//! test to the driver, and collects a [`RunReport`]. Testharness tests go
//! through [`BrowserDriver::run_testharness`](crate::driver::BrowserDriver::run_testharness);
//! reftests are rendered alongside each of their references and compared with
//! [`compare`](crate::reftest::compare); crashtests just load the page and
//! pass when the browser survives
//! ([`BrowserDriver::load_without_crash`](crate::driver::BrowserDriver::load_without_crash)).

use std::time::Duration;

use crate::driver::{BrowserDriver, DriverError, Screenshot, Viewport};
use crate::manifest::{Manifest, TestItem, TestKind};
use crate::reftest::{self, Comparison, RefRel, ReftestTolerance};
use crate::results::HarnessResult;

/// What to run and how.
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// Substring filters: a test runs only if its id contains at least one
    /// entry. Empty means "no include filter".
    pub include: Vec<String>,
    /// Substring filters: a test never runs if its id contains any entry.
    pub exclude: Vec<String>,
    /// Test kinds to run. Empty means "all kinds" (non-automated tests are
    /// still skipped, see below).
    pub kinds: Vec<TestKind>,
    /// Per-test timeout for testharness tests.
    pub timeout: Duration,
    /// Viewport for reftest rendering.
    pub viewport: Viewport,
    /// Screenshot comparison tolerance for reftests.
    pub tolerance: ReftestTolerance,
}

impl Default for RunConfig {
    fn default() -> Self {
        RunConfig {
            include: Vec::new(),
            exclude: Vec::new(),
            kinds: Vec::new(),
            timeout: Duration::from_secs(60),
            viewport: Viewport::DEFAULT,
            tolerance: ReftestTolerance::EXACT,
        }
    }
}

impl RunConfig {
    /// Whether `item` passes the include/exclude/kind filters.
    pub fn wants(&self, item: &TestItem) -> bool {
        if !self.kinds.is_empty() && !self.kinds.contains(&item.kind) {
            return false;
        }
        if !self.include.is_empty() && !self.include.iter().any(|s| item.id.contains(s)) {
            return false;
        }
        if self.exclude.iter().any(|s| item.id.contains(s)) {
            return false;
        }
        true
    }
}

/// The outcome of one test.
#[derive(Debug, Clone)]
pub enum TestOutcome {
    /// A testharness test produced a report.
    Harness(HarnessResult),
    /// A reftest rendered and compared against its references.
    Reftest { passed: bool, checks: Vec<RefCheck> },
    /// A crashtest: `passed` is false when loading the page killed the
    /// browser. Never waits on a testharness completion callback.
    Crashtest { passed: bool },
    /// Not attempted; `reason` says why (manual test, no references, ...).
    Skipped { reason: String },
    /// The driver itself failed (timeout, crashed browser, unimplemented).
    DriverFailed(DriverError),
    /// The harness could not read a file it needed (reftest source,
    /// reference). This is harness-side I/O, not a driver problem, so it
    /// gets its own outcome instead of being lumped into [`TestOutcome::DriverFailed`].
    FilesystemFailure(String),
}

impl TestOutcome {
    /// Did the test pass?
    pub fn passed(&self) -> bool {
        match self {
            TestOutcome::Harness(r) => r.is_green(),
            TestOutcome::Reftest { passed, .. } | TestOutcome::Crashtest { passed } => *passed,
            TestOutcome::Skipped { .. }
            | TestOutcome::DriverFailed(_)
            | TestOutcome::FilesystemFailure(_) => false,
        }
    }
}

/// One reference checked during a reftest.
#[derive(Debug, Clone)]
pub struct RefCheck {
    pub reference: String,
    pub rel: RefRel,
    pub comparison: Comparison,
    /// Whether this reference check went the way its `rel` demands.
    pub ok: bool,
}

/// One test's report.
#[derive(Debug, Clone)]
pub struct TestReport {
    pub id: String,
    pub kind: TestKind,
    pub outcome: TestOutcome,
}

/// Aggregate counts for a run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunSummary {
    pub total: usize,
    pub harness_pass: usize,
    pub harness_fail: usize,
    pub crashtest_pass: usize,
    pub crashtest_fail: usize,
    pub reftest_pass: usize,
    pub reftest_fail: usize,
    pub skipped: usize,
    pub driver_failed: usize,
    /// Harness-side I/O failures (unreadable test or reference files).
    pub filesystem_failures: usize,
}

impl RunSummary {
    pub fn passed(&self) -> usize {
        self.harness_pass + self.crashtest_pass + self.reftest_pass
    }
}

/// Everything a run produced.
#[derive(Debug, Clone)]
pub struct RunReport {
    pub reports: Vec<TestReport>,
    pub summary: RunSummary,
    pub driver_name: String,
}

impl RunReport {
    /// One-line-per-section human summary.
    pub fn summary_text(&self) -> String {
        let s = &self.summary;
        format!(
            "driver: {}\n\
             total: {}\n\
             testharness: {} passed, {} failed\n\
             crashtest: {} passed, {} failed\n\
             reftest: {} passed, {} failed\n\
             skipped: {}\n\
             driver failures: {}\n\
             filesystem failures: {}",
            self.driver_name,
            s.total,
            s.harness_pass,
            s.harness_fail,
            s.crashtest_pass,
            s.crashtest_fail,
            s.reftest_pass,
            s.reftest_fail,
            s.skipped,
            s.driver_failed,
            s.filesystem_failures,
        )
    }
}

/// Runs a manifest against a driver.
pub struct Runner<D> {
    driver: D,
    config: RunConfig,
}

impl<D: BrowserDriver> Runner<D> {
    pub fn new(driver: D, config: RunConfig) -> Runner<D> {
        Runner { driver, config }
    }

    pub fn driver(&self) -> &D {
        &self.driver
    }

    pub fn driver_mut(&mut self) -> &mut D {
        &mut self.driver
    }

    /// Run every selected test in manifest order.
    pub fn run(&mut self, manifest: &Manifest) -> RunReport {
        let mut reports = Vec::new();
        for item in manifest.items() {
            if !self.config.wants(item) {
                continue;
            }
            let outcome = self.run_one(item);
            reports.push(TestReport {
                id: item.id.clone(),
                kind: item.kind.clone(),
                outcome,
            });
        }
        let mut summary = RunSummary::default();
        for r in &reports {
            tally(&mut summary, &r.outcome);
        }
        RunReport {
            reports,
            summary,
            driver_name: self.driver.name().to_owned(),
        }
    }

    pub(crate) fn run_one(&mut self, item: &TestItem) -> TestOutcome {
        if !item.kind.is_automated() {
            return TestOutcome::Skipped {
                reason: format!("{} tests need a human", item.kind),
            };
        }
        match item.kind {
            TestKind::Testharness => match self.driver.run_testharness(item, self.config.timeout) {
                Ok(result) => TestOutcome::Harness(result),
                Err(e) => TestOutcome::DriverFailed(e),
            },
            // Crashtests never wait on a testharness completion callback:
            // they pass when the page loads without killing the browser.
            TestKind::Crashtest => self.run_crashtest(item),
            TestKind::Reftest | TestKind::PrintReftest => self.run_reftest(item),
            _ => TestOutcome::Skipped {
                reason: format!("no runner for {}", item.kind),
            },
        }
    }

    /// Run one crashtest: load the page, pass if the browser survives.
    ///
    /// A dead browser is a *test* failure (the page under test crashed it),
    /// not a driver failure: the driver itself is fine. Any other driver
    /// error (timeout, unimplemented) is a driver failure.
    fn run_crashtest(&mut self, item: &TestItem) -> TestOutcome {
        match self.driver.load_without_crash(&item.url) {
            Ok(()) => TestOutcome::Crashtest { passed: true },
            Err(DriverError::BrowserGone(_)) => TestOutcome::Crashtest { passed: false },
            Err(e) => TestOutcome::DriverFailed(e),
        }
    }

    fn run_reftest(&mut self, item: &TestItem) -> TestOutcome {
        // Prefer the reference list parsed at manifest time. Items that
        // carry a local file but no parsed references (walk output,
        // hand-built items) fall back to scraping the file for
        // <link rel=match|mismatch> tags.
        let scraped;
        let refs: &[reftest::Reference] = if item.references.is_empty() {
            match &item.path {
                Some(path) => match reftest::extract_references(path) {
                    Ok(r) => {
                        scraped = r;
                        &scraped
                    }
                    // The reference source could not be read: harness-side
                    // I/O, not a driver problem.
                    Err(reftest::ReftestError::Io(e)) => {
                        return TestOutcome::FilesystemFailure(e.to_string());
                    }
                    Err(e) => {
                        return TestOutcome::DriverFailed(DriverError::BadReport(e.to_string()));
                    }
                },
                None => {
                    return TestOutcome::Skipped {
                        reason: "no references: manifest entry has no reference list \
                                 and no local file"
                            .into(),
                    };
                }
            }
        } else {
            &item.references
        };
        if refs.is_empty() {
            return TestOutcome::Skipped {
                reason: "no <link rel=match|mismatch> references found".into(),
            };
        }
        let test_shot = match self.driver.render(&item.url, self.config.viewport) {
            Ok(s) => s,
            Err(e) => return TestOutcome::DriverFailed(e),
        };
        let mut checks = Vec::with_capacity(refs.len());
        for r in refs {
            let ref_url = resolve_reference_url(&item.url, &r.href);
            let ref_shot: Screenshot = match self.driver.render(&ref_url, self.config.viewport) {
                Ok(s) => s,
                Err(e) => return TestOutcome::DriverFailed(e),
            };
            let comparison = match reftest::compare(&test_shot, &ref_shot, self.config.tolerance) {
                Ok(c) => c,
                Err(e) => return TestOutcome::DriverFailed(DriverError::BadReport(e.to_string())),
            };
            let ok = match r.rel {
                RefRel::Match => comparison.equal,
                RefRel::Mismatch => !comparison.equal,
            };
            checks.push(RefCheck {
                reference: ref_url,
                rel: r.rel,
                comparison,
                ok,
            });
        }
        let passed = checks.iter().all(|c| c.ok);
        TestOutcome::Reftest { passed, checks }
    }
}

/// Fold one test's outcome into a run's aggregate counts.
fn tally(summary: &mut RunSummary, outcome: &TestOutcome) {
    summary.total += 1;
    match outcome {
        TestOutcome::Harness(h) if h.is_green() => summary.harness_pass += 1,
        TestOutcome::Harness(_) => summary.harness_fail += 1,
        TestOutcome::Crashtest { passed: true } => summary.crashtest_pass += 1,
        TestOutcome::Crashtest { passed: false } => summary.crashtest_fail += 1,
        TestOutcome::Reftest { passed: true, .. } => summary.reftest_pass += 1,
        TestOutcome::Reftest { passed: false, .. } => summary.reftest_fail += 1,
        TestOutcome::Skipped { .. } => summary.skipped += 1,
        TestOutcome::DriverFailed(_) => summary.driver_failed += 1,
        TestOutcome::FilesystemFailure(_) => summary.filesystem_failures += 1,
    }
}

/// Resolve a reftest reference `href` against the test's URL.
///
/// Absolute-path hrefs (`/css/ref.html`) and absolute URLs (anything with
/// a scheme) are used as-is; anything else is resolved against the
/// directory of `test_url` the way a browser would resolve it,
/// normalizing `.` and `..` segments (`/dir/sub/test.html` + `../ref.html`
/// becomes `/dir/ref.html`). `..` past the root clamps at the root.
fn resolve_reference_url(test_url: &str, href: &str) -> String {
    // Absolute path, or an absolute URL (a scheme is present).
    if href.starts_with('/') || href.contains("://") {
        return href.to_owned();
    }
    if let Some(scheme_end) = href.find(':')
        && !href[..scheme_end].contains('/')
    {
        return href.to_owned();
    }
    let dir = test_url.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
    let mut segments: Vec<&str> = dir.split('/').filter(|s| !s.is_empty()).collect();
    for seg in href.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            s => segments.push(s),
        }
    }
    format!("/{}", segments.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::{StubDriver, Viewport};
    use crate::manifest::TestKind;
    use crate::reftest::Reference;
    use crate::results::{HarnessResult, HarnessStatus, SubtestResult, SubtestStatus};
    use std::cell::{Cell, RefCell};

    fn item(id: &str, kind: TestKind) -> TestItem {
        TestItem {
            id: id.into(),
            url: format!("/{id}"),
            kind,
            path: None,
            references: Vec::new(),
        }
    }

    /// A driver with canned answers, so the runner logic is testable without
    /// a browser.
    struct FakeDriver {
        report: HarnessResult,
        shot: Screenshot,
        rendered_urls: RefCell<Vec<String>>,
    }

    impl BrowserDriver for FakeDriver {
        fn run_testharness(
            &self,
            _test: &TestItem,
            _timeout: Duration,
        ) -> Result<HarnessResult, DriverError> {
            Ok(self.report.clone())
        }

        fn render(&self, url: &str, _viewport: Viewport) -> Result<Screenshot, DriverError> {
            self.rendered_urls.borrow_mut().push(url.to_owned());
            Ok(self.shot.clone())
        }

        fn name(&self) -> &str {
            "fake"
        }
    }

    fn fake_driver() -> FakeDriver {
        FakeDriver {
            report: green_report(),
            shot: Screenshot::solid(2, 2, 0, 0, 0, 255),
            rendered_urls: RefCell::new(Vec::new()),
        }
    }

    fn green_report() -> HarnessResult {
        HarnessResult {
            test: "x".into(),
            status: HarnessStatus::Ok,
            duration_ms: Some(1),
            subtests: vec![SubtestResult {
                name: "a".into(),
                status: SubtestStatus::Pass,
                message: None,
            }],
        }
    }

    /// Build a [`Manifest`] from items via the JSON parser, so tests exercise
    /// the same path as a real MANIFEST.json.
    fn manifest_of(items: Vec<TestItem>) -> Manifest {
        let mut by_kind: std::collections::HashMap<String, Vec<&TestItem>> =
            std::collections::HashMap::new();
        for t in &items {
            by_kind.entry(t.kind.to_string()).or_default().push(t);
        }
        let mut objs = Vec::new();
        for (kind, ts) in &by_kind {
            let es: Vec<String> = ts
                .iter()
                .map(|t| format!("\"{}\": [[null, {{}}]]", t.id))
                .collect();
            objs.push(format!("\"{kind}\": {{{}}}", es.join(",")));
        }
        let json = format!(
            "{{\"version\": 9, \"url_base\": \"/\", \"items\": {{{}}}}}",
            objs.join(",")
        );
        Manifest::from_json(&json).unwrap()
    }

    #[test]
    fn filters_include_exclude_kind() {
        let m = manifest_of(vec![
            item("dom/a.html", TestKind::Testharness),
            item("dom/b.html", TestKind::Testharness),
            item("css/c.html", TestKind::Reftest),
        ]);
        let driver = FakeDriver {
            report: green_report(),
            shot: Screenshot::solid(2, 2, 0, 0, 0, 255),
            rendered_urls: RefCell::new(Vec::new()),
        };
        // Reftest without a local path is skipped, not failed.
        let mut r = Runner::new(
            driver,
            RunConfig {
                include: vec!["dom/".into()],
                ..RunConfig::default()
            },
        );
        let report = r.run(&m);
        assert_eq!(report.summary.total, 2);
        assert_eq!(report.summary.harness_pass, 2);

        let driver = FakeDriver {
            report: green_report(),
            shot: Screenshot::solid(2, 2, 0, 0, 0, 255),
            rendered_urls: RefCell::new(Vec::new()),
        };
        let mut r = Runner::new(
            driver,
            RunConfig {
                exclude: vec!["b.html".into()],
                kinds: vec![TestKind::Testharness],
                ..RunConfig::default()
            },
        );
        let report = r.run(&m);
        assert_eq!(report.summary.total, 1);
        assert_eq!(report.reports[0].id, "dom/a.html");
    }

    #[test]
    fn manual_tests_are_skipped_with_reason() {
        let m = manifest_of(vec![item("html/m.html", TestKind::Manual)]);
        let mut r = Runner::new(
            FakeDriver {
                report: green_report(),
                shot: Screenshot::solid(1, 1, 0, 0, 0, 255),
                rendered_urls: RefCell::new(Vec::new()),
            },
            RunConfig::default(),
        );
        let report = r.run(&m);
        assert_eq!(report.summary.skipped, 1);
        match &report.reports[0].outcome {
            TestOutcome::Skipped { reason } => {
                assert!(reason.contains("manual"), "reason was: {reason}");
            }
            o => panic!("unexpected outcome: {o:?}"),
        }
    }

    #[test]
    fn stub_driver_reports_driver_failures() {
        let m = manifest_of(vec![item("dom/a.html", TestKind::Testharness)]);
        let mut r = Runner::new(StubDriver, RunConfig::default());
        let report = r.run(&m);
        assert_eq!(report.summary.driver_failed, 1);
        assert!(matches!(
            report.reports[0].outcome,
            TestOutcome::DriverFailed(DriverError::Unimplemented(_))
        ));
        let text = report.summary_text();
        assert!(text.contains("driver: stub"));
        assert!(text.contains("driver failures: 1"));
    }

    #[test]
    fn reftest_match_pass_and_mismatch_fail() {
        let dir = std::env::temp_dir().join(format!("bowser_wpt_run_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("t.html"),
            "<link rel=match href=\"same.html\">\n<link rel=mismatch href=\"diff.html\">",
        )
        .unwrap();
        std::fs::write(dir.join("same.html"), "x").unwrap();
        std::fs::write(dir.join("diff.html"), "y").unwrap();
        let test = TestItem {
            id: "t.html".into(),
            url: "/t.html".into(),
            kind: TestKind::Reftest,
            path: Some(dir.join("t.html")),
            references: Vec::new(),
        };
        // The path does not survive a manifest round-trip, so drive this
        // single test directly instead of going through Runner::run.
        let mut r = Runner::new(
            FakeDriver {
                report: green_report(),
                shot: Screenshot::solid(2, 2, 7, 7, 7, 255),
                rendered_urls: RefCell::new(Vec::new()),
            },
            RunConfig::default(),
        );
        let outcome = r.run_one(&test);
        match &outcome {
            TestOutcome::Reftest { passed, checks } => {
                assert!(!passed);
                assert_eq!(checks.len(), 2);
                assert!(checks[0].ok); // match, identical
                assert!(!checks[1].ok); // mismatch, identical -> bad
            }
            o => panic!("unexpected outcome: {o:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn reftest_runs_from_manifest_references_without_local_files() {
        // A manifest-sourced item: no local file, references parsed from
        // MANIFEST.json at manifest time.
        let test = TestItem {
            id: "css/t.html".into(),
            url: "/css/t.html".into(),
            kind: TestKind::Reftest,
            path: None,
            references: vec![
                Reference {
                    rel: RefRel::Match,
                    href: "same.html".into(),
                },
                Reference {
                    rel: RefRel::Mismatch,
                    href: "sub/diff.html".into(),
                },
            ],
        };
        let mut r = Runner::new(fake_driver(), RunConfig::default());
        let outcome = r.run_one(&test);
        match &outcome {
            TestOutcome::Reftest { passed, checks } => {
                assert_eq!(checks.len(), 2);
                assert!(checks[0].ok); // match, identical
                assert!(!checks[1].ok); // mismatch, identical -> bad
                assert!(!passed);
                // Reports carry the resolved reference URLs.
                assert_eq!(checks[0].reference, "/css/same.html");
                assert_eq!(checks[1].reference, "/css/sub/diff.html");
            }
            o => panic!("reftest from manifest references must run, got: {o:?}"),
        }
        // The driver saw the test URL plus both resolved reference URLs.
        assert_eq!(
            r.driver().rendered_urls.borrow().clone(),
            vec!["/css/t.html", "/css/same.html", "/css/sub/diff.html"]
        );
    }

    #[test]
    fn reftest_without_references_or_file_is_skipped() {
        let test = TestItem {
            id: "css/u.html".into(),
            url: "/css/u.html".into(),
            kind: TestKind::Reftest,
            path: None,
            references: Vec::new(),
        };
        let mut r = Runner::new(fake_driver(), RunConfig::default());
        let outcome = r.run_one(&test);
        assert!(matches!(outcome, TestOutcome::Skipped { .. }));
    }

    #[test]
    fn reference_url_resolution() {
        // Relative hrefs resolve against the test's directory.
        assert_eq!(
            resolve_reference_url("/dir/test.html", "ref.html"),
            "/dir/ref.html"
        );
        assert_eq!(
            resolve_reference_url("/dir/test.html", "sub/ref.html"),
            "/dir/sub/ref.html"
        );
        // Dot segments normalize.
        assert_eq!(
            resolve_reference_url("/dir/sub/test.html", "../ref.html"),
            "/dir/ref.html"
        );
        assert_eq!(
            resolve_reference_url("/dir/test.html", "./ref.html"),
            "/dir/ref.html"
        );
        // The test URL already carries url_base; it is preserved.
        assert_eq!(
            resolve_reference_url("/_wpt_/dir/test.html", "ref.html"),
            "/_wpt_/dir/ref.html"
        );
        // Absolute paths and full URLs pass through untouched.
        assert_eq!(
            resolve_reference_url("/dir/test.html", "/abs/ref.html"),
            "/abs/ref.html"
        );
        assert_eq!(
            resolve_reference_url("/dir/test.html", "https://example.com/ref.html"),
            "https://example.com/ref.html"
        );
    }

    /// A driver that records which entry point the runner used, so tests can
    /// prove crashtests never wait on the testharness completion callback.
    struct CrashDriver {
        /// What `load_without_crash` returns; `None` means `Ok(())`.
        crash: RefCell<Option<DriverError>>,
        load_calls: RefCell<Vec<String>>,
        harness_calls: Cell<usize>,
    }

    impl BrowserDriver for CrashDriver {
        fn run_testharness(
            &self,
            _test: &TestItem,
            _timeout: Duration,
        ) -> Result<HarnessResult, DriverError> {
            self.harness_calls.set(self.harness_calls.get() + 1);
            panic!("crashtests must not wait for a testharness.js completion callback");
        }

        fn load_without_crash(&self, url: &str) -> Result<(), DriverError> {
            self.load_calls.borrow_mut().push(url.to_owned());
            match self.crash.take() {
                Some(e) => Err(e),
                None => Ok(()),
            }
        }

        fn render(&self, _url: &str, _viewport: Viewport) -> Result<Screenshot, DriverError> {
            unimplemented!("crashtest path never renders")
        }

        fn name(&self) -> &str {
            "crash-fake"
        }
    }

    #[test]
    fn crashtest_passes_without_harness_callback() {
        let mut r = Runner::new(
            CrashDriver {
                crash: RefCell::new(None),
                load_calls: RefCell::new(Vec::new()),
                harness_calls: Cell::new(0),
            },
            RunConfig::default(),
        );
        let outcome = r.run_one(&item("crash/x.html", TestKind::Crashtest));
        assert!(matches!(outcome, TestOutcome::Crashtest { passed: true }));
        assert!(outcome.passed());
        assert_eq!(r.driver().harness_calls.get(), 0);
        assert_eq!(
            r.driver().load_calls.borrow().clone(),
            vec!["/crash/x.html".to_owned()]
        );
    }

    #[test]
    fn crashtest_browser_crash_is_a_test_failure() {
        let mut r = Runner::new(
            CrashDriver {
                crash: RefCell::new(Some(DriverError::BrowserGone("segfault".into()))),
                load_calls: RefCell::new(Vec::new()),
                harness_calls: Cell::new(0),
            },
            RunConfig::default(),
        );
        let m = manifest_of(vec![item("crash/y.html", TestKind::Crashtest)]);
        let report = r.run(&m);
        assert_eq!(report.summary.crashtest_fail, 1);
        assert_eq!(report.summary.crashtest_pass, 0);
        assert_eq!(report.summary.driver_failed, 0);
        assert!(matches!(
            report.reports[0].outcome,
            TestOutcome::Crashtest { passed: false }
        ));
        assert!(!report.reports[0].outcome.passed());
    }

    #[test]
    fn crashtest_driver_errors_are_driver_failures() {
        let mut r = Runner::new(
            CrashDriver {
                crash: RefCell::new(Some(DriverError::Timeout {
                    test: "crash/z.html".into(),
                    after: Duration::from_secs(1),
                })),
                load_calls: RefCell::new(Vec::new()),
                harness_calls: Cell::new(0),
            },
            RunConfig::default(),
        );
        let outcome = r.run_one(&item("crash/z.html", TestKind::Crashtest));
        assert!(matches!(outcome, TestOutcome::DriverFailed(_)));
    }

    #[test]
    fn unreadable_reftest_file_is_filesystem_failure() {
        // A missing reftest file is harness-side I/O: it must be reported
        // as a filesystem failure, never as a driver report failure.
        let dir =
            std::env::temp_dir().join(format!("bowser_wpt_run_missing_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let test = TestItem {
            id: "t.html".into(),
            url: "/t.html".into(),
            kind: TestKind::Reftest,
            path: Some(dir.join("t.html")),
            references: Vec::new(),
        };
        let mut r = Runner::new(
            FakeDriver {
                report: green_report(),
                shot: Screenshot::solid(1, 1, 0, 0, 0, 255),
                rendered_urls: RefCell::new(Vec::new()),
            },
            RunConfig::default(),
        );
        let outcome = r.run_one(&test);
        assert!(
            matches!(outcome, TestOutcome::FilesystemFailure(_)),
            "unreadable reftest file must be FilesystemFailure, got: {outcome:?}"
        );
        assert!(!outcome.passed());
    }

    #[test]
    fn filesystem_failures_are_tallied() {
        let mut summary = RunSummary::default();
        tally(
            &mut summary,
            &TestOutcome::FilesystemFailure("no such file".into()),
        );
        assert_eq!(summary.total, 1);
        assert_eq!(summary.filesystem_failures, 1);
        assert_eq!(summary.driver_failed, 0);
    }
}
