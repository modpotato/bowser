//! Running a manifest against a [`BrowserDriver`](crate::driver::BrowserDriver).
//!
//! [`Runner`] filters a [`Manifest`](crate::manifest::Manifest), hands each
//! test to the driver, and collects a [`RunReport`]. Testharness tests go
//! through [`BrowserDriver::run_testharness`](crate::driver::BrowserDriver::run_testharness);
//! reftests are rendered alongside each of their references and compared with
//! [`compare`](crate::reftest::compare).

use std::time::Duration;

use crate::driver::{BrowserDriver, DriverError, Screenshot, Viewport};
use crate::manifest::{Manifest, TestItem, TestKind};
use crate::reftest::{self, Comparison, RefRel, Reference, ReftestTolerance};
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
    /// Not attempted; `reason` says why (manual test, no references, ...).
    Skipped { reason: String },
    /// The driver itself failed (timeout, crashed browser, unimplemented).
    DriverFailed(DriverError),
}

impl TestOutcome {
    /// Did the test pass?
    pub fn passed(&self) -> bool {
        match self {
            TestOutcome::Harness(r) => r.is_green(),
            TestOutcome::Reftest { passed, .. } => *passed,
            TestOutcome::Skipped { .. } | TestOutcome::DriverFailed(_) => false,
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
    pub reftest_pass: usize,
    pub reftest_fail: usize,
    pub skipped: usize,
    pub driver_failed: usize,
}

impl RunSummary {
    pub fn passed(&self) -> usize {
        self.harness_pass + self.reftest_pass
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
             reftest: {} passed, {} failed\n\
             skipped: {}\n\
             driver failures: {}",
            self.driver_name,
            s.total,
            s.harness_pass,
            s.harness_fail,
            s.reftest_pass,
            s.reftest_fail,
            s.skipped,
            s.driver_failed,
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
            summary.total += 1;
            match &r.outcome {
                TestOutcome::Harness(h) if h.is_green() => summary.harness_pass += 1,
                TestOutcome::Harness(_) => summary.harness_fail += 1,
                TestOutcome::Reftest { passed: true, .. } => summary.reftest_pass += 1,
                TestOutcome::Reftest { passed: false, .. } => summary.reftest_fail += 1,
                TestOutcome::Skipped { .. } => summary.skipped += 1,
                TestOutcome::DriverFailed(_) => summary.driver_failed += 1,
            }
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
            TestKind::Testharness | TestKind::Crashtest => {
                match self.driver.run_testharness(item, self.config.timeout) {
                    Ok(result) => TestOutcome::Harness(result),
                    Err(e) => TestOutcome::DriverFailed(e),
                }
            }
            TestKind::Reftest | TestKind::PrintReftest => self.run_reftest(item),
            _ => TestOutcome::Skipped {
                reason: format!("no runner for {}", item.kind),
            },
        }
    }

    fn run_reftest(&mut self, item: &TestItem) -> TestOutcome {
        // Prefer the reference list parsed at manifest time. Items that
        // carry a local file but no parsed references (walk output,
        // hand-built items) fall back to scraping the file for
        // <link rel=match|mismatch> tags.
        let scraped;
        let refs: &[Reference] = if item.references.is_empty() {
            match &item.path {
                Some(path) => match reftest::extract_references(path) {
                    Ok(r) => {
                        scraped = r;
                        &scraped
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

/// Resolve a reftest reference `href` against the test's served URL.
///
/// `item.url` already includes the manifest's `url_base`, so resolving
/// against it covers both. Absolute URLs and absolute paths pass through
/// untouched; relative hrefs resolve against the test URL's directory,
/// with `.`/`..` segments normalized.
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
    use crate::results::{HarnessResult, HarnessStatus, SubtestResult, SubtestStatus};

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
    /// a browser. Records every URL it was asked to render.
    struct FakeDriver {
        report: HarnessResult,
        shot: Screenshot,
        rendered_urls: Vec<String>,
    }

    impl BrowserDriver for FakeDriver {
        fn run_testharness(
            &mut self,
            _test: &TestItem,
            _timeout: Duration,
        ) -> Result<HarnessResult, DriverError> {
            Ok(self.report.clone())
        }

        fn render(&mut self, url: &str, _viewport: Viewport) -> Result<Screenshot, DriverError> {
            self.rendered_urls.push(url.to_owned());
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
            rendered_urls: Vec::new(),
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
                .map(|t| format!("\"{}\": [\"item\", [\"/{}\", {{}}]]", t.id, t.id))
                .collect();
            objs.push(format!("\"{kind}\": {{{}}}", es.join(",")));
        }
        let json = format!("{{\"items\": {{{}}}}}", objs.join(","));
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
            rendered_urls: Vec::new(),
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
            rendered_urls: Vec::new(),
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
    fn manual_tests_are_skipped() {
        let m = manifest_of(vec![item("html/m.html", TestKind::Manual)]);
        let mut r = Runner::new(
            FakeDriver {
                report: green_report(),
                shot: Screenshot::solid(1, 1, 0, 0, 0, 255),
                rendered_urls: Vec::new(),
            },
            RunConfig::default(),
        );
        let report = r.run(&m);
        assert_eq!(report.summary.skipped, 1);
        assert!(matches!(
            report.reports[0].outcome,
            TestOutcome::Skipped { .. }
        ));
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
                rendered_urls: Vec::new(),
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
            r.driver().rendered_urls,
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
}
