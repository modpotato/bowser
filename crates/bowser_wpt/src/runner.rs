//! Running a manifest against a [`BrowserDriver`](crate::driver::BrowserDriver).
//!
//! [`Runner`] filters a [`Manifest`](crate::manifest::Manifest), hands each
//! test to the driver, and collects a [`RunReport`]. Testharness tests go
//! through [`BrowserDriver::run_testharness`](crate::driver::BrowserDriver::run_testharness);
//! reftests are rendered alongside each of their references and compared with
//! [`compare`](crate::reftest::compare); crashtests just load the page and
//! pass when the browser survives
//! ([`BrowserDriver::load_without_crash`](crate::driver::BrowserDriver::load_without_crash)).
//!
//! Every driver call runs on a worker thread bounded by
//! [`RunConfig::timeout`]: a misbehaving driver that never returns cannot
//! stall the run. Expiry is reported as [`DriverError::Timeout`]; the
//! abandoned worker thread is left running detached (safe Rust cannot kill
//! it) and its late result is discarded.
//!
//! A timeout can also mean the driver itself is wedged: the abandoned
//! worker may still be inside a driver call holding a driver-internal
//! lock, in which case every subsequent test would time out too. After a
//! timeout the runner therefore probes the driver with the cheap
//! [`BrowserDriver::name`] call bounded by [`RunConfig::probe_timeout`];
//! if the probe does not answer, the driver is retired — no further tests
//! are scheduled against it, the remaining tests are marked skipped (never
//! attempted), and the run is flagged aborted in its summary. A wedge is
//! reported as [`DriverError::DriverWedged`], never misattributed as
//! independent test timeouts.
//!
//! Abandoned threads are counted, and once [`MAX_ABANDONED_DRIVERS`] have
//! been abandoned the run aborts with [`DriverError::TooManyTimeouts`]
//! instead of leaking threads without bound. After any timeout the driver
//! is permanently borrowed by the abandoned worker, so
//! [`Runner::driver_mut`] can never succeed again: that is
//! unusable-by-design, not a bug to work around.
//!
//! The runner composes with multiplexing drivers — drivers that serve
//! many operations over one long-lived child process behind an internal
//! mutex (see `bowser_app::wpt_driver::SessionDriver`) — without any
//! special-casing: each driver call still runs on its own worker thread
//! under the same watchdog, and the calls simply serialize on the
//! driver's lock. A multiplexing driver may also enforce its own inner
//! per-operation timeouts; those compose with the runner's outer bound,
//! and whichever fires first reports the timeout.

use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;

use crate::driver::{BrowserDriver, DriverError, Screenshot, Viewport};
use crate::manifest::{Manifest, TestExtras, TestItem, TestKind};
use crate::reftest::{self, Comparison, RefRel, ReftestTolerance};
use crate::results::{
    ExpectedEntry, ExpectedMetadata, ExpectedStatus, HarnessResult, HarnessStatus, SubtestResult,
    SubtestStatus,
};

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
    /// Per-test timeout, enforced by the runner on a worker thread around
    /// every driver call (testharness runs, crashtest loads, reftest
    /// renders). A driver that never returns is reported as
    /// [`DriverError::Timeout`] instead of stalling the run.
    pub timeout: Duration,
    /// Bound on the driver health probe after a timeout (see the module
    /// docs) and on the final [`BrowserDriver::name`] call when the report
    /// is built. `name()` is a pure metadata getter and should answer
    /// instantly; this only needs to be long enough to rule out scheduling
    /// jitter.
    pub probe_timeout: Duration,
    /// Viewport for reftest rendering. A test's manifest
    /// `"viewport_size"` extra overrides this per test (see
    /// [`TestExtras::viewport`](crate::manifest::TestExtras::viewport)).
    pub viewport: Viewport,
    /// Screenshot comparison tolerance for reftests. A test's manifest
    /// `"fuzzy"` extra overrides this per test (see
    /// [`TestExtras::tolerance`](crate::manifest::TestExtras::tolerance)).
    pub tolerance: ReftestTolerance,
    /// Base URL that manifest-relative test URLs are joined against
    /// before they reach the driver. Drivers hand the URL to the browser
    /// verbatim (`bowser --headless <url>`), and a manifest-relative URL
    /// like `/css/css-tables/x.html` makes the browser exit with "missing
    /// scheme". Set this to the fixture server's base URL, e.g.
    /// `http://127.0.0.1:41237`. Empty (the default) means "hand URLs to
    /// the driver untouched", which keeps drivers that join internally
    /// working: [`BrowserDriver::run_testharness`] takes the whole
    /// [`TestItem`](crate::manifest::TestItem), so its joining is the
    /// driver's business, and the runner's report-attribution check
    /// compares the report's url against the item's manifest url.
    pub server_base_url: String,
    /// Expected-failure metadata: tests whose actual outcome matches
    /// their expectation are reported as expected-fail rather than
    /// failures. Empty by default.
    pub metadata: ExpectedMetadata,
}

impl Default for RunConfig {
    fn default() -> Self {
        RunConfig {
            include: Vec::new(),
            exclude: Vec::new(),
            kinds: Vec::new(),
            timeout: Duration::from_secs(60),
            probe_timeout: Duration::from_secs(5),
            viewport: Viewport::DEFAULT,
            tolerance: ReftestTolerance::EXACT,
            metadata: ExpectedMetadata::default(),
            server_base_url: String::new(),
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
    /// The runner-side watchdog expired before the driver produced a
    /// report: the driver call never returned within
    /// [`RunConfig::timeout`], so no `BOWSER_HARNESS_REPORT` arrived.
    /// Deliberately distinct from
    /// [`HarnessStatus::Timeout`](crate::results::HarnessStatus::Timeout),
    /// which is testharness.js reporting its *own* subtest timeout while
    /// the driver still answered — conflating the two made the encoding
    /// suite's harness-reported TIMEOUTs unreadable against real hangs.
    /// Still a test outcome, never a driver failure: the post-timeout
    /// health probe already retires a genuinely wedged driver.
    DriverTimeout {
        id: String,
        url: String,
        after: Duration,
    },
    /// A reftest rendered and compared against its references.
    Reftest { passed: bool, checks: Vec<RefCheck> },
    /// A crashtest: `passed` is false when loading the page killed the
    /// browser. Never waits on a testharness completion callback.
    Crashtest { passed: bool },
    /// Not attempted; `reason` says why (manual test, no references, ...).
    Skipped { reason: String },
    /// The driver itself failed (crashed browser, unimplemented call,
    /// wedged driver, abandonment-cap trip). Watchdog timeouts are *test*
    /// timeouts — [`TestOutcome::DriverTimeout`] for testharness — never
    /// driver failures.
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
            | TestOutcome::FilesystemFailure(_)
            | TestOutcome::DriverTimeout { .. } => false,
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
    /// The outcome matched the test's expected-failure metadata (see
    /// [`is_expected_failure`]): reported as expected-fail rather than a
    /// failure in the summary.
    pub expected_fail: bool,
}

/// Aggregate counts for a run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunSummary {
    pub total: usize,
    pub harness_pass: usize,
    pub harness_fail: usize,
    /// Testharness tests where the driver reported the browser died
    /// ([`DriverError::BrowserGone`]).
    pub harness_crash: usize,
    /// Testharness tests that loaded but produced no subtest data: the v1
    /// driver's "loaded, subtests unknown"
    /// ([`HarnessStatus::SubtestsUnknown`](crate::results::HarnessStatus::SubtestsUnknown)).
    /// Neither a pass nor a failure.
    pub harness_unknown: usize,
    /// Testharness tests where the runner-side watchdog expired before
    /// the driver produced a report
    /// ([`TestOutcome::DriverTimeout`]). Never conflated with
    /// harness-reported `TIMEOUT`s, which tally under `harness_fail`.
    pub driver_timeouts: usize,
    pub crashtest_pass: usize,
    pub crashtest_fail: usize,
    pub reftest_pass: usize,
    pub reftest_fail: usize,
    /// Tests whose outcome matched their expected-failure metadata (see
    /// [`is_expected_failure`]). Counted here instead of in the failure
    /// buckets above.
    pub expected_fail: usize,
    pub skipped: usize,
    pub driver_failed: usize,
    /// Harness-side I/O failures (unreadable test or reference files).
    pub filesystem_failures: usize,
    /// The run stopped handing tests to the driver before finishing: a
    /// driver wedge was detected or the abandonment cap was reached. The
    /// tests never attempted are reported as skipped, so they also count
    /// toward `skipped`; `unattempted` counts exactly those.
    pub aborted: bool,
    /// Tests never handed to the driver because the run aborted. A subset
    /// of `skipped`.
    pub unattempted: usize,
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
    /// Wall-clock epoch millis when [`Runner::run`] started the test loop.
    /// Exported as wptreport `time_start` (see `wptreport.py`'s formatter,
    /// which always emits both timestamps).
    pub time_start_ms: u64,
    /// Wall-clock epoch millis when [`Runner::run`] finished the test loop.
    /// Exported as wptreport `time_end`.
    pub time_end_ms: u64,
}

/// Wall-clock time as milliseconds since the Unix epoch, for wptreport
/// `time_start`/`time_end`.
fn epoch_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

impl RunReport {
    /// One-line-per-section human summary.
    pub fn summary_text(&self) -> String {
        let s = &self.summary;
        let mut text = format!(
            "driver: {}\n\
             total: {}\n\
             testharness: {} passed, {} failed\n\
             testharness crashes: {}\n\
             testharness unknown (loaded, no subtests observed): {}\n\
             watchdog timeouts (driver produced no report): {}\n\
             crashtest: {} passed, {} failed\n\
             reftest: {} passed, {} failed\n\
             expected failures: {}\n\
             skipped: {}\n\
             driver failures: {}\n\
             filesystem failures: {}",
            self.driver_name,
            s.total,
            s.harness_pass,
            s.harness_fail,
            s.harness_crash,
            s.harness_unknown,
            s.driver_timeouts,
            s.crashtest_pass,
            s.crashtest_fail,
            s.reftest_pass,
            s.reftest_fail,
            s.expected_fail,
            s.skipped,
            s.driver_failed,
            s.filesystem_failures,
        );
        // An aborted run must not read as a complete one: say so, and say
        // how many tests never got attempted.
        if s.aborted {
            text.push_str(&format!(
                "\naborted: true ({} test{} never attempted)",
                s.unattempted,
                if s.unattempted == 1 { "" } else { "s" }
            ));
        }
        text
    }
}

/// Runs a manifest against a driver.
///
/// The driver lives behind an `Arc` so each driver call can run on a
/// worker thread (see [`RunConfig::timeout`]) while the runner itself
/// stays sequential.
pub struct Runner<D> {
    driver: Arc<D>,
    config: RunConfig,
    /// Worker threads abandoned to driver timeouts so far this run. Each
    /// one holds an `Arc` clone of the driver until its call returns, so
    /// [`Runner::driver_mut`] can never succeed again once this is
    /// non-zero. Once [`MAX_ABANDONED_DRIVERS`] calls have timed out, the
    /// run aborts instead of leaking threads without bound.
    abandoned: usize,
    /// The driver failed its health probe after a timeout and has been
    /// retired: no further tests are scheduled against it, and
    /// [`Runner::call_with_timeout`] refuses new calls. Set once, never
    /// cleared.
    wedged: bool,
}

/// Cap on abandoned (timed-out) driver worker threads per run. A timeout
/// leaks one thread plus one `Arc<D>` clone — safe Rust cannot kill the
/// worker — so an unbounded run of timeouts would exhaust threads; once
/// this many driver calls have timed out, the run aborts with
/// [`DriverError::TooManyTimeouts`].
pub const MAX_ABANDONED_DRIVERS: usize = 32;

/// Factor by which [`RunConfig::timeout`] is scaled for tests whose
/// manifest extras say `"timeout": "long"` (see
/// [`effective_timeout`](Runner::effective_timeout)). Bowser's choice,
/// not an upstream constant: long enough for the slow layout/paint
/// tests that carry the marker without making a hung driver wait
/// forever.
pub const LONG_TIMEOUT_FACTOR: u32 = 10;

/// Whether `outcome` matches the metadata expectation `expected`: the
/// test failed the way the metadata said it would, so the summary
/// reports it as expected-fail instead of a failure.
///
/// Matching is deliberately narrow:
/// - `FAIL` matches a test that ran and did not pass: a harness result
///   that is not green with a finished status (`OK` with failing
///   subtests, or top-level `FAIL`), or a failed reftest/crashtest.
///   Timeouts, harness errors, crashes, "subtests unknown", and genuinely
///   unknown statuses do not match `FAIL` — timeouts and errors have their
///   own expectations, a crash is never "just" a failure, and an
///   indeterminate status is never "failed as expected".
/// - `TIMEOUT` matches a harness-reported
///   [`HarnessStatus::Timeout`] or a runner watchdog expiry
///   ([`TestOutcome::DriverTimeout`]): either way the test timed out, so
///   `expected: TIMEOUT` metadata keeps working for both.
/// - `ERROR` matches only [`HarnessStatus::Error`].
///
/// Skips, driver failures, and filesystem failures never match: those
/// are infrastructure outcomes, not test results. (An unexpected pass —
/// a test that passes despite an expectation — is not flagged; that is
/// future work, not this wave.)
pub fn is_expected_failure(outcome: &TestOutcome, expected: &ExpectedEntry) -> bool {
    match expected.expected {
        ExpectedStatus::Fail => match outcome {
            TestOutcome::Harness(h) => {
                // Genuinely unknown statuses (`HarnessStatus::Unknown`)
                // are indeterminate: never matched as expected-FAIL (or
                // TIMEOUT/ERROR), whatever the subtests say.
                !h.is_green() && matches!(h.status, HarnessStatus::Ok | HarnessStatus::Fail)
            }
            TestOutcome::Reftest { passed, .. } | TestOutcome::Crashtest { passed } => !passed,
            _ => false,
        },
        ExpectedStatus::Timeout => {
            matches!(outcome, TestOutcome::Harness(h) if h.status == HarnessStatus::Timeout)
                || matches!(outcome, TestOutcome::DriverTimeout { .. })
        }
        ExpectedStatus::Error => {
            matches!(outcome, TestOutcome::Harness(h) if h.status == HarnessStatus::Error)
        }
    }
}

/// Run `op` on a worker thread with a timeout, without touching run state.
///
/// For driver calls made outside the test loop — the health probe, the
/// final `name()` — where counting an abandoned worker toward the cap or
/// probing for a wedge would be wrong. Expiry abandons the worker the
/// same way [`Runner::call_with_timeout`] does, but silently.
fn detached_call<R>(op: impl FnOnce() -> R + Send + 'static, timeout: Duration) -> Option<R>
where
    R: Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(op());
    });
    rx.recv_timeout(timeout).ok()
}

impl<D: BrowserDriver> Runner<D> {
    pub fn new(driver: D, config: RunConfig) -> Runner<D> {
        Runner {
            driver: Arc::new(driver),
            config,
            abandoned: 0,
            wedged: false,
        }
    }

    pub fn driver(&self) -> &D {
        &self.driver
    }

    /// How many driver worker threads have been abandoned to timeouts so
    /// far. Non-zero means a timed-out worker may still be inside a driver
    /// call, holding an `Arc` clone of the driver.
    pub fn abandoned_drivers(&self) -> usize {
        self.abandoned
    }

    /// Mutable access to the driver. Only succeeds while no driver call
    /// has ever timed out: a timed-out worker thread keeps its `Arc`
    /// clone until its call returns (safe Rust cannot kill it), so the
    /// `Arc` is never uniquely owned again and handing out `&mut` would
    /// race with the abandoned worker still inside a driver call.
    ///
    /// By design, then, the driver is unusable for mutation once any call
    /// has timed out — finish or drop the run instead of working around
    /// this. (Panics rather than returning `Result` because the pre-
    /// timeout API promises `&mut D`; changing the signature would ripple
    /// through every caller for a state that is always a programming
    /// error to reach for.)
    pub fn driver_mut(&mut self) -> &mut D {
        Arc::get_mut(&mut self.driver).expect(
            "driver is still borrowed by a timed-out worker thread: after a \
             timeout the runner can never hand out &mut again, because the \
             abandoned worker may still be inside a driver call and mutating \
             the driver from here would race with it. By design the driver \
             is unusable for mutation once any call has timed out; finish \
             or drop the run instead",
        )
    }
}

impl<D: BrowserDriver + Send + Sync + 'static> Runner<D> {
    /// Whether the run has aborted: no more tests may be handed to the
    /// driver, either because a wedge retired it or because the
    /// abandonment cap was reached.
    fn run_aborted(&self) -> bool {
        self.wedged || self.abandoned >= MAX_ABANDONED_DRIVERS
    }

    /// Why the run aborted, for the skip reason on never-attempted tests.
    fn abort_reason(&self) -> String {
        if self.wedged {
            "run aborted: driver wedge detected (driver stopped answering); \
             test never attempted"
                .into()
        } else {
            format!(
                "run aborted: {} driver timeouts reached the cap of {}; \
                 test never attempted",
                self.abandoned, MAX_ABANDONED_DRIVERS
            )
        }
    }

    /// Probe whether the driver still responds after a timed-out call.
    ///
    /// Runs the cheapest driver call ([`BrowserDriver::name`]) on a worker
    /// thread bounded by [`RunConfig::probe_timeout`]. If it answers, the
    /// timeout was (presumably) that one test's problem and the run
    /// continues. If it does not answer, the driver is wedged — the
    /// abandoned worker is likely still inside a driver call holding a
    /// driver-internal lock — and the driver is retired on the spot.
    ///
    /// The probe is a heuristic, not a proof: a driver whose `name()`
    /// never touches a lock can answer while still wedged on real calls.
    /// What it does guarantee is the common case: a driver deadlocked on
    /// every call is caught after one probe timeout instead of one full
    /// test timeout per remaining test.
    ///
    /// An expired probe abandons its worker the same way a timed-out test
    /// call does, but the worker is not counted toward
    /// [`MAX_ABANDONED_DRIVERS`]: the driver is retired either way, so no
    /// further tests are scheduled against it.
    fn probe_driver_health(&self) -> bool {
        let driver = Arc::clone(&self.driver);
        detached_call(move || driver.name().to_owned(), self.config.probe_timeout).is_some()
    }

    /// Driver name for the report, under the same watchdog philosophy as
    /// every other driver call: a hung `name()` yields a fallback name
    /// instead of hanging report construction forever. Best-effort
    /// metadata after the run: it never counts toward the abandonment cap
    /// and never probes for a wedge.
    fn driver_display_name(&self) -> String {
        let driver = Arc::clone(&self.driver);
        detached_call(move || driver.name().to_owned(), self.config.probe_timeout)
            .unwrap_or_else(|| "unknown (driver did not answer name())".into())
    }

    /// Run `op` on a worker thread, enforcing `timeout`.
    ///
    /// The [`BrowserDriver`] contract asks implementations to respect
    /// their timeout, but nothing bounds a misbehaving driver, so the
    /// runner enforces it itself: if `op` is still running after
    /// `timeout`, [`DriverError::Timeout`] is returned for `label` and
    /// the run continues with the next test. The worker thread is left
    /// running detached (it cannot be killed from safe Rust); its result
    /// is discarded when — and if — it ever arrives. Each abandoned
    /// worker holds an `Arc` clone of the driver, which is also what
    /// permanently retires [`Runner::driver_mut`]; the count is visible
    /// via [`Runner::abandoned_drivers`].
    ///
    /// A timeout may mean the driver itself is wedged rather than just
    /// this test being slow, so before returning, the runner probes the
    /// driver with the cheap [`BrowserDriver::name`] call (see
    /// [`Runner::probe_driver_health`]): if the probe does not answer,
    /// the driver is retired and [`DriverError::DriverWedged`] is
    /// returned — a driver malfunction, never misattributed as an
    /// independent test timeout.
    ///
    /// Once [`MAX_ABANDONED_DRIVERS`] workers have been abandoned, the run
    /// aborts with [`DriverError::TooManyTimeouts`] instead of leaking
    /// threads without bound.
    fn call_with_timeout<R>(
        &mut self,
        label: &str,
        timeout: Duration,
        op: impl FnOnce() -> Result<R, DriverError> + Send + 'static,
    ) -> Result<R, DriverError>
    where
        R: Send + 'static,
    {
        // A retired driver is never scheduled against: the wedge was
        // already reported, and burning another timeout here would
        // contradict the retirement.
        if self.wedged {
            return Err(DriverError::DriverWedged {
                test: label.to_owned(),
            });
        }
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let _ = tx.send(op());
        });
        match rx.recv_timeout(timeout) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => {
                self.abandoned += 1;
                // The abandoned worker may still be inside the driver call
                // holding a driver-internal lock, which would wedge every
                // subsequent test. Probe before scheduling anything else.
                if !self.probe_driver_health() {
                    self.wedged = true;
                    return Err(DriverError::DriverWedged {
                        test: label.to_owned(),
                    });
                }
                if self.abandoned >= MAX_ABANDONED_DRIVERS {
                    Err(DriverError::TooManyTimeouts {
                        abandoned: self.abandoned,
                    })
                } else {
                    Err(DriverError::Timeout {
                        test: label.to_owned(),
                        after: timeout,
                    })
                }
            }
            Err(RecvTimeoutError::Disconnected) => Err(DriverError::BadReport(
                "driver worker thread died without producing a result".into(),
            )),
        }
    }

    /// Run every selected test in manifest order.
    ///
    /// Stops handing tests to the driver once the run aborts — a driver
    /// wedge ([`DriverError::DriverWedged`]) or [`MAX_ABANDONED_DRIVERS`]
    /// timed-out calls: the test that trips the abort is reported with the
    /// corresponding error, and every remaining test is marked skipped
    /// (never attempted) instead of burning a timeout each. The summary
    /// records the abort (`aborted`, `unattempted`) so an aborted run
    /// cannot be mistaken for a complete one.
    pub fn run(&mut self, manifest: &Manifest) -> RunReport {
        let time_start_ms = epoch_millis();
        let mut reports = Vec::new();
        let mut aborted = false;
        let mut unattempted = 0;
        for item in manifest.items() {
            if !self.config.wants(item) {
                continue;
            }
            let timeout = self.effective_timeout(manifest, item);
            let extras = manifest.extras_for(&item.id);
            let outcome = if self.run_aborted() {
                aborted = true;
                unattempted += 1;
                TestOutcome::Skipped {
                    reason: self.abort_reason(),
                }
            } else {
                self.run_one(item, timeout, extras)
            };
            let expected_fail = self
                .config
                .metadata
                .expectation_for(&item.id)
                .is_some_and(|e| is_expected_failure(&outcome, e));
            reports.push(TestReport {
                id: item.id.clone(),
                kind: item.kind.clone(),
                outcome,
                expected_fail,
            });
        }
        let mut summary = RunSummary::default();
        for r in &reports {
            tally(&mut summary, &r.outcome, r.expected_fail);
        }
        summary.aborted = aborted;
        summary.unattempted = unattempted;
        let time_end_ms = epoch_millis();
        RunReport {
            reports,
            summary,
            driver_name: self.driver_display_name(),
            time_start_ms,
            time_end_ms,
        }
    }

    /// The watchdog bound for one test: [`RunConfig::timeout`], scaled by
    /// [`LONG_TIMEOUT_FACTOR`] when the manifest marks the test
    /// `"timeout": "long"`. The scaled bound applies to every driver call
    /// for the test (the harness run, the crashtest load, each reftest
    /// render).
    fn effective_timeout(&self, manifest: &Manifest, item: &TestItem) -> Duration {
        if manifest.extras_for(&item.id).timeout_long {
            self.config.timeout.saturating_mul(LONG_TIMEOUT_FACTOR)
        } else {
            self.config.timeout
        }
    }

    /// Run one test: the manifest extras drive the per-test timeout
    /// (computed by the caller via
    /// [`effective_timeout`](Runner::effective_timeout)), the reftest
    /// viewport (`viewport_size`), and the reftest comparison tolerance
    /// (`fuzzy`).
    pub(crate) fn run_one(
        &mut self,
        item: &TestItem,
        timeout: Duration,
        extras: TestExtras,
    ) -> TestOutcome {
        if !item.kind.is_automated() {
            return TestOutcome::Skipped {
                reason: format!("{} tests need a human", item.kind),
            };
        }
        match item.kind {
            TestKind::Testharness => self.run_testharness(item, timeout),
            // Crashtests never wait on a testharness completion callback:
            // they pass when the page loads without killing the browser.
            TestKind::Crashtest => self.run_crashtest(item, timeout),
            TestKind::Reftest | TestKind::PrintReftest => self.run_reftest(item, timeout, extras),
            _ => TestOutcome::Skipped {
                reason: format!("no runner for {}", item.kind),
            },
        }
    }

    /// Run one testharness test under the runner-side watchdog.
    ///
    /// Besides bounding the driver call with [`RunConfig::timeout`], this
    /// verifies report attribution: a report whose (`test`, `url`) pair is
    /// not the test that ran is rejected as a harness failure — with the
    /// mismatch spelled out — instead of silently poisoning the scoreboard.
    /// The url participates because `.any.js` multi-global variants share
    /// one id with different urls; the id alone is not enough.
    /// `PRECONDITION_FAILED` maps to a skip like upstream WPT (see
    /// [`classify_harness_result`], shared with the wptreport import path);
    /// a dead browser maps to a crash result, not a driver failure. A
    /// watchdog expiry maps to [`TestOutcome::DriverTimeout`]: the driver
    /// produced no report at all, which is not the same as testharness.js
    /// reporting its own subtest `TIMEOUT` (that stays
    /// [`TestOutcome::Harness`] with [`HarnessStatus::Timeout`](crate::results::HarnessStatus::Timeout)).
    /// A hung test is still a test timeout, not a driver failure — the
    /// health probe already retires a genuinely wedged driver, so a plain
    /// [`DriverError::Timeout`] is the test's own problem — and
    /// `is_expected_failure`'s TIMEOUT arm still matches it, so
    /// `expected: TIMEOUT` metadata keeps working for both.
    fn run_testharness(&mut self, item: &TestItem, timeout: Duration) -> TestOutcome {
        let driver = Arc::clone(&self.driver);
        let item = item.clone();
        // `id`/`url` stay outside the worker closure: the closure moves
        // `item`, but the arms below still need them for attribution and
        // messages.
        let id = item.id.clone();
        let url = item.url.clone();
        let outcome =
            self.call_with_timeout(&id, timeout, move || driver.run_testharness(&item, timeout));
        match outcome {
            Err(DriverError::BrowserGone(_)) => TestOutcome::Harness(HarnessResult {
                test: id.clone(),
                url: Some(url.clone()),
                status: HarnessStatus::Crash,
                message: None,
                duration_ms: None,
                subtests: Vec::new(),
            }),
            // A hung test is a test timeout, not a driver failure: the
            // watchdog expired while the (probe-healthy) driver was still
            // inside the test. Recorded as a watchdog kill — distinct from
            // a harness-reported TIMEOUT, where the driver answered with
            // testharness.js's own timeout report.
            Err(DriverError::Timeout { after, .. }) => TestOutcome::DriverTimeout {
                id: id.clone(),
                url: url.clone(),
                after,
            },
            Err(e) => TestOutcome::DriverFailed(e),
            Ok(result) if result.test != id || result.url.as_deref() != Some(url.as_str()) => {
                let got_test = result.test;
                let got_url = result.url.unwrap_or_default();
                TestOutcome::Harness(HarnessResult {
                    test: id.clone(),
                    url: Some(url.clone()),
                    status: HarnessStatus::Error,
                    message: None,
                    duration_ms: None,
                    subtests: vec![SubtestResult {
                        name: "report attribution".into(),
                        status: SubtestStatus::Fail,
                        message: Some(format!(
                            "driver returned a report for test ({got_test:?}, {got_url:?}), \
                             expected ({id:?}, {url:?}); stale or misattributed reports \
                             are rejected"
                        )),
                    }],
                })
            }
            Ok(result) => classify_harness_result(&id, result),
        }
    }

    /// Run one crashtest: load the page, pass if the browser survives.
    ///
    /// The page URL is joined against
    /// [`RunConfig::server_base_url`]: the driver hands it to the browser
    /// verbatim, so a manifest-relative URL would fail with "missing
    /// scheme".
    ///
    /// A dead browser is a *test* failure (the page under test crashed it),
    /// not a driver failure: the driver itself is fine. A watchdog expiry
    /// is also a test failure — the page hung instead of loading — reported
    /// as a failed crashtest. Any other driver error (unimplemented call,
    /// wedge, abandonment-cap trip) is a driver failure. The runner
    /// enforces [`RunConfig::timeout`] around the load itself, so a hung
    /// page cannot stall the run.
    fn run_crashtest(&mut self, item: &TestItem, timeout: Duration) -> TestOutcome {
        let driver = Arc::clone(&self.driver);
        let url = self.absolute_url(&item.url);
        let label = url.clone();
        match self.call_with_timeout(&label, timeout, move || driver.load_without_crash(&url)) {
            Ok(()) => TestOutcome::Crashtest { passed: true },
            Err(DriverError::BrowserGone(_)) => TestOutcome::Crashtest { passed: false },
            Err(DriverError::Timeout { .. }) => TestOutcome::Crashtest { passed: false },
            Err(e) => TestOutcome::DriverFailed(e),
        }
    }

    /// Render `url` at `viewport` under the runner-side watchdog.
    ///
    /// The url is joined against
    /// [`RunConfig::server_base_url`] first: the driver hands it to the
    /// browser verbatim (`bowser --headless <url>`), and a
    /// manifest-relative URL would make the browser exit with "missing
    /// scheme".
    ///
    /// A dead browser surfaces as [`DriverError::BrowserGone`] for the
    /// caller to classify; anything else is returned as-is.
    fn render_guarded(
        &mut self,
        url: &str,
        timeout: Duration,
        viewport: Viewport,
    ) -> Result<Screenshot, DriverError> {
        let driver = Arc::clone(&self.driver);
        let url = self.absolute_url(url);
        let label = url.clone();
        self.call_with_timeout(&label, timeout, move || driver.render(&url, viewport))
    }

    /// Join a manifest-relative URL against
    /// [`RunConfig::server_base_url`] so the driver receives an absolute
    /// URL it can hand to the browser verbatim (see
    /// [`resolve_server_url`]).
    fn absolute_url(&self, url: &str) -> String {
        resolve_server_url(&self.config.server_base_url, url)
    }

    /// Run one reftest: render the test and each reference at the
    /// test's viewport (its `"viewport_size"` manifest extra, else
    /// [`RunConfig::viewport`]) and compare with the test's tolerance
    /// (its `"fuzzy"` manifest extra, else [`RunConfig::tolerance`]).
    /// Test and reference URLs are joined against
    /// [`RunConfig::server_base_url`] so the driver renders absolute
    /// URLs (a manifest-relative URL would make the browser exit with
    /// "missing scheme").
    fn run_reftest(
        &mut self,
        item: &TestItem,
        timeout: Duration,
        extras: TestExtras,
    ) -> TestOutcome {
        // The fuzzy pixel budget is a fraction of the rendered area, so
        // the tolerance is resolved against the *effective* viewport:
        // both shots render at it, and `compare` requires equal
        // dimensions anyway.
        let viewport = extras.viewport(self.config.viewport);
        let tolerance = extras.tolerance(
            viewport.width as u64 * viewport.height as u64,
            self.config.tolerance,
        );
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
        let test_shot = match self.render_guarded(&item.url, timeout, viewport) {
            Ok(s) => s,
            // The browser died rendering the test page: the test failed,
            // it is not a driver malfunction.
            Err(DriverError::BrowserGone(_)) => {
                return TestOutcome::Reftest {
                    passed: false,
                    checks: Vec::new(),
                };
            }
            // The test page hung under the watchdog: the test failed, it
            // is not a driver malfunction (a wedged driver is retired by
            // the health probe instead).
            Err(DriverError::Timeout { .. }) => {
                return TestOutcome::Reftest {
                    passed: false,
                    checks: Vec::new(),
                };
            }
            Err(e) => return TestOutcome::DriverFailed(e),
        };
        let mut checks = Vec::with_capacity(refs.len());
        for r in refs {
            // The report carries the absolute URL the driver rendered,
            // not just the manifest-relative resolution.
            let ref_url = self.absolute_url(&resolve_reference_url(&item.url, &r.href));
            // A reference the driver's fixture root cannot serve (pruned
            // from a sparse checkout, e.g.) is harness-side I/O, never
            // an engine pixel failure.
            if !self.driver.resource_exists(&ref_url) {
                return TestOutcome::FilesystemFailure(format!(
                    "reftest reference missing from fixture root: {ref_url}"
                ));
            }
            let ref_shot: Screenshot = match self.render_guarded(&ref_url, timeout, viewport) {
                Ok(s) => s,
                Err(DriverError::BrowserGone(_)) => {
                    return TestOutcome::Reftest {
                        passed: false,
                        checks: Vec::new(),
                    };
                }
                // A hung reference render is a test failure, not a driver
                // failure, same as a hung test render above.
                Err(DriverError::Timeout { .. }) => {
                    return TestOutcome::Reftest {
                        passed: false,
                        checks: Vec::new(),
                    };
                }
                Err(e) => return TestOutcome::DriverFailed(e),
            };
            let comparison = match reftest::compare(&test_shot, &ref_shot, tolerance) {
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
        // Upstream WPT rule: if any `match` references exist, at least one
        // must match; if any `mismatch` references exist, all of them must
        // mismatch. (A lone `match` list needs one hit; a lone `mismatch`
        // list needs every entry to differ.)
        let match_ok = !checks.iter().any(|c| c.rel == RefRel::Match)
            || checks.iter().any(|c| c.rel == RefRel::Match && c.ok);
        let mismatch_ok = checks
            .iter()
            .filter(|c| c.rel == RefRel::Mismatch)
            .all(|c| c.ok);
        let passed = match_ok && mismatch_ok;
        TestOutcome::Reftest { passed, checks }
    }
}

/// Classify one finished testharness result the way the live runner does.
///
/// Shared by [`Runner`] (via `run_testharness`) and the wptreport import
/// path so both agree: `PRECONDITION_FAILED` is a skip — like upstream
/// WPT — never a failure; every other harness-produced result becomes
/// [`TestOutcome::Harness`], whose pass/fail comes from
/// [`HarnessResult::is_green`]. The skip predicate lives on
/// [`HarnessStatus::is_skip`](crate::results::HarnessStatus::is_skip):
/// this function defers to it rather than matching on the status itself.
pub fn classify_harness_result(id: &str, result: HarnessResult) -> TestOutcome {
    if result.status.is_skip() {
        TestOutcome::Skipped {
            reason: format!("precondition failed for {id}"),
        }
    } else {
        TestOutcome::Harness(result)
    }
}

/// Fold one test's outcome into a run's aggregate counts.
///
/// A test flagged `expected_fail` (its outcome matched its
/// expected-failure metadata) counts toward `expected_fail` instead of
/// any failure bucket: it is a known failure, not a regression.
fn tally(summary: &mut RunSummary, outcome: &TestOutcome, expected_fail: bool) {
    summary.total += 1;
    if expected_fail {
        summary.expected_fail += 1;
        return;
    }
    match outcome {
        TestOutcome::Harness(h) if h.status == HarnessStatus::Crash => summary.harness_crash += 1,
        // "Loaded, subtests unknown" (v1 driver) is neither a pass nor a
        // failure: it must be bucketed before the `is_green` arm, where an
        // empty subtest list would otherwise count as a pass.
        TestOutcome::Harness(h) if h.status == HarnessStatus::SubtestsUnknown => {
            summary.harness_unknown += 1
        }
        TestOutcome::Harness(h) if h.is_green() => summary.harness_pass += 1,
        TestOutcome::Harness(_) => summary.harness_fail += 1,
        // A watchdog kill is its own bucket: it must never be tallied as
        // a harness-reported TIMEOUT (harness_fail).
        TestOutcome::DriverTimeout { .. } => summary.driver_timeouts += 1,
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
/// Absolute URLs (anything with a scheme) are used as-is. Absolute-path
/// hrefs (`/css/ref.html`) resolve against the test URL's origin, so a
/// test served from `https://example.com/dir/test.html` keeps its
/// scheme+host instead of degrading to a bare path. Anything else is
/// resolved against the directory of `test_url` the way a browser would
/// resolve it, normalizing `.` and `..` segments (`/dir/sub/test.html` +
/// `../ref.html` becomes `/dir/ref.html`). `..` past the root clamps at
/// the root.
fn resolve_reference_url(test_url: &str, href: &str) -> String {
    // Split an absolute test URL into its "scheme://authority" origin and
    // its path; a relative test URL has no origin.
    let (origin, path) = match test_url.find("://") {
        Some(i) => {
            let rest = &test_url[i + 3..];
            match rest.find('/') {
                Some(j) => (&test_url[..i + 3 + j], &rest[j..]),
                None => (test_url, "/"),
            }
        }
        None => ("", test_url),
    };
    // An absolute URL (a scheme is present) is used as-is.
    if href.contains("://") {
        return href.to_owned();
    }
    if let Some(scheme_end) = href.find(':')
        && !href[..scheme_end].contains('/')
    {
        return href.to_owned();
    }
    // An absolute-path href resolves against the origin.
    if let Some(abs) = href.strip_prefix('/') {
        return format!("{origin}/{abs}");
    }
    // A relative href resolves against the test's directory, normalizing
    // `.` and `..` segments.
    let dir = path.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
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
    format!("{origin}/{}", segments.join("/"))
}

/// Join a manifest-relative URL against the server base URL so the driver
/// receives an absolute URL it can hand to the browser verbatim.
///
/// Drivers spawn the browser with the URL exactly as given (`bowser
/// --headless <url>`); a manifest-relative URL like `/css/x.html` fails
/// there with "missing scheme". Absolute URLs (anything carrying a
/// scheme) pass through untouched, and an empty base leaves the URL
/// untouched — the joining is the driver's business when it wants it.
/// Idempotent: joining an already-joined URL is a no-op.
///
/// The base's trailing slash is ignored (`http://h:1/` + `/css/x.html`
/// becomes `http://h:1/css/x.html`, not a doubled `//`); a bare path
/// without a leading slash is appended under the base
/// (`http://h` + `css/x.html` becomes `http://h/css/x.html`).
fn resolve_server_url(base: &str, url: &str) -> String {
    let base = base.strip_suffix('/').unwrap_or(base);
    // No base configured: leave the URL for the driver to interpret.
    // Already absolute (a scheme is present): used as-is.
    if base.is_empty() || has_scheme(url) {
        return url.to_owned();
    }
    match url.strip_prefix('/') {
        Some(path) => format!("{base}/{path}"),
        None => format!("{base}/{url}"),
    }
}

/// Whether `url` carries a URI scheme (`https://x/y`, `data:...`, ...):
/// a colon with no `/` before it.
fn has_scheme(url: &str) -> bool {
    match url.find(':') {
        Some(i) => !url[..i].contains('/'),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::{StubDriver, Viewport};
    use crate::manifest::TestKind;
    use crate::reftest::Reference;
    use crate::results::{
        HarnessResult, HarnessStatus, SubtestResult, SubtestStatus, parse_wptreport,
    };
    use std::sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    fn item(id: &str, kind: TestKind) -> TestItem {
        TestItem {
            id: id.into(),
            url: format!("/{id}"),
            kind,
            path: None,
            references: Vec::new(),
        }
    }

    /// Drive one test with the runner's configured timeout (extras-aware
    /// scaling lives in [`Runner::run`], which has the manifest; direct
    /// `run_one` callers pass the extras explicitly).
    fn run1<D: BrowserDriver + Send + Sync + 'static>(
        r: &mut Runner<D>,
        item: &TestItem,
    ) -> TestOutcome {
        let timeout = r.config.timeout;
        r.run_one(item, timeout, TestExtras::default())
    }

    /// A driver with canned answers, so the runner logic is testable without
    /// a browser.
    struct FakeDriver {
        report: HarnessResult,
        shot: Screenshot,
        rendered_urls: Mutex<Vec<String>>,
    }

    impl BrowserDriver for FakeDriver {
        fn run_testharness(
            &self,
            test: &TestItem,
            _timeout: Duration,
        ) -> Result<HarnessResult, DriverError> {
            // A well-behaved driver attributes its report to the test it ran:
            // both the id and the served url (multi-global variants share an
            // id across different urls).
            let mut report = self.report.clone();
            report.test = test.id.clone();
            report.url = Some(test.url.clone());
            Ok(report)
        }

        fn render(&self, url: &str, _viewport: Viewport) -> Result<Screenshot, DriverError> {
            self.rendered_urls.lock().unwrap().push(url.to_owned());
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
            rendered_urls: Mutex::new(Vec::new()),
        }
    }

    fn green_report() -> HarnessResult {
        HarnessResult {
            test: "x".into(),
            url: None,
            status: HarnessStatus::Ok,
            message: None,
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
            rendered_urls: Mutex::new(Vec::new()),
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
            rendered_urls: Mutex::new(Vec::new()),
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
                rendered_urls: Mutex::new(Vec::new()),
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
                rendered_urls: Mutex::new(Vec::new()),
            },
            RunConfig::default(),
        );
        let outcome = run1(&mut r, &test);
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
        let outcome = run1(&mut r, &test);
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
            r.driver().rendered_urls.lock().unwrap().clone(),
            vec!["/css/t.html", "/css/same.html", "/css/sub/diff.html"]
        );
    }

    /// A driver whose fixture root serves nothing: every reference
    /// lookup misses.
    struct MissingRefDriver {
        inner: FakeDriver,
    }

    impl BrowserDriver for MissingRefDriver {
        fn run_testharness(
            &self,
            test: &TestItem,
            timeout: Duration,
        ) -> Result<HarnessResult, DriverError> {
            self.inner.run_testharness(test, timeout)
        }

        fn render(&self, url: &str, viewport: Viewport) -> Result<Screenshot, DriverError> {
            self.inner.render(url, viewport)
        }

        fn resource_exists(&self, _url: &str) -> bool {
            false
        }
    }

    #[test]
    fn missing_reftest_reference_is_filesystem_failure() {
        let test = TestItem {
            id: "css/t.html".into(),
            url: "/css/t.html".into(),
            kind: TestKind::Reftest,
            path: None,
            references: vec![Reference {
                rel: RefRel::Match,
                href: "gone.html".into(),
            }],
        };
        let mut r = Runner::new(
            MissingRefDriver {
                inner: fake_driver(),
            },
            RunConfig::default(),
        );
        let outcome = run1(&mut r, &test);
        assert!(
            matches!(outcome, TestOutcome::FilesystemFailure(_)),
            "a reference the driver cannot serve must be FilesystemFailure, got: {outcome:?}"
        );
        assert!(!outcome.passed());
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
        let outcome = run1(&mut r, &test);
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
        // Absolute test URLs keep their scheme+host: relative hrefs
        // resolve under the origin, and absolute-path hrefs resolve
        // against the origin instead of degrading to a bare path.
        assert_eq!(
            resolve_reference_url("https://example.com/dir/test.html", "ref.html"),
            "https://example.com/dir/ref.html"
        );
        assert_eq!(
            resolve_reference_url("https://example.com/dir/test.html", "sub/ref.html"),
            "https://example.com/dir/sub/ref.html"
        );
        assert_eq!(
            resolve_reference_url("https://example.com/dir/sub/test.html", "../ref.html"),
            "https://example.com/dir/ref.html"
        );
        assert_eq!(
            resolve_reference_url("https://example.com/dir/test.html", "/abs/ref.html"),
            "https://example.com/abs/ref.html"
        );
        assert_eq!(
            resolve_reference_url(
                "https://example.com/dir/test.html",
                "https://other.com/ref.html"
            ),
            "https://other.com/ref.html"
        );
        assert_eq!(
            resolve_reference_url("https://example.com", "ref.html"),
            "https://example.com/ref.html"
        );
    }

    #[test]
    fn server_url_resolution() {
        // No base configured: the URL reaches the driver untouched, so
        // drivers that join internally keep their behavior.
        assert_eq!(resolve_server_url("", "/css/t.html"), "/css/t.html");
        // Manifest-relative URLs join against the base.
        assert_eq!(
            resolve_server_url("http://127.0.0.1:8080", "/css/t.html"),
            "http://127.0.0.1:8080/css/t.html"
        );
        // A trailing slash on the base never doubles the path separator.
        assert_eq!(
            resolve_server_url("http://127.0.0.1:8080/", "/css/t.html"),
            "http://127.0.0.1:8080/css/t.html"
        );
        // A bare path without a leading slash appends under the base.
        assert_eq!(
            resolve_server_url("http://127.0.0.1:8080", "css/t.html"),
            "http://127.0.0.1:8080/css/t.html"
        );
        // Absolute URLs pass through untouched even with a base set.
        assert_eq!(
            resolve_server_url("http://127.0.0.1:8080", "https://example.com/t.html"),
            "https://example.com/t.html"
        );
        assert_eq!(
            resolve_server_url("http://127.0.0.1:8080", "data:text/html,x"),
            "data:text/html,x"
        );
    }

    #[test]
    fn reftest_urls_are_joined_against_server_base() {
        // The wave-44 triage's #1 blocker: the runner handed
        // manifest-relative URLs to `BrowserDriver::render`, which spawns
        // `bowser --headless <url>` verbatim and exits 1 ("missing
        // scheme"). With a server base configured, the driver must see
        // absolute URLs for both the test page and its references.
        let mut r = Runner::new(
            fake_driver(),
            RunConfig {
                server_base_url: "http://127.0.0.1:8080".into(),
                ..RunConfig::default()
            },
        );
        let outcome = run1(&mut r, &reftest_item(&[("match", "ref.html")]));
        match &outcome {
            TestOutcome::Reftest { passed, checks } => {
                assert!(passed, "identical screenshots must pass, got: {outcome:?}");
                // The report carries the absolute URLs that were rendered.
                assert_eq!(checks[0].reference, "http://127.0.0.1:8080/css/ref.html");
            }
            o => panic!("expected a reftest outcome, got: {o:?}"),
        }
        assert_eq!(
            r.driver().rendered_urls.lock().unwrap().clone(),
            vec![
                "http://127.0.0.1:8080/css/t.html",
                "http://127.0.0.1:8080/css/ref.html",
            ]
        );
    }

    #[test]
    fn crashtest_url_is_joined_against_server_base() {
        // `load_without_crash` also receives the URL verbatim, so it gets
        // the same joining as the render path.
        let mut r = Runner::new(
            CrashDriver {
                crash: Mutex::new(None),
                load_calls: Mutex::new(Vec::new()),
                harness_calls: AtomicUsize::new(0),
            },
            RunConfig {
                server_base_url: "http://127.0.0.1:8080".into(),
                ..RunConfig::default()
            },
        );
        let outcome = run1(&mut r, &item("crash/x.html", TestKind::Crashtest));
        assert!(matches!(outcome, TestOutcome::Crashtest { passed: true }));
        assert_eq!(
            r.driver().load_calls.lock().unwrap().clone(),
            vec!["http://127.0.0.1:8080/crash/x.html".to_owned()]
        );
    }

    #[test]
    fn absolute_test_urls_pass_through_with_server_base() {
        // A test that is already absolute is never re-joined: the
        // manifest may carry absolute urls (or url_base may), and
        // joining those onto the base would produce garbage.
        let mut r = Runner::new(
            fake_driver(),
            RunConfig {
                server_base_url: "http://127.0.0.1:8080".into(),
                ..RunConfig::default()
            },
        );
        let mut test = reftest_item(&[("match", "ref.html")]);
        test.url = "https://example.com/css/t.html".into();
        let outcome = run1(&mut r, &test);
        assert!(matches!(outcome, TestOutcome::Reftest { passed: true, .. }));
        assert_eq!(
            r.driver().rendered_urls.lock().unwrap().clone(),
            vec![
                "https://example.com/css/t.html",
                "https://example.com/css/ref.html",
            ]
        );
    }

    /// A driver that records which entry point the runner used, so tests can
    /// prove crashtests never wait on the testharness completion callback.
    struct CrashDriver {
        /// What `load_without_crash` returns; `None` means `Ok(())`.
        crash: Mutex<Option<DriverError>>,
        load_calls: Mutex<Vec<String>>,
        harness_calls: AtomicUsize,
    }

    impl BrowserDriver for CrashDriver {
        fn run_testharness(
            &self,
            _test: &TestItem,
            _timeout: Duration,
        ) -> Result<HarnessResult, DriverError> {
            self.harness_calls.fetch_add(1, Ordering::SeqCst);
            panic!("crashtests must not wait for a testharness.js completion callback");
        }

        fn load_without_crash(&self, url: &str) -> Result<(), DriverError> {
            self.load_calls.lock().unwrap().push(url.to_owned());
            match self.crash.lock().unwrap().take() {
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
                crash: Mutex::new(None),
                load_calls: Mutex::new(Vec::new()),
                harness_calls: AtomicUsize::new(0),
            },
            RunConfig::default(),
        );
        let outcome = run1(&mut r, &item("crash/x.html", TestKind::Crashtest));
        assert!(matches!(outcome, TestOutcome::Crashtest { passed: true }));
        assert!(outcome.passed());
        assert_eq!(r.driver().harness_calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            r.driver().load_calls.lock().unwrap().clone(),
            vec!["/crash/x.html".to_owned()]
        );
    }

    #[test]
    fn crashtest_browser_crash_is_a_test_failure() {
        let mut r = Runner::new(
            CrashDriver {
                crash: Mutex::new(Some(DriverError::BrowserGone("segfault".into()))),
                load_calls: Mutex::new(Vec::new()),
                harness_calls: AtomicUsize::new(0),
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
        // A driver-side malfunction (here: unimplemented) is a driver
        // failure. A watchdog timeout is deliberately NOT in this bucket:
        // it is a test failure (see `watchdog_bounds_a_hung_crashtest_load`).
        let mut r = Runner::new(
            CrashDriver {
                crash: Mutex::new(Some(DriverError::Unimplemented(
                    "load_without_crash not implemented".into(),
                ))),
                load_calls: Mutex::new(Vec::new()),
                harness_calls: AtomicUsize::new(0),
            },
            RunConfig::default(),
        );
        let outcome = run1(&mut r, &item("crash/z.html", TestKind::Crashtest));
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
                rendered_urls: Mutex::new(Vec::new()),
            },
            RunConfig::default(),
        );
        let outcome = run1(&mut r, &test);
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
            false,
        );
        assert_eq!(summary.total, 1);
        assert_eq!(summary.filesystem_failures, 1);
        assert_eq!(summary.driver_failed, 0);
    }

    /// A driver whose calls never return, proving the runner-side watchdog
    /// fires instead of stalling the run forever.
    struct HangingDriver;

    impl BrowserDriver for HangingDriver {
        fn run_testharness(
            &self,
            _test: &TestItem,
            _timeout: Duration,
        ) -> Result<HarnessResult, DriverError> {
            std::thread::sleep(Duration::from_secs(30));
            Ok(green_report())
        }

        fn load_without_crash(&self, _url: &str) -> Result<(), DriverError> {
            std::thread::sleep(Duration::from_secs(30));
            Ok(())
        }

        fn render(&self, _url: &str, _viewport: Viewport) -> Result<Screenshot, DriverError> {
            std::thread::sleep(Duration::from_secs(30));
            Ok(Screenshot::solid(1, 1, 0, 0, 0, 255))
        }

        fn name(&self) -> &str {
            "hanging"
        }
    }

    fn hanging_runner() -> Runner<HangingDriver> {
        Runner::new(
            HangingDriver,
            RunConfig {
                timeout: Duration::from_millis(50),
                ..RunConfig::default()
            },
        )
    }

    #[test]
    fn watchdog_bounds_a_hung_testharness() {
        let mut r = hanging_runner();
        let outcome = run1(&mut r, &item("dom/hang.html", TestKind::Testharness));
        // A hung test is a test timeout, not a driver failure — and a
        // watchdog kill, not a harness-reported TIMEOUT: the driver never
        // produced a report. The TIMEOUT expectation still matches it.
        match &outcome {
            TestOutcome::DriverTimeout { id, url, after } => {
                assert_eq!(id, "dom/hang.html");
                assert_eq!(url, "/dom/hang.html");
                assert_eq!(*after, Duration::from_millis(50));
            }
            o => panic!("a hung test must report a watchdog kill, got: {o:?}"),
        }
        assert!(!outcome.passed());
    }

    /// A report shaped like testharness.js's own timeout: the driver
    /// answered instantly with `harness_status: "TIMEOUT"` (what the
    /// encoding suite triage saw on every one of its 12 TIMEOUTs — each
    /// child printed `BOWSER_HARNESS_REPORT` and exited in seconds).
    fn harness_timeout_report() -> HarnessResult {
        HarnessResult {
            test: "x".into(),
            url: None,
            status: HarnessStatus::Timeout,
            message: None,
            duration_ms: Some(1200),
            subtests: vec![SubtestResult {
                name: "decoding".into(),
                status: SubtestStatus::Timeout,
                message: None,
            }],
        }
    }

    #[test]
    fn harness_reported_timeout_stays_a_harness_timeout() {
        // The driver answered with testharness.js's own TIMEOUT report:
        // this must stay a Harness outcome, never a watchdog kill.
        let mut r = Runner::new(
            FakeDriver {
                report: harness_timeout_report(),
                shot: Screenshot::solid(2, 2, 0, 0, 0, 255),
                rendered_urls: Mutex::new(Vec::new()),
            },
            RunConfig::default(),
        );
        let m = manifest_of(vec![item("encoding/t.html", TestKind::Testharness)]);
        let report = r.run(&m);
        match &report.reports[0].outcome {
            TestOutcome::Harness(h) => {
                assert_eq!(h.status, HarnessStatus::Timeout);
                assert_eq!(h.test, "encoding/t.html");
                // The harness's own subtest data survives: this is a real
                // report, not a synthesized watchdog placeholder.
                assert_eq!(h.subtests.len(), 1);
                assert_eq!(h.subtests[0].status, SubtestStatus::Timeout);
            }
            o => panic!("a harness-reported TIMEOUT must stay Harness, got: {o:?}"),
        }
        assert!(!report.reports[0].outcome.passed());
        assert_eq!(report.summary.harness_fail, 1);
        assert_eq!(report.summary.driver_timeouts, 0);
        let text = report.summary_text();
        assert!(
            text.contains("watchdog timeouts (driver produced no report): 0"),
            "summary text was:\n{text}"
        );
    }

    #[test]
    fn watchdog_kill_is_disambiguated_from_harness_timeout() {
        // The driver hangs past a tiny watchdog and produces no report:
        // this must be a DriverTimeout, tallied apart from the
        // harness-reported TIMEOUTs above.
        let mut r = hanging_runner();
        let m = manifest_of(vec![item("dom/hang.html", TestKind::Testharness)]);
        let report = r.run(&m);
        assert!(
            matches!(
                &report.reports[0].outcome,
                TestOutcome::DriverTimeout { .. }
            ),
            "a hung driver must report a watchdog kill, got: {:?}",
            report.reports[0].outcome
        );
        assert_eq!(report.summary.driver_timeouts, 1);
        assert_eq!(report.summary.harness_fail, 0);
        assert_eq!(report.summary.driver_failed, 0);
        let text = report.summary_text();
        assert!(
            text.contains("watchdog timeouts (driver produced no report): 1"),
            "summary text was:\n{text}"
        );
        // `expected: TIMEOUT` metadata still matches a watchdog kill, so
        // existing expectations keep working for both timeout flavors.
        assert!(is_expected_failure(
            &report.reports[0].outcome,
            &ExpectedEntry {
                expected: ExpectedStatus::Timeout,
                bug: None,
            },
        ));
        // The wptreport export keeps upstream's TIMEOUT status (the schema
        // has no distinct code) and carries the disambiguation in the
        // message.
        let json =
            crate::results::export_wptreport(&report, &crate::results::ExpectedMetadata::default());
        assert!(
            json.contains(r#""status":"TIMEOUT""#),
            "export was:\n{json}"
        );
        assert!(
            json.contains("watchdog"),
            "export must name the watchdog, was:\n{json}"
        );
    }

    #[test]
    fn watchdog_kills_are_tallied() {
        let mut summary = RunSummary::default();
        tally(
            &mut summary,
            &TestOutcome::DriverTimeout {
                id: "x".into(),
                url: "/x".into(),
                after: Duration::from_millis(50),
            },
            false,
        );
        assert_eq!(summary.total, 1);
        assert_eq!(summary.driver_timeouts, 1);
        assert_eq!(summary.harness_fail, 0);
        assert_eq!(summary.driver_failed, 0);
    }

    #[test]
    fn watchdog_bounds_a_hung_crashtest_load() {
        let mut r = hanging_runner();
        let outcome = run1(&mut r, &item("crash/hang.html", TestKind::Crashtest));
        assert!(
            matches!(outcome, TestOutcome::Crashtest { passed: false }),
            "a hung crashtest load must fail the test, not fail the driver, got: {outcome:?}"
        );
    }

    #[test]
    fn watchdog_bounds_a_hung_reftest_render() {
        let test = TestItem {
            id: "css/hang.html".into(),
            url: "/css/hang.html".into(),
            kind: TestKind::Reftest,
            path: None,
            references: vec![Reference {
                rel: RefRel::Match,
                href: "same.html".into(),
            }],
        };
        let mut r = hanging_runner();
        let outcome = run1(&mut r, &test);
        assert!(
            matches!(outcome, TestOutcome::Reftest { passed: false, .. }),
            "a hung reftest render must fail the test, not fail the driver, got: {outcome:?}"
        );
    }

    /// A driver where every call hangs, including the cheap `name()`
    /// probe: the abandoned worker is wedged inside a driver call, e.g.
    /// holding a driver-internal lock.
    struct WedgedDriver {
        harness_calls: AtomicUsize,
    }

    impl BrowserDriver for WedgedDriver {
        fn run_testharness(
            &self,
            _test: &TestItem,
            _timeout: Duration,
        ) -> Result<HarnessResult, DriverError> {
            self.harness_calls.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_secs(30));
            Ok(green_report())
        }

        fn render(&self, _url: &str, _viewport: Viewport) -> Result<Screenshot, DriverError> {
            std::thread::sleep(Duration::from_secs(30));
            Ok(Screenshot::solid(1, 1, 0, 0, 0, 255))
        }

        fn name(&self) -> &str {
            std::thread::sleep(Duration::from_secs(30));
            "wedged"
        }
    }

    #[test]
    fn wedged_driver_is_retired_after_one_probe() {
        // Every call hangs, including the `name()` health probe: the first
        // timeout must retire the driver instead of burning a full timeout
        // on every remaining test, and the wedge must not be misattributed
        // as independent test timeouts.
        let mut r = Runner::new(
            WedgedDriver {
                harness_calls: AtomicUsize::new(0),
            },
            RunConfig {
                timeout: Duration::from_millis(50),
                probe_timeout: Duration::from_millis(50),
                ..RunConfig::default()
            },
        );
        let m = manifest_of(vec![
            item("dom/a.html", TestKind::Testharness),
            item("dom/b.html", TestKind::Testharness),
            item("dom/c.html", TestKind::Testharness),
        ]);
        let report = r.run(&m);
        // The tripping test reports the wedge explicitly...
        assert!(
            matches!(
                report.reports[0].outcome,
                TestOutcome::DriverFailed(DriverError::DriverWedged { .. })
            ),
            "tripping test must report the wedge, got: {:?}",
            report.reports[0].outcome
        );
        // ...and the rest are marked skipped, never attempted: only one
        // driver call was ever made.
        assert_eq!(r.driver().harness_calls.load(Ordering::SeqCst), 1);
        for rep in &report.reports[1..] {
            assert!(
                matches!(&rep.outcome, TestOutcome::Skipped { reason } if reason.contains("never attempted")),
                "remaining tests must be skipped as never attempted, got: {:?}",
                rep.outcome
            );
        }
        assert!(report.summary.aborted);
        assert_eq!(report.summary.unattempted, 2);
        assert_eq!(report.summary.skipped, 2);
        assert_eq!(report.summary.driver_failed, 1);
        // Exactly one worker abandoned: no 33-worker pileup.
        assert_eq!(r.abandoned_drivers(), 1);
        // A hung `name()` cannot hang report construction either.
        assert_eq!(report.driver_name, "unknown (driver did not answer name())");
        let text = report.summary_text();
        assert!(text.contains("aborted: true"), "summary text was:\n{text}");
        assert!(
            text.contains("2 tests never attempted"),
            "summary text was:\n{text}"
        );
    }

    #[test]
    fn healthy_driver_survives_an_isolated_timeout() {
        // HangingDriver hangs on real calls but answers `name()` instantly:
        // the probe passes, so isolated timeouts stay timeouts and the run
        // continues instead of retiring the driver.
        let mut r = Runner::new(
            HangingDriver,
            RunConfig {
                timeout: Duration::from_millis(50),
                probe_timeout: Duration::from_millis(50),
                ..RunConfig::default()
            },
        );
        let m = manifest_of(vec![
            item("dom/a.html", TestKind::Testharness),
            item("dom/b.html", TestKind::Testharness),
        ]);
        let report = r.run(&m);
        // Hung tests are test timeouts now, not driver failures: two
        // watchdog kills, zero driver failures, and the run continues
        // instead of retiring the driver. They tally as watchdog timeouts,
        // not harness-reported TIMEOUTs.
        assert_eq!(report.summary.driver_failed, 0);
        assert_eq!(report.summary.driver_timeouts, 2);
        assert_eq!(report.summary.harness_fail, 0);
        assert!(!report.summary.aborted);
        assert_eq!(report.summary.unattempted, 0);
        assert!(
            report
                .reports
                .iter()
                .all(|rep| matches!(&rep.outcome, TestOutcome::DriverTimeout { .. }))
        );
        assert_eq!(r.abandoned_drivers(), 2);
        assert_eq!(report.driver_name, "hanging");
        assert!(!report.summary_text().contains("aborted"));
    }

    #[test]
    fn abandonment_cap_trips_at_32() {
        // The docs promise the run aborts once 32 driver calls have timed
        // out: the 32nd timeout (not the 33rd) must trip the cap.
        //
        // The bounds are generous on purpose: the full test binary runs
        // every timing-sensitive watchdog test in parallel on a small
        // box, and a tight (20ms) probe bound flaked when the scheduler
        // took too long to wake the instant-answering `name()` probe —
        // misattributing jitter as a driver wedge.
        let mut r = Runner::new(
            HangingDriver,
            RunConfig {
                timeout: Duration::from_millis(100),
                probe_timeout: Duration::from_millis(100),
                ..RunConfig::default()
            },
        );
        for i in 0..31 {
            let outcome = run1(&mut r, &item("dom/h.html", TestKind::Testharness));
            assert!(
                matches!(&outcome, TestOutcome::DriverTimeout { .. }),
                "timeout {i} must stay a plain watchdog kill, got: {outcome:?}"
            );
        }
        assert_eq!(r.abandoned_drivers(), 31);
        let outcome = run1(&mut r, &item("dom/h.html", TestKind::Testharness));
        assert!(
            matches!(
                outcome,
                TestOutcome::DriverFailed(DriverError::TooManyTimeouts { abandoned: 32 })
            ),
            "the 32nd timeout must trip the cap, got: {outcome:?}"
        );
        assert_eq!(r.abandoned_drivers(), 32);
    }

    /// Renders a color picked by the URL, so reftests can have references
    /// that match or differ on demand: "red" in the URL renders red,
    /// "green" renders green, anything else renders blue.
    struct ColorDriver;

    impl BrowserDriver for ColorDriver {
        fn run_testharness(
            &self,
            _test: &TestItem,
            _timeout: Duration,
        ) -> Result<HarnessResult, DriverError> {
            unimplemented!("reftest path never runs testharness")
        }

        fn render(&self, url: &str, _viewport: Viewport) -> Result<Screenshot, DriverError> {
            let (rr, gg, bb) = if url.contains("red") {
                (255, 0, 0)
            } else if url.contains("green") {
                (0, 255, 0)
            } else {
                (0, 0, 255)
            };
            Ok(Screenshot::solid(2, 2, rr, gg, bb, 255))
        }

        fn name(&self) -> &str {
            "color"
        }
    }

    /// A reftest item whose test page renders blue, with manifest-parsed
    /// references (no local files needed).
    fn reftest_item(refs: &[(&str, &str)]) -> TestItem {
        TestItem {
            id: "css/t.html".into(),
            url: "/css/t.html".into(),
            kind: TestKind::Reftest,
            path: None,
            references: refs
                .iter()
                .map(|(rel, href)| Reference {
                    rel: match *rel {
                        "match" => RefRel::Match,
                        "mismatch" => RefRel::Mismatch,
                        r => panic!("bad rel {r}"),
                    },
                    href: (*href).into(),
                })
                .collect(),
        }
    }

    #[test]
    fn reftest_match_needs_only_one_hit() {
        // Upstream WPT rule: with `match` references, at least one must
        // match. The old code demanded every one.
        let mut r = Runner::new(ColorDriver, RunConfig::default());
        let outcome = run1(
            &mut r,
            &reftest_item(&[("match", "red.html"), ("match", "blue.html")]),
        );
        match &outcome {
            TestOutcome::Reftest { passed, checks } => {
                assert!(
                    *passed,
                    "one matching match-ref must pass, checks: {checks:?}"
                );
                assert_eq!(checks.len(), 2);
                assert!(!checks[0].ok); // red.html differs from the blue test
                assert!(checks[1].ok); // blue.html matches
            }
            o => panic!("expected a reftest outcome, got: {o:?}"),
        }
    }

    #[test]
    fn reftest_mismatch_needs_every_entry_to_differ() {
        let mut r = Runner::new(ColorDriver, RunConfig::default());
        // Every mismatch reference differs from the blue test: pass.
        let outcome = run1(
            &mut r,
            &reftest_item(&[("mismatch", "red.html"), ("mismatch", "green.html")]),
        );
        assert!(
            outcome.passed(),
            "all-differing mismatch refs must pass: {outcome:?}"
        );
        // One mismatch reference equals the test: fail.
        let outcome = run1(
            &mut r,
            &reftest_item(&[("mismatch", "red.html"), ("mismatch", "blue.html")]),
        );
        assert!(
            !outcome.passed(),
            "an equal mismatch ref must fail: {outcome:?}"
        );
    }

    #[test]
    fn reftest_mixed_match_and_mismatch() {
        let mut r = Runner::new(ColorDriver, RunConfig::default());
        // One match hit plus all mismatches differing: pass.
        let outcome = run1(
            &mut r,
            &reftest_item(&[
                ("match", "red.html"),
                ("match", "blue.html"),
                ("mismatch", "green.html"),
            ]),
        );
        assert!(
            outcome.passed(),
            "satisfied mixed refs must pass: {outcome:?}"
        );
        // Same shape, but a mismatch reference equals the test: fail even
        // though a match reference hits.
        let outcome = run1(
            &mut r,
            &reftest_item(&[
                ("match", "blue.html"),
                ("mismatch", "red.html"),
                ("mismatch", "blue.html"),
            ]),
        );
        assert!(
            !outcome.passed(),
            "an equal mismatch ref must fail the test: {outcome:?}"
        );
    }

    /// A driver that always returns a report for the wrong test, simulating
    /// a stale or misattributed driver report.
    struct WrongIdDriver;

    impl BrowserDriver for WrongIdDriver {
        fn run_testharness(
            &self,
            _test: &TestItem,
            _timeout: Duration,
        ) -> Result<HarnessResult, DriverError> {
            let mut report = green_report();
            report.test = "dom/stale.html".into();
            Ok(report)
        }

        fn render(&self, _url: &str, _viewport: Viewport) -> Result<Screenshot, DriverError> {
            unimplemented!("testharness path never renders")
        }

        fn name(&self) -> &str {
            "wrong-id"
        }
    }

    #[test]
    fn mismatched_report_id_is_a_harness_failure() {
        // A report for the wrong test must not silently poison the
        // scoreboard: the runner rejects it as a harness failure with the
        // mismatch spelled out.
        let m = manifest_of(vec![item("dom/a.html", TestKind::Testharness)]);
        let mut r = Runner::new(WrongIdDriver, RunConfig::default());
        let report = r.run(&m);
        assert_eq!(report.summary.total, 1);
        assert_eq!(report.summary.harness_fail, 1);
        assert_eq!(report.summary.harness_pass, 0);
        match &report.reports[0].outcome {
            TestOutcome::Harness(h) => {
                assert!(!h.is_green());
                assert!(!report.reports[0].outcome.passed());
                let msg = h.subtests[0].message.as_deref().unwrap_or("");
                assert!(msg.contains("dom/stale.html"), "message was: {msg}");
                assert!(msg.contains("dom/a.html"), "message was: {msg}");
            }
            o => panic!("mismatched report must be a harness failure, got: {o:?}"),
        }
    }

    #[test]
    fn matching_report_id_passes_attribution() {
        // The well-behaved path: FakeDriver attributes its report to the
        // item it ran, so a green report tallies as a pass.
        let m = manifest_of(vec![item("dom/a.html", TestKind::Testharness)]);
        let mut r = Runner::new(fake_driver(), RunConfig::default());
        let report = r.run(&m);
        assert_eq!(report.summary.harness_pass, 1);
        assert_eq!(report.summary.harness_fail, 0);
    }

    #[test]
    fn wptreport_precondition_failed_shares_runner_skip_classification() {
        // The wptreport import path must agree with the live runner: a
        // PRECONDITION_FAILED entry is a skip, never a failure. Both go
        // through the shared `classify_harness_result` (which defers to
        // `HarnessStatus::is_skip`); nothing re-matches the status.
        let doc = r#"{"results": [
            {"test": "/dom/p.html", "status": "PRECONDITION_FAILED",
             "subtests": [{"name": "a", "status": "PASS"}]}
        ]}"#;
        let (results, skipped) = parse_wptreport(doc).unwrap();
        assert_eq!(skipped, 0);
        assert_eq!(results.len(), 1);
        let result = results.into_iter().next().unwrap();
        assert!(result.status.is_skip());
        match classify_harness_result("/dom/p.html", result) {
            TestOutcome::Skipped { reason } => {
                assert!(reason.contains("precondition"), "reason was: {reason}")
            }
            o => panic!("wptreport PRECONDITION_FAILED must be a skip, got: {o:?}"),
        }
    }

    #[test]
    fn subtests_unknown_tallies_separately() {
        // The v1 driver's "loaded, subtests unknown" is neither a pass
        // nor a failure: it must land in its own tally bucket.
        let mut report = green_report();
        report.status = HarnessStatus::SubtestsUnknown;
        report.subtests.clear();
        let mut r = Runner::new(
            FakeDriver {
                report,
                shot: Screenshot::solid(1, 1, 0, 0, 0, 255),
                rendered_urls: Mutex::new(Vec::new()),
            },
            RunConfig::default(),
        );
        let m = manifest_of(vec![item("dom/u.html", TestKind::Testharness)]);
        let report = r.run(&m);
        assert_eq!(report.summary.total, 1);
        assert_eq!(report.summary.harness_unknown, 1);
        assert_eq!(report.summary.harness_pass, 0);
        assert_eq!(report.summary.harness_fail, 0);
        assert!(!report.reports[0].outcome.passed());
        assert!(report.summary_text().contains("testharness unknown"));
    }

    #[test]
    fn precondition_failed_is_a_skip() {
        let mut report = green_report();
        report.status = HarnessStatus::PreconditionFailed;
        let mut r = Runner::new(
            FakeDriver {
                report,
                shot: Screenshot::solid(1, 1, 0, 0, 0, 255),
                rendered_urls: Mutex::new(Vec::new()),
            },
            RunConfig::default(),
        );
        let m = manifest_of(vec![item("dom/p.html", TestKind::Testharness)]);
        let report = r.run(&m);
        assert_eq!(report.summary.total, 1);
        assert_eq!(report.summary.skipped, 1);
        assert_eq!(report.summary.harness_fail, 0);
        assert_eq!(report.summary.harness_pass, 0);
        match &report.reports[0].outcome {
            TestOutcome::Skipped { reason } => {
                assert!(reason.contains("precondition"), "reason was: {reason}")
            }
            o => panic!("PRECONDITION_FAILED must be a skip, got: {o:?}"),
        }
    }

    /// A driver that fails testharness/render calls with a canned error.
    struct ErrorDriver {
        harness_err: Option<DriverError>,
        render_err: Option<DriverError>,
    }

    impl BrowserDriver for ErrorDriver {
        fn run_testharness(
            &self,
            _test: &TestItem,
            _timeout: Duration,
        ) -> Result<HarnessResult, DriverError> {
            Err(self.harness_err.clone().expect("test must set harness_err"))
        }

        fn render(&self, _url: &str, _viewport: Viewport) -> Result<Screenshot, DriverError> {
            Err(self.render_err.clone().expect("test must set render_err"))
        }

        fn name(&self) -> &str {
            "error"
        }
    }

    #[test]
    fn browser_gone_in_testharness_is_a_crash() {
        let mut r = Runner::new(
            ErrorDriver {
                harness_err: Some(DriverError::BrowserGone("segfault".into())),
                render_err: None,
            },
            RunConfig::default(),
        );
        let m = manifest_of(vec![item("dom/c.html", TestKind::Testharness)]);
        let report = r.run(&m);
        assert_eq!(report.summary.total, 1);
        assert_eq!(report.summary.harness_crash, 1);
        assert_eq!(report.summary.harness_fail, 0);
        assert_eq!(report.summary.driver_failed, 0);
        match &report.reports[0].outcome {
            TestOutcome::Harness(h) => {
                assert_eq!(h.status, HarnessStatus::Crash);
                assert!(!h.is_green());
            }
            o => panic!("BrowserGone must be a crash outcome, got: {o:?}"),
        }
        assert!(!report.reports[0].outcome.passed());
        assert!(report.summary_text().contains("testharness crashes: 1"));
    }

    #[test]
    fn browser_gone_in_reftest_is_a_reftest_failure() {
        let mut r = Runner::new(
            ErrorDriver {
                harness_err: None,
                render_err: Some(DriverError::BrowserGone("segfault".into())),
            },
            RunConfig::default(),
        );
        let test = TestItem {
            id: "css/c.html".into(),
            url: "/css/c.html".into(),
            kind: TestKind::Reftest,
            path: None,
            references: vec![Reference {
                rel: RefRel::Match,
                href: "same.html".into(),
            }],
        };
        let outcome = run1(&mut r, &test);
        match &outcome {
            TestOutcome::Reftest { passed, checks } => {
                assert!(!passed);
                assert!(checks.is_empty());
            }
            o => panic!("BrowserGone in reftest must be a reftest failure, got: {o:?}"),
        }
        assert!(!outcome.passed());
    }

    /// A multiplexing-style driver: one shared "session child" behind a
    /// mutex, like `bowser_app::wpt_driver::SessionDriver`. Operations
    /// serialize on the lock; the "child" spawns once per driver.
    struct SessionStyleDriver {
        spawns: AtomicUsize,
        ops: AtomicUsize,
        session: Mutex<SessionChildFake>,
    }

    struct SessionChildFake {
        alive: bool,
    }

    impl SessionStyleDriver {
        fn new() -> SessionStyleDriver {
            SessionStyleDriver {
                spawns: AtomicUsize::new(0),
                ops: AtomicUsize::new(0),
                session: Mutex::new(SessionChildFake { alive: false }),
            }
        }

        fn ensure_child(&self) {
            let mut child = self.session.lock().unwrap();
            if !child.alive {
                child.alive = true;
                self.spawns.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    impl BrowserDriver for SessionStyleDriver {
        fn run_testharness(
            &self,
            test: &TestItem,
            _timeout: Duration,
        ) -> Result<HarnessResult, DriverError> {
            self.ensure_child();
            // Serialize like a real session child: one op at a time.
            let _child = self.session.lock().unwrap();
            self.ops.fetch_add(1, Ordering::SeqCst);
            let mut report = green_report();
            report.test = test.id.clone();
            report.url = Some(test.url.clone());
            Ok(report)
        }

        fn render(&self, url: &str, _viewport: Viewport) -> Result<Screenshot, DriverError> {
            self.ensure_child();
            let _child = self.session.lock().unwrap();
            self.ops.fetch_add(1, Ordering::SeqCst);
            let _ = url;
            Ok(Screenshot::solid(2, 2, 0, 0, 0, 255))
        }

        fn name(&self) -> &str {
            "session-style"
        }
    }

    #[test]
    fn multiplexing_driver_gets_one_session_per_run() {
        // The runner hands tests to the driver sequentially; a
        // multiplexing driver must see exactly one session for the whole
        // run — no per-test respawn — and every test must complete.
        let m = manifest_of(vec![
            item("dom/a.html", TestKind::Testharness),
            item("dom/b.html", TestKind::Testharness),
            item("dom/c.html", TestKind::Testharness),
        ]);
        let mut r = Runner::new(SessionStyleDriver::new(), RunConfig::default());
        let report = r.run(&m);
        assert_eq!(report.summary.harness_pass, 3);
        assert_eq!(r.driver().spawns.load(Ordering::SeqCst), 1);
        assert_eq!(r.driver().ops.load(Ordering::SeqCst), 3);
        assert_eq!(report.driver_name, "session-style");
    }

    /// A multiplexing driver whose session hangs mid-operation: the
    /// runner-side watchdog must still bound it — the internal mutex
    /// must not deadlock the watchdog or poison later tests.
    struct HangingSessionDriver {
        session: Mutex<()>,
    }

    impl BrowserDriver for HangingSessionDriver {
        fn run_testharness(
            &self,
            _test: &TestItem,
            _timeout: Duration,
        ) -> Result<HarnessResult, DriverError> {
            let _guard = self.session.lock().unwrap();
            std::thread::sleep(Duration::from_secs(30));
            Ok(green_report())
        }

        fn render(&self, _url: &str, _viewport: Viewport) -> Result<Screenshot, DriverError> {
            unimplemented!("testharness path never renders")
        }

        fn name(&self) -> &str {
            "hanging-session"
        }
    }

    #[test]
    fn watchdog_bounds_a_hung_session_driver() {
        let mut r = Runner::new(
            HangingSessionDriver {
                session: Mutex::new(()),
            },
            RunConfig {
                timeout: Duration::from_millis(50),
                ..RunConfig::default()
            },
        );
        let m = manifest_of(vec![
            item("dom/a.html", TestKind::Testharness),
            item("dom/b.html", TestKind::Testharness),
        ]);
        let report = r.run(&m);
        // Both time out — the second test's worker blocks on the mutex
        // held by the first test's abandoned worker, so it times out too
        // — the run completes, and nothing hangs. Hung tests report as
        // watchdog kills, not driver failures and not harness TIMEOUTs.
        assert_eq!(report.summary.driver_failed, 0);
        assert_eq!(report.summary.driver_timeouts, 2);
        assert_eq!(report.summary.harness_fail, 0);
        assert!(
            report
                .reports
                .iter()
                .all(|rep| matches!(&rep.outcome, TestOutcome::DriverTimeout { .. }))
        );
        assert_eq!(r.abandoned_drivers(), 2);
    }

    /// A driver that records the timeout it was given per testharness
    /// test, so the extras-driven scaling is observable.
    struct TimeoutRecorder {
        seen: Mutex<Vec<(String, Duration)>>,
    }

    impl BrowserDriver for TimeoutRecorder {
        fn run_testharness(
            &self,
            test: &TestItem,
            timeout: Duration,
        ) -> Result<HarnessResult, DriverError> {
            self.seen.lock().unwrap().push((test.id.clone(), timeout));
            let mut report = green_report();
            report.test = test.id.clone();
            report.url = Some(test.url.clone());
            Ok(report)
        }

        fn load_without_crash(&self, _url: &str) -> Result<(), DriverError> {
            unimplemented!("timeout scaling test only runs testharness tests")
        }

        fn render(&self, _url: &str, _viewport: Viewport) -> Result<Screenshot, DriverError> {
            unimplemented!("timeout scaling test only runs testharness tests")
        }

        fn name(&self) -> &str {
            "timeout-recorder"
        }
    }

    #[test]
    fn timeout_long_scales_the_watchdog() {
        // `fuzzy` and `viewport_size` extras ride along in the slow
        // test's extras object: understood keys, never an error.
        let json = r#"{
            "version": 9, "url_base": "/",
            "items": {"testharness": {
                "dom/normal.html": [["/dom/normal.html", {}]],
                "dom/slow.html": [["/dom/slow.html",
                    {"timeout": "long",
                     "fuzzy": [[null, [[0, 1], [0, 10]]]],
                     "viewport_size": [800, 600]}]]
            }}
        }"#;
        let m = Manifest::from_json(json).unwrap();
        assert!(m.extras_for("dom/slow.html").timeout_long);
        assert!(!m.extras_for("dom/normal.html").timeout_long);

        let mut r = Runner::new(
            TimeoutRecorder {
                seen: Mutex::new(Vec::new()),
            },
            RunConfig {
                timeout: Duration::from_secs(2),
                ..RunConfig::default()
            },
        );
        let report = r.run(&m);
        assert_eq!(report.summary.harness_pass, 2);

        let seen = r.driver().seen.lock().unwrap();
        let timeout_for = |id: &str| {
            seen.iter()
                .find(|(seen_id, _)| seen_id == id)
                .map(|(_, t)| *t)
        };
        assert_eq!(timeout_for("dom/normal.html"), Some(Duration::from_secs(2)));
        assert_eq!(
            timeout_for("dom/slow.html"),
            Some(Duration::from_secs(2).saturating_mul(LONG_TIMEOUT_FACTOR))
        );
    }

    /// A driver whose test page renders gray and whose "jitter" reference
    /// renders every pixel off by 5 in the red channel.
    struct JitterDriver;

    impl BrowserDriver for JitterDriver {
        fn run_testharness(
            &self,
            _test: &TestItem,
            _timeout: Duration,
        ) -> Result<HarnessResult, DriverError> {
            unimplemented!("reftest path never runs testharness")
        }

        fn render(&self, url: &str, _viewport: Viewport) -> Result<Screenshot, DriverError> {
            let v = if url.contains("jitter") { 105 } else { 100 };
            Ok(Screenshot::solid(2, 2, v, 100, 100, 255))
        }

        fn name(&self) -> &str {
            "jitter"
        }
    }

    fn fuzzy_manifest(fuzzy: &str) -> Manifest {
        let json = format!(
            r#"{{"version": 9, "url_base": "/", "items": {{
            "reftest": {{"css": {{
                "t.html": [[null, [["/css/jitter.html", "=="]],
                    {{"fuzzy": {fuzzy}}}]]
            }}}}
        }}}}"#
        );
        Manifest::from_json(&json).unwrap()
    }

    #[test]
    fn reftest_fuzzy_extras_relax_comparison() {
        // The test renders (100,100,100), the reference (105,100,100):
        // all 4 pixels differ by 5 in one channel, so exact comparison
        // fails.
        let no_fuzzy = Manifest::from_json(
            r#"{"version": 9, "url_base": "/", "items": {
            "reftest": {"css": {
                "t.html": [[null, [["/css/jitter.html", "=="]], {}]]
            }}}}"#,
        )
        .unwrap();
        let mut r = Runner::new(JitterDriver, RunConfig::default());
        let report = r.run(&no_fuzzy);
        assert_eq!(report.summary.reftest_fail, 1);

        // maxDifference=5 absorbs the channel slack and totalPixels=4
        // covers all 4 differing pixels: pass.
        let mut r = Runner::new(JitterDriver, RunConfig::default());
        let report = r.run(&fuzzy_manifest(r#"[[null, [[0, 5], [0, 4]]]]"#));
        assert_eq!(report.summary.reftest_pass, 1);

        // maxDifference=1 leaves all 4 pixels differing, but totalPixels=2
        // only allows 2: still fails, proving the pixel budget (not just
        // the channel slack) is honored.
        let mut r = Runner::new(JitterDriver, RunConfig::default());
        let report = r.run(&fuzzy_manifest(r#"[[null, [[0, 1], [0, 2]]]]"#));
        assert_eq!(report.summary.reftest_fail, 1);
    }

    /// A driver that records the viewport of every render and returns a
    /// solid screenshot at exactly that size.
    struct ViewportRecorder {
        seen: Mutex<Vec<Viewport>>,
    }

    impl BrowserDriver for ViewportRecorder {
        fn run_testharness(
            &self,
            _test: &TestItem,
            _timeout: Duration,
        ) -> Result<HarnessResult, DriverError> {
            unimplemented!("reftest path never runs testharness")
        }

        fn render(&self, _url: &str, viewport: Viewport) -> Result<Screenshot, DriverError> {
            self.seen.lock().unwrap().push(viewport);
            Ok(Screenshot::solid(
                viewport.width,
                viewport.height,
                0,
                0,
                0,
                255,
            ))
        }

        fn name(&self) -> &str {
            "viewport-recorder"
        }
    }

    #[test]
    fn reftest_viewport_size_extras_override_render_viewport() {
        let m = Manifest::from_json(
            r#"{"version": 9, "url_base": "/", "items": {
            "reftest": {"css": {
                "big.html": [[null, [["/css/r.html", "=="]],
                    {"viewport_size": "320x240"}]],
                "plain.html": [[null, [["/css/r.html", "=="]], {}]]
            }}}}"#,
        )
        .unwrap();
        let mut r = Runner::new(
            ViewportRecorder {
                seen: Mutex::new(Vec::new()),
            },
            RunConfig {
                viewport: Viewport {
                    width: 800,
                    height: 600,
                },
                ..RunConfig::default()
            },
        );
        let report = r.run(&m);
        assert_eq!(report.summary.reftest_pass, 2);
        // Each reftest renders its test page plus one reference, in id
        // order: the extras test at 320x240, the plain one at the run
        // default 800x600.
        assert_eq!(
            r.driver().seen.lock().unwrap().clone(),
            vec![
                Viewport {
                    width: 320,
                    height: 240
                },
                Viewport {
                    width: 320,
                    height: 240
                },
                Viewport {
                    width: 800,
                    height: 600
                },
                Viewport {
                    width: 800,
                    height: 600
                },
            ]
        );
    }

    /// A driver whose every testharness test fails one subtest.
    struct FailDriver;

    impl BrowserDriver for FailDriver {
        fn run_testharness(
            &self,
            test: &TestItem,
            _timeout: Duration,
        ) -> Result<HarnessResult, DriverError> {
            let mut report = green_report();
            report.test = test.id.clone();
            report.url = Some(test.url.clone());
            report.subtests[0].status = SubtestStatus::Fail;
            Ok(report)
        }

        fn load_without_crash(&self, _url: &str) -> Result<(), DriverError> {
            unimplemented!("expected-fail test only runs testharness tests")
        }

        fn render(&self, _url: &str, _viewport: Viewport) -> Result<Screenshot, DriverError> {
            unimplemented!("expected-fail test only runs testharness tests")
        }

        fn name(&self) -> &str {
            "fail-driver"
        }
    }

    #[test]
    fn expected_failures_tally_separately() {
        let m = manifest_of(vec![
            item("dom/a.html", TestKind::Testharness),
            item("dom/b.html", TestKind::Testharness),
        ]);
        let mut r = Runner::new(
            FailDriver,
            RunConfig {
                metadata: ExpectedMetadata::parse("[dom/a.html]\nexpected: FAIL\n").unwrap(),
                ..RunConfig::default()
            },
        );
        let report = r.run(&m);
        assert_eq!(report.summary.total, 2);
        // The expected failure is not counted as a failure.
        assert_eq!(report.summary.expected_fail, 1);
        assert_eq!(report.summary.harness_fail, 1);
        assert!(report.reports[0].expected_fail);
        assert!(!report.reports[1].expected_fail);
        let text = report.summary_text();
        assert!(text.contains("expected failures: 1"));
    }

    #[test]
    fn expected_failure_matching_rules() {
        let entry = |s| ExpectedEntry {
            expected: s,
            bug: None,
        };
        let harness = |status: HarnessStatus, subtests: Vec<SubtestResult>| {
            TestOutcome::Harness(HarnessResult {
                test: "x".into(),
                url: None,
                status,
                message: None,
                duration_ms: None,
                subtests,
            })
        };
        let failing_sub = || {
            vec![SubtestResult {
                name: "a".into(),
                status: SubtestStatus::Fail,
                message: None,
            }]
        };

        // FAIL matches a finished-but-failing result ...
        assert!(is_expected_failure(
            &harness(HarnessStatus::Ok, failing_sub()),
            &entry(ExpectedStatus::Fail)
        ));
        assert!(is_expected_failure(
            &harness(HarnessStatus::Fail, vec![]),
            &entry(ExpectedStatus::Fail)
        ));
        assert!(is_expected_failure(
            &TestOutcome::Reftest {
                passed: false,
                checks: vec![]
            },
            &entry(ExpectedStatus::Fail)
        ));
        assert!(is_expected_failure(
            &TestOutcome::Crashtest { passed: false },
            &entry(ExpectedStatus::Fail)
        ));
        // ... but not timeouts, errors, crashes, subtests-unknown, passes,
        // or genuinely unknown statuses ("indeterminate" is never "failed
        // as expected").
        assert!(!is_expected_failure(
            &harness(HarnessStatus::Timeout, vec![]),
            &entry(ExpectedStatus::Fail)
        ));
        assert!(!is_expected_failure(
            &harness(HarnessStatus::Error, vec![]),
            &entry(ExpectedStatus::Fail)
        ));
        assert!(!is_expected_failure(
            &harness(HarnessStatus::Crash, vec![]),
            &entry(ExpectedStatus::Fail)
        ));
        assert!(!is_expected_failure(
            &harness(HarnessStatus::SubtestsUnknown, vec![]),
            &entry(ExpectedStatus::Fail)
        ));
        assert!(!is_expected_failure(
            &harness(HarnessStatus::Unknown("FROB".into()), vec![]),
            &entry(ExpectedStatus::Fail)
        ));
        assert!(!is_expected_failure(
            &harness(HarnessStatus::Unknown("FROB".into()), vec![]),
            &entry(ExpectedStatus::Timeout)
        ));
        assert!(!is_expected_failure(
            &harness(HarnessStatus::Unknown("FROB".into()), vec![]),
            &entry(ExpectedStatus::Error)
        ));
        assert!(!is_expected_failure(
            &harness(HarnessStatus::Ok, vec![]),
            &entry(ExpectedStatus::Fail)
        ));
        // TIMEOUT and ERROR match only their own status.
        assert!(is_expected_failure(
            &harness(HarnessStatus::Timeout, vec![]),
            &entry(ExpectedStatus::Timeout)
        ));
        // A runner watchdog kill also satisfies `expected: TIMEOUT` — the
        // test timed out either way — but never `expected: FAIL`.
        let watchdog_kill = || TestOutcome::DriverTimeout {
            id: "x".into(),
            url: "/x".into(),
            after: Duration::from_millis(50),
        };
        assert!(is_expected_failure(
            &watchdog_kill(),
            &entry(ExpectedStatus::Timeout)
        ));
        assert!(!is_expected_failure(
            &watchdog_kill(),
            &entry(ExpectedStatus::Fail)
        ));
        assert!(!is_expected_failure(
            &harness(HarnessStatus::Error, vec![]),
            &entry(ExpectedStatus::Timeout)
        ));
        assert!(is_expected_failure(
            &harness(HarnessStatus::Error, vec![]),
            &entry(ExpectedStatus::Error)
        ));
        // Infrastructure outcomes never match.
        for outcome in [
            TestOutcome::Skipped { reason: "x".into() },
            TestOutcome::DriverFailed(DriverError::Unimplemented("x".into())),
            TestOutcome::FilesystemFailure("x".into()),
        ] {
            assert!(!is_expected_failure(&outcome, &entry(ExpectedStatus::Fail)));
            assert!(!is_expected_failure(
                &outcome,
                &entry(ExpectedStatus::Timeout)
            ));
            assert!(!is_expected_failure(
                &outcome,
                &entry(ExpectedStatus::Error)
            ));
        }
    }
}
