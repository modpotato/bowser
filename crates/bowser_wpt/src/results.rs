//! Parsing `testharness.js` completion reports.
//!
//! Two schemas, one per entry point, with no cross-fallback: each parser
//! accepts only its own schema and rejects the other, so a report from the
//! wrong schema can never be silently reinterpreted.
//!
//! The Bowser-native schema (emitted by the driver's injected completion
//! callback, accepted only by [`HarnessResult::from_json`]):
//!
//! ```json
//! {"test": "/dom/a.html", "harness_status": "OK", "duration_ms": 12,
//!  "tests": [{"name": "smoke", "status": "PASS", "message": null}]}
//! ```
//!
//! Required: `test` (string), `harness_status` (string; unknown strings are
//! kept as [`HarnessStatus::Unknown`]), `tests` (array of subtest objects,
//! each with required `name` and `status` strings and an optional `message`).
//! Optional: `duration_ms` (number; `null` or other types are ignored). Any
//! other field — including the wptreport names `status` and `subtests` — is
//! an error.
//!
//! The wptreport schema (produced by `wpt run --log-wptreport`, accepted
//! only by [`parse_wptreport`]):
//!
//! ```json
//! {"results": [{"test": "/a.html", "status": "OK",
//!               "subtests": [{"name": "one", "status": "PASS"}]}]}
//! ```
//!
//! Each entry requires `test` and `status` strings; `subtests` is optional
//! and has the same subtest shape as the native schema (required `name` and
//! `status`, optional `message`). Entries that are missing `test` or
//! `status`, or that carry the Bowser-native fields `harness_status` or
//! `tests`, are skipped and counted in the skip count — never parsed.
//! Other unknown fields on an entry are ignored, since real wptreport logs
//! carry many optional ones (`message`, `expected`, `duration`, ...).
//!
//! [`export_wptreport`] serializes a [`RunReport`](crate::runner::RunReport)
//! back into this schema, so a Bowser run can feed the same tooling that
//! consumes `wpt run --log-wptreport` output. The export round-trips
//! through [`parse_wptreport`].
//!
//! Expected-failure metadata ([`ExpectedMetadata`]) is a minimal ini
//! format, one file per suite under `metadata/`:
//!
//! ```ini
//! # lines starting with # or ; are comments
//! [dom/a.html]
//! expected: FAIL
//! bug: https://github.com/example/project/issues/1
//! ```
//!
//! - Section names are test ids; a leading `/` is stripped before
//!   matching, so `[dom/a.html]` and `[/dom/a.html]` are the same.
//! - `expected` is required per section and is one of `FAIL`, `TIMEOUT`,
//!   `ERROR`.
//! - `bug` is optional free text (usually a tracker URL).
//! - Unknown keys are ignored for forward compatibility. Malformed
//!   lines, key/value pairs outside any section, empty section names,
//!   duplicate sections, a missing `expected`, or an unknown `expected`
//!   value are parse errors.

use std::collections::HashMap;
use std::fmt;

use crate::json::{self, Json};
use crate::runner::{RunReport, TestOutcome};

/// How the harness itself finished a test.
///
/// Testharness tests report `OK`/`ERROR`/`TIMEOUT`/`PRECONDITION_FAILED`,
/// while reftest and crashtest tests report `PASS`/`FAIL` directly (they
/// have no subtests).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarnessStatus {
    /// The harness ran to completion (`done()` was called).
    Ok,
    /// The harness hit its timeout.
    Timeout,
    /// The page declared its preconditions unmet. Upstream WPT maps this
    /// to a skip, not a failure; the runner routes it to
    /// [`TestOutcome::Skipped`](crate::runner::TestOutcome::Skipped).
    PreconditionFailed,
    /// The page errored before finishing (uncaught exception, load failure).
    Error,
    /// The browser process died.
    Crash,
    /// The page loaded and ran, but no subtests were observed. This is what
    /// the v1 driver reports for a clean load: the completion-callback
    /// injection protocol does not exist yet, so "loaded" is all it can
    /// say. Deliberately not [`HarnessStatus::Ok`]: unlike a top-level `OK`
    /// from a real report, this state must never count as a pass (see
    /// [`HarnessResult::is_green`]).
    SubtestsUnknown,
    /// Reftest/crashtest outcome from `wptreport`: the test passed.
    Pass,
    /// Reftest/crashtest outcome from `wptreport`: the test failed.
    Fail,
    /// Forward-compatible: a status string we do not know yet.
    Unknown(String),
}

impl HarnessStatus {
    fn parse(s: &str) -> HarnessStatus {
        match s {
            "OK" => HarnessStatus::Ok,
            "TIMEOUT" => HarnessStatus::Timeout,
            "PRECONDITION_FAILED" => HarnessStatus::PreconditionFailed,
            "ERROR" => HarnessStatus::Error,
            "CRASH" => HarnessStatus::Crash,
            "SUBTESTS_UNKNOWN" => HarnessStatus::SubtestsUnknown,
            "PASS" => HarnessStatus::Pass,
            "FAIL" => HarnessStatus::Fail,
            other => HarnessStatus::Unknown(other.to_owned()),
        }
    }

    /// Whether this harness status means "the page declined to run its
    /// tests": upstream WPT reports these as skips, never as failures.
    ///
    /// The single classification point shared by the live runner
    /// ([`classify_harness_result`](crate::runner::classify_harness_result))
    /// and the wptreport import path, so both agree on what counts as a
    /// skip. Do not re-match on [`HarnessStatus::PreconditionFailed`]
    /// elsewhere; extend this predicate instead.
    pub fn is_skip(&self) -> bool {
        matches!(self, HarnessStatus::PreconditionFailed)
    }

    pub fn as_str(&self) -> &str {
        match self {
            HarnessStatus::Ok => "OK",
            HarnessStatus::Timeout => "TIMEOUT",
            HarnessStatus::PreconditionFailed => "PRECONDITION_FAILED",
            HarnessStatus::Error => "ERROR",
            HarnessStatus::Crash => "CRASH",
            HarnessStatus::SubtestsUnknown => "SUBTESTS_UNKNOWN",
            HarnessStatus::Pass => "PASS",
            HarnessStatus::Fail => "FAIL",
            HarnessStatus::Unknown(s) => s,
        }
    }
}

/// The outcome of one `test()` / `async_test()` inside a testharness page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubtestStatus {
    Pass,
    Fail,
    Timeout,
    NotRun,
    Unknown(String),
}

impl SubtestStatus {
    fn parse(s: &str) -> SubtestStatus {
        match s {
            "PASS" => SubtestStatus::Pass,
            "FAIL" => SubtestStatus::Fail,
            "TIMEOUT" => SubtestStatus::Timeout,
            "NOTRUN" => SubtestStatus::NotRun,
            other => SubtestStatus::Unknown(other.to_owned()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            SubtestStatus::Pass => "PASS",
            SubtestStatus::Fail => "FAIL",
            SubtestStatus::Timeout => "TIMEOUT",
            SubtestStatus::NotRun => "NOTRUN",
            SubtestStatus::Unknown(s) => s,
        }
    }
}

/// One subtest result: name, status, optional failure message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubtestResult {
    pub name: String,
    pub status: SubtestStatus,
    pub message: Option<String>,
}

/// The full result of running one testharness test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessResult {
    /// Test id, as in the manifest.
    pub test: String,
    /// Served URL the report claims to be for. Drivers fill this from the
    /// [`TestItem`](crate::manifest::TestItem) they ran; neither report
    /// schema carries it, so parsed reports leave it [`None`]. The runner
    /// checks the (`test`, `url`) pair for attribution: `.any.js`
    /// multi-global variants share one id with different urls, so the id
    /// alone is not enough.
    pub url: Option<String>,
    pub status: HarnessStatus,
    /// Harness-level message, e.g. the `TestsStatus.message` the
    /// completion callback carries on harness failure. Optional in the
    /// native schema; absent in older reports.
    pub message: Option<String>,
    pub duration_ms: Option<u64>,
    pub subtests: Vec<SubtestResult>,
}

impl HarnessResult {
    /// Parse one Bowser-native completion report.
    ///
    /// Accepts only the Bowser-native schema documented in the module docs:
    /// a missing or mistyped required field, or any unknown field (notably
    /// the wptreport names `status` and `subtests`), is an error.
    pub fn from_json(text: &str) -> Result<HarnessResult, ResultsError> {
        let doc = json::parse(text).map_err(|e| ResultsError::invalid(e.to_string()))?;
        parse_native_report(&doc)
    }

    /// True when the harness ran cleanly and nothing failed.
    ///
    /// A testharness run is green when the harness itself finished (`OK`)
    /// and every subtest passed. Reftest and crashtest entries from
    /// `wptreport` carry the outcome in the top-level status (`PASS`/`FAIL`)
    /// and usually have no subtests; those are green when the status is
    /// `PASS` and all of their (possibly zero) subtests passed. Like
    /// upstream WPT, a top-level `OK` with an empty subtest list counts as
    /// a pass: an empty `tests` array is not a failure.
    ///
    /// [`HarnessStatus::SubtestsUnknown`] — the v1 driver's "loaded, no
    /// subtests observed" — is never green: it is neither `OK` nor `PASS`,
    /// so a page whose assertions fail but which loads cleanly can never
    /// count as a pass through this path.
    pub fn is_green(&self) -> bool {
        match self.status {
            HarnessStatus::Ok | HarnessStatus::Pass => self
                .subtests
                .iter()
                .all(|t| t.status == SubtestStatus::Pass),
            _ => false,
        }
    }

    pub fn passed(&self) -> usize {
        self.subtests
            .iter()
            .filter(|t| t.status == SubtestStatus::Pass)
            .count()
    }

    pub fn failed(&self) -> usize {
        self.subtests
            .iter()
            .filter(|t| !matches!(t.status, SubtestStatus::Pass))
            .count()
    }
}

/// Parse one Bowser-native completion report.
///
/// Strict: only the fields `test`, `harness_status`, `duration_ms`, `tests`
/// exist in this schema. There is no fallback to the wptreport names
/// `status`/`subtests`; a document using those is rejected outright.
fn parse_native_report(doc: &Json) -> Result<HarnessResult, ResultsError> {
    let obj = doc
        .as_obj()
        .ok_or_else(|| ResultsError::invalid("report is not an object".into()))?;
    for (k, _) in obj {
        if !matches!(
            k.as_str(),
            "test" | "harness_status" | "message" | "duration_ms" | "tests"
        ) {
            return Err(ResultsError::invalid(format!(
                "report has unknown field {k:?}"
            )));
        }
    }
    let test = doc
        .get("test")
        .and_then(Json::as_str)
        .ok_or_else(|| ResultsError::invalid("report has no \"test\"".into()))?
        .to_owned();
    let status = doc
        .get("harness_status")
        .and_then(Json::as_str)
        .map(HarnessStatus::parse)
        .ok_or_else(|| ResultsError::invalid("report has no \"harness_status\"".into()))?;
    // Harness-level message (e.g. TestsStatus.message): optional, and a
    // JSON null is the same as absent.
    let message = doc.get("message").and_then(Json::as_str).map(str::to_owned);
    // Fractional durations round to the nearest millisecond instead of
    // silently truncating; negative durations are invalid and dropped.
    let duration_ms = doc
        .get("duration_ms")
        .and_then(Json::as_f64)
        .filter(|n| n.is_finite() && *n >= 0.0)
        .map(|n| n.round() as u64);
    let tests = doc
        .get("tests")
        .and_then(Json::as_arr)
        .ok_or_else(|| ResultsError::invalid("report has no \"tests\"".into()))?;
    let mut subtests = Vec::with_capacity(tests.len());
    for t in tests {
        subtests.push(parse_native_subtest(t)?);
    }
    Ok(HarnessResult {
        test,
        url: None,
        status,
        message,
        duration_ms,
        subtests,
    })
}

/// Parse one subtest of a Bowser-native report.
///
/// Strict: only `name`, `status`, `message` are allowed.
fn parse_native_subtest(t: &Json) -> Result<SubtestResult, ResultsError> {
    let obj = t
        .as_obj()
        .ok_or_else(|| ResultsError::invalid("subtest is not an object".into()))?;
    for (k, _) in obj {
        if !matches!(k.as_str(), "name" | "status" | "message") {
            return Err(ResultsError::invalid(format!(
                "subtest has unknown field {k:?}"
            )));
        }
    }
    let name = t
        .get("name")
        .and_then(Json::as_str)
        .ok_or_else(|| ResultsError::invalid("subtest has no \"name\"".into()))?
        .to_owned();
    let status = t
        .get("status")
        .and_then(Json::as_str)
        .map(SubtestStatus::parse)
        .ok_or_else(|| ResultsError::invalid("subtest has no \"status\"".into()))?;
    let message = t.get("message").and_then(Json::as_str).map(str::to_owned);
    Ok(SubtestResult {
        name,
        status,
        message,
    })
}

/// Parse one entry of a `wpt run --log-wptreport` document.
///
/// Strict: only the wptreport names `status`/`subtests` are read. Entries
/// carrying the Bowser-native fields `harness_status`/`tests` indicate a
/// schema mix-up and are rejected (never reinterpreted); all other unknown
/// fields are ignored since real wptreport logs carry many optional ones.
fn parse_wptreport_entry(r: &Json) -> Result<HarnessResult, ResultsError> {
    for key in ["harness_status", "tests"] {
        if r.get(key).is_some() {
            return Err(ResultsError::invalid(format!(
                "wptreport entry has Bowser-native field {key:?}"
            )));
        }
    }
    let test = r
        .get("test")
        .and_then(Json::as_str)
        .ok_or_else(|| ResultsError::invalid("wptreport entry has no \"test\"".into()))?
        .to_owned();
    let status = r
        .get("status")
        .and_then(Json::as_str)
        .map(HarnessStatus::parse)
        .ok_or_else(|| ResultsError::invalid("wptreport entry has no \"status\"".into()))?;
    let mut subtests = Vec::new();
    if let Some(tests) = r.get("subtests").and_then(Json::as_arr) {
        for t in tests {
            subtests.push(parse_wpt_subtest(t)?);
        }
    }
    Ok(HarnessResult {
        test,
        url: None,
        status,
        message: None,
        duration_ms: None,
        subtests,
    })
}

/// Parse one subtest of a wptreport entry.
///
/// Same field names as the native subtest shape (`name`, `status`,
/// `message`), so no cross-schema ambiguity can arise here; other unknown
/// fields are ignored like on the entry itself.
fn parse_wpt_subtest(t: &Json) -> Result<SubtestResult, ResultsError> {
    let name = t
        .get("name")
        .and_then(Json::as_str)
        .ok_or_else(|| ResultsError::invalid("subtest has no \"name\"".into()))?
        .to_owned();
    let status = t
        .get("status")
        .and_then(Json::as_str)
        .map(SubtestStatus::parse)
        .ok_or_else(|| ResultsError::invalid("subtest has no \"status\"".into()))?;
    let message = t.get("message").and_then(Json::as_str).map(str::to_owned);
    Ok(SubtestResult {
        name,
        status,
        message,
    })
}

/// Parse a `wpt run --log-wptreport` document into per-test results.
///
/// Accepts only the wptreport schema documented in the module docs: each
/// entry must carry `test` and `status` (with an optional `subtests` array).
/// Entries that do not — including entries in the Bowser-native shape — are
/// skipped and counted in the returned skip count. Unknown `status` strings
/// and missing `subtests` arrays are tolerated.
pub fn parse_wptreport(text: &str) -> Result<(Vec<HarnessResult>, usize), ResultsError> {
    let doc = json::parse(text).map_err(|e| ResultsError::invalid(e.to_string()))?;
    let results = doc
        .get("results")
        .and_then(Json::as_arr)
        .ok_or_else(|| ResultsError::invalid("wptreport has no \"results\" array".into()))?;
    let mut out = Vec::with_capacity(results.len());
    let mut skipped = 0;
    for r in results {
        match parse_wptreport_entry(r) {
            Ok(h) => out.push(h),
            Err(_) => skipped += 1,
        }
    }
    Ok((out, skipped))
}

/// A result report was malformed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResultsError {
    msg: String,
}

impl ResultsError {
    fn invalid(msg: String) -> ResultsError {
        ResultsError { msg }
    }
}

impl fmt::Display for ResultsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid result report: {}", self.msg)
    }
}

impl std::error::Error for ResultsError {}

/// An expected test status from expected-failure metadata.
///
/// Minimal subset of upstream WPT's expectation values: just the three
/// this runner can match against actual outcomes (see
/// [`is_expected_failure`](crate::runner::is_expected_failure)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectedStatus {
    /// The test is expected to run and fail (not green).
    Fail,
    /// The test is expected to time out.
    Timeout,
    /// The test is expected to error.
    Error,
}

impl ExpectedStatus {
    fn parse(s: &str) -> Option<ExpectedStatus> {
        match s {
            "FAIL" => Some(ExpectedStatus::Fail),
            "TIMEOUT" => Some(ExpectedStatus::Timeout),
            "ERROR" => Some(ExpectedStatus::Error),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            ExpectedStatus::Fail => "FAIL",
            ExpectedStatus::Timeout => "TIMEOUT",
            ExpectedStatus::Error => "ERROR",
        }
    }
}

/// One test's expected-failure metadata entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedEntry {
    pub expected: ExpectedStatus,
    /// Optional free text, usually a tracker URL.
    pub bug: Option<String>,
}

/// Expected-failure metadata: which tests are expected to fail and how.
///
/// Parsed from `metadata/*.ini` files (see the module docs for the
/// format). The runner consults this when tallying: a test whose actual
/// outcome matches its expectation is reported as expected-fail instead
/// of a failure (see
/// [`is_expected_failure`](crate::runner::is_expected_failure)).
#[derive(Debug, Clone, Default)]
pub struct ExpectedMetadata {
    entries: HashMap<String, ExpectedEntry>,
}

impl ExpectedMetadata {
    /// Parse one metadata ini document into expectations.
    ///
    /// Format rules are documented on the module docs. Section names are
    /// normalized (trimmed, leading `/` stripped) at parse time so
    /// lookups only strip the query side.
    pub fn parse(text: &str) -> Result<ExpectedMetadata, MetadataError> {
        struct Pending {
            name: String,
            expected: Option<ExpectedStatus>,
            bug: Option<String>,
        }
        fn flush(
            pending: &mut Option<Pending>,
            entries: &mut HashMap<String, ExpectedEntry>,
        ) -> Result<(), MetadataError> {
            if let Some(p) = pending.take() {
                let expected = p.expected.ok_or_else(|| {
                    MetadataError::invalid(format!("section [{}] has no \"expected\"", p.name))
                })?;
                if entries.contains_key(&p.name) {
                    return Err(MetadataError::invalid(format!(
                        "duplicate section [{}]",
                        p.name
                    )));
                }
                entries.insert(
                    p.name,
                    ExpectedEntry {
                        expected,
                        bug: p.bug,
                    },
                );
            }
            Ok(())
        }

        let mut entries = HashMap::new();
        let mut pending: Option<Pending> = None;
        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
                continue;
            }
            if let Some(name) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                flush(&mut pending, &mut entries)?;
                let name = name.trim().strip_prefix('/').unwrap_or(name.trim());
                if name.is_empty() {
                    return Err(MetadataError::invalid(format!(
                        "line {}: empty section name",
                        lineno + 1
                    )));
                }
                pending = Some(Pending {
                    name: name.to_owned(),
                    expected: None,
                    bug: None,
                });
                continue;
            }
            let Some(p) = pending.as_mut() else {
                return Err(MetadataError::invalid(format!(
                    "line {}: key/value outside any section",
                    lineno + 1
                )));
            };
            let (key, value) = split_kv(line).ok_or_else(|| {
                MetadataError::invalid(format!(
                    "line {}: expected \"key: value\", got {line:?}",
                    lineno + 1
                ))
            })?;
            match key {
                "expected" => {
                    p.expected = Some(ExpectedStatus::parse(value).ok_or_else(|| {
                        MetadataError::invalid(format!(
                            "line {}: unknown expected status {value:?} \
                             (want FAIL, TIMEOUT, or ERROR)",
                            lineno + 1
                        ))
                    })?);
                }
                "bug" => p.bug = Some(value.to_owned()),
                // Unknown keys are ignored for forward compatibility; a
                // misspelled `expected` still errors via the missing-
                // expected check at flush time.
                _ => {}
            }
        }
        flush(&mut pending, &mut entries)?;
        Ok(ExpectedMetadata { entries })
    }

    /// Merge another document's entries into this one. On duplicate test
    /// ids the other document wins; the CLI loads files in sorted order
    /// so the merge is deterministic.
    pub fn merge(&mut self, other: ExpectedMetadata) {
        self.entries.extend(other.entries);
    }

    /// The expectation for `test_id`, if any. A leading `/` is stripped
    /// before matching.
    pub fn expectation_for(&self, test_id: &str) -> Option<&ExpectedEntry> {
        let id = test_id.strip_prefix('/').unwrap_or(test_id);
        self.entries.get(id)
    }

    /// How many tests have expectations.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no test has an expectation.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Split an ini `key: value` (or `key = value`) line. Both the key and
/// the value must be non-empty after trimming.
fn split_kv(line: &str) -> Option<(&str, &str)> {
    let i = line.find([':', '='])?;
    let key = line[..i].trim();
    let value = line[i + 1..].trim();
    if key.is_empty() || value.is_empty() {
        None
    } else {
        Some((key, value))
    }
}

/// Expected-failure metadata was malformed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataError {
    msg: String,
}

impl MetadataError {
    fn invalid(msg: String) -> MetadataError {
        MetadataError { msg }
    }
}

impl fmt::Display for MetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid expected-failure metadata: {}", self.msg)
    }
}

impl std::error::Error for MetadataError {}

/// Serialize a [`RunReport`] as wptreport JSON.
///
/// The shape mirrors `wpt run --log-wptreport` closely enough that the
/// output round-trips through [`parse_wptreport`]:
///
/// ```json
/// {"run_info": {"product": "bowser", "os": "linux", "arch": "x86_64",
///               "driver": "bowser", "time_start": 1700000000000,
///               "time_end": 1700000060000},
///  "results": [{"test": "/a.html", "subsuite": "", "status": "OK",
///               "message": null, "expected": "FAIL",
///               "known_intermittent": [], "duration": 12,
///               "subtests": [{"name": "one", "status": "PASS",
///                             "message": null}]}]}
/// ```
///
/// Every test becomes one entry:
/// - testharness results keep their harness status, subtests, and
///   duration;
/// - runner watchdog timeouts report `TIMEOUT` with a message naming the
///   watchdog: upstream's status vocabulary has no distinct code for "the
///   runner killed the test before any report arrived", so the message
///   carries the disambiguation from the harness-reported `TIMEOUT`;
/// - reftests and crashtests report `PASS`/`FAIL` with no subtests;
/// - skips report `SKIP` with the skip reason as the message;
/// - driver and filesystem failures report `ERROR` with the error text
///   as the message.
///
/// Entries with metadata expectations carry `"expected"`, so an expected
/// failure reads as `"status": "FAIL", "expected": "FAIL"` exactly like
/// upstream. ([`ExpectedEntry::bug`] stays in the internal metadata
/// struct; upstream's schema has no `bug` field, so it is not
/// serialized.) Every entry carries `"subsuite": ""` like the upstream
/// formatter. Standard library only —
/// this crate takes no dependencies, so JSON is built by hand (see
/// [`JsonObj`]).
pub fn export_wptreport(report: &RunReport, metadata: &ExpectedMetadata) -> String {
    let mut out = String::from("{\"run_info\":");
    out.push_str(&run_info_json(
        &report.driver_name,
        report.time_start_ms,
        report.time_end_ms,
    ));
    out.push_str(",\"results\":[");
    for (i, r) in report.reports.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push_entry(&mut out, r, metadata.expectation_for(&r.id));
    }
    out.push_str("]}");
    out
}

fn run_info_json(driver_name: &str, time_start_ms: u64, time_end_ms: u64) -> String {
    let mut o = JsonObj::new();
    o.field_str("product", Some("bowser"));
    o.field_str("os", Some(std::env::consts::OS));
    o.field_str("arch", Some(std::env::consts::ARCH));
    o.field_str("driver", Some(driver_name));
    // Upstream's formatter always emits these as epoch millis.
    o.field_raw("time_start", &time_start_ms.to_string());
    o.field_raw("time_end", &time_end_ms.to_string());
    o.finish()
}

/// The wptreport fields for one test, extracted from its outcome before
/// JSON building. (A named struct instead of a tuple: the 4-tuple trips
/// `clippy::type_complexity`.)
struct ExportEntry {
    status: String,
    message: Option<String>,
    duration: Option<u64>,
    subtests: Vec<ExportSubtest>,
}

struct ExportSubtest {
    name: String,
    status: String,
    message: Option<String>,
}

/// One wptreport entry for a test report.
fn push_entry(out: &mut String, r: &crate::runner::TestReport, expected: Option<&ExpectedEntry>) {
    let entry = match &r.outcome {
        TestOutcome::Harness(h) => ExportEntry {
            status: h.status.as_str().to_owned(),
            message: None,
            duration: h.duration_ms,
            subtests: h
                .subtests
                .iter()
                .map(|s| ExportSubtest {
                    name: s.name.clone(),
                    status: s.status.as_str().to_owned(),
                    message: s.message.clone(),
                })
                .collect(),
        },
        TestOutcome::Reftest { passed, .. } => ExportEntry {
            status: if *passed { "PASS" } else { "FAIL" }.to_owned(),
            message: None,
            duration: None,
            subtests: Vec::new(),
        },
        TestOutcome::Crashtest { passed } => ExportEntry {
            status: if *passed { "PASS" } else { "FAIL" }.to_owned(),
            message: None,
            duration: None,
            subtests: Vec::new(),
        },
        TestOutcome::DriverTimeout { after, .. } => ExportEntry {
            status: "TIMEOUT".to_owned(),
            message: Some(format!(
                "runner watchdog expired after {after:?}: driver produced no harness report"
            )),
            duration: None,
            subtests: Vec::new(),
        },
        TestOutcome::Skipped { reason } => ExportEntry {
            status: "SKIP".to_owned(),
            message: Some(reason.clone()),
            duration: None,
            subtests: Vec::new(),
        },
        TestOutcome::DriverFailed(e) => ExportEntry {
            status: "ERROR".to_owned(),
            message: Some(e.to_string()),
            duration: None,
            subtests: Vec::new(),
        },
        TestOutcome::FilesystemFailure(e) => ExportEntry {
            status: "ERROR".to_owned(),
            message: Some(e.clone()),
            duration: None,
            subtests: Vec::new(),
        },
    };
    let mut subs = String::from("[");
    for (i, s) in entry.subtests.iter().enumerate() {
        if i > 0 {
            subs.push(',');
        }
        let mut o = JsonObj::new();
        o.field_str("name", Some(&s.name));
        o.field_str("status", Some(&s.status));
        o.field_str("message", s.message.as_deref());
        subs.push_str(&o.finish());
    }
    subs.push(']');

    let mut o = JsonObj::new();
    o.field_str("test", Some(&r.id));
    // Upstream's formatter always emits this, even though bowser never
    // runs named subsuites.
    o.field_str("subsuite", Some(""));
    o.field_str("status", Some(&entry.status));
    o.field_str("message", entry.message.as_deref());
    if let Some(d) = entry.duration {
        o.field_raw("duration", &d.to_string());
    }
    // No `bug` field: upstream's schema has no such key. The internal
    // [`ExpectedEntry::bug`] is kept for other consumers.
    if let Some(e) = expected {
        o.field_str("expected", Some(e.expected.as_str()));
    }
    o.field_raw("subtests", &subs);
    o.field_raw("known_intermittent", "[]");
    out.push_str(&o.finish());
}

/// Tiny JSON object builder: comma tracking plus string escaping, so
/// callers cannot emit malformed JSON by forgetting either.
struct JsonObj {
    buf: String,
    first: bool,
}

impl JsonObj {
    fn new() -> JsonObj {
        JsonObj {
            buf: String::from("{"),
            first: true,
        }
    }

    fn key(&mut self, key: &str) {
        if !self.first {
            self.buf.push(',');
        }
        self.first = false;
        push_json_str(&mut self.buf, key);
        self.buf.push(':');
    }

    /// `field_str("message", None)` writes `"message":null`.
    fn field_str(&mut self, key: &str, value: Option<&str>) {
        self.key(key);
        match value {
            Some(v) => push_json_str(&mut self.buf, v),
            None => self.buf.push_str("null"),
        }
    }

    /// `raw_json` is inserted verbatim; the caller guarantees it is
    /// valid JSON.
    fn field_raw(&mut self, key: &str, raw_json: &str) {
        self.key(key);
        self.buf.push_str(raw_json);
    }

    fn finish(mut self) -> String {
        self.buf.push('}');
        self.buf
    }
}

/// Append `s` as a JSON string literal with escaping. `str` is valid
/// UTF-8 by construction, so no surrogate handling is needed; control
/// characters below U+0020 are escaped per the JSON grammar.
fn push_json_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPORT: &str = r#"{
        "test": "/dom/a.html",
        "harness_status": "OK",
        "duration_ms": 12,
        "tests": [
            {"name": "smoke", "status": "PASS", "message": null},
            {"name": "broken", "status": "FAIL", "message": "expected true, got false"}
        ]
    }"#;

    #[test]
    fn parses_bowser_report() {
        let r = HarnessResult::from_json(REPORT).unwrap();
        assert_eq!(r.test, "/dom/a.html");
        assert_eq!(r.status, HarnessStatus::Ok);
        assert_eq!(r.duration_ms, Some(12));
        assert_eq!(r.subtests.len(), 2);
        assert_eq!(r.subtests[0].name, "smoke");
        assert_eq!(r.subtests[0].status, SubtestStatus::Pass);
        assert_eq!(r.subtests[0].message, None);
        assert_eq!(r.subtests[1].status, SubtestStatus::Fail);
        assert_eq!(
            r.subtests[1].message.as_deref(),
            Some("expected true, got false")
        );
        assert_eq!(r.passed(), 1);
        assert_eq!(r.failed(), 1);
        assert!(!r.is_green());
    }

    #[test]
    fn green_requires_all_pass() {
        let r = HarnessResult::from_json(
            r#"{"test": "x", "harness_status": "OK",
                "tests": [{"name": "a", "status": "PASS"}]}"#,
        )
        .unwrap();
        assert!(r.is_green());
    }

    #[test]
    fn harness_message_parses_and_defaults_to_none() {
        let r = HarnessResult::from_json(
            r#"{"test": "x", "harness_status": "ERROR",
                "message": "boom", "tests": []}"#,
        )
        .unwrap();
        assert_eq!(r.status, HarnessStatus::Error);
        assert_eq!(r.message.as_deref(), Some("boom"));
        // Absent and JSON null both parse to None.
        for doc in [
            r#"{"test": "x", "harness_status": "OK", "tests": []}"#,
            r#"{"test": "x", "harness_status": "OK", "message": null, "tests": []}"#,
        ] {
            let r = HarnessResult::from_json(doc).unwrap();
            assert_eq!(r.message, None);
        }
    }

    #[test]
    fn ok_with_no_subtests_is_green() {
        // Upstream treats a top-level OK as a pass even when the page
        // reported no subtests.
        let r = HarnessResult::from_json(r#"{"test": "x", "harness_status": "OK", "tests": []}"#)
            .unwrap();
        assert!(r.is_green());
    }

    #[test]
    fn subtests_unknown_is_never_green() {
        // The v1 driver's "loaded, subtests unknown": an empty subtest
        // list with this status must not count as a pass the way a
        // top-level OK with zero subtests does.
        let r = HarnessResult {
            test: "x".into(),
            url: Some("/x".into()),
            status: HarnessStatus::SubtestsUnknown,
            message: None,
            duration_ms: Some(1),
            subtests: Vec::new(),
        };
        assert!(!r.is_green());
        assert_eq!(HarnessStatus::SubtestsUnknown.as_str(), "SUBTESTS_UNKNOWN");
        // Round-trips through the native schema's status strings.
        let parsed = HarnessResult::from_json(
            r#"{"test": "x", "harness_status": "SUBTESTS_UNKNOWN", "tests": []}"#,
        )
        .unwrap();
        assert_eq!(parsed.status, HarnessStatus::SubtestsUnknown);
        assert!(!parsed.is_green());
        // Failing assertions behind a clean load are still not a pass:
        // the status alone decides, never the subtest list.
        let r = HarnessResult {
            subtests: vec![SubtestResult {
                name: "a".into(),
                status: SubtestStatus::Fail,
                message: Some("assertion failed, but the page loaded".into()),
            }],
            ..r
        };
        assert!(!r.is_green());
        assert!(!r.status.is_skip());
    }

    #[test]
    fn precondition_failed_parses_and_is_not_green() {
        let r = HarnessResult::from_json(
            r#"{"test": "x", "harness_status": "PRECONDITION_FAILED",
                "tests": [{"name": "a", "status": "PASS"}]}"#,
        )
        .unwrap();
        assert_eq!(r.status, HarnessStatus::PreconditionFailed);
        assert_eq!(r.status.as_str(), "PRECONDITION_FAILED");
        assert!(!r.is_green());
        // The shared skip classification: the runner and the wptreport
        // import path both route through `is_skip`, never a local match.
        assert!(r.status.is_skip());
        assert!(!HarnessStatus::Ok.is_skip());
        assert!(!HarnessStatus::Error.is_skip());
    }

    #[test]
    fn timeout_and_unknown_statuses() {
        let r = HarnessResult::from_json(
            r#"{"test": "x", "harness_status": "TIMEOUT",
                "tests": [{"name": "a", "status": "TIMEOUT"},
                          {"name": "b", "status": "WEIRD"}]}"#,
        )
        .unwrap();
        assert_eq!(r.status, HarnessStatus::Timeout);
        assert_eq!(r.subtests[0].status, SubtestStatus::Timeout);
        assert_eq!(r.subtests[1].status, SubtestStatus::Unknown("WEIRD".into()));
        assert_eq!(HarnessStatus::Unknown("Z".into()).as_str(), "Z");
        assert_eq!(SubtestStatus::NotRun.as_str(), "NOTRUN");
    }

    #[test]
    fn native_schema_requires_tests() {
        assert!(HarnessResult::from_json(r#"{"test": "x", "harness_status": "OK"}"#).is_err());
    }

    #[test]
    fn from_json_rejects_wptreport_shape() {
        // no silent cross-fallback: the wptreport names are unknown here.
        let wpt_shaped = r#"{"test": "/a.html", "status": "OK",
            "subtests": [{"name": "one", "status": "PASS"}]}"#;
        assert!(HarnessResult::from_json(wpt_shaped).is_err());
    }

    #[test]
    fn from_json_rejects_unrelated_status_field() {
        // the W14 scenario: missing harness_status plus an unrelated status
        // field must not parse "successfully".
        let report = r#"{"test": "/a.html", "status": "SOMETHING_ELSE",
            "tests": [{"name": "one", "status": "PASS"}]}"#;
        assert!(HarnessResult::from_json(report).is_err());
    }

    #[test]
    fn from_json_rejects_unknown_fields() {
        assert!(
            HarnessResult::from_json(
                r#"{"test": "x", "harness_status": "OK", "bogus": 1, "tests": []}"#,
            )
            .is_err()
        );
        assert!(
            HarnessResult::from_json(
                r#"{"test": "x", "harness_status": "OK",
                "tests": [{"name": "a", "status": "PASS", "extra": 1}]}"#,
            )
            .is_err()
        );
    }

    #[test]
    fn from_json_rejects_mistyped_required_fields() {
        assert!(
            HarnessResult::from_json(r#"{"test": "x", "harness_status": 42, "tests": []}"#,)
                .is_err()
        );
        assert!(
            HarnessResult::from_json(r#"{"test": "x", "harness_status": "OK", "tests": {}}"#,)
                .is_err()
        );
        assert!(
            HarnessResult::from_json(
                r#"{"test": "x", "harness_status": "OK",
                "tests": [{"name": "a"}]}"#,
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_malformed() {
        assert!(HarnessResult::from_json("nope").is_err());
        assert!(HarnessResult::from_json(r#"{"harness_status": "OK"}"#).is_err());
        let e = HarnessResult::from_json("nope").unwrap_err();
        assert!(e.to_string().contains("invalid result report"));
    }

    #[test]
    fn parses_wptreport() {
        let doc = r#"{
            "run_info": {"os": "linux"},
            "results": [
                {"test": "/a.html", "status": "OK", "message": null,
                 "subtests": [{"name": "one", "status": "PASS", "message": null}]},
                {"test": "/b.html", "status": "TIMEOUT", "subtests": []},
                {"no-test-field": true}
            ]
        }"#;
        let (results, skipped) = parse_wptreport(doc).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(skipped, 1);
        assert_eq!(results[0].test, "/a.html");
        assert_eq!(results[0].status, HarnessStatus::Ok);
        assert!(results[0].is_green());
        assert_eq!(results[1].status, HarnessStatus::Timeout);
    }

    #[test]
    fn wptreport_requires_results_array() {
        assert!(parse_wptreport(r#"{"run_info": {}}"#).is_err());
    }

    #[test]
    fn duration_ms_rounds_and_rejects_negative() {
        let r = HarnessResult::from_json(
            r#"{"test": "x", "harness_status": "OK", "duration_ms": 12.7,
                "tests": [{"name": "a", "status": "PASS"}]}"#,
        )
        .unwrap();
        assert_eq!(r.duration_ms, Some(13));
        let r = HarnessResult::from_json(
            r#"{"test": "x", "harness_status": "OK", "duration_ms": -5,
                "tests": [{"name": "a", "status": "PASS"}]}"#,
        )
        .unwrap();
        assert_eq!(r.duration_ms, None);
        let r = HarnessResult::from_json(
            r#"{"test": "x", "harness_status": "OK", "duration_ms": "12",
                "tests": [{"name": "a", "status": "PASS"}]}"#,
        )
        .unwrap();
        assert_eq!(r.duration_ms, None);
    }

    #[test]
    fn wptreport_maps_reftest_and_crashtest_statuses() {
        // Realistic mixed `--log-wptreport` output: testharness entries use
        // OK/ERROR at the top level with PASS/FAIL subtests, while reftest
        // and crashtest entries report PASS/FAIL at the top level with no
        // subtests.
        let doc = r#"{
            "run_info": {"os": "linux", "product": "bowser"},
            "results": [
                {"test": "/dom/a.html", "status": "OK", "message": null,
                 "expected": "OK", "known_intermittent": [],
                 "subtests": [{"name": "smoke", "status": "PASS",
                               "message": null, "expected": "PASS",
                               "known_intermittent": []}]},
                {"test": "/css/css-transforms/2d-rotate-001.html", "status": "PASS",
                 "message": null, "expected": "PASS", "known_intermittent": [],
                 "subtests": []},
                {"test": "/css/css-transforms/2d-rotate-002.html", "status": "FAIL",
                 "message": "REFTEST IMAGE 1 (TEST): image comparison failed",
                 "expected": "PASS", "known_intermittent": [],
                 "subtests": []},
                {"test": "/crash/crash-001.html", "status": "PASS",
                 "message": null, "expected": "PASS", "known_intermittent": [],
                 "subtests": []},
                {"test": "/dom/b.html", "status": "ERROR",
                 "message": "uncaught exception", "expected": "OK",
                 "known_intermittent": [], "subtests": []}
            ]
        }"#;
        let (results, skipped) = parse_wptreport(doc).unwrap();
        assert_eq!(skipped, 0);
        assert_eq!(results.len(), 5);

        assert_eq!(results[0].status, HarnessStatus::Ok);
        assert!(results[0].is_green());

        assert_eq!(results[1].status, HarnessStatus::Pass);
        assert!(results[1].is_green());
        assert_eq!(results[1].passed(), 0);
        assert_eq!(results[1].failed(), 0);

        assert_eq!(results[2].status, HarnessStatus::Fail);
        assert!(!results[2].is_green());

        assert_eq!(results[3].status, HarnessStatus::Pass);
        assert!(results[3].is_green());

        assert_eq!(results[4].status, HarnessStatus::Error);
        assert!(!results[4].is_green());

        assert_eq!(HarnessStatus::Pass.as_str(), "PASS");
        assert_eq!(HarnessStatus::Fail.as_str(), "FAIL");
    }

    #[test]
    fn wptreport_skips_native_shape() {
        // no silent cross-fallback in the other direction: a Bowser-native
        // entry is skipped, not reinterpreted.
        let doc = r#"{"results": [
            {"test": "/a.html", "harness_status": "OK",
             "tests": [{"name": "one", "status": "PASS"}]},
            {"test": "/b.html", "status": "OK",
             "subtests": [{"name": "two", "status": "PASS"}]},
            {"test": "/c.html", "subtests": []}
        ]}"#;
        let (results, skipped) = parse_wptreport(doc).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(skipped, 2);
        assert_eq!(results[0].test, "/b.html");
        assert_eq!(results[0].status, HarnessStatus::Ok);
        assert!(results[0].is_green());
    }

    #[test]
    fn metadata_parses_minimal_ini() {
        let m = ExpectedMetadata::parse(
            "# a comment\n\
             ; another comment\n\
             \n\
             [dom/a.html]\n\
             expected: FAIL\n\
             bug: https://example.com/issue/1\n\
             \n\
             [/css/b.html]\n\
             expected = TIMEOUT\n",
        )
        .unwrap();
        assert_eq!(m.len(), 2);
        assert!(!m.is_empty());
        let a = m.expectation_for("dom/a.html").unwrap();
        assert_eq!(a.expected, ExpectedStatus::Fail);
        assert_eq!(a.bug.as_deref(), Some("https://example.com/issue/1"));
        // Leading slashes are normalized on both sides.
        assert!(m.expectation_for("/dom/a.html").is_some());
        let b = m.expectation_for("/css/b.html").unwrap();
        assert_eq!(b.expected, ExpectedStatus::Timeout);
        assert_eq!(b.bug, None);
        assert!(m.expectation_for("dom/c.html").is_none());
        assert_eq!(ExpectedStatus::Error.as_str(), "ERROR");
    }

    #[test]
    fn metadata_rejects_bad_documents() {
        // missing expected
        assert!(ExpectedMetadata::parse("[a.html]\nbug: x\n").is_err());
        // unknown expected value
        assert!(ExpectedMetadata::parse("[a.html]\nexpected: SOMETIMES\n").is_err());
        // key/value outside any section
        assert!(ExpectedMetadata::parse("expected: FAIL\n").is_err());
        // malformed line
        assert!(ExpectedMetadata::parse("[a.html]\njust some words\n").is_err());
        // empty section name
        assert!(ExpectedMetadata::parse("[]\nexpected: FAIL\n").is_err());
        // duplicate sections
        assert!(
            ExpectedMetadata::parse("[a.html]\nexpected: FAIL\n[a.html]\nexpected: ERROR\n")
                .is_err()
        );
        // unknown keys are ignored, not errors
        let m = ExpectedMetadata::parse("[a.html]\nexpected: ERROR\nflavor: spicy\n").unwrap();
        assert_eq!(
            m.expectation_for("a.html").unwrap().expected,
            ExpectedStatus::Error
        );
    }

    #[test]
    fn metadata_merge_prefers_later_files() {
        let mut a = ExpectedMetadata::parse("[x.html]\nexpected: FAIL\n").unwrap();
        let b = ExpectedMetadata::parse("[x.html]\nexpected: TIMEOUT\n").unwrap();
        a.merge(b);
        assert_eq!(
            a.expectation_for("x.html").unwrap().expected,
            ExpectedStatus::Timeout
        );
        assert_eq!(a.len(), 1);
        assert!(ExpectedMetadata::default().is_empty());
    }

    /// Build a [`RunReport`](crate::runner::RunReport) with one of every
    /// outcome, for export tests.
    fn mixed_report() -> crate::runner::RunReport {
        use crate::runner::{TestOutcome, TestReport};
        let harness = |test: &str, status: HarnessStatus, subtests: Vec<SubtestResult>| {
            TestOutcome::Harness(HarnessResult {
                test: test.into(),
                url: None,
                status,
                message: None,
                duration_ms: Some(7),
                subtests,
            })
        };
        let sub = |name: &str, status: SubtestStatus, message: Option<&str>| SubtestResult {
            name: name.into(),
            status,
            message: message.map(str::to_owned),
        };
        let reports = vec![
            TestReport {
                id: "/dom/a.html".into(),
                kind: crate::manifest::TestKind::Testharness,
                outcome: harness(
                    "/dom/a.html",
                    HarnessStatus::Ok,
                    vec![
                        sub("one", SubtestStatus::Pass, None),
                        sub("two", SubtestStatus::Fail, Some("boom \"quoted\"\nnewline")),
                    ],
                ),
                expected_fail: false,
            },
            TestReport {
                id: "/css/r.html".into(),
                kind: crate::manifest::TestKind::Reftest,
                outcome: TestOutcome::Reftest {
                    passed: false,
                    checks: Vec::new(),
                },
                expected_fail: true,
            },
            TestReport {
                id: "/crash/c.html".into(),
                kind: crate::manifest::TestKind::Crashtest,
                outcome: TestOutcome::Crashtest { passed: true },
                expected_fail: false,
            },
            TestReport {
                id: "/dom/s.html".into(),
                kind: crate::manifest::TestKind::Testharness,
                outcome: TestOutcome::Skipped {
                    reason: "manual tests need a human".into(),
                },
                expected_fail: false,
            },
            TestReport {
                id: "/dom/d.html".into(),
                kind: crate::manifest::TestKind::Testharness,
                outcome: TestOutcome::DriverFailed(crate::driver::DriverError::Timeout {
                    test: "/dom/d.html".into(),
                    after: std::time::Duration::from_secs(1),
                }),
                expected_fail: false,
            },
        ];
        crate::runner::RunReport {
            reports,
            summary: crate::runner::RunSummary::default(),
            driver_name: "fake".into(),
            time_start_ms: 1_700_000_000_000,
            time_end_ms: 1_700_000_060_000,
        }
    }

    #[test]
    fn export_wptreport_has_expected_shape() {
        let metadata =
            ExpectedMetadata::parse("[dom/a.html]\nexpected: FAIL\nbug: https://ex.com/9\n")
                .unwrap();
        let json = export_wptreport(&mixed_report(), &metadata);
        let doc = crate::json::parse(&json).unwrap();
        let run_info = doc.get("run_info").unwrap();
        assert_eq!(
            run_info.get("product").and_then(crate::json::Json::as_str),
            Some("bowser")
        );
        assert_eq!(
            run_info.get("driver").and_then(crate::json::Json::as_str),
            Some("fake")
        );
        // Upstream's formatter always emits epoch-millis timestamps.
        assert_eq!(
            run_info
                .get("time_start")
                .and_then(crate::json::Json::as_i64),
            Some(1_700_000_000_000)
        );
        assert_eq!(
            run_info.get("time_end").and_then(crate::json::Json::as_i64),
            Some(1_700_000_060_000)
        );
        let results = doc
            .get("results")
            .and_then(crate::json::Json::as_arr)
            .unwrap();
        assert_eq!(results.len(), 5);

        let get = |i: usize, k: &str| results[i].get(k).and_then(crate::json::Json::as_str);
        // Harness entry keeps status, subtests, duration, and the metadata expectation.
        assert_eq!(get(0, "test"), Some("/dom/a.html"));
        assert_eq!(get(0, "subsuite"), Some(""));
        assert_eq!(get(0, "status"), Some("OK"));
        assert_eq!(get(0, "expected"), Some("FAIL"));
        // `bug` is not part of the upstream schema: it stays internal.
        assert!(results[0].get("bug").is_none());
        let subs = results[0]
            .get("subtests")
            .and_then(crate::json::Json::as_arr)
            .unwrap();
        assert_eq!(subs.len(), 2);
        assert_eq!(
            subs[1].get("message").and_then(crate::json::Json::as_str),
            Some("boom \"quoted\"\nnewline")
        );
        // Reftest/crashtest map to PASS/FAIL with no subtests.
        assert_eq!(get(1, "status"), Some("FAIL"));
        assert_eq!(get(1, "expected"), None);
        assert_eq!(get(2, "status"), Some("PASS"));
        // Skips and driver failures carry their reasons as messages.
        assert_eq!(get(3, "status"), Some("SKIP"));
        assert!(get(3, "message").unwrap().contains("manual"));
        assert_eq!(get(4, "status"), Some("ERROR"));
        assert!(get(4, "message").unwrap().contains("/dom/d.html"));
    }

    #[test]
    fn export_round_trips_through_parse_wptreport() {
        let metadata = ExpectedMetadata::parse("[/css/r.html]\nexpected: FAIL\n").unwrap();
        let report = mixed_report();
        let json = export_wptreport(&report, &metadata);
        let (results, skipped) = parse_wptreport(&json).unwrap();
        assert_eq!(skipped, 0);
        assert_eq!(results.len(), report.reports.len());
        // Statuses survive the round trip, including the non-standard ones
        // (SKIP, SUBTESTS_UNKNOWN) via the Unknown/parse path.
        let statuses: Vec<&str> = results.iter().map(|r| r.status.as_str()).collect();
        assert_eq!(statuses, vec!["OK", "FAIL", "PASS", "SKIP", "ERROR"]);
        assert_eq!(results[0].subtests.len(), 2);
        assert_eq!(results[0].subtests[1].status, SubtestStatus::Fail);
        assert_eq!(
            results[0].subtests[1].message.as_deref(),
            Some("boom \"quoted\"\nnewline")
        );
        // `duration` is informational only: the import path does not read
        // it back (the native schema's `duration_ms` is a different field).
        assert_eq!(results[0].duration_ms, None);
        assert!(results[3].status.is_skip() || !results[3].is_green());
    }
}
