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

use std::fmt;

use crate::json::{self, Json};

/// How the harness itself finished a testharness test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarnessStatus {
    /// The harness ran to completion (`done()` was called).
    Ok,
    /// The harness hit its timeout.
    Timeout,
    /// The page errored before finishing (uncaught exception, load failure).
    Error,
    /// The browser process died.
    Crash,
    /// Forward-compatible: a status string we do not know yet.
    Unknown(String),
}

impl HarnessStatus {
    fn parse(s: &str) -> HarnessStatus {
        match s {
            "OK" => HarnessStatus::Ok,
            "TIMEOUT" => HarnessStatus::Timeout,
            "ERROR" => HarnessStatus::Error,
            "CRASH" => HarnessStatus::Crash,
            other => HarnessStatus::Unknown(other.to_owned()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            HarnessStatus::Ok => "OK",
            HarnessStatus::Timeout => "TIMEOUT",
            HarnessStatus::Error => "ERROR",
            HarnessStatus::Crash => "CRASH",
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
    pub status: HarnessStatus,
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

    /// True when the harness finished and every subtest passed.
    pub fn is_green(&self) -> bool {
        self.status == HarnessStatus::Ok
            && !self.subtests.is_empty()
            && self
                .subtests
                .iter()
                .all(|t| t.status == SubtestStatus::Pass)
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
            "test" | "harness_status" | "duration_ms" | "tests"
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
    let duration_ms = doc.get("duration_ms").and_then(|v| match v {
        Json::Num(n) if *n >= 0.0 => Some(*n as u64),
        _ => None,
    });
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
        status,
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
        status,
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
}
