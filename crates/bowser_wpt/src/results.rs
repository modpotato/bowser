//! Parsing `testharness.js` completion reports.
//!
//! When the driver finishes a testharness test it produces one JSON report.
//! The Bowser-native shape (emitted by the driver's injected completion
//! callback) is:
//!
//! ```json
//! {"test": "/dom/a.html", "harness_status": "OK", "duration_ms": 12,
//!  "tests": [{"name": "smoke", "status": "PASS", "message": null}]}
//! ```
//!
//! [`parse_wptreport`] additionally reads the `wpt run --log-wptreport`
//! format (`{"results": [{"test": ..., "status": ..., "subtests": [...]}]}`)
//! so runs recorded by other harnesses can be imported.

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
    pub fn from_json(text: &str) -> Result<HarnessResult, ResultsError> {
        let doc = json::parse(text).map_err(|e| ResultsError::invalid(e.to_string()))?;
        parse_report(&doc)
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

fn parse_report(doc: &Json) -> Result<HarnessResult, ResultsError> {
    let test = doc
        .get("test")
        .and_then(Json::as_str)
        .ok_or_else(|| ResultsError::invalid("report has no \"test\"".into()))?
        .to_owned();
    let status = doc
        .get("harness_status")
        .or_else(|| doc.get("status"))
        .and_then(Json::as_str)
        .map(HarnessStatus::parse)
        .unwrap_or(HarnessStatus::Unknown("missing".into()));
    // Fractional durations round to the nearest millisecond instead of
    // silently truncating; negative durations are invalid and dropped.
    let duration_ms = doc
        .get("duration_ms")
        .and_then(Json::as_f64)
        .filter(|n| n.is_finite() && *n >= 0.0)
        .map(|n| n.round() as u64);
    let mut subtests = Vec::new();
    if let Some(tests) = doc
        .get("tests")
        .or_else(|| doc.get("subtests"))
        .and_then(Json::as_arr)
    {
        for t in tests {
            subtests.push(parse_subtest(t)?);
        }
    }
    Ok(HarnessResult {
        test,
        status,
        duration_ms,
        subtests,
    })
}

fn parse_subtest(t: &Json) -> Result<SubtestResult, ResultsError> {
    let name = t
        .get("name")
        .and_then(Json::as_str)
        .ok_or_else(|| ResultsError::invalid("subtest has no \"name\"".into()))?
        .to_owned();
    let status = t
        .get("status")
        .and_then(Json::as_str)
        .map(SubtestStatus::parse)
        .unwrap_or(SubtestStatus::Unknown("missing".into()));
    let message = t.get("message").and_then(Json::as_str).map(str::to_owned);
    Ok(SubtestResult {
        name,
        status,
        message,
    })
}

/// Parse a `wpt run --log-wptreport` document into per-test results.
///
/// Unknown `status` strings and missing `subtests` arrays are tolerated;
/// entries without a `test` field are skipped and counted in the returned
/// skip count.
pub fn parse_wptreport(text: &str) -> Result<(Vec<HarnessResult>, usize), ResultsError> {
    let doc = json::parse(text).map_err(|e| ResultsError::invalid(e.to_string()))?;
    let results = doc
        .get("results")
        .and_then(Json::as_arr)
        .ok_or_else(|| ResultsError::invalid("wptreport has no \"results\" array".into()))?;
    let mut out = Vec::with_capacity(results.len());
    let mut skipped = 0;
    for r in results {
        match parse_report(r) {
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
    fn missing_subtests_is_empty() {
        let r = HarnessResult::from_json(r#"{"test": "x", "harness_status": "OK"}"#).unwrap();
        assert!(r.subtests.is_empty());
        assert!(!r.is_green());
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
}
