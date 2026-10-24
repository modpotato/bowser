//! End-to-end testdriver.js Action API (phase 1) against the real
//! `bowser` binary.
//!
//! Serves `tests/fixtures/wpt` over the std-only
//! [`FixtureServer`](bowser_app::wpt_driver::FixtureServer) (no external
//! network), loads `td_actions.html` in a headless child (`--check`),
//! and asserts the `TESTDRIVER_E2E` marker the page prints on the
//! child's stdout. The page drives the real upstream `testdriver.js`
//! (vendored at `tests/fixtures/wpt/resources/testdriver.js`) through
//! the vendor shim the fixture server appends and the native
//! `window.__bowser_testdriver` bridge.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use bowser_app::wpt_driver::BowserDriver;

/// Marker the fixture page prints once click + send_keys + bless +
/// in_automation have all reported.
const MARKER: &str = "TESTDRIVER_E2E ";
const EXPECTED: &str = "TESTDRIVER_E2E click=PASS(clicked) send_keys=PASS(down:a,up:a,down:Tab,up:Tab) bless=PASS(true) in_automation=PASS(true)";

/// Marker the unarmed fixture page prints: direct bridge calls without
/// the vendor shim must be refused (nothing dispatched).
const UNARMED_MARKER: &str = "TESTDRIVER_UNARMED ";
const UNARMED_EXPECTED: &str = "TESTDRIVER_UNARMED click=PASS(none) keys=PASS(none)";

/// Marker the action_sequence fixture prints once drag + click +
/// key-chord + wheel have all reported.
const ACTIONS2_MARKER: &str = "TESTDRIVER_ACTIONS2 ";
const ACTIONS2_EXPECTED: &str = "TESTDRIVER_ACTIONS2 drag=PASS(pointerdown@a,mousedown@a:shift=false,pointerup@b,mouseup@b) click=PASS(pointerdown@a,mousedown@a:shift=false,pointerup@a,mouseup@a,click@a) chord=PASS(keydown@a:Shift,keydown@a:a,pointerdown@a,mousedown@a:shift=true,pointerup@a,mouseup@a,click@a) wheel=PASS(120)";

/// Load `url` headless and capture the child's stdout
/// until the marker appears or the deadline passes.
fn run_child(url: &str, exe: &PathBuf, marker_prefix: &str) -> String {
    let mut child = Command::new(exe)
        .arg("--check")
        .arg(url)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn bowser --check");
    let stdout = child.stdout.take().expect("child stdout");
    let reader = BufReader::new(stdout);
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut marker = String::new();
    for line in reader.lines() {
        let line = line.unwrap_or_default();
        if line.starts_with(marker_prefix) {
            marker = line;
            break;
        }
        if Instant::now() > deadline {
            break;
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    marker
}

#[test]
fn testdriver_click_send_keys_bless_e2e() {
    let exe = PathBuf::from(env!("CARGO_BIN_EXE_bowser"));
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wpt");
    assert!(
        fixtures.join("td_actions.html").is_file(),
        "fixture page missing: {}",
        fixtures.join("td_actions.html").display()
    );
    assert!(
        fixtures.join("resources/testdriver.js").is_file(),
        "vendored testdriver.js missing"
    );
    let d = BowserDriver::new(exe.clone(), &fixtures).expect("fixture server failed to start");
    let url = format!("{}/td_actions.html", d.server_base);
    let marker = run_child(&url, &exe, MARKER);
    assert_eq!(marker, EXPECTED, "testdriver actions did not all pass");
}

#[test]
fn testdriver_unarmed_bridge_calls_refused_e2e() {
    // The vendor shim never runs on this page (no testdriver.js), so
    // the bridge stays unarmed: direct `__bowser_testdriver` calls must
    // produce no trusted dispatch and no user-activation grant.
    let exe = PathBuf::from(env!("CARGO_BIN_EXE_bowser"));
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wpt");
    assert!(
        fixtures.join("td_unarmed.html").is_file(),
        "fixture page missing: {}",
        fixtures.join("td_unarmed.html").display()
    );
    let d = BowserDriver::new(exe.clone(), &fixtures).expect("fixture server failed to start");
    let url = format!("{}/td_unarmed.html", d.server_base);
    let marker = run_child(&url, &exe, UNARMED_MARKER);
    assert_eq!(
        marker, UNARMED_EXPECTED,
        "unarmed bridge calls were not refused"
    );
}

#[test]
fn testdriver_action_sequence_e2e() {
    // Phase 2: test_driver.action_sequence drives the embedder
    // executor — a drag across two elements (no click), a click
    // (click after mouseup on the press target), a Shift chord
    // tick-aligned with a pointer press (modifier state reaches the
    // synthesized mousedown), and a wheel scroll (default scroll
    // moves window.scrollY).
    let exe = PathBuf::from(env!("CARGO_BIN_EXE_bowser"));
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wpt");
    assert!(
        fixtures.join("td_actions2.html").is_file(),
        "fixture page missing: {}",
        fixtures.join("td_actions2.html").display()
    );
    let d = BowserDriver::new(exe.clone(), &fixtures).expect("fixture server failed to start");
    let url = format!("{}/td_actions2.html", d.server_base);
    let marker = run_child(&url, &exe, ACTIONS2_MARKER);
    assert_eq!(
        marker, ACTIONS2_EXPECTED,
        "action_sequence did not all pass"
    );
}
