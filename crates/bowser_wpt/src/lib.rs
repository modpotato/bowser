//! Bowser's web-platform-tests harness.
//!
//! This crate is the project's scoreboard: it knows how to enumerate the
//! tests in a local [web-platform-tests](https://github.com/web-platform-tests/wpt)
//! checkout, how to parse the results the browser reports, and how to drive a
//! browser through the suite. The [`driver`] module exposes a pluggable
//! [`BrowserDriver`](driver::BrowserDriver) trait plus a
//! [`StubDriver`](driver::StubDriver) that fails every operation with
//! [`DriverError::Unimplemented`](driver::DriverError::Unimplemented),
//! handy for plumbing-only runs. Real wiring landed:
//! `crates/bowser_app/src/wpt_driver.rs` implements
//! [`BrowserDriver`](driver::BrowserDriver) as `BowserDriver`, gated behind
//! the `wpt-driver` cargo feature.
//!
//! Layout:
//!
//! - [`manifest`] — enumerate tests from `MANIFEST.json` or by walking a checkout.
//! - [`results`] — parse `testharness.js` completion reports.
//! - [`driver`] — pluggable browser-driver trait; [`StubDriver`](driver::StubDriver)
//!   for plumbing-only runs, real runs via `BowserDriver` in
//!   `crates/bowser_app/src/wpt_driver.rs` (feature `wpt-driver`).
//! - [`runner`] — run a filtered subset of the manifest against a driver.
//! - [`reftest`] — screenshot comparison primitives for reftests.
//!
//! Standard library only. No dependencies.

pub mod driver;
pub mod manifest;
pub mod reftest;
pub mod results;
pub mod runner;

mod json;
