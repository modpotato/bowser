//! Pluggable browser driver.
//!
//! The runner never talks to a browser directly; it talks to a
//! [`BrowserDriver`]. Today the only implementation is [`StubDriver`], which
//! fails every call with [`DriverError::Unimplemented`]. The real driver
//! (spawning `bowser_app` headless, serving tests over `wptserve`-compatible
//! HTTP, injecting the completion callback, capturing screenshots) lands
//! with the renderer.

use std::fmt;
use std::time::Duration;

use crate::manifest::TestItem;
use crate::results::HarnessResult;

/// Viewport used when rendering a test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
}

impl Viewport {
    /// The default WPT viewport (800x600).
    pub const DEFAULT: Viewport = Viewport {
        width: 800,
        height: 600,
    };
}

/// An 8-bit RGBA screenshot captured by the driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Screenshot {
    pub width: u32,
    pub height: u32,
    /// Row-major RGBA bytes, `width * height * 4` long.
    pub rgba: Vec<u8>,
}

impl Screenshot {
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> Result<Screenshot, DriverError> {
        let expected = width as usize * height as usize * 4;
        if rgba.len() != expected {
            return Err(DriverError::BadScreenshot {
                expected,
                got: rgba.len(),
            });
        }
        Ok(Screenshot {
            width,
            height,
            rgba,
        })
    }

    /// Solid-color screenshot, handy for tests and for drivers that cannot
    /// render yet but want to exercise the reftest plumbing.
    pub fn solid(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> Screenshot {
        let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
        for _ in 0..width as usize * height as usize {
            rgba.extend_from_slice(&[r, g, b, a]);
        }
        Screenshot {
            width,
            height,
            rgba,
        }
    }
}

/// How the driver talks to the browser under test.
///
/// All methods take `&self`: a driver is expected to be shareable between
/// threads, so a future parallel runner can hand one driver to many
/// workers. An implementation that holds mutable session state (a browser
/// process handle, an HTTP server) synchronizes it internally, e.g. behind
/// a `Mutex`.
pub trait BrowserDriver {
    /// Load `test.url`, wait for the `testharness.js` completion callback,
    /// and return the parsed report. Must respect `timeout`.
    fn run_testharness(
        &self,
        test: &TestItem,
        timeout: Duration,
    ) -> Result<HarnessResult, DriverError>;

    /// Render `url` in `viewport` and return the screenshot.
    fn render(&self, url: &str, viewport: Viewport) -> Result<Screenshot, DriverError>;

    /// Load `url` and return `Ok(())` if the browser survives the load,
    /// `Err(DriverError::BrowserGone)` if it crashes. Crashtests assert the
    /// browser does not die on their content; other test kinds go through
    /// [`BrowserDriver::run_testharness`] or [`BrowserDriver::render`].
    ///
    /// The default implementation fails with [`DriverError::Unimplemented`]
    /// so drivers that do not support crashtests need not touch this.
    fn load_without_crash(&self, url: &str) -> Result<(), DriverError> {
        Err(DriverError::Unimplemented(format!(
            "load_without_crash is not implemented by this driver ({url})"
        )))
    }

    /// Human-readable name for logs (e.g. "bowser-headless/0.1.0").
    fn name(&self) -> &str {
        "unknown"
    }
}

/// A driver that is not wired to any browser yet.
///
/// Every operation fails with [`DriverError::Unimplemented`]. It exists so
/// the runner, the CLI, and the report plumbing can be built and tested
/// before the renderer exists.
#[derive(Debug, Default, Clone, Copy)]
pub struct StubDriver;

impl BrowserDriver for StubDriver {
    fn run_testharness(
        &self,
        test: &TestItem,
        _timeout: Duration,
    ) -> Result<HarnessResult, DriverError> {
        Err(DriverError::Unimplemented(format!(
            "no browser behind the driver yet; cannot run {}",
            test.id
        )))
    }

    fn render(&self, url: &str, _viewport: Viewport) -> Result<Screenshot, DriverError> {
        Err(DriverError::Unimplemented(format!(
            "no browser behind the driver yet; cannot render {url}"
        )))
    }

    fn load_without_crash(&self, url: &str) -> Result<(), DriverError> {
        Err(DriverError::Unimplemented(format!(
            "no browser behind the driver yet; cannot load {url}"
        )))
    }

    fn name(&self) -> &str {
        "stub"
    }
}

/// Something went wrong while driving the browser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriverError {
    /// The operation needs a real browser; the stub cannot do it.
    Unimplemented(String),
    /// The testharness completion callback never fired in time.
    Timeout { test: String, after: Duration },
    /// The browser process died or the connection broke.
    BrowserGone(String),
    /// The screenshot bytes did not match the advertised dimensions.
    BadScreenshot { expected: usize, got: usize },
    /// The driver produced a report that does not parse.
    BadReport(String),
}

impl fmt::Display for DriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DriverError::Unimplemented(msg) => write!(f, "unimplemented: {msg}"),
            DriverError::Timeout { test, after } => {
                write!(f, "timed out waiting for {test} after {after:?}")
            }
            DriverError::BrowserGone(msg) => write!(f, "browser gone: {msg}"),
            DriverError::BadScreenshot { expected, got } => {
                write!(f, "bad screenshot: expected {expected} bytes, got {got}")
            }
            DriverError::BadReport(msg) => write!(f, "bad report: {msg}"),
        }
    }
}

impl std::error::Error for DriverError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::TestKind;

    fn item() -> TestItem {
        TestItem {
            id: "dom/a.html".into(),
            url: "/dom/a.html".into(),
            kind: TestKind::Testharness,
            path: None,
            references: Vec::new(),
        }
    }

    #[test]
    fn stub_driver_is_unimplemented() {
        let d = StubDriver;
        assert_eq!(d.name(), "stub");
        let err = d
            .run_testharness(&item(), Duration::from_secs(1))
            .unwrap_err();
        assert!(matches!(err, DriverError::Unimplemented(_)));
        assert!(err.to_string().contains("dom/a.html"));
        let err = d.render("/dom/a.html", Viewport::DEFAULT).unwrap_err();
        assert!(matches!(err, DriverError::Unimplemented(_)));
        assert!(err.to_string().contains("/dom/a.html"));
        let err = d.load_without_crash("/dom/a.html").unwrap_err();
        assert!(matches!(err, DriverError::Unimplemented(_)));
        assert!(err.to_string().contains("/dom/a.html"));
    }

    #[test]
    fn screenshot_validates_dimensions() {
        assert!(Screenshot::new(2, 2, vec![0u8; 16]).is_ok());
        let err = Screenshot::new(2, 2, vec![0u8; 15]).unwrap_err();
        assert_eq!(
            err,
            DriverError::BadScreenshot {
                expected: 16,
                got: 15
            }
        );
        let s = Screenshot::solid(2, 1, 255, 0, 0, 255);
        assert_eq!(s.rgba, vec![255, 0, 0, 255, 255, 0, 0, 255]);
    }

    #[test]
    fn error_display() {
        let e = DriverError::Timeout {
            test: "x".into(),
            after: Duration::from_secs(5),
        };
        assert!(e.to_string().contains("timed out waiting for x"));
    }
}
