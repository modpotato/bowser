//! WPT browser driver: [`BowserDriver`] plus a tiny static file server.
//!
//! [`BowserDriver`] implements [`bowser_wpt::driver::BrowserDriver`] by
//! spawning the `bowser` binary in headless mode, one child process per
//! operation. [`FixtureServer`] is a std-only static file server on
//! 127.0.0.1 used to serve WPT fixtures to the browser under test.
//!
//! [`BowserDriver::run_reftest`] renders a test and its references through
//! the real headless pipeline and compares pixels with a documented
//! [`bowser_wpt::reftest::ReftestTolerance`], reporting pass/fail.
//! [`is_blank_canvas`] and [`average_color`] are the pixel-sanity helpers
//! fixture tests use ("non-blank" and "expected dominant color").
//!
//! ## Subprocess protocol (`bowser --headless ...`)
//!
//! This module defines the contract; the `bowser` binary's `main.rs`
//! worker implements it. One child per operation; no session state.
//!
//! ```text
//! bowser --headless --viewport {W}x{H} {url}
//! ```
//! Render one frame at `W`x`H` CSS px, then exit.
//! - stdout: raw RGBA8 pixels, row-major, top-left origin, exactly
//!   `W * H * 4` bytes. No header, no framing.
//! - stderr: human-readable logs. Ignored on success; attached to the
//!   error on failure.
//! - exit 0: frame rendered and fully written to stdout.
//! - exit >= 1: render failed (bad arguments, load failure, init
//!   failure). The driver reports [`DriverError::BrowserGone`] with the
//!   stderr tail.
//! - killed by signal: crash. The driver reports
//!   [`DriverError::BrowserGone`].
//!
//! ```text
//! bowser --check {url} (--check is already headless)
//! ```
//! Load the page through the full headless pipeline (fetch -> parse ->
//! style -> layout -> display list; the document's `<script>` elements
//! execute synchronously as the load installs, exactly like `--headless`
//! minus the rasterization), then exit without rendering.
//!
//! - stdout: ignored by `--check` itself, but the testharness reporter
//!   injected by the driver's harness endpoint prints its report line
//!   here via `console.log`; `run_testharness` scans stdout for it (see
//!   the `run_testharness` mapping below).
//! - stderr: human-readable logs.
//! - exit 0: the page loaded and every script ran with no uncaught
//!   errors.
//! - exit 1: page-level failure (fetch/load failure, the `--timeout`
//!   deadline, or an uncaught script error). The browser itself is alive.
//! - exit >= 2: no intentional producer on this path (bad arguments are
//!   rejected by the argument parser with exit 2 before `--check` runs;
//!   the check worker dying without a verdict exits 2). A panic in the
//!   pipeline exits 101; like a signal death, the driver reads any
//!   abnormal exit as [`DriverError::BrowserGone`].
//! - killed by signal: crash.
//!
//! ## `run_testharness` mapping (v2)
//!
//! `run_testharness` serves the test through the harness endpoint
//! (`{base}/__bowser_harness__/<test-path>?__bowser_harness=1&id=<pct-id>&nonce=<nonce>`)
//! instead of the raw test URL. The fixture server injects
//! [`REPORTER_JS`] into the served page (just before `</body>`); the
//! reporter registers testharness.js's `add_completion_callback`, which
//! receives the full `tests` array with per-test status fields, builds one
//! Bowser-native report
//! (`{"test","harness_status","tests":[{"name","status","message"}]}` —
//! the schema [`HarnessResult::from_json`] accepts), and prints it to the
//! child's stdout with [`REPORT_MARKER`], the run's nonce, and the JSON
//! (`BOWSER_HARNESS_REPORT:<nonce>:{...}`) via `console.log`. The driver
//! takes the last marker line carrying this run's nonce from the child's
//! stdout and parses it: the nonce is a per-run random value sealed in
//! the injected reporter's closure, so a page that registers its own
//! completion callback (or uses a load handler) cannot forge a report
//! line the driver will accept — it cannot know the nonce, and report
//! lines without it are ignored.
//!
//! The reporter also registers `add_start_callback` and
//! `add_result_callback` when the harness defines them, printing one
//! [`START_MARKER`] line per test start (the JSON-escaped test name) and
//! one [`RESULT_MARKER`] line per test result (the same subtest shape as
//! the report's `tests` entries). The driver parses these into
//! [`TestharnessMarkers`]: informational observations — which subtests
//! started, each result as it completed — while the completion report
//! stays the authoritative result.
//!
//! The reporter observes whatever the harness completes while the child
//! pumps the page's event loop: synchronous tests, promise continuations
//! through the per-script microtask drain, and macrotask-queue tests
//! (`setTimeout`, `async_test` steps) — `--check` pumps until the
//! reporter's completion callback fires (observed through the completion
//! flag the reporter sets, see
//! [`crate::script::HARNESS_COMPLETION_FLAG`]) or the harness wait
//! budget elapses, so async tests that need more event-loop time than
//! the fixed headless budget still report instead of falling back to
//! the no-report mapping below.
//!
//! | child outcome                              | `run_testharness` result                          |
//! |--------------------------------------------|---------------------------------------------------|
//! | report line on stdout                      | parsed report (attributed to the test's id+url)   |
//! | report for a different test id             | `Err(DriverError::BadReport)`                     |
//! | marker line that does not parse            | `Err(DriverError::BadReport)`                     |
//! | exit 0, no report                          | `Ok(HarnessResult { status: SubtestsUnknown, .. })` |
//! | exit 1, no report                          | `Ok(HarnessResult { status: Error, .. })`         |
//! | exit >= 2                                  | `Err(DriverError::BrowserGone)`                   |
//! | killed by signal                           | `Err(DriverError::BrowserGone)`                   |
//! | no exit within `timeout`                   | child killed; `Err(DriverError::Timeout)`         |
//!
//! A harness report with [`HarnessStatus::Ok`] and all subtests
//! [`SubtestStatus::Pass`](bowser_wpt::results::SubtestStatus::Pass) is
//! green per [`HarnessResult::is_green`]: unlike v1, a passing page can
//! now actually report a pass.
//!
//! `load_without_crash` treats exit 0 and exit 1 as `Ok(())` (a page
//! error is not a crash) and signals as [`DriverError::BrowserGone`].
//!
//! ## Session protocol (`bowser --session`)
//!
//! Spawning one child per operation pays process startup, the system
//! font-directory walk ([`FontCache::new`](bowser_paint::fonts::FontCache::new)),
//! and GPU bring-up per test. [`SessionDriver`] instead spawns a single
//! long-lived `bowser --session` child and multiplexes test operations
//! over its stdin/stdout. The child is implemented here
//! ([`run_session_child`]); the `bowser` binary's `main.rs` only needs
//! to parse the `--session` flag and call it (that glue lives outside
//! this module: `main.rs` is a separate binary crate).
//!
//! The child prints `HELLO bowser-session/1` on startup (the driver's
//! spawn handshake), then loops reading `\n`-terminated command lines on
//! stdin and writing framed responses on stdout; diagnostics go to
//! stderr. Commands are processed strictly sequentially: at most one is
//! ever in flight, so responses need no interleaving logic.
//!
//! ```text
//! PING <id> <nonce>                          -> PONG <id> <nonce>
//! RENDER <id> <nonce> <w> <h> <url>           -> OK <id> <nonce> RENDER <len>\n<len bytes RGBA8>
//! CHECK <id> <nonce> <url>                    -> OK <id> <nonce> CHECK clean
//!                                            -> OK <id> <nonce> CHECK page-failed <message>
//! HARNESS <id> <nonce> <url>                  -> OK <id> <nonce> HARNESS clean
//!                                            -> OK <id> <nonce> HARNESS page-failed <message>
//! QUIT                                       -> child exits 0
//! ERR <id> <nonce> BROWSER <message>          (pipeline-internal failure)
//! ERR 0 - BADCMD <message>                   (malformed command; child stays alive)
//! ```
//!
//! `<id>` is a driver-chosen u64; `<nonce>` is a driver-chosen 32-hex-char
//! per-command secret echoed in every response line. The nonce is what
//! keeps a hostile page's `console.log` spray from forging a response:
//! the driver's reader demuxes child stdout into response lines (which
//! must carry the in-flight command's id *and* nonce) and console lines
//! (everything else, attributed to the in-flight command). Page console
//! output during `HARNESS` therefore flows to the driver exactly like the
//! one-shot `--check` path, and the same report scanning
//! ([`last_report_line`], [`TestharnessMarkers`]) applies unchanged.
//!
//! Timeouts are enforced driver-side only: the child runs each command
//! to completion on its main thread (one [`FontCache`](bowser_paint::fonts::FontCache)
//! and one thread-local GPU backend for the whole session — that reuse
//! is the saving), and the driver kills a child that outlives the
//! operation's budget. A pipeline panic is caught per command: the child
//! answers `ERR ... BROWSER` and exits, and the driver restarts the
//! session.
//!
//! Driver policy ([`SessionDriver`]): lazy spawn on first use, `HELLO`
//! handshake bounded by `spawn_timeout`, a `PING` health probe before an
//! operation when the child has been idle longer than `probe_interval`,
//! eager restart after a failure (at most one restart per incident; three
//! consecutive failures permanently disable the session), and fallback
//! to the one-shot [`BowserDriver`] path whenever the session is
//! unavailable — e.g. a `bowser` binary built without the `wpt-driver`
//! feature, which rejects `--session` and fails the handshake.

use std::collections::HashMap;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex, Weak};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::{ImageIdMap, font_key_for_style, load_tab_headless};
use bowser_css::ComputedStyle;
use bowser_dom::Document;
use bowser_layout::{LayoutBox, Size};
use bowser_paint::WgpuBackend;
use bowser_paint::display_list::{DecodedImage, ImageId, build_display_list};
use bowser_paint::fonts::{FontCache, FontKey};
use bowser_paint::render::render_layout_to_rgba_with_scroll;
use bowser_wpt::driver::{BrowserDriver, DriverError, Screenshot, Viewport};
use bowser_wpt::manifest::{TestExtras, TestItem};
use bowser_wpt::reftest::{self, RefRel, ReftestTolerance};
use bowser_wpt::results::{HarnessResult, HarnessStatus, SubtestResult};

/// Default wall-clock budget for `--viewport` renders (shader compile and
/// first paint can be slow on a cold GPU).
pub const DEFAULT_RENDER_TIMEOUT: Duration = Duration::from_secs(60);
/// Default wall-clock budget for `--check` page loads.
pub const DEFAULT_CHECK_TIMEOUT: Duration = Duration::from_secs(30);

/// Cap on the raw bytes of one HTTP request (request line + headers).
const MAX_REQUEST_BYTES: usize = 64 * 1024;

/// Cap on concurrent fixture-server connections. Past this many, new
/// connections are rejected instead of spawning a thread per connection
/// without bound: a runaway page opening connections in a loop could
/// otherwise exhaust threads (and memory) on the driver host.
const MAX_CONNECTIONS: usize = 128;

/// How often the child reaper re-checks for the child's exit while it is
/// still alive. The calling thread blocks the whole time (no spin); this
/// only bounds how promptly a finished child is noticed and how quickly
/// a kill request is serviced after a timeout.
const REAP_QUANTUM: Duration = Duration::from_millis(10);

/// How long to wait for the reaper to kill and reap a timed-out child
/// before giving up and reporting the timeout anyway. The child may
/// linger in that wedge case, but the driver never hangs on it.
const KILL_GRACE: Duration = Duration::from_secs(5);

/// Driver that spawns the `bowser` headless binary, one child per
/// operation, per the protocol in the module docs.
///
/// Construct with [`BowserDriver::new`], which also starts the fixture
/// server serving `doc_root`; the server lives as long as the driver.
pub struct BowserDriver {
    /// Path to the `bowser` binary implementing the `--headless` protocol.
    pub exe: PathBuf,
    /// Base URL of the fixture server, e.g. `http://127.0.0.1:41237`.
    pub server_base: String,
    /// Wall-clock budget for [`BrowserDriver::render`].
    pub render_timeout: Duration,
    /// Wall-clock budget for [`BrowserDriver::load_without_crash`] and the
    /// `--check` run inside [`BrowserDriver::run_testharness`].
    pub check_timeout: Duration,
    /// Owns the fixture server; dropping it stops the server. Never read.
    _server: Option<Shutdown>,
    /// Fixture-server document root. Kept so served URLs can be mapped
    /// back to files (e.g. [`BrowserDriver::resource_exists`] checks).
    doc_root: PathBuf,
}

/// Join a path against the fixture-server base URL, handling both
/// `path` forms (with or without a leading `/`) and a base URL with or
/// without a trailing `/`. The single join point for every URL the
/// driver builds against the fixture server.
fn join_server_url(base: &str, path: &str) -> String {
    let base = base.trim_end_matches('/');
    if path.starts_with('/') {
        format!("{base}{path}")
    } else {
        format!("{base}/{path}")
    }
}

impl BowserDriver {
    /// Point at the built `bowser` binary and start a fixture server over
    /// `doc_root`. The server shuts down when the driver is dropped.
    pub fn new(exe: PathBuf, doc_root: &Path) -> std::io::Result<BowserDriver> {
        let (server_base, shutdown) = FixtureServer::serve_dir(doc_root)?;
        // Canonicalize for the probe in
        // [`BrowserDriver::resource_exists`]: it compares canonicalized
        // request paths against this root, exactly like the server's own
        // prefix check in [`read_doc_file`] (the server canonicalizes
        // its own copy in `serve_dir`).
        let doc_root = doc_root.canonicalize()?;
        Ok(BowserDriver {
            exe,
            server_base,
            render_timeout: DEFAULT_RENDER_TIMEOUT,
            check_timeout: DEFAULT_CHECK_TIMEOUT,
            _server: Some(shutdown),
            doc_root,
        })
    }

    /// A [`RunConfig`][bowser_wpt::runner::RunConfig] pre-wired to this
    /// driver's fixture server: [`server_base_url`][bowser_wpt::runner::RunConfig::server_base_url]
    /// is set to [`Self::server_base`], so the runner joins manifest-relative
    /// URLs against the fixture server before handing absolute URLs to the
    /// driver (see [`Runner::absolute_url`][bowser_wpt::runner::Runner]). The
    /// remaining fields are left at default for the caller (the `bowser-wpt`
    /// CLI) to fill in.
    pub fn run_config(&self) -> bowser_wpt::runner::RunConfig {
        bowser_wpt::runner::RunConfig {
            server_base_url: self.server_base.clone(),
            ..Default::default()
        }
    }

    /// Absolute URL for a manifest test served by the fixture server.
    fn test_url(&self, test: &TestItem) -> String {
        join_server_url(&self.server_base, &test.url)
    }

    /// Instrumented URL for a testharness run: the fixture server serves
    /// the test page with the completion-callback reporter injected (see
    /// the module docs).
    ///
    /// The test path keeps its directory structure under
    /// [`HARNESS_PREFIX`], so the page's relative subresources resolve
    /// under the same prefix and are served raw. `__bowser_harness=1`
    /// marks the request for injection (subresource fetches never carry
    /// it); `id` stamps the manifest id into the report for attribution,
    /// and `nonce` carries this run's report nonce (see
    /// [`new_report_nonce`]) so the injected reporter prints a report
    /// line only the driver will accept. The test's own query string, if
    /// any, is preserved ahead of the harness parameters.
    fn harness_url(&self, test: &TestItem, nonce: &str) -> String {
        let (path, query) = match test.url.split_once('?') {
            Some((p, q)) => (p, format!("{q}&")),
            None => (test.url.as_str(), String::new()),
        };
        let path = path.strip_prefix('/').unwrap_or(path);
        format!(
            "{}{HARNESS_PREFIX}{path}?{query}__bowser_harness=1&id={}&nonce={}",
            self.server_base,
            pct_encode(&test.id),
            pct_encode(nonce),
        )
    }

    /// Absolute URL for a reference href, resolved against the test's own
    /// directory on the fixture server. A leading `/` means server root;
    /// anything else is relative to the directory containing the test
    /// (e.g. test `/dom/a.html` + `ref.html` -> `/dom/ref.html`).
    fn reference_url(&self, test: &TestItem, href: &str) -> String {
        if href.starts_with('/') {
            return join_server_url(&self.server_base, href);
        }
        // The directory may itself lack a leading slash when the manifest
        // URL does (e.g. `sub/t.html`); join_server_url handles both.
        let path = match test.url.rsplit_once('/') {
            Some((dir, _)) if !dir.is_empty() => format!("{dir}/{href}"),
            _ => href.to_string(),
        };
        join_server_url(&self.server_base, &path)
    }

    /// Run a reftest: render the test page and each of its references at
    /// `viewport` through the real headless pipeline (navigate -> layout
    /// -> `render_to_rgba`), compare pixels with `tolerance`, and report
    /// pass/fail.
    ///
    /// ## Tolerance (from `bowser_wpt::reftest`)
    ///
    /// A pixel counts as different when any of its four channels differs
    /// by more than `tolerance.per_channel`; the two renderings count as
    /// equal when the fraction of differing pixels is at most
    /// `tolerance.max_diff_fraction`. `ReftestTolerance::EXACT` is
    /// pixel-perfect (0 channel slack, 0.0 fraction).
    ///
    /// A `match` reference passes when the renderings are equal under the
    /// tolerance; a `mismatch` reference passes when they differ. The run
    /// passes under the upstream WPT reftest rule: at least one `match`
    /// reference must match when any exist, and every `mismatch`
    /// reference must differ (see [`reftest_passed`]). Tests with no
    /// references pass vacuously; the WPT runner treats "no references"
    /// as a skip before calling this.
    pub fn run_reftest(
        &self,
        test: &TestItem,
        viewport: Viewport,
        tolerance: ReftestTolerance,
    ) -> Result<ReftestRun, DriverError> {
        let test_shot = self.render(&self.test_url(test), viewport)?;
        let mut comparisons = Vec::with_capacity(test.references.len());
        for reference in &test.references {
            let ref_shot = self.render(&self.reference_url(test, &reference.href), viewport)?;
            let cmp = reftest::compare(&test_shot, &ref_shot, tolerance).map_err(|e| {
                DriverError::BadReport(format!("reftest {} vs {}: {e}", test.id, reference.href))
            })?;
            let satisfied = match reference.rel {
                RefRel::Match => cmp.equal,
                RefRel::Mismatch => !cmp.equal,
            };
            comparisons.push(ReferenceComparison {
                href: reference.href.clone(),
                rel: reference.rel,
                equal: cmp.equal,
                diff_pixels: cmp.diff_pixels,
                diff_fraction: cmp.diff_fraction,
                satisfied,
            });
        }
        let passed = reftest_passed(&comparisons);
        Ok(ReftestRun {
            test: test.id.clone(),
            passed,
            comparisons,
        })
    }

    /// Run a reftest with the viewport and tolerance resolved from the
    /// test's manifest extras (`viewport_size`, `fuzzy`), falling back to
    /// `viewport` and `tolerance` when the extras carry none.
    ///
    /// This is what a manifest-driven run calls: the extras come from
    /// [`Manifest::extras_for`](bowser_wpt::manifest::Manifest::extras_for).
    /// It resolves via
    /// [`TestExtras::viewport`](bowser_wpt::manifest::TestExtras::viewport)
    /// and
    /// [`TestExtras::tolerance`](bowser_wpt::manifest::TestExtras::tolerance)
    /// — the same resolution the
    /// [`bowser_wpt::runner`](bowser_wpt::runner::Runner) applies — and
    /// calls [`run_reftest`](BowserDriver::run_reftest) with the result.
    /// The fuzzy pixel budget is a fraction of the effective viewport
    /// area, so the tolerance is resolved against the extras viewport,
    /// not the default one.
    pub fn run_reftest_for_extras(
        &self,
        test: &TestItem,
        extras: &TestExtras,
        viewport: Viewport,
        tolerance: ReftestTolerance,
    ) -> Result<ReftestRun, DriverError> {
        let viewport = extras.viewport(viewport);
        let tolerance = extras.tolerance(viewport.width as u64 * viewport.height as u64, tolerance);
        self.run_reftest(test, viewport, tolerance)
    }

    /// Run the headless binary with `args`, killing it after `timeout`.
    fn run_child(&self, args: &[&str], timeout: Duration) -> Result<ChildOutput, ChildError> {
        let child = Command::new(&self.exe)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                ChildError::Spawn(format!("failed to spawn {}: {e}", self.exe.display()))
            })?;
        run_with_timeout(child, timeout)
    }
}

/// Upstream WPT reftest verdict over one comparison per reference.
///
/// A run passes iff (there are no `match` references, or at least one
/// `match` reference is satisfied) AND every `mismatch` reference is
/// satisfied. Notably, multiple `match` references are alternatives: one
/// matching is enough. Requiring every `match` reference to match
/// (`passed &= satisfied`) is the wrong rule and fails valid tests.
fn reftest_passed(comparisons: &[ReferenceComparison]) -> bool {
    let mut any_match = false;
    let mut match_satisfied = false;
    for comparison in comparisons {
        match comparison.rel {
            RefRel::Match => {
                any_match = true;
                match_satisfied |= comparison.satisfied;
            }
            RefRel::Mismatch => {
                if !comparison.satisfied {
                    return false;
                }
            }
        }
    }
    !any_match || match_satisfied
}

/// Last ~512 bytes of stderr, for attaching to error messages.
fn stderr_tail(stderr: &[u8]) -> String {
    let tail = if stderr.len() > 512 {
        &stderr[stderr.len() - 512..]
    } else {
        stderr
    };
    String::from_utf8_lossy(tail).into_owned()
}

/// Prefix of the one report line the injected reporter prints to the
/// child's stdout via `console.log`. `--check` writes nothing else to
/// stdout, so any line with this prefix is a harness report; the driver
/// takes the last one.
const REPORT_MARKER: &str = "BOWSER_HARNESS_REPORT:";

/// Prefix of the per-test-start lines the injected reporter prints to
/// the child's stdout via `console.log`: `BOWSER_HARNESS_START:` followed
/// by the subtest name as a JSON string literal, one line per
/// `add_start_callback` firing.
const START_MARKER: &str = "BOWSER_HARNESS_START:";

/// Prefix of the per-test-result lines the injected reporter prints:
/// `BOWSER_HARNESS_RESULT:` followed by a JSON `{"name","status",
/// "message"}` object — the same shape as one entry of the completion
/// report's `tests` array — one line per `add_result_callback` firing.
const RESULT_MARKER: &str = "BOWSER_HARNESS_RESULT:";

/// The last marker-prefixed line of the child's stdout that carries
/// this run's `nonce`, with the marker and nonce stripped: the
/// Bowser-native JSON report, or `None` when the reporter never printed
/// (the completion callback never fired). Lines without the nonce are
/// ignored, so a page cannot forge a report with a second completion
/// callback or a load handler: only the injected reporter knows the
/// nonce, and it is sealed in its closure.
fn last_report_line(stdout: &[u8], nonce: &str) -> Option<String> {
    let prefix = format!("{REPORT_MARKER}{nonce}:");
    String::from_utf8_lossy(stdout)
        .lines()
        .filter_map(|line| line.strip_prefix(&prefix))
        .next_back()
        .map(str::to_owned)
}

/// Counter mixed into every report nonce so two runs in the same
/// nanosecond still differ.
static NONCE_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A fresh per-run report nonce: 32 hex chars of process-unique
/// randomness, std-only. The SipHash behind [`RandomState`] is keyed by
/// OS randomness per process; time, pid, and a counter are mixed in so
/// the value is unpredictable to the page under test. Stamped into the
/// injected reporter for this run only (see [`inject_reporter`]); the
/// driver accepts only report lines carrying it
/// (see [`last_report_line`]).
fn new_report_nonce() -> String {
    let counter = NONCE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    let mut h = RandomState::new().build_hasher();
    h.write_usize(counter);
    h.write_u64(nanos as u64);
    h.write_u64(nanos.wrapping_shr(64) as u64);
    h.write_u32(pid);
    let lo = h.finish();
    let mut h = RandomState::new().build_hasher();
    h.write_usize(counter ^ 0x9e3779b97f4a7c15);
    h.write_u64(!nanos as u64);
    h.write_u64((!nanos).wrapping_shr(64) as u64);
    h.write_u32(!pid);
    let hi = h.finish();
    format!("{lo:016x}{hi:016x}")
}

/// Decode the payload of one `BOWSER_HARNESS_START:` line (the subtest
/// name as a JSON string literal) through the native report schema: the
/// payload is wrapped as the name of a synthetic subtest and parsed by
/// [`HarnessResult::from_json`], so start lines get the same strictness
/// as real reports (unknown fields rejected, malformed lines skipped).
/// One line yields at most one name — the first subtest — so a crafted
/// payload cannot smuggle extra entries past the scanner.
fn parse_start_payload(payload: &str) -> Option<String> {
    let wrapped = format!(
        "{{\"test\":\"\",\"harness_status\":\"OK\",\"tests\":[{{\"name\":{payload},\"status\":\"PASS\"}}]}}"
    );
    HarnessResult::from_json(&wrapped)
        .ok()?
        .subtests
        .into_iter()
        .next()
        .map(|t| t.name)
}

/// Decode the payload of one `BOWSER_HARNESS_RESULT:` line (a JSON
/// `{"name","status","message"}` object) through the native report
/// schema, same wrapping discipline as [`parse_start_payload`].
fn parse_result_payload(payload: &str) -> Option<SubtestResult> {
    let wrapped = format!("{{\"test\":\"\",\"harness_status\":\"OK\",\"tests\":[{payload}]}}");
    HarnessResult::from_json(&wrapped)
        .ok()?
        .subtests
        .into_iter()
        .next()
}

/// The per-test start/result marker lines from one child run's stdout,
/// parsed alongside the completion report (see [`REPORTER_JS`]).
///
/// Informational only: the completion report stays the authoritative
/// result. Lines whose payload is not well-formed JSON of the expected
/// shape are skipped, and a line contributes at most one entry, so a
/// hostile page cannot smuggle extra entries past the scanner.
#[derive(Debug, Clone, Default)]
pub struct TestharnessMarkers {
    /// Subtest names, in the order the start callback fired.
    pub started: Vec<String>,
    /// One entry per result marker, in completion order, with the same
    /// name/status/message mapping as the completion report's subtests.
    pub results: Vec<SubtestResult>,
    /// True when the child's stdout hit the [`MAX_DRAIN_BYTES`] cap and
    /// was cut: marker lines past the cut are lost, so a truncated run
    /// must be treated as incomplete, never as silently clean.
    pub truncated: bool,
}

impl TestharnessMarkers {
    /// Parse every `BOWSER_HARNESS_START:` / `BOWSER_HARNESS_RESULT:`
    /// line in `stdout`, in order. Like [`last_report_line`], only
    /// line-leading markers count.
    pub fn parse(stdout: &[u8]) -> TestharnessMarkers {
        Self::parse_capped(stdout, false)
    }

    /// [`parse`](Self::parse) over a buffer that may have been cut at
    /// the drain cap: a trailing partial line cannot parse (payloads go
    /// through the strict report schema), and `truncated` records the
    /// cut on the result.
    pub fn parse_capped(stdout: &[u8], truncated: bool) -> TestharnessMarkers {
        let mut markers = TestharnessMarkers {
            truncated,
            ..TestharnessMarkers::default()
        };
        for line in String::from_utf8_lossy(stdout).lines() {
            if let Some(payload) = line.strip_prefix(START_MARKER)
                && let Some(name) = parse_start_payload(payload)
            {
                markers.started.push(name);
            } else if let Some(payload) = line.strip_prefix(RESULT_MARKER)
                && let Some(result) = parse_result_payload(payload)
            {
                markers.results.push(result);
            }
        }
        markers
    }
}

/// Path prefix the harness endpoint lives under: requests to
/// `{base}/__bowser_harness__/<test-path>` serve the file at
/// `<test-path>` from the doc root. With the `__bowser_harness=1&id=…`
/// query (only ever added by [`BowserDriver::harness_url`]) the
/// completion-callback reporter is injected; without it the file is
/// served raw, which is how the instrumented page's relative
/// subresources resolve.
const HARNESS_PREFIX: &str = "/__bowser_harness__/";

/// Reporter injected into testharness pages by [`serve_harness`].
/// `__BOWSER_TEST_ID__` is substituted with the JSON-escaped manifest id
/// before injection, `__BOWSER_NONCE__` with the JSON-escaped report
/// nonce, and `__BOWSER_COMPLETION_FLAG__` with the JSON-escaped name of
/// the completion-flag global (see
/// [`crate::script::HARNESS_COMPLETION_FLAG`).
///
/// Kept to conservative ES5 (`var`, function expressions, no `Date` —
/// the engine has no `Date` yet): it only needs `add_completion_callback`
/// (defined by testharness.js itself), `JSON.stringify`, `String`, and
/// `console.log` (wired to the child's stdout). `add_start_callback` and
/// `add_result_callback` are registered when the harness defines them;
/// the completion report stays authoritative and the start/result lines
/// are best-effort observations.
///
/// Registration order is safe: the script is injected immediately after
/// the `testharness.js` `<script>` tag (falling back to end of body only
/// when the tag cannot be located), so it runs before the test's own
/// scripts, and `add_completion_callback` receives the *full* `tests`
/// array — results that completed before registration are still
/// reported. Numeric statuses are mapped to the Bowser-native strings
/// (`SubtestStatus` / `HarnessStatus` parse unknown strings
/// forward-compatibly, so an out-of-range value degrades to `Unknown`,
/// never to a misread pass). The harness-level status arrives as a
/// `TestsStatus` object (numeric code on `.status`), which the reporter
/// unwraps before mapping; its `.message` is carried on the report's
/// `message` field.
const REPORTER_JS: &str = r#"(function(){
var TEST_ID = __BOWSER_TEST_ID__;
var NONCE = __BOWSER_NONCE__;
var MARKER = "BOWSER_HARNESS_REPORT:";
var START_MARKER = "BOWSER_HARNESS_START:";
var RESULT_MARKER = "BOWSER_HARNESS_RESULT:";
function nameOf(n, names) {
  if (typeof n === "number" && n >= 0 && n < names.length) { return names[n]; }
  if (typeof n === "string") {
    var u = n.toUpperCase();
    for (var i = 0; i < names.length; i++) { if (names[i] === u) { return u; } }
  }
  return "UNKNOWN";
}
function subtestName(s) { return nameOf(s, ["PASS","FAIL","TIMEOUT","NOTRUN"]); }
// The completion callback receives a TestsStatus *object* (numeric code
// on `.status`, message on `.message`), not a bare number: unwrap it
// before mapping. Bare numbers/strings still map (synthetic harnesses).
function harnessName(s) {
  if (s !== null && typeof s === "object" && typeof s.status === "number") { s = s.status; }
  return nameOf(s, ["OK","ERROR","TIMEOUT","PRECONDITION_FAILED"]);
}
function str(v) { try { return v == null ? null : String(v); } catch (e) { return null; } }
function testName(t) { var nm = "?"; try { nm = String(t.name); } catch (e) {} return nm; }
function setFlag(v) {
  // Best-effort: `window` is the global object in real page realms, but
  // the reporter must also survive bare-VM harnesses (synthetic tests)
  // where it is undefined. The report line stays authoritative; the
  // flag only lets the headless pump stop early.
  try { window[__BOWSER_COMPLETION_FLAG__] = v; } catch (e) {}
}
try {
  if (typeof add_completion_callback !== "function") { return; }
  // Arm the headless pump's completion flag: `--check` keeps pumping
  // the page's event loop until the completion callback below sets it,
  // so async tests past the fixed headless pump budget still report.
  setFlag(false);
  // Headless: the harness's visual `#log` output is never seen, and the
  // parse-phase document shim cannot render it (no `createElementNS`),
  // so disable it. Must run before the first `test()` call: `Output`
  // latches `enabled` on harness setup and a later `setup({output})`
  // cannot re-enable it, so this script is injected right after
  // `testharness.js` loads rather than at end of body.
  if (typeof setup === "function") { try { setup({output:false}); } catch (e) {} }
  // Per-test start: one marker line with the JSON-escaped test name.
  // Guarded — older harnesses may not define it — and best-effort: the
  // completion report below stays authoritative.
  if (typeof add_start_callback === "function") {
    add_start_callback(function(t) {
      console.log(START_MARKER + JSON.stringify(testName(t)));
    });
  }
  // Per-test result as each test completes, same subtest shape as the
  // completion report's `tests` entries.
  if (typeof add_result_callback === "function") {
    add_result_callback(function(t) {
      var st = "UNKNOWN";
      try { st = subtestName(t.status); } catch (e) {}
      console.log(RESULT_MARKER + JSON.stringify(
        {name: testName(t), status: st, message: str(t.message)}));
    });
  }
  add_completion_callback(function(tests, status) {
    var out = [];
    for (var i = 0; i < tests.length; i++) {
      var t = tests[i];
      var nm = "?";
      try { nm = String(t.name); } catch (e) {}
      var st = "UNKNOWN";
      try { st = subtestName(t.status); } catch (e) {}
      out.push({name: nm, status: st, message: str(t.message)});
    }
    // The nonce is a per-run random value sealed in this closure: a
    // page that registers its own completion callback cannot forge a
    // report line the driver accepts, because report lines without the
    // nonce are ignored (see `last_report_line`).
    // `status ? status.message : null`: the message rides on the
    // TestsStatus object; a falsy bare-number status (synthetic
    // harnesses) carries no message instead of stringifying to "0".
    console.log(MARKER + NONCE + ":" + JSON.stringify(
      {test: TEST_ID, harness_status: harnessName(status),
       message: str(status ? status.message : null), tests: out}));
    // Signal the headless pump (same flag as above): the report line is
    // on its way to the driver's stdout scan, so the pump can stop now.
    setFlag(true);
  });
} catch (e) {}
})();"#;

/// Escape `s` as a JSON string literal (quotes included), safe to splice
/// into a `<script>` block: `<`, `>`, `&` are `\u`-escaped so a hostile
/// id can never break out of the script element, and `"`, `\`, and
/// control characters use JSON escapes (valid JS string escapes too).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '&' => out.push_str("\\u0026"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Percent-encode for a URL query value: every byte outside the
/// unreserved set becomes `%XX` (uppercase hex, which `percent_decode`
/// accepts), so `&`, `=`, `/`, and non-ASCII never survive literally.
fn pct_encode(s: &str) -> String {
    const UNRESERVED: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~";
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if UNRESERVED.contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Case-insensitive reverse search for an ASCII needle; returns the byte
/// index into `haystack`. ASCII-only comparison keeps the index valid
/// for slicing the original (unlike lowercasing first, which can shift
/// indices on non-ASCII text).
fn find_ascii_ci(haystack: &[u8], needle: &str) -> Option<usize> {
    let n = needle.as_bytes();
    if n.is_empty() || haystack.len() < n.len() {
        return None;
    }
    (0..=haystack.len() - n.len()).find(|&i| {
        haystack[i..i + n.len()]
            .iter()
            .zip(n.iter())
            .all(|(a, b)| a.to_ascii_lowercase() == *b)
    })
}

fn rfind_ascii_ci(haystack: &[u8], needle: &str) -> Option<usize> {
    let n = needle.as_bytes();
    if n.is_empty() || haystack.len() < n.len() {
        return None;
    }
    (0..=haystack.len() - n.len()).rev().find(|&i| {
        haystack[i..i + n.len()]
            .iter()
            .zip(n.iter())
            .all(|(a, b)| a.to_ascii_lowercase() == *b)
    })
}

/// Splice the reporter `<script>` into `html` just before `</body>`
/// (case-insensitive), falling back to before `</html>`, then to the end
/// of the document. Late injection is deliberate: the reporter must run
/// after testharness.js defines `add_completion_callback`, and the
/// completion callback receives the full `tests` array, so results that
/// completed before registration are still observed.
///
/// `nonce` is this run's report nonce (see [`new_report_nonce`]),
/// spliced as a closure `var` the page cannot reach from outside the
/// IIFE; both placeholders are JSON-escaped.
fn inject_reporter(html: &str, test_id: &str, nonce: &str) -> String {
    let page = inject_reporter_bytes(html.as_bytes(), test_id, nonce);
    // The UTF-8 path splices at char boundaries (the markers are ASCII,
    // so byte and char indices agree); the byte splice is valid UTF-8.
    String::from_utf8(page).expect("reporter injection preserves UTF-8")
}

/// Byte-level reporter injection for genuinely non-UTF-8 pages (e.g.
/// big5-encoded WPT files): the splice markers are pure ASCII, so the
/// search and the inserted `<script>` block (ASCII by construction) never
/// split a multibyte sequence. The page bytes pass through untouched.
fn inject_reporter_bytes(page: &[u8], test_id: &str, nonce: &str) -> Vec<u8> {
    let script = format!(
        "<script>{}</script>",
        REPORTER_JS
            .replace("__BOWSER_TEST_ID__", &json_escape(test_id))
            .replace("__BOWSER_NONCE__", &json_escape(nonce))
            .replace(
                "__BOWSER_COMPLETION_FLAG__",
                &json_escape(crate::script::HARNESS_COMPLETION_FLAG)
            )
    );
    let script = script.as_bytes();
    // The reporter disables the harness's visual `#log` output via
    // `setup({output:false})`, which only takes effect before the first
    // `test()` call, so inject immediately after the `testharness.js`
    // `<script>` tag rather than at end of body. Fall back to the old
    // end-of-body position when the tag cannot be located.
    let idx = find_ascii_ci(page, "testharness.js")
        .and_then(|i| find_ascii_ci(&page[i..], "</script>").map(|j| i + j + "</script>".len()))
        .or_else(|| rfind_ascii_ci(page, "</body>"))
        .or_else(|| rfind_ascii_ci(page, "</html>"))
        .unwrap_or(page.len());
    let mut out = Vec::with_capacity(page.len() + script.len());
    out.extend_from_slice(&page[..idx]);
    out.extend_from_slice(script);
    out.extend_from_slice(&page[idx..]);
    out
}

impl BowserDriver {
    /// Run one testharness test through the harness endpoint, returning
    /// the completion report plus the parsed start/result marker lines
    /// (see [`TestharnessMarkers`]).
    ///
    /// [`BrowserDriver::run_testharness`] delegates here and drops the
    /// markers; callers that want the per-test observations — which
    /// subtests started, each result as it completed — use this directly.
    /// The markers are informational: the completion report stays the
    /// authoritative result.
    pub fn run_testharness_with_markers(
        &self,
        test: &TestItem,
        timeout: Duration,
    ) -> Result<(HarnessResult, TestharnessMarkers), DriverError> {
        // Serve the test through the harness endpoint so the fixture
        // server injects the completion-callback reporter (see the module
        // docs): when testharness.js completes during the load, the
        // reporter prints one Bowser-native JSON report line to the
        // child's stdout. The report is attributed to the TestItem that
        // ran: both the id (stamped into the report by the server, and
        // verified here) and the manifest url (which the schema does not
        // carry), which is what the runner checks.
        // A fresh nonce per run: only the injected reporter knows it,
        // so only its report line is accepted (see `last_report_line`).
        let nonce = new_report_nonce();
        let url = self.harness_url(test, &nonce);
        let start = Instant::now();
        let out = match self.run_child(&["--check", &url], timeout) {
            Ok(out) => out,
            Err(ChildError::Timeout) => {
                return Err(DriverError::Timeout {
                    test: test.id.clone(),
                    after: timeout,
                });
            }
            Err(ChildError::Spawn(msg)) | Err(ChildError::Gone(msg)) => {
                return Err(DriverError::BrowserGone(format!(
                    "testharness run of {}: {msg}",
                    test.id
                )));
            }
        };
        let duration_ms = Some(start.elapsed().as_millis() as u64);
        // The exit-code mapping (see the module docs): clean / page-level
        // failure / browser-internal. The report scanning below is shared
        // with the session transport.
        let verdict = match out.status.code() {
            Some(0) => RunVerdict::Clean,
            Some(1) => RunVerdict::PageFailed,
            Some(code) => {
                return Err(DriverError::BrowserGone(format!(
                    "testharness run of {}: browser exited with code {code}: {}",
                    test.id,
                    stderr_tail(&out.stderr.bytes)
                )));
            }
            None => {
                return Err(DriverError::BrowserGone(format!(
                    "testharness run of {}: browser killed by signal",
                    test.id
                )));
            }
        };
        interpret_harness_output(
            test,
            &nonce,
            verdict,
            &out.stdout.bytes,
            out.stdout.truncated,
            duration_ms,
        )
    }
}

/// Verdict class of a harness/check child run, shared by the one-shot
/// (`--check` exit codes) and session (`CHECK`/`HARNESS` verdicts)
/// transports so both map to [`HarnessResult`] identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunVerdict {
    /// The page loaded and every script ran with no uncaught errors.
    Clean,
    /// Page-level failure: fetch/load failure or an uncaught script
    /// error. The browser itself is alive.
    PageFailed,
}

/// Turn one harness run's captured console output into the completion
/// report plus parsed start/result markers, shared by the one-shot and
/// session transports.
///
/// Scans `console` for the last report line carrying `nonce` (see
/// [`last_report_line`]): when present it is parsed, attributed to the
/// test that ran (id stamped by the server, url from the [`TestItem`]),
/// and returned. With no report line the completion callback never fired
/// — the page never loaded testharness.js, it errored first, or its
/// tests need the event loop the headless pipeline does not pump — and
/// the verdict decides: a clean load is [`HarnessStatus::SubtestsUnknown`]
/// (never a pass), a page-level failure is [`HarnessStatus::Error`].
/// Marker parsing runs over the capped buffer only, and the truncation
/// cut is recorded on the markers so a truncated run is never silently
/// reported as clean.
fn interpret_harness_output(
    test: &TestItem,
    nonce: &str,
    verdict: RunVerdict,
    console: &[u8],
    truncated: bool,
    duration_ms: Option<u64>,
) -> Result<(HarnessResult, TestharnessMarkers), DriverError> {
    let markers = TestharnessMarkers::parse_capped(console, truncated);
    if let Some(line) = last_report_line(console, nonce) {
        let mut report = HarnessResult::from_json(&line).map_err(|e| {
            DriverError::BadReport(format!(
                "testharness run of {}: malformed harness report: {e}",
                test.id
            ))
        })?;
        // The server stamps the id it served; a mismatch means the
        // child navigated elsewhere or the output is not ours — never
        // attribute it to this test.
        if report.test != test.id {
            return Err(DriverError::BadReport(format!(
                "testharness run of {}: harness report is for {:?}",
                test.id, report.test
            )));
        }
        report.url = Some(test.url.clone());
        if report.duration_ms.is_none() {
            report.duration_ms = duration_ms;
        }
        return Ok((report, markers));
    }
    // No report line: fall back to the verdict mapping.
    let status = match verdict {
        RunVerdict::Clean => HarnessStatus::SubtestsUnknown,
        RunVerdict::PageFailed => HarnessStatus::Error,
    };
    Ok((
        HarnessResult {
            test: test.id.clone(),
            url: Some(test.url.clone()),
            status,
            message: None,
            duration_ms,
            subtests: Vec::new(),
        },
        markers,
    ))
}

impl BowserDriver {
    /// Mirror of [`synthesize_wrapper`] for the
    /// [`BrowserDriver::resource_exists`] probe: reports whether the
    /// server would 200 a request for a synthesized
    /// `<name>.any.html` / `<name>.window.html` wrapper, including the
    /// `.any` variant's `// META: global=` exposure check.
    fn synthesized_wrapper_exists(&self, decoded_path: &str) -> bool {
        for (req_suffix, js_suffix, _global_shim, exposure_check) in
            WRAPPER_SYNTHESIS.iter().copied()
        {
            let Some(stem) = decoded_path.strip_suffix(req_suffix) else {
                continue;
            };
            let js_path = format!("{stem}{js_suffix}");
            let fs_path = self.doc_root.join(js_path.trim_start_matches('/'));
            // Same symlink discipline as `synthesize_wrapper`: resolve
            // and re-check against the root, and require a regular file.
            let sibling = match fs_path.canonicalize() {
                Ok(p) if p.starts_with(&self.doc_root) && p.is_file() => p,
                _ => return false,
            };
            if exposure_check {
                let Ok(js) = std::fs::read(&sibling) else {
                    return false;
                };
                let meta = parse_wrapper_meta(&js);
                if !any_window_exposed(meta.global.as_deref()) {
                    return false;
                }
            }
            return true;
        }
        false
    }
}

impl BrowserDriver for BowserDriver {
    fn name(&self) -> &str {
        "bowser"
    }

    /// Map a fixture-server URL back to the document root and report
    /// whether the server would answer 200. Mirrors the server's own
    /// routing (percent-decode, `..` rejection in [`route`], the
    /// directory → `index.html` step, the symlink prefix check, and the
    /// `.any.html` / `.window.html` wrapper synthesis in
    /// [`read_doc_file`]): a pruned-file 404 must never reach the pixel
    /// comparison as a FAIL. Non-local URLs (remote references) cannot
    /// be checked and answer `true`.
    fn resource_exists(&self, url: &str) -> bool {
        let Some(path) = url.strip_prefix(&self.server_base) else {
            return true;
        };
        let path = path.split(['?', '#']).next().unwrap_or(path);
        let Some(decoded) = percent_decode(path) else {
            return false;
        };
        if !decoded.starts_with('/') || decoded.split('/').any(|seg| seg == "..") {
            return false;
        }
        // Directory requests serve `index.html`, like `read_doc_file`.
        let mut fs_path = self.doc_root.join(decoded.trim_start_matches('/'));
        if fs_path.is_dir() {
            fs_path.push("index.html");
        }
        match fs_path.canonicalize() {
            // A symlink escape canonicalizes outside the root; the
            // server 404s those, so the probe answers `false` rather
            // than reporting the escaping target as servable.
            Ok(p) if p.starts_with(&self.doc_root) && p.is_file() => true,
            Ok(_) => false,
            // A missing file may still be satisfiable from a `.js`
            // sibling via wrapper synthesis, like `read_doc_file`.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                self.synthesized_wrapper_exists(&decoded)
            }
            Err(_) => false,
        }
    }

    fn render(&self, url: &str, viewport: Viewport) -> Result<Screenshot, DriverError> {
        let dims = format!("{}x{}", viewport.width, viewport.height);
        let out = match self.run_child(
            &["--headless", "--viewport", &dims, url],
            self.render_timeout,
        ) {
            Ok(out) => out,
            Err(ChildError::Timeout) => {
                return Err(DriverError::Timeout {
                    test: url.to_owned(),
                    after: self.render_timeout,
                });
            }
            Err(ChildError::Spawn(msg)) | Err(ChildError::Gone(msg)) => {
                return Err(DriverError::BrowserGone(format!("render of {url}: {msg}")));
            }
        };
        if !out.status.success() {
            return Err(DriverError::BrowserGone(format!(
                "render of {url} failed: exit={:?} stderr={}",
                out.status,
                stderr_tail(&out.stderr.bytes)
            )));
        }
        let expected = viewport.width as u64 * viewport.height as u64 * 4;
        let got = out.stdout.bytes.len() as u64;
        if got != expected {
            return Err(DriverError::BadScreenshot {
                expected: expected as usize,
                got: got as usize,
            });
        }
        Screenshot::new(viewport.width, viewport.height, out.stdout.bytes)
    }

    fn run_testharness(
        &self,
        test: &TestItem,
        timeout: Duration,
    ) -> Result<HarnessResult, DriverError> {
        self.run_testharness_with_markers(test, timeout)
            .map(|(report, _)| report)
    }

    fn load_without_crash(&self, url: &str) -> Result<(), DriverError> {
        let out = match self.run_child(&["--check", url], self.check_timeout) {
            Ok(out) => out,
            Err(ChildError::Timeout) => {
                return Err(DriverError::Timeout {
                    test: url.to_owned(),
                    after: self.check_timeout,
                });
            }
            Err(ChildError::Spawn(msg)) | Err(ChildError::Gone(msg)) => {
                return Err(DriverError::BrowserGone(format!("load of {url}: {msg}")));
            }
        };
        // Exit 0 (clean) and exit 1 (page-level failure) both mean the
        // browser survived; only a signal means it crashed.
        match out.status.code() {
            Some(_) => Ok(()),
            None => Err(DriverError::BrowserGone(format!(
                "browser crashed while loading {url}"
            ))),
        }
    }
}

/// Pass/fail outcome of [`BowserDriver::run_reftest`].
#[derive(Debug, Clone)]
pub struct ReftestRun {
    /// Test id, as in the manifest.
    pub test: String,
    /// True under the upstream WPT reftest rule: no `match` references, or
    /// at least one satisfied `match` reference, and every `mismatch`
    /// reference satisfied. See [`reftest_passed`].
    pub passed: bool,
    /// One entry per reference, in manifest order.
    pub comparisons: Vec<ReferenceComparison>,
}

/// What [`BowserDriver::run_reftest`] found for one reference page.
#[derive(Debug, Clone)]
pub struct ReferenceComparison {
    /// The reference href, as given in the test's reference list.
    pub href: String,
    pub rel: RefRel,
    /// Whether the two renderings counted as equal under the tolerance.
    pub equal: bool,
    pub diff_pixels: u64,
    pub diff_fraction: f64,
    /// Whether this reference's requirement was satisfied: `match`
    /// requires equality, `mismatch` requires inequality.
    pub satisfied: bool,
}

/// True when the screenshot is indistinguishable from an untouched canvas:
/// every pixel opaque white.
///
/// The headless pipeline starts from an opaque white canvas (see
/// `bowser_paint::render`), so a page that paints nothing renders
/// all-white. Fixture tests use this as the "non-blank" sanity check:
/// a page that paints its background must not be blank.
pub fn is_blank_canvas(shot: &Screenshot) -> bool {
    let (pixels, rest) = shot.rgba.as_chunks::<4>();
    // A malformed buffer is not a blank canvas.
    rest.is_empty() && pixels.iter().all(|p| *p == [255, 255, 255, 255])
}

/// Mean color of the frame, per channel (alpha included), rounded down.
///
/// For solid-fill fixtures this is the fill color; compare it against the
/// expected color with an explicit per-channel tolerance at the call site
/// (the pipeline may round during rasterization). Returns `[0, 0, 0, 0]`
/// for an empty frame.
pub fn average_color(shot: &Screenshot) -> [u8; 4] {
    let (pixels, _) = shot.rgba.as_chunks::<4>();
    if pixels.is_empty() {
        return [0, 0, 0, 0];
    }
    let mut acc = [0u64; 4];
    for p in pixels {
        for (a, b) in acc.iter_mut().zip(p.iter()) {
            *a += u64::from(*b);
        }
    }
    let n = pixels.len() as u64;
    [
        (acc[0] / n) as u8,
        (acc[1] / n) as u8,
        (acc[2] / n) as u8,
        (acc[3] / n) as u8,
    ]
}

/// Captured output of one child run.
struct ChildOutput {
    status: std::process::ExitStatus,
    stdout: DrainedOutput,
    stderr: DrainedOutput,
}

/// How a child run can fail before producing an exit status.
#[derive(Debug)]
enum ChildError {
    /// The process never started.
    Spawn(String),
    /// Still alive after the timeout; it has been killed.
    Timeout,
    /// Died abnormally (signal) or the wait failed.
    Gone(String),
}

/// Spawn already done; wait for `child` with a timeout, killing it if the
/// timeout expires. Stdout/stderr are drained on threads so a large
/// screenshot can never deadlock the waiter against a full pipe buffer.
/// The child is reaped in all cases.
///
/// The calling thread blocks on a channel (a blocking wait with a
/// timeout), never spinning on `try_wait`: a dedicated reaper thread owns
/// the child and reports its exit status. The reaper notices the exit
/// within [`REAP_QUANTUM`] and services a kill request within the same
/// bound after a timeout. The reaper — not the caller — owns the child,
/// so killing needs no shared lock and cannot deadlock.
///
/// Assumes the headless binary is single-process: on a normal exit its
/// write ends are closed, so the drain threads always see EOF.
fn run_with_timeout(mut child: Child, timeout: Duration) -> Result<ChildOutput, ChildError> {
    let stdout_reader = thread::spawn({
        let pipe = child.stdout.take();
        move || drain(pipe)
    });
    let stderr_reader = thread::spawn({
        let pipe = child.stderr.take();
        move || drain(pipe)
    });
    let (done_tx, done_rx) = mpsc::channel();
    let (kill_tx, kill_rx) = mpsc::channel::<()>();
    let reaper = thread::Builder::new()
        .name("wpt-child-reaper".into())
        .spawn(move || {
            let status = loop {
                match child.try_wait() {
                    Ok(Some(status)) => break Ok(status),
                    Ok(None) => {}
                    Err(e) => break Err(format!("failed waiting for child: {e}")),
                }
                // Block until asked to kill or the next status check: one
                // wakeup per REAP_QUANTUM while the child is alive, not a
                // spin.
                match kill_rx.recv_timeout(REAP_QUANTUM) {
                    Ok(()) => {
                        let _ = child.kill();
                        // Reap so the child never lingers as a zombie;
                        // wait() after kill() returns promptly.
                        break child
                            .wait()
                            .map_err(|e| format!("failed reaping child: {e}"));
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        // The caller is gone; fall back to a plain
                        // blocking wait so the child is still reaped.
                        break child
                            .wait()
                            .map_err(|e| format!("failed waiting for child: {e}"));
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                }
            };
            let _ = done_tx.send(status);
        })
        .map_err(|e| ChildError::Gone(format!("failed to spawn child reaper: {e}")))?;
    // Blocking wait with a timeout: this thread parks until the reaper
    // reports the exit status or the deadline passes.
    match done_rx.recv_timeout(timeout) {
        Ok(Ok(status)) => Ok(ChildOutput {
            status,
            stdout: stdout_reader.join().unwrap_or_default(),
            stderr: stderr_reader.join().unwrap_or_default(),
        }),
        Ok(Err(msg)) => Err(ChildError::Gone(msg)),
        Err(RecvTimeoutError::Disconnected) => Err(ChildError::Gone(
            "child reaper died without reporting an exit status".into(),
        )),
        Err(RecvTimeoutError::Timeout) => {
            // Ask the reaper to kill the child, then wait (bounded) for
            // it to kill and reap.
            let _ = kill_tx.send(());
            let reaped = done_rx.recv_timeout(KILL_GRACE).is_ok();
            if reaped {
                // The reaper killed and reaped the child; join to
                // release it.
                let _ = reaper.join();
            }
            // If the reaper is wedged, detach: joining it could hang the
            // driver. It owns the child, so nothing here can deadlock.
            //
            // Do not join the drain threads here: a misbehaving child
            // could keep a pipe write end open (via a grandchild) and
            // joining would hang the driver. They exit on their own once
            // the pipes hit EOF.
            Err(ChildError::Timeout)
        }
    }
}

/// Cap on bytes kept from one child pipe: a test page spewing stdout
/// grows driver RAM at pipe throughput only until the timeout kill
/// lands, so without a cap a hostile page could force tens of gigabytes
/// before the kill. 16 MiB is far above any legitimate output (a
/// screenshot is bounded by the viewport; marker lines are bytes), and
/// the cut is recorded on the result.
const MAX_DRAIN_BYTES: usize = 16 * 1024 * 1024;

/// One child pipe's captured output. `bytes` is capped at
/// [`MAX_DRAIN_BYTES`]; `truncated` records the cut so a truncated run
/// is never silently reported as clean.
#[derive(Debug, Default)]
struct DrainedOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

/// Read a child pipe to EOF, keeping at most [`MAX_DRAIN_BYTES`];
/// `None` (pipe not captured) yields empty. Reading stops at the cap:
/// the child is killed on timeout anyway, and the drain thread must be
/// joinable on the success path, so it cannot keep blocking on a hostile
/// pipe. (On the timeout path the threads are deliberately detached; a
/// capped Vec bounds their memory until the pipes hit EOF.)
fn drain(pipe: Option<impl Read>) -> DrainedOutput {
    let mut out = DrainedOutput::default();
    let Some(mut p) = pipe else {
        return out;
    };
    let mut chunk = [0u8; 8192];
    loop {
        let n = match p.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        let room = MAX_DRAIN_BYTES.saturating_sub(out.bytes.len());
        let take = n.min(room);
        out.bytes.extend_from_slice(&chunk[..take]);
        if take < n {
            out.truncated = true;
            break;
        }
    }
    out
}

/// Tiny std-only static file server for WPT fixtures.
///
/// Serves `GET` requests from `doc_root` on 127.0.0.1 with an ephemeral
/// port. Both origin-form (`/path`) and absolute-form
/// (`http://127.0.0.1:port/path`) request targets are accepted; the bowser
/// fetch stack sends absolute-form. Paths with `..` segments are rejected
/// (404); directories serve their `index.html`; anything else 404s.
/// Content-Type is sniffed from the file extension. One thread per
/// connection, up to [`MAX_CONNECTIONS`] concurrent; past the cap new
/// connections are rejected instead of spawning threads without bound.
/// The accept loop runs on a detached thread that exits when the
/// [`Shutdown`] handle is dropped.
///
/// wptserve subset: `?pipe=` directives (`status(N)`,
/// `header(Name,Value[,Append])`, `trickle(d1)` / `trickle(100:d1)` /
/// `trickle(100:d1:r2)`), applied in order, and `<path>.headers` sidecar
/// files (`key: value` lines, `[section]` sub-paths, `#` comments). A
/// `trickle` script streams the body in chunks with delays between
/// them; each delay is capped at [`MAX_TRICKLE_MS`] and total sleep at
/// [`MAX_TRICKLE_TOTAL_MS`] per request. Unknown pipe functions and bad
/// arguments are 500s, and pipes never apply to error responses, all
/// matching upstream wptserve.
/// wptserve wrapper synthesis: a request for `<name>.any.html` (or
/// `<name>.window.html`) with no such file is served from the
/// `<name>.any.js` (resp. `.window.js`) sibling, generating the same
/// window-variant wrapper upstream produces (`self.GLOBAL` shim,
/// testharness scripts, `// META:` handling). Worker variants
/// (`.any.worker.html`, …) 404: no worker/global plumbing here.
///
/// `.py` handlers: a request whose path (after decoding) ends in `.py`
/// executes the file as a wptserve `FunctionHandler` (`main(request,
/// response)`), via a persistent `python3` daemon running an embedded
/// wptserve-emulation shim (see [`PY_WPT_SHIM`]). The daemon keeps the
/// server-wide stash across requests; if `python3` is unavailable every
/// `.py` request is a 500. Static files stay `GET`-only; `.py` accepts
/// any method (preflight `OPTIONS`, `POST` echoes, …) and honors
/// `?pipe=` on value-returning handlers, like upstream.
///
/// Subdomains: the server also binds the loopback aliases
/// `127.0.0.2`–`127.0.0.6` on the same port (best effort: an alias that
/// cannot bind is skipped and the server runs primary-only) and routes
/// every `Host` to the same checkout (host-based routing with `Host`
/// handling, but no host allow-list: anything that reaches a listener
/// is served). Files
/// with `.sub.` in the name get wptserve `sub()` template substitution
/// (`{{host}}`, `{{domains[www]}}`, `{{hosts[alt][]}}`,
/// `{{ports[http][0]}}`, `{{location[…]}}`, `{{headers[…]}}`,
/// `{{GET[…]}}`, `{{uuid()}}`, `{{$v:…}}`), so cross-origin tests can
/// discover the alias origins. Upstream spells subdomains
/// `www.127.0.0.1`; here they are additional loopback IP literals
/// because sandbox DNS resolves `*.localhost`/`web-platform.test` to
/// non-loopback egress IPs and `/etc/hosts` is read-only, while IP
/// literals need no DNS at all.
pub struct FixtureServer;

impl FixtureServer {
    /// Start serving `doc_root`. Returns the base URL
    /// (`http://127.0.0.1:{port}`) and a [`Shutdown`] handle; dropping
    /// the handle stops the server.
    pub fn serve_dir(doc_root: &Path) -> std::io::Result<(String, Shutdown)> {
        let doc_root = doc_root.canonicalize()?;
        let primary = TcpListener::bind("127.0.0.1:0")?;
        let port = primary.local_addr()?.port();
        // Subdomain aliases (see [`alias_ip`]): same port, extra
        // loopback IPs, one accept loop each. Best effort: an alias
        // that cannot bind degrades to primary-only instead of failing
        // the whole server.
        let mut listeners = vec![primary];
        for ip in SUBDOMAIN_ALIAS_IPS {
            match TcpListener::bind(format!("{ip}:{port}")) {
                Ok(l) => listeners.push(l),
                Err(e) => eprintln!(
                    "wpt fixture server: subdomain alias {ip} unavailable ({e}); \
                     continuing with 127.0.0.1 only"
                ),
            }
        }
        let addrs: Vec<SocketAddr> = listeners
            .iter()
            .map(TcpListener::local_addr)
            .collect::<std::io::Result<_>>()?;
        let flag = Arc::new(AtomicBool::new(false));
        let active = Arc::new(AtomicUsize::new(0));
        let state = ServerState {
            flag: Arc::clone(&flag),
            addrs,
        };
        let ctx = Arc::new(ServerCtx {
            port,
            py: Arc::new(PyDaemonHandle::new()),
        });
        for listener in listeners {
            let doc_root = doc_root.clone();
            let flag = Arc::clone(&flag);
            let active = Arc::clone(&active);
            let ctx = Arc::clone(&ctx);
            thread::Builder::new()
                .name("wpt-fixture-server".into())
                .spawn(move || accept_loop(listener, doc_root, flag, active, ctx))
                .map_err(|e| {
                    std::io::Error::other(format!("failed to spawn server thread: {e}"))
                })?;
        }
        Ok((format!("http://127.0.0.1:{port}"), Shutdown { state }))
    }
}

/// Shared state used to stop the server.
struct ServerState {
    flag: Arc<AtomicBool>,
    addrs: Vec<SocketAddr>,
}

/// Stops the fixture server when dropped. Every accept loop is woken
/// with a self-connection and the detached threads then exit.
pub struct Shutdown {
    state: ServerState,
}

impl Shutdown {
    /// Stop the server now (equivalent to dropping the handle).
    pub fn shutdown(self) {}
}

impl Drop for Shutdown {
    fn drop(&mut self) {
        self.state.flag.store(true, Ordering::SeqCst);
        // Wake the blocking accept()s so the loops see the flag.
        for addr in &self.state.addrs {
            let _ = TcpStream::connect_timeout(addr, Duration::from_millis(500));
        }
    }
}

/// Per-server context shared by every connection thread: the ephemeral
/// port (for `{{ports[…]}}` substitution) and the lazily spawned `.py`
/// handler daemon.
struct ServerCtx {
    port: u16,
    py: Arc<PyDaemonHandle>,
}

fn accept_loop(
    listener: TcpListener,
    doc_root: PathBuf,
    flag: Arc<AtomicBool>,
    active: Arc<AtomicUsize>,
    ctx: Arc<ServerCtx>,
) {
    for stream in listener.incoming() {
        if flag.load(Ordering::SeqCst) {
            break;
        }
        let Ok(stream) = stream else { continue };
        let Some(_slot) = try_acquire_slot(&active, MAX_CONNECTIONS) else {
            // Past the cap: reject without spawning a thread. The
            // connection is closed; the client sees EOF/reset.
            drop(stream);
            continue;
        };
        let doc_root = doc_root.clone();
        let ctx = Arc::clone(&ctx);
        thread::spawn(move || {
            // Held for the life of the connection; dropping it releases
            // the slot even if handling panics.
            let _slot = _slot;
            handle_connection(stream, &doc_root, &ctx);
        });
    }
}

/// One held connection slot; dropping it releases the slot.
struct ConnectionSlot {
    active: Arc<AtomicUsize>,
}

impl Drop for ConnectionSlot {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Take one connection slot if `cap` is not reached, `None` otherwise.
/// The check-and-take is a single atomic increment, so the active count
/// can never exceed the cap however connections interleave.
fn try_acquire_slot(active: &Arc<AtomicUsize>, cap: usize) -> Option<ConnectionSlot> {
    let prev = active.fetch_add(1, Ordering::SeqCst);
    if prev >= cap {
        active.fetch_sub(1, Ordering::SeqCst);
        None
    } else {
        Some(ConnectionSlot {
            active: Arc::clone(active),
        })
    }
}

fn handle_connection(stream: TcpStream, doc_root: &Path, ctx: &ServerCtx) {
    let outcome = match read_request(&stream) {
        Ok(Some((req, mut reader))) => {
            let mut req = req;
            // `.py` handlers accept request bodies (POST/PUT/OPTIONS…);
            // the static routes ignore (and never read) them.
            if is_py_target(&req.target) {
                match read_request_body(&mut reader, &req.headers) {
                    Ok(body) => req.body = body,
                    Err(status) => {
                        let mut stream = stream;
                        let _ =
                            Response::empty(status, status_reason(status)).write_to(&mut stream);
                        return;
                    }
                }
            }
            route(&req, doc_root, ctx)
        }
        Ok(None) | Err(_) => RouteOutcome::Response(Response::empty(400, "Bad Request")),
    };
    let mut stream = stream;
    match outcome {
        RouteOutcome::Response(response) => {
            let _ = response.write_to(&mut stream);
        }
        RouteOutcome::PyStream(py) => {
            let _ = py.drive(&mut stream);
        }
    }
}

/// One parsed HTTP request: the request line, headers, and — for `.py`
/// targets only — the body.
struct HttpRequest {
    method: String,
    target: String,
    /// `(name, value)` in wire order; names keep their original case,
    /// values are trimmed.
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    /// Peer socket address, for `request.client_address`.
    peer: String,
}

/// Case-insensitive header lookup.
fn header_value<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}

/// Read the request line and headers. Returns `None` on empty/EOF. The
/// `BufReader` is returned so the caller can continue reading the body
/// from the same buffer.
fn read_request(
    stream: &TcpStream,
) -> std::io::Result<Option<(HttpRequest, BufReader<&TcpStream>)>> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let peer = stream
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_default();
    let mut reader = BufReader::new(stream);
    let mut total = 0usize;
    let Some(line) = read_line_capped(&mut reader, &mut total)? else {
        return Ok(None);
    };
    let line = String::from_utf8(line)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "non-utf8 request"))?;
    let mut parts = line.split_whitespace();
    let (Some(method), Some(target), Some(_version)) = (parts.next(), parts.next(), parts.next())
    else {
        return Ok(None);
    };
    let mut headers = Vec::new();
    while let Some(header) = read_line_capped(&mut reader, &mut total)? {
        if header == b"\r\n" || header == b"\n" {
            break;
        }
        // Lenient: a header line without a colon is skipped rather than
        // failing the request; non-UTF-8 bytes become lossy.
        let header = String::from_utf8_lossy(&header);
        if let Some((name, value)) = header.split_once(':') {
            let name = name.trim();
            if !name.is_empty() {
                headers.push((name.to_owned(), value.trim().to_owned()));
            }
        }
    }
    Ok(Some((
        HttpRequest {
            method: method.to_owned(),
            target: target.to_owned(),
            headers,
            body: Vec::new(),
            peer,
        },
        reader,
    )))
}

/// Cap on a `.py` request body: unbounded reads let a hostile client
/// grow connection memory without bound. 16 MiB is far above any
/// legitimate WPT handler body.
const MAX_PY_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Read the request body for a `.py` target, per `Content-Length`,
/// continuing on the request's `BufReader` so already-buffered bytes
/// are not lost. Chunked request bodies are not supported (501).
/// Returns the HTTP status to answer with when the body cannot be read.
fn read_request_body(
    reader: &mut BufReader<&TcpStream>,
    headers: &[(String, String)],
) -> Result<Vec<u8>, u16> {
    if header_value(headers, "transfer-encoding").is_some_and(|v| !v.is_empty()) {
        return Err(501);
    }
    let len: usize = match header_value(headers, "content-length") {
        None => return Ok(Vec::new()),
        Some(v) => v.trim().parse().map_err(|_| 400u16)?,
    };
    if len > MAX_PY_BODY_BYTES {
        return Err(413);
    }
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body).map_err(|_| 400u16)?;
    Ok(body)
}

/// Whether the request target (after decoding) names a `.py` handler.
fn is_py_target(target: &str) -> bool {
    let target = strip_absolute_form(target);
    let (path, _) = split_query(target);
    percent_decode(path).is_some_and(|p| p.ends_with(".py"))
}

fn read_line_capped(
    reader: &mut BufReader<&TcpStream>,
    total: &mut usize,
) -> std::io::Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    let n = reader.read_until(b'\n', &mut line)?;
    if n == 0 {
        return Ok(None);
    }
    *total += n;
    if *total > MAX_REQUEST_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "request too large",
        ));
    }
    Ok(Some(line))
}

/// What routing produced: either a complete response, or a `.py`
/// handler that already sent its head and must now stream body chunks
/// straight to the connection.
enum RouteOutcome {
    Response(Response),
    PyStream(PyStream),
}

/// Route one request. `Host`-based routing: every host that reaches a
/// listener (the primary `127.0.0.1` or any subdomain alias) is served
/// from the same checkout; the `Host` header only feeds `request.url`
/// for `.py` handlers and `{{location}}` substitution.
fn route(req: &HttpRequest, doc_root: &Path, ctx: &ServerCtx) -> RouteOutcome {
    // The bowser fetch stack sends absolute-form request targets
    // (`GET http://127.0.0.1:port/path HTTP/1.1`); reduce those to
    // origin-form before routing.
    let target = strip_absolute_form(&req.target);
    let (path, query) = split_query(target);
    // The testharness harness endpoint: serve the test page with the
    // completion-callback reporter injected — or raw, for the page's
    // relative subresources, which resolve under the same prefix.
    if let Some(inner) = path.strip_prefix(HARNESS_PREFIX) {
        if req.method != "GET" {
            return RouteOutcome::Response(Response::empty(405, "Method Not Allowed"));
        }
        return serve_harness(inner, query, req, doc_root, ctx);
    }
    let decoded = match percent_decode(path) {
        Some(p) => p,
        None => return RouteOutcome::Response(Response::empty(400, "Bad Request")),
    };
    if !decoded.starts_with('/') {
        return RouteOutcome::Response(Response::empty(400, "Bad Request"));
    }
    // Reject traversal before joining: no `..` segment may survive.
    if decoded.split('/').any(|seg| seg == "..") {
        return RouteOutcome::Response(Response::empty(404, "Not Found"));
    }
    // `.py` handlers: executed, never served as source. Unlike the
    // static routes they accept any method (preflight OPTIONS, POST
    // echoes, …), like upstream wptserve.
    if decoded.ends_with(".py") {
        return serve_py(req, &decoded, query, doc_root, ctx);
    }
    if req.method != "GET" {
        return RouteOutcome::Response(Response::empty(405, "Method Not Allowed"));
    }
    // Upstream wptserve raises 404 before the pipeline runs, so `?pipe=`
    // directives only apply to successfully read files; a bad pipe is a
    // 500, like upstream's uncaught pipeline exceptions.
    let sub = SubCtx {
        port: ctx.port,
        headers: &req.headers,
        query,
        url_path: &decoded,
        doc_root,
    };
    let doc = match read_doc_file(&decoded, doc_root, Some(&sub)) {
        Ok(doc) => doc,
        Err(status) => {
            return RouteOutcome::Response(Response::empty(status, status_reason(status)));
        }
    };
    // testdriver.js automation: the vendor shim implementing
    // `test_driver_internal` rides on the served testdriver.js (see
    // `crate::testdriver`); every other file is byte-identical.
    let bytes = crate::testdriver::maybe_append_vendor_shim(&decoded, doc.bytes);
    // wptserve `?pipe=` directives, parsed in order. Trickle delays are
    // applied while the body is written (see `TricklePlan::write_body`),
    // so a trickling response holds no resources other than its
    // connection thread.
    let directives = match query
        .and_then(|q| query_param(q, "pipe"))
        .and_then(percent_decode)
        .map(|p| parse_pipe(&p))
        .transpose()
    {
        Ok(d) => d.unwrap_or_default(),
        Err(_) => return RouteOutcome::Response(Response::empty(500, status_reason(500))),
    };
    let response = Response::new(200, "OK", doc.content_type, bytes);
    let response = apply_sidecar_headers(response, &doc.sidecar_path, doc_root);
    RouteOutcome::Response(apply_pipe(response, &directives))
}

/// One parsed `?pipe=` sub-function. wptserve applies pipe functions in
/// the order they appear, separated by `|`.
#[derive(Debug, Clone, PartialEq)]
enum PipeDirective {
    /// Override the response status code: `status(N)`.
    Status(u16),
    /// Set or append a response header: `header(Name,Value[,Append])`.
    /// Like upstream wptserve, the header REPLACES any same-named
    /// header unless `Append` is `true`/`1` (case-insensitive).
    Header {
        name: String,
        value: String,
        append: bool,
    },
    /// Stream the body in chunks with delays between them:
    /// `trickle(d1)` / `trickle(100:d1)` / `trickle(100:d1:r2)`.
    Trickle(TricklePlan),
}

/// Parsed `trickle(...)` script: delays in milliseconds and byte counts
/// applied in order, plus an optional even repeat count when the last
/// token is `rN` (upstream parity).
#[derive(Debug, Clone, PartialEq)]
struct TricklePlan {
    ops: Vec<TrickleOp>,
    /// Repeat the last `rN` ops (N even) until the whole body is sent.
    repeat_last: Option<usize>,
}

/// One `trickle` step: sleep, or send this many body bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrickleOp {
    DelayMs(u64),
    Bytes(usize),
}

/// Why a `?pipe=` value failed to parse. Any of these is a 500 from the
/// fixture server, matching upstream wptserve (unknown pipe functions
/// raise `KeyError`, arg-count and value errors raise `ValueError`, and
/// the server maps uncaught pipeline exceptions to 500).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipeError {
    /// No pipe function with this name.
    UnknownFunction,
    /// Wrong number of arguments for the function.
    ArgCount,
    /// An argument failed to convert (bad int, bad boolean, bad trickle
    /// component, unbalanced parenthesis, ...).
    BadValue,
}

/// Per-directive cap on one `trickle` delay: a trickling response holds
/// the harness's connection open, and an unbounded delay would stall
/// the run. Upstream has no cap; this is a deliberate hardening.
const MAX_TRICKLE_MS: u64 = 5_000;

/// Total trickle sleep budget per request: repeats can otherwise stretch
/// one response's delays without bound.
const MAX_TRICKLE_TOTAL_MS: u64 = 5_000;

/// Parse a `?pipe=` value into ordered directives. Unknown pipe
/// functions, wrong argument counts, and unconvertible arguments are
/// errors (HTTP 500), matching upstream wptserve — not silently
/// skipped.
///
/// Arguments split on unescaped commas; a backslash escapes the next
/// character (`\,` → `,`, `\n` → newline, ...), per upstream's
/// `PipeTokenizer`.
fn parse_pipe(pipe: &str) -> Result<Vec<PipeDirective>, PipeError> {
    let bytes = pipe.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    let mut directives = Vec::new();
    while i < n {
        while i < n && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= n {
            break;
        }
        // Function name: up to the argument list or the next function.
        let start = i;
        while i < n && bytes[i] != b'(' && bytes[i] != b'|' {
            i += 1;
        }
        let name = pipe[start..i].trim();
        // Argument list: comma-separated, backslash escapes honored; the
        // first unescaped `)` closes the list.
        let mut args: Vec<String> = Vec::new();
        if i < n && bytes[i] == b'(' {
            i += 1;
            let mut cur = String::new();
            let mut closed = false;
            while i < n {
                let c = bytes[i];
                if c == b'\\' {
                    if i + 1 < n {
                        match bytes[i + 1] {
                            b'n' => cur.push('\n'),
                            b'r' => cur.push('\r'),
                            b't' => cur.push('\t'),
                            e => cur.push(e as char),
                        }
                        i += 2;
                    } else {
                        cur.push('\\');
                        i += 1;
                    }
                } else if c == b',' {
                    args.push(std::mem::take(&mut cur));
                    i += 1;
                } else if c == b')' {
                    args.push(std::mem::take(&mut cur));
                    i += 1;
                    closed = true;
                    break;
                } else {
                    cur.push(c as char);
                    i += 1;
                }
            }
            if !closed {
                return Err(PipeError::BadValue);
            }
        }
        while i < n && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < n {
            if bytes[i] == b'|' {
                i += 1;
            } else {
                return Err(PipeError::BadValue);
            }
        }
        directives.push(directive_from(name, &args)?);
    }
    Ok(directives)
}

/// Build one directive from its parsed name and arguments.
fn directive_from(name: &str, args: &[String]) -> Result<PipeDirective, PipeError> {
    match name {
        "status" => {
            if args.len() != 1 {
                return Err(PipeError::ArgCount);
            }
            args[0]
                .trim()
                .parse::<u16>()
                .map(PipeDirective::Status)
                .map_err(|_| PipeError::BadValue)
        }
        "header" => {
            // Upstream: `header(name, value[, append])`; append defaults
            // to false (REPLACE), `true`/`1` appends. Name/value are
            // trimmed (a bowser normalization; upstream keeps spaces).
            if args.len() < 2 || args.len() > 3 {
                return Err(PipeError::ArgCount);
            }
            let hname = args[0].trim();
            if hname.is_empty() {
                return Err(PipeError::BadValue);
            }
            let append = if args.len() == 3 {
                match args[2].trim().to_ascii_lowercase().as_str() {
                    "true" | "1" => true,
                    "false" | "0" => false,
                    _ => return Err(PipeError::BadValue),
                }
            } else {
                false
            };
            Ok(PipeDirective::Header {
                name: hname.to_owned(),
                value: args[1].trim().to_owned(),
                append,
            })
        }
        "trickle" => {
            if args.len() != 1 {
                return Err(PipeError::ArgCount);
            }
            parse_trickle(&args[0]).map(PipeDirective::Trickle)
        }
        _ => Err(PipeError::UnknownFunction),
    }
}

/// Parse a `trickle(...)` delay script, upstream microsyntax: a single
/// colon-separated string where `d<float>` is a delay in SECONDS, bare
/// integers are byte counts, and a final `rN` (N even) repeats the
/// previous N instructions until the whole body is sent. Consecutive
/// same-kind components accumulate, as upstream.
///
/// A repeat block that would send no bytes (e.g. `trickle(d1:r2)`) is
/// dropped instead of hanging the connection the way upstream does.
fn parse_trickle(delays: &str) -> Result<TricklePlan, PipeError> {
    let mut ops: Vec<TrickleOp> = Vec::new();
    let mut repeat: Option<usize> = None;
    let parts: Vec<&str> = delays.split(':').collect();
    for (idx, item) in parts.iter().enumerate() {
        let last = idx + 1 == parts.len();
        let item = item.trim();
        if let Some(rest) = item.strip_prefix('d') {
            let secs: f64 = rest.parse().map_err(|_| PipeError::BadValue)?;
            if !secs.is_finite() || secs < 0.0 {
                return Err(PipeError::BadValue);
            }
            let ms = (secs * 1000.0).round() as u64;
            let ms = ms.min(MAX_TRICKLE_MS);
            match ops.last_mut() {
                Some(TrickleOp::DelayMs(prev)) => {
                    *prev = (*prev).saturating_add(ms).min(MAX_TRICKLE_MS)
                }
                _ => ops.push(TrickleOp::DelayMs(ms)),
            }
        } else if let Some(rest) = item.strip_prefix('r') {
            let r: usize = rest.parse().map_err(|_| PipeError::BadValue)?;
            // Upstream raises unless the repeat count is even (it must
            // cover whole delay+bytes pairs).
            if r == 0 || !r.is_multiple_of(2) {
                return Err(PipeError::BadValue);
            }
            if last {
                if repeat.is_some() {
                    return Err(PipeError::BadValue);
                }
                repeat = Some(r);
            }
            // A non-final `rN` is ignored, as upstream does.
        } else {
            let bytes: usize = item.parse().map_err(|_| PipeError::BadValue)?;
            match ops.last_mut() {
                Some(TrickleOp::Bytes(prev)) => *prev = prev.saturating_add(bytes),
                _ => ops.push(TrickleOp::Bytes(bytes)),
            }
        }
    }
    if ops.is_empty() {
        // `trickle()` / `trickle(r2)`: upstream raises on the empty
        // script, so this is a 500 too.
        return Err(PipeError::BadValue);
    }
    if let Some(r) = repeat {
        let block = &ops[ops.len().saturating_sub(r)..];
        if !block
            .iter()
            .any(|op| matches!(op, TrickleOp::Bytes(n) if *n > 0))
        {
            repeat = None;
        }
    }
    Ok(TricklePlan {
        ops,
        repeat_last: repeat,
    })
}

/// Apply parsed `?pipe=` directives in order: `status(N)` rewrites the
/// status line (last one wins), `header(Name,Value[,Append])` replaces
/// same-named headers by default or appends with `Append=true`/`1`
/// (`trickle` is applied while the body is written).
fn apply_pipe(mut response: Response, directives: &[PipeDirective]) -> Response {
    for d in directives {
        match d {
            PipeDirective::Status(n) => response = response.with_status(*n),
            PipeDirective::Header {
                name,
                value,
                append,
            } => {
                response = if *append {
                    response.push_header(name.clone(), value.clone())
                } else {
                    response.set_header(name.clone(), value.clone())
                };
            }
            PipeDirective::Trickle(plan) => {
                // Upstream resolves the body before re-chunking, so a
                // second trickle replaces the first: last one wins here.
                response.trickle = Some(plan.clone());
                // Upstream's trickle disables caching for the response.
                response = response
                    .set_header(
                        "Cache-Control".into(),
                        "no-cache, no-store, must-revalidate".into(),
                    )
                    .set_header("Pragma".into(), "no-cache".into())
                    .set_header("Expires".into(), "0".into());
            }
        }
    }
    response
}

/// Apply a `<path>.headers` sidecar's headers to a response, in file
/// order. A `Content-Type` entry overrides the sniffed type; everything
/// else is appended (repeatable names are emitted in order).
fn apply_sidecar_headers(response: Response, decoded_path: &str, doc_root: &Path) -> Response {
    let mut response = response;
    for (name, value) in sidecar_headers(decoded_path, doc_root) {
        response = response.push_header(name, value);
    }
    response
}

/// Headers from a `<path>.headers` sidecar file (wptserve convention):
/// `key: value` lines; `[section]` selects headers for one relative
/// path (`*` matches any); `#` comments and blank lines are ignored.
/// Top-level (unsectioned) headers apply to the file's own responses.
fn parse_headers_file(text: &str, rel_path: &str) -> Vec<(String, String)> {
    let mut headers = Vec::new();
    let mut section: Option<&str> = None;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(name) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            section = Some(name.trim());
            continue;
        }
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim();
        if name.is_empty() || name.bytes().any(|b| b.is_ascii_whitespace()) {
            continue;
        }
        let applies = match section {
            None => true,
            Some(s) => s == rel_path || s == "*",
        };
        if applies {
            headers.push((name.to_owned(), value.trim().to_owned()));
        }
    }
    headers
}

/// Read the `<path>.headers` sidecar for `decoded_path` (already
/// traversal-checked), resolved with the same symlink safety as
/// [`read_doc_file`]: directories resolve to their `index.html`, symlinks
/// are canonicalized and re-checked against the doc root. A missing or
/// unreadable sidecar yields no headers.
fn sidecar_headers(decoded_path: &str, doc_root: &Path) -> Vec<(String, String)> {
    let mut fs_path = doc_root.join(decoded_path.trim_start_matches('/'));
    if fs_path.is_dir() {
        fs_path.push("index.html");
    }
    let Ok(canonical) = fs_path.canonicalize() else {
        return Vec::new();
    };
    if !canonical.starts_with(doc_root) {
        return Vec::new();
    }
    let Some(file_name) = canonical.file_name().and_then(|n| n.to_str()) else {
        return Vec::new();
    };
    let sidecar = canonical.with_file_name(format!("{file_name}.headers"));
    let text = match std::fs::read_to_string(&sidecar) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let rel = canonical
        .strip_prefix(doc_root)
        .ok()
        .and_then(|p| p.to_str())
        .unwrap_or("")
        .replace('\\', "/");
    parse_headers_file(&text, &rel)
}

/// Serve one harness-endpoint request: `inner` is the still-
/// percent-encoded path under [`HARNESS_PREFIX`], `query` the raw query
/// string when the target carried one.
///
/// With `__bowser_harness=1&id=…&nonce=…` the test page is served with
/// the reporter injected; without the flag the file is served raw so the
/// instrumented page's relative subresources keep working. The `id` is
/// required for injection (it is stamped into the report); the `nonce`
/// is required too (it is sealed into the reporter's closure so only
/// this run's report lines are accepted). The nonce gate runs before
/// the file lookup: without a valid run nonce the request cannot learn
/// whether a path exists under the prefix, let alone get a reporter
/// injected. The traversal and symlink
/// checks are the same as the normal file route. Both arms honor
/// `?pipe=` and the `<path>.headers` sidecar (sidecar first, pipe
/// directives after), like the plain file route.
fn serve_harness(
    inner: &str,
    query: Option<&str>,
    req: &HttpRequest,
    doc_root: &Path,
    ctx: &ServerCtx,
) -> RouteOutcome {
    let inject = query.and_then(|q| query_param(q, "__bowser_harness")) == Some("1");
    let decoded_inner = match percent_decode(inner) {
        Some(p) => p,
        None => return RouteOutcome::Response(Response::empty(400, "Bad Request")),
    };
    // The prefix strip leaves the path without its leading slash;
    // restore it so the checks below see the absolute form.
    let decoded = format!("/{decoded_inner}");
    if decoded.split('/').any(|seg| seg == "..") {
        return RouteOutcome::Response(Response::empty(404, "Not Found"));
    }
    // `.py` handlers execute even under the harness prefix: a test
    // page's relative subresource may be a `.py` URL, and serving
    // handler source would leak code (and break the subresource).
    if decoded.ends_with(".py") {
        return serve_py(req, &decoded, query, doc_root, ctx);
    }
    // `{{…}}` substitution context for `.sub.` files served here.
    let sub = SubCtx {
        port: ctx.port,
        headers: &req.headers,
        query,
        url_path: &decoded,
        doc_root,
    };
    if !inject {
        // Raw path: the file is served as-is, but `?pipe=` and the
        // `.headers` sidecar still apply, like upstream wptserve. Error
        // responses carry no pipes (upstream raises before the pipeline
        // runs).
        let doc = match read_doc_file(&decoded, doc_root, Some(&sub)) {
            Ok(doc) => doc,
            Err(status) => {
                return RouteOutcome::Response(Response::empty(status, status_reason(status)));
            }
        };
        let directives = match parse_pipe_query(query) {
            Ok(d) => d,
            Err(status) => {
                return RouteOutcome::Response(Response::empty(status, status_reason(status)));
            }
        };
        // Same testdriver.js vendor shim as the plain route: a test's
        // relative subresource may resolve testdriver.js under the
        // harness prefix.
        let bytes = crate::testdriver::maybe_append_vendor_shim(&decoded, doc.bytes);
        let response = Response::new(200, "OK", doc.content_type, bytes);
        let response = apply_sidecar_headers(response, &doc.sidecar_path, doc_root);
        return RouteOutcome::Response(apply_pipe(response, &directives));
    }
    let Some(id) = query
        .and_then(|q| query_param(q, "id"))
        .and_then(percent_decode)
    else {
        return RouteOutcome::Response(Response::empty(400, "Bad Request"));
    };
    // Alphanumeric only: the nonce is spliced into a `<script>` block,
    // and while `json_escape` would neutralize anything hostile, a
    // well-formed run always sends the hex the driver generated.
    let nonce_ok = query
        .and_then(|q| query_param(q, "nonce"))
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_alphanumeric()));
    if !nonce_ok {
        return RouteOutcome::Response(Response::empty(400, "Bad Request"));
    }
    let nonce = query
        .and_then(|q| query_param(q, "nonce"))
        .unwrap_or_default();
    let doc = match read_doc_file(&decoded, doc_root, Some(&sub)) {
        Ok(doc) => doc,
        Err(status) => {
            return RouteOutcome::Response(Response::empty(status, status_reason(status)));
        }
    };
    let bytes = doc.bytes;
    // A genuinely non-UTF-8 page (e.g. a big5-encoded test file) pipes its
    // raw bytes through with a byte-level reporter splice instead of
    // 500ing: the markers are ASCII, so the splice never splits a
    // multibyte sequence.
    let page = match String::from_utf8(bytes) {
        Ok(html) => inject_reporter(&html, &id, nonce).into_bytes(),
        Err(e) => inject_reporter_bytes(e.as_bytes(), &id, nonce),
    };
    // The injected page gets the same `?pipe=` / `.headers` treatment
    // as a raw file: sidecar first, pipe directives after.
    let directives = match parse_pipe_query(query) {
        Ok(d) => d,
        Err(status) => {
            return RouteOutcome::Response(Response::empty(status, status_reason(status)));
        }
    };
    // `text/html` with no charset, like upstream wptserve: the page's own
    // `<meta charset>` must win (e.g. encoding/big5-encoder.html).
    let response = Response::new(200, "OK", "text/html", page);
    let response = apply_sidecar_headers(response, &doc.sidecar_path, doc_root);
    RouteOutcome::Response(apply_pipe(response, &directives))
}

/// Parse the `?pipe=` query parameter into ordered directives;
/// `Err(status)` is the HTTP status to answer with (500 on a bad pipe,
/// like upstream wptserve).
fn parse_pipe_query(query: Option<&str>) -> Result<Vec<PipeDirective>, u16> {
    query
        .and_then(|q| query_param(q, "pipe"))
        .and_then(percent_decode)
        .map(|p| parse_pipe(&p))
        .transpose()
        .map(|d| d.unwrap_or_default())
        .map_err(|_| 500)
}

/// A file read from the doc root (or synthesized from a `.js` sibling).
struct DocFile {
    bytes: Vec<u8>,
    content_type: &'static str,
    /// URL path used for the `<path>.headers` sidecar lookup. Usually the
    /// requested path; for a synthesized wrapper it is the `.js` sibling's
    /// path, matching upstream wptserve (which loads headers for the
    /// underlying `.js` file).
    sidecar_path: String,
}

/// A synthesized wrapper: the document plus the `.js` sibling's URL path
/// (used for the `.headers` sidecar lookup).
struct SynthWrapper {
    html: Vec<u8>,
    js_path: String,
}

/// Wrapper kinds this server synthesizes: `(request suffix, sibling
/// suffix, self.GLOBAL shim, global= exposure check)`. Mirrors upstream
/// `AnyHtmlHandler` (window variant) and `WindowHandler` in
/// `tools/serve/serve.py`.
const WRAPPER_SYNTHESIS: &[(&str, &str, bool, bool)] = &[
    (".any.html", ".any.js", true, true),
    (".window.html", ".window.js", false, false),
];

/// Synthesize the wptserve wrapper for a `<name>.any.html` /
/// `<name>.window.html` request from the `.js` sibling on disk: the
/// `self.GLOBAL` shim (`.any.html` only), `/resources/testharness.js` +
/// `/resources/testharnessreport.js`, the `// META: script=` extras, and
/// the `.js` itself. `// META: title=` / `timeout=long` become `<title>` /
/// `<meta name=timeout>`; a `// META: global=` that excludes `window` is a
/// 404 for the `.any.html` variant, like upstream's exposure check.
///
/// `?variant` queries need no special handling: the query is split off
/// before the file lookup and stays on the page URL, so the `.js` reads it
/// via `location.search` exactly like upstream.
///
/// Returns `Ok(None)` when `decoded_path` is not a synthesized-wrapper
/// request or no sibling exists. Worker variants (`.any.worker.html`,
/// `.any.sharedworker.html`, …) are NOT synthesized and 404: serving them
/// needs worker/global plumbing outside this server's scope.
fn synthesize_wrapper(decoded_path: &str, doc_root: &Path) -> Result<Option<SynthWrapper>, u16> {
    let Some((stem, js_suffix, global_shim, exposure_check)) =
        WRAPPER_SYNTHESIS.iter().find_map(|(req, js, shim, check)| {
            decoded_path
                .strip_suffix(req)
                .map(|s| (s, *js, *shim, *check))
        })
    else {
        return Ok(None);
    };
    let js_path = format!("{stem}{js_suffix}");
    let fs_path = doc_root.join(js_path.trim_start_matches('/'));
    // Same symlink discipline as `read_doc_file`: resolve and re-check.
    // A directory (or an escape) is not a usable sibling: 404. Any other
    // sibling-open failure (permission denied, symlink loop) is also a
    // 404, like upstream wptserve.
    let fs_path = match fs_path.canonicalize() {
        Ok(p) if p.starts_with(doc_root) && p.is_file() => p,
        Ok(_) => return Err(404),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(404),
    };
    let js = match std::fs::read(&fs_path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(404),
    };
    let meta = parse_wrapper_meta(&js);
    if exposure_check && !any_window_exposed(meta.global.as_deref()) {
        return Err(404);
    }
    Ok(Some(SynthWrapper {
        html: render_wrapper(&js_path, &meta, global_shim).into_bytes(),
        js_path,
    }))
}

/// `// META:` directives from the head of a `.js` file, mirroring upstream
/// `read_script_metadata` (`//\s*META:\s*(\w*)=(.*)$` per line; scanning
/// stops at the first non-directive line).
#[derive(Default)]
struct WrapperMeta {
    /// Rendered `<title>` / `<meta name=timeout>` lines, in file order.
    head_lines: Vec<String>,
    /// Rendered `<script src=…>` lines from `// META: script=`, in order.
    script_lines: Vec<String>,
    /// Raw `// META: global=` value (first one wins, like upstream's
    /// exposure check).
    global: Option<String>,
}

fn parse_wrapper_meta(js: &[u8]) -> WrapperMeta {
    let mut meta = WrapperMeta::default();
    for line in js.split(|b| *b == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let Some(rest) = line.strip_prefix(b"//") else {
            break;
        };
        let rest = trim_ascii_ws(rest);
        let Some(rest) = rest.strip_prefix(b"META:") else {
            break;
        };
        let rest = trim_ascii_ws(rest);
        let key_len = rest
            .iter()
            .take_while(|b| b.is_ascii_alphanumeric() || **b == b'_')
            .count();
        let (key, rest) = rest.split_at(key_len);
        let Some(value) = rest.strip_prefix(b"=") else {
            break;
        };
        // Upstream decodes the value as UTF-8; lossy is the std-only
        // equivalent for the rare non-UTF-8 byte.
        let value = String::from_utf8_lossy(value).into_owned();
        match key {
            b"title" => meta
                .head_lines
                .push(format!("<title>{}</title>", escape_html_text(&value))),
            b"timeout" if value == "long" => meta
                .head_lines
                .push("<meta name=\"timeout\" content=\"long\">".to_owned()),
            b"script" => meta.script_lines.push(format!(
                "<script src=\"{}\"></script>",
                escape_html_attr(&value)
            )),
            b"global" if meta.global.is_none() => meta.global = Some(value),
            _ => {}
        }
    }
    meta
}

/// Whether the window-variant wrapper is exposed for this `// META:
// global=` value, mirroring upstream `parse_variants` plus the
/// `AnyHtmlHandler` exposure check. Absent or empty means the default
/// (`window,dedicatedworker`). Only a literal `window` item exposes the
/// window wrapper: `worker` longhands to the three worker kinds and
/// `shadowrealm` to the shadowrealm kinds, neither of which is `window`.
fn any_window_exposed(global: Option<&str>) -> bool {
    let Some(value) = global else {
        return true;
    };
    // Upstream `parse_variants("")` yields the default variants
    // (`window,dedicatedworker`), so an empty value exposes too.
    if value.is_empty() {
        return true;
    }
    value.split(',').map(str::trim).any(|item| item == "window")
}

/// Render the wrapper document in the exact shape of the upstream
/// templates (`tools/serve/serve.py`): `%(meta)s` / `%(script)s` are the
/// `\n`-joined directive lines (leaving a blank line when empty),
/// `%(path)s` the `.js` URL path.
fn render_wrapper(js_url_path: &str, meta: &WrapperMeta, global_shim: bool) -> String {
    let mut out = String::with_capacity(
        560 + js_url_path.len()
            + meta.head_lines.iter().map(String::len).sum::<usize>()
            + meta.script_lines.iter().map(String::len).sum::<usize>(),
    );
    out.push_str("<!doctype html>\n<meta charset=utf-8>\n");
    for line in &meta.head_lines {
        out.push_str(line);
        out.push('\n');
    }
    if meta.head_lines.is_empty() {
        out.push('\n');
    }
    if global_shim {
        out.push_str(
            "<script>\nself.GLOBAL = {\n  isWindow: function() { return true; },\n  isWorker: function() { return false; },\n  isShadowRealm: function() { return false; },\n};\n</script>\n",
        );
    }
    out.push_str(
        "<script src=\"/resources/testharness.js\"></script>\n<script src=\"/resources/testharnessreport.js\"></script>\n",
    );
    for line in &meta.script_lines {
        out.push_str(line);
        out.push('\n');
    }
    if meta.script_lines.is_empty() {
        out.push('\n');
    }
    out.push_str("<div id=log></div>\n");
    // Request-derived path, like every other sink in this function:
    // escaping keeps a crafted path from injecting markup.
    out.push_str(&format!(
        "<script src=\"{}\"></script>\n",
        escape_html_attr(js_url_path)
    ));
    out
}

/// Trim ASCII whitespace (Python `\s` on bytes: tab, LF, FF, CR, space)
/// from both ends.
fn trim_ascii_ws(mut b: &[u8]) -> &[u8] {
    while let Some((&c, rest)) = b.split_first() {
        if matches!(c, b'\t' | b'\n' | 0x0C | b'\r' | b' ') {
            b = rest;
        } else {
            break;
        }
    }
    while let Some((&c, rest)) = b.split_last() {
        if matches!(c, b'\t' | b'\n' | 0x0C | b'\r' | b' ') {
            b = rest;
        } else {
            break;
        }
    }
    b
}

/// Escape `&` and `<` for HTML text (upstream title handling).
fn escape_html_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            _ => out.push(c),
        }
    }
    out
}

/// Escape `&` and `"` for a double-quoted attribute value (upstream script
/// handling).
fn escape_html_attr(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

/// Read `decoded_path` (traversal-checked, starts with `/`) from under
/// `doc_root`: directories serve their `index.html`; symlinks are
/// resolved and re-checked against the doc root. A request for
/// `<name>.any.html` (or `<name>.window.html`) with no such file on disk
/// falls back to synthesizing the wptserve wrapper from the `.js` sibling
/// (see [`synthesize_wrapper`]). A file whose name contains `.sub.` gets
/// wptserve `sub()` template substitution (see [`apply_sub_template`])
/// when `sub` is provided; a template error is a 500, like upstream.
/// Returns the file, or the HTTP status for the failure.
fn read_doc_file(
    decoded_path: &str,
    doc_root: &Path,
    sub: Option<&SubCtx<'_>>,
) -> Result<DocFile, u16> {
    let mut fs_path = doc_root.join(decoded_path.trim_start_matches('/'));
    if fs_path.is_dir() {
        fs_path.push("index.html");
    }
    // Resolve symlinks: a symlink inside the doc root can point anywhere,
    // and the `..` check above cannot see through it. `doc_root` is
    // already canonical (see `serve_dir`), so the prefix check keeps every
    // served path under it.
    let fs_path = match fs_path.canonicalize() {
        Ok(p) if p.starts_with(doc_root) => p,
        Ok(_) => return Err(404),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // No such file: the request may still be satisfiable from a
            // `.js` sibling (wptserve generates these wrappers).
            return match synthesize_wrapper(decoded_path, doc_root)? {
                Some(s) => Ok(DocFile {
                    bytes: s.html,
                    content_type: "text/html",
                    sidecar_path: s.js_path,
                }),
                None => Err(404),
            };
        }
        Err(_) => return Err(500),
    };
    let content_type = content_type(&fs_path);
    match std::fs::read(&fs_path) {
        Ok(bytes) => {
            let bytes = match sub {
                Some(s) if decoded_path.contains(".sub.") => {
                    let escape_html = matches!(
                        fs_path.extension().and_then(|e| e.to_str()),
                        Some("html" | "htm" | "xht" | "xhtml" | "xml" | "svg")
                    );
                    apply_sub_template(&bytes, s, escape_html).map_err(|_| 500u16)?
                }
                _ => bytes,
            };
            Ok(DocFile {
                bytes,
                content_type,
                sidecar_path: decoded_path.to_owned(),
            })
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(404),
        Err(_) => Err(500),
    }
}

/// Reason phrase for a synthesized status. Also used when
/// `?pipe=status(N)` rewrites the status line.
fn status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        301 => "Moved Permanently",
        302 => "Found",
        304 => "Not Modified",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        418 => "I'm a teapot",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        _ => "Error",
    }
}

/// Split a request target into path and query at the first `?`.
fn split_query(target: &str) -> (&str, Option<&str>) {
    match target.split_once('?') {
        Some((path, query)) => (path, Some(query)),
        None => (target, None),
    }
}

/// Value of the `name` query parameter, or `None` when absent. The first
/// match wins; the value is still percent-encoded.
fn query_param<'a>(query: &'a str, name: &str) -> Option<&'a str> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k == name).then_some(v)
    })
}

/// Strip `scheme://authority` from an absolute-form request target,
/// returning the origin-form path (`http://h:1/a?b` -> `/a?b`).
/// Anything without an `http(s)://` prefix passes through unchanged; a
/// target with no path at all maps to `/`.
fn strip_absolute_form(target: &str) -> &str {
    let rest = target
        .strip_prefix("http://")
        .or_else(|| target.strip_prefix("https://"));
    let Some(rest) = rest else {
        return target;
    };
    match rest.find('/') {
        Some(i) => &rest[i..],
        None => "/",
    }
}

/// Decode `%XX` escapes. `+` is left alone (it is literal in a path).
/// Returns `None` on malformed escapes.
fn percent_decode_bytes(s: &str) -> Option<Vec<u8>> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && bytes.len() - i >= 3 {
            let hi = hex_val(bytes[i + 1])?;
            let lo = hex_val(bytes[i + 2])?;
            out.push(hi << 4 | lo);
            i += 3;
        } else if bytes[i] == b'%' {
            return None;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    Some(out)
}

/// Decode `%XX` escapes. `+` is left alone (it is literal in a path).
/// Returns `None` on malformed escapes or non-UTF-8 output.
fn percent_decode(s: &str) -> Option<String> {
    String::from_utf8(percent_decode_bytes(s)?).ok()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Content-Type by file extension.
///
/// `.html`/`.htm` map to bare `text/html`, matching upstream wptserve
/// (`tools/serve/serve.py`): no `charset` parameter. This is not a
/// charset-override fix — bowser's document pipeline never consults the
/// transport charset for documents (`Document::parse_html(&response.body)`
/// → `decode_input`: BOM, then `<meta charset>` prescan, then UTF-8
/// fallback), so a charset parameter would be inert here either way.
/// The bare header just matches upstream.
fn content_type(path: &Path) -> &'static str {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext.eq_ignore_ascii_case("html") || ext.eq_ignore_ascii_case("htm") {
        "text/html"
    } else if ext.eq_ignore_ascii_case("js") || ext.eq_ignore_ascii_case("mjs") {
        "text/javascript; charset=utf-8"
    } else if ext.eq_ignore_ascii_case("css") {
        "text/css; charset=utf-8"
    } else if ext.eq_ignore_ascii_case("json") {
        "application/json"
    } else if ext.eq_ignore_ascii_case("png") {
        "image/png"
    } else if ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("jpeg") {
        "image/jpeg"
    } else if ext.eq_ignore_ascii_case("gif") {
        "image/gif"
    } else if ext.eq_ignore_ascii_case("svg") {
        "image/svg+xml"
    } else if ext.eq_ignore_ascii_case("webp") {
        "image/webp"
    } else if ext.eq_ignore_ascii_case("txt") {
        "text/plain; charset=utf-8"
    } else if ext.eq_ignore_ascii_case("xml") {
        "application/xml"
    } else if ext.eq_ignore_ascii_case("xhtml") {
        "application/xhtml+xml"
    } else if ext.eq_ignore_ascii_case("wasm") {
        "application/wasm"
    } else if ext.eq_ignore_ascii_case("woff") {
        "font/woff"
    } else if ext.eq_ignore_ascii_case("woff2") {
        "font/woff2"
    } else if ext.eq_ignore_ascii_case("ttf") {
        "font/ttf"
    } else {
        "application/octet-stream"
    }
}

/// One HTTP response. `extra_headers` carry the repeatable headers from
/// `?pipe=header(...)` and `<path>.headers` sidecars, emitted in order
/// after `content-type`.
struct Response {
    status: u16,
    reason: std::borrow::Cow<'static, str>,
    content_type: std::borrow::Cow<'static, str>,
    extra_headers: Vec<(String, String)>,
    body: Vec<u8>,
    /// Attached by `?pipe=trickle(...)`: the body is streamed in chunks
    /// with delays instead of going out in one write.
    trickle: Option<TricklePlan>,
    /// When false, the `content-length` line is not auto-emitted: the
    /// handler supplied its own (`.py` responses only).
    auto_content_length: bool,
}

impl Response {
    fn new(
        status: u16,
        reason: &'static str,
        content_type: &'static str,
        body: Vec<u8>,
    ) -> Response {
        Response {
            status,
            reason: reason.into(),
            content_type: content_type.into(),
            extra_headers: Vec::new(),
            body,
            trickle: None,
            auto_content_length: true,
        }
    }

    fn empty(status: u16, reason: &'static str) -> Response {
        Response::new(status, reason, "text/plain; charset=utf-8", Vec::new())
    }

    /// A daemon-supplied reason phrase or header name/value must not
    /// carry CR or LF: the daemon runs test-checkout code, and writing
    /// them verbatim into the status line / header block would be
    /// response splitting. Rejected with `InvalidData` (surfaced as a
    /// 500); the same check guards the streaming path in [`serve_py`].
    fn py_head_parts_valid(reason: &[u8], headers: &[(Vec<u8>, Vec<u8>)]) -> std::io::Result<()> {
        fn clean(b: &[u8]) -> bool {
            !b.iter().any(|&c| c == b'\r' || c == b'\n')
        }
        if clean(reason) && headers.iter().all(|(n, v)| clean(n) && clean(v)) {
            Ok(())
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "py-daemon head contains CR or LF",
            ))
        }
    }

    /// Build a response from a `.py` daemon result: `reason` is the
    /// handler's reason phrase (falling back to the standard one when
    /// empty), `headers` are emitted verbatim except `content-type`,
    /// which overrides the sniffed type like [`Response::push_header`].
    /// An explicit `content-length` header disables the auto-emitted
    /// one, matching upstream wptserve. A CR/LF in the reason or any
    /// header name/value is `InvalidData` (response splitting).
    fn from_py(
        status: u16,
        reason: &[u8],
        headers: Vec<(Vec<u8>, Vec<u8>)>,
        body: Vec<u8>,
    ) -> std::io::Result<Response> {
        Self::py_head_parts_valid(reason, &headers)?;
        let reason = if reason.is_empty() {
            std::borrow::Cow::Borrowed(status_reason(status))
        } else {
            std::borrow::Cow::Owned(String::from_utf8_lossy(reason).into_owned())
        };
        let mut response = Response {
            status,
            reason,
            content_type: "".into(),
            extra_headers: Vec::new(),
            body,
            trickle: None,
            auto_content_length: true,
        };
        for (name, value) in headers {
            let value = String::from_utf8_lossy(&value).into_owned();
            if name.eq_ignore_ascii_case(b"content-type") {
                response.content_type = value.into();
            } else {
                if name.eq_ignore_ascii_case(b"content-length") {
                    response.auto_content_length = false;
                }
                response
                    .extra_headers
                    .push((String::from_utf8_lossy(&name).into_owned(), value));
            }
        }
        Ok(response)
    }

    /// Rewrite the status line (used by `?pipe=status(N)`).
    fn with_status(mut self, status: u16) -> Response {
        self.status = status;
        self.reason = status_reason(status).into();
        self
    }

    /// Append a raw response header. A `Content-Type` header overrides
    /// the sniffed type instead of duplicating it; any other name is
    /// appended verbatim, so repeatable names are emitted in order.
    fn push_header(mut self, name: String, value: String) -> Response {
        if name.eq_ignore_ascii_case("content-type") {
            self.content_type = std::borrow::Cow::Owned(value);
        } else {
            self.extra_headers.push((name, value));
        }
        self
    }

    /// Set a response header, replacing any same-named header
    /// (case-insensitive). This is `?pipe=header(Name,Value)` without
    /// the append flag, matching upstream wptserve.
    fn set_header(mut self, name: String, value: String) -> Response {
        if name.eq_ignore_ascii_case("content-type") {
            self.content_type = std::borrow::Cow::Owned(value);
        } else {
            self.extra_headers
                .retain(|(n, _)| !n.eq_ignore_ascii_case(&name));
            self.extra_headers.push((name, value));
        }
        self
    }

    fn head_bytes(&self) -> Vec<u8> {
        let mut head = format!("HTTP/1.1 {} {}\r\n", self.status, self.reason).into_bytes();
        // Upstream wptserve emits no content-type at all when the
        // handler never set one (`.py` responses only; the static
        // routes always pass a sniffed type).
        if !self.content_type.is_empty() {
            head.extend_from_slice(b"content-type: ");
            head.extend_from_slice(self.content_type.as_bytes());
            head.extend_from_slice(b"\r\n");
        }
        for (name, value) in &self.extra_headers {
            head.extend_from_slice(name.as_bytes());
            head.extend_from_slice(b": ");
            head.extend_from_slice(value.as_bytes());
            head.extend_from_slice(b"\r\n");
        }
        if self.auto_content_length {
            head.extend_from_slice(format!("content-length: {}\r\n", self.body.len()).as_bytes());
        }
        head.extend_from_slice(b"connection: close\r\n\r\n");
        head
    }

    /// Write the response to the connection: head first, then the body —
    /// streamed in chunks with delays when a `trickle` plan is attached,
    /// in one write otherwise.
    fn write_to(&self, stream: &mut TcpStream) -> std::io::Result<()> {
        stream.write_all(&self.head_bytes())?;
        match &self.trickle {
            None => stream.write_all(&self.body),
            Some(plan) => plan.write_body(stream, &self.body),
        }
    }
}

impl TricklePlan {
    /// Stream `body` through `stream` per the parsed script, mirroring
    /// upstream wptserve's generator: `Bytes(n)` sends the next n bytes,
    /// `DelayMs` sleeps, a final `rN` re-runs the last N ops until the
    /// whole body is out, and any unsent remainder goes out at the end.
    /// Total sleep is bounded by [`MAX_TRICKLE_TOTAL_MS`]; past the
    /// budget the remaining delays are skipped (bytes are always sent).
    fn write_body(&self, stream: &mut TcpStream, body: &[u8]) -> std::io::Result<()> {
        let mut slept_ms: u64 = 0;
        let mut offset = 0usize;
        let run = |ops: &[TrickleOp],
                   offset: &mut usize,
                   slept_ms: &mut u64,
                   stream: &mut TcpStream|
         -> std::io::Result<()> {
            for op in ops {
                match op {
                    TrickleOp::DelayMs(ms) => {
                        let budget_left = MAX_TRICKLE_TOTAL_MS.saturating_sub(*slept_ms);
                        let sleep_ms = (*ms).min(budget_left);
                        if sleep_ms > 0 {
                            thread::sleep(Duration::from_millis(sleep_ms));
                            *slept_ms += sleep_ms;
                        }
                    }
                    TrickleOp::Bytes(n) => {
                        let end = (*offset + n).min(body.len());
                        stream.write_all(&body[*offset..end])?;
                        *offset = end;
                    }
                }
            }
            Ok(())
        };
        match self.repeat_last {
            Some(r) => {
                let start = self.ops.len().saturating_sub(r);
                let block = &self.ops[start..];
                // A block that sends no bytes would loop forever; the
                // parser already drops those, this is belt-and-braces.
                if block
                    .iter()
                    .any(|op| matches!(op, TrickleOp::Bytes(n) if *n > 0))
                {
                    while offset < body.len() {
                        run(block, &mut offset, &mut slept_ms, &mut *stream)?;
                    }
                    return Ok(());
                }
                run(&self.ops, &mut offset, &mut slept_ms, &mut *stream)?;
            }
            None => run(&self.ops, &mut offset, &mut slept_ms, &mut *stream)?,
        }
        if offset < body.len() {
            stream.write_all(&body[offset..])?;
        }
        Ok(())
    }
}

// ============================================================
// `.py` handlers: python3 daemon + subdomain aliases + `{{…}}`
// substitution.
// ============================================================

/// Design choice (why not a Python-subset interpreter, and why not
/// native reimplementation): upstream wptserve executes `.py` handlers
/// in-process as `main(request, response)`, and the handlers in the
/// target suites use arbitrary Python (`uuid` validation, `hmac`-less
/// auth parsing, `time.sleep` trickles, `json`, `base64`, `datetime`
/// cookie expiry, `re`, `urllib`) plus the wptserve API surface
/// (`request.GET/POST/headers/body/cookies/auth/server.stash`,
/// `response.writer` streaming, `wptserve.utils.isomorphic_*`,
/// `wptserve.handlers.json_handler`). A Python-subset interpreter is
/// unbounded work; reimplementing handler-by-handler is a long tail of
/// subtle divergence. So the fixture server shells out to `python3`,
/// which exists on every WPT test host: one persistent daemon process
/// (not one spawn per request) amortizes interpreter startup and,
/// critically, keeps `request.server.stash` alive across requests —
/// stash-put/take roundtrips, redirect loops, and preflight
/// correlation all depend on server-wide mutable state. The daemon
/// runs the embedded [`PY_WPT_SHIM`], a std-only emulation of exactly
/// the wptserve pieces those handlers use (semantics verified against
/// wptserve 4.0.3). If `python3` is unavailable, every `.py` request is
/// a 500 (logged once); `.py` URLs never fall back to serving source.
///
/// Protocol: the daemon reads one JSON object per line on stdin
/// (`id`, `script`, `method`, `target`, `url`, `peer`, `headers`,
/// `body_b64`, `doc_root`; or `{"id","abort":true}`) and writes
/// little-endian binary frames on stdout (see the shim header).
/// stdout is unbuffered (`-u`) so streaming handlers flush promptly.
///
/// Known gaps, all documented at the use site: multipart
/// `request.POST` (urlencoded only), `{{file_hash(…)}}` (no crypto in
/// the std-only server), `.h2.py` handlers (HTTP/1-only server: no
/// `main` → 500, exactly upstream's "No main function" path), and
/// `Server`/`Date` default headers (this server never emits them).
const PY_WPT_SHIM: &str = r###"# Bowser FixtureServer `.py` handler daemon: a std-only emulation of the
# wptserve pieces that `.py` handlers in the WPT checkout actually use
# (wptserve 4.x semantics: FunctionHandler, PythonScriptHandler,
# JsonHandler, Request/Response/ResponseWriter, MultiDict, Stash).
#
# Protocol (stdin): one JSON object per line per request:
#   {"id": N, "script": "/abs/handler.py", "method": "GET",
#    "target": "/path?query", "url": "http://host:port/path?query",
#    "peer": "127.0.0.1:1234", "headers": [[name, value], ...],
#    "body_b64": "<base64>", "doc_root": "/abs/docroot"}
# or {"id": N, "abort": true} to stop a streaming response.
#
# Protocol (stdout): binary frames, little-endian:
#   u64 id, u8 kind (0=Response, 1=Head, 2=Chunk, 3=End),
#   for 0: u8 flags (bit 0: the Rust side should apply ?pipe=, i.e. the
#          handler returned a value without streaming, like upstream's
#          wrap_pipeline running only when rv is not None),
#   for 0/1: u16 status, u16 reason_len, reason, u32 nheaders,
#            per header u32 nlen, name, u32 vlen, value,
#   for 0/2: u64 body_len, body.
#
# One thread per request (like wptserve). The stash is process-global so
# it persists across requests. Handler tracebacks go to stderr.

import base64
import io
import json
import os
import struct
import sys
import threading
import traceback
import types
from datetime import datetime, timedelta, timezone
from http.cookies import Morsel
from urllib.parse import urlsplit, parse_qsl


# --------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------

_FH_RESPONSE, _FH_HEAD, _FH_CHUNK, _FH_END = 0, 1, 2, 3

_out = sys.stdout.buffer
# Frames ride on stdout, so a handler's stray print() would corrupt the
# frame stream and deafen the daemon; rebind stdout to stderr and swap
# stdin for an empty buffer so input() cannot swallow request lines
# either. The real stdin pipe is captured first for the main loop.
_stdin = sys.stdin.buffer
sys.stdout = sys.stderr
sys.stdin = io.StringIO()
_out_lock = threading.Lock()


def _to_bytes(v):
    if isinstance(v, bytes):
        return v
    return str(v).encode("iso-8859-1")


def _send(rid, kind, status=0, reason=b"", headers=(), body=b"", flags=0):
    frame = struct.pack("<QB", rid, kind)
    if kind == _FH_RESPONSE:
        frame += struct.pack("<B", flags)
    if kind in (_FH_RESPONSE, _FH_HEAD):
        rb = _to_bytes(reason)
        frame += struct.pack("<H", status)
        frame += struct.pack("<H", len(rb)) + rb
        frame += struct.pack("<I", len(headers))
        for k, v in headers:
            kb, vb = _to_bytes(k), _to_bytes(v)
            frame += struct.pack("<I", len(kb)) + kb
            frame += struct.pack("<I", len(vb)) + vb
    if kind in (_FH_RESPONSE, _FH_CHUNK):
        frame += struct.pack("<Q", len(body)) + body
    with _out_lock:
        _out.write(frame)
        _out.flush()


# --------------------------------------------------------------------------
# wptserve API stubs
# --------------------------------------------------------------------------

def isomorphic_encode(s):
    if isinstance(s, bytes):
        return s
    if isinstance(s, str):
        return s.encode("iso-8859-1")
    raise TypeError("Unexpected value (expecting string-like): %r" % (s,))


def isomorphic_decode(s):
    if isinstance(s, str):
        return s
    if isinstance(s, bytes):
        return s.decode("iso-8859-1")
    raise TypeError("Unexpected value (expecting string-like): %r" % (s,))


class HTTPException(Exception):
    def __init__(self, code, message=""):
        super().__init__(message)
        self.code = code
        self.message = message


def _install_module(name, parent_name=None):
    mod = types.ModuleType(name)
    if parent_name:
        parent = sys.modules[parent_name]
        setattr(parent, name.split(".")[-1], mod)
    sys.modules[name] = mod
    return mod


_wptserve = _install_module("wptserve")
_wptserve_utils = _install_module("wptserve.utils", "wptserve")
_wptserve_utils.isomorphic_encode = isomorphic_encode
_wptserve_utils.isomorphic_decode = isomorphic_decode
_wptserve_handlers = _install_module("wptserve.handlers", "wptserve")


def json_handler(func):
    """Mirror of wptserve.handlers.json_handler: JSON-encode the return
    value (or the last tuple element), with Content-Type/Length set, and
    return it for the normal FunctionHandler mapping."""

    def handle_request(request, response):
        rv = func(request, response)
        response.headers.set("Content-Type", "application/json")
        if isinstance(rv, tuple):
            rv = list(rv)
            value = tuple(rv[:-1] + [json.dumps(rv[-1])])
            length = len(value[-1])
        else:
            value = json.dumps(rv)
            length = len(value)
        response.headers.set("Content-Length", str(length))
        return value

    handle_request.__name__ = getattr(func, "__name__", "json_handler")
    return handle_request


_wptserve_handlers.json_handler = json_handler


def load_source(name, path):
    """Minimal stand-in for tools.wpt.utils.load_source: exec a file and
    return it as a module."""
    with open(path, "rb") as f:
        source = f.read()
    mod = types.ModuleType(name or "__wpt_load_source__")
    mod.__file__ = path
    code = compile(source, path, "exec")
    exec(code, mod.__dict__)
    return mod


_tools = _install_module("tools")
_tools_wpt = _install_module("tools.wpt", "tools")
_tools_wpt_utils = _install_module("tools.wpt.utils", "tools.wpt")
_tools_wpt_utils.load_source = load_source


# --------------------------------------------------------------------------
# Request side
# --------------------------------------------------------------------------

_MISSING = object()


class MultiDict:
    def __init__(self):
        self._d = {}

    def add(self, name, value):
        self._d.setdefault(name, []).append(value)

    def __setitem__(self, name, value):
        self._d[name] = [value]

    def __contains__(self, name):
        return name in self._d

    def __getitem__(self, name):
        return self.first(name)

    def first(self, name, default=_MISSING):
        if name in self._d and self._d[name]:
            return self._d[name][0]
        if default is not _MISSING:
            return default
        raise KeyError(name)

    def get(self, name, default=None):
        return self.first(name, default)

    def get_list(self, name):
        return list(self._d.get(name, []))

    def items(self):
        return [(k, list(v)) for k, v in self._d.items()]


class RequestHeaders:
    def __init__(self, items):
        self._d = {}
        for k, v in items:
            self._d.setdefault(k.lower(), []).append(v)

    def __contains__(self, name):
        return isomorphic_encode(name).lower() in self._d

    def __getitem__(self, name):
        values = self._d[isomorphic_encode(name).lower()]
        if len(values) == 1:
            return values[0]
        return b", ".join(values)

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def __iter__(self):
        return iter(self._d)


class RawHeaders:
    """Keeps the raw header lines so str(request.raw_headers) works."""

    def __init__(self, items):
        self._items = items

    def __str__(self):
        return "".join(
            "%s: %s\r\n" % (isomorphic_decode(k), isomorphic_decode(v))
            for k, v in self._items
        )


class CookieValue:
    def __init__(self, value):
        self.value = value


def _parse_cookies(header_value):
    cookies = {}
    for part in header_value.split(b";"):
        if b"=" not in part:
            continue
        name, _, value = part.partition(b"=")
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[:1] == value[-1:] == b'"':
            value = value[1:-1]
        if name and name not in cookies:
            cookies[name] = CookieValue(value)
    return cookies


class Authentication:
    def __init__(self, headers):
        self.username = None
        self.password = None
        if "authorization" in headers:
            header = headers.get("authorization")
            parts = header.split(b" ", 1)
            if len(parts) != 2:
                raise ValueError("Bad Authorization header")
            auth_type, data = parts
            if auth_type != b"Basic":
                raise HTTPException(400, "Unsupported authentication scheme")
            decoded = base64.b64decode(data)
            user, _, password = decoded.partition(b":")
            self.username = user
            self.password = password


class StashError(Exception):
    pass


class Stash:
    """Write-once, read-once (destructive take), path-scoped key/value
    store. Keys are arbitrary strings (upstream wptserve imposes no
    format); callers conventionally use UUIDs to avoid collisions."""

    def __init__(self):
        self._d = {}
        self._lock = threading.Lock()

    def _wrap(self, key, path):
        if isinstance(key, bytes):
            key = key.decode("ascii")
        if not isinstance(key, str) or not key:
            raise ValueError('Invalid stash key %r' % (key,))
        return (isomorphic_encode(path), key)

    def put(self, key, value, path=None):
        if value is None:
            raise ValueError("SharedStash value may not be set to None")
        if path is None:
            raise ValueError("stash path required")
        k = self._wrap(key, path)
        with self._lock:
            if k in self._d:
                raise StashError(
                    "Tried to overwrite existing shared stash value for key %s" % (k,)
                )
            self._d[k] = value

    def take(self, key, path=None):
        if path is None:
            raise ValueError("stash path required")
        k = self._wrap(key, path)
        with self._lock:
            return self._d.pop(k, None)


_STASH = Stash()


class Server:
    def __init__(self, request):
        self._request = request

    @property
    def stash(self):
        return _StashProxy(self._request.url_parts.path)


class _StashProxy:
    """Binds the request's default path like wptserve's per-request Stash."""

    def __init__(self, default_path):
        self._default_path = default_path

    def put(self, key, value, path=None):
        _STASH.put(key, value, self._default_path if path is None else path)

    def take(self, key, path=None):
        return _STASH.take(key, self._default_path if path is None else path)


class Request:
    def __init__(self, method, target, url, headers, body, doc_root, peer):
        self.method = method
        self.target = target
        self.url = url
        self.url_parts = urlsplit(url)
        self.doc_root = doc_root
        self.request_path = target
        self.url_base = "/"
        self.client_address = peer
        self.raw_headers = RawHeaders(headers)
        self._headers = RequestHeaders(headers)
        self._body = body
        self._GET = None
        self._POST = None
        self._cookies = None
        self._auth = None
        self.server = Server(self)

    @property
    def headers(self):
        return self._headers

    @property
    def GET(self):
        if self._GET is None:
            params = MultiDict()
            for k, v in parse_qsl(
                self.url_parts.query, keep_blank_values=True, encoding="iso-8859-1"
            ):
                params.add(isomorphic_encode(k), isomorphic_encode(v))
            self._GET = params
        return self._GET

    @property
    def POST(self):
        # Upstream uses cgi.FieldStorage (multipart-capable); urlencoded
        # covers the handlers in the target suites.
        if self._POST is None:
            params = MultiDict()
            ctype = self._headers.get(b"content-type", b"").split(b";")[0].strip()
            if ctype == b"application/x-www-form-urlencoded":
                for k, v in parse_qsl(
                    self._body.decode("iso-8859-1"),
                    keep_blank_values=True,
                    encoding="iso-8859-1",
                ):
                    params.add(isomorphic_encode(k), isomorphic_encode(v))
            self._POST = params
        return self._POST

    @property
    def body(self):
        return self._body

    @property
    def cookies(self):
        if self._cookies is None:
            self._cookies = _parse_cookies(self._headers.get(b"cookie", b""))
        return self._cookies

    @property
    def auth(self):
        if self._auth is None:
            self._auth = Authentication(self._headers)
        return self._auth


# --------------------------------------------------------------------------
# Response side
# --------------------------------------------------------------------------

_REASONS = {
    200: "OK", 201: "Created", 204: "No Content",
    301: "Moved Permanently", 302: "Found", 304: "Not Modified",
    400: "Bad Request", 401: "Unauthorized", 403: "Forbidden",
    404: "Not Found", 405: "Method Not Allowed",
    418: "I'm a teapot", 500: "Internal Server Error",
}


class ResponseHeaders:
    def __init__(self):
        self._d = {}

    def set(self, name, value):
        name = isomorphic_encode(name)
        value = isomorphic_encode(value)
        self._d[name.lower()] = (name, [value])

    def append(self, name, value):
        name = isomorphic_encode(name)
        value = isomorphic_encode(value)
        if name.lower() in self._d:
            self._d[name.lower()][1].append(value)
        else:
            self.set(name, value)

    def get(self, name, default=_MISSING):
        try:
            return self[name]
        except KeyError:
            if default is _MISSING:
                return []
            return default

    def __getitem__(self, name):
        return list(self._d[isomorphic_encode(name).lower()][1])

    def __contains__(self, name):
        return isomorphic_encode(name).lower() in self._d

    def update(self, items):
        for name, value in items:
            self.append(name, value)

    def items(self):
        for name, values in self._d.values():
            for value in values:
                yield name, value


class ResponseWriter:
    """Buffers the head until end_headers(), then emits one Head frame;
    body chunks become Chunk frames. write() returning False means the
    client went away (upstream: OSError on the socket)."""

    def __init__(self, response):
        self._response = response
        self._status = None
        self._head = []
        self._seen = set()
        self._complete = False
        self.content_written = False

    def _aborted(self):
        return self._response._aborted.is_set()

    def write_status(self, code, message=None):
        code = int(code)
        if message is None:
            message = _REASONS.get(code, "")
        if self._status is None:
            self._status = (code, message)
        return not self._aborted()

    def write_header(self, name, value):
        if self._status is None:
            self.write_status(200)
        name = isomorphic_encode(name)
        value = isomorphic_encode(value) if not isinstance(value, int) else str(value).encode()
        self._seen.add(name.lower())
        self._head.append((name, value))
        return not self._aborted()

    def end_headers(self):
        # Upstream also emits Server/Date here; this server never does.
        if self._response.add_required_headers:
            content = self._response.content
            if isinstance(content, (bytes, str)) and b"content-length" not in self._seen:
                self._head.append(
                    (b"Content-Length", str(len(isomorphic_encode(content))).encode())
                )
                self._seen.add(b"content-length")
        if b"content-length" not in self._seen:
            self._response.close_connection = True
        if self._status is None:
            self._status = (200, _REASONS.get(200, ""))
        code, message = self._status
        _send(self._response._rid, _FH_HEAD, code, message, self._head)
        self._complete = True
        return not self._aborted()

    def write_content(self, data):
        # Upstream: the first write_content finalizes the head, sizing
        # Content-Length from this chunk when the content is bytes/str.
        if self._status is None:
            self.write_status(200)
        if not self._complete:
            self._response.content = data
            self.end_headers()
        return self._write_raw(data)

    def write(self, data):
        return self._write_raw(data)

    def _write_raw(self, data):
        if isinstance(data, str):
            data = data.encode(self._response.encoding)
        data = bytes(data)
        self.content_written = True
        if self._aborted():
            return False
        if not self._complete:
            # Upstream would emit body bytes with no head; choose the
            # sane behavior instead.
            self.end_headers()
        _send(self._response._rid, _FH_CHUNK, body=data)
        return not self._aborted()


class Response:
    def __init__(self, rid, request, aborted):
        self._rid = rid
        self._aborted = aborted
        self.request = request
        self.encoding = "utf8"
        self.add_required_headers = True
        self.send_body_for_head_request = False
        self.close_connection = False
        self._status = (200, None)
        self.headers = ResponseHeaders()
        self.content = []
        self.writer = ResponseWriter(self)

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError("status tuple must be (code, message)")
            code = int(value[0])
            message = value[1]
            if not isinstance(message, (bytes, str)):
                message = str(message)
            self._status = (code, message)
        else:
            self._status = (int(value), None)

    def write_status_headers(self):
        self.writer.write_status(*self._status)
        for name, value in self.headers.items():
            self.writer.write_header(name, value)
        self.writer.end_headers()

    def set_cookie(self, name, value, path="/", domain=None, max_age=None,
                   expires=None, samesite=None, secure=False, httponly=False,
                   comment=None):
        if value is None:
            value = b""
            max_age = 0
            expires = timedelta(days=-1)
        name = isomorphic_decode(name)
        value = isomorphic_decode(value)
        months = {i + 1: m for i, m in enumerate(
            ["jan", "feb", "mar", "apr", "may", "jun",
             "jul", "aug", "sep", "oct", "nov", "dec"])}
        if isinstance(expires, timedelta):
            expires = datetime.now(timezone.utc) + expires
        if expires is not None:
            expires = (expires.strftime("%d %%s %Y %H:%M:%S GMT") % months[expires.month])
        if max_age is not None:
            if hasattr(max_age, "total_seconds"):
                max_age = int(max_age.total_seconds())
            max_age = "%.0d" % max_age

        m = Morsel()

        def maybe_set(key, v):
            if v is not None and v is not False:
                m[key] = v

        m.set(name, value, value)
        maybe_set("path", path)
        maybe_set("domain", domain)
        maybe_set("comment", comment)
        maybe_set("expires", expires)
        maybe_set("max-age", max_age)
        maybe_set("secure", secure)
        maybe_set("httponly", httponly)
        maybe_set("samesite", samesite)
        self.headers.append("Set-Cookie", m.OutputString())

    def set_error(self, code, err=None):
        if err is None:
            message = ""
        elif isinstance(err, BaseException):
            message = traceback.format_exc()
        else:
            message = str(err)
        data = json.dumps({"error": {"code": code, "message": message}})
        self.status = code
        self.headers = ResponseHeaders()
        self.headers.append("Content-Type", "application/json")
        self.headers.append("Content-Length", str(len(data)))
        self.content = data


# --------------------------------------------------------------------------
# Handler dispatch (FunctionHandler + PythonScriptHandler semantics)
# --------------------------------------------------------------------------

def _coerce_content(content):
    if content is None:
        return b""
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode("utf8")
    if isinstance(content, (list, tuple)):
        return b"".join(_coerce_content(c) for c in content)
    if hasattr(content, "read"):
        return content.read()
    raise ValueError("Unsupported response content type %r" % type(content))


_ABORTS = {}
_ABORTS_LOCK = threading.Lock()


def _run_request(msg):
    rid = msg["id"]
    aborted = threading.Event()
    with _ABORTS_LOCK:
        _ABORTS[rid] = aborted
    try:
        script = msg["script"]
        headers = [(isomorphic_encode(k), isomorphic_encode(v)) for k, v in msg["headers"]]
        body = base64.b64decode(msg["body_b64"]) if msg.get("body_b64") else b""
        peer = msg.get("peer", "")
        host, _, port = peer.partition(":") if isinstance(peer, str) else ("", "", "")
        request = Request(
            msg["method"], msg["target"], msg["url"], headers, body,
            msg["doc_root"], (host, port),
        )
        response = Response(rid, request, aborted)

        failed = False
        pipes = False
        try:
            with open(script, "rb") as f:
                source = f.read()
        except OSError:
            response.set_error(404, "No such handler")
            failed = True
        else:
            namespace = {"__file__": script, "__name__": "__wpt_handler__"}
            try:
                exec(compile(source, script, "exec"), namespace)
            except Exception:
                response.set_error(500, traceback.format_exc())
                failed = True
            else:
                main = namespace.get("main")
                if main is None:
                    # Like upstream: .h2.py-style scripts with only
                    # handle_headers/handle_data have no main on H1.
                    response.set_error(500, "No main function in script %s" % script)
                    failed = True
                else:
                    try:
                        rv = main(request, response)
                    except HTTPException as e:
                        response.set_error(e.code, e.message)
                        failed = True
                    except Exception:
                        response.set_error(500, traceback.format_exc())
                        failed = True
                    else:
                        pipes = False
                        head_sent = (response.writer._complete
                                     or response.writer.content_written)
                        if not head_sent:
                            # FunctionHandler return-value mapping.
                            # Upstream runs ?pipe= only when rv is not None.
                            pipes = rv is not None
                            if rv is not None:
                                if isinstance(rv, tuple):
                                    if len(rv) == 3:
                                        status, hdrs, content = rv
                                        response.status = status
                                    elif len(rv) == 2:
                                        hdrs, content = rv
                                    else:
                                        raise ValueError("bad handler tuple")
                                    response.headers.update(hdrs)
                                else:
                                    content = rv
                                response.content = content
        head_sent = response.writer._complete or response.writer.content_written
        code, message = response.status
        if message is None:
            message = _REASONS.get(code, "")
        if head_sent:
            _send(rid, _FH_END)
        else:
            _send(rid, _FH_RESPONSE, code, message,
                  list(response.headers.items()),
                  _coerce_content(response.content),
                  flags=1 if pipes else 0)
    except Exception:
        # Last resort: never leave the Rust side hanging.
        try:
            data = json.dumps({"error": {"code": 500, "message": traceback.format_exc()}})
            _send(rid, _FH_RESPONSE, 500, "Internal Server Error",
                  [(b"Content-Type", b"application/json")], data.encode())
        except Exception:
            pass
    finally:
        with _ABORTS_LOCK:
            _ABORTS.pop(rid, None)


def _main_loop():
    # The real stdin pipe, captured before stdout/stdin were rebound:
    # handlers must never see the request stream.
    data = _stdin
    # Sibling imports (none in fetch/xhr, but cheap) resolve after stdlib.
    while True:
        line = data.readline()
        if not line:
            break
        try:
            msg = json.loads(line)
        except Exception:
            continue
        if not isinstance(msg, dict) or "id" not in msg:
            continue
        if msg.get("abort"):
            with _ABORTS_LOCK:
                ev = _ABORTS.get(msg["id"])
            if ev is not None:
                ev.set()
            continue
        for p in (msg.get("doc_root", ""), os.path.dirname(msg.get("script", ""))):
            if p and p not in sys.path:
                sys.path.append(p)
        t = threading.Thread(target=_run_request, args=(msg,), daemon=True)
        t.start()


if __name__ == "__main__":
    _main_loop()
"###;

/// Subdomain aliases bound by [`FixtureServer::serve_dir`] alongside
/// the primary `127.0.0.1` (see [`alias_ip`]).
const SUBDOMAIN_ALIAS_IPS: &[&str] = &[
    "127.0.0.2",
    "127.0.0.3",
    "127.0.0.4",
    "127.0.0.5",
    "127.0.0.6",
];

/// Loopback IP standing in for a wptserve subdomain host: `host_set`
/// is `""` (the primary host) or `"alt"`, `subdomain` is `""`,
/// `"www"`, `"www1"` or `"www2"`. `None` for anything else (the
/// template then 500s, like upstream's `KeyError` on an unknown
/// domain).
///
/// Upstream spells these `www.127.0.0.1`; here they are additional
/// loopback IP literals because sandbox DNS resolves `*.localhost`
/// and `web-platform.test` to non-loopback egress IPs and
/// `/etc/hosts` is read-only, while IP literals need no DNS at all.
/// Tests only ever see these values through `{{…}}` substitution, so
/// the spelling is self-consistent.
fn alias_ip(host_set: &str, subdomain: &str) -> Option<&'static str> {
    match (host_set, subdomain) {
        ("", "") => Some("127.0.0.1"),
        ("", "www") => Some("127.0.0.2"),
        ("", "www1") => Some("127.0.0.3"),
        ("", "www2") => Some("127.0.0.4"),
        ("alt", "") => Some("127.0.0.5"),
        ("alt", "www") => Some("127.0.0.6"),
        _ => None,
    }
}

/// Standard-alphabet base64 encoder (for daemon request bodies).
fn b64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    let (chunks, rem) = data.as_chunks::<3>();
    for c in chunks {
        let n = (u32::from(c[0]) << 16) | (u32::from(c[1]) << 8) | u32::from(c[2]);
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(ALPHABET[(n >> 6) as usize & 63] as char);
        out.push(ALPHABET[n as usize & 63] as char);
    }
    if !rem.is_empty() {
        let mut n = 0u32;
        for (i, &b) in rem.iter().enumerate() {
            n |= u32::from(b) << (16 - 8 * i);
        }
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        if rem.len() == 2 {
            out.push(ALPHABET[(n >> 6) as usize & 63] as char);
        } else {
            out.push('=');
        }
        out.push('=');
    }
    out
}

/// One frame from the `.py` daemon (wire format documented on
/// [`PY_WPT_SHIM`]).
#[derive(Debug)]
enum PyFrame {
    Response {
        id: u64,
        /// Whether the Rust side should apply `?pipe=`: true only when
        /// the handler returned a value without streaming, mirroring
        /// upstream `wrap_pipeline` (which runs only for `rv is not
        /// None`).
        apply_pipes: bool,
        status: u16,
        reason: Vec<u8>,
        headers: Vec<(Vec<u8>, Vec<u8>)>,
        body: Vec<u8>,
    },
    Head {
        id: u64,
        status: u16,
        reason: Vec<u8>,
        headers: Vec<(Vec<u8>, Vec<u8>)>,
    },
    Chunk {
        id: u64,
        data: Vec<u8>,
    },
    End {
        id: u64,
    },
}

impl PyFrame {
    fn id(&self) -> u64 {
        match self {
            PyFrame::Response { id, .. }
            | PyFrame::Head { id, .. }
            | PyFrame::Chunk { id, .. }
            | PyFrame::End { id } => *id,
        }
    }
}

/// Caps on one daemon frame: the daemon runs test-checkout code, so a
/// buggy handler must not OOM the driver with a lying length prefix.
const MAX_PY_FRAME_HEADERS: usize = 1024;
const MAX_PY_HEADER_BYTES: usize = 256 * 1024;
const MAX_PY_FRAME_BODY_BYTES: usize = 256 * 1024 * 1024;

/// Little-endian frame field reader over the daemon's stdout.
struct PyFrameReader<'a, R: BufRead> {
    reader: &'a mut R,
}

impl<R: BufRead> PyFrameReader<'_, R> {
    fn invalid(&self) -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "bad py-daemon frame")
    }

    fn u16le(&mut self) -> std::io::Result<u16> {
        let mut b = [0u8; 2];
        self.reader.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    fn u32le(&mut self) -> std::io::Result<u32> {
        let mut b = [0u8; 4];
        self.reader.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn bytes(&mut self, n: usize) -> std::io::Result<Vec<u8>> {
        let mut b = vec![0u8; n];
        self.reader.read_exact(&mut b)?;
        Ok(b)
    }
}

fn read_py_frame(reader: &mut impl BufRead) -> std::io::Result<PyFrame> {
    let mut r = PyFrameReader { reader };
    let mut head = [0u8; 9];
    r.reader.read_exact(&mut head)?;
    let id = u64::from_le_bytes(head[0..8].try_into().map_err(|_| r.invalid())?);
    let kind = head[8];
    // Status, reason, and headers ride on Response/Head frames.
    let (apply_pipes, status, reason, headers) = if kind == 0 || kind == 1 {
        let apply_pipes = if kind == 0 {
            let mut b = [0u8; 1];
            r.reader.read_exact(&mut b)?;
            b[0] & 1 != 0
        } else {
            false
        };
        let status = r.u16le()?;
        let rlen = r.u16le()? as usize;
        if rlen > MAX_PY_HEADER_BYTES {
            return Err(r.invalid());
        }
        let reason = r.bytes(rlen)?;
        let nh = r.u32le()? as usize;
        if nh > MAX_PY_FRAME_HEADERS {
            return Err(r.invalid());
        }
        let mut headers = Vec::with_capacity(nh.min(16));
        for _ in 0..nh {
            let nl = r.u32le()? as usize;
            if nl > MAX_PY_HEADER_BYTES {
                return Err(r.invalid());
            }
            let name = r.bytes(nl)?;
            let vl = r.u32le()? as usize;
            if vl > MAX_PY_HEADER_BYTES {
                return Err(r.invalid());
            }
            let value = r.bytes(vl)?;
            headers.push((name, value));
        }
        (apply_pipes, status, reason, headers)
    } else {
        (false, 0, Vec::new(), Vec::new())
    };
    // Bodies ride on Response/Chunk frames.
    let body = if kind == 0 || kind == 2 {
        let mut b = [0u8; 8];
        r.reader.read_exact(&mut b)?;
        let bl = u64::from_le_bytes(b) as usize;
        if bl > MAX_PY_FRAME_BODY_BYTES {
            return Err(r.invalid());
        }
        r.bytes(bl)?
    } else {
        Vec::new()
    };
    match kind {
        0 => Ok(PyFrame::Response {
            id,
            apply_pipes,
            status,
            reason,
            headers,
            body,
        }),
        1 => Ok(PyFrame::Head {
            id,
            status,
            reason,
            headers,
        }),
        2 => Ok(PyFrame::Chunk { id, data: body }),
        3 => Ok(PyFrame::End { id }),
        _ => Err(r.invalid()),
    }
}

/// How long [`serve_py`] waits for the daemon's first frame before
/// giving up with a 500. No handler in the target suites sleeps this
/// long before answering.
const PY_FIRST_BYTE_TIMEOUT: Duration = Duration::from_secs(30);

/// Minimum seconds between `.py` dispatch-failure log lines: a missing
/// `python3` (or a dead daemon) fails every `.py` request, and must not
/// flood stderr once per request.
const PY_DISPATCH_LOG_INTERVAL_SECS: u64 = 60;
static LAST_PY_DISPATCH_LOG_SECS: AtomicU64 = AtomicU64::new(0);

/// Log a `.py` dispatch failure, at most once per
/// [`PY_DISPATCH_LOG_INTERVAL_SECS`]. Returns whether this call logged.
fn note_py_dispatch_failed(msg: &str) -> bool {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let last = LAST_PY_DISPATCH_LOG_SECS.load(Ordering::Relaxed);
    if now.wrapping_sub(last) < PY_DISPATCH_LOG_INTERVAL_SECS {
        return false;
    }
    if LAST_PY_DISPATCH_LOG_SECS
        .compare_exchange_weak(last, now, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return false;
    }
    eprintln!(
        "wpt fixture server: .py dispatch failed: {msg} \
         (further failures are logged at most once a minute)"
    );
    true
}

/// The lazily spawned `python3` `.py`-handler daemon, shared by every
/// connection of one [`FixtureServer`].
struct PyDaemonHandle {
    state: Mutex<PyDaemonState>,
}

enum PyDaemonState {
    Unstarted,
    Running(Arc<PyDaemon>),
    /// Spawn failed, or the daemon died mid-run: `.py` requests fail
    /// instead of hanging. `respawnable` is false when `python3` itself
    /// is missing (retrying would burn time on every request) and true
    /// after a mid-run death, so one transient EPIPE does not poison
    /// later requests: [`PyDaemonHandle::daemon`] retries with backoff.
    Failed(String, bool),
}

struct PyDaemon {
    /// Serializes request dispatch: id assignment, pending
    /// registration, and the stdin write happen under this one lock so
    /// JSON lines can never interleave on the pipe.
    req_lock: Mutex<()>,
    stdin: Mutex<ChildStdin>,
    pending: Mutex<HashMap<u64, mpsc::Sender<PyFrame>>>,
    next_id: Mutex<u64>,
    /// Killed on drop so no `python3` outlives the server.
    child: Mutex<Child>,
    /// Back-pointer for the reader thread: on stdout EOF it marks the
    /// handle failed promptly instead of waiting for the next stdin
    /// write to discover the death.
    handle: Weak<PyDaemonHandle>,
}

impl Drop for PyDaemon {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
        }
    }
}

/// SIGKILL the daemon child and reap it on a detached thread, so no
/// `python3` survives a failure and no zombie lingers. `wait()` blocks
/// until the child exits, so it never runs on a request path; the kill
/// guarantees it returns.
fn reap_daemon(d: &Arc<PyDaemon>) {
    if let Ok(mut child) = d.child.lock() {
        let _ = child.kill();
    }
    let d = Arc::clone(d);
    let _ = thread::Builder::new()
        .name("wpt-py-reaper".into())
        .spawn(move || {
            if let Ok(mut child) = d.child.lock() {
                let _ = child.wait();
            }
        });
}

impl Drop for PyDaemonHandle {
    fn drop(&mut self) {
        // The reader thread's `Arc<PyDaemon>` clone keeps the daemon
        // itself alive past this point, so dropping the handle must
        // kill the child explicitly: otherwise `python3` blocks on
        // stdin readline forever and leaks. The kill EOFs stdout, which
        // lets the reader thread exit.
        self.fail_permanent("server shut down".to_owned());
    }
}

/// How many extra spawn attempts [`PyDaemonHandle::daemon`] makes after
/// a mid-run daemon death before failing fast.
const PY_RESPAWN_RETRIES: u32 = 2;
/// Backoff between respawn attempts, multiplied by the attempt number.
const PY_RESPAWN_BACKOFF: Duration = Duration::from_millis(100);

impl PyDaemonHandle {
    fn new() -> PyDaemonHandle {
        PyDaemonHandle {
            state: Mutex::new(PyDaemonState::Unstarted),
        }
    }

    /// Mark the daemon failed: pending receivers see a disconnect
    /// (their requests become 500s). A later [`daemon`](Self::daemon)
    /// call retries the spawn with backoff before failing fast, so a
    /// transient mid-run death does not poison later requests.
    fn fail(&self, msg: String) {
        self.fail_inner(msg, true);
    }

    /// Like [`fail`](Self::fail), but permanent: no respawn is ever
    /// attempted (used when `python3` is missing, and on shutdown).
    fn fail_permanent(&self, msg: String) {
        self.fail_inner(msg, false);
    }

    fn fail_inner(&self, msg: String, respawnable: bool) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let PyDaemonState::Running(d) = &*state {
            d.pending.lock().unwrap_or_else(|e| e.into_inner()).clear();
            reap_daemon(d);
        }
        *state = PyDaemonState::Failed(msg, respawnable);
    }

    /// Mark the daemon failed, but only if it is still the running one
    /// (called by the reader thread on stdout EOF). Never overwrites an
    /// existing failure: a concurrent respawn or a permanent failure
    /// (shutdown, missing `python3`) wins.
    fn fail_if_running(&self, msg: String) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let PyDaemonState::Running(d) = &*state {
            d.pending.lock().unwrap_or_else(|e| e.into_inner()).clear();
            reap_daemon(d);
            *state = PyDaemonState::Failed(msg, true);
        }
    }

    /// The running daemon, spawning it on first use. A mid-run death
    /// retries the spawn with backoff ([`PY_RESPAWN_RETRIES`] attempts);
    /// a missing `python3` fails fast and permanently. Concurrent calls
    /// may spawn twice; the loser is dropped (and killed) once the
    /// winner is installed.
    fn daemon(self: &Arc<Self>) -> Result<Arc<PyDaemon>, String> {
        let respawnable = {
            let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            match &*state {
                PyDaemonState::Running(d) => return Ok(Arc::clone(d)),
                PyDaemonState::Failed(msg, respawnable) => {
                    if !respawnable {
                        return Err(msg.clone());
                    }
                    true
                }
                PyDaemonState::Unstarted => false,
            }
        };
        let attempts = if respawnable {
            1 + PY_RESPAWN_RETRIES
        } else {
            1
        };
        let mut last_err = String::new();
        for attempt in 0..attempts {
            if attempt > 0 {
                thread::sleep(PY_RESPAWN_BACKOFF * attempt);
            }
            match self.spawn_one() {
                Ok(d) => return Ok(d),
                Err(e) => last_err = e,
            }
        }
        // Still dead after bounded retries: fail fast (and permanently)
        // instead of retry-storming on every request.
        let msg = last_err.clone();
        self.fail_permanent(last_err);
        Err(msg)
    }

    /// Spawn one daemon, start its reader, and install it as `Running`,
    /// returning the winner if a concurrent call installed one first.
    fn spawn_one(self: &Arc<Self>) -> Result<Arc<PyDaemon>, String> {
        let daemon = match spawn_py_daemon() {
            Ok(mut d) => {
                d.handle = Arc::downgrade(self);
                Arc::new(d)
            }
            Err(e) => {
                let msg = format!("python3 daemon unavailable: {e}");
                eprintln!("wpt fixture server: {msg}");
                return Err(msg);
            }
        };
        // The reader routes frames to per-request channels by id; on
        // EOF (or a corrupt frame) it clears every pending sender so
        // receivers fail instead of hanging, marks the handle failed so
        // the death is visible immediately, and reaps the child.
        let reader = Arc::clone(&daemon);
        if let Err(e) = thread::Builder::new()
            .name("wpt-py-reader".into())
            .spawn(move || py_reader(reader))
        {
            eprintln!("wpt fixture server: failed to spawn py reader: {e}");
            return Err("python3 daemon reader failed to spawn".to_owned());
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match &*state {
            PyDaemonState::Running(d) => Ok(Arc::clone(d)),
            _ => {
                *state = PyDaemonState::Running(Arc::clone(&daemon));
                Ok(daemon)
            }
        }
    }
}

/// Spawn the `.py` handler daemon: `python3 -u -c <shim>`, one JSON
/// request line per stdin line, binary frames on stdout. Handler
/// tracebacks go to our stderr — the interesting half of debugging a
/// `.py` handler. A handler's stray `print()` lands on stderr too: the
/// shim rebinds its stdout to stderr because stdout carries the binary
/// frame protocol, and one stray line would corrupt it permanently.
fn spawn_py_daemon() -> std::io::Result<PyDaemon> {
    let mut child = Command::new("python3")
        .arg("-u")
        .arg("-c")
        .arg(PY_WPT_SHIM)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| std::io::Error::other("python3 daemon: no stdin"))?;
    if child.stdout.is_none() {
        return Err(std::io::Error::other("python3 daemon: no stdout"));
    }
    Ok(PyDaemon {
        req_lock: Mutex::new(()),
        stdin: Mutex::new(stdin),
        pending: Mutex::new(HashMap::new()),
        next_id: Mutex::new(1),
        child: Mutex::new(child),
        // Filled in by `PyDaemonHandle::spawn_one`, which owns the `Arc`.
        handle: Weak::new(),
    })
}

/// A frame that answers its request completely: after one of these,
/// nothing more may arrive for the id, so the `pending` entry is
/// forgotten. (`Head` is not terminal — chunks follow; `Chunk` is not
/// terminal — `End` follows.)
fn py_frame_is_terminal(frame: &PyFrame) -> bool {
    matches!(frame, PyFrame::End { .. } | PyFrame::Response { .. })
}

/// Frame router: read frames from the daemon's stdout and hand each to
/// its request's channel. On EOF (or a corrupt frame) every pending
/// receiver is released with a disconnect, the handle is marked failed
/// so the death is visible without waiting for the next stdin write to
/// fail, and the child is reaped.
fn py_reader(daemon: Arc<PyDaemon>) {
    let stdout = {
        let mut child = daemon.child.lock().unwrap_or_else(|e| e.into_inner());
        child.stdout.take()
    };
    let Some(stdout) = stdout else { return };
    let mut reader = BufReader::new(stdout);
    while let Ok(frame) = read_py_frame(&mut reader) {
        let id = frame.id();
        let terminal = py_frame_is_terminal(&frame);
        let mut pending = daemon.pending.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(tx) = pending.get(&id)
            && (tx.send(frame).is_err() || terminal)
        {
            pending.remove(&id);
        }
    }
    daemon
        .pending
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
    if let Some(handle) = daemon.handle.upgrade() {
        handle.fail_if_running("python3 daemon died".to_owned());
    } else {
        // The handle is gone (server shutting down): make sure the
        // child cannot outlive the reader.
        reap_daemon(&daemon);
    }
}

/// One `.py` handler invocation's inputs.
struct PyHandlerRequest<'a> {
    script: &'a Path,
    method: &'a str,
    /// Origin-form target (path + query), for `request.target`.
    target: &'a str,
    /// Full URL, for `request.url` / `request.url_parts`.
    url: String,
    headers: &'a [(String, String)],
    body: &'a [u8],
    doc_root: &'a Path,
    peer: &'a str,
}

impl PyHandlerRequest<'_> {
    /// The daemon's JSON request line, without the trailing newline.
    fn json_line(&self, id: u64) -> String {
        let mut out = String::with_capacity(256 + self.body.len() * 2);
        out.push_str(&format!("{{\"id\":{id},"));
        out.push_str(&format!(
            "\"script\":{},",
            json_escape(&self.script.to_string_lossy())
        ));
        out.push_str(&format!("\"method\":{},", json_escape(self.method)));
        out.push_str(&format!("\"target\":{},", json_escape(self.target)));
        out.push_str(&format!("\"url\":{},", json_escape(&self.url)));
        out.push_str(&format!("\"peer\":{},", json_escape(self.peer)));
        out.push_str("\"headers\":[");
        for (i, (name, value)) in self.headers.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!("[{},{}]", json_escape(name), json_escape(value)));
        }
        out.push_str("],");
        out.push_str(&format!("\"body_b64\":\"{}\",", b64_encode(self.body)));
        out.push_str(&format!(
            "\"doc_root\":{}",
            json_escape(&self.doc_root.to_string_lossy())
        ));
        out.push('}');
        out
    }
}

/// Dispatch one `.py` handler invocation. Returns the request id (for
/// [`py_abort`]) and the frame receiver.
fn py_call(
    handle: &Arc<PyDaemonHandle>,
    req: &PyHandlerRequest<'_>,
) -> Result<(u64, mpsc::Receiver<PyFrame>), String> {
    let daemon = handle.daemon()?;
    let _dispatch = daemon.req_lock.lock().unwrap_or_else(|e| e.into_inner());
    let id = {
        let mut next_id = daemon.next_id.lock().unwrap_or_else(|e| e.into_inner());
        let id = *next_id;
        *next_id = next_id.wrapping_add(1);
        id
    };
    let (tx, rx) = mpsc::channel();
    daemon
        .pending
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(id, tx);
    let line = req.json_line(id);
    let write_ok = {
        let mut stdin = daemon.stdin.lock().unwrap_or_else(|e| e.into_inner());
        stdin
            .write_all(line.as_bytes())
            .and_then(|()| stdin.write_all(b"\n"))
            .is_ok()
    };
    drop(_dispatch);
    if !write_ok {
        daemon
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&id);
        handle.fail("python3 daemon stdin write failed".to_owned());
        return Err("python3 daemon died".to_owned());
    }
    Ok((id, rx))
}

/// Ask the daemon to stop a streaming response, and forget the
/// request's pending entry. Best effort: never spawns or respawns a
/// daemon just to deliver an abort. Lock order is req_lock →
/// pending → stdin, matching [`py_call`].
fn py_abort(handle: &Arc<PyDaemonHandle>, id: u64) {
    let daemon = {
        let state = handle.state.lock().unwrap_or_else(|e| e.into_inner());
        match &*state {
            PyDaemonState::Running(d) => Arc::clone(d),
            _ => return,
        }
    };
    let _dispatch = daemon.req_lock.lock().unwrap_or_else(|e| e.into_inner());
    daemon
        .pending
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&id);
    let mut stdin = daemon.stdin.lock().unwrap_or_else(|e| e.into_inner());
    let _ = stdin.write_all(format!("{{\"id\":{id},\"abort\":true}}\n").as_bytes());
}

/// Serve a `.py` request by executing the handler in the python3
/// daemon. `decoded` is the traversal-checked decoded path (ends in
/// `.py`), `query` the raw query string.
fn serve_py(
    req: &HttpRequest,
    decoded: &str,
    query: Option<&str>,
    doc_root: &Path,
    ctx: &ServerCtx,
) -> RouteOutcome {
    let failed =
        |status: u16| RouteOutcome::Response(Response::empty(status, status_reason(status)));
    // Resolve the script with the same symlink discipline as
    // `read_doc_file`: a missing handler is a 404, like upstream.
    let script = match doc_root
        .join(decoded.trim_start_matches('/'))
        .canonicalize()
    {
        Ok(p) if p.starts_with(doc_root) && p.is_file() => p,
        Ok(_) => return failed(404),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return failed(404),
        Err(_) => return failed(500),
    };
    // `request.url`, as upstream builds it: scheme, the Host header (or
    // the server address), the port, and the origin-form target.
    let target = strip_absolute_form(&req.target);
    let host = header_value(&req.headers, "host")
        .and_then(|h| h.split(':').next())
        .filter(|h| !h.is_empty())
        .unwrap_or("127.0.0.1");
    let py_req = PyHandlerRequest {
        script: &script,
        method: &req.method,
        target,
        url: format!("http://{host}:{}{target}", ctx.port),
        headers: &req.headers,
        body: &req.body,
        doc_root,
        peer: &req.peer,
    };
    let (id, rx) = match py_call(&ctx.py, &py_req) {
        Ok(v) => v,
        Err(e) => {
            // A missing `python3` fails every `.py` request: rate-limit
            // the log line instead of flooding stderr per request.
            note_py_dispatch_failed(&e);
            return failed(500);
        }
    };
    let first = match rx.recv_timeout(PY_FIRST_BYTE_TIMEOUT) {
        Ok(frame) => frame,
        Err(_) => {
            // The handler never answered: stop it and 500. The
            // pending entry is forgotten so a late answer goes
            // nowhere.
            py_abort(&ctx.py, id);
            return failed(500);
        }
    };
    match first {
        PyFrame::Response {
            apply_pipes,
            status,
            reason,
            headers,
            body,
            ..
        } => {
            let mut response = match Response::from_py(status, &reason, headers, body) {
                // A CR/LF smuggled into the reason or headers is a
                // response-splitting attempt: 500, don't emit it.
                Ok(r) => r,
                Err(_) => return failed(500),
            };
            if apply_pipes {
                let directives = match parse_pipe_query(query) {
                    Ok(d) => d,
                    Err(s) => return failed(s),
                };
                response = apply_pipe(response, &directives);
            }
            RouteOutcome::Response(response)
        }
        PyFrame::Head {
            status,
            reason,
            headers,
            ..
        } => {
            // Same CR/LF bar as the buffered path: the streaming head
            // is written verbatim by `PyStream::drive`.
            if Response::py_head_parts_valid(&reason, &headers).is_err() {
                py_abort(&ctx.py, id);
                return failed(500);
            }
            RouteOutcome::PyStream(PyStream {
                status,
                reason,
                headers,
                rx,
                py: Arc::clone(&ctx.py),
                id,
            })
        }
        // A stream frame first is a daemon protocol violation.
        PyFrame::Chunk { .. } | PyFrame::End { .. } => {
            py_abort(&ctx.py, id);
            failed(500)
        }
    }
}

/// A `.py` handler that sent its head via `response.write_status_headers`
/// (or `writer.write_header` + `end_headers`): body chunks stream
/// straight to the connection; the daemon, not a [`Response`], owns
/// the framing.
struct PyStream {
    status: u16,
    reason: Vec<u8>,
    headers: Vec<(Vec<u8>, Vec<u8>)>,
    rx: mpsc::Receiver<PyFrame>,
    py: Arc<PyDaemonHandle>,
    id: u64,
}

/// How long [`PyStream::drive`] waits for one body chunk after the
/// head before aborting the handler: a wedged handler must not pin a
/// connection thread forever.
const PY_CHUNK_TIMEOUT: Duration = Duration::from_secs(30);

impl PyStream {
    /// Write the head, then forward chunks until `End`, a daemon
    /// protocol violation, or a failed client write (which aborts the
    /// handler, mirroring upstream's write-returns-False path).
    fn drive(&self, stream: &mut TcpStream) -> std::io::Result<()> {
        self.drive_with_timeout(stream, PY_CHUNK_TIMEOUT)
    }

    /// [`drive`](Self::drive) with an explicit chunk timeout (the
    /// parameter exists so tests can use a short one).
    fn drive_with_timeout(
        &self,
        stream: &mut TcpStream,
        chunk_timeout: Duration,
    ) -> std::io::Result<()> {
        let mut head = format!(
            "HTTP/1.1 {} {}\r\n",
            self.status,
            String::from_utf8_lossy(&self.reason)
        )
        .into_bytes();
        for (name, value) in &self.headers {
            head.extend_from_slice(name);
            head.extend_from_slice(b": ");
            head.extend_from_slice(value);
            head.extend_from_slice(b"\r\n");
        }
        // The daemon's head is authoritative (it mirrors upstream's
        // `end_headers`); only add `connection: close` when the handler
        // didn't set `Connection` itself.
        if !self
            .headers
            .iter()
            .any(|(n, _)| n.eq_ignore_ascii_case(b"connection"))
        {
            head.extend_from_slice(b"connection: close\r\n");
        }
        head.extend_from_slice(b"\r\n");
        stream.write_all(&head)?;
        loop {
            match self.rx.recv_timeout(chunk_timeout) {
                Ok(PyFrame::Chunk { data, .. }) => {
                    if stream.write_all(&data).is_err() {
                        py_abort(&self.py, self.id);
                        return Ok(());
                    }
                }
                Ok(PyFrame::End { .. }) | Err(RecvTimeoutError::Disconnected) => return Ok(()),
                Err(RecvTimeoutError::Timeout) => {
                    // The handler wedged mid-stream: stop it and close
                    // the connection rather than pinning this thread
                    // forever.
                    py_abort(&self.py, self.id);
                    return Ok(());
                }
                // Protocol violation: stop rather than emit garbage.
                Ok(_) => {
                    py_abort(&self.py, self.id);
                    return Ok(());
                }
            }
        }
    }
}

// ============================================================
// `{{…}}` template substitution (wptserve `sub()` pipe).
// ============================================================

/// Context for `{{…}}` substitution in `.sub.` files.
struct SubCtx<'a> {
    port: u16,
    headers: &'a [(String, String)],
    query: Option<&'a str>,
    /// Decoded request path, for `{{location}}` and relative `fs_path()`.
    url_path: &'a str,
    doc_root: &'a Path,
}

/// Why a `{{…}}` substitution failed. Any of these is a 500, like
/// upstream wptserve (an unhandled template exception → HTTP 500).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubError {
    BadToken,
    Undefined,
    BadIndex,
    BadValue,
}

#[derive(Debug, Clone)]
enum SubToken {
    /// `$name:` — assignment target (colon stripped, `$` kept).
    Var(String),
    /// `[$]?word`.
    Ident(String),
    Index(SubIndex),
    /// `(a, b, …)`.
    Args(Vec<String>),
}

#[derive(Debug, Clone)]
enum SubIndex {
    Int(i64),
    Key(String),
}

fn is_sub_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Tokenize one `{{…}}` body, mirroring upstream's `re.Scanner`
/// (`\$\w+:`, `\$?\w+`, `\[[^\]]*\]`, `\([^)]*\)`, tried in that
/// order). A trailing unscannable remainder is dropped, as upstream
/// does.
fn tokenize_sub(s: &[u8]) -> Vec<SubToken> {
    let mut toks = Vec::new();
    let mut i = 0;
    'scan: while i < s.len() {
        let rest = &s[i..];
        // var: `$\w+:`
        if rest[0] == b'$' {
            let mut j = 1;
            while j < rest.len() && is_sub_word(rest[j]) {
                j += 1;
            }
            if j > 1 && j < rest.len() && rest[j] == b':' {
                toks.push(SubToken::Var(
                    String::from_utf8_lossy(&rest[..j]).into_owned(),
                ));
                i += j + 1;
                continue 'scan;
            }
        }
        // ident: `\$?\w+`
        {
            let mut j = usize::from(rest[0] == b'$');
            while j < rest.len() && is_sub_word(rest[j]) {
                j += 1;
            }
            if j > usize::from(rest[0] == b'$') {
                toks.push(SubToken::Ident(
                    String::from_utf8_lossy(&rest[..j]).into_owned(),
                ));
                i += j;
                continue 'scan;
            }
        }
        // index: `\[[^\]]*\]`
        if rest[0] == b'['
            && let Some(end) = rest.iter().position(|&b| b == b']')
        {
            let text = String::from_utf8_lossy(&rest[1..end]).into_owned();
            let idx = text
                .parse::<i64>()
                .map(SubIndex::Int)
                .unwrap_or(SubIndex::Key(text));
            toks.push(SubToken::Index(idx));
            i += end + 1;
            continue 'scan;
        }
        // arguments: `\([^)]*\)`
        if rest[0] == b'('
            && let Some(end) = rest.iter().position(|&b| b == b')')
        {
            let inner = String::from_utf8_lossy(&rest[1..end]);
            let args = if inner.is_empty() {
                Vec::new()
            } else {
                inner.split(',').map(|a| a.trim().to_owned()).collect()
            };
            toks.push(SubToken::Args(args));
            i += end + 1;
            continue 'scan;
        }
        // Unscannable remainder: dropped, like upstream.
        break;
    }
    toks
}

#[derive(Debug, Clone)]
enum SubVal {
    Str(String),
    Int(i64),
    Map(Vec<(String, SubVal)>),
    List(Vec<SubVal>),
    /// `{{headers[…]}}`: the request headers.
    Headers,
    /// `{{GET[…]}}`: the query parameters (missing → `""`).
    Get,
    Func(SubFunc),
}

#[derive(Debug, Clone, Copy)]
enum SubFunc {
    Uuid,
    FileHash,
    FsPath,
    HeaderOrDefault,
}

fn eval_sub(
    inner: &[u8],
    ctx: &SubCtx<'_>,
    vars: &mut HashMap<String, SubVal>,
) -> Result<SubVal, SubError> {
    let mut toks = tokenize_sub(inner).into_iter();
    let (var, mut value) = match toks.next().ok_or(SubError::BadToken)? {
        SubToken::Var(name) => {
            let SubToken::Ident(ident) = toks.next().ok_or(SubError::BadToken)? else {
                return Err(SubError::BadToken);
            };
            (Some(name), resolve_sub_ident(&ident, ctx, vars)?)
        }
        SubToken::Ident(ident) => (None, resolve_sub_ident(&ident, ctx, vars)?),
        _ => return Err(SubError::BadToken),
    };
    for tok in toks {
        match tok {
            SubToken::Index(idx) => value = index_sub_val(value, idx, ctx)?,
            SubToken::Args(args) => value = call_sub_func(value, &args, ctx)?,
            _ => return Err(SubError::BadToken),
        }
    }
    // Upstream asserts the final value is int/bytes/str.
    match &value {
        SubVal::Str(_) | SubVal::Int(_) => {}
        _ => return Err(SubError::BadValue),
    }
    if let Some(name) = var {
        vars.insert(name, value.clone());
    }
    Ok(value)
}

fn resolve_sub_ident(
    name: &str,
    ctx: &SubCtx<'_>,
    vars: &HashMap<String, SubVal>,
) -> Result<SubVal, SubError> {
    if let Some(v) = vars.get(name) {
        return Ok(v.clone());
    }
    match name {
        "host" => Ok(SubVal::Str("127.0.0.1".to_owned())),
        "hosts" => Ok(SubVal::Map(vec![
            ("".to_owned(), domains_sub_val("")),
            ("alt".to_owned(), domains_sub_val("alt")),
        ])),
        "domains" => Ok(domains_sub_val("")),
        "ports" => {
            // One port serves every scheme here: there is no TLS or WS
            // listener, so every protocol's port list is the HTTP port.
            // Substitution stays total; fetching an https:// URL fails
            // at the TLS handshake, as it does everywhere else in this
            // server.
            let p = || SubVal::List(vec![SubVal::Int(i64::from(ctx.port))]);
            Ok(SubVal::Map(vec![
                ("http".to_owned(), p()),
                ("https".to_owned(), p()),
                ("http-local".to_owned(), p()),
                ("http-public".to_owned(), p()),
                ("https-local".to_owned(), p()),
                ("https-public".to_owned(), p()),
                ("ws".to_owned(), p()),
                ("wss".to_owned(), p()),
            ]))
        }
        "location" => {
            let host = request_host(ctx);
            Ok(SubVal::Map(vec![
                (
                    "server".to_owned(),
                    SubVal::Str(format!("http://{host}:{}", ctx.port)),
                ),
                ("scheme".to_owned(), SubVal::Str("http".to_owned())),
                (
                    "host".to_owned(),
                    SubVal::Str(format!("{host}:{}", ctx.port)),
                ),
                ("hostname".to_owned(), SubVal::Str(host)),
                ("port".to_owned(), SubVal::Int(i64::from(ctx.port))),
                ("path".to_owned(), SubVal::Str(ctx.url_path.to_owned())),
                ("pathname".to_owned(), SubVal::Str(ctx.url_path.to_owned())),
                (
                    "query".to_owned(),
                    SubVal::Str(match ctx.query {
                        Some(q) => format!("?{q}"),
                        None => "?".to_owned(),
                    }),
                ),
            ]))
        }
        "headers" => Ok(SubVal::Headers),
        "GET" => Ok(SubVal::Get),
        "uuid" => Ok(SubVal::Func(SubFunc::Uuid)),
        "file_hash" => Ok(SubVal::Func(SubFunc::FileHash)),
        "fs_path" => Ok(SubVal::Func(SubFunc::FsPath)),
        "header_or_default" => Ok(SubVal::Func(SubFunc::HeaderOrDefault)),
        "url_base" => Ok(SubVal::Str("/".to_owned())),
        _ => Err(SubError::Undefined),
    }
}

/// `{{domains}}` / one `{{hosts[…]}}` entry: subdomain → alias IP
/// (see [`alias_ip`]).
fn domains_sub_val(host_set: &str) -> SubVal {
    let mut map = Vec::new();
    for sub in ["", "www", "www1", "www2"] {
        if let Some(ip) = alias_ip(host_set, sub) {
            map.push((sub.to_owned(), SubVal::Str(ip.to_owned())));
        }
    }
    SubVal::Map(map)
}

/// Request host without the port, for `{{location}}` (the `Host`
/// header, or the primary address when absent).
fn request_host(ctx: &SubCtx<'_>) -> String {
    header_value(ctx.headers, "host")
        .and_then(|h| h.split(':').next())
        .filter(|h| !h.is_empty())
        .unwrap_or("127.0.0.1")
        .to_owned()
}

fn index_sub_val(value: SubVal, idx: SubIndex, ctx: &SubCtx<'_>) -> Result<SubVal, SubError> {
    match (value, idx) {
        (SubVal::Map(m), SubIndex::Key(k)) => m
            .into_iter()
            .find(|(key, _)| key == &k)
            .map(|(_, v)| v)
            .ok_or(SubError::Undefined),
        (SubVal::List(l), SubIndex::Int(n)) => {
            let n: usize = n.try_into().map_err(|_| SubError::BadIndex)?;
            l.into_iter().nth(n).ok_or(SubError::BadIndex)
        }
        (SubVal::Headers, SubIndex::Key(k)) => header_value(ctx.headers, &k)
            .map(|v| SubVal::Str(v.to_owned()))
            .ok_or(SubError::Undefined),
        // Upstream: iso-8859-1 parse, then a strict UTF-8 decode of the
        // value; a missing key is `""` (FirstWrapper).
        (SubVal::Get, SubIndex::Key(k)) => match ctx.query.and_then(|q| query_param(q, &k)) {
            None => Ok(SubVal::Str(String::new())),
            Some(v) => {
                let bytes = percent_decode_bytes(v).ok_or(SubError::BadValue)?;
                String::from_utf8(bytes)
                    .map(SubVal::Str)
                    .map_err(|_| SubError::BadValue)
            }
        },
        _ => Err(SubError::BadIndex),
    }
}

fn call_sub_func(value: SubVal, args: &[String], ctx: &SubCtx<'_>) -> Result<SubVal, SubError> {
    match value {
        SubVal::Func(SubFunc::Uuid) => {
            if !args.is_empty() {
                return Err(SubError::BadValue);
            }
            Ok(SubVal::Str(new_uuid()))
        }
        SubVal::Func(SubFunc::HeaderOrDefault) => {
            if args.len() != 2 {
                return Err(SubError::BadValue);
            }
            Ok(SubVal::Str(
                header_value(ctx.headers, &args[0])
                    .unwrap_or(&args[1])
                    .to_owned(),
            ))
        }
        SubVal::Func(SubFunc::FsPath) => {
            if args.len() != 1 {
                return Err(SubError::BadValue);
            }
            fs_path_sub(ctx, &args[0]).map(SubVal::Str)
        }
        // No crypto in the std-only fixture server.
        SubVal::Func(SubFunc::FileHash) => Err(SubError::BadValue),
        _ => Err(SubError::BadValue),
    }
}

/// Pseudo-random UUID v4 for `{{uuid()}}` (stash keys). Unique, not
/// unpredictable: xorshift64* seeded from the clock, pid, and a
/// counter — uniqueness per call is what matters. Never use the output
/// where an attacker must not guess the value (no secret tokens, no
/// nonces); it is not cryptographic randomness.
fn new_uuid() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut x = nanos
        .wrapping_add((std::process::id() as u64).wrapping_mul(0x9E3779B97F4A7C15))
        .wrapping_add(
            COUNTER
                .fetch_add(1, Ordering::SeqCst)
                .wrapping_mul(0xBF58476D1CE4E5B9),
        );
    let mut next = || {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        x = x.wrapping_mul(0x2545F4914F6CDD1D);
        x
    };
    let hi = next();
    let lo = next();
    // Version 4 and RFC 4122 variant bits.
    let b = [
        (hi >> 56) as u8,
        (hi >> 48) as u8,
        (hi >> 40) as u8,
        (hi >> 32) as u8,
        (hi >> 24) as u8,
        (hi >> 16) as u8,
        ((hi >> 8) as u8 & 0x0f) | 0x40,
        hi as u8,
        ((lo >> 56) as u8 & 0x3f) | 0x80,
        (lo >> 48) as u8,
        (lo >> 40) as u8,
        (lo >> 32) as u8,
        (lo >> 24) as u8,
        (lo >> 16) as u8,
        (lo >> 8) as u8,
        lo as u8,
    ];
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-\
         {:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0],
        b[1],
        b[2],
        b[3],
        b[4],
        b[5],
        b[6],
        b[7],
        b[8],
        b[9],
        b[10],
        b[11],
        b[12],
        b[13],
        b[14],
        b[15]
    )
}

/// `{{fs_path(path)}}`: absolute path of a file under the doc root, as
/// upstream (relative paths resolve against the request's directory).
/// Escaping the root is a 500.
fn fs_path_sub(ctx: &SubCtx<'_>, path: &str) -> Result<String, SubError> {
    let rel = if let Some(p) = path.strip_prefix('/') {
        p.to_owned()
    } else {
        let dir = ctx.url_path.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
        format!("{dir}/{path}")
    };
    let abs = ctx.doc_root.join(rel);
    // Lexical normalization; symlinks are the caller's business, as
    // upstream.
    let mut normal = PathBuf::new();
    for comp in abs.components() {
        match comp {
            std::path::Component::ParentDir => {
                if !normal.pop() {
                    return Err(SubError::BadValue);
                }
            }
            std::path::Component::CurDir => {}
            _ => normal.push(comp.as_os_str()),
        }
    }
    if !normal.starts_with(ctx.doc_root) {
        return Err(SubError::BadValue);
    }
    normal.to_str().map(str::to_owned).ok_or(SubError::BadValue)
}

/// wptserve `sub()` template substitution over `content` (bytes):
/// every `{{…}}` span is replaced; `escape_html` selects the HTML
/// escaping upstream uses for markup files. A substitution error is a
/// 500 (upstream: unhandled template exception → HTTP 500).
fn apply_sub_template(
    content: &[u8],
    ctx: &SubCtx<'_>,
    escape_html: bool,
) -> Result<Vec<u8>, SubError> {
    let mut out = Vec::with_capacity(content.len());
    let mut vars: HashMap<String, SubVal> = HashMap::new();
    let mut i = 0;
    while i < content.len() {
        // `{{([^}]*)}}`: find `{{`, then the first `}` must be doubled.
        let start = match content[i..].windows(2).position(|w| w == b"{{") {
            Some(s) => i + s,
            None => break,
        };
        let inner_start = start + 2;
        let rest = &content[inner_start..];
        let end_rel = rest.iter().position(|&b| b == b'}');
        let Some(end_rel) = end_rel else {
            // No closing brace at all: the rest is literal.
            break;
        };
        if rest.get(end_rel + 1) != Some(&b'}') {
            // `{{` not followed by `}}`: literal, like the regex
            // failing to match here; keep scanning after it.
            out.extend_from_slice(&content[i..inner_start]);
            i = inner_start;
            continue;
        }
        out.extend_from_slice(&content[i..start]);
        let value = eval_sub(&rest[..end_rel], ctx, &mut vars)?;
        let bytes = sub_value_bytes(&value)?;
        if escape_html {
            html_escape_into(&mut out, &bytes);
        } else {
            out.extend_from_slice(&bytes);
        }
        i = inner_start + end_rel + 2;
    }
    out.extend_from_slice(&content[i..]);
    Ok(out)
}

/// Render a substituted value: upstream asserts int/bytes/str and
/// stringifies ints.
fn sub_value_bytes(value: &SubVal) -> Result<Vec<u8>, SubError> {
    match value {
        SubVal::Str(s) => Ok(s.as_bytes().to_vec()),
        SubVal::Int(n) => Ok(n.to_string().into_bytes()),
        _ => Err(SubError::BadValue),
    }
}

/// `html.escape(s, quote=True)`: `&<>"'` →
/// `&amp;&lt;&gt;&quot;&#x27;`.
fn html_escape_into(out: &mut Vec<u8>, bytes: &[u8]) {
    for &b in bytes {
        match b {
            b'&' => out.extend_from_slice(b"&amp;"),
            b'<' => out.extend_from_slice(b"&lt;"),
            b'>' => out.extend_from_slice(b"&gt;"),
            b'"' => out.extend_from_slice(b"&quot;"),
            b'\'' => out.extend_from_slice(b"&#x27;"),
            _ => out.push(b),
        }
    }
}

// ============================================================
// Session protocol: `bowser --session` child + `SessionDriver`.
// See the module docs for the wire format and driver policy.
// ============================================================

/// Wire-protocol version spoken by [`run_session_child`] and
/// [`SessionDriver`]. The child prints it in its `HELLO` line; the
/// driver requires an exact match.
pub const SESSION_PROTO_VERSION: u32 = 1;

/// First line the session child prints on stdout after startup: the
/// driver's spawn handshake.
pub const SESSION_HELLO: &str = "HELLO bowser-session/1";

/// Maximum viewport dimension in CSS px accepted by the session child.
/// Mirrors `main.rs`'s `MAX_VIEWPORT_DIM`: a 16384x16384 RGBA8 frame is
/// already 1 GiB; anything larger is rejected instead of attempting a
/// multi-gigabyte allocation.
const MAX_SESSION_VIEWPORT_DIM: u32 = 16_384;

/// Viewport for `CHECK`/`HARNESS` commands: the headless pipeline needs
/// one, and the one-shot `--check` path uses the binary's default
/// viewport (1280x800) since the driver never passes `--viewport` there.
/// Fixed here for verdict parity between the two transports.
const SESSION_CHECK_VIEWPORT: (u32, u32) = (1280, 800);

/// Cap on console bytes the driver's session reader keeps per operation.
/// Mirrors [`MAX_DRAIN_BYTES`]: past it, bytes are dropped and the cut is
/// recorded, so a chatty page cannot grow driver RAM without bound.
const MAX_SESSION_CONSOLE_BYTES: usize = MAX_DRAIN_BYTES;

/// Outcome class of a session `CHECK`/`HARNESS` command, produced by the
/// child and consumed by [`SessionDriver`]. Maps onto [`RunVerdict`] for
/// the shared harness interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionVerdict {
    /// The page loaded and every script ran with no uncaught errors.
    Clean,
    /// Page-level failure: fetch/load failure or an uncaught script
    /// error. The child itself is alive.
    PageFailed,
}

/// One parsed `bowser --session` command line.
#[derive(Debug)]
enum SessionCommand {
    Ping {
        id: u64,
        nonce: String,
    },
    Render {
        id: u64,
        nonce: String,
        width: u32,
        height: u32,
        url: String,
    },
    Check {
        id: u64,
        nonce: String,
        url: String,
    },
    Harness {
        id: u64,
        nonce: String,
        url: String,
    },
    Quit,
}

fn parse_session_id(s: &str) -> Result<u64, String> {
    s.parse::<u64>()
        .map_err(|_| format!("bad id {s:?}: expected a u64"))
}

/// Nonces are 32 lowercase/uppercase hex chars (see [`new_report_nonce`]).
fn parse_session_nonce(s: &str) -> Result<String, String> {
    if s.len() == 32 && s.bytes().all(|b| b.is_ascii_hexdigit()) {
        Ok(s.to_owned())
    } else {
        Err(format!(
            "bad nonce {s:?}: expected 32 hex chars, like the driver generates"
        ))
    }
}

/// Parse one session command line (without the trailing newline).
fn parse_session_command(line: &str) -> Result<SessionCommand, String> {
    let line = line.trim_end_matches(['\r', '\n']);
    if line == "QUIT" {
        return Ok(SessionCommand::Quit);
    }
    let (cmd, args) = line
        .split_once(' ')
        .ok_or_else(|| format!("malformed command {line:?}: expected CMD args"))?;
    match cmd {
        "PING" => {
            let (id, nonce) = args
                .split_once(' ')
                .ok_or_else(|| format!("malformed PING {line:?}: expected PING <id> <nonce>"))?;
            if nonce.contains(' ') {
                return Err(format!("malformed PING {line:?}: nonce must be one token"));
            }
            Ok(SessionCommand::Ping {
                id: parse_session_id(id)?,
                nonce: parse_session_nonce(nonce)?,
            })
        }
        "RENDER" => {
            // RENDER <id> <nonce> <w> <h> <url>; the url runs to end of
            // line so query strings and (pathological) spaces survive.
            let mut parts = args.splitn(5, ' ');
            let (id, nonce, w, h, url) = match (
                parts.next(),
                parts.next(),
                parts.next(),
                parts.next(),
                parts.next(),
            ) {
                (Some(id), Some(nonce), Some(w), Some(h), Some(url)) => (id, nonce, w, h, url),
                _ => {
                    return Err(format!(
                        "malformed RENDER {line:?}: expected RENDER <id> <nonce> <w> <h> <url>"
                    ));
                }
            };
            if url.is_empty() {
                return Err(format!("malformed RENDER {line:?}: url is empty"));
            }
            let width = parse_session_dim(w)?;
            let height = parse_session_dim(h)?;
            Ok(SessionCommand::Render {
                id: parse_session_id(id)?,
                nonce: parse_session_nonce(nonce)?,
                width,
                height,
                url: url.to_owned(),
            })
        }
        "CHECK" | "HARNESS" => {
            let mut parts = args.splitn(3, ' ');
            let (id, nonce, url) = match (parts.next(), parts.next(), parts.next()) {
                (Some(id), Some(nonce), Some(url)) if !url.is_empty() => (id, nonce, url),
                _ => {
                    return Err(format!(
                        "malformed {cmd} {line:?}: expected {cmd} <id> <nonce> <url>"
                    ));
                }
            };
            let id = parse_session_id(id)?;
            let nonce = parse_session_nonce(nonce)?;
            let url = url.to_owned();
            Ok(if cmd == "HARNESS" {
                SessionCommand::Harness { id, nonce, url }
            } else {
                SessionCommand::Check { id, nonce, url }
            })
        }
        other => Err(format!("unknown session command {other:?}")),
    }
}

fn parse_session_dim(s: &str) -> Result<u32, String> {
    let dim: u32 = s
        .parse()
        .map_err(|_| format!("bad dimension {s:?}: expected a positive integer"))?;
    if dim == 0 || dim > MAX_SESSION_VIEWPORT_DIM {
        return Err(format!(
            "bad dimension {s:?}: must be within 1..={MAX_SESSION_VIEWPORT_DIM}"
        ));
    }
    Ok(dim)
}

/// Collapse to one line for embedding in a protocol response line, and
/// cap the length: these messages cross into driver errors.
fn one_line(s: &str) -> String {
    const CAP: usize = 512;
    let mut out = String::with_capacity(s.len().min(CAP));
    for c in s.chars().take(CAP) {
        match c {
            '\r' | '\n' => out.push(' '),
            c => out.push(c),
        }
    }
    out
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

/// The per-command browser work the session child performs.
///
/// `render` rasterizes `url` at `width`x`height` CSS px to raw RGBA8
/// pixels (row-major, `4*w*h` bytes). `check` loads `url` through the
/// headless pipeline: `Ok(())` is clean, `Err(msg)` is page-failed (load
/// failure or uncaught script errors). `harness` is the check that also
/// carries the page's console output: the default impl defers to `check`
/// and ignores `console`, which is what the real pipeline does — its
/// `console.log` writes to the process stdout directly, the same stream
/// the protocol responses go to, so console lines naturally precede the
/// `OK` line and the driver demuxes them. Test stubs override `harness`
/// to inject canned console lines through `console` instead.
trait SessionOps {
    fn render(&mut self, url: &str, width: u32, height: u32) -> Result<Vec<u8>, String>;
    fn check(&mut self, url: &str) -> Result<(), String>;
    fn harness(&mut self, url: &str, console: &mut dyn Write) -> Result<(), String> {
        let _ = console;
        self.check(url)
    }
}

/// The real pipeline, owning the session's single [`FontCache`]: it is
/// built once here and threaded through every render, instead of once
/// per child process. Together with the thread-local GPU backend (see
/// [`WgpuBackend::render_offscreen`], brought up once on this thread and
/// reused), that is the per-test spawn cost the session amortizes.
///
/// The cache travels behind a [`Mutex`] because the reftest handshake
/// ([`settle_reftest_wait`](crate::window::settle_reftest_wait)) re-lays
/// out through the windowed pipeline's shared font-cache convention; the
/// session thread is the only locker here.
struct PipelineOps {
    fonts: Mutex<FontCache>,
}

impl PipelineOps {
    fn new() -> PipelineOps {
        PipelineOps {
            fonts: Mutex::new({
                let mut fonts = FontCache::new();
                fonts.install_test_fonts();
                fonts
            }),
        }
    }
}

/// The layout tree a session render builds its display list from: the
/// paint tree (nested scroll offsets baked in) once the pipeline has
/// baked one, else the layout tree. Mirrors `main.rs`'s
/// `headless_render_root`; kept in sync with it.
fn session_render_root(tab: &crate::Tab) -> Option<&LayoutBox> {
    tab.paint_root().or(tab.layout_root.as_deref())
}

/// Round `viewport` to whole px; `None` for a zero-area viewport.
/// Mirrors `main.rs`'s private `viewport_whole`; kept in sync with it.
fn viewport_whole_px(viewport: Size) -> Option<(u32, u32)> {
    let w = viewport.w.max(0.0).round() as u32;
    let h = viewport.h.max(0.0).round() as u32;
    (w > 0 && h > 0).then_some((w, h))
}

/// Composite straight-alpha RGBA8 pixels over opaque white, in place.
/// Mirrors `main.rs`'s private `composite_over_white`; kept in sync with
/// it.
fn composite_over_white(pixels: &mut [u8]) {
    for px in pixels.as_chunks_mut::<4>().0 {
        let (r, g, b, a) = (
            u32::from(px[0]),
            u32::from(px[1]),
            u32::from(px[2]),
            u32::from(px[3]),
        );
        let inv = 255 - a;
        px[0] = ((r * a + 255 * inv + 127) / 255) as u8;
        px[1] = ((g * a + 255 * inv + 127) / 255) as u8;
        px[2] = ((b * a + 255 * inv + 127) / 255) as u8;
        px[3] = 255;
    }
}

/// Render through a caller-owned [`FontCache`] instead of constructing
/// one per call. Mirrors `main.rs`'s private
/// `render_layout_to_rgba_with_fonts`; kept in sync with it: the GPU
/// path builds the display list resolving font keys against `fonts` and
/// renders offscreen (the backend persists thread-locally across
/// commands), and the adapter-less fallback is the plain
/// [`render_layout_to_rgba_with_scroll`] entry point (identical pixels).
///
/// `scroll` is the viewport scroll offset (CSS px), subtracted from
/// content at draw time like the windowed path's per-frame uniform.
fn render_with_fonts(
    root: &LayoutBox,
    doc: &Document,
    viewport: Size,
    image_resolver: &mut dyn FnMut(&str) -> Option<(ImageId, DecodedImage)>,
    fonts: &mut FontCache,
    scroll: (f32, f32),
) -> Vec<u8> {
    let Some((w, h)) = viewport_whole_px(viewport) else {
        return Vec::new();
    };
    if WgpuBackend::new_headless().is_some() {
        let display_list = {
            let mut resolve_fonts =
                |style: &ComputedStyle| -> FontKey { font_key_for_style(fonts, style) };
            build_display_list(
                root,
                doc,
                &mut resolve_fonts,
                &mut *image_resolver,
                viewport,
            )
        };
        if let Some(mut pixels) =
            WgpuBackend::render_offscreen_with_scroll(&display_list, w, h, fonts, scroll)
        {
            composite_over_white(&mut pixels);
            return pixels;
        }
        // Adapter vanished between probe and render; fall through to the
        // CPU path below.
    }
    render_layout_to_rgba_with_scroll(root, doc, viewport, image_resolver, scroll)
}

impl SessionOps for PipelineOps {
    /// Mirror of `main.rs`'s `render_headless_pixels`, but through the
    /// session's shared [`FontCache`].
    fn render(&mut self, url: &str, width: u32, height: u32) -> Result<Vec<u8>, String> {
        let viewport = Size {
            w: width as f32,
            h: height as f32,
        };
        let mut tab = load_tab_headless(url, viewport).map_err(|e| format!("load failed: {e}"))?;
        // Reftest handshake (see `settle_reftest_wait`): the session
        // child carries no deadline of its own — timeouts are enforced
        // driver-side by killing the child at `render_timeout` — so the
        // handshake's own budget sits well under it.
        crate::window::settle_reftest_wait(
            &mut tab,
            &self.fonts,
            Instant::now() + crate::window::REFTEST_SETTLE_BUDGET,
        );
        let (Some(document), Some(root)) = (tab.document_ref(), session_render_root(&tab)) else {
            return Err("load produced no document to render".to_string());
        };
        let document_url = tab
            .current_url()
            .cloned()
            .ok_or("load produced a tab with no URL")?;
        // Same `src` -> `ImageId` mapping the windowed paint path uses.
        let mut image_ids = ImageIdMap::new(&document, Some(&document_url));
        let mut resolve_images =
            |src: &str| -> Option<(ImageId, DecodedImage)> { image_ids.resolve(src) };
        // The session thread is the only locker; recover from a poisoned
        // mutex like every other lock site in this file.
        let mut fonts = self.fonts.lock().unwrap_or_else(|e| e.into_inner());
        // The live viewport offset rides into the rasterizer like the
        // windowed path's per-frame uniform: a WPT run that scrolled the
        // page (e.g. via `scrollIntoView` in the test) screenshots at
        // the scrolled position, not the top of the document.
        let scroll = tab.scroll().offset();
        Ok(render_with_fonts(
            root,
            &document,
            viewport,
            &mut resolve_images,
            &mut fonts,
            scroll,
        ))
    }

    /// Mirror of `main.rs`'s `check_page` verdict: `Ok` only when the
    /// load succeeded and no script threw an uncaught error.
    fn check(&mut self, url: &str) -> Result<(), String> {
        let viewport = Size {
            w: SESSION_CHECK_VIEWPORT.0 as f32,
            h: SESSION_CHECK_VIEWPORT.1 as f32,
        };
        let tab = load_tab_headless(url, viewport).map_err(|e| format!("load failed: {e}"))?;
        let errors = tab
            .script_report()
            .map(|report| report.errors.len())
            .unwrap_or(0);
        if errors == 0 {
            return Ok(());
        }
        let first = tab
            .script_report()
            .and_then(|report| report.errors.first())
            .map(|error| error.to_string())
            .unwrap_or_default();
        Err(format!(
            "{} uncaught script error{}: {first}",
            errors,
            if errors == 1 { "" } else { "s" }
        ))
    }
}

/// Run the session protocol loop: print `HELLO`, then serve commands
/// from `input`, writing framed responses to `output`, until `QUIT`,
/// stdin EOF, or an I/O error. Returns the process exit code.
///
/// Commands run strictly sequentially on the calling thread. A command
/// that panics the pipeline is caught: the child answers
/// `ERR <id> <nonce> BROWSER` and exits 1, so the driver restarts the
/// session instead of continuing on possibly-corrupt cached state (font
/// cache, GPU backend). Timeouts are enforced driver-side — the child
/// has no deadline of its own; the driver kills it.
fn run_session_loop(
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    ops: &mut dyn SessionOps,
) -> i32 {
    let _ = writeln!(output, "{SESSION_HELLO}");
    let _ = output.flush();
    let mut line = String::new();
    loop {
        line.clear();
        match input.read_line(&mut line) {
            Ok(0) => return 0, // The driver went away; exit cleanly.
            Ok(_) => {}
            Err(_) => return 1,
        }
        let cmd = match parse_session_command(&line) {
            Ok(cmd) => cmd,
            Err(msg) => {
                // Malformed command: our driver's bug, not the child's —
                // say so and stay alive for the next command.
                let _ = writeln!(output, "ERR 0 - BADCMD {}", one_line(&msg));
                let _ = output.flush();
                continue;
            }
        };
        match cmd {
            SessionCommand::Quit => return 0,
            SessionCommand::Ping { id, nonce } => {
                let _ = writeln!(output, "PONG {id} {nonce}");
                let _ = output.flush();
            }
            SessionCommand::Render {
                id,
                nonce,
                width,
                height,
                url,
            } => {
                let outcome =
                    std::panic::catch_unwind(AssertUnwindSafe(|| ops.render(&url, width, height)));
                match outcome {
                    Ok(Ok(pixels)) => {
                        let _ = writeln!(output, "OK {id} {nonce} RENDER {}", pixels.len());
                        let _ = output.write_all(&pixels);
                        let _ = output.flush();
                    }
                    Ok(Err(msg)) => {
                        let _ = writeln!(output, "ERR {id} {nonce} BROWSER {}", one_line(&msg));
                        let _ = output.flush();
                    }
                    Err(payload) => {
                        let _ = writeln!(
                            output,
                            "ERR {id} {nonce} BROWSER pipeline panic: {}",
                            one_line(&panic_message(&payload))
                        );
                        let _ = output.flush();
                        return 1;
                    }
                }
            }
            SessionCommand::Check { id, nonce, url } => {
                let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| ops.check(&url)));
                match outcome {
                    Ok(Ok(())) => {
                        let _ = writeln!(output, "OK {id} {nonce} CHECK clean");
                    }
                    Ok(Err(msg)) => {
                        let _ = writeln!(
                            output,
                            "OK {id} {nonce} CHECK page-failed {}",
                            one_line(&msg)
                        );
                    }
                    Err(payload) => {
                        let _ = writeln!(
                            output,
                            "ERR {id} {nonce} BROWSER pipeline panic: {}",
                            one_line(&panic_message(&payload))
                        );
                        let _ = output.flush();
                        return 1;
                    }
                }
                let _ = output.flush();
            }
            SessionCommand::Harness { id, nonce, url } => {
                // The op gets the output handle so test stubs can inject
                // console lines ahead of the OK line; the real pipeline
                // ignores it — its `console.log` writes to the process
                // stdout directly, which is this same stream, so those
                // lines naturally precede the OK line and the driver
                // demuxes them (see the module docs).
                let outcome =
                    std::panic::catch_unwind(AssertUnwindSafe(|| ops.harness(&url, output)));
                match outcome {
                    Ok(Ok(())) => {
                        let _ = writeln!(output, "OK {id} {nonce} HARNESS clean");
                    }
                    Ok(Err(msg)) => {
                        let _ = writeln!(
                            output,
                            "OK {id} {nonce} HARNESS page-failed {}",
                            one_line(&msg)
                        );
                    }
                    Err(payload) => {
                        let _ = writeln!(
                            output,
                            "ERR {id} {nonce} BROWSER pipeline panic: {}",
                            one_line(&panic_message(&payload))
                        );
                        let _ = output.flush();
                        return 1;
                    }
                }
                let _ = output.flush();
            }
        }
    }
}

/// The `bowser --session` child entry point: serve the session protocol
/// on stdin/stdout (see the module docs). Never returns.
///
/// `main.rs` glue (kept there because this module is feature-gated and
/// `main.rs` is a separate binary crate):
/// ```ignore
/// // in the flag parser:
/// "session" => { args.session = true; }
/// // in main(), before the other modes, under cfg(feature = "wpt-driver"):
/// if args.session { bowser_app::wpt_driver::run_session_child(); }
/// ```
pub fn run_session_child() -> ! {
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let mut output = std::io::stdout();
    let mut ops = PipelineOps::new();
    let code = run_session_loop(&mut input, &mut output, &mut ops);
    std::process::exit(code);
}

// ============================================================
// `SessionDriver`: the driver side of the session protocol.
// ============================================================

/// Default idle time after which the next operation is preceded by a
/// `PING` health probe.
pub const DEFAULT_SESSION_PROBE_INTERVAL: Duration = Duration::from_secs(30);
/// Default deadline for the session child's `HELLO` handshake after spawn.
pub const DEFAULT_SESSION_SPAWN_TIMEOUT: Duration = Duration::from_secs(10);
/// Deadline for a `PING` probe round-trip.
const SESSION_PROBE_TIMEOUT: Duration = Duration::from_secs(5);
/// Extra wait past the operation's own timeout for the session reader:
/// the child enforces no deadline of its own, so a response can
/// legitimately arrive just as the budget expires; the grace keeps the
/// driver from killing a child that was about to answer. The runner's
/// own watchdog remains the outer bound.
const SESSION_OP_GRACE: Duration = Duration::from_secs(5);
/// Consecutive session failures (transport errors and failed restarts)
/// before the session is permanently disabled in favor of the one-shot
/// fallback: a deterministically broken child must not cost a
/// spawn+handshake on every remaining test.
const MAX_CONSECUTIVE_SESSION_FAILURES: u32 = 3;

/// Counters describing how a [`SessionDriver`] served its operations.
/// See [`SessionDriver::session_stats`].
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    /// Operations completed through the session child.
    pub session_ops: u64,
    /// Operations completed through the one-shot fallback.
    pub fallback_ops: u64,
    /// Session child spawns, including restarts.
    pub spawns: u32,
    /// Restarts after a failure (a spawn that followed a dead child).
    pub restarts: u32,
    /// `PING` health probes sent.
    pub probes: u64,
}

/// [`BrowserDriver`] that multiplexes test operations over one
/// long-lived `bowser --session` child (see the module docs) instead of
/// spawning one child per operation.
///
/// The child is spawned lazily on first use. Every operation goes
/// through the session while it is healthy; when the session cannot be
/// established or fails, the driver restarts it (eagerly, once per
/// incident) and falls back to the one-shot [`BowserDriver`] path for
/// operations the session cannot serve. After
/// [`MAX_CONSECUTIVE_SESSION_FAILURES`] consecutive failures the session
/// is permanently disabled and every operation uses the fallback, so a
/// broken child degrades to exactly the old behavior instead of burning
/// a spawn+handshake on every remaining test.
///
/// Like [`BowserDriver`], all methods take `&self`: session state lives
/// behind a mutex, and operations serialize on it — the child processes
/// commands strictly sequentially.
pub struct SessionDriver {
    /// The one-shot driver: fixture server, binary path, and timeouts.
    /// Also the fallback transport.
    inner: BowserDriver,
    state: Mutex<SessionState>,
    /// Idle time after which the next operation is preceded by a `PING`
    /// health probe.
    pub probe_interval: Duration,
    /// Deadline for the session child's `HELLO` handshake after spawn.
    pub spawn_timeout: Duration,
}

struct SessionState {
    child: SessionChildState,
    /// The session is permanently disabled: every operation uses the
    /// one-shot fallback. Set once, never cleared.
    disabled: bool,
    /// The last spawn followed a dead child (counts toward
    /// [`SessionStats::restarts`]).
    failed: bool,
    consecutive_failures: u32,
    stats: SessionStats,
}

enum SessionChildState {
    Down,
    Up(SessionChild),
}

struct SessionChild {
    proc: Child,
    stdin: ChildStdin,
    /// Shared with the per-operation reader thread: it locks, reads one
    /// full response (line plus any payload), and unlocks. Operations
    /// serialize on the session mutex, so at most one reader exists.
    stdout: Arc<Mutex<BufReader<ChildStdout>>>,
    /// Capped stderr tail, filled by a detached drain thread.
    stderr: Arc<Mutex<DrainedOutput>>,
    next_id: u64,
    last_ok: Instant,
}

/// One driver operation as a session command.
enum SessionOp {
    Render {
        url: String,
        width: u32,
        height: u32,
    },
    Check {
        url: String,
    },
    Harness {
        url: String,
    },
}

impl SessionOp {
    fn command_line(&self, id: u64, nonce: &str) -> String {
        match self {
            SessionOp::Render { url, width, height } => {
                format!("RENDER {id} {nonce} {width} {height} {url}")
            }
            SessionOp::Check { url } => format!("CHECK {id} {nonce} {url}"),
            SessionOp::Harness { url } => format!("HARNESS {id} {nonce} {url}"),
        }
    }

    fn kind(&self) -> OpKind {
        match self {
            SessionOp::Render { width, height, .. } => OpKind::Render {
                width: *width,
                height: *height,
            },
            SessionOp::Check { .. } => OpKind::Check,
            SessionOp::Harness { .. } => OpKind::Harness,
        }
    }
}

/// What the session reader expects, so it can validate payloads.
#[derive(Debug, Clone, Copy)]
enum OpKind {
    Ping,
    Render { width: u32, height: u32 },
    Check,
    Harness,
}

/// Why [`SessionDriver::transact`] did not use the session.
enum TransactError {
    /// The session is unavailable (never started, failed to spawn, or
    /// permanently disabled): the caller falls back to the one-shot path.
    Unavailable,
    /// The session was used and the operation failed: the error is
    /// terminal for this operation (timeout, dead child, protocol
    /// violation) and must not be retried silently.
    Failed(DriverError),
}

enum ExchangeError {
    Transport(TransportFailure),
    /// Our parser and our child disagree on the protocol shape:
    /// deterministic, never transient.
    Protocol(String),
}

enum TransportFailure {
    /// The operation outlived its budget; the child was killed.
    Timeout,
    /// The child is gone (clean EOF).
    Eof,
    Io(String),
}

/// Random starting command id for a session child. Ids are sequential
/// from here, and the page under test can read the command nonce from
/// `location.search`, so a fixed start of 1 would let a hostile page
/// spray forged verdicts for i=1..N. `RandomState` keys are fresh OS
/// randomness per instance, so each child gets an unpredictable start;
/// 0 is avoided (the child uses `ERR 0 -` for id-less rejections).
fn random_session_start_id() -> u64 {
    RandomState::new().build_hasher().finish().max(1)
}

impl SessionDriver {
    /// Point at the built `bowser` binary and start a fixture server over
    /// `doc_root`, like [`BowserDriver::new`]. The session child itself
    /// is spawned lazily on first use. The server shuts down when the
    /// driver is dropped.
    pub fn new(exe: PathBuf, doc_root: &Path) -> std::io::Result<SessionDriver> {
        Ok(SessionDriver {
            inner: BowserDriver::new(exe, doc_root)?,
            state: Mutex::new(SessionState {
                child: SessionChildState::Down,
                disabled: false,
                failed: false,
                consecutive_failures: 0,
                stats: SessionStats::default(),
            }),
            probe_interval: DEFAULT_SESSION_PROBE_INTERVAL,
            spawn_timeout: DEFAULT_SESSION_SPAWN_TIMEOUT,
        })
    }

    /// The wrapped one-shot driver: fixture server base URL, binary path,
    /// and the `render_timeout`/`check_timeout` budgets the session
    /// operations use.
    pub fn inner(&self) -> &BowserDriver {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut BowserDriver {
        &mut self.inner
    }

    /// Counters describing how operations were served so far. Useful for
    /// measuring the session's spawn savings in a run.
    pub fn session_stats(&self) -> SessionStats {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stats
            .clone()
    }

    fn kill_child(child: &mut SessionChild) {
        Self::kill_and_reap(&mut child.proc);
    }

    /// Kill and reap: dropping a `Child` without waiting leaves a zombie,
    /// and a long WPT run would accumulate one per restart. `wait`
    /// returns promptly after SIGKILL.
    fn kill_and_reap(proc: &mut Child) {
        let _ = proc.kill();
        let _ = proc.wait();
    }

    /// Note the permanent switch to the one-shot fallback, once: silent
    /// degradation would hide a broken session from WPT run logs.
    fn note_disabled(&self, st: &mut SessionState, reason: &str) {
        if !st.disabled {
            st.disabled = true;
            eprintln!(
                "bowser: wpt session disabled ({reason}); \
                 falling back to one child per operation"
            );
        }
    }

    /// Spawn the session child and run the `HELLO` handshake, bounded by
    /// [`SessionDriver::spawn_timeout`]. A `bowser` binary without
    /// `--session` support (built without the `wpt-driver` feature)
    /// prints usage to stderr and exits, which fails the handshake here
    /// and sends the driver to the one-shot fallback.
    fn spawn_session_child(&self) -> Result<SessionChild, String> {
        let mut proc = Command::new(&self.inner.exe)
            .arg("--session")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to spawn {}: {e}", self.inner.exe.display()))?;
        let stdin: ChildStdin = proc.stdin.take().ok_or("session child has no stdin")?;
        let stdout: ChildStdout = proc.stdout.take().ok_or("session child has no stdout")?;
        let stderr: ChildStderr = proc.stderr.take().ok_or("session child has no stderr")?;
        // Detached drain: the child's stderr must never block it, and the
        // tail is attached to crash errors. The thread owns the pipe FD
        // and exits on EOF once the child dies; it is never joined.
        // Bound: one drain thread per spawn attempt, and every spawn
        // attempt either hands the child over (its death is observed via
        // try_wait/ping) or kills it on the handshake paths below. The
        // child spawns no grandchildren, so EOF after the kill is
        // prompt; repeated spawn failures disable the session after
        // MAX_CONSECUTIVE_SESSION_FAILURES.
        let err_buf = Arc::new(Mutex::new(DrainedOutput::default()));
        thread::Builder::new()
            .name("wpt-session-stderr".into())
            .spawn({
                let err_buf = Arc::clone(&err_buf);
                move || {
                    *err_buf.lock().unwrap_or_else(|e| e.into_inner()) = drain(Some(stderr));
                }
            })
            .map_err(|e| format!("failed to spawn session stderr drain: {e}"))?;
        let (tx, rx) = mpsc::channel();
        thread::Builder::new()
            .name("wpt-session-hello".into())
            .spawn(move || {
                let mut reader = BufReader::new(stdout);
                let mut line = String::new();
                let result = reader.read_line(&mut line).map(|_| line);
                let _ = tx.send((result, reader));
            })
            .map_err(|e| format!("failed to spawn session handshake reader: {e}"))?;
        let (result, reader) = match rx.recv_timeout(self.spawn_timeout) {
            Ok(pair) => pair,
            Err(RecvTimeoutError::Timeout) => {
                // The kill turns the handshake reader's blocked
                // read_line into EOF so it exits; like the stderr
                // drain, at most one abandoned thread per spawn
                // attempt, and spawn attempts are failure-capped.
                Self::kill_and_reap(&mut proc);
                return Err(format!(
                    "session child did not print HELLO within {:?}",
                    self.spawn_timeout
                ));
            }
            Err(RecvTimeoutError::Disconnected) => {
                Self::kill_and_reap(&mut proc);
                return Err("session handshake reader died".to_string());
            }
        };
        let line = result.map_err(|e| {
            Self::kill_and_reap(&mut proc);
            format!("session child closed stdout before HELLO: {e}")
        })?;
        if line.trim_end_matches(['\r', '\n']) != SESSION_HELLO {
            Self::kill_and_reap(&mut proc);
            return Err(format!(
                "bad session HELLO: {line:?} (want {SESSION_HELLO:?})"
            ));
        }
        // The child may have exited right after HELLO; fail fast instead
        // of handing out a dead child.
        if proc.try_wait().map(|s| s.is_some()).unwrap_or(true) {
            return Err("session child exited right after HELLO".to_string());
        }
        Ok(SessionChild {
            proc,
            stdin,
            stdout: Arc::new(Mutex::new(reader)),
            stderr: err_buf,
            next_id: random_session_start_id(),
            last_ok: Instant::now(),
        })
    }

    /// Run one `PING` round-trip against a live child. The caller holds
    /// the session mutex.
    fn ping_locked(&self, child: &mut SessionChild) -> Result<(), ExchangeError> {
        let id = child.next_id;
        child.next_id += 1;
        let nonce = new_report_nonce();
        let cmd = format!("PING {id} {nonce}");
        match self.exchange_locked(child, &cmd, id, &nonce, OpKind::Ping, SESSION_PROBE_TIMEOUT)? {
            SessionResponse {
                kind: ResponseKind::Pong,
                ..
            } => Ok(()),
            _ => Err(ExchangeError::Protocol(
                "expected PONG to session PING".to_string(),
            )),
        }
    }

    /// Ensure a live session child: drop one that exited on its own,
    /// probe an idle one, and (re)spawn when down. Returns false when
    /// the session is unavailable and the caller must use the one-shot
    /// fallback. The caller holds the session mutex.
    fn ensure_up(&self, st: &mut SessionState) -> bool {
        if let SessionChildState::Up(child) = &mut st.child {
            let dead = child.proc.try_wait().map(|s| s.is_some()).unwrap_or(true);
            if dead {
                st.child = SessionChildState::Down;
                st.failed = true;
            }
        }
        if let SessionChildState::Up(child) = &mut st.child
            && child.last_ok.elapsed() > self.probe_interval
        {
            match self.ping_locked(child) {
                Ok(()) => {
                    st.stats.probes += 1;
                    st.consecutive_failures = 0;
                    child.last_ok = Instant::now();
                }
                Err(_) => {
                    Self::kill_child(child);
                    st.child = SessionChildState::Down;
                    st.failed = true;
                    st.consecutive_failures += 1;
                }
            }
        }
        if matches!(st.child, SessionChildState::Down) && !st.disabled {
            if st.consecutive_failures >= MAX_CONSECUTIVE_SESSION_FAILURES {
                self.note_disabled(
                    st,
                    &format!("{MAX_CONSECUTIVE_SESSION_FAILURES} consecutive session failures"),
                );
            } else {
                match self.spawn_session_child() {
                    Ok(child) => {
                        st.stats.spawns += 1;
                        if st.failed {
                            st.stats.restarts += 1;
                            st.failed = false;
                        }
                        st.child = SessionChildState::Up(child);
                    }
                    Err(e) => self.note_disabled(st, &format!("spawn failed: {e}")),
                }
            }
        }
        matches!(st.child, SessionChildState::Up(_))
    }

    /// Write one command and read its full response (line plus any
    /// payload) on a reader thread, bounded by `timeout`. The caller
    /// holds the session mutex; the child is borrowed mutably for the
    /// whole exchange, so commands never interleave.
    fn exchange_locked(
        &self,
        child: &mut SessionChild,
        cmd: &str,
        id: u64,
        nonce: &str,
        kind: OpKind,
        timeout: Duration,
    ) -> Result<SessionResponse, ExchangeError> {
        if let Err(e) = writeln!(child.stdin, "{cmd}").and_then(|()| child.stdin.flush()) {
            return Err(ExchangeError::Transport(TransportFailure::Io(format!(
                "write to session child: {e}"
            ))));
        }
        // The reader thread owns the stdout lock for one response; a
        // blocking pipe read has no timeout, so the bound lives here on
        // the channel instead. On timeout the child is killed (the
        // abandoned reader then sees EOF and exits); it is never joined,
        // since a wedged child could hold the pipe open via a grandchild.
        //
        // Leak bound: at most one abandoned reader thread plus its pipe
        // FD per timed-out operation. The timeout arm below kills the
        // child immediately, so the reader's blocked read_line turns
        // into EOF and the thread exits promptly. It can only linger if
        // a grandchild inherited the pipe — the session child (`bowser
        // --session`) never spawns grandchildren — and repeated timeouts
        // disable the session after MAX_CONSECUTIVE_SESSION_FAILURES
        // consecutive failures, so accumulation is bounded.
        let stdout = Arc::clone(&child.stdout);
        let (tx, rx) = mpsc::channel();
        let nonce = nonce.to_owned();
        if let Err(e) = thread::Builder::new()
            .name("wpt-session-reader".into())
            .spawn(move || {
                let mut locked = stdout.lock().unwrap_or_else(|e| e.into_inner());
                let outcome = read_session_response(&mut locked, id, &nonce, kind);
                let _ = tx.send(outcome);
            })
        {
            return Err(ExchangeError::Transport(TransportFailure::Io(format!(
                "spawn session reader: {e}"
            ))));
        }
        match rx.recv_timeout(timeout + SESSION_OP_GRACE) {
            Ok(ReadOutcome::Response(resp)) => Ok(resp),
            Ok(ReadOutcome::Eof) => Err(ExchangeError::Transport(TransportFailure::Eof)),
            Ok(ReadOutcome::Io(msg)) => Err(ExchangeError::Transport(TransportFailure::Io(msg))),
            Ok(ReadOutcome::Protocol(msg)) => Err(ExchangeError::Protocol(msg)),
            Err(RecvTimeoutError::Timeout) => {
                // Kill promptly: closing the child's pipe ends turns the
                // abandoned reader's blocked read_line into EOF so it
                // exits instead of lingering on the FD. transact()
                // kills again on this path; kill is idempotent.
                let _ = child.proc.kill();
                Err(ExchangeError::Transport(TransportFailure::Timeout))
            }
            Err(RecvTimeoutError::Disconnected) => Err(ExchangeError::Transport(
                TransportFailure::Io("session reader died without responding".to_string()),
            )),
        }
    }

    /// Run one operation through the session child, applying the
    /// restart policy. The session mutex is held for the whole call.
    /// `nonce` is the command nonce, echoed by the child in every
    /// response line; callers that need the nonce elsewhere (the
    /// harness report nonce, which doubles as the command nonce)
    /// generate it themselves.
    fn transact(
        &self,
        op: &SessionOp,
        nonce: &str,
        timeout: Duration,
        label: &str,
    ) -> Result<SessionResponse, TransactError> {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !self.ensure_up(&mut st) {
            return Err(TransactError::Unavailable);
        }
        let child = match &mut st.child {
            SessionChildState::Up(child) => child,
            SessionChildState::Down => return Err(TransactError::Unavailable),
        };
        let id = child.next_id;
        child.next_id += 1;
        let cmd = op.command_line(id, nonce);
        let kind = op.kind();
        match self.exchange_locked(child, &cmd, id, nonce, kind, timeout) {
            Ok(resp) => {
                child.last_ok = Instant::now();
                st.stats.session_ops += 1;
                st.consecutive_failures = 0;
                Ok(resp)
            }
            Err(ExchangeError::Protocol(msg)) => {
                // Deterministic: our parser and our child disagree on the
                // protocol shape (a wrong id/nonce line is console
                // output, not a protocol error — the reader scans past
                // it). Restarting cannot help; stop paying
                // spawn+handshake and use the one-shot path from here on.
                Self::kill_child(child);
                st.child = SessionChildState::Down;
                self.note_disabled(&mut st, &format!("protocol error: {msg}"));
                Err(TransactError::Failed(DriverError::BadReport(format!(
                    "wpt session protocol error during {label}: {msg}"
                ))))
            }
            Err(ExchangeError::Transport(failure)) => {
                let err_tail =
                    stderr_tail(&child.stderr.lock().unwrap_or_else(|e| e.into_inner()).bytes);
                Self::kill_child(child);
                st.child = SessionChildState::Down;
                st.failed = true;
                st.consecutive_failures += 1;
                // Eager restart: the next operation should not pay the
                // spawn on its critical path. One attempt; a failed
                // restart (or the failure cap) disables the session.
                let restarted = st.consecutive_failures < MAX_CONSECUTIVE_SESSION_FAILURES
                    && match self.spawn_session_child() {
                        Ok(new_child) => {
                            st.stats.spawns += 1;
                            st.stats.restarts += 1;
                            st.failed = false;
                            st.child = SessionChildState::Up(new_child);
                            true
                        }
                        Err(_) => false,
                    };
                if !restarted {
                    self.note_disabled(
                        &mut st,
                        &format!("{MAX_CONSECUTIVE_SESSION_FAILURES} consecutive session failures"),
                    );
                }
                Err(TransactError::Failed(map_transport_failure(
                    &failure, label, timeout, &err_tail,
                )))
            }
        }
    }

    fn fallback_render(&self, url: &str, viewport: Viewport) -> Result<Screenshot, DriverError> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stats
            .fallback_ops += 1;
        self.inner.render(url, viewport)
    }

    fn fallback_harness(
        &self,
        test: &TestItem,
        timeout: Duration,
    ) -> Result<(HarnessResult, TestharnessMarkers), DriverError> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stats
            .fallback_ops += 1;
        self.inner.run_testharness_with_markers(test, timeout)
    }

    fn fallback_check(&self, url: &str) -> Result<(), DriverError> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stats
            .fallback_ops += 1;
        self.inner.load_without_crash(url)
    }

    /// Run one testharness test through the session child, returning the
    /// completion report plus the parsed start/result marker lines (see
    /// [`TestharnessMarkers`]).
    ///
    /// [`BrowserDriver::run_testharness`] delegates here and drops the
    /// markers. Console lines the page prints during the run are demuxed
    /// from the session stream and scanned exactly like the one-shot
    /// path's stdout.
    pub fn run_testharness_with_markers(
        &self,
        test: &TestItem,
        timeout: Duration,
    ) -> Result<(HarnessResult, TestharnessMarkers), DriverError> {
        // The same nonce doubles as the session command nonce and the
        // harness report nonce: the driver accepts only the response line
        // carrying it, and only the injected reporter knows it (see
        // `last_report_line`).
        let nonce = new_report_nonce();
        let url = self.inner.harness_url(test, &nonce);
        let start = Instant::now();
        let op = SessionOp::Harness { url };
        let resp = match self.transact(&op, &nonce, timeout, &test.id) {
            Ok(resp) => resp,
            Err(TransactError::Unavailable) => return self.fallback_harness(test, timeout),
            Err(TransactError::Failed(e)) => return Err(e),
        };
        let duration_ms = Some(start.elapsed().as_millis() as u64);
        match resp.kind {
            ResponseKind::Harness(verdict) => interpret_harness_output(
                test,
                &nonce,
                match verdict {
                    SessionVerdict::Clean => RunVerdict::Clean,
                    SessionVerdict::PageFailed => RunVerdict::PageFailed,
                },
                &resp.console,
                resp.truncated,
                duration_ms,
            ),
            ResponseKind::BrowserError(msg) => Err(DriverError::BrowserGone(format!(
                "testharness run of {}: {msg}",
                test.id
            ))),
            ResponseKind::BadCmd(msg) => Err(DriverError::BadReport(format!(
                "session child rejected HARNESS of {}: {msg}",
                test.id
            ))),
            _ => Err(DriverError::BadReport(format!(
                "unexpected session response for testharness run of {}",
                test.id
            ))),
        }
    }
}

impl Drop for SessionDriver {
    fn drop(&mut self) {
        // Kill, don't QUIT: Drop must not block on a wedged child, and
        // the child holds no state worth flushing (every response is
        // flushed before it is sent). Recover from a poisoned mutex like
        // every other lock site in this file: a panic mid-transact must
        // not leak the child.
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let SessionChildState::Up(child) = &mut st.child {
            SessionDriver::kill_child(child);
        }
        st.child = SessionChildState::Down;
    }
}

impl BrowserDriver for SessionDriver {
    fn name(&self) -> &str {
        "bowser-session"
    }

    fn render(&self, url: &str, viewport: Viewport) -> Result<Screenshot, DriverError> {
        let op = SessionOp::Render {
            url: url.to_owned(),
            width: viewport.width,
            height: viewport.height,
        };
        let timeout = self.inner.render_timeout;
        let nonce = new_report_nonce();
        let resp = match self.transact(&op, &nonce, timeout, url) {
            Ok(resp) => resp,
            Err(TransactError::Unavailable) => return self.fallback_render(url, viewport),
            Err(TransactError::Failed(e)) => return Err(e),
        };
        match resp.kind {
            ResponseKind::Render(pixels) => {
                Screenshot::new(viewport.width, viewport.height, pixels)
            }
            ResponseKind::BrowserError(msg) => {
                Err(DriverError::BrowserGone(format!("render of {url}: {msg}")))
            }
            ResponseKind::BadCmd(msg) => Err(DriverError::BadReport(format!(
                "session child rejected RENDER of {url}: {msg}"
            ))),
            _ => Err(DriverError::BadReport(format!(
                "unexpected session response for render of {url}"
            ))),
        }
    }

    fn run_testharness(
        &self,
        test: &TestItem,
        timeout: Duration,
    ) -> Result<HarnessResult, DriverError> {
        self.run_testharness_with_markers(test, timeout)
            .map(|(report, _)| report)
    }

    fn load_without_crash(&self, url: &str) -> Result<(), DriverError> {
        let op = SessionOp::Check {
            url: url.to_owned(),
        };
        let timeout = self.inner.check_timeout;
        let nonce = new_report_nonce();
        let resp = match self.transact(&op, &nonce, timeout, url) {
            Ok(resp) => resp,
            Err(TransactError::Unavailable) => return self.fallback_check(url),
            Err(TransactError::Failed(e)) => return Err(e),
        };
        match resp.kind {
            // Parity with the one-shot path: any exit code — clean,
            // page-failed, even a pipeline-internal BROWSER error from a
            // live child — means the browser survived the load; only a
            // dead child (EOF/signal) is a crash.
            ResponseKind::Check(SessionVerdict::Clean)
            | ResponseKind::Check(SessionVerdict::PageFailed)
            | ResponseKind::BrowserError(_) => Ok(()),
            ResponseKind::BadCmd(msg) => Err(DriverError::BadReport(format!(
                "session child rejected CHECK of {url}: {msg}"
            ))),
            _ => Err(DriverError::BadReport(format!(
                "unexpected session response for load of {url}"
            ))),
        }
    }
}

/// Map a mid-operation transport failure to the [`DriverError`] the
/// operation reports. The session has already been killed and the
/// restart attempted by the time this runs.
fn map_transport_failure(
    failure: &TransportFailure,
    label: &str,
    timeout: Duration,
    err_tail: &str,
) -> DriverError {
    match failure {
        TransportFailure::Timeout => DriverError::Timeout {
            test: label.to_owned(),
            after: timeout,
        },
        TransportFailure::Eof => {
            DriverError::BrowserGone(format!("wpt session child died during {label}: {err_tail}"))
        }
        TransportFailure::Io(msg) => {
            DriverError::BrowserGone(format!("wpt session child I/O error during {label}: {msg}"))
        }
    }
}

/// What the session reader thread found on the child's stdout.
enum ReadOutcome {
    Response(SessionResponse),
    /// Clean EOF: the child is gone.
    Eof,
    Io(String),
    Protocol(String),
}

/// One full session response: the parsed response plus the console lines
/// seen while its command was in flight.
struct SessionResponse {
    kind: ResponseKind,
    /// Non-response lines seen while this command was in flight: the
    /// page's console output. Scanned for harness reports by HARNESS.
    console: Vec<u8>,
    /// The console buffer hit [`MAX_SESSION_CONSOLE_BYTES`] and was cut.
    truncated: bool,
}

#[derive(Debug)]
enum ResponseKind {
    Pong,
    Render(Vec<u8>),
    Check(SessionVerdict),
    Harness(SessionVerdict),
    BrowserError(String),
    BadCmd(String),
}

fn done_response(kind: ResponseKind, console: Vec<u8>, truncated: bool) -> ReadOutcome {
    ReadOutcome::Response(SessionResponse {
        kind,
        console,
        truncated,
    })
}

/// Read one full session response: the response line plus, for `RENDER`,
/// the exact pixel payload. Non-response lines are console output from
/// the page (or the child) and are collected for the caller; the nonce
/// is what separates a real response from a hostile page's
/// `console.log` spray — only a line carrying the in-flight command's
/// id *and* nonce counts, and response-shaped lines with the wrong
/// id/nonce are skipped as console while scanning for it.
fn read_session_response(
    reader: &mut BufReader<ChildStdout>,
    expect_id: u64,
    expect_nonce: &str,
    kind: OpKind,
) -> ReadOutcome {
    let mut console = Vec::new();
    let mut truncated = false;
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => return ReadOutcome::Eof,
            Ok(_) => {}
            Err(e) => return ReadOutcome::Io(format!("read session response: {e}")),
        }
        let text = line.trim_end_matches(['\r', '\n']);
        match classify_response_line(text, expect_id, expect_nonce) {
            LineKind::Console => {
                if !truncated {
                    // +1 for the newline re-added below.
                    if console.len() + text.len() + 1 > MAX_SESSION_CONSOLE_BYTES {
                        truncated = true;
                    } else {
                        console.extend_from_slice(text.as_bytes());
                        console.push(b'\n');
                    }
                }
            }
            LineKind::Malformed(msg) => {
                return ReadOutcome::Protocol(format!(
                    "malformed session response ({msg}): {text:?}"
                ));
            }
            LineKind::BadCmd(msg) => {
                return done_response(ResponseKind::BadCmd(msg), console, truncated);
            }
            LineKind::Pong => {
                // A PONG for a non-PING command means the stream is
                // desynchronized (or the child is buggy): deterministic,
                // never transient.
                if !matches!(kind, OpKind::Ping) {
                    return ReadOutcome::Protocol(format!(
                        "unexpected PONG response to a non-ping command: {text:?}"
                    ));
                }
                return done_response(ResponseKind::Pong, console, truncated);
            }
            LineKind::Render { len } => {
                let expected = match kind {
                    OpKind::Render { width, height } => width as u64 * height as u64 * 4,
                    _ => {
                        return ReadOutcome::Protocol(format!(
                            "unexpected RENDER response to a non-render command: {text:?}"
                        ));
                    }
                };
                if len != expected {
                    return ReadOutcome::Protocol(format!(
                        "RENDER byte count {len} does not match {expected} \
                         for the requested viewport"
                    ));
                }
                let mut pixels = vec![0u8; len as usize];
                match reader.read_exact(&mut pixels) {
                    Ok(()) => {
                        return done_response(ResponseKind::Render(pixels), console, truncated);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                        return ReadOutcome::Eof;
                    }
                    Err(e) => {
                        return ReadOutcome::Io(format!("read session pixels: {e}"));
                    }
                }
            }
            LineKind::Check(verdict) => {
                if !matches!(kind, OpKind::Check) {
                    return ReadOutcome::Protocol(format!(
                        "unexpected CHECK response to a non-check command: {text:?}"
                    ));
                }
                return done_response(ResponseKind::Check(verdict), console, truncated);
            }
            LineKind::Harness(verdict) => {
                if !matches!(kind, OpKind::Harness) {
                    return ReadOutcome::Protocol(format!(
                        "unexpected HARNESS response to a non-harness command: {text:?}"
                    ));
                }
                return done_response(ResponseKind::Harness(verdict), console, truncated);
            }
            LineKind::BrowserError(msg) => {
                return done_response(ResponseKind::BrowserError(msg), console, truncated);
            }
        }
    }
}

/// Classification of one child stdout line during a session operation.
enum LineKind {
    /// Not a response line: console output, attributed to the in-flight
    /// command. This includes right-shaped `TAG <id> <nonce>` lines with
    /// the *wrong* id or nonce: a page can print those at will (e.g.
    /// `console.log("OK", n, uuid)`), so they are console output and the
    /// reader keeps scanning for an exact match. Only an exact
    /// id+nonce forgery is accepted as a response, as before.
    Console,
    /// Our id and nonce, but the rest does not parse.
    Malformed(String),
    /// `ERR 0 - BADCMD ...`: the child rejected our command line.
    BadCmd(String),
    Pong,
    Render {
        len: u64,
    },
    Check(SessionVerdict),
    Harness(SessionVerdict),
    BrowserError(String),
}

/// Classify one child stdout line. A line counts as a response only when
/// it has the full `TAG <id> <32-hex-nonce>` shape with the in-flight
/// command's id and nonce; a right-shaped line with wrong values is
/// console output from the page (which can print such lines at will) —
/// the reader keeps scanning — and anything else, including a page
/// logging "OK computer", is console output too. A genuinely
/// unparseable protocol line (right id and nonce, bad body) is
/// [`LineKind::Malformed`].
fn classify_response_line(line: &str, expect_id: u64, expect_nonce: &str) -> LineKind {
    // The child rejects a malformed command without an id/nonce to echo
    // (there was nothing sane to echo); anything else under `ERR 0 -` is
    // not ours — treat it as console rather than killing the session on
    // a guess.
    if let Some(msg) = line.strip_prefix("ERR 0 - BADCMD ") {
        return LineKind::BadCmd(msg.to_owned());
    }
    if line.starts_with("ERR 0 - ") {
        return LineKind::Console;
    }
    let mut parts = line.splitn(4, ' ');
    let (tag, id_s, nonce_s) = match (parts.next(), parts.next(), parts.next()) {
        (Some(tag), Some(id_s), Some(nonce_s)) => (tag, id_s, nonce_s),
        _ => return LineKind::Console,
    };
    if !matches!(tag, "PONG" | "OK" | "ERR") {
        return LineKind::Console;
    }
    let shaped = id_s.parse::<u64>().is_ok()
        && nonce_s.len() == 32
        && nonce_s.bytes().all(|b| b.is_ascii_hexdigit());
    if !shaped {
        return LineKind::Console;
    }
    if id_s.parse::<u64>().ok() != Some(expect_id) || nonce_s != expect_nonce {
        // Right shape, wrong values: the page (or anything else on the
        // pipe) printed a response-shaped line. Console output, not a
        // protocol violation — keep scanning for the exact match.
        return LineKind::Console;
    }
    let rest = parts.next().unwrap_or("");
    let mut words = rest.split_whitespace();
    match tag {
        "PONG" => {
            if words.next().is_none() {
                LineKind::Pong
            } else {
                LineKind::Malformed("PONG takes no arguments".to_string())
            }
        }
        "OK" => match words.next() {
            Some("RENDER") => match (words.next(), words.next()) {
                (Some(len_s), None) => match len_s.parse::<u64>() {
                    Ok(len) => LineKind::Render { len },
                    Err(_) => LineKind::Malformed("bad RENDER byte count".to_string()),
                },
                _ => LineKind::Malformed("bad RENDER response".to_string()),
            },
            // A page-failed verdict may carry a message after the
            // verdict token; the driver does not need it.
            Some("CHECK") => match words.next() {
                Some("clean") => LineKind::Check(SessionVerdict::Clean),
                Some("page-failed") => LineKind::Check(SessionVerdict::PageFailed),
                _ => LineKind::Malformed("bad CHECK verdict".to_string()),
            },
            Some("HARNESS") => match words.next() {
                Some("clean") => LineKind::Harness(SessionVerdict::Clean),
                Some("page-failed") => LineKind::Harness(SessionVerdict::PageFailed),
                _ => LineKind::Malformed("bad HARNESS verdict".to_string()),
            },
            _ => LineKind::Malformed("bad OK response".to_string()),
        },
        "ERR" => {
            if let Some(msg) = rest.strip_prefix("BROWSER ") {
                LineKind::BrowserError(msg.to_owned())
            } else if rest == "BROWSER" {
                LineKind::BrowserError(String::new())
            } else {
                LineKind::Malformed("bad ERR response".to_string())
            }
        }
        _ => LineKind::Malformed("bad response tag".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bowser_js::{Value, Vm};
    use bowser_wpt::manifest::{FuzzySpec, TestKind};
    use std::os::unix::fs::PermissionsExt;

    /// Unique scratch dir for a test; removed when the test finishes.
    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("bowser-wpt-driver-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_exe(dir: &Path, name: &str, script: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, script).unwrap();
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();
        path
    }

    fn item(id: &str, url: &str) -> TestItem {
        TestItem {
            id: id.into(),
            url: url.into(),
            kind: TestKind::Testharness,
            path: None,
            references: Vec::new(),
        }
    }

    fn driver_with(exe: PathBuf) -> BowserDriver {
        let root = scratch("docroot");
        let mut d = BowserDriver::new(exe, &root).unwrap();
        d.render_timeout = Duration::from_secs(10);
        d.check_timeout = Duration::from_secs(10);
        let _ = std::fs::remove_dir_all(&root);
        d
    }

    /// Raw HTTP GET against the fixture server; returns (status, headers, body).
    fn http_get(base: &str, target: &str) -> (u16, Vec<(String, String)>, Vec<u8>) {
        let addr = base.strip_prefix("http://").unwrap();
        let mut stream = TcpStream::connect(addr).unwrap();
        stream
            .write_all(format!("GET {target} HTTP/1.1\r\nHost: x\r\n\r\n").as_bytes())
            .unwrap();
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).unwrap();
        let split = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .expect("no header/body split");
        let head = String::from_utf8(raw[..split].to_vec()).unwrap();
        let mut lines = head.lines();
        let status_line = lines.next().unwrap();
        let status: u16 = status_line
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let headers = lines
            .map(|l| {
                let (k, v) = l.split_once(':').unwrap();
                (k.trim().to_lowercase(), v.trim().to_owned())
            })
            .collect();
        (status, headers, raw[split + 4..].to_vec())
    }

    fn header<'a>(headers: &'a [(String, String)], name: &str) -> &'a str {
        headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
            .unwrap()
    }

    #[test]
    fn server_serves_files_with_content_type() {
        let root = scratch("serve");
        std::fs::write(root.join("index.html"), "<html>hi</html>").unwrap();
        std::fs::write(root.join("app.js"), "console.log(1);").unwrap();
        std::fs::write(root.join("style.css"), "p { color: red; }").unwrap();

        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();
        assert!(base.starts_with("http://127.0.0.1:"));

        let (status, headers, body) = http_get(&base, "/index.html");
        assert_eq!(status, 200);
        // Bare `text/html` like upstream wptserve: no charset parameter,
        // so `<meta charset>` in the page stays authoritative.
        assert_eq!(header(&headers, "content-type"), "text/html");
        assert_eq!(body, b"<html>hi</html>");

        let (status, headers, body) = http_get(&base, "/app.js?x=1");
        assert_eq!(status, 200);
        assert_eq!(
            header(&headers, "content-type"),
            "text/javascript; charset=utf-8"
        );
        assert_eq!(body, b"console.log(1);");

        let (status, headers, _) = http_get(&base, "/style.css");
        assert_eq!(status, 200);
        assert_eq!(header(&headers, "content-type"), "text/css; charset=utf-8");

        // Directory falls back to index.html.
        let (status, _, body) = http_get(&base, "/");
        assert_eq!(status, 200);
        assert_eq!(body, b"<html>hi</html>");

        shutdown.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn server_appends_testdriver_vendor_shim() {
        // The vendor shim implementing `test_driver_internal` rides on
        // the served `/resources/testdriver.js` (see
        // `crate::testdriver`); the WPT runner's one-shot browser
        // child cannot be reached over HTTP, so the in-page shim +
        // native bridge is the driver<->page channel.
        let root = scratch("tdshim");
        std::fs::create_dir_all(root.join("resources")).unwrap();
        let upstream = "window.test_driver={};window.test_driver_internal={};";
        std::fs::write(root.join("resources/testdriver.js"), upstream).unwrap();
        std::fs::write(root.join("resources/other.js"), "var x = 1;").unwrap();

        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        // The shim is appended after the upstream bytes.
        let (status, headers, body) = http_get(&base, "/resources/testdriver.js");
        assert_eq!(status, 200);
        assert_eq!(
            header(&headers, "content-type"),
            "text/javascript; charset=utf-8"
        );
        let body = String::from_utf8(body).unwrap();
        assert!(body.starts_with(upstream));
        assert!(body.ends_with(crate::testdriver::VENDOR_SHIM_JS));
        assert!(body.contains("in_automation=true"));

        // Query strings (e.g. `?feature=bidi`) do not defeat the append.
        let (status, _, body) = http_get(&base, "/resources/testdriver.js?feature=bidi");
        assert_eq!(status, 200);
        assert!(
            String::from_utf8(body)
                .unwrap()
                .ends_with(crate::testdriver::VENDOR_SHIM_JS)
        );

        // Every other file is byte-identical.
        let (status, _, body) = http_get(&base, "/resources/other.js");
        assert_eq!(status, 200);
        assert_eq!(body, b"var x = 1;");

        // A missing testdriver.js still 404s (no synthetic shim).
        let (status, _, _) = http_get(&base, "/resources/nope-testdriver.js");
        assert_eq!(status, 404);

        shutdown.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn server_accepts_absolute_form_targets() {
        let root = scratch("absform");
        std::fs::write(root.join("page.html"), "<html>abs</html>").unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        // Absolute-form, as sent by the bowser fetch stack.
        let (status, _, body) = http_get(&base, &format!("{base}/page.html?x=1"));
        assert_eq!(status, 200);
        assert_eq!(body, b"<html>abs</html>");

        // No path at all reduces to "/" (404: no index.html here).
        let (status, _, _) = http_get(&base, &base);
        assert_eq!(status, 404);

        shutdown.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn strip_absolute_form_unit() {
        assert_eq!(strip_absolute_form("/a/b?c"), "/a/b?c");
        assert_eq!(strip_absolute_form("http://h:1/a?b"), "/a?b");
        assert_eq!(strip_absolute_form("https://h/a"), "/a");
        assert_eq!(strip_absolute_form("http://h"), "/");
        assert_eq!(strip_absolute_form("OPTIONS *"), "OPTIONS *");
    }

    #[test]
    fn connection_slots_cap_concurrency() {
        // The accept loop's admission control: slots are handed out up to
        // the cap, refused past it, and released on drop.
        let active = Arc::new(AtomicUsize::new(0));
        let mut slots = Vec::new();
        for _ in 0..8 {
            slots.push(try_acquire_slot(&active, 8).expect("slot under cap"));
        }
        assert_eq!(active.load(Ordering::SeqCst), 8);
        assert!(try_acquire_slot(&active, 8).is_none());
        assert_eq!(active.load(Ordering::SeqCst), 8);
        drop(slots.pop());
        assert_eq!(active.load(Ordering::SeqCst), 7);
        let slot = try_acquire_slot(&active, 8).expect("drop freed a slot");
        assert_eq!(active.load(Ordering::SeqCst), 8);
        drop(slot);
        assert_eq!(active.load(Ordering::SeqCst), 7);
    }

    #[test]
    fn run_config_wires_server_base_url() {
        let d = BowserDriver {
            exe: PathBuf::new(),
            server_base: "http://127.0.0.1:9".into(),
            render_timeout: Duration::from_secs(1),
            check_timeout: Duration::from_secs(1),
            _server: None,
            doc_root: PathBuf::new(),
        };
        let config = d.run_config();
        assert_eq!(config.server_base_url, "http://127.0.0.1:9");
        // Everything else stays at the default for the caller to fill in.
        let default = bowser_wpt::runner::RunConfig::default();
        assert_eq!(config.timeout, default.timeout);
        assert!(config.include.is_empty() && config.exclude.is_empty());
    }

    #[test]
    fn join_server_url_handles_both_forms() {
        assert_eq!(join_server_url("http://h:1", "/a"), "http://h:1/a");
        assert_eq!(join_server_url("http://h:1", "a"), "http://h:1/a");
        assert_eq!(join_server_url("http://h:1/", "/a"), "http://h:1/a");
        assert_eq!(join_server_url("http://h:1/", "a"), "http://h:1/a");
        assert_eq!(
            join_server_url("http://h:1", "sub/ref.html"),
            "http://h:1/sub/ref.html"
        );
    }

    #[test]
    fn resource_exists_maps_fixture_urls_to_docroot() {
        use bowser_wpt::driver::BrowserDriver;
        let root = scratch("resexists");
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/ref.html"), b"<html></html>").unwrap();
        let d = BowserDriver::new(PathBuf::from("bowser"), &root).unwrap();
        let base = d.server_base.clone();
        // Present file, with and without query string.
        assert!(d.resource_exists(&format!("{base}/sub/ref.html")));
        assert!(d.resource_exists(&format!("{base}/sub/ref.html?x=1")));
        // Missing file: pruned checkout, not an engine result.
        assert!(!d.resource_exists(&format!("{base}/sub/gone.html")));
        assert!(!d.resource_exists(&format!("{base}/nope.html")));
        // Traversal can never resolve: report missing, never escape.
        assert!(!d.resource_exists(&format!("{base}/../secret.txt")));
        assert!(!d.resource_exists(&format!("{base}/sub/%2e%2e/secret.txt")));
        // Remote references cannot be checked: assume servable.
        assert!(d.resource_exists("https://example.com/ref.html"));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The probe must mirror `read_doc_file`'s directory and
    /// wrapper-synthesis resolutions: servable references must not
    /// report as missing (or the pruned-file logic never fires for
    /// them), and unsynthesizable requests must report as missing.
    #[test]
    fn resource_exists_mirrors_server_resolutions() {
        use bowser_wpt::driver::BrowserDriver;
        let root = scratch("resexists-mirror");
        std::fs::create_dir_all(root.join("dir")).unwrap();
        std::fs::write(root.join("dir/index.html"), b"<html></html>").unwrap();
        std::fs::create_dir_all(root.join("emptydir")).unwrap();
        // Synthesized wrappers: only the `.js` sibling exists on disk.
        std::fs::write(
            root.join("t.any.js"),
            b"// META: title=x\ntest(()=>{}, 't');",
        )
        .unwrap();
        std::fs::write(root.join("w.window.js"), b"test(()=>{}, 'w');").unwrap();
        // `.any` wrapper whose `// META: global=` excludes `window`:
        // the server 404s the synthesis, so the probe must too.
        std::fs::write(
            root.join("nw.any.js"),
            b"// META: global=worker\ntest(()=>{}, 'nw');",
        )
        .unwrap();
        let d = BowserDriver::new(PathBuf::from("bowser"), &root).unwrap();
        let base = d.server_base.clone();
        // (a) Directory URLs serve `index.html`, with or without the
        // trailing slash (`resolve_reference_url` strips it).
        assert!(d.resource_exists(&format!("{base}/dir")));
        assert!(d.resource_exists(&format!("{base}/dir/")));
        // A directory with no `index.html` 404s on the server.
        assert!(!d.resource_exists(&format!("{base}/emptydir")));
        // (b) Synthesized wrappers resolve from the `.js` sibling.
        assert!(d.resource_exists(&format!("{base}/t.any.html")));
        assert!(d.resource_exists(&format!("{base}/w.window.html")));
        // No sibling on disk: nothing to synthesize.
        assert!(!d.resource_exists(&format!("{base}/missing.any.html")));
        // Exposure check: `global=worker` excludes window -> 404.
        assert!(!d.resource_exists(&format!("{base}/nw.any.html")));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A symlink inside the doc root pointing outside it is a 404 on
    /// the server (canonicalize prefix check in `read_doc_file`); the
    /// probe must not report the escaping target as servable, or the
    /// server's 404 page would reach the pixel comparison as a FAIL.
    #[cfg(unix)]
    #[test]
    fn resource_exists_rejects_escaping_symlink() {
        use bowser_wpt::driver::BrowserDriver;
        use std::os::unix::fs::symlink;
        let root = scratch("resexists-symlink");
        let outside = scratch("resexists-symlink-outside");
        std::fs::write(outside.join("secret.txt"), b"secret").unwrap();
        symlink(outside.join("secret.txt"), root.join("leak.txt")).unwrap();
        symlink(&outside, root.join("leakdir")).unwrap();
        let d = BowserDriver::new(PathBuf::from("bowser"), &root).unwrap();
        let base = d.server_base.clone();
        assert!(!d.resource_exists(&format!("{base}/leak.txt")));
        assert!(!d.resource_exists(&format!("{base}/leakdir/secret.txt")));
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    /// Manifest URLs may lack a leading slash (e.g. `sub/t.html`), in
    /// which case the directory `sub` also lacks one and the old
    /// `"{base}{dir}/{href}"` produced `http://host:portsub/ref.html`.
    #[test]
    fn reference_url_without_leading_slash_dir() {
        let d = BowserDriver {
            exe: PathBuf::new(),
            server_base: "http://127.0.0.1:9".into(),
            render_timeout: Duration::from_secs(1),
            check_timeout: Duration::from_secs(1),
            _server: None,
            doc_root: PathBuf::new(),
        };
        let t = item("sub/t.html", "sub/t.html");
        assert_eq!(
            d.reference_url(&t, "ref.html"),
            "http://127.0.0.1:9/sub/ref.html"
        );
        assert_eq!(d.test_url(&t), "http://127.0.0.1:9/sub/t.html");
        let t = item("/sub/t.html", "/sub/t.html");
        assert_eq!(
            d.reference_url(&t, "ref.html"),
            "http://127.0.0.1:9/sub/ref.html"
        );
        assert_eq!(
            d.reference_url(&t, "/ref.html"),
            "http://127.0.0.1:9/ref.html"
        );
        assert_eq!(d.test_url(&t), "http://127.0.0.1:9/sub/t.html");
        let t = item("t.html", "t.html");
        assert_eq!(
            d.reference_url(&t, "ref.html"),
            "http://127.0.0.1:9/ref.html"
        );
    }

    #[test]
    fn server_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;
        let root = scratch("symlinkesc");
        let outside = scratch("symlinkout");
        std::fs::write(outside.join("secret.txt"), "TOP SECRET").unwrap();
        // Symlink to a file outside the doc root...
        symlink(outside.join("secret.txt"), root.join("link.txt")).unwrap();
        // ...and to a directory outside the doc root.
        symlink(&outside, root.join("dirlink")).unwrap();

        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, _, body) = http_get(&base, "/link.txt");
        assert_eq!(status, 404);
        assert!(body.is_empty());
        let (status, _, _) = http_get(&base, "/dirlink/secret.txt");
        assert_eq!(status, 404);

        // A symlink that stays inside the doc root still works.
        std::fs::write(root.join("inside.txt"), "ok").unwrap();
        symlink(root.join("inside.txt"), root.join("oklink.txt")).unwrap();
        let (status, _, body) = http_get(&base, "/oklink.txt");
        assert_eq!(status, 200);
        assert_eq!(body, b"ok");

        shutdown.shutdown();
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn content_type_xhtml_is_xhtml() {
        assert_eq!(content_type(Path::new("a.xhtml")), "application/xhtml+xml");
        assert_eq!(content_type(Path::new("a.xml")), "application/xml");
    }

    #[test]
    fn server_404s_and_blocks_traversal() {
        let root = scratch("serve404");
        std::fs::write(root.join("secret.txt"), "top secret").unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, _, _) = http_get(&base, "/missing.html");
        assert_eq!(status, 404);

        // Traversal attempts never escape the doc root.
        for target in ["/../secret.txt", "/..%2fsecret.txt", "/%2e%2e/secret.txt"] {
            let (status, _, body) = http_get(&base, target);
            assert_eq!(status, 404, "target {target}");
            let needle = b"top secret";
            assert!(
                !body.windows(needle.len()).any(|w| w == needle),
                "target {target}"
            );
        }

        drop(shutdown);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn driver_new_spawns_server_and_names_itself() {
        let root = scratch("new");
        let d = BowserDriver::new(PathBuf::from("/nonexistent/bowser"), &root).unwrap();
        assert!(d.server_base.starts_with("http://127.0.0.1:"));
        assert_eq!(d.name(), "bowser");
        assert_eq!(d.render_timeout, DEFAULT_RENDER_TIMEOUT);
        assert_eq!(d.check_timeout, DEFAULT_CHECK_TIMEOUT);
        // The server is reachable through the driver's base URL.
        std::fs::write(root.join("x.txt"), "x").unwrap();
        let (status, _, _) = http_get(&d.server_base, "/x.txt");
        assert_eq!(status, 200);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Fake "browser": emits W*H*4 bytes cycling a 4-byte RGBA pattern,
    /// honouring the --viewport WxH argument like the real protocol.
    const PATTERN_EXE: &str = r#"#!/bin/sh
dims=""
prev=""
for a in "$@"; do
  case "$prev" in
    --viewport) dims="$a" ;;
  esac
  prev="$a"
done
w=${dims%x*}; h=${dims#*x}
n=$((w * h))
i=0
while [ "$i" -lt "$n" ]; do
  case $((i % 4)) in
    0) printf '\377\000\000\377' ;;
    1) printf '\000\377\000\377' ;;
    2) printf '\000\000\377\377' ;;
    3) printf '\377\377\377\377' ;;
  esac
  i=$((i + 1))
done
exit 0
"#;

    fn expected_pattern(w: u32, h: u32) -> Vec<u8> {
        let px: [[u8; 4]; 4] = [
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [255, 255, 255, 255],
        ];
        let mut out = Vec::with_capacity(w as usize * h as usize * 4);
        for i in 0..w as usize * h as usize {
            out.extend_from_slice(&px[i % 4]);
        }
        out
    }

    #[test]
    fn render_captures_rgba_from_child() {
        let dir = scratch("render");
        let exe = write_exe(&dir, "fake-bowser.sh", PATTERN_EXE);
        let d = driver_with(exe);

        let shot = d
            .render(
                "http://127.0.0.1:1/page.html",
                Viewport {
                    width: 4,
                    height: 2,
                },
            )
            .unwrap();
        assert_eq!(shot.width, 4);
        assert_eq!(shot.height, 2);
        assert_eq!(shot.rgba, expected_pattern(4, 2));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn render_rejects_short_output() {
        let dir = scratch("rendershort");
        let exe = write_exe(&dir, "short.sh", "#!/bin/sh\nprintf 'abc'\nexit 0\n");
        let d = driver_with(exe);

        let err = d
            .render(
                "http://x/y",
                Viewport {
                    width: 2,
                    height: 2,
                },
            )
            .unwrap_err();
        assert_eq!(
            err,
            DriverError::BadScreenshot {
                expected: 16,
                got: 3
            }
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn render_nonzero_exit_is_browser_gone() {
        let dir = scratch("renderexit");
        let exe = write_exe(
            &dir,
            "fail.sh",
            "#!/bin/sh\necho 'boom: no gpu' >&2\nexit 3\n",
        );
        let d = driver_with(exe);

        let err = d.render("http://x/y", Viewport::DEFAULT).unwrap_err();
        assert!(matches!(err, DriverError::BrowserGone(_)), "{err:?}");
        assert!(err.to_string().contains("boom: no gpu"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn render_timeout_kills_child() {
        let dir = scratch("rendertimeout");
        let exe = write_exe(&dir, "slow.sh", "#!/bin/sh\nsleep 30\nexit 0\n");
        let root = scratch("rendertimeout-root");
        let mut d = BowserDriver::new(exe, &root).unwrap();
        d.render_timeout = Duration::from_millis(200);

        let start = Instant::now();
        let err = d
            .render(
                "http://x/y",
                Viewport {
                    width: 1,
                    height: 1,
                },
            )
            .unwrap_err();
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "child not killed"
        );
        assert!(
            matches!(err, DriverError::Timeout { test: _, after: _ }),
            "{err:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_without_crash_distinguishes_crash_from_page_error() {
        let dir = scratch("check");
        let ok = write_exe(&dir, "ok.sh", "#!/bin/sh\nexit 0\n");
        let page_err = write_exe(&dir, "pageerr.sh", "#!/bin/sh\nexit 1\n");
        let crash = write_exe(&dir, "crash.sh", "#!/bin/sh\nkill -TERM $$\n");

        // Exit 0: clean load.
        driver_with(ok).load_without_crash("http://x/").unwrap();
        // Exit 1: page-level failure, browser alive.
        driver_with(page_err)
            .load_without_crash("http://x/")
            .unwrap();
        // Signal death: crash.
        let err = driver_with(crash)
            .load_without_crash("http://x/")
            .unwrap_err();
        assert!(matches!(err, DriverError::BrowserGone(_)), "{err:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_testharness_maps_exit_codes() {
        let dir = scratch("harness");
        let ok = write_exe(&dir, "ok.sh", "#!/bin/sh\nexit 0\n");
        let page_err = write_exe(&dir, "pageerr.sh", "#!/bin/sh\nexit 1\n");
        let broken = write_exe(&dir, "broken.sh", "#!/bin/sh\nexit 2\n");
        let crash = write_exe(&dir, "crash.sh", "#!/bin/sh\nkill -TERM $$\n");

        // Exit 0 with no report line -> clean load, subtests unknown
        // (the completion callback never fired). Honest: never Ok, never
        // green.
        let r = driver_with(ok)
            .run_testharness(&item("dom/a.html", "/dom/a.html"), Duration::from_secs(5))
            .unwrap();
        assert_eq!(r.test, "dom/a.html");
        assert_eq!(r.url.as_deref(), Some("/dom/a.html"));
        assert_eq!(r.status, HarnessStatus::SubtestsUnknown);
        assert!(r.subtests.is_empty());
        assert!(r.duration_ms.is_some());
        assert!(!r.is_green());

        // Exit 1 -> harness Error.
        let r = driver_with(page_err)
            .run_testharness(&item("dom/a.html", "/dom/a.html"), Duration::from_secs(5))
            .unwrap();
        assert_eq!(r.status, HarnessStatus::Error);
        assert_eq!(r.url.as_deref(), Some("/dom/a.html"));
        assert!(!r.is_green());

        // Exit >= 2 -> browser itself failed.
        let err = driver_with(broken)
            .run_testharness(&item("dom/a.html", "/dom/a.html"), Duration::from_secs(5))
            .unwrap_err();
        assert!(matches!(err, DriverError::BrowserGone(_)), "{err:?}");

        // Signal -> BrowserGone.
        let err = driver_with(crash)
            .run_testharness(&item("dom/a.html", "/dom/a.html"), Duration::from_secs(5))
            .unwrap_err();
        assert!(matches!(err, DriverError::BrowserGone(_)), "{err:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_testharness_timeout() {
        let dir = scratch("harnesstimeout");
        let exe = write_exe(&dir, "slow.sh", "#!/bin/sh\nsleep 30\nexit 0\n");
        let d = driver_with(exe);

        let err = d
            .run_testharness(
                &item("dom/slow.html", "/dom/slow.html"),
                Duration::from_millis(200),
            )
            .unwrap_err();
        assert_eq!(
            err,
            DriverError::Timeout {
                test: "dom/slow.html".into(),
                after: Duration::from_millis(200),
            }
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn harness_url_keeps_test_path_and_stamps_id() {
        let d = BowserDriver {
            exe: PathBuf::new(),
            server_base: "http://127.0.0.1:9".into(),
            render_timeout: Duration::from_secs(1),
            check_timeout: Duration::from_secs(1),
            _server: None,
            doc_root: PathBuf::new(),
        };
        // The test path keeps its directory structure under the harness
        // prefix so relative subresources resolve; the id is stamped for
        // report attribution and the nonce lets the driver accept only
        // this run's report line.
        let t = item("dom/a.html", "/dom/a.html");
        assert_eq!(
            d.harness_url(&t, "abc123"),
            "http://127.0.0.1:9/__bowser_harness__/dom/a.html?__bowser_harness=1&id=dom%2Fa.html&nonce=abc123"
        );
        // A manifest url without a leading slash still lands under the
        // prefix with its structure intact.
        let t = item("sub/t.html", "sub/t.html");
        assert_eq!(
            d.harness_url(&t, "abc123"),
            "http://127.0.0.1:9/__bowser_harness__/sub/t.html?__bowser_harness=1&id=sub%2Ft.html&nonce=abc123"
        );
        // The test's own query string is preserved ahead of the harness
        // parameters.
        let t = item("q.html", "/q.html?foo=bar");
        assert_eq!(
            d.harness_url(&t, "abc123"),
            "http://127.0.0.1:9/__bowser_harness__/q.html?foo=bar&__bowser_harness=1&id=q.html&nonce=abc123"
        );
    }

    #[test]
    fn last_report_line_takes_the_last_marker_line() {
        let nonce = "abc123";
        assert_eq!(last_report_line(b"noise\n", nonce), None);
        assert_eq!(
            last_report_line(b"BOWSER_HARNESS_REPORT:abc123:{\"a\":1}\n", nonce),
            Some("{\"a\":1}".to_string())
        );
        let out =
            b"log line\nBOWSER_HARNESS_REPORT:abc123:first\nBOWSER_HARNESS_REPORT:abc123:second\n";
        assert_eq!(last_report_line(out, nonce), Some("second".to_string()));
        // A marker mid-line is not a report line.
        assert_eq!(
            last_report_line(b"xx BOWSER_HARNESS_REPORT:abc123:{\"a\":1}", nonce),
            None
        );
        assert_eq!(last_report_line(b"", nonce), None);
    }

    #[test]
    fn last_report_line_ignores_lines_without_the_nonce() {
        let nonce = "abc123";
        // A forged report line without the nonce — e.g. from a page that
        // registered its own completion callback or used a load handler
        // — is ignored, even when it is the last marker line.
        let forged = b"BOWSER_HARNESS_REPORT:{\"evil\":true}\n";
        assert_eq!(last_report_line(forged, nonce), None);
        // A line with the wrong nonce is ignored too.
        let wrong = b"BOWSER_HARNESS_REPORT:deadbeef:{\"evil\":true}\n";
        assert_eq!(last_report_line(wrong, nonce), None);
        // Last-wins applies only within this run's nonce: a later
        // forgery cannot displace the real report.
        let out = b"BOWSER_HARNESS_REPORT:abc123:{\"real\":true}\n\
            BOWSER_HARNESS_REPORT:{\"evil\":true}\n";
        assert_eq!(
            last_report_line(out, nonce),
            Some("{\"real\":true}".to_string())
        );
        // Only forgeries: no report.
        let out = b"BOWSER_HARNESS_REPORT:{\"evil\":true}\n\
            BOWSER_HARNESS_REPORT:deadbeef:{\"evil\":true}\n";
        assert_eq!(last_report_line(out, nonce), None);
    }

    #[test]
    fn report_nonce_is_hex_and_unique_per_call() {
        let a = new_report_nonce();
        let b = new_report_nonce();
        assert_eq!(a.len(), 32);
        assert!(a.bytes().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b, "two runs must not share a nonce");
    }

    #[test]
    fn drain_caps_output_at_16mib_and_flags_truncation() {
        // >16 MiB synthetic output with marker lines before and after
        // the truncation point.
        let mut big = Vec::new();
        big.extend_from_slice(b"BOWSER_HARNESS_START:\"before\"\n");
        big.extend(std::iter::repeat_n(b'x', MAX_DRAIN_BYTES));
        big.extend_from_slice(
            b"BOWSER_HARNESS_RESULT:{\"name\":\"after\",\"status\":\"PASS\",\"message\":null}\n",
        );
        let d = drain(Some(&big[..]));
        assert_eq!(d.bytes.len(), MAX_DRAIN_BYTES);
        assert!(d.truncated);
        // Marker parsing runs over the capped buffer only: the line
        // before the cut parses, the one past it is lost, and the
        // truncation is flagged so the run is never silently reported
        // as clean.
        let m = TestharnessMarkers::parse_capped(&d.bytes, d.truncated);
        assert!(m.truncated);
        assert_eq!(m.started, vec!["before".to_string()]);
        assert!(m.results.is_empty());
    }

    #[test]
    fn drain_passes_small_output_through_unflagged() {
        let d = drain(Some(&b"hello\n"[..]));
        assert_eq!(d.bytes, b"hello\n");
        assert!(!d.truncated);
        let d = drain(None::<&[u8]>);
        assert!(d.bytes.is_empty());
        assert!(!d.truncated);
    }

    #[test]
    fn harness_markers_parse_marker_lines() {
        use bowser_wpt::results::SubtestStatus;
        let out = b"noise\n\
            BOWSER_HARNESS_START:\"alpha\"\n\
            BOWSER_HARNESS_START:\"be\"ta\"\n\
            BOWSER_HARNESS_START:42\n\
            BOWSER_HARNESS_RESULT:{\"name\":\"alpha\",\"status\":\"PASS\",\"message\":null}\n\
            BOWSER_HARNESS_RESULT:{\"name\":\"beta\",\"status\":\"FAIL\",\"message\":\"boom\"}\n\
            BOWSER_HARNESS_RESULT:not json\n\
            BOWSER_HARNESS_REPORT:{\"test\":\"x\",\"harness_status\":\"OK\",\"tests\":[]}\n";
        let m = TestharnessMarkers::parse(out);
        // Well-formed start lines only; the malformed payloads are
        // skipped, and the report line is not a start line.
        assert_eq!(m.started, vec!["alpha".to_string()]);
        assert_eq!(m.results.len(), 2);
        assert_eq!(m.results[0].name, "alpha");
        assert_eq!(m.results[0].status, SubtestStatus::Pass);
        assert_eq!(m.results[0].message, None);
        assert_eq!(m.results[1].name, "beta");
        assert_eq!(m.results[1].status, SubtestStatus::Fail);
        assert_eq!(m.results[1].message.as_deref(), Some("boom"));

        // A marker mid-line is not a marker line.
        let m = TestharnessMarkers::parse(b"xx BOWSER_HARNESS_START:\"alpha\"");
        assert!(m.started.is_empty());
        assert!(m.results.is_empty());
        // Empty stdout: no markers.
        let m = TestharnessMarkers::parse(b"");
        assert!(m.started.is_empty());
        assert!(m.results.is_empty());
        // One line yields at most one entry: a crafted payload cannot
        // smuggle extra subtests past the scanner.
        let m = TestharnessMarkers::parse(
            b"BOWSER_HARNESS_RESULT:{\"name\":\"a\",\"status\":\"PASS\"},{\"name\":\"b\",\"status\":\"FAIL\"}\n",
        );
        assert_eq!(m.results.len(), 1);
        assert_eq!(m.results[0].name, "a");
        let m = TestharnessMarkers::parse(b"BOWSER_HARNESS_START:\"a\",\"b\"\n");
        assert!(m.started.is_empty());
    }

    #[test]
    fn json_escape_never_breaks_out_of_script() {
        assert_eq!(json_escape("abc"), "\"abc\"");
        assert_eq!(json_escape("a\"b\\c"), "\"a\\\"b\\\\c\"");
        assert_eq!(json_escape("a\x01b\n"), "\"a\\u0001b\\n\"");
        // `</script>` in an id cannot terminate the script element.
        let evil = "</script><script>alert(1)</script>";
        let escaped = json_escape(evil);
        assert!(!escaped.contains('<'));
        assert!(!escaped.contains("</script>"));
        assert_eq!(
            escaped,
            "\"\\u003c/script\\u003e\\u003cscript\\u003ealert(1)\\u003c/script\\u003e\""
        );
    }

    #[test]
    fn inject_reporter_goes_before_body_close() {
        let html = "<html><head><script src=\"/resources/testharness.js\"></script></head>\
            <body><script>done();</script></body></html>";
        let page = inject_reporter(html, "dom/a.html", "testnonce");
        let script_pos = page
            .find("<script>(function(){")
            .expect("reporter injected");
        assert!(script_pos < page.find("</body>").expect("body close kept"));
        assert!(page.contains(r#"var TEST_ID = "dom/a.html";"#));
        assert!(page.contains(r#"var NONCE = "testnonce";"#));
        assert!(page.contains("BOWSER_HARNESS_REPORT:"));
        // Original page content intact, in order.
        assert!(page.contains("<script>done();</script>"));

        // Case-insensitive `</BODY>`.
        let page = inject_reporter("<HTML><BODY>hi</BODY></HTML>", "x", "testnonce");
        assert!(page.find("<script>(function(){").unwrap() < page.find("</BODY>").unwrap());

        // No body/html close: appended at the end.
        let page = inject_reporter("<html><p>hi</p>", "x", "testnonce");
        assert!(page.ends_with("</script>"));
        assert!(page.starts_with("<html><p>hi</p>"));

        // A hostile id is escaped, not spliced raw.
        let page = inject_reporter(
            "<html><body></body></html>",
            "</script><script>alert(1)",
            "testnonce",
        );
        assert!(!page.contains("</script><script>alert(1)"));
    }

    #[test]
    fn inject_reporter_follows_testharness_script_tag() {
        // `setup({output:false})` only takes effect before the first
        // `test()` call, so the reporter must land right after
        // `testharness.js` loads, not at end of body.
        let html = "<html><head><script src=\"/resources/testharness.js\"></script></head>\
            <body><script>test(function(){}, \"t\"); done();</script></body></html>";
        let page = inject_reporter(html, "dom/a.html", "testnonce");
        let harness_close = page.find("testharness.js").unwrap();
        let reporter_pos = page
            .find("<script>(function(){")
            .expect("reporter injected");
        let body_test = page.find("test(function(){}").expect("page test kept");
        assert!(
            reporter_pos > harness_close,
            "reporter after testharness.js"
        );
        assert!(reporter_pos < body_test, "reporter before first test()");
        assert!(page.contains("setup({output:false})"));
    }

    #[test]
    fn inject_reporter_registers_start_and_result_callbacks() {
        let html = "<html><head><script src=\"/resources/testharness.js\"></script></head>\
            <body><script>test(function(){}, \"t\"); done();</script></body></html>";
        let page = inject_reporter(html, "dom/a.html", "testnonce");
        // The reporter registers all three harness callbacks...
        assert!(page.contains("add_start_callback"));
        assert!(page.contains("add_result_callback"));
        assert!(page.contains("add_completion_callback"));
        // ...and prints all three marker kinds.
        assert!(page.contains("BOWSER_HARNESS_START:"));
        assert!(page.contains("BOWSER_HARNESS_RESULT:"));
        assert!(page.contains("BOWSER_HARNESS_REPORT:"));
        // ...and arms/sets the headless pump's completion flag (the
        // placeholder is substituted with the shared flag name, not left
        // as literal text).
        assert!(page.contains(crate::script::HARNESS_COMPLETION_FLAG));
        assert!(!page.contains("__BOWSER_COMPLETION_FLAG__"));
    }

    #[test]
    fn inject_reporter_bytes_pipes_non_utf8() {
        // A big5-encoded page (invalid UTF-8): the raw bytes pass through
        // with the reporter spliced at the ASCII markers, never splitting
        // a multibyte sequence.
        let mut page: Vec<u8> =
            b"<html><head><script src=\"/resources/testharness.js\"></script></head><body>"
                .to_vec();
        page.extend_from_slice(&[0xA4, 0xA4, 0xA4, 0xE5]); // big5 bytes (invalid UTF-8)
        page.extend_from_slice(b"</body></html>");
        assert!(String::from_utf8(page.clone()).is_err());
        let out = inject_reporter_bytes(&page, "encoding/big5-encoder.html", "testnonce");
        // Big5 payload intact and in order.
        let payload_pos = out
            .windows(4)
            .position(|w| w == [0xA4, 0xA4, 0xA4, 0xE5])
            .expect("payload preserved");
        // Reporter present, right after the testharness script tag.
        let reporter_pos = out
            .windows(b"<script>(function(){".len())
            .position(|w| w == b"<script>(function(){")
            .expect("reporter injected");
        let harness_pos = out
            .windows(b"testharness.js".len())
            .position(|w| w == b"testharness.js")
            .unwrap();
        assert!(reporter_pos > harness_pos);
        assert!(payload_pos > reporter_pos, "reporter before payload");
        // String path agrees with the byte path on UTF-8 input.
        let html = "<html><body>hi</body></html>";
        assert_eq!(
            inject_reporter_bytes(html.as_bytes(), "x", "n"),
            inject_reporter(html, "x", "n").into_bytes()
        );
    }

    #[test]
    fn server_harness_injects_reporter() {
        let root = scratch("harness-inject");
        std::fs::create_dir_all(root.join("dom")).unwrap();
        let test_html = "<!doctype html><html><head>\
            <script src=\"/resources/testharness.js\"></script></head>\
            <body><script>test(function(){assert_true(true);},\"smoke\");done();</script></body></html>";
        std::fs::write(root.join("dom/a.html"), test_html).unwrap();
        std::fs::write(root.join("dom/helper.js"), "/* helper */").unwrap();

        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        // Injection request: 200, HTML, reporter before </body>, stamped
        // with the manifest id and the run nonce, original content intact.
        let (status, headers, body) = http_get(
            &base,
            &format!(
                "{base}/__bowser_harness__/dom/a.html?__bowser_harness=1&id=dom%2Fa.html&nonce=testnonce123"
            ),
        );
        assert_eq!(status, 200);
        assert_eq!(header(&headers, "content-type"), "text/html");
        let body = String::from_utf8(body).unwrap();
        let marker = body
            .find("BOWSER_HARNESS_REPORT:")
            .expect("reporter present");
        assert!(marker < body.find("</body>").unwrap());
        assert!(body.contains(r#"var TEST_ID = "dom/a.html";"#));
        assert!(body.contains(r#"var NONCE = "testnonce123";"#));
        assert!(body.contains("assert_true(true)"));

        // Same path without the inject flag: raw file, no reporter.
        let (status, _, body) = http_get(&base, &format!("{base}/__bowser_harness__/dom/a.html"));
        assert_eq!(status, 200);
        let body = String::from_utf8(body).unwrap();
        assert!(!body.contains("BOWSER_HARNESS_REPORT:"));
        assert!(body.contains("assert_true(true)"));

        // Relative subresource under the prefix: raw, correct type.
        let (status, headers, body) =
            http_get(&base, &format!("{base}/__bowser_harness__/dom/helper.js"));
        assert_eq!(status, 200);
        assert_eq!(
            header(&headers, "content-type"),
            "text/javascript; charset=utf-8"
        );
        assert_eq!(body, b"/* helper */");

        shutdown.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn server_harness_rejects_bad_requests() {
        let root = scratch("harness-bad");
        std::fs::write(root.join("secret.txt"), "top secret").unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        // Traversal under the prefix stays inside the doc root.
        let (status, _, body) = http_get(
            &base,
            &format!("{base}/__bowser_harness__/%2e%2e/secret.txt"),
        );
        assert_eq!(status, 404);
        assert!(
            !body.windows(10).any(|w| w == b"top secret"),
            "traversal leaked"
        );

        // Inject flag with a valid nonce but no id: 400 (the id is
        // stamped into the report, so it is required).
        let (status, _, _) = http_get(
            &base,
            &format!("{base}/__bowser_harness__/secret.txt?__bowser_harness=1&nonce=abc123"),
        );
        assert_eq!(status, 400);

        // Inject flag with an id but no nonce: 400. The nonce gate runs
        // before the file lookup, so a nonceless request cannot learn
        // whether the path exists.
        let (status, _, _) = http_get(
            &base,
            &format!("{base}/__bowser_harness__/nope.html?__bowser_harness=1&id=x"),
        );
        assert_eq!(status, 400);

        // Missing file with a valid nonce: 404 (file lookup runs last).
        let (status, _, _) = http_get(
            &base,
            &format!("{base}/__bowser_harness__/nope.html?__bowser_harness=1&id=x&nonce=abc123"),
        );
        assert_eq!(status, 404);

        shutdown.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Fake "browser" for harness runs: requires a `__bowser_harness__`
    /// URL among its args (exit 2 otherwise), then prints one report line
    /// to stdout and exits 0 — simulating a child whose page ran the
    /// injected reporter to completion. The report line carries this
    /// run's nonce, pulled from the `nonce=` query param of the harness
    /// URL in its args, mirroring the injected reporter: the driver only
    /// accepts nonce-bearing report lines.
    fn report_exe(report_json: &str) -> String {
        format!(
            "#!/bin/sh\n\
             url=\"\"\n\
             for a in \"$@\"; do\n\
             case \"$a\" in *__bowser_harness__*) url=\"$a\" ;; esac\n\
             done\n\
             if [ -z \"$url\" ]; then echo \"no harness url\" >&2; exit 2; fi\n\
             nonce=\"${{url##*nonce=}}\"\n\
             nonce=\"${{nonce%%&*}}\"\n\
             if [ -z \"$nonce\" ]; then echo \"no nonce\" >&2; exit 2; fi\n\
             printf 'BOWSER_HARNESS_REPORT:%s:%s\\n' \"$nonce\" '{report_json}'\n\
             exit 0\n"
        )
    }

    #[test]
    fn run_testharness_extracts_real_report() {
        use bowser_wpt::results::SubtestStatus;
        let dir = scratch("harness-report");
        let exe = write_exe(
            &dir,
            "report.sh",
            &report_exe(
                r#"{"test":"dom/a.html","harness_status":"OK","tests":[{"name":"smoke","status":"PASS","message":null},{"name":"broken","status":"FAIL","message":"expected true"}]}"#,
            ),
        );
        let d = driver_with(exe);

        let r = d
            .run_testharness(&item("dom/a.html", "/dom/a.html"), Duration::from_secs(5))
            .unwrap();
        assert_eq!(r.test, "dom/a.html");
        // Attribution: the schema carries no url; the driver fills it
        // from the TestItem it ran.
        assert_eq!(r.url.as_deref(), Some("/dom/a.html"));
        assert_eq!(r.status, HarnessStatus::Ok);
        assert_eq!(r.subtests.len(), 2);
        assert_eq!(r.subtests[0].name, "smoke");
        assert_eq!(r.subtests[0].status, SubtestStatus::Pass);
        assert_eq!(r.subtests[0].message, None);
        assert_eq!(r.subtests[1].name, "broken");
        assert_eq!(r.subtests[1].status, SubtestStatus::Fail);
        assert_eq!(r.subtests[1].message.as_deref(), Some("expected true"));
        assert_eq!(r.passed(), 1);
        assert_eq!(r.failed(), 1);
        assert!(!r.is_green());
        assert!(r.duration_ms.is_some());

        // All-pass report is green: a passing page can now report a pass,
        // which v1 could never do.
        let exe = write_exe(
            &dir,
            "green.sh",
            &report_exe(
                r#"{"test":"dom/a.html","harness_status":"OK","tests":[{"name":"smoke","status":"PASS"}]}"#,
            ),
        );
        let r = driver_with(exe)
            .run_testharness(&item("dom/a.html", "/dom/a.html"), Duration::from_secs(5))
            .unwrap();
        assert_eq!(r.status, HarnessStatus::Ok);
        assert!(r.is_green());

        // A harness TIMEOUT maps through, not to a pass.
        let exe = write_exe(
            &dir,
            "timeout.sh",
            &report_exe(
                r#"{"test":"dom/a.html","harness_status":"TIMEOUT","tests":[{"name":"slow","status":"TIMEOUT"}]}"#,
            ),
        );
        let r = driver_with(exe)
            .run_testharness(&item("dom/a.html", "/dom/a.html"), Duration::from_secs(5))
            .unwrap();
        assert_eq!(r.status, HarnessStatus::Timeout);
        assert_eq!(r.subtests[0].status, SubtestStatus::Timeout);
        assert!(!r.is_green());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_testharness_with_markers_parses_start_and_result_lines() {
        use bowser_wpt::results::SubtestStatus;
        let dir = scratch("harness-markers");
        // Fake child printing start/result markers around its report
        // line, the way the injected reporter does; one malformed start
        // line exercises the skip path. The report line carries this
        // run's nonce, taken from the `nonce=` query param of the
        // harness URL in its args (mirroring `report_exe`); the start
        // and result markers are informational and stay nonce-free.
        let exe = write_exe(
            &dir,
            "markers.sh",
            "#!/bin/sh\n\
             url=\"\"\n\
             for a in \"$@\"; do\n\
             case \"$a\" in *__bowser_harness__*) url=\"$a\" ;; esac\n\
             done\n\
             nonce=\"${url##*nonce=}\"\n\
             nonce=\"${nonce%%&*}\"\n\
             printf 'BOWSER_HARNESS_START:\"smoke\"\\n'\n\
             printf 'BOWSER_HARNESS_RESULT:{\"name\":\"smoke\",\"status\":\"PASS\",\"message\":null}\\n'\n\
             printf 'BOWSER_HARNESS_START:\"bro\"ken\"\\n'\n\
             printf 'BOWSER_HARNESS_START:\"broken\"\\n'\n\
             printf 'BOWSER_HARNESS_RESULT:{\"name\":\"broken\",\"status\":\"FAIL\",\"message\":\"expected true\"}\\n'\n\
             printf 'BOWSER_HARNESS_REPORT:%s:%s\\n' \"$nonce\" '{\"test\":\"dom/a.html\",\"harness_status\":\"OK\",\"tests\":[{\"name\":\"smoke\",\"status\":\"PASS\"},{\"name\":\"broken\",\"status\":\"FAIL\",\"message\":\"expected true\"}]}'\n\
             exit 0\n",
        );
        let (report, markers) = driver_with(exe)
            .run_testharness_with_markers(
                &item("dom/a.html", "/dom/a.html"),
                Duration::from_secs(5),
            )
            .unwrap();
        // The completion report is unchanged and authoritative.
        assert_eq!(report.test, "dom/a.html");
        assert_eq!(report.status, HarnessStatus::Ok);
        assert_eq!(report.subtests.len(), 2);
        // Markers: one start and one result per subtest, in order; the
        // malformed start line was skipped.
        assert_eq!(
            markers.started,
            vec!["smoke".to_string(), "broken".to_string()]
        );
        assert_eq!(markers.results.len(), 2);
        assert_eq!(markers.results[0].name, "smoke");
        assert_eq!(markers.results[0].status, SubtestStatus::Pass);
        assert_eq!(markers.results[0].message, None);
        assert_eq!(markers.results[1].name, "broken");
        assert_eq!(markers.results[1].status, SubtestStatus::Fail);
        assert_eq!(markers.results[1].message.as_deref(), Some("expected true"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_testharness_rejects_misattributed_report() {
        let dir = scratch("harness-misattr");
        let exe = write_exe(
            &dir,
            "report.sh",
            &report_exe(r#"{"test":"dom/other.html","harness_status":"OK","tests":[]}"#),
        );
        let err = driver_with(exe)
            .run_testharness(&item("dom/a.html", "/dom/a.html"), Duration::from_secs(5))
            .unwrap_err();
        assert!(matches!(err, DriverError::BadReport(_)), "{err:?}");
        assert!(err.to_string().contains("dom/other.html"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_testharness_rejects_malformed_report() {
        let dir = scratch("harness-malformed");
        let exe = write_exe(&dir, "report.sh", &report_exe("this is not json"));
        let err = driver_with(exe)
            .run_testharness(&item("dom/a.html", "/dom/a.html"), Duration::from_secs(5))
            .unwrap_err();
        assert!(matches!(err, DriverError::BadReport(_)), "{err:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Minimal testharness.js surface for reporter unit tests: `test`,
    /// `async_test` (+ `step`/`done`), `assert_true`, `assert_equals`,
    /// `done`, `add_completion_callback`, `add_start_callback`,
    /// `add_result_callback`, and the numeric statuses the
    /// real harness uses (subtest PASS=0, FAIL=1, TIMEOUT=2, NOTRUN=3;
    /// harness OK=0, ERROR=1, TIMEOUT=2, PRECONDITION_FAILED=3).
    ///
    /// The shim is two pieces: [`SYNTHETIC_TESTHARNESS_BASE_JS`] models a
    /// harness that predates `add_start_callback`/`add_result_callback`
    /// (its `__th_fire_start`/`__th_fire_result` are no-ops), and
    /// [`SYNTHETIC_TESTHARNESS_CALLBACKS_JS`] adds them, firing start
    /// callbacks at test start and result callbacks exactly once at test
    /// completion, like the real testharness.js.
    /// [`full_synthetic_harness()`] concatenates both pieces for the
    /// tests that need the full surface; the base alone exercises the
    /// reporter's guards for harnesses without the newer callbacks.
    ///
    /// The reporter under test maps these numbers to status strings, so
    /// exercising it against this shim exercises the same mapping it
    /// performs against the real testharness.js. `force_timeout` and
    /// `__th_harness_timeout` are deterministic stand-ins for the real
    /// harness's internal timers: the VM's timers run on a wall clock, so
    /// a real timeout would make the tests slow and flaky. They mark the
    /// test/harness TIMEOUT exactly as the harness timer would.
    ///
    /// Everything is installed by plain global assignment (`test =
    /// function ...`, never `function test ...`): one `Vm::eval` is one
    /// script scope in this engine, and declarative bindings (`var`,
    /// `function`) do not survive into the next eval — only sloppy-mode
    /// implicit globals land on the shared global object. Separate evals
    /// sharing globals is also exactly how the production page works
    /// (one `<script>` per testharness.js / reporter / test), so the
    /// shape matches deployment.
    const SYNTHETIC_TESTHARNESS_BASE_JS: &str = r#"
__th_tests = [];
__th_callbacks = [];
__th_finished = false;

// Start/result dispatch: no-ops in the base (a harness predating the
// callbacks), overridden by SYNTHETIC_TESTHARNESS_CALLBACKS_JS.
__th_fire_start = function (t) {};
__th_fire_result = function (t) {};

__th_Test = function (name) {
    this.name = name;
    this.status = 3; // NOTRUN until the body runs / the test completes
    this.message = null;
};

__th_record = function (t, fn) {
    try {
        fn();
    } catch (e) {
        t.status = 1;
        t.message = (e && e.message) ? String(e.message) : String(e);
    }
};

test = function (fn, name) {
    var t = new __th_Test(name);
    __th_tests.push(t);
    __th_fire_start(t);
    __th_record(t, fn);
    if (t.status === 3) {
        t.status = 0; // body ran clean: PASS
    }
    __th_fire_result(t);
    return t;
};

async_test = function (name) {
    var t = new __th_Test(name);
    __th_tests.push(t);
    __th_fire_start(t);
    t.step = function (fn) { __th_record(t, fn); };
    t.done = function () {
        if (t.status === 3) {
            t.status = 0;
        }
        __th_fire_result(t);
    };
    t.force_timeout = function (message) {
        t.status = 2;
        t.message = message || "timed out";
        __th_fire_result(t);
    };
    return t;
};

assert_true = function (v, message) {
    if (!v) {
        throw { message: message || "assert_true failed" };
    }
};

assert_equals = function (actual, expected, message) {
    if (actual !== expected) {
        throw {
            message: (message || "assert_equals failed") +
                ": expected " + String(expected) +
                " but got " + String(actual)
        };
    }
};

__th_finish = function (harness_status) {
    if (__th_finished) {
        return;
    }
    __th_finished = true;
    for (var i = 0; i < __th_callbacks.length; i++) {
        __th_callbacks[i](__th_tests, harness_status);
    }
};

done = function () {
    __th_finish(0); // harness OK
};

__th_harness_timeout = function () {
    __th_finish(2); // harness TIMEOUT
};

add_completion_callback = function (cb) {
    __th_callbacks.push(cb);
};
"#;

    /// `add_start_callback` / `add_result_callback` for the synthetic
    /// harness: start callbacks fire when a test is created, result
    /// callbacks fire exactly once when the test reaches a terminal
    /// status — the real testharness.js semantics the reporter relies on.
    const SYNTHETIC_TESTHARNESS_CALLBACKS_JS: &str = r#"
__th_start_callbacks = [];
__th_result_callbacks = [];

__th_fire_start = function (t) {
    for (var i = 0; i < __th_start_callbacks.length; i++) {
        __th_start_callbacks[i](t);
    }
};

__th_fire_result = function (t) {
    if (t.__th_result_fired) {
        return;
    }
    t.__th_result_fired = true;
    for (var i = 0; i < __th_result_callbacks.length; i++) {
        __th_result_callbacks[i](t);
    }
};

add_start_callback = function (cb) {
    __th_start_callbacks.push(cb);
};

add_result_callback = function (cb) {
    __th_result_callbacks.push(cb);
};
"#;

    /// Full synthetic harness: base plus start/result callbacks. Built
    /// at runtime (`concat!` only takes literals); the pieces are `&str`
    /// constants so the base alone is usable too.
    fn full_synthetic_harness() -> String {
        format!("{SYNTHETIC_TESTHARNESS_BASE_JS}{SYNTHETIC_TESTHARNESS_CALLBACKS_JS}")
    }

    /// Fixed nonce for synthetic reporter runs (see [`new_report_nonce`]
    /// for the real per-run generator).
    const TEST_NONCE: &str = "0123456789abcdef0123456789abcdef";

    /// Run a synthetic testharness page (`page_js`, the body of a
    /// `<script>` in a testharness test) against the full synthetic
    /// harness ([`full_synthetic_harness()`]) and the real injected
    /// [`REPORTER_JS`], returning the parsed completion report plus every
    /// line the reporter printed (start/result markers included).
    ///
    /// `console.log` is stubbed to accumulate lines instead of writing
    /// to the test runner's stdout: the reporter looks `console.log` up
    /// dynamically, so a plain assignment intercepts it. The accumulated
    /// text is read back through [`Vm::get_global`]; the report is the
    /// last [`REPORT_MARKER`] line carrying the run's nonce, mirroring
    /// [`last_report_line`], and is parsed with [`HarnessResult::from_json`],
    /// the same schema the driver's `run_testharness` accepts.
    ///
    /// The stub uses a sloppy-mode implicit global (`__captured = ...`,
    /// not `var`): like the shim above, only implicit globals survive
    /// across this engine's per-`eval` script scopes onto the shared
    /// global object.
    fn run_synthetic_page_lines(page_js: &str, test_id: &str) -> (HarnessResult, Vec<String>) {
        run_synthetic_page_lines_with(page_js, test_id, &full_synthetic_harness())
    }

    /// [`run_synthetic_page_lines`] against an explicit harness shim:
    /// [`SYNTHETIC_TESTHARNESS_BASE_JS`] alone models a harness without
    /// `add_start_callback`/`add_result_callback`.
    fn run_synthetic_page_lines_with(
        page_js: &str,
        test_id: &str,
        shim_js: &str,
    ) -> (HarnessResult, Vec<String>) {
        let mut vm = Vm::new();
        vm.eval(shim_js).expect("shim is valid JS");
        vm.eval("__captured = \"\"; console.log = function (s) { __captured = __captured + String(s) + \"\\n\"; };")
            .expect("capture stub is valid JS");
        let reporter = REPORTER_JS
            .replace("__BOWSER_TEST_ID__", &json_escape(test_id))
            .replace("__BOWSER_NONCE__", &json_escape(TEST_NONCE))
            .replace(
                "__BOWSER_COMPLETION_FLAG__",
                &json_escape(crate::script::HARNESS_COMPLETION_FLAG),
            );
        vm.eval(&reporter).expect("reporter is valid JS");
        vm.eval(page_js).expect("page is valid JS");
        let captured = match vm.get_global("__captured") {
            Some(Value::Str(s)) => s.as_wtf8().to_string(),
            other => panic!("reporter printed no output (captured: {other:?})"),
        };
        let lines: Vec<String> = captured.lines().map(str::to_owned).collect();
        let prefix = format!("{REPORT_MARKER}{TEST_NONCE}:");
        let json = lines
            .iter()
            .rev()
            .find_map(|line| line.strip_prefix(&prefix))
            .unwrap_or_else(|| panic!("report line lacks nonce marker: {captured:?}"));
        (
            HarnessResult::from_json(json).expect("reporter output parses"),
            lines,
        )
    }

    fn run_synthetic_page(page_js: &str, test_id: &str) -> HarnessResult {
        run_synthetic_page_lines(page_js, test_id).0
    }

    #[test]
    fn synthetic_page_all_pass() {
        use bowser_wpt::results::SubtestStatus;
        let r = run_synthetic_page(
            r#"test(function () { assert_true(1 + 1 === 2); }, "arithmetic");
               test(function () { assert_equals("a" + "b", "ab"); }, "concat");
               done();"#,
            "synth/all-pass.html",
        );
        assert_eq!(r.test, "synth/all-pass.html");
        assert_eq!(r.status, HarnessStatus::Ok);
        assert_eq!(r.subtests.len(), 2);
        assert_eq!(r.subtests[0].name, "arithmetic");
        assert_eq!(r.subtests[1].name, "concat");
        assert!(r.subtests.iter().all(|t| t.status == SubtestStatus::Pass));
        assert!(r.subtests.iter().all(|t| t.message.is_none()));
        assert!(r.is_green());
    }

    #[test]
    fn synthetic_page_harness_status_object_unwraps() {
        // The real testharness.js passes a TestsStatus *object* (numeric
        // code on `.status`) to the completion callback; the reporter
        // must unwrap it before mapping. A bare-number-only mapping
        // would degrade this to UNKNOWN and fail is_green().
        let r = run_synthetic_page(
            r#"test(function () { assert_true(true); }, "ok");
               __th_finish({status: 0, message: null});"#,
            "synth/status-object.html",
        );
        assert_eq!(r.status, HarnessStatus::Ok);
        assert_eq!(r.message, None);
        assert!(r.is_green());
    }

    #[test]
    fn synthetic_page_harness_error_carries_message() {
        // A harness-level failure: the TestsStatus message rides on the
        // report's `message` field and the run is not green.
        let r = run_synthetic_page(
            r#"test(function () { assert_true(true); }, "ok");
               __th_finish({status: 1, message: "setup blew up"});"#,
            "synth/status-error.html",
        );
        assert_eq!(r.status, HarnessStatus::Error);
        assert_eq!(r.message.as_deref(), Some("setup blew up"));
        assert!(!r.is_green());
    }

    #[test]
    fn synthetic_page_one_failing_subtest() {
        use bowser_wpt::results::SubtestStatus;
        let r = run_synthetic_page(
            r#"test(function () { assert_true(true); }, "passes");
               test(function () { assert_equals(1, 2, "math is broken"); }, "fails");
               done();"#,
            "synth/one-fail.html",
        );
        assert_eq!(r.status, HarnessStatus::Ok);
        assert_eq!(r.subtests.len(), 2);
        assert_eq!(r.subtests[0].status, SubtestStatus::Pass);
        assert_eq!(r.subtests[1].name, "fails");
        assert_eq!(r.subtests[1].status, SubtestStatus::Fail);
        let msg = r.subtests[1].message.as_deref().expect("failure message");
        assert!(msg.contains("math is broken"), "message: {msg}");
        assert_eq!(r.passed(), 1);
        assert_eq!(r.failed(), 1);
        // Overall = fail when any subtest fails: the harness ran cleanly
        // (OK) but the run is not green.
        assert!(!r.is_green());
    }

    #[test]
    fn synthetic_page_async_test() {
        use bowser_wpt::results::SubtestStatus;
        let r = run_synthetic_page(
            r#"var t = async_test("async addition");
               t.step(function () { assert_equals(2 * 3, 6); });
               t.done();
               test(function () { assert_true(true); }, "sync still counted");
               done();"#,
            "synth/async.html",
        );
        assert_eq!(r.status, HarnessStatus::Ok);
        assert_eq!(r.subtests.len(), 2);
        assert_eq!(r.subtests[0].name, "async addition");
        assert_eq!(r.subtests[0].status, SubtestStatus::Pass);
        assert_eq!(r.subtests[1].status, SubtestStatus::Pass);
        assert!(r.is_green());

        // An async_test whose step throws fails like a sync test.
        let r = run_synthetic_page(
            r#"var t = async_test("async failure");
               t.step(function () { assert_true(false, "step blew up"); });
               t.done();
               done();"#,
            "synth/async-fail.html",
        );
        assert_eq!(r.subtests.len(), 1);
        assert_eq!(r.subtests[0].name, "async failure");
        assert_eq!(r.subtests[0].status, SubtestStatus::Fail);
        assert!(
            r.subtests[0]
                .message
                .as_deref()
                .unwrap()
                .contains("step blew up")
        );
        assert!(!r.is_green());
    }

    #[test]
    fn synthetic_page_timeout() {
        use bowser_wpt::results::SubtestStatus;
        // Subtest timeout: the async test never completes; the shim marks
        // it TIMEOUT the way the real harness's internal timer would.
        let r = run_synthetic_page(
            r#"test(function () { assert_true(true); }, "fast");
               var t = async_test("slow");
               t.step(function () { assert_true(true); });
               t.force_timeout("timed out waiting for done()");
               done();"#,
            "synth/subtest-timeout.html",
        );
        assert_eq!(r.status, HarnessStatus::Ok);
        assert_eq!(r.subtests.len(), 2);
        assert_eq!(r.subtests[0].status, SubtestStatus::Pass);
        assert_eq!(r.subtests[1].name, "slow");
        assert_eq!(r.subtests[1].status, SubtestStatus::Timeout);
        assert_eq!(
            r.subtests[1].message.as_deref(),
            Some("timed out waiting for done()")
        );
        assert!(!r.is_green());

        // Harness timeout: done() never runs; the harness itself times out.
        // The unfinished async test stays NOTRUN; the timeout is carried
        // by the harness status.
        let r = run_synthetic_page(
            r#"test(function () { assert_true(true); }, "fast");
               var t = async_test("slow");
               __th_harness_timeout();"#,
            "synth/harness-timeout.html",
        );
        assert_eq!(r.status, HarnessStatus::Timeout);
        assert_eq!(r.subtests.len(), 2);
        assert_eq!(r.subtests[0].status, SubtestStatus::Pass);
        assert_eq!(r.subtests[1].status, SubtestStatus::NotRun);
        assert!(!r.is_green());
    }

    #[test]
    fn synthetic_page_start_and_result_callbacks() {
        use bowser_wpt::results::SubtestStatus;
        let (report, lines) = run_synthetic_page_lines(
            r#"test(function () { assert_true(true); }, "first");
               test(function () { assert_equals(1, 2, "boom"); }, "second \"quoted\"");
               var t = async_test("third");
               t.step(function () { assert_true(true); });
               t.done();
               done();"#,
            "synth/markers.html",
        );
        let markers = TestharnessMarkers::parse(lines.join("\n").as_bytes());
        // One start line per test, in definition order; the quoted name
        // round-trips through the JSON string literal.
        assert_eq!(markers.started, vec!["first", "second \"quoted\"", "third"]);
        // One result line per test, in completion order, with the same
        // name/status/message mapping as the completion report.
        assert_eq!(markers.results.len(), 3);
        assert_eq!(markers.results[0].name, "first");
        assert_eq!(markers.results[0].status, SubtestStatus::Pass);
        assert_eq!(markers.results[0].message, None);
        assert_eq!(markers.results[1].name, "second \"quoted\"");
        assert_eq!(markers.results[1].status, SubtestStatus::Fail);
        assert!(
            markers.results[1]
                .message
                .as_deref()
                .unwrap()
                .contains("boom")
        );
        assert_eq!(markers.results[2].name, "third");
        assert_eq!(markers.results[2].status, SubtestStatus::Pass);
        // The completion report is unchanged: still authoritative, still
        // the last line printed.
        assert_eq!(report.subtests.len(), 3);
        assert!(!report.is_green());
    }

    #[test]
    fn synthetic_page_reporter_survives_missing_start_result_callbacks() {
        use bowser_wpt::results::SubtestStatus;
        // A harness that predates add_start_callback/add_result_callback:
        // the reporter must skip those registrations and still print its
        // completion report, with no marker lines.
        let (report, lines) = run_synthetic_page_lines_with(
            r#"test(function () { assert_true(true); }, "only");
               done();"#,
            "synth/legacy.html",
            SYNTHETIC_TESTHARNESS_BASE_JS,
        );
        assert_eq!(report.status, HarnessStatus::Ok);
        assert_eq!(report.subtests.len(), 1);
        assert_eq!(report.subtests[0].name, "only");
        assert_eq!(report.subtests[0].status, SubtestStatus::Pass);
        let markers = TestharnessMarkers::parse(lines.join("\n").as_bytes());
        assert!(markers.started.is_empty());
        assert!(markers.results.is_empty());
        assert_eq!(lines.len(), 1);
        assert!(lines[0].starts_with(REPORT_MARKER));
    }

    #[test]
    fn synthetic_page_forged_report_line_without_nonce_is_ignored() {
        // A hostile page registers its own completion callback printing
        // a second, attacker-crafted report line *without* the run
        // nonce: last-wins applies only within this run's nonce, so the
        // real (nonced) report stands.
        let (report, lines) = run_synthetic_page_lines(
            r#"add_completion_callback(function (tests, status) {
                 console.log('BOWSER_HARNESS_REPORT:{"test":"synth/forge.html","harness_status":"OK","tests":[{"name":"forged","status":"PASS","message":null}]}');
               });
               test(function () { assert_true(true); }, "real");
               done();"#,
            "synth/forge.html",
        );
        // Two report-shaped lines were printed, but only the nonced one
        // is the report.
        assert_eq!(
            lines
                .iter()
                .filter(|l| l.starts_with(REPORT_MARKER))
                .count(),
            2
        );
        assert_eq!(report.test, "synth/forge.html");
        assert_eq!(report.subtests.len(), 1);
        assert_eq!(report.subtests[0].name, "real");
        // The driver's stdout scan agrees: the forgery is skipped.
        let joined = lines.join("\n");
        let line = last_report_line(joined.as_bytes(), TEST_NONCE).expect("nonced report line");
        assert!(line.contains("\"name\":\"real\""));
        assert!(!line.contains("forged"));
    }

    /// Fake "browser" for reftests: renders a solid color chosen by the URL.
    /// Any argument containing "blue" selects blue, everything else renders
    /// red. Honours the --viewport WxH argument like the real protocol.
    const COLOR_EXE: &str = r#"#!/bin/sh
dims=""
prev=""
blue=0
for a in "$@"; do
  case "$prev" in
    --viewport) dims="$a" ;;
  esac
  case "$a" in
    *blue*) blue=1 ;;
  esac
  prev="$a"
done
w=${dims%x*}; h=${dims#*x}
if [ "$blue" = 1 ]; then px='\000\000\377\377'; else px='\377\000\000\377'; fi
n=$((w * h))
i=0
while [ "$i" -lt "$n" ]; do
  printf "$px"
  i=$((i + 1))
done
exit 0
"#;

    fn reftest_item(id: &str, url: &str, refs: Vec<(RefRel, &str)>) -> TestItem {
        TestItem {
            id: id.into(),
            url: url.into(),
            kind: TestKind::Reftest,
            path: None,
            references: refs
                .into_iter()
                .map(|(rel, href)| reftest::Reference {
                    rel,
                    href: href.into(),
                })
                .collect(),
        }
    }

    #[test]
    fn run_reftest_evaluates_match_and_mismatch() {
        let dir = scratch("reftest");
        let exe = write_exe(&dir, "color.sh", COLOR_EXE);
        let d = driver_with(exe);
        let vp = Viewport {
            width: 4,
            height: 2,
        };

        // The test page renders red; blue.html renders blue.
        let t = reftest_item(
            "t.html",
            "/t.html",
            vec![
                (RefRel::Mismatch, "blue.html"), // red vs blue: differ -> satisfied
                (RefRel::Match, "red.html"),     // red vs red: equal -> satisfied
            ],
        );
        let run = d.run_reftest(&t, vp, ReftestTolerance::EXACT).unwrap();
        assert_eq!(run.test, "t.html");
        assert!(run.passed);
        assert_eq!(run.comparisons.len(), 2);
        assert!(!run.comparisons[0].equal);
        assert!(run.comparisons[0].satisfied);
        assert_eq!(run.comparisons[0].diff_pixels, 8);
        assert_eq!(run.comparisons[0].diff_fraction, 1.0);
        assert!(run.comparisons[1].equal);
        assert!(run.comparisons[1].satisfied);

        // A match reference that renders differently fails the run.
        let t = reftest_item("t.html", "/t.html", vec![(RefRel::Match, "blue.html")]);
        let run = d.run_reftest(&t, vp, ReftestTolerance::EXACT).unwrap();
        assert!(!run.passed);
        assert!(!run.comparisons[0].satisfied);

        // A mismatch reference that renders identically fails the run.
        let t = reftest_item("t.html", "/t.html", vec![(RefRel::Mismatch, "red.html")]);
        let run = d.run_reftest(&t, vp, ReftestTolerance::EXACT).unwrap();
        assert!(!run.passed);
        assert!(!run.comparisons[0].satisfied);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_reftest_match_references_are_alternatives() {
        let dir = scratch("reftest-alt");
        let exe = write_exe(&dir, "color.sh", COLOR_EXE);
        let d = driver_with(exe);
        let vp = Viewport {
            width: 4,
            height: 2,
        };

        // The test page renders red. Upstream WPT rule: multiple `match`
        // references are alternatives, so one matching reference is
        // enough even though the other does not match.
        let t = reftest_item(
            "t.html",
            "/t.html",
            vec![(RefRel::Match, "red.html"), (RefRel::Match, "blue.html")],
        );
        let run = d.run_reftest(&t, vp, ReftestTolerance::EXACT).unwrap();
        assert!(run.passed, "{run:?}");
        assert!(run.comparisons[0].satisfied);
        assert!(!run.comparisons[1].satisfied);

        // No `match` reference matching still fails.
        let t = reftest_item("t.html", "/t.html", vec![(RefRel::Match, "blue.html")]);
        let run = d.run_reftest(&t, vp, ReftestTolerance::EXACT).unwrap();
        assert!(!run.passed, "{run:?}");

        // A violated `mismatch` reference still fails the run even when a
        // `match` reference matches.
        let t = reftest_item(
            "t.html",
            "/t.html",
            vec![(RefRel::Match, "red.html"), (RefRel::Mismatch, "red.html")],
        );
        let run = d.run_reftest(&t, vp, ReftestTolerance::EXACT).unwrap();
        assert!(!run.passed, "{run:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Like COLOR_EXE, but URLs containing `near` render red off by one
    /// in the red channel (254 instead of 255).
    const NEAR_EXE: &str = r#"#!/bin/sh
dims=""
prev=""
near=0
for a in "$@"; do
  case "$prev" in
    --viewport) dims="$a" ;;
  esac
  case "$a" in
    *near*) near=1 ;;
  esac
  prev="$a"
done
w=${dims%x*}; h=${dims#*x}
if [ "$near" = 1 ]; then px='\376\000\000\377'; else px='\377\000\000\377'; fi
n=$((w * h))
i=0
while [ "$i" -lt "$n" ]; do
  printf "$px"
  i=$((i + 1))
done
exit 0
"#;

    #[test]
    fn run_reftest_for_extras_uses_fuzzy_and_viewport() {
        let dir = scratch("reftest-extras");
        let exe = write_exe(&dir, "near.sh", NEAR_EXE);
        let d = driver_with(exe);
        // The test renders red; the reference renders red off-by-one.
        let t = reftest_item("t.html", "/t.html", vec![(RefRel::Match, "nearred.html")]);
        let vp = Viewport {
            width: 4,
            height: 2,
        };
        // Exact comparison fails on the off-by-one channel.
        let run = d.run_reftest(&t, vp, ReftestTolerance::EXACT).unwrap();
        assert!(!run.passed);
        assert_eq!(run.comparisons[0].diff_pixels, 8);

        // With fuzzy extras (maxDifference=1) the off-by-one passes: the
        // per-channel slack counts every pixel as same, so nothing differs.
        let extras = TestExtras {
            fuzzy: Some(FuzzySpec {
                max_pixels: 100,
                max_diff: 1,
            }),
            viewport_size: Some((2, 2)),
            ..TestExtras::default()
        };
        let run = d
            .run_reftest_for_extras(&t, &extras, vp, ReftestTolerance::EXACT)
            .unwrap();
        assert!(run.passed, "{run:?}");
        assert!(run.comparisons[0].equal);
        assert_eq!(run.comparisons[0].diff_pixels, 0);

        // The viewport_size extra (2x2) renders 4 pixels instead of 8:
        // without the fuzzy slack every off-by-one pixel counts.
        let sharp_extras = TestExtras {
            viewport_size: Some((2, 2)),
            ..TestExtras::default()
        };
        let run = d
            .run_reftest_for_extras(&t, &sharp_extras, vp, ReftestTolerance::EXACT)
            .unwrap();
        assert!(!run.passed);
        assert_eq!(run.comparisons[0].diff_pixels, 4);

        let _ = std::fs::remove_dir_all(&dir);
    }
    fn comparison(rel: RefRel, satisfied: bool) -> ReferenceComparison {
        ReferenceComparison {
            href: "ref.html".into(),
            rel,
            equal: satisfied,
            diff_pixels: 0,
            diff_fraction: 0.0,
            satisfied,
        }
    }

    #[test]
    fn reftest_passed_follows_upstream_rule() {
        use RefRel::{Match, Mismatch};
        // No references: vacuous pass.
        assert!(reftest_passed(&[]));
        // Single match.
        assert!(reftest_passed(&[comparison(Match, true)]));
        assert!(!reftest_passed(&[comparison(Match, false)]));
        // Multiple match references are alternatives: one is enough.
        assert!(reftest_passed(&[
            comparison(Match, true),
            comparison(Match, false),
        ]));
        assert!(reftest_passed(&[
            comparison(Match, false),
            comparison(Match, true),
        ]));
        assert!(!reftest_passed(&[
            comparison(Match, false),
            comparison(Match, false),
        ]));
        // Every mismatch reference must differ.
        assert!(reftest_passed(&[comparison(Mismatch, true)]));
        assert!(!reftest_passed(&[comparison(Mismatch, false)]));
        assert!(reftest_passed(&[
            comparison(Mismatch, true),
            comparison(Mismatch, true),
        ]));
        assert!(!reftest_passed(&[
            comparison(Mismatch, true),
            comparison(Mismatch, false),
        ]));
        // Mixed: a matching `match` does not excuse a violated `mismatch`.
        assert!(reftest_passed(&[
            comparison(Match, true),
            comparison(Mismatch, true),
        ]));
        assert!(!reftest_passed(&[
            comparison(Match, true),
            comparison(Mismatch, false),
        ]));
        // Mismatch-only run: no `match` refs, so all-differing passes.
        assert!(reftest_passed(&[
            comparison(Mismatch, true),
            comparison(Mismatch, true),
        ]));
    }

    #[test]
    fn run_reftest_resolves_reference_urls() {
        let dir = scratch("reftesturls");
        let exe = write_exe(&dir, "color.sh", COLOR_EXE);
        let d = driver_with(exe);
        let vp = Viewport {
            width: 2,
            height: 1,
        };

        // Relative href resolves against the test's directory: /sub/sib.html
        // renders red, matching the red test page.
        let t = reftest_item(
            "sub/t.html",
            "/sub/t.html",
            vec![(RefRel::Match, "sib.html")],
        );
        let run = d.run_reftest(&t, vp, ReftestTolerance::EXACT).unwrap();
        assert!(run.passed, "{run:?}");

        // Absolute href resolves against the server root: /blue.html
        // renders blue, so the mismatch is satisfied.
        let t = reftest_item(
            "sub/t.html",
            "/sub/t.html",
            vec![(RefRel::Mismatch, "/blue.html")],
        );
        let run = d.run_reftest(&t, vp, ReftestTolerance::EXACT).unwrap();
        assert!(run.passed, "{run:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_reftest_propagates_render_failure() {
        let dir = scratch("reftestfail");
        let exe = write_exe(&dir, "fail.sh", "#!/bin/sh\nexit 1\n");
        let d = driver_with(exe);
        let t = reftest_item("t.html", "/t.html", vec![]);
        let err = d
            .run_reftest(&t, Viewport::DEFAULT, ReftestTolerance::EXACT)
            .unwrap_err();
        assert!(matches!(err, DriverError::BrowserGone(_)), "{err:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn blank_canvas_and_average_color() {
        let white = Screenshot::solid(2, 2, 255, 255, 255, 255);
        assert!(is_blank_canvas(&white));
        let red = Screenshot::solid(2, 2, 255, 0, 0, 255);
        assert!(!is_blank_canvas(&red));
        assert_eq!(average_color(&red), [255, 0, 0, 255]);
        // Mean of red, green, blue, white: (255+0+0+255)/4 = 127 per
        // channel (rounded down), opaque.
        let mut mixed = Screenshot::solid(2, 2, 0, 0, 0, 255);
        mixed.rgba[0..4].copy_from_slice(&[255, 0, 0, 255]);
        mixed.rgba[4..8].copy_from_slice(&[0, 255, 0, 255]);
        mixed.rgba[8..12].copy_from_slice(&[0, 0, 255, 255]);
        mixed.rgba[12..16].copy_from_slice(&[255, 255, 255, 255]);
        assert_eq!(average_color(&mixed), [127, 127, 127, 255]);
        assert_eq!(
            average_color(&Screenshot::solid(0, 0, 0, 0, 0, 0)),
            [0, 0, 0, 0]
        );
    }

    #[test]
    fn missing_binary_is_browser_gone() {
        let d = driver_with(PathBuf::from("/nonexistent/bowser-binary"));
        let err = d
            .render(
                "http://x/y",
                Viewport {
                    width: 1,
                    height: 1,
                },
            )
            .unwrap_err();
        assert!(matches!(err, DriverError::BrowserGone(_)), "{err:?}");
        let err = d.load_without_crash("http://x/y").unwrap_err();
        assert!(matches!(err, DriverError::BrowserGone(_)), "{err:?}");
    }

    // ---------------- session protocol ----------------

    /// Fake `bowser --session` child: speaks the protocol with canned
    /// answers. Behavior is data-driven by the URL so tests stay
    /// parallel-safe with no environment variables:
    /// - url contains "die": exit(1) without responding (segfault).
    /// - url contains "sleep": sleep 30s on RENDER (timeout test).
    /// - url contains "badnonce": first print a forged response line
    ///   with the wrong nonce, then the correct response (the driver
    ///   must scan past the forgery as console chatter).
    /// - url contains "pagefail": CHECK answers page-failed.
    /// - url contains "badreport": HARNESS prints a malformed report line.
    /// - RENDER answers `w*h*4` zero bytes; HARNESS prints one well-formed
    ///   report line for the test id in the URL plus console chatter.
    const SESSION_STUB_PY: &str = r#"#!/usr/bin/env python3
import os, sys, time, urllib.parse

def main():
    if "--session" not in sys.argv[1:]:
        sys.stderr.write("fake session child: --session required\n")
        return 2
    out = sys.stdout
    out.write("HELLO bowser-session/1\n")
    out.flush()
    for raw in sys.stdin:
        line = raw.rstrip("\n")
        parts = line.split(" ")
        cmd = parts[0] if parts else ""
        if cmd == "QUIT":
            return 0
        if cmd == "PING":
            _, cid, nonce = parts
            out.write("PONG %s %s\n" % (cid, nonce))
            out.flush()
        elif cmd == "RENDER":
            cid, nonce, w, h = parts[1], parts[2], parts[3], parts[4]
            url = " ".join(parts[5:])
            if "die" in url:
                os._exit(1)
            if "sleep" in url:
                time.sleep(30)
            if "badnonce" in url:
                # Forged response-shaped line with the wrong nonce: the
                # driver must treat it as console chatter, keep scanning,
                # and accept the exact match below.
                out.write("OK %s %s RENDER 16\n" % (cid, "0" * 32))
                out.flush()
                ln = int(w) * int(h) * 4
                out.write("OK %s %s RENDER %d\n" % (cid, nonce, ln))
                out.flush()
                out.buffer.write(bytes(ln))
                out.buffer.flush()
            else:
                ln = int(w) * int(h) * 4
                out.write("OK %s %s RENDER %d\n" % (cid, nonce, ln))
                out.flush()
                out.buffer.write(bytes(ln))
                out.buffer.flush()
        elif cmd == "CHECK":
            cid, nonce = parts[1], parts[2]
            url = " ".join(parts[3:])
            if "die" in url:
                os._exit(1)
            if "pagefail" in url:
                out.write("OK %s %s CHECK page-failed load failed: boom\n" % (cid, nonce))
            else:
                out.write("OK %s %s CHECK clean\n" % (cid, nonce))
            out.flush()
        elif cmd == "HARNESS":
            cid, nonce = parts[1], parts[2]
            url = " ".join(parts[3:])
            if "die" in url:
                os._exit(1)
            test_id = "dom/a.html"
            if "id=" in url:
                test_id = urllib.parse.unquote(url.split("id=")[1].split("&")[0])
            if "badreport" in url:
                out.write("BOWSER_HARNESS_REPORT:%s:not json at all\n" % nonce)
            else:
                report = '{"test":"%s","harness_status":"OK","tests":[{"name":"t","status":"PASS"}]}' % test_id
                out.write("BOWSER_HARNESS_REPORT:%s:%s\n" % (nonce, report))
            out.write("console chatter\n")
            out.write("OK %s %s HARNESS clean\n" % (cid, nonce))
            out.flush()
        else:
            out.write("ERR 0 - BADCMD unknown command\n")
            out.flush()
    return 0

sys.exit(main())
"#;

    fn session_stub(dir: &Path) -> PathBuf {
        write_exe(dir, "session-stub", SESSION_STUB_PY)
    }

    fn session_driver(exe: PathBuf) -> SessionDriver {
        // Unique docroot per driver: tests run in parallel, and sharing
        // one scratch dir would race on its create/remove.
        static DOCROOT_COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = DOCROOT_COUNTER.fetch_add(1, Ordering::SeqCst);
        let root = scratch(&format!("session-docroot-{n}"));
        let mut d = SessionDriver::new(exe, &root).unwrap();
        d.inner_mut().render_timeout = Duration::from_secs(10);
        d.inner_mut().check_timeout = Duration::from_secs(10);
        let _ = std::fs::remove_dir_all(&root);
        d
    }

    #[test]
    fn parse_session_commands() {
        let n = "0123456789abcdef0123456789abcdef";
        assert!(matches!(
            parse_session_command("QUIT"),
            Ok(SessionCommand::Quit)
        ));
        match parse_session_command(&format!("PING 7 {n}")) {
            Ok(SessionCommand::Ping { id, nonce }) => {
                assert_eq!(id, 7);
                assert_eq!(nonce, n);
            }
            c => panic!("bad PING parse: {c:?}"),
        }
        match parse_session_command(&format!("RENDER 8 {n} 800 600 http://x/y?a=b&c=d")) {
            Ok(SessionCommand::Render {
                id,
                nonce,
                width,
                height,
                url,
            }) => {
                assert_eq!(id, 8);
                assert_eq!(nonce, n);
                assert_eq!((width, height), (800, 600));
                assert_eq!(url, "http://x/y?a=b&c=d");
            }
            c => panic!("bad RENDER parse: {c:?}"),
        }
        // The URL runs to end of line, so query strings survive.
        match parse_session_command(&format!("CHECK 9 {n} http://x/a?b=c")) {
            Ok(SessionCommand::Check { url, .. }) => assert_eq!(url, "http://x/a?b=c"),
            c => panic!("bad CHECK parse: {c:?}"),
        }
        match parse_session_command(&format!("HARNESS 10 {n} http://x/h")) {
            Ok(SessionCommand::Harness { id, .. }) => assert_eq!(id, 10),
            c => panic!("bad HARNESS parse: {c:?}"),
        }
        // Rejections: malformed, bad id/nonce, bad dimensions, empty url.
        for bad in [
            "",
            "BOGUS",
            "QUIT x",
            "PING 7",
            "PING x aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "PING 7 short",
            "PING 7 zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
            "RENDER 8 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 800 600",
            "RENDER 8 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 0 600 http://x/",
            "RENDER 8 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 20000 600 http://x/",
            "RENDER 8 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 800 x http://x/",
            "CHECK 9 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "HARNESS 10 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ",
        ] {
            assert!(parse_session_command(bad).is_err(), "must reject {bad:?}");
        }
    }

    #[test]
    fn classify_response_lines() {
        let n = "0123456789abcdef0123456789abcdef";
        let id = 42u64;
        assert!(matches!(
            classify_response_line(&format!("PONG {id} {n}"), id, n),
            LineKind::Pong
        ));
        assert!(matches!(
            classify_response_line(&format!("OK {id} {n} RENDER 16"), id, n),
            LineKind::Render { len: 16 }
        ));
        assert!(matches!(
            classify_response_line(&format!("OK {id} {n} CHECK clean"), id, n),
            LineKind::Check(SessionVerdict::Clean)
        ));
        assert!(matches!(
            classify_response_line(&format!("OK {id} {n} CHECK page-failed boom"), id, n),
            LineKind::Check(SessionVerdict::PageFailed)
        ));
        assert!(matches!(
            classify_response_line(&format!("OK {id} {n} HARNESS page-failed"), id, n),
            LineKind::Harness(SessionVerdict::PageFailed)
        ));
        assert!(matches!(
            classify_response_line(&format!("ERR {id} {n} BROWSER segfault"), id, n),
            LineKind::BrowserError(m) if m == "segfault"
        ));
        assert!(matches!(
            classify_response_line("ERR 0 - BADCMD nope", id, n),
            LineKind::BadCmd(m) if m == "nope"
        ));
        // Console lines are never responses — even ones that start with
        // a response tag.
        for console in [
            "OK computer",
            "PONG",
            "hello world",
            "BOWSER_HARNESS_REPORT:xyz:{...}",
            "OK 42 nothex",
            "",
            "ERR 0 - something else",
        ] {
            assert!(
                matches!(classify_response_line(console, id, n), LineKind::Console),
                "must be console: {console:?}"
            );
        }
        // Right shape, wrong values: console output, not a protocol
        // error — the reader scans past these for an exact match. This
        // is the hostile-page case (e.g. console.log("OK", n,
        // uuid.replace(/-/g,"")): UUIDs are exactly 32 hex chars).
        let wrong_nonce = "f".repeat(32);
        for sprayed in [
            format!("PONG {id} {wrong_nonce}"),
            format!("PONG 43 {n}"),
            format!("OK 7 {wrong_nonce} CHECK clean"),
            format!("OK {id} {wrong_nonce} RENDER 16"),
            format!("ERR 9 {n} BROWSER boom"),
        ] {
            assert!(
                matches!(classify_response_line(&sprayed, id, n), LineKind::Console),
                "sprayed line must be console, not a protocol error: {sprayed:?}"
            );
        }
        // Our id and nonce, bad body: malformed.
        assert!(matches!(
            classify_response_line(&format!("OK {id} {n} FROBNICATE"), id, n),
            LineKind::Malformed(_)
        ));
        assert!(matches!(
            classify_response_line(&format!("OK {id} {n} RENDER x"), id, n),
            LineKind::Malformed(_)
        ));
        assert!(matches!(
            classify_response_line(&format!("PONG {id} {n} extra"), id, n),
            LineKind::Malformed(_)
        ));
    }

    #[test]
    fn one_line_sanitizes() {
        assert_eq!(one_line("a\nb\rc"), "a b c");
        assert_eq!(one_line(&"x".repeat(600)).len(), 512);
    }

    /// Stub browser work for driving [`run_session_loop`] in memory.
    struct StubSessionOps {
        log: Vec<String>,
        fail_render: bool,
    }

    impl SessionOps for StubSessionOps {
        fn render(&mut self, url: &str, width: u32, height: u32) -> Result<Vec<u8>, String> {
            self.log.push(format!("render {url} {width}x{height}"));
            if self.fail_render {
                return Err("render boom".into());
            }
            Ok(vec![0xABu8; width as usize * height as usize * 4])
        }

        fn check(&mut self, url: &str) -> Result<(), String> {
            self.log.push(format!("check {url}"));
            if url.contains("fail") {
                Err("page boom".into())
            } else {
                Ok(())
            }
        }

        fn harness(&mut self, url: &str, console: &mut dyn Write) -> Result<(), String> {
            self.log.push(format!("harness {url}"));
            // Console lines must precede the OK line in the byte stream.
            writeln!(console, "BOWSER_HARNESS_START: \"t\"").unwrap();
            self.check(url)
        }
    }

    fn run_loop_with(ops: &mut StubSessionOps, input: &str) -> (i32, Vec<u8>) {
        let mut input = std::io::Cursor::new(input.as_bytes().to_vec());
        let mut output = Vec::new();
        let code = run_session_loop(&mut input, &mut output, ops);
        (code, output)
    }

    #[test]
    fn session_loop_serves_commands_in_memory() {
        let (a, b, c, d, e) = (
            "a".repeat(32),
            "b".repeat(32),
            "c".repeat(32),
            "d".repeat(32),
            "e".repeat(32),
        );
        let input = format!(
            "PING 7 {a}\n\
             RENDER 8 {b} 2 2 http://example/x\n\
             CHECK 9 {c} http://example/ok\n\
             CHECK 10 {d} http://example/fail\n\
             HARNESS 11 {e} http://example/h\n\
             BOGUS\n\
             QUIT\n"
        );
        let mut ops = StubSessionOps {
            log: Vec::new(),
            fail_render: false,
        };
        let (code, output) = run_loop_with(&mut ops, &input);
        assert_eq!(code, 0);
        let mut expected = Vec::new();
        expected.extend_from_slice(b"HELLO bowser-session/1\n");
        expected.extend_from_slice(format!("PONG 7 {a}\n").as_bytes());
        expected.extend_from_slice(format!("OK 8 {b} RENDER 16\n").as_bytes());
        expected.extend_from_slice(&[0xABu8; 16]);
        expected.extend_from_slice(format!("OK 9 {c} CHECK clean\n").as_bytes());
        expected.extend_from_slice(format!("OK 10 {d} CHECK page-failed page boom\n").as_bytes());
        expected.extend_from_slice(b"BOWSER_HARNESS_START: \"t\"\n");
        expected.extend_from_slice(format!("OK 11 {e} HARNESS clean\n").as_bytes());
        assert!(
            output.starts_with(&expected),
            "output diverged:\n{}",
            String::from_utf8_lossy(&output)
        );
        // The malformed command is rejected and the child stays alive
        // for QUIT after it.
        let tail = &output[expected.len()..];
        assert!(
            tail.starts_with(b"ERR 0 - BADCMD "),
            "tail was: {:?}",
            String::from_utf8_lossy(tail)
        );
        assert!(tail.ends_with(b"\n"));
        assert_eq!(
            ops.log,
            vec![
                "render http://example/x 2x2",
                "check http://example/ok",
                "check http://example/fail",
                "harness http://example/h",
                "check http://example/h",
            ]
        );
    }

    #[test]
    fn session_loop_exits_zero_on_eof() {
        let mut ops = StubSessionOps {
            log: Vec::new(),
            fail_render: false,
        };
        let (code, output) = run_loop_with(&mut ops, "PING 1 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n");
        assert_eq!(code, 0);
        assert_eq!(
            output,
            b"HELLO bowser-session/1\nPONG 1 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        );
    }

    #[test]
    fn session_loop_reports_render_failure() {
        let b = "b".repeat(32);
        let mut ops = StubSessionOps {
            log: Vec::new(),
            fail_render: true,
        };
        let (code, output) =
            run_loop_with(&mut ops, &format!("RENDER 8 {b} 2 2 http://x/\nQUIT\n"));
        assert_eq!(code, 0);
        let text = String::from_utf8(output).unwrap();
        assert!(
            text.contains(&format!("ERR 8 {b} BROWSER render boom\n")),
            "output was: {text:?}"
        );
    }

    #[test]
    fn session_driver_name() {
        let dir = scratch("session-name");
        let d = session_driver(session_stub(&dir));
        assert_eq!(d.name(), "bowser-session");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_multiplexes_renders_over_one_child() {
        let dir = scratch("session-renders");
        let d = session_driver(session_stub(&dir));
        let vp = Viewport {
            width: 2,
            height: 2,
        };
        for i in 0..3 {
            let shot = d.render(&format!("http://x/r{i}"), vp).unwrap();
            assert_eq!(shot.rgba, vec![0u8; 16]);
            assert_eq!((shot.width, shot.height), (2, 2));
        }
        // Three operations, one child spawn: the per-test spawn cost is
        // gone.
        let stats = d.session_stats();
        assert_eq!(stats.spawns, 1, "{stats:?}");
        assert_eq!(stats.session_ops, 3, "{stats:?}");
        assert_eq!(stats.fallback_ops, 0, "{stats:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_harness_parses_report_through_nonce() {
        let dir = scratch("session-harness");
        let d = session_driver(session_stub(&dir));
        let t = item("dom/a.html", "/dom/a.html");
        let (report, markers) = d
            .run_testharness_with_markers(&t, Duration::from_secs(10))
            .unwrap();
        // The stub prints the report line with the command nonce and
        // console chatter around it; the driver must demux, attribute,
        // and parse exactly like the one-shot path.
        assert_eq!(report.test, "dom/a.html");
        assert_eq!(report.url.as_deref(), Some("/dom/a.html"));
        assert_eq!(report.status, HarnessStatus::Ok);
        assert!(report.is_green());
        assert_eq!(report.subtests.len(), 1);
        assert_eq!(report.subtests[0].name, "t");
        assert!(markers.started.is_empty() && markers.results.is_empty());
        assert!(!markers.truncated);
        assert_eq!(d.session_stats().session_ops, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_harness_bad_report() {
        let dir = scratch("session-badreport");
        let d = session_driver(session_stub(&dir));
        let t = item("dom/a.html", "/dom/a.html?badreport");
        let err = d.run_testharness(&t, Duration::from_secs(10)).unwrap_err();
        assert!(matches!(err, DriverError::BadReport(_)), "{err:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_check_verdicts() {
        let dir = scratch("session-check");
        let d = session_driver(session_stub(&dir));
        // Clean and page-failed both mean the browser survived the load:
        // parity with the one-shot exit-code mapping.
        d.load_without_crash("http://x/ok").unwrap();
        d.load_without_crash("http://x/pagefail").unwrap();
        assert_eq!(d.session_stats().session_ops, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_falls_back_when_child_lacks_session() {
        // A `bowser` binary without --session (built without the
        // wpt-driver feature) fails the HELLO handshake; the driver must
        // degrade to one child per operation instead of erroring.
        let dir = scratch("session-fallback");
        let exe = write_exe(
            &dir,
            "no-session",
            "#!/usr/bin/env python3\nimport sys\nsys.exit(2)\n",
        );
        let d = session_driver(exe);
        let vp = Viewport {
            width: 2,
            height: 2,
        };
        let err = d.render("http://x/y", vp).unwrap_err();
        assert!(matches!(err, DriverError::BrowserGone(_)), "{err:?}");
        let stats = d.session_stats();
        assert_eq!(stats.spawns, 0, "{stats:?}");
        assert_eq!(stats.fallback_ops, 1, "{stats:?}");
        // The disable is sticky: no spawn retry on the next operation.
        let _ = d.render("http://x/y", vp);
        assert_eq!(d.session_stats().spawns, 0);
        assert_eq!(d.session_stats().fallback_ops, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_restarts_after_child_death() {
        let dir = scratch("session-restart");
        let d = session_driver(session_stub(&dir));
        let vp = Viewport {
            width: 2,
            height: 2,
        };
        // The stub dies mid-render: the browser is gone for this op...
        let err = d.render("http://x/die", vp).unwrap_err();
        assert!(matches!(err, DriverError::BrowserGone(_)), "{err:?}");
        // ...but the session restarts, and the next op succeeds on it.
        let shot = d.render("http://x/ok", vp).unwrap();
        assert_eq!(shot.rgba.len(), 16);
        let stats = d.session_stats();
        assert_eq!(stats.spawns, 2, "{stats:?}");
        assert_eq!(stats.restarts, 1, "{stats:?}");
        assert_eq!(stats.session_ops, 1, "{stats:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_probes_idle_child() {
        let dir = scratch("session-probe");
        let mut d = session_driver(session_stub(&dir));
        d.probe_interval = Duration::from_millis(50);
        let vp = Viewport {
            width: 2,
            height: 2,
        };
        d.render("http://x/a", vp).unwrap();
        std::thread::sleep(Duration::from_millis(120));
        d.render("http://x/b", vp).unwrap();
        let stats = d.session_stats();
        assert_eq!(stats.probes, 1, "{stats:?}");
        assert_eq!(stats.spawns, 1, "{stats:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_op_timeout_kills_and_restarts() {
        let dir = scratch("session-timeout");
        let mut d = session_driver(session_stub(&dir));
        d.inner_mut().render_timeout = Duration::from_millis(100);
        let vp = Viewport {
            width: 2,
            height: 2,
        };
        let err = d.render("http://x/sleep", vp).unwrap_err();
        match err {
            DriverError::Timeout { test, after } => {
                assert_eq!(test, "http://x/sleep");
                assert_eq!(after, Duration::from_millis(100));
            }
            e => panic!("expected Timeout, got {e:?}"),
        }
        // The wedged child was killed and the session restarted eagerly.
        let stats = d.session_stats();
        assert_eq!(stats.spawns, 2, "{stats:?}");
        assert_eq!(stats.restarts, 1, "{stats:?}");
        // And the fresh session serves the next op.
        d.render("http://x/ok", vp).unwrap();
        assert_eq!(d.session_stats().session_ops, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_disables_after_consecutive_failures() {
        let dir = scratch("session-disable");
        let d = session_driver(session_stub(&dir));
        let vp = Viewport {
            width: 2,
            height: 2,
        };
        for _ in 0..3 {
            let err = d.render("http://x/die", vp).unwrap_err();
            assert!(matches!(err, DriverError::BrowserGone(_)), "{err:?}");
        }
        let stats = d.session_stats();
        assert_eq!(stats.spawns, 3, "{stats:?}");
        assert_eq!(stats.restarts, 2, "{stats:?}");
        // Now disabled: the next op goes straight to the one-shot
        // fallback (the stub exits 2 without --session) with no new
        // spawn.
        let err = d.render("http://x/die", vp).unwrap_err();
        assert!(matches!(err, DriverError::BrowserGone(_)), "{err:?}");
        let stats = d.session_stats();
        assert_eq!(stats.spawns, 3, "{stats:?}");
        assert_eq!(stats.fallback_ops, 1, "{stats:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_nonce_mismatch_scans_past_forgery() {
        // A response-shaped line with the wrong nonce is console
        // chatter, not a protocol error: the reader scans past it and
        // accepts the exact id+nonce match. The session stays up and the
        // op succeeds.
        let dir = scratch("session-mismatch");
        let d = session_driver(session_stub(&dir));
        let shot = d
            .render(
                "http://x/badnonce",
                Viewport {
                    width: 2,
                    height: 2,
                },
            )
            .expect("render past a forged nonce line must succeed");
        assert_eq!(shot.rgba.len(), 2 * 2 * 4);
        let stats = d.session_stats();
        assert_eq!(stats.session_ops, 1, "{stats:?}");
        assert_eq!(stats.fallback_ops, 0, "{stats:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Newest modification time under `dir`, recursively. Used by the
    /// smoke test's freshness guard: the real `bowser` binary must not
    /// predate the source tree it is supposed to test.
    fn newest_mtime(dir: &Path) -> std::io::Result<std::time::SystemTime> {
        let mut newest = std::fs::metadata(dir)?.modified()?;
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            for entry in std::fs::read_dir(&d)? {
                let entry = entry?;
                let md = entry.metadata()?;
                if md.is_dir() {
                    stack.push(entry.path());
                } else if let Ok(m) = md.modified()
                    && m > newest
                {
                    newest = m;
                }
            }
        }
        Ok(newest)
    }

    /// Smoke test: the REAL vendored WPT `resources/testharness.js`
    /// (`tests/fixtures/testharness.js`, byte-identical to upstream)
    /// runs two pages through the real `bowser --check` headless
    /// pipeline: one with a passing and a failing synchronous test, and
    /// one all-pass page. The injected reporter's completion reports
    /// carry the real per-test results, and the all-pass page asserts
    /// the harness-level status unwraps from the `TestsStatus` object to
    /// `OK` (a bare-number-only mapping would degrade it to `UNKNOWN`
    /// and fail `is_green()`).
    ///
    /// This needs the real headless binary beside the test executable
    /// (`<target>/debug/bowser` — build with
    /// `cargo build -p bowser_app --features wpt-driver --bin bowser`
    /// in the same target dir), because the driver spawns it as the
    /// `--check` child.
    ///
    /// This depends on the headless pump fix: the real harness only
    /// completes after the window `load` event plus a `setTimeout(0)`,
    /// so `--check` must pump until the reporter's completion flag
    /// fires (see `pump_headless`); otherwise the run falls back to
    /// `SubtestsUnknown`.
    #[test]
    fn real_wpt_testharness_smoke() {
        use bowser_wpt::results::{HarnessStatus, SubtestStatus};

        let dir = scratch("th-smoke");
        // Serve the vendored fixture as a plain subresource of the doc
        // root; the harness endpoint injects the reporter into the page.
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/testharness.js");
        assert!(
            fixture.is_file(),
            "vendored testharness.js fixture missing: {}",
            fixture.display()
        );
        std::fs::copy(&fixture, dir.join("testharness.js")).unwrap();
        std::fs::write(
            dir.join("smoke.html"),
            "<!doctype html>\n\
             <html>\n\
             <head><meta charset=\"utf-8\"><title>smoke</title></head>\n\
             <body>\n\
             <script src=\"testharness.js\"></script>\n\
             <script>\n\
             test(function() { assert_true(true); }, \"trivially passes\");\n\
             test(function() { assert_true(false, \"deliberate failure\"); }, \"deliberately fails\");\n\
             </script>\n\
             </body>\n\
             </html>\n",
        )
        .unwrap();
        // An all-pass page: the harness completes with TestsStatus OK,
        // which exercises the reporter's TestsStatus-object unwrap — a
        // bare-number-only mapping would degrade this to UNKNOWN and
        // fail `is_green()`.
        std::fs::write(
            dir.join("smoke_pass.html"),
            "<!doctype html>\n\
             <html>\n\
             <head><meta charset=\"utf-8\"><title>smoke pass</title></head>\n\
             <body>\n\
             <script src=\"testharness.js\"></script>\n\
             <script>\n\
             test(function() { assert_true(true); }, \"passes one\");\n\
             test(function() { assert_equals(1 + 1, 2); }, \"passes two\");\n\
             </script>\n\
             </body>\n\
             </html>\n",
        )
        .unwrap();
        // The real headless binary, derived from this test executable's
        // location: `<target>/debug/deps/bowser_app-<hash>` ->
        // `<target>/debug/bowser`.
        let exe = std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("bowser");
        assert!(
            exe.is_file(),
            "real bowser binary missing at {} (build it with \
             `cargo build -p bowser_app --features wpt-driver --bin bowser`)",
            exe.display()
        );
        // Freshness guard: `cargo test` does not build
        // `target/debug/bowser`, so a stale binary — from an older tree
        // or another checkout — would be used silently and the smoke
        // test would pass or fail for the wrong reasons. Fail loudly
        // when the binary predates the `bowser_app` source tree.
        let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
        let binary_mtime = std::fs::metadata(&exe).and_then(|m| m.modified()).ok();
        let src_mtime = newest_mtime(&src_dir).ok();
        if let (Some(bin), Some(src)) = (binary_mtime, src_mtime) {
            assert!(
                bin >= src,
                "stale bowser binary at {} (binary built at {:?}, but {} changed at {:?}): \
                 rebuild with `cargo build -p bowser_app --features wpt-driver --bin bowser`",
                exe.display(),
                bin,
                src_dir.display(),
                src
            );
        }
        let d = BowserDriver::new(exe, &dir).unwrap();

        let r = d
            .run_testharness(&item("smoke", "/smoke.html"), Duration::from_secs(90))
            .unwrap();

        // Attribution: the report carries the stamped id, and the url is
        // filled in from the TestItem the driver ran.
        assert_eq!(r.test, "smoke");
        assert_eq!(r.url.as_deref(), Some("/smoke.html"));
        // The real harness completed and reported both subtests: one
        // PASS, one FAIL.
        assert_eq!(r.subtests.len(), 2);
        assert_eq!(r.subtests[0].name, "trivially passes");
        assert_eq!(r.subtests[0].status, SubtestStatus::Pass);
        assert_eq!(r.subtests[0].message, None);
        assert_eq!(r.subtests[1].name, "deliberately fails");
        assert_eq!(r.subtests[1].status, SubtestStatus::Fail);
        // NOTE: the exact failure message is engine-dependent. Today it
        // is "get_stack is not defined": the real harness's
        // `AssertionError` constructor calls the `const get_stack`
        // declared textually after it, and bowser_js does not resolve
        // `let`/`const` bindings from functions defined textually before
        // the declaration. Only require that a message is present.
        assert!(r.subtests[1].message.is_some());
        assert_eq!(r.passed(), 1);
        assert_eq!(r.failed(), 1);
        assert!(!r.is_green());
        assert!(r.duration_ms.is_some());
        // The harness itself completed (the failing subtest does not
        // fail the harness): the TestsStatus object unwraps to OK and no
        // harness message is set.
        assert_eq!(r.status, HarnessStatus::Ok);
        assert_eq!(r.message, None);

        // All-pass page: the reporter's TestsStatus unwrap must map the
        // object to OK, so the run is green.
        let r = d
            .run_testharness(
                &item("smoke_pass", "/smoke_pass.html"),
                Duration::from_secs(90),
            )
            .unwrap();
        assert_eq!(r.test, "smoke_pass");
        assert_eq!(r.url.as_deref(), Some("/smoke_pass.html"));
        assert_eq!(r.status, HarnessStatus::Ok);
        assert_eq!(r.message, None);
        assert_eq!(r.subtests.len(), 2);
        assert!(r.subtests.iter().all(|t| t.status == SubtestStatus::Pass));
        assert!(r.is_green());

        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Tests for the fixture server's wptserve subset: `?pipe=` directives,
/// `<path>.headers` sidecars, plain-variant serving next to a
/// `<name>.any.js` sibling, and `.any.html` / `.window.html` wrapper
/// synthesis. (Owned by the wave-39 FixtureServer worker;
/// the `tests` module above is owned separately.)
///
/// Manual smoke against the real WPT checkout (needs the
/// `<wpt-checkout>` blobs: `git -C <wpt-checkout> sparse-checkout add
/// encoding resources`; not part of this suite by design):
/// ```sh
/// cargo build -p bowser_app --features wpt-driver --bin bowser-wpt --bin bowser
/// ./target/debug/bowser-wpt --wpt-root <wpt-checkout> --kind testharness \
///   --include encoding/big5-encoder.html
/// # .any.html synthesis via the full manifest (worker variants 404 by design):
/// unzstd -c <wpt-checkout>/MANIFEST-full.json.zst > /tmp/MANIFEST-full.json
/// ./target/debug/bowser-wpt --wpt-root <wpt-checkout> \
///   --manifest /tmp/MANIFEST-full.json --kind testharness \
///   --include "encoding/api-basics"
/// ```
#[cfg(test)]
mod fixture_server_pipe_tests {
    use super::*;

    fn scratch_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("bowser-wpt-pipe-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Raw HTTP GET; returns (status, headers, body). Header names are
    /// lowercased for easy lookup.
    fn http_get(base: &str, target: &str) -> (u16, Vec<(String, String)>, Vec<u8>) {
        let mut stream = TcpStream::connect(base.trim_start_matches("http://")).expect("connect");
        write!(
            stream,
            "GET {target} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"
        )
        .unwrap();
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).unwrap();
        let split = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .expect("no header/body split");
        let head = String::from_utf8(raw[..split].to_vec()).unwrap();
        let mut lines = head.lines();
        let status: u16 = lines
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let headers = lines
            .map(|l| {
                let (k, v) = l.split_once(':').unwrap();
                (k.trim().to_lowercase(), v.trim().to_owned())
            })
            .collect();
        (status, headers, raw[split + 4..].to_vec())
    }

    fn get_header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
        headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    fn header_values<'a>(headers: &'a [(String, String)], name: &str) -> Vec<&'a str> {
        headers
            .iter()
            .filter(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
            .collect()
    }

    #[test]
    fn wpt_pipe_parsing_preserves_order() {
        let dirs = parse_pipe("status(418)|header(X-Test,yes)|trickle(d1)").unwrap();
        assert_eq!(
            dirs,
            vec![
                PipeDirective::Status(418),
                PipeDirective::Header {
                    name: "X-Test".into(),
                    value: "yes".into(),
                    append: false,
                },
                PipeDirective::Trickle(TricklePlan {
                    ops: vec![TrickleOp::DelayMs(1000)],
                    repeat_last: None,
                }),
            ]
        );
    }

    #[test]
    fn wpt_pipe_header_replaces_by_default_appends_with_flag() {
        // Upstream wptserve: `header(Name,Value)` REPLACES same-named
        // headers; the third argument enables appending.
        let dirs = parse_pipe("header(X-A,1)|header(X-A,2)").unwrap();
        assert_eq!(
            dirs,
            vec![
                PipeDirective::Header {
                    name: "X-A".into(),
                    value: "1".into(),
                    append: false,
                },
                PipeDirective::Header {
                    name: "X-A".into(),
                    value: "2".into(),
                    append: false,
                },
            ]
        );
        // `header(X,A,True)` is three arguments — value "A", append —
        // not the value "A,True".
        let dirs = parse_pipe("header(X-T,A,True)").unwrap();
        assert_eq!(
            dirs,
            vec![PipeDirective::Header {
                name: "X-T".into(),
                value: "A".into(),
                append: true,
            }]
        );
        for flag in ["True", "true", "1"] {
            let dirs = parse_pipe(&format!("header(X-A,1,{flag})")).unwrap();
            assert!(
                matches!(&dirs[..], [PipeDirective::Header { append: true, .. }]),
                "flag {flag} should append"
            );
        }
        for flag in ["False", "false", "0"] {
            let dirs = parse_pipe(&format!("header(X-A,1,{flag})")).unwrap();
            assert!(
                matches!(&dirs[..], [PipeDirective::Header { append: false, .. }]),
                "flag {flag} should replace"
            );
        }
        // Name and value are trimmed (a bowser normalization).
        let dirs = parse_pipe("header( X-A , 1 )").unwrap();
        assert_eq!(
            dirs,
            vec![PipeDirective::Header {
                name: "X-A".into(),
                value: "1".into(),
                append: false,
            }]
        );
    }

    #[test]
    fn wpt_pipe_header_escapes() {
        // An escaped comma does not split the argument list (upstream
        // PipeTokenizer semantics).
        let dirs = parse_pipe(r"header(X-C,a\,b)").unwrap();
        assert_eq!(
            dirs,
            vec![PipeDirective::Header {
                name: "X-C".into(),
                value: "a,b".into(),
                append: false,
            }]
        );
    }

    #[test]
    fn wpt_pipe_unknown_and_malformed_is_500() {
        // Upstream parity: unknown pipe functions, arg-count mismatches,
        // and unconvertible arguments are errors (the server answers
        // 500), not silently skipped.
        assert_eq!(parse_pipe("bogus(1)"), Err(PipeError::UnknownFunction));
        assert_eq!(parse_pipe("status(abc)"), Err(PipeError::BadValue));
        assert_eq!(parse_pipe("status()"), Err(PipeError::BadValue));
        assert_eq!(parse_pipe("status(200,201)"), Err(PipeError::ArgCount));
        assert_eq!(parse_pipe("header(X)"), Err(PipeError::ArgCount));
        assert_eq!(parse_pipe("header(X,A,maybe)"), Err(PipeError::BadValue));
        assert_eq!(parse_pipe("header(,x)"), Err(PipeError::BadValue));
        assert_eq!(parse_pipe("trickle()"), Err(PipeError::BadValue));
        assert_eq!(parse_pipe("trickle(x)"), Err(PipeError::BadValue));
        assert_eq!(parse_pipe("trickle(d1:r3)"), Err(PipeError::BadValue));
        // Valid pipes still parse.
        assert!(parse_pipe("status(200)|header(X-Test,yes)|trickle(d1)").is_ok());
    }

    #[test]
    fn wpt_pipe_trickle_microsyntax() {
        // dN: delay in SECONDS before the first byte.
        assert_eq!(
            parse_trickle("d1").unwrap(),
            TricklePlan {
                ops: vec![TrickleOp::DelayMs(1000)],
                repeat_last: None,
            }
        );
        // Fractional seconds round to milliseconds.
        assert_eq!(
            parse_trickle("d1.5").unwrap().ops,
            vec![TrickleOp::DelayMs(1500)]
        );
        // Bare ints are byte counts: chunk the body between delays.
        assert_eq!(
            parse_trickle("100:d1").unwrap(),
            TricklePlan {
                ops: vec![TrickleOp::Bytes(100), TrickleOp::DelayMs(1000)],
                repeat_last: None,
            }
        );
        // A trailing rN (even) repeats the previous N ops until the body
        // is sent.
        assert_eq!(
            parse_trickle("100:d1:r2").unwrap(),
            TricklePlan {
                ops: vec![TrickleOp::Bytes(100), TrickleOp::DelayMs(1000)],
                repeat_last: Some(2),
            }
        );
        // Consecutive same-kind components accumulate, like upstream.
        assert_eq!(
            parse_trickle("d1:d2").unwrap().ops,
            vec![TrickleOp::DelayMs(3000)]
        );
        assert_eq!(
            parse_trickle("50:50:d1").unwrap().ops,
            vec![TrickleOp::Bytes(100), TrickleOp::DelayMs(1000)]
        );
        // Per-directive delay cap (a bowser hardening; upstream is
        // uncapped).
        assert_eq!(
            parse_trickle("d999").unwrap().ops,
            vec![TrickleOp::DelayMs(MAX_TRICKLE_MS)]
        );
        // Malformed scripts are errors, not skips.
        for bad in ["", "x", "d", "d-1", "r2", "100:d1:r3"] {
            assert!(parse_trickle(bad).is_err(), "trickle({bad}) should fail");
        }
        // A repeat block that would send no bytes is dropped instead of
        // hanging the connection the way upstream does.
        assert_eq!(
            parse_trickle("d1:r2").unwrap(),
            TricklePlan {
                ops: vec![TrickleOp::DelayMs(1000)],
                repeat_last: None,
            }
        );
    }

    #[test]
    fn wpt_headers_file_parsing() {
        let text = "# comment\n\
                    X-Frame-Options: DENY\n\
                    Content-Type: text/plain\n\
                    \n\
                    [other.html]\n\
                    X-Other: 1\n\
                    [*]\n\
                    X-Star: yes\n";
        let headers = parse_headers_file(text, "page.html");
        assert_eq!(
            headers,
            vec![
                ("X-Frame-Options".to_owned(), "DENY".to_owned()),
                ("Content-Type".to_owned(), "text/plain".to_owned()),
                ("X-Star".to_owned(), "yes".to_owned()),
            ]
        );
        // Sectioned for another path: only the top-level and `*` apply.
        let headers = parse_headers_file(text, "other.html");
        assert!(headers.iter().any(|(k, _)| k == "X-Other"));
        assert!(headers.iter().any(|(k, _)| k == "X-Star"));
        // Malformed lines are skipped.
        let headers = parse_headers_file("no colon here\n: empty\nGood: yes\n", "x");
        assert_eq!(headers, vec![("Good".to_owned(), "yes".to_owned())]);
    }

    #[test]
    fn wpt_serve_pipe_status_and_header_end_to_end() {
        let root = scratch_dir("pipe-e2e");
        std::fs::write(root.join("page.html"), "<html>pipe</html>").unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, headers, body) =
            http_get(&base, "/page.html?pipe=status(418)|header(X-Test,yes)");
        assert_eq!(status, 418);
        assert_eq!(get_header(&headers, "x-test"), Some("yes"));
        assert_eq!(body, b"<html>pipe</html>");

        // Pipe headers replace by default (upstream parity): last one
        // wins.
        let (status, headers, _) = http_get(&base, "/page.html?pipe=header(X-A,1)|header(X-A,2)");
        assert_eq!(status, 200);
        assert_eq!(header_values(&headers, "x-a"), vec!["2"]);

        // ...and append with the third argument.
        let (status, headers, _) = http_get(
            &base,
            "/page.html?pipe=header(X-A,1,True)|header(X-A,2,True)",
        );
        assert_eq!(status, 200);
        assert_eq!(header_values(&headers, "x-a"), vec!["1", "2"]);

        // Escaped commas stay inside the value.
        let (status, headers, _) = http_get(&base, r"/page.html?pipe=header(X-C,a\,b)");
        assert_eq!(status, 200);
        assert_eq!(get_header(&headers, "x-c"), Some("a,b"));

        // header(X,A,True) is three arguments: value "A", append.
        let (status, headers, _) = http_get(&base, "/page.html?pipe=header(X-T,A,True)");
        assert_eq!(status, 200);
        assert_eq!(get_header(&headers, "x-t"), Some("A"));

        // Pipe directives do NOT apply to error responses: upstream
        // raises 404 before the pipeline runs.
        let (status, _, _) = http_get(&base, "/missing.html?pipe=status(201)");
        assert_eq!(status, 404);

        // Unknown pipe functions and bad arguments are 500s.
        let (status, _, _) = http_get(&base, "/page.html?pipe=bogus(1)");
        assert_eq!(status, 500);
        let (status, _, _) = http_get(&base, "/page.html?pipe=status(abc)");
        assert_eq!(status, 500);

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_headers_sidecar_end_to_end() {
        let root = scratch_dir("headers-e2e");
        std::fs::write(root.join("page.html"), "<html>sidecar</html>").unwrap();
        std::fs::write(
            root.join("page.html.headers"),
            "# sidecar\nX-Frame-Options: DENY\nContent-Type: text/plain\n",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, headers, body) = http_get(&base, "/page.html");
        assert_eq!(status, 200);
        assert_eq!(get_header(&headers, "x-frame-options"), Some("DENY"));
        // A Content-Type entry overrides the sniffed type.
        assert_eq!(get_header(&headers, "content-type"), Some("text/plain"));
        assert_eq!(body, b"<html>sidecar</html>");

        // Pipe headers and sidecar headers compose: sidecar first, pipe
        // after, in order.
        let (status, headers, _) = http_get(&base, "/page.html?pipe=header(X-Pipe,yes)");
        assert_eq!(status, 200);
        assert_eq!(get_header(&headers, "x-frame-options"), Some("DENY"));
        assert_eq!(get_header(&headers, "x-pipe"), Some("yes"));

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_html_variant_beside_any_js() {
        // A `.any.js` sibling must not disturb serving the plain `.html`
        // variant (multi-global expansion is manifest work, out of scope).
        let root = scratch_dir("variant");
        std::fs::write(root.join("test.html"), "<html>variant</html>").unwrap();
        std::fs::write(root.join("test.any.js"), "test(()=>{},'x');").unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, headers, body) = http_get(&base, "/test.html?pipe=status(204)");
        assert_eq!(status, 204);
        assert_eq!(get_header(&headers, "content-type"), Some("text/html"));
        assert_eq!(body, b"<html>variant</html>");

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_any_html_synthesized_from_sibling() {
        // A `.any.html` request with no such file is synthesized from the
        // `.any.js` sibling, in the exact shape of upstream wptserve's
        // `AnyHtmlHandler` wrapper.
        let root = scratch_dir("any-synth");
        std::fs::write(root.join("t.any.js"), "test(()=>{},'x');").unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, headers, body) = http_get(&base, "/t.any.html");
        assert_eq!(status, 200);
        assert_eq!(get_header(&headers, "content-type"), Some("text/html"));
        let body = String::from_utf8(body).unwrap();
        // Byte-for-byte the upstream `AnyHtmlHandler` wrapper shape
        // (`tools/serve/serve.py`, cf. the checked-in `foo.any.html`
        // example in `tools/wptserve/tests/functional/docroot/`).
        assert_eq!(
            body,
            concat!(
                "<!doctype html>\n",
                "<meta charset=utf-8>\n",
                "\n",
                "<script>\n",
                "self.GLOBAL = {\n",
                "  isWindow: function() { return true; },\n",
                "  isWorker: function() { return false; },\n",
                "  isShadowRealm: function() { return false; },\n",
                "};\n",
                "</script>\n",
                "<script src=\"/resources/testharness.js\"></script>\n",
                "<script src=\"/resources/testharnessreport.js\"></script>\n",
                "\n",
                "<div id=log></div>\n",
                "<script src=\"/t.any.js\"></script>\n",
            )
        );

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_any_html_meta_directives() {
        // `// META:` title/timeout/script become wrapper head lines; the
        // `.js` sibling itself is served raw with its own content type.
        let root = scratch_dir("any-meta");
        std::fs::write(
            root.join("t.any.js"),
            "// META: title=Encoding: basics & more\n\
             // META: timeout=long\n\
             // META: script=/common/sab.js\n\
             // META: script=resources/helper.js\n\
             // META: global=window,worker\n\
             test(()=>{},'x');",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, _, body) = http_get(&base, "/t.any.html");
        assert_eq!(status, 200);
        let body = String::from_utf8(body).unwrap();
        assert!(body.contains("<title>Encoding: basics &amp; more</title>"));
        assert!(body.contains("<meta name=\"timeout\" content=\"long\">"));
        assert!(body.contains("<script src=\"/common/sab.js\"></script>"));
        assert!(body.contains("<script src=\"resources/helper.js\"></script>"));
        // META order is preserved: title/timeout before the GLOBAL shim,
        // scripts after testharnessreport.js.
        let title = body.find("<title>").unwrap();
        let shim = body.find("self.GLOBAL").unwrap();
        let th = body.find("testharnessreport.js").unwrap();
        let extra = body.find("/common/sab.js").unwrap();
        let under_test = body.find("<script src=\"/t.any.js\">").unwrap();
        assert!(title < shim && shim < th && th < extra && extra < under_test);

        // The sibling `.js` is still served raw underneath.
        let (status, headers, body) = http_get(&base, "/t.any.js");
        assert_eq!(status, 200);
        assert_eq!(
            get_header(&headers, "content-type"),
            Some("text/javascript; charset=utf-8")
        );
        assert!(body.starts_with(b"// META: title="));

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_any_html_variant_query_and_pipe() {
        // Manifest-style `?variant` queries (e.g. `?1-1000`) must not break
        // the wrapper lookup; `?pipe=` still applies to the response.
        let root = scratch_dir("any-variant");
        std::fs::write(root.join("t.any.js"), "test(()=>{},'x');").unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, _, body) = http_get(&base, "/t.any.html?1-1000");
        assert_eq!(status, 200);
        assert!(String::from_utf8(body).unwrap().contains("/t.any.js"));

        let (status, _, _) = http_get(&base, "/t.any.html?variant=2&pipe=status(201)");
        assert_eq!(status, 201);

        // No sibling at all: 404, like any other missing file.
        let (status, _, _) = http_get(&base, "/missing.any.html");
        assert_eq!(status, 404);

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_any_html_global_exposure() {
        // A `// META: global=` without `window` 404s the window wrapper,
        // like upstream's exposure check.
        let root = scratch_dir("any-global");
        std::fs::write(
            root.join("w.any.js"),
            "// META: global=worker\ntest(()=>{},'x');",
        )
        .unwrap();
        std::fs::write(
            root.join("both.any.js"),
            "// META: global=window,dedicatedworker\ntest(()=>{},'x');",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, _, _) = http_get(&base, "/w.any.html");
        assert_eq!(status, 404);
        let (status, _, _) = http_get(&base, "/both.any.html");
        assert_eq!(status, 200);

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_any_html_uses_sibling_sidecar() {
        // Upstream applies the `.any.js` sibling's `.headers` sidecar to
        // the generated wrapper.
        let root = scratch_dir("any-sidecar");
        std::fs::write(root.join("t.any.js"), "test(()=>{},'x');").unwrap();
        std::fs::write(root.join("t.any.js.headers"), "X-Wrapper: yes\n").unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, headers, _) = http_get(&base, "/t.any.html");
        assert_eq!(status, 200);
        assert_eq!(get_header(&headers, "x-wrapper"), Some("yes"));

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_any_worker_html_404s() {
        // Worker variants need worker/global plumbing this server does not
        // have; they 404 (documented limitation, not silent breakage).
        let root = scratch_dir("any-worker");
        std::fs::write(root.join("t.any.js"), "test(()=>{},'x');").unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, _, _) = http_get(&base, "/t.any.worker.html");
        assert_eq!(status, 404);
        let (status, _, _) = http_get(&base, "/t.any.sharedworker.html");
        assert_eq!(status, 404);

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_window_html_synthesized_no_shim() {
        // `.window.html` synthesizes from the `.window.js` sibling with
        // the upstream `WindowHandler` shape: no `self.GLOBAL` shim, no
        // `global=` exposure check.
        let root = scratch_dir("window-synth");
        std::fs::write(
            root.join("w.window.js"),
            "// META: global=window\ntest(()=>{},'x');",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, headers, body) = http_get(&base, "/w.window.html");
        assert_eq!(status, 200);
        assert_eq!(get_header(&headers, "content-type"), Some("text/html"));
        let body = String::from_utf8(body).unwrap();
        assert!(!body.contains("self.GLOBAL"));
        assert!(body.contains("<script src=\"/resources/testharness.js\"></script>"));
        assert!(body.contains("<script src=\"/w.window.js\"></script>"));

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_any_html_real_file_wins() {
        // A real `.any.html` file on disk is served verbatim; the sibling
        // never shadows it.
        let root = scratch_dir("any-real");
        std::fs::write(root.join("t.any.html"), "<html>real</html>").unwrap();
        std::fs::write(root.join("t.any.js"), "test(()=>{},'x');").unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, _, body) = http_get(&base, "/t.any.html");
        assert_eq!(status, 200);
        assert_eq!(body, b"<html>real</html>");

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_any_html_harness_injection_end_to_end() {
        // The driver path: a synthesized wrapper served through the
        // harness endpoint gets the reporter injected after the
        // testharness.js script tag.
        let root = scratch_dir("any-harness");
        std::fs::write(root.join("t.any.js"), "test(()=>{},'x');").unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();
        let prefix = HARNESS_PREFIX.trim_start_matches('/');

        let (status, headers, body) = http_get(
            &base,
            &format!("/{prefix}t.any.html?__bowser_harness=1&id=t1&nonce=abc123"),
        );
        assert_eq!(status, 200);
        assert_eq!(get_header(&headers, "content-type"), Some("text/html"));
        let body = String::from_utf8(body).unwrap();
        assert!(body.contains("BOWSER_HARNESS_REPORT:"));
        // Injected immediately after the testharness.js script tag, so the
        // reporter's `setup({output:false})` precedes the first `test()`.
        let th = body.find("testharness.js").unwrap();
        let injected = body.find("BOWSER_HARNESS_REPORT:").unwrap();
        let under_test = body.find("<script src=\"/t.any.js\">").unwrap();
        assert!(th < injected && injected < under_test);

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn any_window_exposed_variants() {
        // Unit coverage for the exposure check without the server.
        assert!(any_window_exposed(None));
        assert!(any_window_exposed(Some("")));
        assert!(any_window_exposed(Some("window")));
        assert!(any_window_exposed(Some("window,worker")));
        assert!(any_window_exposed(Some(" window , dedicatedworker ")));
        assert!(!any_window_exposed(Some("worker")));
        assert!(!any_window_exposed(Some("dedicatedworker")));
        assert!(!any_window_exposed(Some("shadowrealm")));
        assert!(!any_window_exposed(Some("bogus")));
    }

    #[test]
    fn parse_wrapper_meta_stops_at_first_non_directive() {
        // Upstream stops scanning at the first non-META line; a later
        // `// META:` is code, not metadata. CRLF is tolerated.
        let js = b"// META: title=Real\r\ntest(() => {\r\n  // META: title=Fake\r\n});\r\n";
        let meta = parse_wrapper_meta(js);
        assert_eq!(meta.head_lines.len(), 1);
        assert!(meta.head_lines[0].contains("Real"));
        assert!(meta.global.is_none());

        // A malformed directive (no `=`) ends the block too.
        let js = b"// META: bogus\n// META: title=Late\n";
        let meta = parse_wrapper_meta(js);
        assert!(meta.head_lines.is_empty());
    }

    #[test]
    fn wpt_serve_trickle_delays_first_byte() {
        let root = scratch_dir("trickle");
        std::fs::write(root.join("page.html"), "<html>trickle</html>").unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        // Canonical form: `dN` is a pre-first-byte delay in SECONDS.
        let start = Instant::now();
        let (status, _, _) = http_get(&base, "/page.html?pipe=trickle(d1)");
        let elapsed = start.elapsed();
        assert_eq!(status, 200);
        assert!(
            elapsed >= Duration::from_millis(900),
            "trickle delay too short: {elapsed:?}"
        );

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_trickle_chunks_body_with_repeat() {
        let root = scratch_dir("trickle-chunk");
        let body = vec![b'x'; 100];
        std::fs::write(root.join("page.html"), &body).unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        // 10 bytes, 50ms delay, repeated via r2 until the body is out:
        // 10 passes * 50ms ~= 500ms, and the full body still arrives.
        let start = Instant::now();
        let (status, _, got) = http_get(&base, "/page.html?pipe=trickle(10:d0.05:r2)");
        let elapsed = start.elapsed();
        assert_eq!(status, 200);
        assert_eq!(got, body);
        assert!(
            elapsed >= Duration::from_millis(400),
            "trickle chunking too fast: {elapsed:?}"
        );

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_harness_endpoint_applies_pipe_and_sidecar() {
        let root = scratch_dir("harness-pipe");
        std::fs::write(root.join("page.html"), "<html><body>hi</body></html>").unwrap();
        std::fs::write(root.join("page.html.headers"), "X-Frame-Options: DENY\n").unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();
        let prefix = HARNESS_PREFIX.trim_start_matches('/');

        // Raw arm (no __bowser_harness flag): pipe + sidecar apply.
        let (status, headers, body) = http_get(
            &base,
            &format!("/{prefix}page.html?pipe=status(418)|header(X-Test,yes)"),
        );
        assert_eq!(status, 418);
        assert_eq!(get_header(&headers, "x-test"), Some("yes"));
        assert_eq!(get_header(&headers, "x-frame-options"), Some("DENY"));
        assert_eq!(body, b"<html><body>hi</body></html>");

        // Inject arm: the reporter goes in and the pipe still applies.
        let (status, headers, body) = http_get(
            &base,
            &format!(
                "/{prefix}page.html?__bowser_harness=1&id=t1&nonce=abc123&pipe=header(X-Pipe,yes)"
            ),
        );
        assert_eq!(status, 200);
        assert_eq!(get_header(&headers, "x-pipe"), Some("yes"));
        assert_eq!(get_header(&headers, "x-frame-options"), Some("DENY"));
        let body = String::from_utf8(body).unwrap();
        assert!(body.contains("BOWSER_HARNESS_REPORT:"));

        // A bad pipe is a 500 on the harness endpoint too.
        let (status, _, _) = http_get(&base, &format!("/{prefix}page.html?pipe=bogus(1)"));
        assert_eq!(status, 500);

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    /// Raw HTTP request with an explicit method, headers, and body;
    /// returns (status, headers, body) like [`http_get`].
    fn http_raw(
        base: &str,
        method: &str,
        target: &str,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> (u16, Vec<(String, String)>, Vec<u8>) {
        let mut stream = TcpStream::connect(base.trim_start_matches("http://")).expect("connect");
        write!(stream, "{method} {target} HTTP/1.1\r\n").unwrap();
        for (k, v) in headers {
            write!(stream, "{k}: {v}\r\n").unwrap();
        }
        if !body.is_empty() {
            write!(stream, "Content-Length: {}\r\n", body.len()).unwrap();
        }
        write!(stream, "Connection: close\r\n\r\n").unwrap();
        stream.write_all(body).unwrap();
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).unwrap();
        let split = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .expect("no header/body split");
        let head = String::from_utf8(raw[..split].to_vec()).unwrap();
        let mut lines = head.lines();
        let status: u16 = lines
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let headers = lines
            .map(|l| {
                let (k, v) = l.split_once(':').unwrap();
                (k.trim().to_lowercase(), v.trim().to_owned())
            })
            .collect();
        (status, headers, raw[split + 4..].to_vec())
    }

    #[test]
    fn wpt_serve_py_echo_get_and_post() {
        // `.py` handlers execute: the source is never served, the
        // return value maps like upstream's FunctionHandler.
        let root = scratch_dir("py-echo");
        std::fs::write(
            root.join("echo.py"),
            "def main(request, response):\n    return [(b\"X-Method\", request.method.encode()),\n            (b\"Content-Type\", b\"text/plain\")], request.body\n",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, headers, body) = http_get(&base, "/echo.py");
        assert_eq!(status, 200);
        assert_eq!(get_header(&headers, "x-method"), Some("GET"));
        assert_eq!(get_header(&headers, "content-type"), Some("text/plain"));
        assert!(body.is_empty());
        // The source must not leak through.
        assert!(!body.windows(4).any(|w| w == b"def "));

        let (status, headers, body) =
            http_raw(&base, "POST", "/echo.py", &[("Host", "x")], b"hello");
        assert_eq!(status, 200);
        assert_eq!(get_header(&headers, "x-method"), Some("POST"));
        assert_eq!(body, b"hello");

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_py_status_tuple_and_string() {
        let root = scratch_dir("py-tuple");
        std::fs::write(
            root.join("t.py"),
            "def main(request, response):\n    return (201, b\"Created\"), [(b\"X-A\", b\"1\")], b\"hi\"\n",
        )
        .unwrap();
        std::fs::write(
            root.join("s.py"),
            "def main(request, response):\n    return 'plain'\n",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, headers, body) = http_get(&base, "/t.py");
        assert_eq!(status, 201);
        assert_eq!(get_header(&headers, "x-a"), Some("1"));
        assert_eq!(body, b"hi");

        let (status, _, body) = http_get(&base, "/s.py");
        assert_eq!(status, 200);
        assert_eq!(body, b"plain");

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_py_get_params_and_post_form() {
        let root = scratch_dir("py-params");
        std::fs::write(
            root.join("p.py"),
            "def main(request, response):\n    a = request.GET.first(b\"a\", b\"none\")\n    b = request.POST.first(b\"b\", b\"none\")\n    return a + b\"|\" + b\n",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, _, body) = http_get(&base, "/p.py?a=1&a=2");
        assert_eq!(status, 200);
        assert_eq!(body, b"1|none");

        let (status, _, body) = http_raw(
            &base,
            "POST",
            "/p.py",
            &[
                ("Host", "x"),
                ("Content-Type", "application/x-www-form-urlencoded"),
            ],
            b"b=2",
        );
        assert_eq!(status, 200);
        assert_eq!(body, b"none|2");

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_py_stash_roundtrip() {
        // The daemon-wide stash persists across connections; `take`
        // is destructive, like upstream.
        let root = scratch_dir("py-stash");
        std::fs::write(
            root.join("put.py"),
            "def main(request, response):\n    key = request.GET.first(b\"key\")\n    request.server.stash.put(key, b\"v\", \"/\")\n    return b\"ok\"\n",
        )
        .unwrap();
        std::fs::write(
            root.join("take.py"),
            "from wptserve.handlers import json_handler\n\n@json_handler\ndef main(request, response):\n    key = request.GET.first(b\"key\")\n    v = request.server.stash.take(key, \"/\")\n    return v.decode() if v else None\n",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, _, _) = http_get(&base, "/put.py?key=k1");
        assert_eq!(status, 200);
        let (status, headers, body) = http_get(&base, "/take.py?key=k1");
        assert_eq!(status, 200);
        assert_eq!(
            get_header(&headers, "content-type"),
            Some("application/json")
        );
        assert_eq!(body, b"\"v\"");
        // Second take finds nothing.
        let (status, _, body) = http_get(&base, "/take.py?key=k1");
        assert_eq!(status, 200);
        assert_eq!(body, b"null");

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_py_errors() {
        let root = scratch_dir("py-errors");
        std::fs::write(
            root.join("boom.py"),
            "def main(request, response):\n    raise RuntimeError('kablam')\n",
        )
        .unwrap();
        std::fs::write(root.join("nomain.py"), "x = 1\n").unwrap();
        std::fs::write(
            root.join("h2.py"),
            "def handle_headers(frame, request, response):\n    pass\n",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        // An uncaught exception is a 500 with the upstream JSON error shape.
        let (status, headers, body) = http_get(&base, "/boom.py");
        assert_eq!(status, 500);
        assert_eq!(
            get_header(&headers, "content-type"),
            Some("application/json")
        );
        let body = String::from_utf8(body).unwrap();
        assert!(body.contains("\"code\": 500"));
        assert!(body.contains("kablam"));

        // No `main` (including `.h2.py`-style scripts on H1) is a 500.
        let (status, _, _) = http_get(&base, "/nomain.py");
        assert_eq!(status, 500);
        let (status, _, _) = http_get(&base, "/h2.py");
        assert_eq!(status, 500);

        // A missing handler is a 404, like upstream.
        let (status, _, _) = http_get(&base, "/missing.py");
        assert_eq!(status, 404);

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_py_streaming() {
        // `write_status_headers` + `writer.write` streams with no
        // content-length, like upstream's `end_headers`.
        let root = scratch_dir("py-stream");
        std::fs::write(
            root.join("s.py"),
            "def main(request, response):\n    response.headers.set(b\"X-S\", b\"1\")\n    response.write_status_headers()\n    response.writer.write(b\"ab\")\n    response.writer.write(b\"cd\")\n",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, headers, body) = http_get(&base, "/s.py");
        assert_eq!(status, 200);
        assert_eq!(get_header(&headers, "x-s"), Some("1"));
        assert_eq!(get_header(&headers, "content-length"), None);
        assert_eq!(body, b"abcd");

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_py_pipe_applies_to_return_values() {
        // Upstream runs `?pipe=` on `.py` output only when the handler
        // returned a value without streaming.
        let root = scratch_dir("py-pipe");
        std::fs::write(
            root.join("v.py"),
            "def main(request, response):\n    return b\"x\"\n",
        )
        .unwrap();
        std::fs::write(
            root.join("n.py"),
            "def main(request, response):\n    response.content = b\"y\"\n",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, _, body) = http_get(&base, "/v.py?pipe=status(418)");
        assert_eq!(status, 418);
        assert_eq!(body, b"x");

        // `rv is None`: no pipe processing, like upstream.
        let (status, _, body) = http_get(&base, "/n.py?pipe=status(418)");
        assert_eq!(status, 200);
        assert_eq!(body, b"y");

        // A bad pipe is a 500 on `.py` too.
        let (status, _, _) = http_get(&base, "/v.py?pipe=bogus(1)");
        assert_eq!(status, 500);

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_py_options_reaches_handler() {
        // Preflight-style OPTIONS must reach the handler, not 405.
        let root = scratch_dir("py-options");
        std::fs::write(
            root.join("p.py"),
            "def main(request, response):\n    response.headers.set(b\"X-M\", request.method.encode())\n    return b\"\"\n",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, headers, _) = http_raw(&base, "OPTIONS", "/p.py", &[("Host", "x")], &[]);
        assert_eq!(status, 200);
        assert_eq!(get_header(&headers, "x-m"), Some("OPTIONS"));

        // Static files stay GET-only.
        std::fs::write(root.join("f.html"), "<html></html>").unwrap();
        let (status, _, _) = http_raw(&base, "OPTIONS", "/f.html", &[("Host", "x")], &[]);
        assert_eq!(status, 405);

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_py_set_cookie() {
        let root = scratch_dir("py-cookie");
        std::fs::write(
            root.join("c.py"),
            "import datetime\ndef main(request, response):\n    response.set_cookie(b\"n\", b\"v\", expires=datetime.timedelta(days=1))\n    return b\"\"\n",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, headers, _) = http_get(&base, "/c.py");
        assert_eq!(status, 200);
        let sc = get_header(&headers, "set-cookie").expect("set-cookie");
        assert!(sc.starts_with("n=v"), "{sc}");
        assert!(sc.contains("expires="), "{sc}");
        assert!(sc.contains("GMT"), "{sc}");

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_subdomain_alias_routing() {
        // Every alias IP serves the same checkout.
        let root = scratch_dir("alias");
        std::fs::write(root.join("page.html"), "<html>alias</html>").unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();
        let port: u16 = base
            .trim_start_matches("http://127.0.0.1:")
            .parse()
            .unwrap();

        for ip in [
            "127.0.0.1",
            "127.0.0.2",
            "127.0.0.3",
            "127.0.0.4",
            "127.0.0.5",
            "127.0.0.6",
        ] {
            let alias_base = format!("http://{ip}:{port}");
            let (status, _, body) = http_get(&alias_base, "/page.html");
            assert_eq!(status, 200, "alias {ip}");
            assert_eq!(body, b"<html>alias</html>");
        }

        // An arbitrary Host header still routes to the checkout.
        let (status, _, _) = http_raw(&base, "GET", "/page.html", &[("Host", "evil.example")], &[]);
        assert_eq!(status, 200);

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_sub_template_hosts_and_ports() {
        let root = scratch_dir("sub-basic");
        std::fs::write(
            root.join("t.sub.html"),
            "<a href=\"http://{{host}}:{{ports[http][0]}}/\">x</a>\n\
             <a href=\"http://{{hosts[][www]}}:{{ports[http][0]}}/\">w</a>\n\
             <a href=\"http://{{hosts[alt][]}}:{{ports[http][0]}}/\">a</a>\n\
             <a href=\"http://{{domains[www1]}}:{{ports[http][0]}}/\">w1</a>\n\
             <span>{{location[scheme]}}://{{location[host]}}{{location[path]}}</span>\n",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();
        let port: u16 = base
            .trim_start_matches("http://127.0.0.1:")
            .parse()
            .unwrap();

        let (status, headers, body) = http_get(&base, "/t.sub.html");
        assert_eq!(status, 200);
        assert_eq!(get_header(&headers, "content-type"), Some("text/html"));
        let body = String::from_utf8(body).unwrap();
        assert!(
            body.contains(&format!("http://127.0.0.1:{port}/")),
            "{body}"
        );
        assert!(
            body.contains(&format!("http://127.0.0.2:{port}/")),
            "{body}"
        );
        assert!(
            body.contains(&format!("http://127.0.0.5:{port}/")),
            "{body}"
        );
        assert!(
            body.contains(&format!("http://127.0.0.3:{port}/")),
            "{body}"
        );
        // `{{location}}` reflects the request (Host: x here).
        assert!(body.contains("http://x:"), "{body}");
        assert!(body.contains("/t.sub.html"), "{body}");

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_sub_template_escaping_and_vars() {
        // Markup files escape HTML; `.sub.js` does not. `{{$v:…}}`
        // assignment and `{{GET[…]}}` / `{{uuid()}}` work.
        let root = scratch_dir("sub-esc");
        std::fs::write(
            root.join("e.sub.html"),
            "{{header_or_default(X-Missing, <d&f>)}}\n{{$id:uuid()}}{{$id}}\n",
        )
        .unwrap();
        std::fs::write(root.join("j.sub.js"), "var h = \"{{host}}\"; // <tag>\n").unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, _, body) = http_get(&base, "/e.sub.html");
        assert_eq!(status, 200);
        let body = String::from_utf8(body).unwrap();
        assert!(body.contains("&lt;d&amp;f&gt;"), "{body}");
        // The uuid appears twice (assignment + use) and parses.
        let ids: Vec<&str> = body
            .lines()
            .nth(1)
            .unwrap()
            .as_bytes()
            .chunks(36)
            .map(|c| std::str::from_utf8(c).unwrap())
            .collect();
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], ids[1]);
        assert_eq!(ids[0].len(), 36);
        assert_eq!(&ids[0][14..15], "4"); // version 4

        let (status, _, body) = http_get(&base, "/j.sub.js?x=1");
        assert_eq!(status, 200);
        let body = String::from_utf8(body).unwrap();
        assert!(body.contains("var h = \"127.0.0.1\"; // <tag>"), "{body}");

        // Unknown variables are a 500, like upstream.
        std::fs::write(root.join("b.sub.html"), "{{bogus_variable}}\n").unwrap();
        let (status, _, _) = http_get(&base, "/b.sub.html");
        assert_eq!(status, 500);

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_sub_template_get_param() {
        let root = scratch_dir("sub-get");
        std::fs::write(
            root.join("g.sub.html"),
            "[{{GET[name]}}][{{GET[missing]}}]\n",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, _, body) = http_get(&base, "/g.sub.html?name=va%20l");
        assert_eq!(status, 200);
        assert_eq!(body, b"[va l][]\n");

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn alias_ip_table() {
        assert_eq!(alias_ip("", ""), Some("127.0.0.1"));
        assert_eq!(alias_ip("", "www"), Some("127.0.0.2"));
        assert_eq!(alias_ip("", "www1"), Some("127.0.0.3"));
        assert_eq!(alias_ip("", "www2"), Some("127.0.0.4"));
        assert_eq!(alias_ip("alt", ""), Some("127.0.0.5"));
        assert_eq!(alias_ip("alt", "www"), Some("127.0.0.6"));
        assert_eq!(alias_ip("", "bogus"), None);
        assert_eq!(alias_ip("bogus", ""), None);
        // Every non-primary alias is bound by `serve_dir`.
        for ip in SUBDOMAIN_ALIAS_IPS {
            assert_ne!(*ip, "127.0.0.1");
        }
    }

    #[test]
    fn b64_encode_vectors() {
        assert_eq!(b64_encode(b""), "");
        assert_eq!(b64_encode(b"f"), "Zg==");
        assert_eq!(b64_encode(b"fo"), "Zm8=");
        assert_eq!(b64_encode(b"foo"), "Zm9v");
        assert_eq!(b64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(b64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(b64_encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(b64_encode(&[0xff, 0xfe, 0xfd]), "//79");
    }

    #[test]
    fn tokenize_sub_shapes() {
        let toks = tokenize_sub(b"$id:uuid()");
        assert!(matches!(toks[0], SubToken::Var(ref s) if s == "$id"));
        assert!(matches!(toks[1], SubToken::Ident(ref s) if s == "uuid"));
        assert!(matches!(toks[2], SubToken::Args(ref a) if a.is_empty()));

        let toks = tokenize_sub(b"hosts[alt][www]");
        assert!(matches!(toks[0], SubToken::Ident(ref s) if s == "hosts"));
        assert!(matches!(toks[1], SubToken::Index(SubIndex::Key(ref s)) if s == "alt"));
        assert!(matches!(toks[2], SubToken::Index(SubIndex::Key(ref s)) if s == "www"));

        let toks = tokenize_sub(b"ports[http][0]");
        assert!(matches!(toks[2], SubToken::Index(SubIndex::Int(0))));

        let toks = tokenize_sub(b"header_or_default(X-Test, dflt)");
        assert!(
            matches!(toks[1], SubToken::Args(ref a) if a == &["X-Test".to_owned(), "dflt".to_owned()])
        );
    }

    // ============================================================
    // `.py` daemon hardening (wave-44 review findings)
    // ============================================================

    fn py_test_request<'a>(script: &'a Path, root: &'a Path) -> PyHandlerRequest<'a> {
        PyHandlerRequest {
            script,
            method: "GET",
            target: "/t.py",
            url: "http://127.0.0.1:1/t.py".to_owned(),
            headers: &[],
            body: &[],
            doc_root: root,
            peer: "127.0.0.1:1",
        }
    }

    #[test]
    fn read_py_frame_rejects_bad_kind() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&7u64.to_le_bytes());
        buf.push(9); // unknown kind byte
        let mut cur = std::io::Cursor::new(buf);
        let err = read_py_frame(&mut cur).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn read_py_frame_rejects_overlong_header_len() {
        // A header-name length past the cap is rejected without reading
        // the bytes behind it.
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.push(0); // Response
        buf.push(0); // flags
        buf.extend_from_slice(&200u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes()); // reason_len
        buf.extend_from_slice(&1u32.to_le_bytes()); // nheaders
        buf.extend_from_slice(&(MAX_PY_HEADER_BYTES as u32 + 1).to_le_bytes());
        let mut cur = std::io::Cursor::new(buf);
        let err = read_py_frame(&mut cur).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn read_py_frame_rejects_eof_mid_frame() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.push(1); // Head, then truncated
        let mut cur = std::io::Cursor::new(buf);
        let err = read_py_frame(&mut cur).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn read_py_frame_rejects_lying_body_len() {
        // A 256MB+1 body prefix must fail the cap check, not allocate.
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.push(2); // Chunk
        buf.extend_from_slice(&(MAX_PY_FRAME_BODY_BYTES as u64 + 1).to_le_bytes());
        let mut cur = std::io::Cursor::new(buf);
        let err = read_py_frame(&mut cur).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn py_frame_response_is_terminal() {
        let response = PyFrame::Response {
            id: 1,
            apply_pipes: false,
            status: 200,
            reason: Vec::new(),
            headers: Vec::new(),
            body: Vec::new(),
        };
        assert!(py_frame_is_terminal(&response));
        assert!(py_frame_is_terminal(&PyFrame::End { id: 1 }));
        assert!(!py_frame_is_terminal(&PyFrame::Head {
            id: 1,
            status: 200,
            reason: Vec::new(),
            headers: Vec::new(),
        }));
        assert!(!py_frame_is_terminal(&PyFrame::Chunk {
            id: 1,
            data: Vec::new(),
        }));
    }

    #[test]
    fn py_pending_entry_forgotten_after_response() {
        // A `Response` answers the request completely: the reader must
        // forget the `pending` entry, not leak one per completed request.
        let root = scratch_dir("py-pending");
        let script = root.join("t.py");
        std::fs::write(&script, "def main(request, response):\n    return b\"x\"\n").unwrap();
        let handle = Arc::new(PyDaemonHandle::new());
        let req = py_test_request(&script, &root);
        let (_id, rx) = py_call(&handle, &req).expect("dispatch");
        let frame = rx.recv_timeout(PY_FIRST_BYTE_TIMEOUT).expect("response");
        assert!(matches!(frame, PyFrame::Response { .. }));
        let daemon = handle.daemon().expect("daemon");
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if daemon
                .pending
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty()
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "pending entry leaked after Response frame"
            );
            thread::sleep(Duration::from_millis(10));
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn py_daemon_drop_kills_child() {
        // Dropping the handle must not orphan `python3`: the child is
        // SIGKILLed and reaped even though the reader thread's
        // `Arc<PyDaemon>` clone keeps the daemon struct alive.
        let handle = Arc::new(PyDaemonHandle::new());
        let pid = handle
            .daemon()
            .expect("spawn python3 daemon")
            .child
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .id();
        drop(handle);
        let proc = format!("/proc/{pid}");
        let deadline = Instant::now() + Duration::from_secs(10);
        while std::path::Path::new(&proc).exists() {
            assert!(
                Instant::now() < deadline,
                "python3 child {pid} survived PyDaemonHandle drop"
            );
            thread::sleep(Duration::from_millis(50));
        }
    }

    #[test]
    fn py_stream_chunk_timeout_aborts_wedged_handler() {
        // A handler that sends its head then wedges must not pin the
        // connection thread: `drive` gives up after the chunk timeout.
        let root = scratch_dir("py-chunk-timeout");
        let script = root.join("slow.py");
        std::fs::write(
            &script,
            "import time\ndef main(request, response):\n    response.write_status_headers()\n    time.sleep(5)\n",
        )
        .unwrap();
        let handle = Arc::new(PyDaemonHandle::new());
        let req = py_test_request(&script, &root);
        let (id, rx) = py_call(&handle, &req).expect("dispatch");
        let (status, reason, headers) = match rx.recv_timeout(PY_FIRST_BYTE_TIMEOUT).expect("head")
        {
            PyFrame::Head {
                status,
                reason,
                headers,
                ..
            } => (status, reason, headers),
            _ => panic!("expected Head frame"),
        };
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut stream = TcpStream::connect(addr).unwrap();
        let _ = listener.accept().unwrap();
        let ps = PyStream {
            status,
            reason,
            headers,
            rx,
            py: Arc::clone(&handle),
            id,
        };
        let start = Instant::now();
        ps.drive_with_timeout(&mut stream, Duration::from_millis(500))
            .expect("drive");
        assert!(
            start.elapsed() < Duration::from_secs(4),
            "wedged handler pinned the connection"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn wpt_serve_py_stray_print_does_not_break_daemon() {
        // A handler's stray print() must not corrupt the frame stream:
        // the shim rebinds stdout to stderr. Without the rebind the
        // reader dies on the garbage bytes and later requests stall.
        let root = scratch_dir("py-print");
        std::fs::write(
            root.join("p.py"),
            "def main(request, response):\n    print(\"stray print must not corrupt the frame stream\")\n    return b\"ok\"\n",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();
        for _ in 0..2 {
            let (status, _, body) = http_get(&base, "/p.py");
            assert_eq!(status, 200);
            assert_eq!(body, b"ok");
        }
        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_py_daemon_death_is_500_then_recovers() {
        // The daemon dying mid-run must fail the in-flight request with
        // a 500 (not a hang), and later requests must recover via
        // respawn instead of failing fast forever.
        let root = scratch_dir("py-death");
        std::fs::write(
            root.join("kill.py"),
            "import os\ndef main(request, response):\n    os._exit(1)\n",
        )
        .unwrap();
        std::fs::write(
            root.join("ok.py"),
            "def main(request, response):\n    return b\"alive\"\n",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        let (status, _, _) = http_get(&base, "/kill.py");
        assert_eq!(status, 500);

        let mut recovered = false;
        for _ in 0..20 {
            let (status, _, body) = http_get(&base, "/ok.py");
            if status == 200 && body == b"alive" {
                recovered = true;
                break;
            }
            thread::sleep(Duration::from_millis(200));
        }
        assert!(recovered, "daemon did not recover after mid-run death");

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_py_daemon_death_mid_stream_closes_cleanly() {
        // The head and the first chunk made it out before the daemon
        // died: the connection must close cleanly, not hang.
        let root = scratch_dir("py-death-stream");
        std::fs::write(
            root.join("d.py"),
            "import os\ndef main(request, response):\n    response.write_status_headers()\n    response.writer.write(b\"part\")\n    os._exit(1)\n",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();
        let (status, _, body) = http_get(&base, "/d.py");
        assert_eq!(status, 200);
        assert_eq!(body, b"part");
        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_py_crlf_injection_is_500() {
        // CR/LF smuggled into a header value, the reason phrase, or a
        // streaming head is response splitting: 500, and the bytes must
        // never reach the wire.
        let root = scratch_dir("py-crlf");
        std::fs::write(
            root.join("h.py"),
            "def main(request, response):\n    return [(b\"X-Evil\", b\"1\\r\\nX-Injected: yes\")], b\"x\"\n",
        )
        .unwrap();
        std::fs::write(
            root.join("r.py"),
            "def main(request, response):\n    return (200, b\"OK\\r\\nX-Injected: yes\"), [], b\"x\"\n",
        )
        .unwrap();
        std::fs::write(
            root.join("s.py"),
            "def main(request, response):\n    response.writer.write_status(200, \"Evil\\r\\nX-Injected: yes\")\n    response.write_status_headers()\n",
        )
        .unwrap();
        std::fs::write(
            root.join("clean.py"),
            "def main(request, response):\n    return b\"ok\"\n",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();

        for path in ["/h.py", "/r.py", "/s.py"] {
            let (status, headers, _) = http_get(&base, path);
            assert_eq!(status, 500, "{path}");
            assert_eq!(get_header(&headers, "x-injected"), None, "{path}");
        }
        // The daemon is unpoisoned: a later clean request still works.
        let (status, _, body) = http_get(&base, "/clean.py");
        assert_eq!(status, 200);
        assert_eq!(body, b"ok");

        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn wpt_serve_py_harness_prefix_executes() {
        // A `.py` URL under the harness prefix executes the handler; it
        // must not serve the source.
        let root = scratch_dir("py-harness");
        std::fs::write(
            root.join("h.py"),
            "def main(request, response):\n    return b\"executed\"\n",
        )
        .unwrap();
        let (base, shutdown) = FixtureServer::serve_dir(&root).unwrap();
        let (status, _, body) = http_get(&base, "/__bowser_harness__/h.py");
        assert_eq!(status, 200);
        assert_eq!(body, b"executed");
        let _ = std::fs::remove_dir_all(&root);
        drop(shutdown);
    }

    #[test]
    fn py_dispatch_log_rate_limited() {
        // A stale timestamp lets one line through…
        LAST_PY_DISPATCH_LOG_SECS.store(0, Ordering::Relaxed);
        let _ = note_py_dispatch_failed("test");
        // …and immediate repeats are suppressed.
        assert!(!note_py_dispatch_failed("test"));
        // A recent timestamp also suppresses.
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        LAST_PY_DISPATCH_LOG_SECS.store(now, Ordering::Relaxed);
        assert!(!note_py_dispatch_failed("test"));
    }
}
