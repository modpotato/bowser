//! Per-page JS VM plumbing: `<script>` collection, fetching, execution,
//! event-loop pumping, and inline-handler dispatch.
//!
//! On every full navigation the loader (step 5 of [`crate::loader`])
//! collects the document's `<script>` elements in document order and
//! fetches external sources; [`crate::navigation::install_loaded_document`]
//! then creates a fresh [`PageVm`] — one JS realm per document, like a new
//! `Window` — and evaluates each classic script through `bowser_js::Vm`.
//!
//! # What runs
//!
//! Classic scripts only: no `type` attribute, or a JavaScript MIME type
//! (mirroring `bowser_js`'s executable-type list). `type="module"` and
//! data blocks (`type="application/json"`, ...) are recorded as skipped.
//! `defer` is honored as execution order, and so is `async`: the loader
//! fires `async` external fetches concurrently (they never wait for the
//! document-order stream), and [`PageVm::run`] executes in three passes —
//! classic scripts in document order (the spec's during-parsing
//! execution; the loader hands us scripts post-parse, so this is the
//! faithful mapping), then `async` scripts as one order-independent batch
//! (the spec's execute-as-soon-as-available, mapped onto the pipeline:
//! every source is already fetched, so the batch runs after the sync
//! stream and makes no document-order promise), then deferred external
//! scripts in document order. Both `async` and `defer` run before
//! `DOMContentLoaded`, which fires after parsing completes and before
//! the window `load` event. `async` wins over `defer` when both are
//! present, per spec; `async`/`defer` without `src` are ignored.
//!
//! # v1 divergences (documented, not silently wrong)
//!
//! * The web-platform bindings (`window`, `document`, `location`,
//!   `navigator`, `fetch`, `getComputedStyle`, storage/history, `URL`,
//!   ...) are installed on every page realm created with
//!   [`PageVm::new_for_document`] (via
//!   [`Vm::install_page_web_apis`](bowser_js::Vm::install_page_web_apis),
//!   one install per page realm, before the first script runs). The
//!   timer/microtask globals (`setTimeout`, `setInterval`,
//!   `queueMicrotask`, ...) are installed by the bindings and shadow the
//!   VM's own timer intrinsics; they fire through the bindings pump hook
//!   ([`bowser_js::pump_page_event_loop`]), which [`PageVm::pump`] drives.
//!   Event listeners *do* dispatch:
//!   `addEventListener`/`dispatchEvent` re-enter the running VM through
//!   the engine's sanctioned native-to-JS choke point. `fetch()`
//!   dispatches and its promise settles heap-side on completion; `.then`
//!   reactions run when the pump moves the completion into the loop as a
//!   microtask.
//! * Script failures never fail the navigation (report-and-continue, like
//!   the browser error console): they land in [`ScriptReport::errors`],
//!   and failures from later event-dispatch evals join them through
//!   [`PageVm::record_error`].
//! * No microtask checkpoint runs *during* a script, only between
//!   scripts; a script that never returns still starves later ones, same
//!   as a blocking classic script on the web.
//! * Timers registered by page scripts (the bindings' `setTimeout` and
//!   friends, on the page's shared event loop) fire when something pumps
//!   the loop:
//!   [`PageVm::pump`] is that pump — the dispatcher
//!   ([`crate::window::JsDispatch`]) calls it after every event,
//!   and the window worker calls it on its own tick. Uncaught throws from
//!   timer callbacks (and script tasks) are drained by
//!   [`PageVm::pump`] and land in [`ScriptReport::errors`] as node-less
//!   "page task" errors — the error console sees them instead of the
//!   event loop swallowing them. (Unhandled promise rejections are still
//!   silent: the engine has no unhandled-rejection tracking yet.)
//! * Cross-script state rides the global *object*, not lexical scope:
//!   each `Vm::eval` compiles its script as an isolated function scope,
//!   so top-level `var`/`let`/`const` (and function declarations) do not
//!   survive into later scripts or dispatch evals — a known engine gap
//!   (the spec binds top-level `var`s to the global object, and keeps
//!   `let`/`const` in a global declarative environment). Fixing it is not
//!   an embedding change: `bowser_js` would need a global lexical
//!   environment — the compiler emitting global-scope declarations into
//!   a declarative global env record that the interpreter keeps alive
//!   across evals. Until then, what persists is properties of the global
//!   object — sloppy-mode assignments (`x = 1`), globals planted with
//!   [`PageVm::install_global`], and the builtins. Tests assert this
//!   contract, not the spec's.
//! * Event dispatch is inline handlers only: the dispatcher runs a node's
//!   `onclick` / `onkeydown` attribute source through [`PageVm::eval`].
//!   There is no bubbling — but `addEventListener`/`dispatchEvent` from
//!   the installed bindings do fire listeners (without bubbling or an
//!   `Event` object carrying key state: a handler body cannot see which
//!   key was pressed). Bubbling needs an `Event` object plumbed through
//!   dispatch plus propagation through the bindings' listener lists;
//!   the dispatcher is inline-handlers-only by design for v1.

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;
use std::sync::Mutex;

use bowser_css::Stylesheet;
use bowser_dom::{Document, NodeId, NodeKind, PointerData, Touch, TouchPhase};
use bowser_js::{
    PointerMouseKind, Value, Vm, VmError, drain_page_microtasks, pump_page_event_loop,
    pump_page_event_loop_with_budget,
};
use bowser_layout::LayoutBox;
use bowser_net::cookies::CookieJar;
use bowser_net::fetch::{NavigationContext, Request, fetch_with_jar_in_context};
use bowser_url::Url;

/// Maximum external `<script>` sources fetched per document. Extra
/// `<script src>` elements are skipped, never fetched — the same cap
/// discipline as the loader's stylesheet and image limits.
pub const MAX_SCRIPTS: usize = 128;

// ---------------------------------------------------------------------------
// Collection
// ---------------------------------------------------------------------------

/// One `<script>` element as collected from the document, in document
/// order, before fetching and type classification.
#[derive(Debug, Clone)]
pub struct ScriptRef {
    /// The `<script>` element.
    pub node: NodeId,
    /// Raw `src` attribute value, if present.
    pub src: Option<String>,
    /// Raw `type` attribute value, if present.
    pub script_type: Option<String>,
    /// Whether the `defer` attribute is present. Only meaningful with
    /// `src` (per spec `defer` on an inline script is ignored);
    /// [`resolve_scripts`] turns it into the effective deferral on
    /// [`PageScript::External`].
    pub defer: bool,
    /// Whether the `async` attribute is present. Only meaningful with
    /// `src` (per spec `async` on an inline script is ignored);
    /// [`resolve_scripts`] turns it into the effective flag on
    /// [`PageScript::External`]. Wins over [`ScriptRef::defer`] when both
    /// are present, per spec.
    pub is_async: bool,
    /// The element's raw text (the inline source when there is no `src`).
    pub inline_text: String,
}

/// Collect every `<script>` element in document order.
pub fn collect_script_refs(document: &Document) -> Vec<ScriptRef> {
    let mut out = Vec::new();
    for node in document.descendants(document.root()) {
        let NodeKind::Element(data) = document.kind(node) else {
            continue;
        };
        if data.tag_name != "script" {
            continue;
        }
        out.push(ScriptRef {
            node,
            src: document.get_attribute(node, "src").map(str::to_owned),
            script_type: document.get_attribute(node, "type").map(str::to_owned),
            defer: document.get_attribute(node, "defer").is_some(),
            is_async: document.get_attribute(node, "async").is_some(),
            inline_text: document.text_content(node),
        });
    }
    out
}

/// Why a `<script>` element was not executed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    /// `type="module"`: deferred to the module loader, which
    /// `bowser_js` does not expose to this crate yet.
    Module,
    /// A `type` that is not an executable script type (e.g. a data
    /// block): carries the raw attribute value.
    NonExecutableType(String),
    /// The `src` did not resolve against the base URL.
    BadUrl(String),
    /// The external source could not be fetched (carries the fetch
    /// error). The page loads on regardless.
    Fetch(String),
}

impl fmt::Display for SkipReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Module => f.write_str("type=\"module\" is not executed yet"),
            Self::NonExecutableType(t) => write!(f, "non-executable script type {t:?}"),
            Self::BadUrl(e) => write!(f, "unresolvable script src: {e}"),
            Self::Fetch(e) => write!(f, "script fetch failed: {e}"),
        }
    }
}

/// A `<script>` element resolved to something runnable (or a recorded
/// skip): the unit [`PageVm::run`] executes.
///
/// `External` carries a `Url` + source by value; the variant is large
/// next to `Skipped`, which is fine for a short-lived per-script value.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum PageScript {
    /// Inline classic script: the element's raw text.
    Inline { node: NodeId, text: String },
    /// External classic script: `src` resolved against the base URL, with
    /// the fetched source. `defer` is the effective `defer` attribute
    /// (only ever true with `src`; per spec `defer` on an inline script
    /// is ignored). `is_async` is the effective `async` attribute (same
    /// `src` rule); it wins over `defer` when both are present.
    External {
        node: NodeId,
        url: Url,
        source: String,
        defer: bool,
        is_async: bool,
    },
    /// Not executed; the reason is recorded for diagnostics.
    Skipped { node: NodeId, reason: SkipReason },
}

/// Whether [`PageVm::run`] defers this script to its third pass:
/// deferred external scripts only. `async` wins over `defer` when both
/// are present (spec), so an async+defer script is never deferred.
fn is_deferred_script(script: &PageScript) -> bool {
    matches!(
        script,
        PageScript::External {
            defer: true,
            is_async: false,
            ..
        }
    )
}

/// Whether [`PageVm::run`] runs this script in its second (`async`)
/// pass: async external scripts only. Per spec `async` without `src` is
/// ignored, so inline scripts never match.
fn is_async_script(script: &PageScript) -> bool {
    matches!(script, PageScript::External { is_async: true, .. })
}

/// The `<script>` node a [`PageScript`] came from, whatever its fate.
pub fn script_node(script: &PageScript) -> NodeId {
    match script {
        PageScript::Inline { node, .. }
        | PageScript::External { node, .. }
        | PageScript::Skipped { node, .. } => *node,
    }
}

/// JavaScript MIME types that mark a classic script, mirroring
/// `bowser_js`'s executable-type list (kept in sync by the
/// `executable_script_types_match_bindings` test).
pub(crate) fn is_executable_script_type(t: &str) -> bool {
    let t = t.trim().to_ascii_lowercase();
    t.is_empty()
        || t == "text/javascript"
        || t == "application/javascript"
        || t == "text/ecmascript"
        || t == "application/ecmascript"
}

/// Classify collected scripts and fetch external sources, in document
/// order.
///
/// `fetch` maps a resolved `src` URL to its source text (or a fetch
/// error); the loader passes a fetcher with the document's cookie jar,
/// user agent, and SameSite context. External scripts are fetched
/// sequentially and capped at [`MAX_SCRIPTS`]; anything past the cap, and
/// any fetch failure, becomes [`PageScript::Skipped`] — the page never
/// fails because of a script.
pub fn resolve_scripts(
    refs: Vec<ScriptRef>,
    base_url: &Url,
    mut fetch: impl FnMut(&Url) -> Result<String, String>,
) -> Vec<PageScript> {
    let mut out = Vec::with_capacity(refs.len());
    let mut fetched = 0usize;
    for r in refs {
        if let Some(t) = r.script_type.as_deref()
            && t.trim().eq_ignore_ascii_case("module")
        {
            out.push(PageScript::Skipped {
                node: r.node,
                reason: SkipReason::Module,
            });
            continue;
        }
        if let Some(t) = r.script_type.as_deref()
            && !is_executable_script_type(t)
        {
            out.push(PageScript::Skipped {
                node: r.node,
                reason: SkipReason::NonExecutableType(t.to_owned()),
            });
            continue;
        }
        let Some(src) = r.src.as_deref() else {
            out.push(PageScript::Inline {
                node: r.node,
                text: r.inline_text,
            });
            continue;
        };
        let url = match bowser_url::parse(src.trim(), Some(base_url)) {
            Ok(url) => url,
            Err(e) => {
                out.push(PageScript::Skipped {
                    node: r.node,
                    reason: SkipReason::BadUrl(e.to_string()),
                });
                continue;
            }
        };
        if fetched >= MAX_SCRIPTS {
            out.push(PageScript::Skipped {
                node: r.node,
                reason: SkipReason::Fetch(format!(
                    "more than {MAX_SCRIPTS} external scripts; skipped"
                )),
            });
            continue;
        }
        fetched += 1;
        match fetch(&url) {
            Ok(source) => out.push(PageScript::External {
                node: r.node,
                url,
                source,
                // Per spec `defer`/`async` without `src` are ignored; this
                // branch only runs with `src`, so the attributes are
                // effective (`async` wins over `defer` in `run`'s pass
                // selection).
                defer: r.defer,
                is_async: r.is_async,
            }),
            Err(e) => out.push(PageScript::Skipped {
                node: r.node,
                reason: SkipReason::Fetch(e),
            }),
        }
    }
    out
}

/// Fetch one external script source: a plain GET carrying the loader's
/// `User-Agent`, following redirects, in the document's SameSite top
/// context (not a top-level navigation), with the jar locked only for
/// the request itself — the same semantics as every other subresource
/// fetch in [`crate::loader`].
pub(crate) fn fetch_script_source(
    url: &Url,
    jar: &Mutex<CookieJar>,
    user_agent: &str,
    nav: &NavigationContext,
) -> Result<String, String> {
    let request = Request {
        method: "GET".to_string(),
        url: url.clone(),
        headers: vec![("User-Agent".to_string(), user_agent.to_string())],
        body: None,
    };
    let response = {
        let mut guard = jar.lock().expect("script: cookie jar mutex poisoned");
        fetch_with_jar_in_context(&request, Some(&mut guard), nav).map_err(|e| e.to_string())?
    };
    if !(200..300).contains(&response.status) {
        return Err(format!("HTTP {}", response.status));
    }
    Ok(response.text())
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// How one script's evaluation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScriptErrorKind {
    /// The engine rejected the source: parse error, compile error, an
    /// uncaught throw, or resource exhaustion. Carries the engine's
    /// message.
    Eval(String),
    /// The script burned its execution budget: the engine's per-eval
    /// fuel ran out before the script returned. The page load continues
    /// with the next script — a runaway never hangs the load.
    FuelExhausted,
}

impl fmt::Display for ScriptErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eval(e) => write!(f, "script evaluation failed: {e}"),
            Self::FuelExhausted => f.write_str("script execution budget exhausted (fuel)"),
        }
    }
}

impl ScriptErrorKind {
    /// Classify an engine failure: fuel exhaustion (the execution
    /// budget) gets its own kind so the error console can tell "too
    /// slow" apart from "threw"; everything else is a plain evaluation
    /// failure. The engine reports budget kills as
    /// `VmError::Resource("fuel exhausted")`.
    pub(crate) fn from_vm_error(error: &VmError) -> Self {
        match error {
            VmError::Resource(msg) if msg == "fuel exhausted" => Self::FuelExhausted,
            _ => Self::Eval(error.to_string()),
        }
    }
}

/// One script that failed during [`PageVm::run`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptError {
    /// The `<script>` element that failed, if the failure belongs to
    /// one. `None` for failures with no script element behind them:
    /// uncaught throws from timer callbacks and other event-loop tasks,
    /// surfaced by [`PageVm::pump`].
    pub node: Option<NodeId>,
    /// Resolved `src` for external scripts; `None` for inline ones.
    pub url: Option<Url>,
    /// What went wrong.
    pub kind: ScriptErrorKind,
}

impl fmt::Display for ScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.node, &self.url, &self.kind) {
            (Some(_), Some(url), kind) => {
                write!(f, "{} (external): {kind}", url.serialize())
            }
            (Some(_), None, kind) => write!(f, "inline script: {kind}"),
            (None, _, kind) => write!(f, "page task: {kind}"),
        }
    }
}

impl ScriptError {
    /// Build an evaluation-failure error for `node`: `url` is `Some` when
    /// the script came from an external source, `None` for inline scripts.
    /// Used by [`PageVm::run`] and by dispatchers reporting later eval
    /// failures through [`PageVm::record_error`].
    pub fn eval_error(node: NodeId, url: Option<Url>, message: impl fmt::Display) -> Self {
        ScriptError {
            node: Some(node),
            url,
            kind: ScriptErrorKind::Eval(message.to_string()),
        }
    }

    /// Build an error from an engine failure for `node`: `url` is `Some`
    /// when the script came from an external source, `None` for inline
    /// scripts. Classifies the failure: a fuel-exhausted runaway becomes
    /// [`ScriptErrorKind::FuelExhausted`] rather than a generic eval
    /// error, so the error console can tell "too slow" apart from
    /// "threw".
    pub fn from_vm_error(node: NodeId, url: Option<Url>, error: &VmError) -> Self {
        ScriptError {
            node: Some(node),
            url,
            kind: ScriptErrorKind::from_vm_error(error),
        }
    }

    /// Build an error for a failure with no `<script>` element behind
    /// it: an uncaught throw from a timer callback or another event-loop
    /// task, drained from the engine by [`PageVm::pump`].
    pub fn task_error(error: VmError) -> Self {
        ScriptError {
            node: None,
            url: None,
            kind: ScriptErrorKind::from_vm_error(&error),
        }
    }
}

impl std::error::Error for ScriptError {}

/// What [`PageVm::run`] did with a page's scripts.
#[derive(Debug, Default, Clone)]
pub struct ScriptReport {
    /// Classic scripts evaluated (inline + external), in document order.
    pub executed: usize,
    /// Scripts not executed: modules, data blocks, bad URLs, failed or
    /// over-cap fetches.
    pub skipped: usize,
    /// Evaluation failures, in document order. A failed script never
    /// stops the scripts after it.
    pub errors: Vec<ScriptError>,
}

/// One page's JavaScript realm: a fresh `bowser_js::Vm` per document.
///
/// Created by [`crate::navigation::install_loaded_document`] on every
/// full navigation (a new document gets a new global scope, like a new
/// `Window`); dropped with the tab. Owns script evaluation for the page.
///
/// Build page realms with [`new_for_document`](Self::new_for_document):
/// it installs the web API surface (`window`, `document`, ...) on the
/// fresh global before the first script runs — one install per page
/// realm. [`new`](Self::new) is the bare realm (plain ECMAScript
/// builtins only), kept for tests and non-page uses.
pub struct PageVm {
    vm: Vm,
    report: ScriptReport,
    web_apis: Option<PageWebApis>,
    /// The document URL: the referrer for dynamic `import()` specifiers
    /// in classic scripts (inline scripts and event handlers). Set by
    /// [`install_document_apis`](Self::install_document_apis); external
    /// scripts resolve against their own URL instead.
    base_url: Option<Url>,
    /// Transient user activation for this page (HTML "transient
    /// activation"). `test_driver.click()` / `send_keys()` / `bless()`
    /// grant it, like a real user gesture would; `bless()`'s action
    /// then runs under it. No web API consumes it yet (phase-2 gap) —
    /// the plumbing exists so the flag is observable and consumable.
    user_activation: UserActivation,
}

/// Transient user activation state for one page, per HTML's "transient
/// activation" (a user gesture recently occurred).
///
/// Phase 1 is deliberately minimal: a sticky flag. Real transient
/// activation is consumed by the first activation-gated API call and
/// expires after a timeout; here nothing consumes it yet, and expiry
/// is a phase-2 gap, so the flag stays set once granted. The
/// [`consume`](UserActivation::consume) API exists for the future
/// consumers.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct UserActivation {
    /// Whether transient activation is currently held.
    transient: bool,
}

impl UserActivation {
    /// Grant transient activation (a trusted user gesture occurred).
    pub fn grant(&mut self) {
        self.transient = true;
    }

    /// Whether transient activation is currently held.
    pub fn has_transient(&self) -> bool {
        self.transient
    }

    /// Consume transient activation: returns whether it was held, and
    /// clears it. Activation-gated web APIs call this.
    pub fn consume(&mut self) -> bool {
        std::mem::replace(&mut self.transient, false)
    }
}

/// Name of the page global the WPT harness reporter arms and the
/// testharness.js completion callback sets.
///
/// The driver injects its reporter (`REPORTER_JS` in `wpt_driver.rs`)
/// into testharness pages: the reporter initializes this global to
/// `false` once `add_completion_callback` exists, and the completion
/// callback sets it to `true` after printing the report marker line.
/// The headless pump observes it through
/// [`PageVm::harness_reporter_armed`] / [`PageVm::harness_completed`] to
/// keep pumping an async test's event loop until it reports instead of
/// giving up after the fixed headless budget.
///
/// A hostile page could set the global early or clear it — both only
/// deny the wait (fail-closed): the report nonce still guards which
/// marker lines the driver accepts.
pub const HARNESS_COMPLETION_FLAG: &str = "__bowser_harness_done";

/// A page realm's installed web APIs: torn down with the realm.
struct PageWebApis {
    /// The installation handle from
    /// [`Vm::install_page_web_apis`](bowser_js::Vm::install_page_web_apis);
    /// passed back to
    /// [`Vm::remove_page_web_apis`](bowser_js::Vm::remove_page_web_apis)
    /// on drop (cancels in-flight fetches, drops timer/storage state).
    ctx: u32,
}

impl PageVm {
    /// A fresh realm: new heap, installed builtins, empty report. No web
    /// APIs: page scripts run against plain ECMAScript builtins plus the
    /// VM's event loop. Production page loads use
    /// [`new_for_document`](Self::new_for_document).
    pub fn new() -> Self {
        PageVm {
            vm: Vm::new(),
            report: ScriptReport::default(),
            web_apis: None,
            base_url: None,
            user_activation: UserActivation::default(),
        }
    }

    /// A fresh realm for a document, with the web API surface installed:
    /// `window`/`document`/`location`/`navigator`/`fetch`/
    /// `getComputedStyle`, storage/history, and `URL` (see
    /// [`Vm::install_page_web_apis`](bowser_js::Vm::install_page_web_apis)
    /// for the exact surface and the deliberate timer exclusions). The
    /// single per-page-realm install: call once per document, before
    /// [`run`](Self::run), so the page's scripts see the host objects
    /// from the first script.
    ///
    /// `document` is the tab's live DOM tree: the tab owns the sole
    /// strong `Rc`, shared with the realm (one tree, no copies). The
    /// bindings registry holds only a `Weak` to it, upgraded per native
    /// call (see `WebApis::document` in `bowser_js`); this `PageVm`
    /// deliberately holds no handle of its own. In production the tab
    /// tears the context down with the bundle (`PageVm::Drop`), so a
    /// call after teardown fails at the context lookup layer ("web api
    /// context is gone") before the Weak upgrade is ever reached; the
    /// Weak-failure path ("document is gone") is unreachable through
    /// the public bundle API and is exercised only via degraded/
    /// registry-level probes. Neither path lets script operate on a
    /// tree the renderer never reads.
    /// `base_url` is the document URL (resolves relative `fetch()` URLs,
    /// seeds `location.href` and the `URL` constructor's API base URL).
    /// `user_agent` becomes `navigator.userAgent`. `author` is the
    /// document's author stylesheets in cascade order, seeded into the
    /// page's web APIs so `getComputedStyle` reflects author rules.
    pub fn new_for_document(
        document: Rc<RefCell<Document>>,
        base_url: &Url,
        user_agent: &str,
        author: &[Stylesheet],
    ) -> Result<Self, String> {
        let mut page_vm = PageVm::new();
        page_vm.install_document_apis(document, base_url, user_agent, author)?;
        Ok(page_vm)
    }

    /// Install the page web APIs on this realm's existing VM, binding
    /// them to `document`. This is the document half of
    /// [`new_for_document`](Self::new_for_document), split out so the
    /// parser-inserted script path can run mid-parse scripts through a
    /// `PageVm` first (with only the parse-phase `document` shim) and
    /// install the production web APIs afterwards, on the same realm.
    /// The install overwrites the parse-phase `document` shim.
    ///
    /// `window.addEventListener` registrations made through the
    /// parse-phase shim are adopted into the production listener table
    /// by the install (registration order and seq preserved), so a
    /// mid-parse `window.addEventListener("load", …)` still fires after
    /// the parse.
    ///
    /// The registry keeps only a `Weak` to `document`; the tab keeps
    /// the strong handle, so dropping the tab's install tears down
    /// script access to this tree cleanly.
    pub fn install_document_apis(
        &mut self,
        document: Rc<RefCell<Document>>,
        base_url: &Url,
        user_agent: &str,
        author: &[Stylesheet],
    ) -> Result<(), String> {
        let ctx = self
            .vm
            .install_page_web_apis(document.clone(), base_url.clone(), user_agent)?;
        // Seed the document's author stylesheets into the page's web
        // APIs so `getComputedStyle` reflects author rules.
        self.vm.set_author_stylesheets(ctx, author.to_vec());
        self.web_apis = Some(PageWebApis { ctx });
        // The document URL is the referrer for dynamic `import()` in
        // classic scripts (see `run_one`).
        self.base_url = Some(base_url.clone());
        Ok(())
    }

    /// The VM itself, for the parser-inserted script path (which
    /// installs its own per-script `document` shim before evaluating).
    pub(crate) fn vm_mut(&mut self) -> &mut Vm {
        &mut self.vm
    }

    /// The VM itself, read-only — for inspecting engine state (guard
    /// budgets, stack high-water marks) without running anything.
    /// Test-only for now: production code reaches the VM through
    /// [`PageVm::vm_mut`].
    #[cfg(test)]
    pub(crate) fn vm(&self) -> &Vm {
        &self.vm
    }

    /// Count one parser-inserted script as executed, for the executor
    /// in [`crate::script_executor`].
    pub(crate) fn record_script_executed(&mut self) {
        self.report.executed += 1;
    }

    /// The web-API context installed on this realm, if it was built with
    /// [`new_for_document`](Self::new_for_document).
    pub fn web_api_context(&self) -> Option<u32> {
        self.web_apis.as_ref().map(|w| w.ctx)
    }

    /// Dispatch a trusted `load` (or `error`) event at an `<img>`
    /// element: the headless pump's late-image drain calls this after
    /// image completions, so late-image `load` listeners run. Returns
    /// false when the realm is gone or the target cannot be wrapped.
    pub(crate) fn dispatch_image_event(&mut self, target: NodeId, event_type: &str) -> bool {
        self.vm.dispatch_image_event(target, event_type)
    }

    /// Evaluate every runnable script, recording the outcome in the
    /// [`ScriptReport`].
    ///
    /// Inline and external sources run through [`Vm::eval`] as classic
    /// scripts; the VM's microtask queue is drained between scripts (the
    /// HTML microtask checkpoint). Timers and other macrotasks registered
    /// by scripts do *not* run here — they wait for the first
    /// [`pump`](Self::pump). An evaluation failure is recorded and
    /// execution continues with the next script — scripts never fail the
    /// navigation.
    ///
    /// Ordering is three passes: every classic (non-async, non-deferred)
    /// script first in document order (the spec's during-parsing
    /// execution; the loader hands us scripts post-parse, so this is the
    /// faithful mapping), then every `async` external script as one
    /// order-independent batch (the spec's execute-as-soon-as-available:
    /// the loader fired these fetches concurrently, so by the time this
    /// runs each source is in hand and the batch makes no document-order
    /// promise), then every deferred external script in document order.
    /// Both the `async` batch and the deferred pass run before the window
    /// `load` event.
    ///
    /// Execution budget: each script evaluates under the engine's
    /// per-eval fuel budget — a non-terminating script burns its fuel and
    /// the eval returns `VmError::Resource("fuel exhausted")`, which
    /// lands in the report as [`ScriptErrorKind::FuelExhausted`]. The
    /// page load continues with the next script; a runaway never hangs
    /// the installing thread.
    ///
    /// Scripts share the realm's global *object* (sloppy-mode globals and
    /// anything planted with [`install_global`](Self::install_global)
    /// persist across scripts), but each script's lexical scope is its
    /// own: top-level `var`/`let`/`const` do not carry into later scripts
    /// (engine gap, documented at the module level).
    pub fn run(&mut self, scripts: &[PageScript]) {
        for script in scripts
            .iter()
            .filter(|s| !is_async_script(s) && !is_deferred_script(s))
        {
            self.run_one(script);
        }
        for script in scripts.iter().filter(|s| is_async_script(s)) {
            self.run_one(script);
        }
        for script in scripts.iter().filter(|s| is_deferred_script(s)) {
            self.run_one(script);
        }
    }

    /// Point the VM's dynamic-import referrer at `script_url` (an external
    /// script's own URL), falling back to the document URL for inline
    /// scripts. `None` everywhere clears it: relative `import()`
    /// specifiers then reject with a `TypeError`.
    fn sync_base_url(&mut self, script_url: Option<&Url>) {
        let url = script_url.or(self.base_url.as_ref()).map(|u| u.serialize());
        self.vm.set_base_url(url);
    }

    /// Evaluate one runnable script: the unit of the three [`run`](Self::run)
    /// passes (classic, then async, then deferred).
    fn run_one(&mut self, script: &PageScript) {
        let (node, url, source) = match script {
            PageScript::Inline { node, text } => (*node, None, text.as_str()),
            PageScript::External {
                node, url, source, ..
            } => (*node, Some(url.clone()), source.as_str()),
            PageScript::Skipped { .. } => {
                self.report.skipped += 1;
                return;
            }
        };
        self.report.executed += 1;
        self.sync_base_url(url.as_ref());
        if let Err(e) = self.vm.eval(source) {
            self.report
                .errors
                .push(ScriptError::from_vm_error(node, url, &e));
        }
        self.drain_microtasks();
    }

    /// The between-scripts microtask checkpoint: bindings microtasks
    /// (`queueMicrotask` callbacks, fetch completions) and engine promise
    /// jobs interleave in FIFO order through the pump hook. Realms
    /// without installed web APIs (unit-test realms) use the VM's own
    /// drain.
    fn drain_microtasks(&mut self) {
        // The bindings page-microtask path bypasses `Vm::drain_microtasks`,
        // so run the host dynamic-import hook explicitly: imports queued by
        // the scripts above settle before their `.then` reactions run.
        self.vm.process_pending_dynamic_imports();
        if let Some(w) = &self.web_apis {
            for error in drain_page_microtasks(&mut self.vm, w.ctx) {
                self.report
                    .errors
                    .push(ScriptError::task_error(VmError::Thrown(error)));
            }
        } else {
            self.vm.drain_microtasks();
        }
    }

    /// Evaluate `source` as a classic script in the page's realm,
    /// sharing its globals with the scripts [`run`](Self::run)
    /// evaluated, then drain the microtask queue (the same checkpoint
    /// `run` applies between scripts).
    ///
    /// The entry point for event dispatch: click/key handlers run through
    /// here.
    ///
    /// Like [`run`](Self::run), only the microtask queue is drained:
    /// timers and other macrotasks registered by the handler wait for
    /// the next [`pump`](Self::pump).
    pub fn eval(&mut self, source: &str) -> Result<Value, VmError> {
        // Event handlers behave like inline scripts for import().
        self.sync_base_url(None);
        let result = self.vm.eval(source);
        self.drain_microtasks();
        result
    }

    /// Drive the page's event loop until no macrotask, microtask, or due
    /// timer remains (timers with future deadlines stay pending).
    ///
    /// This is what fires `setTimeout`/`setInterval` callbacks and other
    /// macrotasks page scripts registered: nothing runs them until
    /// something pumps the loop. The window worker calls it on its
    /// event-loop tick; the event dispatcher calls it after every
    /// dispatched event.
    ///
    /// The pump goes through the bindings pump hook
    /// ([`bowser_js::pump_page_event_loop`]): bindings-owned tasks
    /// (timer callbacks, `queueMicrotask` calls, fetch `.then` reactions)
    /// fire through the hidden pump native, and everything else runs
    /// through the VM's own task runner, in the event loop's FIFO order.
    /// Realms without installed web APIs (unit-test realms) use the
    /// VM's own idle drain.
    ///
    /// Uncaught throws from the tasks that ran (a `setTimeout` callback
    /// that threw, a script task that failed) are drained from the engine
    /// and appended to the [`ScriptReport`]'s errors as node-less "page
    /// task" errors — the error console sees them instead of the event
    /// loop swallowing them.
    ///
    /// Known hang, documented: a page that keeps scheduling due work (a
    /// `setInterval` with a 0 ms delay, an always-requeuing macrotask)
    /// makes `pump` spin forever. The per-eval fuel budget does not save
    /// it: every task fires through `Vm::eval`, which resets fuel, so no
    /// single task ever exhausts its budget while the queue never
    /// empties. Callers that cannot tolerate the spin use
    /// [`pump_bounded`](Self::pump_bounded): the engine-side task budget
    /// (maximum tasks per pump) lives in `bowser_js`
    /// (`EventLoop::run_with_task_budget`).
    /// Classic scripts are covered: a non-terminating *script* burns its
    /// own fuel and lands in the report as
    /// [`ScriptErrorKind::FuelExhausted`] (see [`run`](Self::run)).
    pub fn pump(&mut self) {
        // Settle imports queued since the last checkpoint before running
        // the loop (the bindings pump path bypasses `Vm::run_until_idle`).
        self.vm.process_pending_dynamic_imports();
        if let Some(w) = &self.web_apis {
            for error in pump_page_event_loop(&mut self.vm, w.ctx) {
                self.report
                    .errors
                    .push(ScriptError::task_error(VmError::Thrown(error)));
            }
        } else {
            self.vm.run_until_idle();
        }
        // Tasks above may have queued dynamic imports (timer callbacks,
        // fetch completions): settle them and their reactions now rather
        // than leaving the promises pending until the next pump.
        self.drain_microtasks();
        for error in self.vm.take_uncaught_task_errors() {
            self.report.errors.push(ScriptError::task_error(error));
        }
    }

    /// Drive the page's event loop like [`pump`](Self::pump), but run at
    /// most `max_tasks` tasks: every timer/macrotask fire and every
    /// microtask drained around them counts against the budget.
    ///
    /// Returns `true` when the loop drained fully (no macrotask,
    /// microtask, or due timer remains) and `false` when the budget ran
    /// out first — the page that keeps scheduling due work (a
    /// `setInterval` with a 0 ms delay) can never spin one call forever.
    /// Unrun work stays queued, in order, for the next pump.
    ///
    /// When the budget ran out, the between-tasks microtask checkpoint is
    /// skipped: remaining microtasks (and any dynamic imports this pump's
    /// tasks queued) wait for the next pump instead of spinning here.
    /// When the loop drained, the checkpoint runs exactly as in
    /// [`pump`](Self::pump).
    pub fn pump_bounded(&mut self, max_tasks: usize) -> bool {
        // Settle imports queued since the last checkpoint before running
        // the loop (the bindings pump path bypasses `Vm::run_until_idle`).
        self.vm.process_pending_dynamic_imports();
        let drained = if let Some(w) = &self.web_apis {
            let (errors, drained) =
                pump_page_event_loop_with_budget(&mut self.vm, w.ctx, max_tasks);
            for error in errors {
                self.report
                    .errors
                    .push(ScriptError::task_error(VmError::Thrown(error)));
            }
            drained
        } else {
            self.vm
                .event_loop()
                .run_with_task_budget(&mut self.vm, max_tasks)
        };
        // Tasks above may have queued dynamic imports (timer callbacks,
        // fetch completions): settle them and their reactions now rather
        // than leaving the promises pending until the next pump — but
        // only when the loop actually drained. On a budget cut the next
        // pump iteration continues the work; draining here could spin on
        // a page that requeues microtasks forever.
        if drained {
            self.drain_microtasks();
        }
        for error in self.vm.take_uncaught_task_errors() {
            self.report.errors.push(ScriptError::task_error(error));
        }
        drained
    }

    /// Whether the WPT harness reporter armed this page: the
    /// [`HARNESS_COMPLETION_FLAG`] global exists and is a boolean. Only
    /// pages served through the driver's harness endpoint have it (the
    /// injected reporter initializes it); ordinary pages pump against
    /// the fixed headless budget.
    pub fn harness_reporter_armed(&self) -> bool {
        matches!(
            self.vm.get_global(HARNESS_COMPLETION_FLAG),
            Some(Value::Bool(_))
        )
    }

    /// Whether the testharness.js completion callback has fired: the
    /// [`HARNESS_COMPLETION_FLAG`] global is `true`. The injected
    /// reporter sets it right after printing the report marker line, so
    /// by the time the pump observes it the report is on its way to the
    /// driver's stdout scan.
    pub fn harness_completed(&self) -> bool {
        matches!(
            self.vm.get_global(HARNESS_COMPLETION_FLAG),
            Some(Value::Bool(true))
        )
    }

    /// Grant transient user activation on this page: a trusted user
    /// gesture (real or testdriver-driven) occurred. See
    /// [`UserActivation`].
    pub fn grant_transient_activation(&mut self) {
        self.user_activation.grant();
    }

    /// Whether this page currently holds transient user activation.
    pub fn has_transient_activation(&self) -> bool {
        self.user_activation.has_transient()
    }

    /// Consume transient user activation, returning whether it was held.
    /// Activation-gated web APIs call this; currently no caller exists
    /// (phase-2 gap), but the testdriver drain grants through
    /// [`grant_transient_activation`](Self::grant_transient_activation).
    pub fn consume_transient_activation(&mut self) -> bool {
        self.user_activation.consume()
    }

    /// Plant a host object on the page's global scope, overwriting any
    /// existing global of the same name. Install before
    /// [`run`](Self::run) so the page's scripts see the host object
    /// from the first script. (The web APIs themselves are planted by
    /// [`new_for_document`](Self::new_for_document); this is the escape
    /// hatch for additional host objects.)
    pub fn install_global(&mut self, name: &str, value: Value) {
        self.vm.set_global(name, value);
    }

    /// Read a property of the page's global object: the counterpart of
    /// [`install_global`](Self::install_global) — tests read script
    /// state through this. `None` when the global has no such own
    /// property.
    pub fn global(&self, name: &str) -> Option<Value> {
        self.vm.get_global(name)
    }

    /// Record a failure from an eval that happened after
    /// [`run`](Self::run) — an event-dispatch handler that threw, say —
    /// in the page's error console ([`ScriptReport::errors`]).
    ///
    /// Unlike `run`, which records failures itself, [`eval`](Self::eval)
    /// returns them to the caller: the dispatcher calls this to surface
    /// them instead of dropping them.
    pub fn record_error(&mut self, error: ScriptError) {
        self.report.errors.push(error);
    }

    /// The outcome of [`PageVm::run`]: counts plus evaluation failures.
    pub fn report(&self) -> &ScriptReport {
        &self.report
    }
}

impl Default for PageVm {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PageVm {
    /// Tear down the page's web APIs: unregisters the context (cancels
    /// in-flight fetches, drops timer/storage state). Bare realms
    /// ([`new`](Self::new)) have nothing to tear down.
    fn drop(&mut self) {
        if let Some(w) = &self.web_apis {
            self.vm.remove_page_web_apis(w.ctx);
        }
    }
}

// ---------------------------------------------------------------------------
// Production UI-event dispatch
// ---------------------------------------------------------------------------

/// Production [`JsDispatch`](crate::window::JsDispatch): UI events into
/// the page's real JS realm.
///
/// A click runs the target's inline `onclick` attribute source with the
/// dispatched event object bound as `event`, then fires the listeners
/// the page registered via `addEventListener` by dispatching that SAME
/// object as a bubbling, cancelable `click` event on the target's
/// element wrapper (reached via a temporary id, since `querySelector`
/// has no attribute-selector support). A key runs the document
/// element's (falling back to `<body>`'s) inline `onkeydown` source and
/// dispatches a `keydown` on the `document` wrapper, which is where
/// `document.addEventListener('keydown', ...)` listeners live. A wheel
/// event goes straight to the VM's
/// [`dispatch_wheel`](bowser_js::Vm::dispatch_wheel), which builds and
/// dispatches a trusted bubbling, cancelable `wheel` event carrying
/// `deltaX`/`deltaY`/`deltaMode`. After
/// dispatch the VM is pumped — handler-scheduled work like
/// `setTimeout(..., 0)` runs — and DOM dirt is folded into the tab's
/// render flags.
///
/// Returns whether the page prevented the default action: an inline
/// handler whose source evaluates to `false` (the `onclick="return
/// false"` idiom), or `dispatchEvent` reporting the event canceled.
/// Handler throws are recorded in the page's [`ScriptReport`], never
/// propagated: a throwing handler must not break the UI event.
pub struct PageEventDispatch {
    next_marker: u64,
}

/// Modifier-key state carried by one key dispatch (the WebDriver
/// Actions executor's modifier state machine, see
/// `crate::testdriver`): the flags are spliced into the dispatched
/// event's init object, so inline handlers read `event.shiftKey` et
/// al. directly. (`addEventListener` listeners receive a wrapped
/// `Event` whose stored shape carries only `key` today — the
/// bindings have no `KeyboardEvent` wrapper yet, so the flags reach
/// listeners only through pointer events' `PointerData`; documented
/// gap, not silent.)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KeyModifiers {
    /// `Shift` held.
    pub shift: bool,
    /// `Control` held.
    pub ctrl: bool,
    /// `Alt` held.
    pub alt: bool,
    /// `Meta` held.
    pub meta: bool,
}

impl PageEventDispatch {
    /// A fresh dispatcher: the marker counter starts at zero.
    pub fn new() -> Self {
        PageEventDispatch { next_marker: 0 }
    }

    /// Run an inline handler attribute's `source` with the dispatched
    /// event object bound as `event`, then dispatch that SAME object to
    /// the target's `addEventListener` listeners. Returns true when the
    /// page canceled: the inline source evaluated to `false` (the
    /// `return false` idiom) or `dispatchEvent` reported cancellation.
    ///
    /// The event object is created once per dispatch: the inline handler
    /// sees the object `dispatchEvent` delivers (so
    /// `onkeydown="…event.key…"` reads the dispatched key), matching
    /// browser behavior where the inline handler is just another
    /// listener on the same event.
    ///
    /// `target_js` is a JS expression evaluating to the dispatch target
    /// (an element lookup, or `document`); a nullish target skips the
    /// listener dispatch but still runs the inline handler, preserving
    /// the old "detached mid-dispatch reports not canceled" behavior.
    /// Handler throws are recorded in the page's [`ScriptReport`], never
    /// propagated: a throwing handler must not break the UI event.
    fn eval_inline_and_dispatch(
        page_vm: &mut PageVm,
        node: NodeId,
        target_js: &str,
        event_init_js: &str,
        inline: Option<&str>,
    ) -> bool {
        // Inline handlers are function bodies in the DOM: wrap the source
        // so `return false` cancels the default action instead of being a
        // top-level syntax error, and bind the dispatched event object as
        // `event`.
        let inline_expr = match inline {
            Some(source) => format!("(function(event){{{source}}})(ev)"),
            None => "undefined".to_string(),
        };
        let source = format!(
            "(function(){{var ev={event_init_js};var r={inline_expr};\
             var t=({target_js});var d=t?t.dispatchEvent(ev):true;\
             return (r===false||d===false)?false:true;}})()"
        );
        match page_vm.eval(&source) {
            Ok(Value::Bool(false)) => true,
            Ok(_) => false,
            Err(e) => {
                page_vm.record_error(ScriptError::from_vm_error(node, None, &e));
                false
            }
        }
    }

    /// Fire the page's inline `onclick` source (with the dispatched event
    /// object bound as `event`) and its `addEventListener("click", ...)`
    /// listeners at `target`. Returns true when the page canceled the
    /// event.
    fn dispatch_click_with_inline(
        &mut self,
        doc: &RefCell<Document>,
        page_vm: &mut PageVm,
        target: NodeId,
        inline: Option<&str>,
    ) -> bool {
        // The temporary-id lookup and the inline-then-listeners eval
        // are shared with keys and focus events
        // (`dispatch_typed_with_inline`).
        self.dispatch_typed_with_inline(
            doc,
            page_vm,
            target,
            "{type:'click',bubbles:true,cancelable:true}",
            inline,
        )
    }

    /// Dispatch one key press (`keydown`, then `keyup`) targeted at
    /// `target`'s element wrapper: the per-element key path
    /// `test_driver.send_keys()` needs (the [`JsDispatch`](crate::window::JsDispatch)
    /// `dispatch_key` is document-level only).
    ///
    /// Each event runs the target's inline handler attribute (`onkeydown`
    /// / `onkeyup`) first, then the page's `addEventListener` listeners
    /// via a trusted bubbling, cancelable `dispatchEvent` carrying the
    /// UI-Events `key` value — the same temporary-id lookup trick as
    /// clicks, since `querySelector` lacks attribute selectors.
    ///
    /// Deliberate gaps, documented not silent: no `keypress` event, no
    /// key `repeat`, no `code`/`location`, and no default text-input
    /// action (typing into a text field does not change its value —
    /// there is no editing subsystem yet). Scrolling keys do get their
    /// default action (see [`dispatch_key_down_at`](Self::dispatch_key_down_at)).
    /// Returns true when any dispatched event was canceled.
    pub fn dispatch_key_at(
        &mut self,
        tab: &mut crate::tab::Tab,
        target: NodeId,
        key: &str,
    ) -> bool {
        let down = self.dispatch_key_down_at(tab, target, key, KeyModifiers::default());
        let up = self.dispatch_key_up_at(tab, target, key, KeyModifiers::default());
        down || up
    }

    /// Dispatch only the `keydown` half of a key press at `target`,
    /// carrying `modifiers`: the WebDriver Actions executor presses
    /// and releases keys as separate actions (a held modifier must
    /// span many actions), so the halves are separate entry points.
    ///
    /// Default action: when the page did not prevent the `keydown`,
    /// scrolling keys pan the key target's scroll container — the
    /// synthetic-input equivalent of the windowed key path's
    /// [`apply_key_scroll`](crate::window::apply_key_scroll) — and the
    /// movement is reported through the scroll funnel below. Without
    /// this, an Actions `ArrowDown` dispatched its event and moved
    /// nothing.
    ///
    /// Returns true when the page canceled the event.
    pub fn dispatch_key_down_at(
        &mut self,
        tab: &mut crate::tab::Tab,
        target: NodeId,
        key: &str,
        modifiers: KeyModifiers,
    ) -> bool {
        let prevented = self.dispatch_key_phase(tab, target, "keydown", key, modifiers);
        if !prevented {
            // The WebDriver Actions executor's modifier state rides along
            // in `modifiers`: thread Shift through to the default scroll
            // action so Shift+Space pages up, like the windowed key path
            // (`apply_key_scroll_with_shift`).
            let scrolled =
                crate::window::apply_key_scroll_with_shift(tab, Some(target), key, modifiers.shift);
            self.report_synthetic_scroll(tab, &scrolled);
        }
        prevented
    }

    /// Dispatch only the `keyup` half of a key press at `target`;
    /// see [`dispatch_key_down_at`](Self::dispatch_key_down_at).
    pub fn dispatch_key_up_at(
        &mut self,
        tab: &mut crate::tab::Tab,
        target: NodeId,
        key: &str,
        modifiers: KeyModifiers,
    ) -> bool {
        self.dispatch_key_phase(tab, target, "keyup", key, modifiers)
    }

    /// One key phase (`keydown` or `keyup`) at `target`: the shared
    /// body of [`dispatch_key_at`](Self::dispatch_key_at) and its
    /// split halves.
    fn dispatch_key_phase(
        &mut self,
        tab: &mut crate::tab::Tab,
        target: NodeId,
        event_type: &str,
        key: &str,
        modifiers: KeyModifiers,
    ) -> bool {
        // Read the inline source before borrowing the VM: both live on
        // the tab.
        let inline_attr = if event_type == "keydown" {
            "onkeydown"
        } else {
            "onkeyup"
        };
        let inline = match tab.document_ref() {
            Some(doc) => doc.get_attribute(target, inline_attr).map(|s| s.to_owned()),
            None => return false,
        };
        let Some(installed) = tab.installed_mut() else {
            return false;
        };
        // Disjoint field borrow: the DOM handle and the realm, with no
        // `Rc` clone — the tab keeps the sole strong owner.
        let (doc, page_vm) = installed.document_and_vm_mut();
        // Reentrancy gate (see `InstalledDocument::assert_no_live_borrow`):
        // the inline source and the `dispatchEvent` eval below re-enter
        // the page VM, whose natives re-borrow the document through the
        // registry `Weak` — so no `Ref`/`RefMut` may be alive across
        // either eval.
        crate::tab::InstalledDocument::assert_no_live_borrow(doc);
        // One eval: the inline handler runs with the dispatched event
        // object bound as `event`, then the same object goes to the
        // listeners via `dispatchEvent`.
        let prevented = self.dispatch_key_with_inline(
            doc,
            page_vm,
            target,
            event_type,
            key,
            modifiers,
            inline.as_deref(),
        );
        page_vm.pump();
        tab.sync_dirty_flags();
        prevented
    }

    /// Fire the page's inline `onkeydown`/`onkeyup` source (with the
    /// dispatched event object bound as `event`) and its
    /// `addEventListener("keydown"|"keyup", ...)` listeners at `target`,
    /// carrying the UI-Events `key` value and the modifier flags.
    /// Returns true when the page canceled the event. Same
    /// temporary-id lookup as clicks.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_key_with_inline(
        &mut self,
        doc: &RefCell<Document>,
        page_vm: &mut PageVm,
        target: NodeId,
        event_type: &str,
        key: &str,
        modifiers: KeyModifiers,
        inline: Option<&str>,
    ) -> bool {
        let key_lit = js_string_escape(key);
        let event_init = format!(
            "{{type:'{event_type}',key:\"{key_lit}\",bubbles:true,cancelable:true,\
             shiftKey:{},ctrlKey:{},altKey:{},metaKey:{}}}",
            modifiers.shift, modifiers.ctrl, modifiers.alt, modifiers.meta
        );
        self.dispatch_typed_with_inline(doc, page_vm, target, &event_init, inline)
    }

    /// Fire one focus-family event (`focus`/`blur`/`focusin`/`focusout`)
    /// at `target`: the inline `on<type>` handler first, then the
    /// `addEventListener` listeners, through the same temporary-id
    /// dispatch as keys. `focus`/`blur` do not bubble, `focusin`/
    /// `focusout` do; none is cancelable. Used by the testdriver
    /// Actions executor's focus-lite model (see `crate::testdriver`).
    /// Returns true when the page canceled the event (impossible for
    /// these non-cancelable types today; kept symmetric with keys).
    pub fn dispatch_focus_event(
        &mut self,
        tab: &mut crate::tab::Tab,
        target: NodeId,
        event_type: &str,
    ) -> bool {
        debug_assert!(matches!(
            event_type,
            "focus" | "blur" | "focusin" | "focusout"
        ));
        let inline_attr = format!("on{event_type}");
        let inline = match tab.document_ref() {
            Some(doc) => doc
                .get_attribute(target, &inline_attr)
                .map(|s| s.to_owned()),
            None => return false,
        };
        let Some(installed) = tab.installed_mut() else {
            return false;
        };
        let (doc, page_vm) = installed.document_and_vm_mut();
        // Reentrancy gate: see `dispatch_key_phase`.
        crate::tab::InstalledDocument::assert_no_live_borrow(doc);
        let bubbles = matches!(event_type, "focusin" | "focusout");
        let event_init = format!("{{type:'{event_type}',bubbles:{bubbles},cancelable:false}}");
        let prevented =
            self.dispatch_typed_with_inline(doc, page_vm, target, &event_init, inline.as_deref());
        page_vm.pump();
        tab.sync_dirty_flags();
        prevented
    }

    /// Dispatch the event described by `event_init_js` (a JS object
    /// literal) at `target`, running `inline` (the target's `on<type>`
    /// attribute source) first with the event bound as `event`.
    /// Returns true when the page canceled. The temporary-id lookup
    /// is shared by clicks, keys, and focus events: `querySelector`
    /// lacks attribute selectors, so `getElementById` is the path.
    fn dispatch_typed_with_inline(
        &mut self,
        doc: &RefCell<Document>,
        page_vm: &mut PageVm,
        target: NodeId,
        event_init_js: &str,
        inline: Option<&str>,
    ) -> bool {
        self.next_marker = self.next_marker.wrapping_add(1);
        let temp_id = format!("bowser-dispatch-{}", self.next_marker);
        let old_id = doc
            .borrow()
            .get_attribute(target, "id")
            .map(|s| s.to_owned());
        doc.borrow_mut().set_attribute(target, "id", &temp_id);
        // IIFE keeps the temporary out of the page's globals; a missing
        // wrapper (the inline handler detached the target mid-dispatch)
        // reports "not canceled" instead of throwing.
        let target_js = format!("document.getElementById(\"{temp_id}\")");
        // Reentrancy gate (see `InstalledDocument::assert_no_live_borrow`):
        // natives re-borrow the document through the registry `Weak` on
        // every call, so no `Ref`/`RefMut` may be alive across an eval.
        crate::tab::InstalledDocument::assert_no_live_borrow(doc);
        let canceled =
            Self::eval_inline_and_dispatch(page_vm, target, &target_js, event_init_js, inline);
        match old_id {
            Some(id) => doc.borrow_mut().set_attribute(target, "id", &id),
            None => doc.borrow_mut().remove_attribute(target, "id"),
        }
        canceled
    }

    /// Fire the page's `addEventListener("scroll", ...)` /
    /// `addEventListener("scrollend", ...)` listeners at `target` (a
    /// nested scroll container whose offset moved / whose scroll gesture
    /// ended). Like
    /// [`dispatch_click_with_inline`](Self::dispatch_click_with_inline) but
    /// non-bubbling and non-cancelable, matching the `scroll` and
    /// `scrollend` events. `event_type` is `"scroll"` or `"scrollend"`.
    fn dispatch_scroll_listeners(
        &mut self,
        doc: &RefCell<Document>,
        page_vm: &mut PageVm,
        target: NodeId,
        event_type: &str,
    ) {
        self.next_marker = self.next_marker.wrapping_add(1);
        // Same temporary-id lookup as clicks: the DOM engine's
        // `querySelector` does not support attribute selectors, so
        // `getElementById` is the lookup path. Any pre-existing id is
        // saved and restored; no page script runs while the temporary is
        // in place except the dispatch below, which looks it up directly.
        let temp_id = format!("bowser-dispatch-{}", self.next_marker);
        let old_id = doc
            .borrow()
            .get_attribute(target, "id")
            .map(|s| s.to_owned());
        doc.borrow_mut().set_attribute(target, "id", &temp_id);
        // IIFE keeps the temporary out of the page's globals; a missing
        // wrapper (the inline handler detached the target mid-dispatch)
        // reports a dispatch error instead of throwing.
        let source = format!(
            "(function(){{var el=document.getElementById(\"{temp_id}\");\
             return el?el.dispatchEvent({{type:'{event_type}'}}):true;}})()"
        );
        // Reentrancy gate (see `InstalledDocument::assert_no_live_borrow`):
        // natives re-borrow the document through the registry `Weak` on
        // every call, so no `Ref`/`RefMut` may be alive across an eval.
        crate::tab::InstalledDocument::assert_no_live_borrow(doc);
        if let Err(e) = page_vm.eval(&source) {
            page_vm.record_error(ScriptError::from_vm_error(target, None, &e));
        }
        match old_id {
            Some(id) => doc.borrow_mut().set_attribute(target, "id", &id),
            None => doc.borrow_mut().remove_attribute(target, "id"),
        }
    }
}

/// Escape a Rust string as a JS double-quoted string literal body.
///
/// Shared by every embedder eval that splices Rust strings into JS
/// source (key literals in [`PageEventDispatch`]'s dispatch,
/// [`crate::testdriver::TestDriver`]'s completion errors, …): one
/// escaper, not one per call site — the two near-duplicate versions
/// drifted (`\t` escaped in one, not the other).
///
/// U+2028/U+2029 are escaped even though they are valid JS source
/// *outside* strings: bowser_js's lexer rejects them *inside*
/// string literals (`LexError::UnterminatedString`), so a raw
/// paragraph separator in a spliced message would make the eval
/// throw.
pub(crate) fn js_string_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Touch-event dispatch for [`PageEventDispatch`], kept outside the
/// [`crate::window::JsDispatch`] trait because that trait lives in
/// `window.rs` (owned by another worker this wave).
impl PageEventDispatch {
    /// THE HOOK THE TOUCH-PHYSICS WORKER CALLS.
    ///
    /// Dispatch one gesture phase to the page's JS listeners:
    ///
    /// ```rust,ignore
    /// // Signature (stable contract for the physics worker):
    /// pub fn dispatch_touch_event(
    ///     &mut self,
    ///     tab: &mut crate::tab::Tab,
    ///     phase: bowser_dom::TouchPhase,   // Start | Move | End
    ///     touches: &[bowser_dom::Touch],   // changed contacts this phase
    /// ) -> bool
    /// ```
    ///
    /// `phase` selects the event type (`touchstart`/`touchmove`/
    /// `touchend`); `touches` are the contact points that changed in
    /// this phase, each carrying its own target (the element the gesture
    /// started on, per the Touch Events spec). The dispatch target is
    /// the first changed touch with `Some(target)`; when no changed
    /// touch has a target the call fails closed and returns `false`
    /// without running any script.
    ///
    /// The built event is trusted, bubbling, and cancelable, with
    /// `touches`/`changedTouches` set to the `touches` argument and
    /// `targetTouches` to the subset whose target is the dispatch
    /// target. Modifier keys are all false (the gesture recognizer does
    /// not track them yet).
    ///
    /// Returns true when a non-passive listener called
    /// `preventDefault()`: a prevented `touchstart` means the page took
    /// the gesture, so the physics worker MUST cancel its default pan.
    /// Passive listeners' `preventDefault()` calls are ignored. Handler
    /// throws are recorded in the page's [`ScriptReport`], never
    /// propagated.
    pub fn dispatch_touch_event(
        &mut self,
        tab: &mut crate::tab::Tab,
        phase: TouchPhase,
        touches: &[Touch],
    ) -> bool {
        // Per the Touch Events spec the event targets the touch's
        // original target; the first changed touch with one wins.
        let target = match touches.iter().find_map(|t| t.target) {
            Some(n) => n,
            None => return false,
        };
        // Fail closed: no installed document means no JS realm to ask.
        let Some(installed) = tab.installed_mut() else {
            return false;
        };
        // Reentrancy gate (see `InstalledDocument::assert_no_live_borrow`):
        // `dispatch_touch` runs JS listeners, whose natives re-borrow the
        // document through the registry `Weak` — so no `Ref`/`RefMut` may
        // be alive across the dispatch.
        crate::tab::InstalledDocument::assert_no_live_borrow(installed.document());
        let page_vm = installed.page_vm_mut();
        // Trusted, bubbling, cancelable touch event; returns true iff a
        // non-passive listener prevented the default (already fail-closed
        // when the realm is gone).
        let prevented = page_vm.vm_mut().dispatch_touch(phase, touches, target);
        // Same bookkeeping as clicks, keys, and wheels: pump
        // handler-scheduled work, then fold DOM dirt into the tab's
        // render flags.
        page_vm.pump();
        tab.sync_dirty_flags();
        prevented
    }
}

// ---------------------------------------------------------------------------
// Pointer / wheel dispatch at a point (testdriver Actions executor)
// ---------------------------------------------------------------------------
//
// The windowed input path hit-tests through `window::target_at`;
// that function is private to `window.rs`, so the executor's
// dispatch layer carries its own copy of the same walk here (the
// algorithm — deepest/topmost box, nested-scroll shifts and clips,
// `transform: translate()` ancestors — is identical, and both read
// the same scroll tree, so the two cannot disagree about geometry).

/// Whether `rect` contains the point `(x, y)` (CSS px).
fn hit_rect_contains(rect: &bowser_layout::Rect, x: f32, y: f32) -> bool {
    x >= rect.x && x < rect.x + rect.w && y >= rect.y && y < rect.y + rect.h
}

/// Intersection of two rects; `None` when they do not overlap with
/// positive area (mirrors `window::rect_intersect`).
fn hit_rect_intersect(
    a: &bowser_layout::Rect,
    b: &bowser_layout::Rect,
) -> Option<bowser_layout::Rect> {
    let x = a.x.max(b.x);
    let y = a.y.max(b.y);
    let right = (a.x + a.w).min(b.x + b.w);
    let bottom = (a.y + a.h).min(b.y + b.h);
    if right > x && bottom > y {
        Some(bowser_layout::Rect {
            x,
            y,
            w: right - x,
            h: bottom - y,
        })
    } else {
        None
    }
}

/// The CSS Appendix E paint layer of one child box within its
/// parent's stacking context, as a `(layer, z)` key: paint order is
/// ascending by this key, then by document (child-list) order.
///
/// Layers: `0` = stacking contexts with negative `z-index`
/// (ascending `z`), `1` = non-positioned in-flow boxes, `2` =
/// positioned boxes with `z-index: auto` and `z-index: 0` stacking
/// contexts, `3` = stacking contexts with positive `z-index`
/// (ascending `z`).
///
/// `LayoutBox::z_index` is `Some` exactly where `z-index` applies —
/// positioned boxes with a non-`auto` value, plus static flex/grid
/// items (whose non-`auto` `z-index` also forms a stacking context;
/// see the field's docs in `bowser_layout`) — so a `Some` value
/// always participates in z-ordering here, positioned or not, and a
/// positioned box with `z-index: auto` (`None`) lands in layer 2.
fn hit_paint_key(child: &LayoutBox) -> (i32, i32) {
    let positioned = child
        .style
        .as_deref()
        .is_some_and(|s| bowser_layout::position_of(s) != bowser_layout::Position::Static);
    match child.z_index {
        Some(z) if z < 0 => (0, z),
        Some(z) if z > 0 => (3, z),
        Some(_) => (2, 0),
        None if positioned => (2, 0),
        None => (1, 0),
    }
}

/// Child indices of `parent` in hit-test visit order (topmost
/// first): reverse paint order — descending [`hit_paint_key`],
/// with later document order winning ties (later siblings paint
/// above earlier ones at the same layer).
fn hit_visit_order(parent: &LayoutBox) -> Vec<usize> {
    let mut order: Vec<usize> = (0..parent.children.len()).collect();
    order.sort_by(|&a, &b| {
        hit_paint_key(&parent.children[a])
            .cmp(&hit_paint_key(&parent.children[b]))
            .then(a.cmp(&b))
    });
    order.reverse();
    order
}

/// Deepest layout box containing the page point `(x, y)`, or `None`:
/// the same walk as `window::hit_test` (see its docs for the
/// shift/clip/translate invariants), with one deliberate deviation:
/// descent is not pruned by ancestor containment.
///
/// Children are tested even when the point lies outside this box:
/// overflow-visible content (an absolutely positioned box extending
/// past a short root/body box, say) paints — and is hit — outside
/// its ancestors' border boxes. `window::hit_test` prunes the
/// descent on ancestor containment instead; the executor needs the
/// elementFromPoint behavior, because WPT action targets are
/// routinely positioned boxes overflowing a near-empty body.
///
/// Children are visited in reverse paint order (see
/// [`hit_visit_order`]): effective stacking order — `z-index`
/// within the stacking context first, document order as the
/// tiebreak — consistently with how paint orders the same boxes,
/// so an earlier-in-DOM sibling with a higher `z-index` wins over
/// a later sibling, and a negative-`z-index` box loses to in-flow
/// content above it.
///
/// The walk is iterative (explicit stack, two task kinds) rather
/// than recursive: layout trees can be arbitrarily deep and the
/// recursive form could overflow the stack on adversarial pages.
/// A `Visit` task expands a box (pushing its own `Check` below its
/// children's `Visit`s, so children resolve first — deepest wins);
/// a `Check` task reports the box's own node when the point lies
/// in its rect and no child hit.
#[allow(clippy::too_many_arguments)]
fn hit_test_box(
    tree: &crate::scroll::ScrollTree,
    root: &LayoutBox,
    x: f32,
    y: f32,
    shift: (f32, f32),
    tshift: (f32, f32),
    tctx: crate::scroll::TranslateCtx,
    clip: Option<bowser_layout::Rect>,
) -> Option<NodeId> {
    enum HitTask<'a> {
        Visit(
            &'a LayoutBox,
            (f32, f32),
            (f32, f32),
            Option<bowser_layout::Rect>,
        ),
        Check(&'a LayoutBox, bowser_layout::Rect),
    }
    let mut stack: Vec<HitTask<'_>> = vec![HitTask::Visit(root, shift, tshift, clip)];
    while let Some(task) = stack.pop() {
        match task {
            HitTask::Check(b, rect) => {
                // A hit on an anonymous box (`node == None`) is not
                // a result: the recursive form returned `None` here
                // and the caller kept searching siblings, which the
                // tasks remaining on the stack now do.
                if hit_rect_contains(&rect, x, y)
                    && let Some(node) = b.node
                {
                    return Some(node);
                }
            }
            HitTask::Visit(b, shift, tshift, clip) => {
                let (ttx, tty) =
                    crate::scroll::box_translate(b.style.as_deref(), b.rect.w, b.rect.h, tctx);
                let tshift = (tshift.0 + ttx, tshift.1 + tty);
                let rect = bowser_layout::Rect {
                    x: b.rect.x - shift.0 + tshift.0,
                    y: b.rect.y - shift.1 + tshift.1,
                    w: b.rect.w,
                    h: b.rect.h,
                };
                // A scrollport clip from an ancestor bounds the
                // whole subtree: content clipped out of a scroller
                // is not hittable anywhere (not even the box itself).
                if clip.is_some_and(|c| !hit_rect_contains(&c, x, y)) {
                    continue;
                }
                let (child_shift, child_clip) = match b.node.and_then(|n| tree.index_of(n)) {
                    Some(i) => {
                        let container = tree.container(i);
                        let (ox, oy) = container.state.offset();
                        let port = bowser_layout::Rect {
                            x: container.scrollport.x - shift.0 + tshift.0,
                            y: container.scrollport.y - shift.1 + tshift.1,
                            w: container.scrollport.w,
                            h: container.scrollport.h,
                        };
                        let clipped = match clip {
                            Some(c) => hit_rect_intersect(&c, &port),
                            None => Some(port),
                        };
                        let Some(clipped) = clipped else {
                            // The subtree is clipped away entirely;
                            // only the container's own box can still
                            // be hit.
                            stack.push(HitTask::Check(b, rect));
                            continue;
                        };
                        ((shift.0 + ox, shift.1 + oy), Some(clipped))
                    }
                    None => (shift, clip),
                };
                stack.push(HitTask::Check(b, rect));
                // Push children in ascending paint order so the
                // topmost child pops (and resolves) first.
                let mut visit = hit_visit_order(b);
                visit.reverse();
                for idx in visit {
                    stack.push(HitTask::Visit(
                        &b.children[idx],
                        child_shift,
                        tshift,
                        child_clip,
                    ));
                }
            }
        }
    }
    None
}

/// The axis-aligned scale factors of a box's computed `transform`,
/// plus whether the transform contains a component this module
/// does not model (`rotate()`, `skew()`, `matrix()`, the 3D forms —
/// anything retained as `TransformFunction::Other`, and `Rotate`).
///
/// Only `scale()` factors are extracted (the parser folds
/// `scalex()`/`scaley()` into `Scale`); a non-numeric scale
/// argument resolves as `1.0` (the parser rejects those, so this
/// is fail-soft, never silent geometry invention). Used by the
/// in-view-centre geometry (`element_client_center`); hit testing
/// deliberately stays translate-only, matching paint, which does
/// not render scale/rotate either (see `scroll::box_translate`).
fn box_scale(style: Option<&bowser_css::ComputedStyle>) -> ((f32, f32), bool) {
    use bowser_css::{CssValue, TransformFunction};
    let Some(style) = style else {
        return ((1.0, 1.0), false);
    };
    let mut scale = (1.0f32, 1.0f32);
    let mut unmodelled = false;
    let mut apply = |fns: &[TransformFunction]| {
        for f in fns {
            match f {
                TransformFunction::Scale { x, y } => {
                    if let CssValue::Number(n) = x {
                        scale.0 *= n;
                    }
                    if let CssValue::Number(n) = y {
                        scale.1 *= n;
                    }
                }
                TransformFunction::Rotate { .. } | TransformFunction::Other { .. } => {
                    unmodelled = true;
                }
                TransformFunction::Translate { .. } => {}
            }
        }
    };
    match style.get("transform") {
        Some(CssValue::Transform(fns)) => apply(fns),
        Some(CssValue::List(items)) => {
            for item in items {
                if let CssValue::Transform(fns) = item {
                    apply(fns);
                }
            }
        }
        _ => {}
    }
    (scale, unmodelled)
}

/// The centre of `rect` clipped to the viewport and to every rect
/// in `clips` (ancestor scroller clip rects, all in client
/// coordinates): the WebDriver "in-view centre point". `None` when
/// the visible intersection is empty (the element has no in-view
/// point to aim at).
fn in_view_center(
    rect: &bowser_layout::Rect,
    viewport: &bowser_layout::Rect,
    clips: &[bowser_layout::Rect],
) -> Option<(f64, f64)> {
    let mut visible = hit_rect_intersect(rect, viewport)?;
    for clip in clips {
        visible = hit_rect_intersect(&visible, clip)?;
    }
    Some((
        (visible.x + visible.w / 2.0) as f64,
        (visible.y + visible.h / 2.0) as f64,
    ))
}

/// Pointer/mouse and wheel dispatch for [`PageEventDispatch`]: the
/// primitives the testdriver Actions executor (see
/// `crate::testdriver`) is built on. Pointer/mouse events go through
/// [`Vm::dispatch_pointer_mouse`](bowser_js::Vm::dispatch_pointer_mouse)
/// (trusted, full `PointerData`); wheels go through the
/// [`JsDispatch`](crate::window::JsDispatch) wheel path plus the
/// default scroll.
impl PageEventDispatch {
    /// The translate-resolution context for `tab`'s layout tree (the
    /// same bases `window::target_at` uses).
    fn translate_ctx(tab: &crate::tab::Tab, root: &LayoutBox) -> crate::scroll::TranslateCtx {
        crate::scroll::TranslateCtx {
            root_font_size: root
                .style
                .as_deref()
                .map(|s| crate::scroll::px_font_size(s, 16.0))
                .unwrap_or(16.0),
            viewport_w: tab.viewport.w,
            viewport_h: tab.viewport.h,
        }
    }

    /// Hit-test target at a viewport position (CSS px): the deepest,
    /// topmost element box containing the point, accounting for the
    /// viewport scroll offset, nested scroll containers' live
    /// offsets and clips, and `transform: translate()` ancestors.
    /// `None` when nothing is laid out or the point misses every box.
    pub fn hit_test_at_point(&self, tab: &crate::tab::Tab, x: f32, y: f32) -> Option<NodeId> {
        let (sx, sy) = tab.scroll().offset();
        let root = tab.layout_root.as_deref()?;
        let tctx = Self::translate_ctx(tab, root);
        hit_test_box(
            &tab.scroll_tree,
            root,
            x + sx,
            y + sy,
            (0.0, 0.0),
            (0.0, 0.0),
            tctx,
            None,
        )
    }

    /// The border-box rect of `node`'s layout box in *client*
    /// (viewport) coordinates: the layout rect shifted by its scroll
    /// containers' live offsets and its `transform: translate()` /
    /// `scale()` ancestors, minus the viewport scroll offset. The same
    /// geometry `getBoundingClientRect` reports (the bindings compute
    /// it from the same published tree). `None` when the node has no box.
    pub fn element_client_rect(
        &self,
        tab: &crate::tab::Tab,
        node: NodeId,
    ) -> Option<bowser_layout::Rect> {
        self.element_client_geometry(tab, node)
            .map(|(rect, _, _, _)| rect)
    }

    /// The geometry of `node`'s box in client coordinates: its
    /// border-box rect, the accumulated ancestor-scroller clip (the
    /// intersection of every ancestor scroll container's scrollport,
    /// in client coordinates; `None` when no ancestor clips, and
    /// `clip_empty` set when the intersection is empty), and whether
    /// any box on the path carries a transform component this walk
    /// does not model (see [`element_client_center`]).
    ///
    /// The walk is iterative and carries an axis-aligned affine
    /// (`sx, sy, tx, ty`: layout page point -> client point) instead
    /// of the old shift/translate pair: each box's own transform
    /// composes into the affine as a scale about its
    /// `transform-origin` (default `50% 50%`, resolved like layout
    /// resolves origins; an unresolvable origin component falls back
    /// to the centre on that axis) plus its `translate()` offset
    /// (via `scroll::box_translate`, in the ancestor-scaled frame).
    /// A box's own scroll offset folds into the affine translation
    /// for its children only (its own rect does not move), matching
    /// the `hit_test_box` invariants. Uniform and non-uniform
    /// axis-aligned scales are exact under this model.
    fn element_client_geometry(
        &self,
        tab: &crate::tab::Tab,
        node: NodeId,
    ) -> Option<(bowser_layout::Rect, Option<bowser_layout::Rect>, bool, bool)> {
        // An axis-aligned affine: `(x, y) -> (sx * x + tx, sy * y + ty)`.
        #[derive(Clone, Copy)]
        struct Affine {
            sx: f32,
            sy: f32,
            tx: f32,
            ty: f32,
        }
        impl Affine {
            fn apply(&self, x: f32, y: f32) -> (f32, f32) {
                (self.sx * x + self.tx, self.sy * y + self.ty)
            }
            /// The client rect of a page-space rect under this
            /// affine (corner-normalized, so negative scales work).
            fn rect(&self, r: &bowser_layout::Rect) -> bowser_layout::Rect {
                let (x0, y0) = self.apply(r.x, r.y);
                let (x1, y1) = self.apply(r.x + r.w, r.y + r.h);
                bowser_layout::Rect {
                    x: x0.min(x1),
                    y: y0.min(y1),
                    w: (x1 - x0).abs(),
                    h: (y1 - y0).abs(),
                }
            }
        }
        let root = tab.layout_root.as_deref()?;
        let (sx, sy) = tab.scroll().offset();
        let tctx = Self::translate_ctx(tab, root);
        // Stack entries: the box, the affine of its ancestors, the
        // accumulated ancestor clip, whether that clip is already
        // empty, and whether an unmodelled transform was seen on
        // the path so far.
        type GeoWalk<'a> = Vec<(
            &'a LayoutBox,
            Affine,
            Option<bowser_layout::Rect>,
            bool,
            bool,
        )>;
        let initial = Affine {
            sx: 1.0,
            sy: 1.0,
            tx: -sx,
            ty: -sy,
        };
        let mut stack: GeoWalk = vec![(root, initial, None, false, false)];
        while let Some((b, anc, clip, clip_empty, unmodelled)) = stack.pop() {
            let (ttx, tty) =
                crate::scroll::box_translate(b.style.as_deref(), b.rect.w, b.rect.h, tctx);
            let ((osx, osy), own_unmodelled) = box_scale(b.style.as_deref());
            let unmodelled = unmodelled || own_unmodelled;
            // The transform origin in layout coordinates (default
            // `50% 50%` of the border box).
            let (ox, oy) = match b
                .style
                .as_deref()
                .and_then(|s| s.get("transform-origin"))
                .and_then(bowser_css::TransformOrigin::parse)
            {
                Some(origin) => (
                    bowser_css::resolve_absolute_length(&origin.x, b.rect.w)
                        .unwrap_or(b.rect.w / 2.0),
                    bowser_css::resolve_absolute_length(&origin.y, b.rect.h)
                        .unwrap_or(b.rect.h / 2.0),
                ),
                None => (b.rect.w / 2.0, b.rect.h / 2.0),
            };
            let qo = anc.apply(b.rect.x + ox, b.rect.y + oy);
            // This box's own transform, applied in the ancestor
            // frame: scale about the origin point, then translate
            // (the translate resolves in ancestor-scaled units).
            let incl = Affine {
                sx: osx * anc.sx,
                sy: osy * anc.sy,
                tx: osx * anc.tx + (qo.0 - osx * qo.0) + anc.sx * ttx,
                ty: osy * anc.ty + (qo.1 - osy * qo.1) + anc.sy * tty,
            };
            let rect = incl.rect(&b.rect);
            if b.node == Some(node) {
                return Some((rect, clip, clip_empty, unmodelled));
            }
            // Children: fold this box's scroll offset (when it is a
            // scroll container) into the affine, and its scrollport
            // into the accumulated clip.
            let (child_affine, child_clip, child_empty) =
                match b.node.and_then(|n| tab.scroll_tree.index_of(n)) {
                    Some(i) => {
                        let container = tab.scroll_tree.container(i);
                        let (sox, soy) = container.state.offset();
                        let folded = Affine {
                            tx: incl.tx - incl.sx * sox,
                            ty: incl.ty - incl.sy * soy,
                            ..incl
                        };
                        let port = incl.rect(&container.scrollport);
                        let (c, empty) = if clip_empty {
                            (clip, true)
                        } else {
                            match clip {
                                Some(c) => match hit_rect_intersect(&c, &port) {
                                    Some(i2) => (Some(i2), false),
                                    None => (clip, true),
                                },
                                None => (Some(port), false),
                            }
                        };
                        (folded, c, empty)
                    }
                    None => (incl, clip, clip_empty),
                };
            for child in b.children.iter() {
                stack.push((child, child_affine, child_clip, child_empty, unmodelled));
            }
        }
        None
    }

    /// The in-view centre of `node` in client coordinates (the
    /// WebDriver "in-view centre point"): the centre of the
    /// element's border box intersected with the viewport and with
    /// every ancestor scroller's clip rect. `None` when the node
    /// has no box or the visible intersection is empty (WebDriver
    /// "move target out of bounds" for element-origin actions).
    ///
    /// Transform handling: `translate()` and axis-aligned `scale()`
    /// ancestors (uniform or not) are accounted exactly (see
    /// [`element_client_geometry`](Self::element_client_geometry)).
    /// Limitation, documented rather than silent: `rotate()` /
    /// `skew()` / `matrix()` / 3D transform components on the path
    /// are NOT modelled — the centre is then computed from the
    /// unrotated (translate/scale-only) box, which can be wrong for
    /// rotated elements. Paint does not render those transforms
    /// yet either, so the point still matches what is painted; a
    /// rotated hit target is a known gap, not a precise result.
    pub fn element_client_center(&self, tab: &crate::tab::Tab, node: NodeId) -> Option<(f64, f64)> {
        let (rect, clip, clip_empty, _unmodelled) = self.element_client_geometry(tab, node)?;
        if clip_empty {
            return None;
        }
        let viewport = bowser_layout::Rect {
            x: 0.0,
            y: 0.0,
            w: tab.viewport.w,
            h: tab.viewport.h,
        };
        let clips: &[bowser_layout::Rect] = match &clip {
            Some(c) => std::slice::from_ref(c),
            None => &[],
        };
        in_view_center(&rect, &viewport, clips)
    }

    /// Dispatch one trusted pointer/mouse event at an explicit
    /// `target` through the VM (see
    /// [`Vm::dispatch_pointer_mouse`](bowser_js::Vm::dispatch_pointer_mouse)):
    /// the executor uses this for the boundary chains
    /// (`pointerover`/`pointerenter`/...) and the synthesized `click`,
    /// whose targets are known without a hit test. Pumps the VM and
    /// folds DOM dirt afterwards, like every other dispatch here.
    /// Returns true when a non-passive listener prevented the default.
    pub fn dispatch_pointer_mouse_at(
        &mut self,
        tab: &mut crate::tab::Tab,
        target: NodeId,
        event_type: &str,
        kind: PointerMouseKind,
        data: PointerData,
    ) -> bool {
        // Fail closed: no installed document means no JS realm to ask.
        let Some(installed) = tab.installed_mut() else {
            return false;
        };
        // Reentrancy gate (see `InstalledDocument::assert_no_live_borrow`):
        // `dispatch_pointer_mouse` runs JS listeners, whose natives
        // re-borrow the document through the registry `Weak` — so no
        // `Ref`/`RefMut` may be alive across the dispatch.
        crate::tab::InstalledDocument::assert_no_live_borrow(installed.document());
        let page_vm = installed.page_vm_mut();
        let prevented = page_vm
            .vm_mut()
            .dispatch_pointer_mouse(target, event_type, kind, data);
        page_vm.pump();
        tab.sync_dirty_flags();
        prevented
    }

    /// Hit-test the viewport point `(x, y)` and dispatch one trusted
    /// pointer/mouse event at the hit target, filling
    /// `data.offset_x`/`offset_y` from the target's client rect (the
    /// target-relative coordinates the executor cannot know before
    /// the hit). Returns the target (if any) and whether a
    /// non-passive listener prevented the default.
    pub fn dispatch_pointer_mouse_at_point(
        &mut self,
        tab: &mut crate::tab::Tab,
        x: f32,
        y: f32,
        event_type: &str,
        kind: PointerMouseKind,
        mut data: PointerData,
    ) -> (Option<NodeId>, bool) {
        let Some(target) = self.hit_test_at_point(tab, x, y) else {
            return (None, false);
        };
        if let Some(rect) = self.element_client_rect(tab, target) {
            data.offset_x = x as f64 - rect.x as f64;
            data.offset_y = y as f64 - rect.y as f64;
        }
        let prevented = self.dispatch_pointer_mouse_at(tab, target, event_type, kind, data);
        (Some(target), prevented)
    }

    /// Dispatch a `wheel` event at the hit target under the viewport
    /// point `(x, y)` and, when no non-passive listener prevented it,
    /// perform the default scroll: the hit node's innermost scroll
    /// container scrolls by `(delta_x, delta_y)` through the scroll
    /// tree's chain (the remainder chaining outward to the viewport,
    /// exactly like a physical wheel — see `window::apply_wheel`),
    /// then any `scrollsnapchanging` the movement produced fires,
    /// then `scroll` listeners fire for every container that moved
    /// (the funnel order — see [`report_synthetic_scroll`](Self::report_synthetic_scroll)).
    /// Returns true when the page prevented the default; returns
    /// false when there was no hit target to dispatch at (nothing
    /// was dispatched and nothing scrolled).
    pub fn dispatch_wheel_at_point(
        &mut self,
        tab: &mut crate::tab::Tab,
        x: f32,
        y: f32,
        delta_x: f64,
        delta_y: f64,
    ) -> bool {
        let Some(target) = self.hit_test_at_point(tab, x, y) else {
            return false;
        };
        let prevented = <Self as crate::window::JsDispatch>::dispatch_wheel(
            self, tab, target, delta_x, delta_y, 0,
        );
        if prevented {
            return true;
        }
        // Default scroll: start at the hit node's innermost scroll
        // container (an overlay intercepting the hit intercepts the
        // scroll too), falling back to the geometric innermost
        // container when the document is gone.
        let start = match tab.document_ref() {
            Some(doc) => tab.scroll_tree.container_for_node(target, &doc),
            None => tab.scroll_tree.innermost_at(x, y),
        };
        let scrolled = tab
            .scroll_tree
            .scroll_chain(start, delta_x as f32, delta_y as f32)
            .nodes();
        crate::window::sync_nested_paint(tab, &scrolled);
        self.report_synthetic_scroll(tab, &scrolled);
        false
    }

    /// Report movement from a synthetic scroll default action (wheel,
    /// key): stamp the moved containers for `scrollend` bookkeeping —
    /// the mark arms the snap backstop, which may queue
    /// `scrollsnapchanging` and launch a smooth scroll the pump later
    /// completes (headless: instantly) — then drain and fire the queued
    /// snap events BEFORE the `scroll` dispatches. That is the funnel
    /// order [`fire_scroll_listeners_for_tab`](crate::window::fire_scroll_listeners_for_tab)
    /// uses (CSS Scroll Snap 2 §5): without this drain the wheel path
    /// left snap events for the next pump iteration's top/`end_tick`
    /// drain, firing them after the movement's `scroll`. No-op when
    /// nothing moved.
    fn report_synthetic_scroll(&mut self, tab: &mut crate::tab::Tab, scrolled: &[Option<NodeId>]) {
        if scrolled.is_empty() {
            return;
        }
        tab.scroll_tree.mark_scrolled(scrolled);
        // Snap events first: the mark above queued any
        // `scrollsnapchanging` this movement produced.
        let deliveries = tab.scroll_tree.take_snap_events();
        for d in &deliveries {
            <Self as crate::window::JsDispatch>::dispatch_snap_event(
                self,
                tab,
                d.container,
                d.kind.as_str(),
                d.targets,
            );
        }
        for node in scrolled {
            match node {
                Some(n) => <Self as crate::window::JsDispatch>::dispatch_scroll(self, tab, *n),
                None => {
                    // The viewport scrolled: the window-level `scroll`
                    // event (mirrors `WindowApp::fire_window_scroll`).
                    if let Some(installed) = tab.installed_mut() {
                        crate::tab::InstalledDocument::assert_no_live_borrow(installed.document());
                        let page_vm = installed.page_vm_mut();
                        if let Err(e) = page_vm.eval("window.dispatchEvent({type:'scroll'})") {
                            page_vm.record_error(ScriptError::task_error(e));
                        }
                        page_vm.pump();
                        tab.sync_dirty_flags();
                    }
                }
            }
        }
    }
}

impl crate::window::JsDispatch for PageEventDispatch {
    fn dispatch_click(&mut self, tab: &mut crate::tab::Tab, target: NodeId) -> bool {
        // Read the inline source before borrowing the VM: both live on
        // the tab.
        let inline = tab
            .document_ref()
            .and_then(|doc| doc.get_attribute(target, "onclick").map(|s| s.to_owned()));
        let Some(installed) = tab.installed_mut() else {
            return false;
        };
        // Disjoint field borrow: the DOM handle and the realm, with no
        // `Rc` clone — the tab keeps the sole strong owner.
        let (doc, page_vm) = installed.document_and_vm_mut();
        let mut prevented = false;
        // Reentrancy gate (see `InstalledDocument::assert_no_live_borrow`):
        // the inline source and the `dispatchEvent` eval re-enter the page
        // VM while `doc` is held, so no `Ref`/`RefMut` may be alive across
        // it.
        crate::tab::InstalledDocument::assert_no_live_borrow(doc);
        // One eval: the inline handler runs with the dispatched event
        // object bound as `event`, then the same object goes to the
        // listeners via `dispatchEvent`.
        if self.dispatch_click_with_inline(doc, page_vm, target, inline.as_deref()) {
            prevented = true;
        }
        page_vm.pump();
        tab.sync_dirty_flags();
        prevented
    }

    fn dispatch_key(&mut self, tab: &mut crate::tab::Tab, _key: &str) -> bool {
        // Inline `onkeydown` on `<html>`, falling back to `<body>`.
        let (node, inline) = match tab.document_ref() {
            Some(doc) => {
                let html = doc.document_element();
                let body = doc.get_elements_by_tag_name("body").into_iter().next();
                let node = html.or(body).unwrap_or_else(|| doc.root());
                let inline = doc.get_attribute(node, "onkeydown").map(|s| s.to_owned());
                (node, inline)
            }
            None => return false,
        };
        let Some(installed) = tab.installed_mut() else {
            return false;
        };
        // Reentrancy gate (see `InstalledDocument::assert_no_live_borrow`):
        // the inline `onkeydown` source and the `dispatchEvent` eval below
        // re-enter the page VM, whose natives re-borrow the document through
        // the registry `Weak` — so no `Ref`/`RefMut` may be alive across
        // either eval.
        crate::tab::InstalledDocument::assert_no_live_borrow(installed.document());
        let page_vm = installed.page_vm_mut();
        let mut prevented = false;
        // One eval: the inline handler runs with the dispatched event
        // object bound as `event`, then the same object goes to the
        // document-level listeners via `dispatchEvent`.
        if Self::eval_inline_and_dispatch(
            page_vm,
            node,
            "document",
            "{type:'keydown',bubbles:true,cancelable:true}",
            inline.as_deref(),
        ) {
            prevented = true;
        }
        page_vm.pump();
        tab.sync_dirty_flags();
        prevented
    }

    fn dispatch_wheel(
        &mut self,
        tab: &mut crate::tab::Tab,
        target: NodeId,
        delta_x: f64,
        delta_y: f64,
        delta_mode: u32,
    ) -> bool {
        // Fail closed: no installed document means no JS realm to ask.
        let Some(installed) = tab.installed_mut() else {
            return false;
        };
        // Reentrancy gate (see `InstalledDocument::assert_no_live_borrow`):
        // `dispatch_wheel` runs JS listeners, whose natives re-borrow the
        // document through the registry `Weak` — so no `Ref`/`RefMut` may
        // be alive across the dispatch.
        crate::tab::InstalledDocument::assert_no_live_borrow(installed.document());
        let page_vm = installed.page_vm_mut();
        // Trusted, bubbling, cancelable `wheel` event; returns true iff a
        // non-passive listener prevented the default (already fail-closed
        // when the realm is gone).
        let prevented = page_vm
            .vm_mut()
            .dispatch_wheel(target, delta_x, delta_y, delta_mode);
        // Same bookkeeping as clicks and keys: pump handler-scheduled
        // work, then fold DOM dirt into the tab's render flags.
        page_vm.pump();
        tab.sync_dirty_flags();
        prevented
    }

    fn dispatch_scroll(&mut self, tab: &mut crate::tab::Tab, target: NodeId) {
        let Some(installed) = tab.installed_mut() else {
            return;
        };
        // Disjoint field borrow: the DOM handle and the realm, with no
        // `Rc` clone — the tab keeps the sole strong owner.
        let (doc, page_vm) = installed.document_and_vm_mut();
        self.dispatch_scroll_listeners(doc, page_vm, target, "scroll");
        page_vm.pump();
        tab.sync_dirty_flags();
    }

    fn dispatch_scrollend(&mut self, tab: &mut crate::tab::Tab, target: NodeId) {
        let Some(installed) = tab.installed_mut() else {
            return;
        };
        // Disjoint field borrow: the DOM handle and the realm, with no
        // `Rc` clone — the tab keeps the sole strong owner.
        let (doc, page_vm) = installed.document_and_vm_mut();
        self.dispatch_scroll_listeners(doc, page_vm, target, "scrollend");
        page_vm.pump();
        tab.sync_dirty_flags();
    }

    fn dispatch_snap_event(
        &mut self,
        tab: &mut crate::tab::Tab,
        target: Option<NodeId>,
        event_type: &str,
        targets: crate::scroll::SnapTargets,
    ) {
        let Some(installed) = tab.installed_mut() else {
            return;
        };
        // Reentrancy gate (see `InstalledDocument::assert_no_live_borrow`):
        // the dispatch runs JS listeners, whose natives re-borrow the
        // document through the registry `Weak` — so no `Ref`/`RefMut`
        // may be alive across it. The document root is read (and the
        // borrow dropped) before the gate.
        let root = installed.document().borrow().root();
        crate::tab::InstalledDocument::assert_no_live_borrow(installed.document());
        let page_vm = installed.page_vm_mut();
        // A real `SnapEvent` wrapper built in Rust (the plain-object
        // eval the `scroll` dispatch uses cannot carry node payloads):
        // element containers dispatch at the element (no bubbling);
        // the viewport dispatches at the document root with
        // `at_document`, so `event.target` is the document and the
        // event bubbles to `window`.
        match target {
            Some(node) => {
                page_vm.vm_mut().dispatch_snap_event(
                    node,
                    event_type,
                    targets.block,
                    targets.inline,
                    false,
                );
            }
            None => {
                page_vm.vm_mut().dispatch_snap_event(
                    root,
                    event_type,
                    targets.block,
                    targets.inline,
                    true,
                );
            }
        }
        page_vm.pump();
        tab.sync_dirty_flags();
    }
}

impl Default for PageEventDispatch {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn parse(html: &str) -> Document {
        Document::parse_html(html.as_bytes()).0
    }

    fn base() -> Url {
        bowser_url::parse("http://example.com/page.html", None).expect("test url")
    }

    const MIXED_HTML: &str = r#"<!DOCTYPE html><html><head>
        <script>var a = 1;</script>
        <script src="/app.js"></script>
        <script type="module">import x from './m.js';</script>
        <script type="application/json">{"not": "code"}</script>
        <script src="http://[::1">broken</script>
        <script>var b = 2;</script>
        </head><body></body></html>"#;

    fn resolve_all(document: &Document) -> Vec<PageScript> {
        let base = base();
        resolve_scripts(collect_script_refs(document), &base, |url| {
            Ok(format!("// fetched {}", url.serialize()))
        })
    }

    #[test]
    fn collect_finds_scripts_in_document_order() {
        let doc = parse(MIXED_HTML);
        let refs = collect_script_refs(&doc);
        assert_eq!(refs.len(), 6);
        assert_eq!(refs[0].inline_text, "var a = 1;");
        assert_eq!(refs[1].src.as_deref(), Some("/app.js"));
        assert_eq!(refs[2].script_type.as_deref(), Some("module"));
        assert_eq!(refs[3].script_type.as_deref(), Some("application/json"));
        assert_eq!(refs[5].inline_text, "var b = 2;");
    }

    #[test]
    fn resolve_classifies_and_orders_scripts() {
        let doc = parse(MIXED_HTML);
        let scripts = resolve_all(&doc);
        assert_eq!(scripts.len(), 6);

        assert!(matches!(scripts[0], PageScript::Inline { .. }));
        match &scripts[1] {
            PageScript::External { url, source, .. } => {
                assert_eq!(url.serialize(), "http://example.com/app.js");
                assert!(source.contains("fetched"));
            }
            other => panic!("expected external, got {other:?}"),
        }
        assert!(
            matches!(
                &scripts[2],
                PageScript::Skipped {
                    reason: SkipReason::Module,
                    ..
                }
            ),
            "module skipped: {:?}",
            scripts[2]
        );
        assert!(
            matches!(
                &scripts[3],
                PageScript::Skipped {
                    reason: SkipReason::NonExecutableType(_),
                    ..
                }
            ),
            "data block skipped: {:?}",
            scripts[3]
        );
        assert!(
            matches!(
                &scripts[4],
                PageScript::Skipped {
                    reason: SkipReason::BadUrl(_),
                    ..
                }
            ),
            "bad URL skipped: {:?}",
            scripts[4]
        );
        assert!(matches!(scripts[5], PageScript::Inline { .. }));
    }

    #[test]
    fn script_src_resolves_against_base_url() {
        let doc = parse(r#"<script src="js/app.js"></script>"#);
        let base = bowser_url::parse("http://example.com/sub/page.html", None).unwrap();
        let scripts = resolve_scripts(collect_script_refs(&doc), &base, |url| {
            assert_eq!(url.serialize(), "http://example.com/sub/js/app.js");
            Ok(String::new())
        });
        assert!(matches!(scripts[0], PageScript::External { .. }));
    }

    #[test]
    fn executable_types_match_bindings_list() {
        // `type` values the bindings worker treats as classic scripts
        // must resolve to Inline here, and vice versa.
        for t in [
            "",
            "text/javascript",
            "application/javascript",
            "text/ecmascript",
            "application/ecmascript",
            "  TEXT/JAVASCRIPT  ",
        ] {
            let doc = parse(&format!(r#"<script type="{t}">1;</script>"#));
            let scripts = resolve_all(&doc);
            assert!(
                matches!(scripts[0], PageScript::Inline { .. }),
                "type {t:?} should execute"
            );
        }
        for t in ["application/json", "text/plain", "importmap"] {
            let doc = parse(&format!(r#"<script type="{t}">1;</script>"#));
            let scripts = resolve_all(&doc);
            assert!(
                matches!(
                    &scripts[0],
                    PageScript::Skipped {
                        reason: SkipReason::NonExecutableType(_),
                        ..
                    }
                ),
                "type {t:?} should be skipped"
            );
        }
    }

    #[test]
    fn external_scripts_are_capped() {
        let mut html = String::from("<!DOCTYPE html><html><head>");
        for i in 0..MAX_SCRIPTS + 5 {
            html.push_str(&format!(r#"<script src="/s{i}.js"></script>"#));
        }
        html.push_str("</head><body></body></html>");
        let doc = parse(&html);
        let base = base();
        let mut fetched = 0usize;
        let scripts = resolve_scripts(collect_script_refs(&doc), &base, |_| {
            fetched += 1;
            Ok(String::new())
        });
        assert_eq!(fetched, MAX_SCRIPTS);
        assert_eq!(scripts.len(), MAX_SCRIPTS + 5);
        assert!(matches!(
            scripts[MAX_SCRIPTS - 1],
            PageScript::External { .. }
        ));
        assert!(
            matches!(
                &scripts[MAX_SCRIPTS],
                PageScript::Skipped {
                    reason: SkipReason::Fetch(_),
                    ..
                }
            ),
            "over-cap scripts are skipped, not fetched"
        );
    }

    #[test]
    fn fetch_failures_become_skips() {
        let doc = parse(r#"<script src="/missing.js"></script><script>ok();</script>"#);
        let base = base();
        let scripts = resolve_scripts(collect_script_refs(&doc), &base, |_| {
            Err("connection refused".to_string())
        });
        assert!(matches!(
            &scripts[0],
            PageScript::Skipped {
                reason: SkipReason::Fetch(_),
                ..
            }
        ));
        assert!(matches!(scripts[1], PageScript::Inline { .. }));
    }

    #[test]
    fn page_vm_runs_scripts_in_order_sharing_globals() {
        // Cross-script state rides the global object (sloppy-mode
        // assignment): each script's lexical `var`s are per-script in the
        // current engine, so the shared-state path is the global object.
        let doc = parse(
            "<script>total = 10;</script>\
             <script>total += 5;</script>\
             <script>total *= 2;</script>",
        );
        let scripts = resolve_all(&doc);
        let mut page = PageVm::new();
        page.run(&scripts);
        let report = page.report();
        assert_eq!(report.executed, 3);
        assert_eq!(report.skipped, 0);
        assert!(report.errors.is_empty());
        // The realm persists across scripts in one run...
        assert_eq!(page.eval("total").ok(), Some(Value::Int(30)));
    }

    #[test]
    fn page_vm_eval_is_the_event_dispatch_entry_point() {
        let doc = parse("<script>clicks = 0;</script>");
        let scripts = resolve_all(&doc);
        let mut page = PageVm::new();
        page.run(&scripts);
        // A dispatched handler runs in the same realm: it sees the
        // page's globals...
        page.eval("clicks += 1;").expect("dispatch eval");
        // ...and its queued microtasks are drained, like the
        // between-scripts checkpoint.
        page.eval("seen = []; Promise.resolve().then(() => seen.push(clicks));")
            .expect("dispatch eval");
        assert!(js_is_true(&mut page, "seen.join(',') === '1'"));
        assert_eq!(page.eval("clicks").ok(), Some(Value::Int(1)));
        // Evaluation failures surface to the dispatcher, not the report:
        // `run`'s report only covers the page's own scripts.
        assert!(page.eval("notDefinedVariable").is_err());
        assert!(page.report().errors.is_empty());
    }

    #[test]
    fn page_vm_dynamic_import_resolves_against_document_base_url() {
        // A classic page script calling `import()`: the specifier resolves
        // against the document URL (wired through `install_document_apis`
        // in production; set directly here for the bare test realm), and
        // the between-scripts checkpoint settles the promise.
        let doc = parse(
            "<script>globalThis.answer = null; \
             import('./m.js').then(ns => { globalThis.answer = ns.answer; });</script>",
        );
        let scripts = resolve_all(&doc);
        let mut page = PageVm::new();
        page.base_url = Some(base());
        let loader = page.vm_mut().module_loader_mut();
        loader.add_source("http://example.com/m.js", "export const answer = 42;");
        loader.set_allow_net(false);
        page.run(&scripts);
        assert!(page.report().errors.is_empty());
        assert_eq!(page.eval("answer").ok(), Some(Value::Int(42)));
    }

    #[test]
    fn page_vm_records_errors_and_continues() {
        let doc = parse(
            "<script>var x = 1;</script>\
             <script>this is not javascript (;</script>\
             <script>x = 2;</script>\
             <script>throw new Error('boom');</script>\
             <script>x = 3;</script>",
        );
        let scripts = resolve_all(&doc);
        let mut page = PageVm::new();
        page.run(&scripts);

        let report = page.report();
        assert_eq!(report.executed, 5);
        assert_eq!(report.errors.len(), 2, "parse error + throw: {report:?}");
        assert!(report.errors[0].url.is_none(), "inline script");
        assert!(
            report
                .errors
                .iter()
                .all(|e| e.kind.to_string().contains("script evaluation failed")),
            "{report:?}"
        );
        // Later scripts still ran (the sloppy-global writes landed).
        assert_eq!(page.eval("x").ok(), Some(Value::Int(3)));
    }

    #[test]
    fn page_vm_drains_microtasks_between_scripts() {
        // The shared log is a sloppy global (visible across evals); the
        // microtask closure captures it as an upvalue either way.
        let doc = parse(
            "<script>log = []; Promise.resolve().then(() => log.push('micro'));</script>\
             <script>log.push('second');</script>",
        );
        let scripts = resolve_all(&doc);
        let mut page = PageVm::new();
        page.run(&scripts);
        assert!(page.report().errors.is_empty());
        // The microtask queued by the first script ran before the second
        // script's synchronous push: drain between scripts is the
        // checkpoint.
        assert!(js_is_true(&mut page, "log.join(',') === 'micro,second'"));
    }

    #[test]
    fn page_vm_counts_skips() {
        let doc = parse(
            "<script>1;</script>\
             <script type=\"module\">2;</script>\
             <script type=\"application/json\">3</script>",
        );
        let scripts = resolve_all(&doc);
        let mut page = PageVm::new();
        page.run(&scripts);
        let report = page.report();
        assert_eq!(report.executed, 1);
        assert_eq!(report.skipped, 2);
        assert!(report.errors.is_empty());
    }

    #[test]
    fn script_node_returns_the_element() {
        let doc = parse("<script>1;</script><script type=\"module\">2;</script>");
        let scripts = resolve_all(&doc);
        let nodes: Vec<NodeId> = scripts.iter().map(script_node).collect();
        assert_eq!(nodes.len(), 2);
        assert_ne!(nodes[0], nodes[1]);
    }

    /// Evaluate `expr` and expect the boolean `true`: avoids asserting
    /// on `Value`'s `Debug` rendering.
    fn js_is_true(page: &mut PageVm, expr: &str) -> bool {
        matches!(page.eval(expr), Ok(Value::Bool(true)))
    }

    #[test]
    fn pump_fires_due_timers_but_not_future_ones() {
        // Timer effects are observed through sloppy globals: each eval's
        // lexical scope is its own, but the global object is shared.
        let doc = parse(
            "<script>setTimeout(() => { firedNow = true; }, 0);\
             setTimeout(() => { firedLater = true; }, 60000);\
             </script>",
        );
        let scripts = resolve_all(&doc);
        let mut page = PageVm::new();
        page.run(&scripts);
        assert!(page.report().errors.is_empty());

        // Timers do not fire during `run` (microtask checkpoint only)...
        // (`global` reads the shared global object from the host side,
        // sidestepping the engine's throwing `typeof` on missing globals.)
        assert_eq!(page.global("firedNow"), None);
        // ...nor during a dispatch eval.
        page.eval("1 + 1;").expect("dispatch eval");
        assert_eq!(page.global("firedNow"), None);

        page.pump();

        // The 0 ms timer fired; the 60 s one stays pending.
        assert_eq!(page.global("firedNow"), Some(Value::Bool(true)));
        assert_eq!(page.global("firedLater"), None);
    }

    #[test]
    fn pump_drains_microtasks_queued_by_timer_callbacks() {
        let doc = parse(
            "<script>setTimeout(() => {\
                 order = 'macro';\
                 Promise.resolve().then(() => { order += ',micro'; });\
             }, 0);\
             </script>",
        );
        let scripts = resolve_all(&doc);
        let mut page = PageVm::new();
        page.run(&scripts);
        assert!(page.report().errors.is_empty());

        page.pump();

        // The timer callback ran, and the microtask it queued drained
        // after it — the same checkpoint `run` applies between scripts.
        assert!(js_is_true(&mut page, "order === 'macro,micro'"));
    }

    /// A headless page realm with the production web APIs installed: the
    /// path [`PageVm::pump`] drives through the bindings pump hook.
    /// The document's scripts are evaluated, like a real page load.
    fn page_with_web_apis(html: &str) -> PageVm {
        let doc = parse(html);
        let scripts = resolve_all(&doc);
        let doc = Rc::new(RefCell::new(doc));
        let mut page = PageVm::new_for_document(doc, &base(), "bowser-test/1.0", &[])
            .expect("install web apis");
        page.run(&scripts);
        assert!(page.report().errors.is_empty());
        page
    }

    #[test]
    fn page_pump_fires_bindings_timers_and_microtasks() {
        // The bindings' timer/microtask globals are installed on the
        // production page realm (shadowing the VM intrinsics); the pump
        // hook fires their callbacks.
        let mut page = page_with_web_apis("<script>log = [];</script>");
        page.eval("setTimeout(() => log.push('timer'), 0);")
            .expect("setTimeout");
        page.eval("queueMicrotask(() => log.push('micro'));")
            .expect("queueMicrotask");
        // The microtask ran at the between-scripts checkpoint...
        assert!(js_is_true(&mut page, "log.join(',') === 'micro'"));
        assert!(page.report().errors.is_empty());
        // ...the timer waits for the pump.
        page.pump();
        assert!(js_is_true(&mut page, "log.join(',') === 'micro,timer'"));
        assert!(page.report().errors.is_empty());
    }

    #[test]
    fn page_pump_preserves_microtask_fifo_across_bindings_and_engine() {
        // Bindings microtasks and engine promise jobs share one microtask
        // queue: FIFO order holds across the boundary, not per kind.
        let mut page = page_with_web_apis("<script>log = [];</script>");
        page.eval(
            "queueMicrotask(() => log.push('b1')); \
             Promise.resolve().then(() => log.push('e1')); \
             queueMicrotask(() => log.push('b2')); \
             Promise.resolve().then(() => log.push('e2'));",
        )
        .expect("queue microtasks");
        // All four drained at the between-scripts checkpoint, in order.
        assert!(js_is_true(&mut page, "log.join(',') === 'b1,e1,b2,e2'"));
        assert!(page.report().errors.is_empty());
    }

    #[test]
    fn page_pump_timer_ordering_microtasks_before_timers() {
        let mut page = page_with_web_apis("<script>log = [];</script>");
        page.eval(
            "setTimeout(() => log.push('t1'), 0); \
             queueMicrotask(() => log.push('m1')); \
             setTimeout(() => log.push('t2'), 0);",
        )
        .expect("register timers");
        // Microtask checkpoint only: the microtask ran, timers pending.
        assert!(js_is_true(&mut page, "log.join(',') === 'm1'"));
        page.pump();
        // Due timers fire FIFO after the microtask checkpoint.
        assert!(js_is_true(&mut page, "log.join(',') === 'm1,t1,t2'"));
        assert!(page.report().errors.is_empty());
    }

    #[test]
    fn page_pump_fires_set_interval_and_clear_interval_stops_it() {
        // Interval callbacks ride the timers module's holder object and
        // fire through the same pump hook as one-shot timers. (A 0 ms
        // interval would spin the pump forever — the documented v1
        // hazard — so the test uses a real delay and `>=` bounds, robust
        // to scheduling jitter; only the post-clear count is exact.)
        let mut page = page_with_web_apis("<script>log = [];</script>");
        page.eval("id = setInterval(() => log.push('tick'), 100);")
            .expect("setInterval");
        std::thread::sleep(std::time::Duration::from_millis(500));
        page.pump();
        // Fired repeatedly through the hook.
        assert!(js_is_true(&mut page, "log.length >= 2"));
        let before = match page.eval("log.length") {
            Ok(Value::Int(n)) => n,
            other => panic!("log.length not an int: {other:?}"),
        };
        page.eval("clearInterval(id);").expect("clearInterval");
        std::thread::sleep(std::time::Duration::from_millis(300));
        page.pump();
        // Cancelled: no further ticks, deterministically.
        assert_eq!(
            page.eval("log.length").ok(),
            Some(Value::Int(before)),
            "interval fired after clearInterval"
        );
        assert!(page.report().errors.is_empty());
    }

    #[test]
    fn page_pump_records_throwing_timer_callback() {
        // A throwing bindings timer callback lands in the report as a
        // node-less "page task" error (DOM report-and-continue), like a
        // throwing VM-intrinsic timer callback.
        let mut page = page_with_web_apis("<script>log = [];</script>");
        page.eval("setTimeout(() => { throw new Error('boom'); }, 0);")
            .expect("setTimeout");
        page.pump();
        let report = page.report();
        assert_eq!(report.errors.len(), 1, "{report:?}");
        assert!(
            report.errors[0].to_string().contains("boom"),
            "{}",
            report.errors[0]
        );
    }

    #[test]
    fn page_pump_runs_fetch_then_reactions() {
        // The headless page path: fetch() dispatches to the worker pool
        // and the .then reaction runs when the pump moves the completion
        // into the loop — the v1 gap the pump hook closes.
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let body = "hello";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len(),
            );
            stream.write_all(response.as_bytes()).expect("write");
        });

        let mut page = page_with_web_apis("<script>log = [];</script>");
        page.eval(&format!(
            "fetch('http://127.0.0.1:{port}/hello').then(r => log.push('status:' + r.status));"
        ))
        .expect("fetch");
        // The worker pool resolves the fetch asynchronously: spin the
        // pump until the reaction runs (bounded; localhost is fast).
        let mut done = false;
        for _ in 0..500 {
            page.pump();
            if js_is_true(&mut page, "log.length === 1") {
                done = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(
            done,
            "fetch .then reaction never ran: {report:?}",
            report = page.report()
        );
        assert!(js_is_true(&mut page, "log[0] === 'status:200'"));
        assert!(page.report().errors.is_empty());
        server.join().expect("server thread");
    }

    #[test]
    fn pump_surfaces_throwing_timer_callbacks_in_report() {
        let doc = parse("<script>setTimeout(() => { throw new Error('boom'); }, 0);</script>");
        let scripts = resolve_all(&doc);
        let mut page = PageVm::new();
        page.run(&scripts);
        assert!(page.report().errors.is_empty());

        page.pump();

        // The throw is not swallowed: it lands in the error console as a
        // node-less "page task" error carrying the engine's message.
        let errors = &page.report().errors;
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert_eq!(errors[0].node, None);
        let rendered = errors[0].to_string();
        assert!(rendered.starts_with("page task: "), "{rendered}");
        assert!(rendered.contains("boom"), "{rendered}");

        // Drained: a second pump with no new failures adds nothing.
        page.pump();
        assert_eq!(page.report().errors.len(), 1);
    }

    #[test]
    fn pump_bounded_terminates_spinning_interval() {
        // A `setInterval` with a 0 ms delay reschedules immediately-due:
        // the unbounded `pump` would spin forever on this page. The
        // bounded pump runs `max_tasks` tasks and reports the cut.
        let mut page = page_with_web_apis("<script>ticks = 0;</script>");
        page.eval("setInterval(() => { ticks++; }, 0);")
            .expect("setInterval");
        let start = std::time::Instant::now();
        let drained = page.pump_bounded(100);
        assert!(start.elapsed() < std::time::Duration::from_secs(10));
        assert!(!drained, "spinning page cannot drain a 100-task budget");
        assert!(js_is_true(&mut page, "ticks === 100"));
        assert!(page.report().errors.is_empty());
        // The interval is still live: a second bounded pump keeps going
        // where the first stopped — the budget cut lost no work.
        assert!(!page.pump_bounded(50));
        assert!(js_is_true(&mut page, "ticks === 150"));
    }

    #[test]
    fn pump_bounded_drains_quiet_page_fully() {
        let mut page = page_with_web_apis("<script>log = [];</script>");
        page.eval("setTimeout(() => log.push('t'), 0);")
            .expect("setTimeout");
        assert!(page.pump_bounded(10_000));
        assert!(js_is_true(&mut page, "log.join(',') === 't'"));
        assert!(page.report().errors.is_empty());
        // Nothing left: a second bounded pump is a no-op drain.
        assert!(page.pump_bounded(10_000));
    }

    #[test]
    fn harness_completion_flag_accessors() {
        let mut page = page_with_web_apis("<script></script>");
        assert!(!page.harness_reporter_armed());
        assert!(!page.harness_completed());
        // The reporter arms the flag before the completion callback runs.
        page.eval(&format!(
            "window['{flag}'] = false;",
            flag = HARNESS_COMPLETION_FLAG
        ))
        .expect("arm flag");
        assert!(page.harness_reporter_armed());
        assert!(!page.harness_completed());
        page.eval(&format!(
            "window['{flag}'] = true;",
            flag = HARNESS_COMPLETION_FLAG
        ))
        .expect("set flag");
        assert!(page.harness_reporter_armed());
        assert!(page.harness_completed());
    }

    #[test]
    fn defer_scripts_run_after_non_deferred_in_document_order() {
        let doc = parse(
            "<script>order = 'a';</script>\
             <script defer src='one.js'></script>\
             <script>order += 'b';</script>\
             <script defer src='two.js'></script>",
        );
        let refs = collect_script_refs(&doc);
        assert_eq!(refs.len(), 4);
        assert!(!refs[0].defer);
        assert!(refs[1].defer);
        assert!(!refs[2].defer);
        assert!(refs[3].defer);

        let scripts = resolve_scripts(collect_script_refs(&doc), &base(), |url| {
            Ok(match url.serialize().as_str() {
                s if s.ends_with("one.js") => "order += '1';".to_string(),
                s if s.ends_with("two.js") => "order += '2';".to_string(),
                s => panic!("unexpected script fetch: {s}"),
            })
        });
        assert!(matches!(
            &scripts[1],
            PageScript::External { defer: true, .. }
        ));
        assert!(matches!(
            &scripts[3],
            PageScript::External { defer: true, .. }
        ));

        let mut page = PageVm::new();
        page.run(&scripts);
        assert!(
            page.report().errors.is_empty(),
            "{:?}",
            page.report().errors
        );
        // Non-deferred (a, b) first in document order, then deferred (1, 2).
        assert!(
            js_is_true(&mut page, "order === 'ab12'"),
            "got {:?}",
            page.global("order")
        );
    }

    #[test]
    fn defer_without_src_is_ignored() {
        // Per spec `defer` on an inline script is ignored: it runs in
        // document order with the non-deferred scripts.
        let doc = parse("<script defer>order = 'x';</script><script>order += 'y';</script>");
        let scripts = resolve_all(&doc);
        assert!(
            scripts
                .iter()
                .all(|s| matches!(s, PageScript::Inline { .. }))
        );
        let mut page = PageVm::new();
        page.run(&scripts);
        assert!(
            page.report().errors.is_empty(),
            "{:?}",
            page.report().errors
        );
        assert!(js_is_true(&mut page, "order === 'xy'"));
    }

    #[test]
    fn async_scripts_run_after_classic_before_defer() {
        // Three passes: classic in document order, then the async batch,
        // then deferred in document order. The async scripts are out of
        // document order relative to the classic stream (the first
        // script in the document runs in the async batch, after both
        // classic scripts).
        let doc = parse(
            "<script async src='a.js'></script>\
             <script>order = 'c1';</script>\
             <script>order += 'c2';</script>\
             <script defer src='d.js'></script>\
             <script async src='b.js'></script>",
        );
        let refs = collect_script_refs(&doc);
        assert!(refs[0].is_async && refs[4].is_async);
        assert!(refs[3].defer && !refs[3].is_async);
        let scripts = resolve_scripts(refs, &base(), |url| {
            Ok(match url.serialize().as_str() {
                s if s.ends_with("a.js") => "order += 'a';".to_string(),
                s if s.ends_with("b.js") => "order += 'b';".to_string(),
                s if s.ends_with("d.js") => "order += 'd';".to_string(),
                s => panic!("unexpected script fetch: {s}"),
            })
        });
        assert!(matches!(
            &scripts[0],
            PageScript::External { is_async: true, .. }
        ));
        assert!(matches!(
            &scripts[3],
            PageScript::External {
                defer: true,
                is_async: false,
                ..
            }
        ));

        let mut page = PageVm::new();
        page.run(&scripts);
        assert!(
            page.report().errors.is_empty(),
            "{:?}",
            page.report().errors
        );
        assert_eq!(page.report().executed, 5);
        assert!(js_is_true(&mut page, "order === 'c1c2abd'"));
    }

    #[test]
    fn async_wins_over_defer() {
        // Per spec `async` takes precedence when both are present: the
        // script runs in the async batch, not the deferred pass.
        let doc = parse(
            "<script async defer src='x.js'></script>\
             <script defer src='d.js'></script>",
        );
        let refs = collect_script_refs(&doc);
        assert!(refs[0].is_async && refs[0].defer);
        let scripts = resolve_scripts(refs, &base(), |url| {
            Ok(match url.serialize().as_str() {
                s if s.ends_with("x.js") => "order = 'x';".to_string(),
                s if s.ends_with("d.js") => "order += 'd';".to_string(),
                s => panic!("unexpected script fetch: {s}"),
            })
        });
        let mut page = PageVm::new();
        page.run(&scripts);
        assert!(page.report().errors.is_empty());
        assert!(js_is_true(&mut page, "order === 'xd'"));
    }

    #[test]
    fn async_without_src_is_ignored() {
        // Per spec `async` on an inline script is ignored: it runs in
        // document order with the classic scripts, not in the async
        // batch.
        let doc = parse(
            "<script async>order = 'x';</script>\
             <script>order += 'y';</script>",
        );
        let scripts = resolve_all(&doc);
        assert!(
            scripts
                .iter()
                .all(|s| matches!(s, PageScript::Inline { .. }))
        );
        let mut page = PageVm::new();
        page.run(&scripts);
        assert!(page.report().errors.is_empty());
        assert!(js_is_true(&mut page, "order === 'xy'"));
    }

    #[test]
    fn runaway_classic_script_terminates_with_fuel_error() {
        // A non-terminating script burns its per-eval fuel budget and
        // the page load continues: the error is recorded as fuel
        // exhaustion and the next script still runs.
        let doc = parse(
            "<script>while (true) {}</script>\
             <script>reached = true;</script>",
        );
        let scripts = resolve_all(&doc);
        let mut page = PageVm::new();
        page.run(&scripts);
        let report = page.report();
        assert_eq!(report.executed, 2);
        assert_eq!(report.errors.len(), 1, "{report:?}");
        assert_eq!(report.errors[0].kind, ScriptErrorKind::FuelExhausted);
        assert_eq!(report.errors[0].node, Some(script_node(&scripts[0])));
        assert!(report.errors[0].url.is_none(), "inline script");
        assert!(
            report.errors[0].to_string().contains("budget"),
            "{}",
            report.errors[0]
        );
        // The page load continued past the runaway.
        assert_eq!(page.global("reached"), Some(Value::Bool(true)));
    }

    #[test]
    fn runaway_async_script_does_not_block_defer() {
        // The fuel budget applies per script in every pass: a spinning
        // async script is recorded and the deferred pass still runs.
        let doc = parse(
            "<script async src='spin.js'></script>\
             <script defer src='d.js'></script>",
        );
        let scripts = resolve_scripts(collect_script_refs(&doc), &base(), |url| {
            Ok(match url.serialize().as_str() {
                s if s.ends_with("spin.js") => "while (true) {}".to_string(),
                s if s.ends_with("d.js") => "deferredRan = true;".to_string(),
                s => panic!("unexpected script fetch: {s}"),
            })
        });
        let mut page = PageVm::new();
        page.run(&scripts);
        let report = page.report();
        assert_eq!(report.executed, 2);
        assert_eq!(report.errors.len(), 1, "{report:?}");
        assert_eq!(report.errors[0].kind, ScriptErrorKind::FuelExhausted);
        assert_eq!(page.global("deferredRan"), Some(Value::Bool(true)));
    }

    #[test]
    fn fuel_exhausted_kind_renders() {
        assert_eq!(
            ScriptErrorKind::FuelExhausted.to_string(),
            "script execution budget exhausted (fuel)"
        );
    }

    #[test]
    fn install_global_plants_host_objects_before_run() {
        let mut page = PageVm::new();
        // A plain value stands in for a host object here (the web APIs
        // themselves are planted by `new_for_document`).
        page.install_global("hostValue", Value::Int(41));
        // The script reads the host object; the result escapes through a
        // sloppy global (per-script lexical scope does not survive evals).
        let doc = parse("<script>answer = hostValue + 1;</script>");
        page.run(&resolve_all(&doc));
        assert!(page.report().errors.is_empty());

        // The script saw the host object from its first line...
        assert_eq!(page.eval("answer").ok(), Some(Value::Int(42)));
        // ...and the host can read it back through `global`.
        assert_eq!(page.global("hostValue"), Some(Value::Int(41)));
        assert_eq!(page.global("missing"), None);
    }

    #[test]
    fn install_global_overwrites() {
        let mut page = PageVm::new();
        page.install_global("x", Value::Int(1));
        page.install_global("x", Value::Int(2));
        assert!(js_is_true(&mut page, "x === 2"));
    }

    #[test]
    fn record_error_surfaces_dispatch_failures_in_the_report() {
        let doc = parse("<script>var ok = true;</script>");
        let scripts = resolve_all(&doc);
        let node = script_node(&scripts[0]);
        let mut page = PageVm::new();
        page.run(&scripts);
        assert!(page.report().errors.is_empty());

        // A dispatch handler that throws: `eval` returns the failure to
        // the dispatcher instead of recording it...
        // (Thrown as a string: the engine stringifies a thrown `Error`
        // object as "[object Object]" — `Error.prototype.toString` is not
        // implemented in `bowser_js` yet — so an `Error` throw would not
        // carry its message into the report. The string throw exercises
        // the same `record_error` path with a message the engine
        // preserves verbatim.)
        let err = page.eval("throw 'click boom';").unwrap_err();
        assert!(page.report().errors.is_empty());
        // ...and the dispatcher surfaces it through `record_error`.
        page.record_error(ScriptError::from_vm_error(node, None, &err));

        let report = page.report();
        assert_eq!(report.errors.len(), 1);
        assert_eq!(report.errors[0].node, Some(node));
        assert!(report.errors[0].url.is_none());
        assert!(
            report.errors[0].kind.to_string().contains("click boom"),
            "{:?}",
            report.errors[0]
        );
    }

    fn page_for_document(html: &str) -> (PageVm, Rc<RefCell<Document>>) {
        page_for_document_with_styles(html, &[])
    }

    fn page_for_document_with_styles(
        html: &str,
        author: &[Stylesheet],
    ) -> (PageVm, Rc<RefCell<Document>>) {
        let doc = Rc::new(RefCell::new(parse(html)));
        let base = base();
        let page = PageVm::new_for_document(doc.clone(), &base, "TestBrowser/9.9", author)
            .expect("web API install");
        // The test plays the tab's role: it owns the sole strong
        // document handle while the registry keeps only a `Weak`.
        (page, doc)
    }

    #[test]
    fn new_for_document_installs_the_web_api_surface() {
        let (mut page, _doc) = page_for_document(
            "<!DOCTYPE html><html><head><title>hi</title></head>\
             <body><p id=\"greet\">hello</p></body></html>",
        );
        assert!(page.web_api_context().is_some());
        // Bare realms have no context.
        assert!(PageVm::new().web_api_context().is_none());

        // window === globalThis, location is a snapshot of the base URL,
        // navigator carries the user agent.
        assert!(js_is_true(&mut page, "window === globalThis"));
        assert!(js_is_true(
            &mut page,
            "location.href === 'http://example.com/page.html'"
        ));
        assert!(js_is_true(
            &mut page,
            "navigator.userAgent === 'TestBrowser/9.9'"
        ));
        // document: live wrappers over the tab's tree.
        assert!(js_is_true(
            &mut page,
            "document.getElementById('greet').textContent === 'hello'"
        ));
        assert!(js_is_true(&mut page, "document.title === 'hi'"));
        // URL + encoding globals.
        assert!(js_is_true(
            &mut page,
            "new URL('/x', 'http://example.com/').href === 'http://example.com/x'"
        ));
        // Storage.
        assert!(js_is_true(
            &mut page,
            "localStorage.setItem('k', 'v'), localStorage.getItem('k') === 'v'"
        ));
        assert!(js_is_true(
            &mut page,
            "sessionStorage.setItem('s', '1'), sessionStorage.length === 1"
        ));
    }

    #[test]
    fn new_for_document_keeps_the_vm_timer_intrinsics() {
        // The production install deliberately does not shadow the VM's
        // timer globals: page setTimeout keeps firing through `pump`.
        let (mut page, _doc) = page_for_document("<html><body></body></html>");
        page.eval("setTimeout(() => { timerFired = true; }, 0);")
            .expect("eval");
        page.pump();
        assert_eq!(page.global("timerFired"), Some(Value::Bool(true)));
    }

    #[test]
    fn new_for_document_dispatches_events_to_listeners() {
        // addEventListener/dispatchEvent re-enter the running VM through
        // the bindings' invoker (inside the dispatchEvent native).
        let (mut page, _doc) =
            page_for_document("<html><body><button id=\"b\">x</button></body></html>");
        page.eval(
            "seen = []; \
             document.getElementById('b') \
               .addEventListener('click', () => seen.push('clicked'));",
        )
        .expect("eval");
        page.eval("document.getElementById('b').dispatchEvent({type: 'click'});")
            .expect("eval");
        assert!(js_is_true(&mut page, "seen.join(',') === 'clicked'"));
    }

    #[test]
    fn new_for_document_mutations_are_visible_to_the_embedder() {
        // The bindings share the tab's document by Rc: JS DOM mutations
        // land on the same tree the embedder holds.
        let doc = Rc::new(RefCell::new(parse(
            "<html><body><div id=\"a\">x</div></body></html>",
        )));
        let base = base();
        let mut page =
            PageVm::new_for_document(doc.clone(), &base, "TestBrowser/9.9", &[]).expect("install");
        page.eval("document.getElementById('a').setAttribute('data-n', '42');")
            .expect("eval");
        let doc_ref = doc.borrow();
        let node = doc_ref.get_element_by_id("a").expect("element");
        assert_eq!(
            doc_ref.get_attribute(node, "data-n").map(str::to_owned),
            Some("42".to_string())
        );
    }

    /// Canned HTTP server for the fetch test: serves one script, 404s the
    /// rest, and records the `User-Agent` it saw.
    fn spawn_script_server() -> (String, std::sync::Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("test server addr");
        let seen = std::sync::Arc::new(Mutex::new(Vec::<String>::new()));
        let seen_srv = std::sync::Arc::clone(&seen);
        std::thread::spawn(move || {
            for stream in listener.incoming().take(4) {
                let mut stream = match stream {
                    Ok(s) => s,
                    Err(_) => break,
                };
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                loop {
                    match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            buf.extend_from_slice(&chunk[..n]);
                            if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                                break;
                            }
                        }
                    }
                }
                let req = String::from_utf8_lossy(&buf).into_owned();
                let ua = req
                    .lines()
                    .find(|l| l.to_ascii_lowercase().starts_with("user-agent:"))
                    .map(|l| l["user-agent:".len()..].trim().to_owned())
                    .unwrap_or_default();
                seen_srv.lock().unwrap().push(ua);
                let path = req
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("/");
                let path = path
                    .rsplit('/')
                    .next()
                    .map(|l| format!("/{l}"))
                    .unwrap_or_default();
                let (status, body): (u16, &[u8]) = match path.as_str() {
                    "/app.js" => (200, b"var fromServer = 42;"),
                    _ => (404, b"nope"),
                };
                let head = format!(
                    "HTTP/1.1 {status} OK\r\nContent-Type: text/javascript\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(body);
            }
        });
        (format!("http://{addr}"), seen)
    }

    #[test]
    fn fetch_script_source_sends_user_agent_and_reads_body() {
        let (base, seen) = spawn_script_server();
        let jar = Mutex::new(CookieJar::new());
        let nav = NavigationContext {
            top_host: Some("127.0.0.1".to_string()),
            top_secure: Some(false),
            top_level_safe_navigation: false,
        };
        let url = bowser_url::parse(&format!("{base}/app.js"), None).expect("test url");

        let source = fetch_script_source(&url, &jar, "TestBrowser/9.9", &nav).expect("fetch");
        assert_eq!(source, "var fromServer = 42;");
        assert_eq!(seen.lock().unwrap().as_slice(), ["TestBrowser/9.9"]);

        // Non-2xx is an error, not a body.
        let missing = bowser_url::parse(&format!("{base}/missing.js"), None).expect("test url");
        let err = fetch_script_source(&missing, &jar, "TestBrowser/9.9", &nav).unwrap_err();
        assert!(err.contains("404"), "unexpected error: {err}");
    }

    // -- PageEventDispatch (production JsDispatch) ----------------------

    /// A tab with a live document and page realm, for dispatch tests.
    /// The button carries `id="t"`; page JS reaches it via
    /// `getElementById` (element collections do not support indexing).
    fn dispatch_tab(html: &str) -> (crate::tab::Tab, NodeId) {
        use crate::tab::Tab;
        use bowser_layout::Size;

        let doc = Rc::new(RefCell::new(parse(html)));
        let base = base();
        let page =
            PageVm::new_for_document(doc.clone(), &base, "TestBrowser/9.9", &[]).expect("install");
        let target = doc.borrow().get_element_by_id("t").expect("target button");
        let mut tab = Tab::new(Size { w: 800.0, h: 600.0 });
        tab.install_document(doc, page);
        (tab, target)
    }

    #[test]
    fn page_event_dispatch_fires_click_listeners() {
        use crate::window::JsDispatch;

        let (mut tab, button) =
            dispatch_tab("<html><body><button id=\"t\">go</button></body></html>");
        tab.page_vm_mut()
            .expect("page vm")
            .eval("clicked = 0; document.getElementById('t').addEventListener('click', function() { clicked = 41; });")
            .expect("register listener");

        let mut dispatch = PageEventDispatch::new();
        let prevented = dispatch.dispatch_click(&mut tab, button);

        assert!(!prevented, "no handler prevented the default action");
        let clicked = tab
            .page_vm_mut()
            .expect("page vm")
            .eval("clicked")
            .expect("read clicked");
        assert!(
            matches!(clicked, Value::Int(41)),
            "addEventListener click listener ran, got: {clicked:?}"
        );
    }

    #[test]
    fn page_event_dispatch_runs_inline_handlers() {
        use crate::window::JsDispatch;

        let (mut tab, button) = dispatch_tab(
            "<html><body><button id=\"t\" onclick=\"inlineRan = 7;\">go</button></body></html>",
        );

        let mut dispatch = PageEventDispatch::new();
        let prevented = dispatch.dispatch_click(&mut tab, button);

        assert!(!prevented, "plain inline handler does not prevent default");
        let ran = tab
            .page_vm_mut()
            .expect("page vm")
            .eval("inlineRan")
            .expect("read inlineRan");
        assert!(
            matches!(ran, Value::Int(7)),
            "inline onclick source ran, got: {ran:?}"
        );
    }

    #[test]
    fn page_event_dispatch_inline_return_false_prevents_default() {
        use crate::window::JsDispatch;

        let (mut tab, button) = dispatch_tab(
            "<html><body><button id=\"t\" onclick=\"return false;\">go</button></body></html>",
        );

        let mut dispatch = PageEventDispatch::new();
        let prevented = dispatch.dispatch_click(&mut tab, button);

        assert!(
            prevented,
            "onclick=\"return false\" must report the default action prevented"
        );
    }

    #[test]
    fn page_event_dispatch_inline_handler_sees_event_key() {
        // `send_keys` makes inline keyboard handlers a realistic path:
        // the inline source must see the dispatched event object as
        // `event` (finding: `onkeydown="if(event.key==='Enter')…"` threw
        // ReferenceError because `event` was unbound).
        let (mut tab, target) = dispatch_tab(
            "<html><body><input id=\"t\" onkeydown=\"seenDownKey = event.key;\" onkeyup=\"seenUpKey = event.key;\"></body></html>",
        );

        let mut dispatch = PageEventDispatch::new();
        let prevented = dispatch.dispatch_key_at(&mut tab, target, "Enter");

        assert!(!prevented, "plain inline handler does not prevent default");
        let vm = tab.page_vm_mut().expect("page vm");
        let down = vm.eval("seenDownKey").expect("read seenDownKey");
        assert!(
            matches!(down, Value::Str(ref s) if s.as_wtf8().to_utf8_lossy() == "Enter"),
            "inline onkeydown saw event.key, got: {down:?}"
        );
        let up = vm.eval("seenUpKey").expect("read seenUpKey");
        assert!(
            matches!(up, Value::Str(ref s) if s.as_wtf8().to_utf8_lossy() == "Enter"),
            "inline onkeyup saw event.key, got: {up:?}"
        );
    }

    #[test]
    fn page_event_dispatch_inline_click_handler_sees_event() {
        let (mut tab, button) = dispatch_tab(
            "<html><body><button id=\"t\" onclick=\"seenClickType = event.type;\">go</button></body></html>",
        );

        let mut dispatch = PageEventDispatch::new();
        let prevented = {
            use crate::window::JsDispatch;
            dispatch.dispatch_click(&mut tab, button)
        };

        assert!(!prevented, "plain inline handler does not prevent default");
        let seen = tab
            .page_vm_mut()
            .expect("page vm")
            .eval("seenClickType")
            .expect("read seenClickType");
        assert!(
            matches!(seen, Value::Str(ref s) if s.as_wtf8().to_utf8_lossy() == "click"),
            "inline onclick saw event.type, got: {seen:?}"
        );
    }

    #[test]
    fn page_event_dispatch_canceled_event_prevents_default() {
        use crate::window::JsDispatch;

        let (mut tab, button) =
            dispatch_tab("<html><body><button id=\"t\">go</button></body></html>");
        tab.page_vm_mut()
            .expect("page vm")
            .eval("document.getElementById('t').addEventListener('click', function(e) { e.preventDefault(); });")
            .expect("register listener");

        let mut dispatch = PageEventDispatch::new();
        let prevented = dispatch.dispatch_click(&mut tab, button);

        assert!(
            prevented,
            "preventDefault() in a listener must report the default action prevented"
        );
    }

    #[test]
    fn page_event_dispatch_wheel_delivers_deltas_and_prevented() {
        use crate::window::JsDispatch;

        let (mut tab, button) =
            dispatch_tab("<html><body><button id=\"t\">go</button></body></html>");
        tab.page_vm_mut()
            .expect("page vm")
            .eval("gotDx = 0; gotDy = 0; gotMode = 0; document.getElementById('t').addEventListener('wheel', function(e) { gotDx = e.deltaX; gotDy = e.deltaY; gotMode = e.deltaMode; e.preventDefault(); });")
            .expect("register listener");

        let mut dispatch = PageEventDispatch::new();
        let prevented = dispatch.dispatch_wheel(&mut tab, button, 10.0, -20.5, 0);

        assert!(
            prevented,
            "preventDefault() in a wheel listener must report the default action prevented"
        );
        let vm = tab.page_vm_mut().expect("page vm");
        // NOTE: read the values in Rust rather than comparing in JS: the
        // engine parses every integral literal as `Int`, and its `===`
        // is representation-strict, so `Float(10.0) === 10` is false.
        let dx = vm.eval("gotDx").expect("read gotDx");
        let dy = vm.eval("gotDy").expect("read gotDy");
        let mode = vm.eval("gotMode").expect("read gotMode");
        assert!(
            matches!(dx, Value::Float(10.0)),
            "deltaX delivered, got: {dx:?}"
        );
        assert!(
            matches!(dy, Value::Float(-20.5)),
            "deltaY delivered, got: {dy:?}"
        );
        assert!(
            matches!(mode, Value::Float(0.0)),
            "deltaMode delivered, got: {mode:?}"
        );
    }

    #[test]
    fn page_event_dispatch_wheel_unprevented_not_canceled() {
        use crate::window::JsDispatch;

        let (mut tab, button) =
            dispatch_tab("<html><body><button id=\"t\">go</button></body></html>");
        tab.page_vm_mut()
            .expect("page vm")
            .eval("ran = false; document.getElementById('t').addEventListener('wheel', function() { ran = true; });")
            .expect("register listener");

        let mut dispatch = PageEventDispatch::new();
        let prevented = dispatch.dispatch_wheel(&mut tab, button, 0.0, 5.0, 1);

        assert!(!prevented, "no preventDefault: the scroll must proceed");
        let vm = tab.page_vm_mut().expect("page vm");
        assert!(js_is_true(vm, "ran"), "wheel listener still ran");
    }

    #[test]
    fn page_event_dispatch_wheel_fail_closed_without_document() {
        use crate::window::JsDispatch;
        use bowser_layout::Size;

        let doc = parse("<html><body><button id=\"t\">go</button></body></html>");
        let target = doc.get_element_by_id("t").expect("target button");
        let mut tab = crate::tab::Tab::new(Size { w: 800.0, h: 600.0 });
        let mut dispatch = PageEventDispatch::new();

        assert!(
            !dispatch.dispatch_wheel(&mut tab, target, 1.0, 2.0, 0),
            "no document installed: nothing can be prevented"
        );
    }

    #[test]
    fn page_event_dispatch_touch_delivers_lists_and_prevented() {
        let (mut tab, button) =
            dispatch_tab("<html><body><button id=\"t\">go</button></body></html>");
        tab.page_vm_mut()
            .expect("page vm")
            .eval("gotType = ''; gotId = 0; gotX = 0; gotY = 0; gotLen = 0; gotTargetLen = 0; document.getElementById('t').addEventListener('touchstart', function(e) { gotType = e.type; gotId = e.changedTouches[0].identifier; gotX = e.changedTouches[0].clientX; gotY = e.changedTouches[0].clientY; gotLen = e.touches.length; gotTargetLen = e.targetTouches.length; e.preventDefault(); });")
            .expect("register listener");

        let mut dispatch = PageEventDispatch::new();
        let touches = [Touch::new(
            7,
            Some(button),
            10.0,
            20.0,
            1.0,
            2.0,
            10.0,
            20.0,
        )];
        let prevented = dispatch.dispatch_touch_event(&mut tab, TouchPhase::Start, &touches);

        assert!(
            prevented,
            "preventDefault() in a touchstart listener must report the default pan canceled"
        );
        let vm = tab.page_vm_mut().expect("page vm");
        let ttype = vm.eval("gotType").expect("read gotType");
        assert!(
            matches!(ttype, Value::Str(ref s) if s.as_wtf8().to_utf8_lossy() == "touchstart"),
            "event type delivered, got: {ttype:?}"
        );
        // NOTE: read the values in Rust rather than comparing in JS: the
        // engine parses every integral literal as `Int`, and its `===`
        // is representation-strict, so `Float(10.0) === 10` is false.
        let id = vm.eval("gotId").expect("read gotId");
        let x = vm.eval("gotX").expect("read gotX");
        let y = vm.eval("gotY").expect("read gotY");
        let len = vm.eval("gotLen").expect("read gotLen");
        let target_len = vm.eval("gotTargetLen").expect("read gotTargetLen");
        assert!(
            matches!(id, Value::Int(7)),
            "identifier delivered, got: {id:?}"
        );
        assert!(
            matches!(x, Value::Float(10.0)),
            "clientX delivered, got: {x:?}"
        );
        assert!(
            matches!(y, Value::Float(20.0)),
            "clientY delivered, got: {y:?}"
        );
        assert!(matches!(len, Value::Int(1)), "touches length, got: {len:?}");
        assert!(
            matches!(target_len, Value::Int(1)),
            "targetTouches length, got: {target_len:?}"
        );
    }

    #[test]
    fn page_event_dispatch_touch_unprevented_not_canceled() {
        let (mut tab, button) =
            dispatch_tab("<html><body><button id=\"t\">go</button></body></html>");
        tab.page_vm_mut()
            .expect("page vm")
            .eval("ran = false; document.getElementById('t').addEventListener('touchend', function() { ran = true; });")
            .expect("register listener");

        let mut dispatch = PageEventDispatch::new();
        let touches = [Touch::new(
            7,
            Some(button),
            10.0,
            20.0,
            1.0,
            2.0,
            10.0,
            20.0,
        )];
        let prevented = dispatch.dispatch_touch_event(&mut tab, TouchPhase::End, &touches);

        assert!(
            !prevented,
            "no preventDefault: the default pan must proceed"
        );
        let vm = tab.page_vm_mut().expect("page vm");
        assert!(js_is_true(vm, "ran"), "touchend listener still ran");
    }

    #[test]
    fn page_event_dispatch_touch_fail_closed_without_target() {
        let (mut tab, _button) =
            dispatch_tab("<html><body><button id=\"t\">go</button></body></html>");
        let mut dispatch = PageEventDispatch::new();
        // No changed touch carries a target: fail closed, no script runs.
        let touches = [Touch::new(7, None, 10.0, 20.0, 1.0, 2.0, 10.0, 20.0)];
        assert!(
            !dispatch.dispatch_touch_event(&mut tab, TouchPhase::Move, &touches),
            "no touch target: nothing can be prevented"
        );
        assert!(
            !dispatch.dispatch_touch_event(&mut tab, TouchPhase::Move, &[]),
            "empty touch list: nothing can be prevented"
        );
    }

    // -- Hit testing (M4) and in-view centre (M3) --------------------

    fn geo_box(
        rect: (f32, f32, f32, f32),
        node: Option<NodeId>,
        z_index: Option<i32>,
        children: Vec<bowser_layout::LayoutBox>,
    ) -> bowser_layout::LayoutBox {
        bowser_layout::LayoutBox {
            rect: bowser_layout::Rect {
                x: rect.0,
                y: rect.1,
                w: rect.2,
                h: rect.3,
            },
            children,
            box_type: bowser_layout::BoxType::Block,
            style: None,
            text: None,
            node,
            fragments: Vec::new(),
            z_index,
            collapsed_borders: None,
            phantom_w: false,
            phantom_h: false,
        }
    }

    /// A tab whose layout tree is `root_children` under an anonymous
    /// 800x600 root, over a document parsed from `html`.
    fn geo_tab(
        html: &str,
        root_children: Vec<bowser_layout::LayoutBox>,
    ) -> (crate::tab::Tab, Document) {
        let doc = parse(html);
        let mut tab = crate::tab::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
        tab.layout_root = Some(Rc::new(geo_box(
            (0.0, 0.0, 800.0, 600.0),
            None,
            None,
            root_children,
        )));
        (tab, doc)
    }

    #[test]
    fn hit_test_higher_z_index_earlier_sibling_wins() {
        // Two overlapping siblings: the earlier-in-DOM one carries
        // the higher z-index, so it paints (and hits) on top even
        // though document order alone would pick the later one.
        let (tab, doc) = geo_tab(
            "<html><body><div id=\"a\">A</div><div id=\"b\">B</div></body></html>",
            vec![],
        );
        let a = doc.get_element_by_id("a").expect("a");
        let b = doc.get_element_by_id("b").expect("b");
        let mut tab = tab;
        tab.layout_root = Some(Rc::new(geo_box(
            (0.0, 0.0, 800.0, 600.0),
            None,
            None,
            vec![
                geo_box((0.0, 0.0, 100.0, 100.0), Some(a), Some(5), vec![]),
                geo_box((0.0, 0.0, 100.0, 100.0), Some(b), None, vec![]),
            ],
        )));
        let dispatch = PageEventDispatch::new();
        assert_eq!(dispatch.hit_test_at_point(&tab, 50.0, 50.0), Some(a));
    }

    #[test]
    fn hit_test_negative_z_index_loses_to_in_flow_sibling() {
        let (tab, doc) = geo_tab(
            "<html><body><div id=\"a\">A</div><div id=\"b\">B</div></body></html>",
            vec![],
        );
        let a = doc.get_element_by_id("a").expect("a");
        let b = doc.get_element_by_id("b").expect("b");
        let mut tab = tab;
        tab.layout_root = Some(Rc::new(geo_box(
            (0.0, 0.0, 800.0, 600.0),
            None,
            None,
            vec![
                geo_box((0.0, 0.0, 100.0, 100.0), Some(a), Some(-1), vec![]),
                geo_box((0.0, 0.0, 100.0, 100.0), Some(b), None, vec![]),
            ],
        )));
        let dispatch = PageEventDispatch::new();
        // Negative z-index paints below in-flow content: #b wins.
        assert_eq!(dispatch.hit_test_at_point(&tab, 50.0, 50.0), Some(b));
        // Among positive z-indexes the higher value wins regardless
        // of document order.
        tab.layout_root = Some(Rc::new(geo_box(
            (0.0, 0.0, 800.0, 600.0),
            None,
            None,
            vec![
                geo_box((0.0, 0.0, 100.0, 100.0), Some(a), Some(1), vec![]),
                geo_box((0.0, 0.0, 100.0, 100.0), Some(b), Some(3), vec![]),
            ],
        )));
        assert_eq!(dispatch.hit_test_at_point(&tab, 50.0, 50.0), Some(b));
    }

    #[test]
    fn in_view_center_pure_intersection_math() {
        let viewport = bowser_layout::Rect {
            x: 0.0,
            y: 0.0,
            w: 800.0,
            h: 600.0,
        };
        let rect = bowser_layout::Rect {
            x: 700.0,
            y: 0.0,
            w: 200.0,
            h: 100.0,
        };
        // Half the element hangs past the viewport's right edge:
        // the visible part is 700..800, centre x = 750.
        assert_eq!(in_view_center(&rect, &viewport, &[]), Some((750.0, 50.0)));
        // Fully outside the viewport: no in-view point.
        let outside = bowser_layout::Rect { x: 900.0, ..rect };
        assert_eq!(in_view_center(&outside, &viewport, &[]), None);
        // A scroller clip narrower than the viewport clips further.
        let clip = bowser_layout::Rect {
            x: 0.0,
            y: 0.0,
            w: 720.0,
            h: 600.0,
        };
        assert_eq!(
            in_view_center(&rect, &viewport, &[clip]),
            Some((710.0, 50.0))
        );
    }

    #[test]
    fn element_client_center_uses_visible_half_when_viewport_clipped() {
        let (tab, doc) = geo_tab("<html><body><div id=\"a\">A</div></body></html>", vec![]);
        let a = doc.get_element_by_id("a").expect("a");
        let mut tab = tab;
        // Element spans x 700..900 in an 800-wide viewport.
        tab.layout_root = Some(Rc::new(geo_box(
            (0.0, 0.0, 800.0, 600.0),
            None,
            None,
            vec![geo_box((700.0, 100.0, 200.0, 100.0), Some(a), None, vec![])],
        )));
        let dispatch = PageEventDispatch::new();
        assert_eq!(
            dispatch.element_client_center(&tab, a),
            Some((750.0, 150.0)),
            "centre of the visible half, not the border-box centre (800, 150)"
        );
    }

    #[test]
    fn element_client_center_errors_when_scroller_clips_element_away() {
        // A real layout + scroll tree: #inner sits below its
        // scroller's clip (scrolled content outside the port), so
        // its in-view centre does not exist.
        let (doc, root) = {
            let (doc, _) = bowser_dom::Document::parse_html(
                b"<html><body><div id=\"sc\"><div id=\"inner\">x</div></div></body></html>",
            );
            let sheet = bowser_css::parse_stylesheet(
                "#sc { overflow: hidden; width: 100px; height: 100px; } \
                 #inner { width: 50px; height: 50px; margin-top: 300px; }",
            );
            let root = bowser_layout::layout(
                &doc,
                &[&sheet],
                bowser_layout::Size { w: 800.0, h: 600.0 },
                &bowser_layout::FixedFontMetrics,
            );
            (doc, root)
        };
        let inner = doc.get_element_by_id("inner").expect("inner");
        let mut tab = crate::tab::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
        tab.scroll_tree = crate::scroll::ScrollTree::build(
            &doc,
            &root,
            bowser_layout::Size { w: 800.0, h: 600.0 },
            tab.scroll(),
            tab.element_scroll(),
        );
        tab.layout_root = Some(Rc::new(root));
        let dispatch = PageEventDispatch::new();
        // The element has a border box (margin pushes it to y=300)
        // but the scroller clips y to 0..100: no in-view centre.
        assert!(dispatch.element_client_rect(&tab, inner).is_some());
        assert_eq!(dispatch.element_client_center(&tab, inner), None);
    }

    #[test]
    fn element_client_center_accounts_for_uniform_scale_ancestor() {
        // Real layout: a scale(2) wrapper doubles its child's
        // geometry about the wrapper's centre (the default
        // transform-origin). Wrapper is 100x100 at (0,0), child
        // 50x50 at its top-left: the child's centre (25,25) maps to
        // centre + 2 * ((25,25) - (50,50)) = (0,0)... the child's
        // visible box is (-50,-50)-(50,50), clipped by the viewport
        // to (0,0)-(50,50), whose centre is (25,25).
        let (doc, root) = {
            let (doc, _) = bowser_dom::Document::parse_html(
                b"<html><body><div id=\"w\"><div id=\"inner\">x</div></div></body></html>",
            );
            let sheet = bowser_css::parse_stylesheet(
                "#w { transform: scale(2); width: 100px; height: 100px; } \
                 #inner { width: 50px; height: 50px; }",
            );
            let root = bowser_layout::layout(
                &doc,
                &[&sheet],
                bowser_layout::Size { w: 800.0, h: 600.0 },
                &bowser_layout::FixedFontMetrics,
            );
            (doc, root)
        };
        let inner = doc.get_element_by_id("inner").expect("inner");
        let mut tab = crate::tab::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
        tab.layout_root = Some(Rc::new(root));
        let dispatch = PageEventDispatch::new();
        let rect = dispatch
            .element_client_rect(&tab, inner)
            .expect("inner rect");
        assert_eq!(
            (rect.x, rect.y, rect.w, rect.h),
            (-50.0, -50.0, 100.0, 100.0),
            "scaled rect about the wrapper centre"
        );
        assert_eq!(
            dispatch.element_client_center(&tab, inner),
            Some((25.0, 25.0)),
            "centre of the viewport-visible part of the scaled box"
        );
    }

    #[test]
    fn get_computed_style_without_author_sheets_uses_ua_only() {
        // No author sheets: only the UA stylesheet applies (a `div` is
        // `block` per the UA sheet).
        let (mut page, _doc) =
            page_for_document("<html><body><div id=\"d\">hi</div></body></html>");
        assert!(js_is_true(
            &mut page,
            "window.getComputedStyle(document.getElementById('d')).getPropertyValue('display') === 'block'"
        ));
    }

    #[test]
    fn get_computed_style_sees_author_stylesheets() {
        // Author sheets seeded via `new_for_document` reach
        // `getComputedStyle`: the `#d` rule's red wins over the UA
        // stylesheet, and serializes as rgb(255, 0, 0).
        let sheet = bowser_css::parse_stylesheet("#d { color: red; }");
        let (mut page, _doc) = page_for_document_with_styles(
            "<html><body><div id=\"d\">hi</div></body></html>",
            &[sheet],
        );
        assert!(js_is_true(
            &mut page,
            "window.getComputedStyle(document.getElementById('d')).getPropertyValue('color') === 'rgb(255, 0, 0)'"
        ));
    }

    /// The wheel default-action path drains queued snap events BEFORE
    /// the movement's `scroll` dispatches (the scroll funnel order,
    /// CSS Scroll Snap 2 §5): a wheel scroll that changes the snap
    /// target fires `scrollsnapchanging` then `scroll` — before this
    /// fix the `scrollsnapchanging` sat in the queue until the next
    /// pump iteration's drain, firing after `scroll`.
    #[test]
    fn wheel_drains_snap_events_before_scroll() {
        // 400px snap areas in a 400px-tall mandatory snapport: a 250px
        // wheel lands between snap points and the backstop retargets
        // to the nearest one (#b at 400).
        let sheet = bowser_css::parse_stylesheet(
            "#sc { overflow-y: auto; width: 400px; height: 400px; scroll-snap-type: y mandatory; } \
             #a { height: 400px; scroll-snap-align: start; } \
             #b { height: 400px; scroll-snap-align: start; } \
             #c { height: 400px; scroll-snap-align: start; }",
        );
        let (page, doc) = page_for_document_with_styles(
            "<html><body><div id=\"sc\"><div id=\"a\">x</div><div id=\"b\">y</div><div id=\"c\">z</div></div></body></html>",
            std::slice::from_ref(&sheet),
        );
        let root = bowser_layout::layout(
            &doc.borrow(),
            &[&sheet],
            bowser_layout::Size { w: 800.0, h: 600.0 },
            &bowser_layout::FixedFontMetrics,
        );
        let mut tab = crate::tab::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
        tab.author.push(sheet);
        tab.install_document(doc, page);
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        // The initial layout-snap deliveries are not this test's
        // subject: drop them so only the wheel's own drain is observed.
        let _ = tab.scroll_tree.take_snap_events();
        tab.page_vm_mut()
            .expect("page vm")
            .eval(
                "window.__events = []; var sc = document.getElementById('sc'); \
                 sc.addEventListener('scrollsnapchanging', function() { window.__events.push('scrollsnapchanging'); }); \
                 sc.addEventListener('scroll', function() { window.__events.push('scroll'); });",
            )
            .expect("register listeners");
        let mut dispatch = PageEventDispatch::new();
        let prevented = dispatch.dispatch_wheel_at_point(&mut tab, 200.0, 200.0, 0.0, 250.0);
        assert!(!prevented, "no wheel listener prevents the default scroll");
        assert_eq!(
            tab.scroll_tree
                .container(
                    tab.scroll_tree
                        .index_of(
                            tab.document_ref()
                                .expect("document")
                                .get_element_by_id("sc")
                                .expect("sc")
                        )
                        .expect("sc container")
                )
                .state
                .offset(),
            (0.0, 250.0),
            "the wheel default action scrolled the snap container"
        );
        match tab
            .page_vm_mut()
            .expect("page vm")
            .eval("window.__events.join(',')")
        {
            Ok(bowser_js::Value::Str(s)) => assert_eq!(
                s.as_wtf8().to_utf8_lossy().as_ref(),
                "scrollsnapchanging,scroll",
                "snap events fire before the movement's scroll"
            ),
            other => panic!("unexpected events value: {other:?}"),
        }
    }

    /// The synthetic keydown default action threads the WebDriver
    /// Actions modifier state into the scroll: Shift+Space pages up,
    /// like the windowed key handler's `apply_key_scroll_with_shift` —
    /// before this fix `dispatch_key_down_at` dropped `modifiers.shift`
    /// and Shift+Space paged down on the synthetic path while paging up
    /// on the windowed path.
    #[test]
    fn key_down_at_threads_shift_into_space_scroll() {
        let sheet = bowser_css::parse_stylesheet(
            "#sc { overflow-y: auto; height: 100px; } #tall { height: 500px; }",
        );
        let (page, doc) = page_for_document_with_styles(
            "<html><body><div id=\"sc\"><div id=\"tall\">x</div></div></body></html>",
            std::slice::from_ref(&sheet),
        );
        let root = bowser_layout::layout(
            &doc.borrow(),
            &[&sheet],
            bowser_layout::Size { w: 800.0, h: 600.0 },
            &bowser_layout::FixedFontMetrics,
        );
        let sc = doc.borrow().get_element_by_id("sc").expect("sc");
        let mut tab = crate::tab::Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 });
        tab.author.push(sheet);
        tab.install_document(doc, page);
        tab.layout_root = Some(Rc::new(root));
        tab.rebuild_scroll_after_layout();
        let isc = tab.scroll_tree.index_of(sc).expect("sc container");

        let mut dispatch = PageEventDispatch::new();
        // Plain Space pages down one page (the 100px scrollport).
        let down = dispatch.dispatch_key_down_at(&mut tab, sc, " ", KeyModifiers::default());
        assert!(!down, "no keydown listener prevents the default scroll");
        assert_eq!(
            tab.scroll_tree.container(isc).state.offset(),
            (0.0, 100.0),
            "Space pages down"
        );
        // Shift+Space pages back up.
        let shift = KeyModifiers {
            shift: true,
            ..Default::default()
        };
        let down = dispatch.dispatch_key_down_at(&mut tab, sc, " ", shift);
        assert!(!down, "no keydown listener prevents the default scroll");
        assert_eq!(
            tab.scroll_tree.container(isc).state.offset(),
            (0.0, 0.0),
            "Shift+Space pages up"
        );
    }
}
