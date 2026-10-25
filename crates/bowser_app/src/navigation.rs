//! Navigation: loading URLs into a tab, session history, fragment scrolling.
//!
//! The loading itself is delegated to [`crate::loader::load_document_with_user_agent`],
//! which runs on a worker thread (never the UI thread). This module owns
//! the tab-level concerns around a load: URL resolution, same-document
//! (fragment) fast paths, session-history bookkeeping, title extraction,
//! and render invalidation.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::Duration;

use bowser_dom::{Document, NodeId};
use bowser_js::ScrollBehavior;
use bowser_layout::IntrinsicDims;
use bowser_net::cookies::CookieJar;
use bowser_net::fetch::FetchError;
use bowser_paint::fonts::{FontCache, FontStyle, SharedFontCache};
use bowser_url::{ParseError, Url};

use crate::loader::{self, LoadError, LoadedDocument};
use crate::script::PageVm;
use crate::script_executor::{ParserExecutor, filter_handled_scripts};
use crate::scroll::RevealAlign;
use crate::tab::{Tab, mark_all_dirty};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Every way a navigation can fail.
#[derive(Debug)]
pub enum NavError {
    /// The document fetch itself failed (DNS, connect, TLS, redirects...).
    Fetch(FetchError),
    /// The URL string failed to parse (resolved against the tab's current
    /// URL as base).
    Url(ParseError),
    /// The document fetch completed with a non-2xx status (3xx chains are
    /// already followed by fetch, so this is 4xx/5xx in practice).
    HttpStatus(u16),
    /// The document is not HTML: carries the offending media type.
    UnsupportedContentType(String),
    /// The loader worker thread died (or its report channel broke) before
    /// delivering a result.
    Cancelled,
    /// The overall navigation deadline fired before the load finished.
    /// The loader worker is detached and abandoned (Rust cannot kill a
    /// thread stuck in blocking I/O); any cookie updates it was carrying
    /// are discarded with it. Carries the deadline that fired.
    Timeout(Duration),
}

impl fmt::Display for NavError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fetch(e) => write!(f, "navigation fetch failed: {e}"),
            Self::Url(e) => write!(f, "navigation URL error: {e}"),
            Self::HttpStatus(s) => write!(f, "navigation got HTTP {s}"),
            Self::UnsupportedContentType(t) => {
                write!(f, "navigation to non-HTML document: {t}")
            }
            Self::Cancelled => f.write_str("navigation was cancelled"),
            Self::Timeout(d) => write!(f, "navigation timed out after {}s", d.as_secs()),
        }
    }
}

impl std::error::Error for NavError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Fetch(e) => Some(e),
            Self::Url(e) => Some(e),
            _ => None,
        }
    }
}

impl From<FetchError> for NavError {
    fn from(e: FetchError) -> Self {
        Self::Fetch(e)
    }
}

impl From<ParseError> for NavError {
    fn from(e: ParseError) -> Self {
        Self::Url(e)
    }
}

impl From<LoadError> for NavError {
    fn from(e: LoadError) -> Self {
        match e {
            LoadError::Fetch(fe) => Self::Fetch(fe),
            LoadError::HttpStatus(s) => Self::HttpStatus(s),
            LoadError::UnsupportedContentType(t) => Self::UnsupportedContentType(t),
        }
    }
}

// ---------------------------------------------------------------------------
// Core loading
// ---------------------------------------------------------------------------

/// Overall navigation deadline for [`load_url`].
///
/// The loader's net stack only has per-stage socket timeouts, so a
/// drip-feeding server (bytes arriving slower than the socket read
/// timeout, forever) would stall navigation indefinitely. This bounds
/// the whole load instead. Matches the headless `--timeout` default in
/// `main.rs`, which applies the same deadline to its direct fetches.
pub(crate) const NAVIGATION_TIMEOUT: Duration = Duration::from_secs(30);

/// Swap the profile's cookie jar out of its mutex, leaving a fresh jar
/// behind.
///
/// The loader holds the jar lock for the whole load, so the worker gets an
/// owned jar: the profile lock is only held for the swap, never across a
/// load, so an abandoned worker cannot block later navigations on it.
/// Shared by the blocking [`load_url`] path and the window worker's async
/// loads (`window::spawn_load`).
pub(crate) fn take_profile_jar(jar: &Mutex<CookieJar>) -> CookieJar {
    let mut guard = jar.lock().expect("navigation: cookie jar mutex poisoned");
    std::mem::replace(&mut *guard, CookieJar::new())
}

/// Put a worker's jar back into the profile mutex, with whatever cookies
/// the load set.
pub(crate) fn restore_profile_jar(jar: &Mutex<CookieJar>, owned: CookieJar) {
    *jar.lock().expect("navigation: cookie jar mutex poisoned") = owned;
}

/// RAII guard over a profile cookie jar taken out for a load.
///
/// [`take_profile_jar`] swaps the profile's jar out, leaving a fresh jar
/// behind; if the load never reports back (deadline timeout, dead worker,
/// or a panic before the restore), the profile would keep that empty jar
/// and lose every stored cookie. This guard owns the taken jar and puts
/// it back on [`Drop`], so the jar is restored on every path. The
/// finished-load path calls [`commit`](Self::commit) with the worker's
/// jar (carrying the load's new cookies) instead, which installs it and
/// disarms the drop restore.
///
/// The worker is handed a clone of the taken jar ([`worker_jar`](Self::worker_jar)):
/// the guard keeps the original, so the failure paths restore exactly the
/// pre-load jar and discard only the abandoned load's cookie updates.
pub(crate) struct ProfileJarGuard<'a> {
    jar: &'a Mutex<CookieJar>,
    taken: CookieJar,
    committed: bool,
}

impl<'a> ProfileJarGuard<'a> {
    /// Swap the profile's jar out of its mutex and hold it.
    pub(crate) fn take(jar: &'a Mutex<CookieJar>) -> Self {
        let taken = take_profile_jar(jar);
        Self {
            jar,
            taken,
            committed: false,
        }
    }

    /// A worker-owned copy of the taken jar, carrying the pre-load
    /// cookies into the load.
    pub(crate) fn worker_jar(&self) -> CookieJar {
        self.taken.clone()
    }

    /// Install the worker's jar (with the load's cookie updates) back
    /// into the profile mutex and disarm the drop restore.
    pub(crate) fn commit(&mut self, owned: CookieJar) {
        restore_profile_jar(self.jar, owned);
        self.committed = true;
    }
}

impl Drop for ProfileJarGuard<'_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        // Never panic in drop: a poisoned mutex while unwinding must not
        // abort the process, and if the lock is gone there is nothing to
        // save the jar into anyway.
        if let Ok(mut guard) = self.jar.lock() {
            *guard = std::mem::replace(&mut self.taken, CookieJar::new());
        }
    }
}

/// The outcome of running [`crate::loader::load_document_with_user_agent`] to completion
/// on a worker thread: the loader result (or the panic payload when the
/// loader panicked) plus the cookie jar reclaimed from the worker lock.
pub(crate) struct LoaderOutcome {
    pub result: std::thread::Result<Result<LoadedDocument, LoadError>>,
    pub jar: CookieJar,
}

/// Run the loader on the calling thread with an owned `jar`, catching
/// panics; the jar is reclaimed even if the loader poisoned its lock by
/// panicking while holding it. Both load paths (blocking and async) run
/// their worker body through here so a loader panic is handled one way.
pub(crate) fn run_loader(url: &Url, jar: CookieJar, user_agent: &str) -> LoaderOutcome {
    let worker_jar = Mutex::new(jar);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        loader::load_document_with_user_agent(url, &worker_jar, user_agent)
    }));
    let jar = worker_jar
        .into_inner()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    LoaderOutcome { result, jar }
}

// ---------------------------------------------------------------------------
// Page-realm native-stack guard sizing
// ---------------------------------------------------------------------------

/// Target native-stack budget for the page realm: the test262 harness
/// recipe — a 7 MiB guard on an 8 MiB thread, leaving
/// [`PAGE_REALM_STACK_GUARD_HEADROOM`] of headroom to unwind past the
/// trip point.
const PAGE_REALM_STACK_GUARD_TARGET: usize = 7 << 20;
/// Bytes of native stack reserved above the guard trip point so the
/// `RangeError`-class unwind itself (destructors, catch landing pads)
/// never touches the real stack end.
const PAGE_REALM_STACK_GUARD_HEADROOM: usize = 1 << 20;

/// `RLIMIT_STACK` soft limit in bytes — the main thread's stack size on
/// Linux — probed once per process. `None` when the probe is unavailable
/// (non-Linux, or `/proc` unreadable); the guard computation fails safe
/// on `None`.
static MAIN_THREAD_STACK_LIMIT: OnceLock<Option<usize>> = OnceLock::new();

/// Read the `RLIMIT_STACK` soft limit without libc (this crate is
/// std-only): the kernel publishes it in `/proc/self/limits` as the
/// `Max stack size` row, in bytes. Arch-independent, unlike a raw
/// `getrlimit` syscall whose number varies per architecture.
#[cfg(target_os = "linux")]
fn probe_main_thread_stack_limit() -> Option<usize> {
    let text = std::fs::read_to_string("/proc/self/limits").ok()?;
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let is_stack_row = fields.next() == Some("Max")
            && fields.next() == Some("stack")
            && fields.next() == Some("size");
        if !is_stack_row {
            continue;
        }
        return match fields.next()? {
            // Soft limit raised to infinity (up to the hard limit): the
            // main thread can grow without bound, so the target guard
            // always applies.
            "unlimited" => Some(usize::MAX),
            bytes => bytes.parse::<usize>().ok(),
        };
    }
    None
}

/// Non-Linux: no std-only probe for the main thread's stack size, so
/// there is nothing to clamp against — the guard computation uses the
/// conservative fallback instead.
#[cfg(not(target_os = "linux"))]
fn probe_main_thread_stack_limit() -> Option<usize> {
    None
}

/// Cached `RLIMIT_STACK` probe. Lazy — this module has no earlier hook,
/// and the limit is effectively fixed for the process's life: lowering
/// it later can only make the clamp more conservative, and raising it
/// lets the kernel grow the main stack mapping, which the fresh read
/// observes.
fn main_thread_stack_limit() -> Option<usize> {
    *MAIN_THREAD_STACK_LIMIT.get_or_init(probe_main_thread_stack_limit)
}

/// Native-stack budget to install on the page realm: the 7 MiB test262
/// target, clamped to `main_thread_stack − headroom` so the guard always
/// sits *above* the real stack end, where the engine's exhaustion check
/// trips it into a catchable `RangeError` instead of the process dying
/// on a real stack overflow (SIGSEGV/SIGABRT) the too-high guard never
/// tripped.
///
/// The clamp is keyed off the main thread's limit because that is the
/// only installing thread whose size the code does not choose: the
/// `--check`/`--headless` workers are spawned with 8 MiB stacks, so
/// `min(7 MiB, rlimit − 1 MiB)` is exact for them whenever
/// `rlimit ≥ 8 MiB`, and merely conservative (still safe — still below
/// their real 8 MiB end) when the operator shrank `ulimit -s` below
/// that. On the main thread the clamp is exact: with the default 8 MiB
/// `RLIMIT_STACK` the full 7 MiB applies and testharness.js-sized parses
/// run clean; with a shrunk limit the guard shrinks with it and deep
/// recursion fails closed with `RangeError` instead of aborting. (A
/// sub-headroom limit clamps the guard to zero: scripts then fail
/// closed on every call — still no abort.)
///
/// Fallback (probe unavailable): the engine's 1 MiB default — documented
/// safe on 2 MiB threads — so an unknown main-thread stack can never get
/// a guard above its real end. Deep script recursion may `RangeError`
/// there; that is clean and catchable, unlike a stack abort.
fn page_realm_native_stack_guard() -> usize {
    match main_thread_stack_limit() {
        Some(limit) => {
            PAGE_REALM_STACK_GUARD_TARGET.min(limit.saturating_sub(PAGE_REALM_STACK_GUARD_HEADROOM))
        }
        None => 1 << 20,
    }
}

/// Install a finished load into the tab: document, stylesheets, image
/// intrinsics, title; a fresh [`PageVm`] running the page's `<script>`
/// elements; caches dropped, scroll reset, everything dirty.
/// Single implementation shared by the blocking [`load_url`] path and the
/// window worker's async completion (`window::poll_pending`).
///
/// Parser-inserted scripts execute mid-parse, in document order, through
/// the page's own realm (WHATWG HTML §13.2.6): the body is re-parsed here
/// — on the installing thread, because the `Vm` is `!Send` — with a
/// [`ParserExecutor`](crate::script_executor::ParserExecutor) attached,
/// so `document.write` and mid-parse DOM reads see the tree under
/// construction. The production web API surface (`window`, `document`,
/// ...) is installed on the same realm afterwards, overwriting the
/// parse-phase `document` shim; the scripts the parser declined
/// (`async`/`defer`, modules, bad URLs, failed or over-cap fetches) then
/// run through the post-parse path. The bindings borrow the tab's shared
/// DOM tree (one tree, no copies): script mutations land in the document
/// the renderer reads, and record dirt for
/// [`Tab::sync_dirty_flags`](crate::tab::Tab::sync_dirty_flags).
/// `user_agent` seeds `navigator.userAgent` and should be the same UA the
/// load itself sent. If the web API install fails, the page keeps the
/// bare realm — the mid-parse scripts already ran, and the post-parse
/// scripts still get their turn.
pub(crate) fn install_loaded_document(
    tab: &mut Tab,
    loaded: LoadedDocument,
    user_agent: &str,
    fonts: &Arc<Mutex<FontCache>>,
    cookie_jar: &Arc<Mutex<CookieJar>>,
) {
    // @font-face install, before first layout and before the
    // scripted parse: the loader fetched these synchronously (no
    // async arrival in this slice), so installing them here — inside
    // the one install every load path funnels through — means the
    // first layout measures with the webfonts on BOTH paths. The old
    // split (initial loads via `install_final_document`, push
    // navigations via `complete_navigation`) dropped webfonts on the
    // push path, which called this function directly. The rule's
    // descriptors win over the file's internals (see
    // `install_font_owned_as`); installing twice (loader +
    // install) is harmless by (bytes, family, weight, style) dedupe.
    {
        let mut guard = fonts.lock().expect("navigation: font cache lock poisoned");
        for webfont in &loaded.webfonts {
            guard.install_font_owned_as(
                webfont.bytes.clone(),
                &webfont.family,
                webfont.weight,
                FontStyle::from_descriptor(&webfont.style),
            );
        }
    }
    let url = loaded.url.clone();
    // The loader's discovery DOM fed subresource collection, but the
    // scripted parse happens here, on the installing thread: the `Vm` is
    // `!Send` and bound to its creating thread, so the page realm — and
    // the parse that runs parser-inserted scripts through it — must be
    // created here, not in the loader. The base URL comes from the
    // discovery document (the same base the loader used when prefetching
    // `script_sources`), because the executor needs it before the
    // scripted parse exists.
    let base = loader::effective_base_url(&loaded.document, &url);
    // One JS realm per document, like a new `Window`: the old realm (and
    // its globals) is dropped with the old document. Parser-inserted
    // classic scripts run synchronously mid-parse, in document order, on
    // the installing thread — the windowed path's event thread. A
    // non-terminating script does not stall it forever: every script
    // evaluates under the engine's per-eval fuel budget, and a runaway
    // lands in the [`ScriptReport`](crate::script::ScriptReport) as
    // [`ScriptErrorKind::FuelExhausted`](crate::script::ScriptErrorKind)
    // while the parse continues with the next script. Script failures
    // never fail the navigation: they land in the
    // [`ScriptReport`](crate::script::ScriptReport). Scripting is always
    // on for a browser page load: the page runs scripts, so `<noscript>`
    // parses as raw text per WHATWG 13.2.6 — exactly like the loader's
    // own discovery parse.
    let mut page_vm = PageVm::new();
    // Native-stack guard for the page realm. Parser-inserted scripts —
    // notably WPT's `resources/testharness.js` — evaluate mid-parse,
    // *below* the guard anchor captured in `Vm::new`: every HTML
    // tokenizer, tree-builder, and executor frame counts against the
    // budget, and each JS call nests a fresh `Interp::run` dispatch loop
    // of several KiB. The 1 MiB default trips a few dozen nested calls
    // in, failing every testharness.js page with `RangeError: stack
    // overflow`. Raise it toward 7 MiB — the test262 harness recipe (7 MiB
    // guard on an 8 MiB thread, 1 MiB of headroom left to unwind) — via
    // `page_realm_native_stack_guard`.
    //
    // INVARIANT (enforced, not assumed): the installed guard never sits
    // above the installing thread's real stack end. Production install
    // threads: the `--check`/`--headless` workers (spawned with 8 MiB
    // stacks in `main.rs`), and the main thread for `--dump-display-list`
    // (via `load_tab_headless`), the windowed event thread
    // (`run_windowed_with_js`), and the `--session` WPT child
    // (`wpt_driver::run_session_child` → `run_session_loop` →
    // `load_tab_headless`). The main thread's size is `RLIMIT_STACK`,
    // which the operator can shrink via `ulimit -s`; the guard is
    // clamped to `min(7 MiB, rlimit − 1 MiB)` (probed once per process,
    // see `page_realm_native_stack_guard`), so a shrunk main thread
    // fails closed with a catchable `RangeError` instead of aborting on
    // a real stack overflow the too-high guard never tripped. (Unit tests
    // install on 2 MiB test threads, where the raised guard is simply
    // inert — no test recurses anywhere near either limit.)
    page_vm
        .vm_mut()
        .set_max_native_stack(page_realm_native_stack_guard());
    let (document, outcome) = {
        let mut executor = ParserExecutor::new(
            &mut page_vm,
            &base,
            &loaded.script_sources,
            Some((tab.viewport.w as u32, tab.viewport.h as u32)),
        );
        let (document, quirks) =
            Document::parse_html_with_executor(&loaded.body, true, &mut executor);
        // w77 quirks plumbing: the mode travels on the document itself
        // (`Document::quirks_mode`); layout/style read it from there.
        debug_assert_eq!(document.quirks_mode(), quirks);
        let outcome = executor.outcome();
        (document, outcome)
    };
    // One shared DOM tree: the tab owns the `Rc`, the new page realm
    // borrows the same allocation — no per-navigation clone.
    let shared = Rc::new(RefCell::new(document));
    // The scripted parse minted fresh node IDs, so re-key the loader's
    // URL-keyed intrinsics against the scripted document's nodes.
    let intrinsics: HashMap<NodeId, IntrinsicDims> = {
        let document = shared.borrow();
        let base = loader::effective_base_url(&document, &url);
        let mut intrinsics = HashMap::new();
        for (node, img_url) in loader::collect_images(&document, &base) {
            if let Some(size) = loaded.image_sizes.get(&img_url.serialize()) {
                intrinsics.insert(node, *size);
            }
        }
        intrinsics
    };
    tab.author = loaded.stylesheets;
    tab.image_intrinsics = intrinsics;
    tab.layout_root = None;
    tab.display_list = None;
    // Install the production web APIs on the same realm that ran the
    // parser-inserted scripts. The install overwrites the parse-phase
    // `document` shim, so post-parse scripts and event handlers see the
    // full binding surface. On failure the page keeps the bare realm —
    // the mid-parse scripts already ran, and the post-parse scripts
    // still get their turn below.
    // `readyState` transition hook: a successful install flips
    // `document.readyState` from the shim's `"loading"` data prop to the
    // `"interactive"` accessor (the parse has stopped). Post-parse
    // scripts below therefore observe `"interactive"`.
    if let Err(e) = page_vm.install_document_apis(shared.clone(), &url, user_agent, &tab.author) {
        eprintln!("bowser: web API install failed ({e}); page runs without bindings");
    }
    // Share the tab's scroll state before the first post-parse script
    // runs, so `window.scrollTo` during load operates on `Tab::scroll()`
    // (clamped to zero until the first layout publishes the content
    // bounds). Skipped when the web API install failed, which leaves no
    // web APIs to share with.
    if let Some(ctx) = page_vm.web_api_context() {
        bowser_js::share_scroll_state(ctx, Rc::clone(tab.scroll()));
        bowser_js::share_element_scroll_states(ctx, Rc::clone(tab.element_scroll()));
        bowser_js::share_completed_image_loads(ctx, Rc::clone(tab.completed_image_loads()));
        // Share the window's LIVE cache (not a copy): a `FontFace.load()`
        // in this realm installs into the same cache layout measures
        // with. This replaces any parse-shim cache the install above
        // adopted, exactly as `share_font_cache` documents. Without
        // it, a JS-loaded face lands in a lazily created orphan cache
        // and layout never sees it.
        bowser_js::share_font_cache(ctx, SharedFontCache::from_arc(Arc::clone(fonts)));
        // Share the profile's LIVE jar (not a copy): a same-origin
        // `FontFace.load()` in this realm sends the profile's cookies
        // and stores its `Set-Cookie` back into the profile jar, like
        // the loader's `@font-face` path. Sharing a snapshot copy
        // would diverge in both directions and is not wiring.
        bowser_js::share_cookie_jar(ctx, Arc::clone(cookie_jar));
    }
    // The post-parse scripts below run with the production web APIs
    // installed but before any pump: install the forced-sync-layout
    // slot now, so a geometry getter in those scripts (or in the
    // `DOMContentLoaded` / non-deferred `load` handlers) forces a
    // synchronous reflow instead of reading the no-layout answer.
    // The document is not installed in the tab yet, so the slot is
    // built from raw handles over the tab's render pipeline and the
    // fresh document's cell (see
    // [`AppLayoutEnsurer::for_pending_install`]); the guard drops
    // before `install_document` below.
    //
    // Fresh-document scroll state moves up here for the same reason:
    // the script phase (and any sync layout it forces) must see the
    // new page unscrolled, not the previous document's offsets and
    // containers. `install_document` still performs the bundle swap;
    // these resets are idempotent.
    let defer_load = tab.defer_load;
    tab.scroll().reset();
    tab.element_scroll().reset();
    tab.scroll_tree =
        crate::scroll::ScrollTree::for_viewport(tab.scroll(), tab.viewport, (true, true));
    // A forced sync layout during the script phase must take the full
    // path: drop the previous document's retained dirty report (its
    // roots name dead nodes) and mark style+layout dirty for the fresh
    // tree. `mark_all_dirty` below re-marks everything anyway.
    tab.clear_dirty_report();
    tab.render.dirty.style = true;
    tab.render.dirty.layout = true;
    let _ensurer_guard = page_vm.web_api_context().map(|ctx| {
        // SAFETY: `&mut tab.render` is converted to a raw pointer and
        // the borrow ends immediately. Until the guard drops, the
        // install code does not touch `tab.render` (or `tab` at all):
        // the script phase below uses only `page_vm`, `shared`, and
        // locals. The `shared` Rc outlives the guard, keeping the
        // document cell alive; `fonts` outlives it by the caller's
        // contract (the installing thread never holds the font lock
        // across the install, so the sync layout's lock cannot
        // deadlock). Mirrors the `reenter_page` slot contract.
        let ensurer = Rc::new(crate::tab::AppLayoutEnsurer::for_pending_install(
            &mut tab.render,
            &shared,
            ctx,
            fonts,
        ));
        crate::tab::install_layout_ensurer(ctx, ensurer)
    });
    // Post-parse path: the scripts the parser declined (`async` — whose
    // fetches the loader fired concurrently — `defer`, modules, data
    // blocks, bad URLs, failed or over-cap fetches) run here, in the same
    // realm the mid-parse scripts already ran in: classic scripts left
    // over from the parse first, then the `async` batch, then deferred
    // scripts, all before the window `load` event below.
    // `filter_handled_scripts` drops the refs the parser already
    // executed — so nothing runs twice — and attributes the mid-parse
    // failures to their elements.
    let (unhandled, mid_parse_errors) = filter_handled_scripts(
        crate::script::collect_script_refs(&shared.borrow()),
        &outcome,
    );
    for error in mid_parse_errors {
        page_vm.record_error(error);
    }
    let scripts = crate::script::resolve_scripts(unhandled, &base, |url| {
        loaded
            .script_sources
            .get(&url.serialize())
            .cloned()
            .unwrap_or_else(|| Err("external script was not prefetched by the loader".to_string()))
    });
    page_vm.run(&scripts);
    // The document (and its scripts) finished loading: fire
    // `DOMContentLoaded` (while readyState is still "interactive"), then
    // the window `load` event so page and harness handlers run. Listener
    // exceptions are swallowed by the dispatch (DOM report-and-continue);
    // a dispatch failure only means the web APIs never installed, which
    // was already reported above.
    if let Err(e) = page_vm.vm_mut().dispatch_dom_content_loaded() {
        eprintln!("bowser: DOMContentLoaded dispatch failed ({e}); page runs without it");
    }
    // `readyState` transition hook: the dispatch flips
    // `document.readyState` to `"complete"`, but only after the `load`
    // listeners have run — a listener reading `readyState` still sees
    // `"interactive"`.
    //
    // Headless (WPT driver) defers the `load` event until after the
    // first layout publishes: `load` handlers must observe geometry
    // (WPT `position-absolute-scrollable-overflow-001`), and at this
    // point no layout has run yet. `tab.defer_load` (set by
    // `load_tab_headless`) records the intent; the loader fires the
    // event after its first `refresh`.
    if defer_load {
        tab.pending_load = true;
    } else if let Err(e) = page_vm.vm_mut().dispatch_window_load() {
        eprintln!("bowser: load event dispatch failed ({e}); page runs without load handlers");
    }
    // The script phase is over: clear the forced-sync-layout slot
    // before the bundle swap below (the fresh document's cell moves
    // into the tab, whose sole-ownership gate the raw pointer does
    // not disturb, but the slot must not outlive this scope).
    drop(_ensurer_guard);
    // The single atomic swap: document and page realm install together
    // (see [`Tab::install_document`](crate::tab::Tab::install_document)).
    tab.install_document(shared, page_vm);
    // Fold any DOM mutations the page's scripts made into the render
    // flags, and drain the document's dirty set so it starts clean;
    // `mark_all_dirty` below then requests every stage anyway.
    tab.sync_dirty_flags();
    tab.title = tab
        .document_ref()
        .map(|doc| document_title(&doc))
        .unwrap_or_default();
    // Fresh document: unscrolled, and unscrollable until the first layout
    // publishes the content bounds. The scroll resets themselves moved
    // earlier (before the script phase); the paint bake still goes away
    // here: until the first layout re-discovers the containers the tree
    // is viewport-only.
    tab.paint_root = None;
    mark_all_dirty(tab);
}

/// Commit a finished load to the tab: push the final URL after redirects
/// to session history (when `push_history`), then install the document.
/// This is the single "what happens when a load lands" rule: the blocking
/// [`load_url`] path and the window worker's async completion both funnel
/// through here, so history and the installed document can never disagree
/// between the two paths.
pub(crate) fn complete_navigation(
    tab: &mut Tab,
    loaded: LoadedDocument,
    push_history: bool,
    user_agent: &str,
    fonts: &Arc<Mutex<FontCache>>,
    cookie_jar: &Arc<Mutex<CookieJar>>,
) {
    if push_history {
        tab.history.push(loaded.url.clone());
    }
    install_loaded_document(tab, loaded, user_agent, fonts, cookie_jar);
}

/// Same-document (fragment-only) fast path, shared by [`navigate`] and the
/// window worker's link-click flow.
///
/// When `url` addresses the same document as the tab's current URL
/// (differs only in fragment) and a document is loaded, the URL is
/// committed to history and the tab scrolls to the fragment anchor — no
/// refetch. Returns `Some` with the scrolled containers
/// innermost-first (`None` = viewport) when the fast path applied. The
/// fast path requires a loaded document: a tab whose history entry was recorded
/// without a fetch (e.g. right after [`crate::browser::Browser::new_tab`])
/// always does a full load, otherwise the tab would keep an empty
/// document forever.
pub(crate) fn navigate_fragment(tab: &mut Tab, url: &Url) -> Option<Vec<Option<NodeId>>> {
    if let Some(current) = tab.current_url()
        && same_document(current, url)
        && tab.has_document()
    {
        tab.history.push(url.clone());
        return Some(scroll_to_fragment(tab, url.fragment()));
    }
    None
}

/// Run the loader on a worker thread and block until it reports or
/// `timeout` fires.
///
/// `thread::scope` cannot be used here: the scope would join a stuck
/// worker and the deadline would never fire. On timeout the worker is
/// detached and abandoned (a thread stuck in blocking I/O cannot be
/// killed); any cookie updates it was carrying are discarded with its jar.
/// The tab is left completely untouched on error, and the profile cookie
/// jar keeps its pre-load cookies: a [`ProfileJarGuard`] holds the taken
/// jar for the whole load and puts it back on drop, so no failure path
/// can leave the profile with the empty swap jar.
fn load_document_blocking(
    url: &Url,
    jar: &Arc<Mutex<CookieJar>>,
    timeout: Duration,
    user_agent: &str,
) -> Result<LoadedDocument, NavError> {
    let mut guard = ProfileJarGuard::take(jar);
    let worker_jar = guard.worker_jar();
    let (tx, rx) = mpsc::channel();
    let url = url.clone();
    let user_agent = user_agent.to_owned();
    std::thread::spawn(move || {
        let outcome = run_loader(&url, worker_jar, &user_agent);
        // After a timeout the receiver is gone; the send failure just
        // means nobody is waiting anymore.
        let _ = tx.send(outcome);
    });
    let outcome = match rx.recv_timeout(timeout) {
        Ok(outcome) => outcome,
        // Abandoned load: the guard's drop restores the pre-load jar.
        Err(RecvTimeoutError::Timeout) => return Err(NavError::Timeout(timeout)),
        // The send is unconditional, so a disconnect means the worker
        // vanished without reporting; treat it as a dead worker.
        Err(RecvTimeoutError::Disconnected) => return Err(NavError::Cancelled),
    };
    // The load reported: install the worker's jar (with whatever cookies
    // the load set), disarming the guard's restore.
    guard.commit(outcome.jar);
    // A loader panic is reported as `Cancelled` rather than unwinding the
    // navigation caller.
    outcome
        .result
        .map_err(|_| NavError::Cancelled)?
        .map_err(NavError::from)
}

/// Full navigation to `url`: fetch, parse, install into the tab.
///
/// Runs the loader on a worker thread and blocks the caller on its report
/// channel (v1: navigation is synchronous from the caller's perspective;
/// the window worker must call this off the UI thread). `jar` is the
/// profile's shared cookie jar, as an `Arc` handle: the loader takes
/// `&Mutex<_>` (not `RefCell`) through deref coercion so the jar can
/// be shared across loader threads, and the install path shares this
/// same `Arc` with the page's JS context (see
/// [`install_loaded_document`]).
///
/// The load is bounded by `timeout`: if the report has not arrived by the
/// deadline, the worker is detached and abandoned (a thread stuck in
/// blocking I/O cannot be killed) and [`NavError::Timeout`] is returned.
/// Cookie updates from an abandoned load are discarded with its jar.
///
/// When `push_history`, the final URL after redirects is committed as a
/// new session-history entry (forward entries truncated); traversals and
/// reloads pass `false` so history is left untouched.
///
/// Returns the final URL after redirects. On error the tab is left
/// completely untouched (history and document are only swapped in after a
/// successful load).
fn load_url(
    tab: &mut Tab,
    jar: &Arc<Mutex<CookieJar>>,
    url: &Url,
    timeout: Duration,
    push_history: bool,
    user_agent: &str,
    fonts: &Arc<Mutex<FontCache>>,
) -> Result<Url, NavError> {
    // `jar` is the shared profile handle: the loader's `&Mutex`
    // paths take it through deref coercion, and the install path
    // shares this same `Arc` with the page's JS context.
    let loaded = load_document_blocking(url, jar, timeout, user_agent)?;
    let final_url = loaded.url.clone();
    complete_navigation(tab, loaded, push_history, user_agent, fonts, jar);
    Ok(final_url)
}

/// Two URLs address the same document when they differ only in fragment.
fn same_document(a: &Url, b: &Url) -> bool {
    a.serialize_without_fragment() == b.serialize_without_fragment()
}

/// v1 document title: the first `<title>` element's text content.
fn document_title(document: &bowser_dom::Document) -> String {
    document
        .get_elements_by_tag_name("title")
        .first()
        .map(|n| document.text_content(n))
        .unwrap_or_default()
}

/// Scroll the tab to a fragment anchor.
///
/// Finds the element with the given id and reveals it in every scroll
/// container up the chain (nested scrollers first, then the viewport).
/// v1 fallbacks, documented: if the element is not found, if layout has
/// not run yet (no `layout_root`), or if the layout root is stale (the
/// `layout` dirty flag is set, script mutated the DOM after layout ran),
/// the tab scrolls to the top instead of the anchor. A missing fragment
/// scrolls to the top.
///
/// Returns the scrolled containers innermost-first (`None` = viewport),
/// for `scroll` event dispatch. Historical behavior preserved: a fragment
/// navigation resets the horizontal offset.
///
/// Like the wheel/keyboard/`scrollIntoView()` paths, nested-container
/// movement is folded into the paint tree and paint is invalidated
/// ([`crate::window::sync_nested_paint`]): without it the display list
/// rebuild would lag one tick behind the scroll and the next script pump
/// would see live != baked offsets and fire the `scroll` listeners a
/// second time.
fn scroll_to_fragment(tab: &mut Tab, fragment: Option<&str>) -> Vec<Option<NodeId>> {
    // A dirty layout root is stale: its boxes may have moved since
    // layout ran. Don't trust it; fall back to the top.
    let node = fragment.and_then(|frag| {
        if tab.dirty.layout {
            return None;
        }
        tab.document_ref()?.get_element_by_id(frag)
    });
    let scrolled = match node {
        Some(n) => {
            // Fragment navigation keeps its historical behavior:
            // block-start, horizontal offset kept, auto (smooth when the
            // scroller's CSS `scroll-behavior` says so).
            let scrolled = scroll_to_node(
                tab,
                n,
                RevealAlign::Start,
                RevealAlign::Keep,
                ScrollBehavior::Auto,
            );
            // Historical behavior: a fragment navigation resets the
            // horizontal offset.
            let (before_x, _) = tab.scroll().offset();
            if before_x != 0.0 {
                tab.scroll().set_offset(0.0, tab.scroll().offset().1);
                let mut scrolled = scrolled;
                if !scrolled.contains(&None) {
                    scrolled.push(None);
                }
                scrolled
            } else {
                scrolled
            }
        }
        None => {
            // A missing fragment (or element) scrolls to the top.
            let before = tab.scroll().offset();
            tab.scroll().set_offset(0.0, 0.0);
            if tab.scroll().offset() != before {
                vec![None]
            } else {
                Vec::new()
            }
        }
    };
    crate::window::sync_nested_paint(tab, &scrolled);
    scrolled
}

/// Scroll `node` into view in every scroll container up the chain: the
/// `scrollIntoView()` backend (and fragment navigation, with
/// `Start`/`Keep`/`Auto`).
///
/// The bindings only record the target node and the parsed options
/// (they cannot see the layout tree); the window worker drains the
/// request through here once per event-loop tick (see
/// `window::pump_scripts`). Same stale-layout guard as
/// [`scroll_to_fragment`]. Returns the scrolled containers
/// innermost-first (`None` = viewport), for `scroll` event dispatch.
///
/// `behavior` resolves per container against that container's CSS
/// `scroll-behavior` default: smooth containers animate (their movement
/// is reported by the animation ticks, not here).
pub(crate) fn scroll_to_node(
    tab: &mut Tab,
    node: bowser_dom::NodeId,
    block: RevealAlign,
    inline: RevealAlign,
    behavior: ScrollBehavior,
) -> Vec<Option<NodeId>> {
    if tab.dirty.layout {
        return Vec::new();
    }
    let Some(root) = tab.layout_root.as_ref() else {
        return Vec::new();
    };
    tab.scroll_tree
        .reveal_node_with_options(root, node, block, inline, behavior)
}

// ---------------------------------------------------------------------------
// Public API: synchronous navigation.
//
// The windowed event loop cannot block on these (it uses the async
// `spawn_load`/`poll_pending` path in `window.rs`, which shares the load
// primitives above: jar swap, worker body, fragment fast path, and load
// completion). These are the blocking API for synchronous callers —
// headless diagnostics (`--dump-display-list`), tests, and the next wave
// of browser workers.
// ---------------------------------------------------------------------------

/// Navigate the tab to `url_str`, resolved against the tab's current URL
/// as base, bounded by the default [`NAVIGATION_TIMEOUT`].
///
/// See [`navigate_with_timeout`] for the semantics.
pub fn navigate(
    tab: &mut Tab,
    jar: &Arc<Mutex<CookieJar>>,
    url_str: &str,
    user_agent: &str,
    fonts: &Arc<Mutex<FontCache>>,
) -> Result<(), NavError> {
    navigate_with_timeout(tab, jar, url_str, NAVIGATION_TIMEOUT, user_agent, fonts)
}

/// Navigate the tab to `url_str`, resolved against the tab's current URL
/// as base, bounded by `timeout`.
///
/// * Fragment-only change (same URL ignoring fragment): same-document
///   navigation, no refetch. The URL is committed to history, the document
///   is kept, and the tab scrolls to the fragment anchor
///   ([`scroll_to_fragment`]).
/// * `about:blank`: the built-in welcome page installs with no fetch —
///   the same page the windowed and headless initial-load paths render.
/// * Anything else: full navigation. The document (plus stylesheets and
///   image intrinsics) loads on a worker thread via
///   [`crate::loader::load_document_with_user_agent`]; the final URL after redirects is
///   pushed to history (forward entries truncated); all dirty flags are
///   set so the window worker rebuilds style, layout, and the display
///   list. The document's `<script>` elements run through a fresh
///   [`PageVm`](crate::script::PageVm) when the load installs (see
///   [`install_loaded_document`]).
///
/// v1 blocks the caller until the load finishes or the overall
/// navigation deadline fires; call off the UI thread.
/// There is no bfcache yet: every full navigation re-parses from the
/// fetched bytes.
pub fn navigate_with_timeout(
    tab: &mut Tab,
    jar: &Arc<Mutex<CookieJar>>,
    url_str: &str,
    timeout: Duration,
    user_agent: &str,
    fonts: &Arc<Mutex<FontCache>>,
) -> Result<(), NavError> {
    let url = bowser_url::parse(url_str, tab.current_url()).map_err(NavError::Url)?;
    // Exact same URL (fragment included): reload in place instead of
    // pushing a duplicate history entry. Honors the caller's timeout.
    if tab.current_url().is_some_and(|current| current == &url) {
        return reload_with_timeout(tab, jar, timeout, user_agent, fonts);
    }
    if navigate_fragment(tab, &url).is_some() {
        return Ok(());
    }
    if url.serialize() == "about:blank" {
        // No fetch: push the entry, then install the built-in welcome
        // page — the same two steps `complete_navigation` performs for a
        // fetched load.
        tab.history.push(url.clone());
        crate::window::install_welcome_page(tab, &url, user_agent, fonts, jar);
        return Ok(());
    }
    load_url(tab, jar, &url, timeout, true, user_agent, fonts)?;
    Ok(())
}
/// Same-document (fragment-only) traversal fast path, shared by
/// [`go_back`], [`go_forward`], and the window worker's async traversal
/// (which cannot call the blocking traversals: they would race the
/// in-flight async load's cookie-jar handoff).
///
/// When the target entry addresses the same document as the tab's
/// current URL (differs only in fragment) and a document is loaded, the
/// history index moves and the tab scrolls to the fragment anchor — no
/// refetch, no JS realm replacement. Returns `Some` with the scrolled
/// containers innermost-first (`None` = viewport) when the fast path
/// applied. Like [`navigate_fragment`], the fast path requires a loaded
/// document: without one the tab falls through to a full load.
pub(crate) fn traverse_fragment(
    tab: &mut Tab,
    target: usize,
    url: &Url,
) -> Option<Vec<Option<NodeId>>> {
    if !tab.has_document() {
        return None;
    }
    let current = tab.current_url()?;
    if !same_document(current, url) {
        return None;
    }
    if target < tab.history.index() {
        tab.history.back();
    } else {
        tab.history.forward();
    }
    Some(scroll_to_fragment(tab, url.fragment()))
}

/// Go back one session-history entry.
///
/// v1 re-fetches and re-parses the entry (bfcache deferred) — except
/// same-document (fragment-only) entries, which scroll without
/// refetching and keep the JS realm ([`traverse_fragment`]). At the
/// oldest entry this is a no-op returning `Ok`. On load failure the
/// history index is left unchanged. When the entry's URL redirects, the
/// entry is rewritten to the final URL (the same rule [`navigate`] uses
/// when it pushes), so `current_url` matches the loaded document.
pub fn go_back(
    tab: &mut Tab,
    jar: &Arc<Mutex<CookieJar>>,
    user_agent: &str,
    fonts: &Arc<Mutex<FontCache>>,
) -> Result<(), NavError> {
    if !tab.history.can_go_back() {
        return Ok(());
    }
    let target = tab.history.index() - 1;
    let url = tab
        .history
        .entry(target)
        .expect("can_go_back implies the target entry exists")
        .clone();
    if traverse_fragment(tab, target, &url).is_some() {
        return Ok(());
    }
    // Load first: only move the history index on success, so a failed
    // back leaves the tab on the current entry.
    let final_url = load_url(tab, jar, &url, NAVIGATION_TIMEOUT, false, user_agent, fonts)?;
    if final_url != url {
        replace_history_entry(tab, target, final_url);
    }
    tab.history.back();
    Ok(())
}
/// Rewrite the session-history entry at `index` to `url` in place.
///
/// Session-history entries always store the final URL after redirects:
/// [`navigate`] pushes the final URL for a fresh navigation, and the
/// back/forward traversals plus the window worker's async initial-load
/// completion (`window::WindowApp::poll_pending`, where the entry was
/// recorded at tab creation with the pre-redirect URL) re-apply it here,
/// so [`Tab::current_url`] agrees with the loaded document no matter
/// which path arrived at the entry. Forward entries are untouched.
pub(crate) fn replace_history_entry(tab: &mut Tab, index: usize, url: Url) {
    tab.history.replace(index, url);
}

/// Go forward one session-history entry.
///
/// Same v1 re-fetch semantics as [`go_back`], including the
/// same-document fragment fast path ([`traverse_fragment`]): the entry
/// is rewritten to the final URL when its URL redirects. Past the
/// newest entry this is a no-op returning `Ok`.
pub fn go_forward(
    tab: &mut Tab,
    jar: &Arc<Mutex<CookieJar>>,
    user_agent: &str,
    fonts: &Arc<Mutex<FontCache>>,
) -> Result<(), NavError> {
    if !tab.history.can_go_forward() {
        return Ok(());
    }
    let target = tab.history.index() + 1;
    let url = tab
        .history
        .entry(target)
        .expect("can_go_forward implies the target entry exists")
        .clone();
    if traverse_fragment(tab, target, &url).is_some() {
        return Ok(());
    }
    let final_url = load_url(tab, jar, &url, NAVIGATION_TIMEOUT, false, user_agent, fonts)?;
    if final_url != url {
        replace_history_entry(tab, target, final_url);
    }
    tab.history.forward();
    Ok(())
}
/// Reload the current entry without touching history.
///
/// v1 re-fetches unconditionally (no validators / conditional GET yet).
/// On a tab with no history this is a no-op returning `Ok`. When the
/// entry's URL redirects, the entry is rewritten to the final URL after
/// redirects — the same rule [`navigate`], the back/forward traversals,
/// and the windowed reload path
/// (`window::install_final_document`) apply — so `current_url` matches
/// the loaded document.
///
/// `about:blank` has no network origin: reloading it reinstalls the
/// built-in welcome page in place (no fetch, no history change) instead
/// of failing on a fetch of a non-fetchable URL.
pub fn reload(
    tab: &mut Tab,
    jar: &Arc<Mutex<CookieJar>>,
    user_agent: &str,
    fonts: &Arc<Mutex<FontCache>>,
) -> Result<(), NavError> {
    reload_with_timeout(tab, jar, NAVIGATION_TIMEOUT, user_agent, fonts)
}

/// [`reload`] with an explicit load deadline, shared with the same-URL
/// fast path of [`navigate_with_timeout`] so a same-URL navigation
/// honors the caller's timeout rather than the default navigation one.
fn reload_with_timeout(
    tab: &mut Tab,
    jar: &Arc<Mutex<CookieJar>>,
    timeout: Duration,
    user_agent: &str,
    fonts: &Arc<Mutex<FontCache>>,
) -> Result<(), NavError> {
    let Some(url) = tab.current_url().cloned() else {
        return Ok(());
    };
    if url.serialize() == "about:blank" {
        crate::window::install_welcome_page(tab, &url, user_agent, fonts, jar);
        return Ok(());
    }
    let final_url = load_url(tab, jar, &url, timeout, false, user_agent, fonts)?;
    if final_url != url {
        replace_history_entry(tab, tab.history.index(), final_url);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use bowser_js::Value;
    use bowser_layout::{BoxType, LayoutBox, Rect, Size};
    use bowser_net::cookies::RequestContext;

    use crate::tab::DirtyFlags;

    /// Default UA for navigation tests (the tests assert plumbing, not the
    /// value; the configured-UA case has its own test).
    const UA: &str = bowser_net::fetch::USER_AGENT;

    const PAGE_ONE: &str = "<!DOCTYPE html><html><head><title>Page One</title>\
        <style>p{color:red}</style></head>\
        <body><div id=\"frag\">hello</div></body></html>";
    const PAGE_TWO: &str =
        "<!DOCTYPE html><html><head><title>Page Two</title></head><body>two</body></html>";
    const PAGE_FINAL: &str =
        "<!DOCTYPE html><html><head><title>Final Page</title></head><body>final</body></html>";
    // Cross-script state in the fixtures rides sloppy-mode globals on the
    // shared global object: the engine scopes each script's lexical
    // `var`s to that script's eval, so `var counter` would not survive
    // into the next `<script>` (documented engine gap).
    const PAGE_SCRIPTS: &str = "<!DOCTYPE html><html><head><title>Scripts</title>\
        <script>counter = 1;</script>\
        <script src=\"/counter.js\"></script>\
        <script>counter += 1;</script>\
        <script>throw new Error('boom');</script>\
        <script>counter += 10;</script></head>\
        <body>scripts</body></html>";
    const PAGE_SCRIPTS_TWO: &str = "<!DOCTYPE html><html><head><title>Scripts Two</title>\
        <script>counter = 100;</script></head><body>two</body></html>";
    const COUNTER_JS: &str = "counter += 41;";
    const PAGE_FRAG_SCRIPTS: &str = "<!DOCTYPE html><html><head><title>Frag Scripts</title>\
        <script>marker = 7;</script></head>\
        <body><div id=\"frag\">anchor</div></body></html>";

    fn test_tab() -> Tab {
        Tab::new(Size { w: 800.0, h: 600.0 })
    }

    fn test_jar() -> Arc<Mutex<CookieJar>> {
        Arc::new(Mutex::new(CookieJar::new()))
    }

    /// Profile jar seeded with one cookie, so a wipe would be visible.
    fn test_jar_with_cookie() -> Arc<Mutex<CookieJar>> {
        let jar = test_jar();
        jar.lock().expect("test jar").set_cookie(
            "session=abc123; Path=/",
            &RequestContext {
                host: "127.0.0.1",
                path: "/",
                secure: false,
                http_api: true,
                top_host: None,
                top_secure: None,
                top_level_safe_navigation: false,
            },
            0,
        );
        assert_eq!(jar.lock().expect("test jar").len(), 1);
        jar
    }

    /// Canned HTTP/1.1 server on 127.0.0.1. Returns the base URL and a
    /// per-request hit counter.
    fn spawn_server() -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("test server addr");
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_srv = Arc::clone(&hits);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = match stream {
                    Ok(s) => s,
                    Err(_) => break,
                };
                let hits = Arc::clone(&hits_srv);
                std::thread::spawn(move || {
                    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 4096];
                    loop {
                        match stream.read(&mut chunk) {
                            Ok(0) => break,
                            Ok(n) => {
                                buf.extend_from_slice(&chunk[..n]);
                                if buf.windows(4).any(|w| w == b"\r\n\r\n") || buf.len() > 65_536 {
                                    break;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                    hits.fetch_add(1, Ordering::SeqCst);
                    let (status, reason, extra_headers, body) = route(request_path(&buf));
                    let head = format!(
                        "HTTP/1.1 {status} {reason}\r\n\
                         Content-Type: text/html; charset=utf-8\r\n\
                         Content-Length: {}\r\n\
                         {extra_headers}Connection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(head.as_bytes());
                    let _ = stream.write_all(body.as_bytes());
                });
            }
        });
        (format!("http://{addr}"), hits)
    }

    fn request_path(buf: &[u8]) -> &str {
        let head = std::str::from_utf8(buf).unwrap_or("");
        let target = head
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or("/");
        // The fetch client sends absolute-form request targets
        // ("GET http://host:port/path HTTP/1.1"); strip to origin-form.
        let origin = target
            .strip_prefix("http://")
            .or_else(|| target.strip_prefix("https://"))
            .and_then(|rest| rest.find('/').map(|i| &rest[i..]))
            .unwrap_or(target);
        origin.split('?').next().unwrap_or(origin)
    }

    fn route(path: &str) -> (u16, &'static str, &'static str, &'static str) {
        match path {
            "/page" => (200, "OK", "", PAGE_ONE),
            "/other" => (200, "OK", "", PAGE_TWO),
            "/redir" => (302, "Found", "Location: /final\r\n", ""),
            "/final" => (200, "OK", "", PAGE_FINAL),
            "/scripts" => (200, "OK", "", PAGE_SCRIPTS),
            "/scripts2" => (200, "OK", "", PAGE_SCRIPTS_TWO),
            "/fragscripts" => (200, "OK", "", PAGE_FRAG_SCRIPTS),
            "/counter.js" => (200, "OK", "", COUNTER_JS),
            _ => (404, "Not Found", "", "<title>nope</title>"),
        }
    }

    fn hit_count(hits: &Arc<AtomicUsize>) -> usize {
        hits.load(Ordering::SeqCst)
    }

    #[test]
    fn navigate_loads_document_title_and_invalidates() {
        let (base, _hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar();

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .expect("navigate");

        assert_eq!(tab.title, "Page One");
        assert_eq!(
            tab.current_url().expect("history").serialize(),
            format!("{base}/page")
        );
        assert!(tab.has_document());
        assert_eq!(tab.author.len(), 1, "inline <style> collected");
        assert!(tab.layout_root.is_none());
        assert!(tab.display_list.is_none());
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
        assert!(tab.dirty.style && tab.dirty.layout && tab.dirty.paint);
    }

    #[test]
    fn relative_url_resolves_against_current() {
        let (base, _hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar();

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        navigate(
            &mut tab,
            &jar,
            "other",
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();

        assert_eq!(tab.current_url().expect("history").path(), "/other");
        assert_eq!(tab.title, "Page Two");
    }

    #[test]
    fn fragment_navigation_does_not_refetch() {
        let (base, hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar();

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        assert_eq!(hit_count(&hits), 1);

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page#frag"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page#other"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();

        assert_eq!(hit_count(&hits), 1, "fragment nav must not refetch");
        assert_eq!(tab.title, "Page One", "same document kept");
        assert!(tab.has_document());
        assert_eq!(
            tab.current_url().expect("history").fragment(),
            Some("other")
        );
        // v1: no layout has run, so the anchor cannot be resolved; scroll
        // falls back to the top.
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
        // Fragment navigations commit history entries.
        assert_eq!(tab.history.len(), 3);
        assert!(tab.dirty.style && tab.dirty.layout && tab.dirty.paint);
    }

    #[test]
    fn fragment_navigation_scrolls_to_anchor_when_laid_out() {
        let (base, _hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar();

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        let anchor = tab
            .document_ref()
            .expect("document")
            .get_element_by_id("frag")
            .expect("anchor element exists");

        // Fake a laid-out tree: the anchor sits at y=321 in a document
        // 1000px tall (scrollable: max scroll offset is (0, 400)).
        // Faking layout also means faking it clean — a dirty layout root
        // is stale and `scroll_to_fragment` would (correctly) ignore it —
        // and faking what `refresh` would publish into the shared scroll
        // state (the content bounds).
        tab.layout_root = Some(Rc::new(LayoutBox {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                w: 800.0,
                h: 600.0,
            },
            children: vec![LayoutBox {
                rect: Rect {
                    x: 0.0,
                    y: 321.0,
                    w: 800.0,
                    h: 679.0,
                },
                children: vec![],
                box_type: BoxType::Block,
                style: None,
                text: None,
                node: Some(anchor),
                fragments: vec![],
                z_index: None,
                collapsed_borders: None,
                phantom_w: false,
                phantom_h: false,
            }],
            box_type: BoxType::Block,
            style: None,
            text: None,
            node: None,
            fragments: vec![],
            z_index: None,
            collapsed_borders: None,
            phantom_w: false,
            phantom_h: false,
        }));
        tab.dirty.layout = false;
        tab.scroll().set_max_offset((0.0, 400.0));

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page#frag"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        assert_eq!(tab.scroll().offset(), (0.0, 321.0));
    }

    #[test]
    fn fragment_navigation_clamps_anchor_to_content_bounds() {
        let (base, _hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar();

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        let anchor = tab
            .document_ref()
            .expect("document")
            .get_element_by_id("frag")
            .expect("anchor element exists");

        // Fake a laid-out tree: the anchor sits at y=900, but the
        // document is only 1000px tall in a 600px viewport, so the max
        // scroll offset is (0, 400) — the anchor is unreachable and the
        // scroll clamps instead of overscrolling.
        tab.layout_root = Some(Rc::new(LayoutBox {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                w: 800.0,
                h: 600.0,
            },
            children: vec![LayoutBox {
                rect: Rect {
                    x: 0.0,
                    y: 900.0,
                    w: 800.0,
                    h: 100.0,
                },
                children: vec![],
                box_type: BoxType::Block,
                style: None,
                text: None,
                node: Some(anchor),
                fragments: vec![],
                z_index: None,
                collapsed_borders: None,
                phantom_w: false,
                phantom_h: false,
            }],
            box_type: BoxType::Block,
            style: None,
            text: None,
            node: None,
            fragments: vec![],
            z_index: None,
            collapsed_borders: None,
            phantom_w: false,
            phantom_h: false,
        }));
        tab.dirty.layout = false;
        tab.scroll().set_max_offset((0.0, 400.0));

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page#frag"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        assert_eq!(tab.scroll().offset(), (0.0, 400.0));
    }

    #[test]
    fn fragment_navigation_ignores_stale_layout() {
        let (base, _hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar();

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        let anchor = tab
            .document_ref()
            .expect("document")
            .get_element_by_id("frag")
            .expect("anchor element exists");

        // A layout root is present, but the layout flag is dirty (script
        // mutated the DOM after layout ran): the root is stale and must
        // not be trusted — scroll falls back to the top.
        tab.layout_root = Some(Rc::new(LayoutBox {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                w: 800.0,
                h: 600.0,
            },
            children: vec![LayoutBox {
                rect: Rect {
                    x: 0.0,
                    y: 321.0,
                    w: 800.0,
                    h: 679.0,
                },
                children: vec![],
                box_type: BoxType::Block,
                style: None,
                text: None,
                node: Some(anchor),
                fragments: vec![],
                z_index: None,
                collapsed_borders: None,
                phantom_w: false,
                phantom_h: false,
            }],
            box_type: BoxType::Block,
            style: None,
            text: None,
            node: None,
            fragments: vec![],
            z_index: None,
            collapsed_borders: None,
            phantom_w: false,
            phantom_h: false,
        }));
        assert!(tab.dirty.layout, "navigation leaves layout dirty");
        tab.scroll().set_max_offset((0.0, 400.0));

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page#frag"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        assert_eq!(tab.scroll().offset(), (0.0, 0.0));
    }

    #[test]
    fn history_back_and_forward_refetch() {
        let (base, hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar();

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        navigate(
            &mut tab,
            &jar,
            &format!("{base}/other"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        assert_eq!(tab.title, "Page Two");
        assert_eq!(hit_count(&hits), 2);

        go_back(&mut tab, &jar, UA, &Arc::new(Mutex::new(FontCache::new()))).unwrap();
        assert_eq!(tab.current_url().expect("history").path(), "/page");
        assert_eq!(tab.title, "Page One");
        assert_eq!(hit_count(&hits), 3, "back re-fetches in v1");

        go_forward(&mut tab, &jar, UA, &Arc::new(Mutex::new(FontCache::new()))).unwrap();
        assert_eq!(tab.current_url().expect("history").path(), "/other");
        assert_eq!(tab.title, "Page Two");

        // Past the newest entry: no-op.
        go_forward(&mut tab, &jar, UA, &Arc::new(Mutex::new(FontCache::new()))).unwrap();
        assert_eq!(tab.history.index(), 1);
        assert_eq!(hit_count(&hits), 4);
    }

    #[test]
    fn back_at_oldest_entry_is_noop() {
        let (base, hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar();

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        go_back(&mut tab, &jar, UA, &Arc::new(Mutex::new(FontCache::new()))).expect("no-op back");

        assert_eq!(tab.history.index(), 0);
        assert_eq!(tab.title, "Page One");
        assert_eq!(hit_count(&hits), 1);
    }

    #[test]
    fn new_navigation_truncates_forward_history() {
        let (base, _hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar();

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        navigate(
            &mut tab,
            &jar,
            &format!("{base}/other"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        go_back(&mut tab, &jar, UA, &Arc::new(Mutex::new(FontCache::new()))).unwrap();
        navigate(
            &mut tab,
            &jar,
            &format!("{base}/final"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();

        assert_eq!(tab.history.len(), 2);
        assert_eq!(tab.history.index(), 1);
        assert_eq!(tab.current_url().expect("history").path(), "/final");
    }

    #[test]
    fn redirect_history_records_final_url() {
        let (base, hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar();

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/redir"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();

        assert_eq!(tab.current_url().expect("history").path(), "/final");
        assert_eq!(tab.title, "Final Page");
        assert_eq!(tab.history.len(), 1);
        assert_eq!(hit_count(&hits), 2, "initial URL plus redirect target");
    }

    #[test]
    fn back_rewrites_redirected_entry_to_final_url() {
        let (base, _hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar();

        // Plant an entry whose URL redirects when re-fetched (e.g. the
        // server's redirect mapping changed since the entry was recorded):
        // `navigate` itself never stores such entries, it pushes the
        // final URL, but traversal must apply the same rule.
        tab.history
            .push(bowser_url::parse(&format!("{base}/redir"), None).unwrap());
        tab.history
            .push(bowser_url::parse(&format!("{base}/other"), None).unwrap());

        go_back(&mut tab, &jar, UA, &Arc::new(Mutex::new(FontCache::new()))).unwrap();

        // The entry was rewritten to the final URL, so `current_url`
        // agrees with the loaded document; forward history is intact.
        assert_eq!(tab.history.index(), 0);
        assert_eq!(tab.history.len(), 2);
        assert_eq!(tab.history.entry(0).expect("entry").path(), "/final");
        assert_eq!(tab.current_url().expect("history").path(), "/final");
        assert_eq!(tab.title, "Final Page");

        go_forward(&mut tab, &jar, UA, &Arc::new(Mutex::new(FontCache::new()))).unwrap();
        assert_eq!(tab.current_url().expect("history").path(), "/other");
        assert_eq!(tab.title, "Page Two");
    }

    #[test]
    fn forward_rewrites_redirected_entry_to_final_url() {
        let (base, _hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar();

        tab.history
            .push(bowser_url::parse(&format!("{base}/page"), None).unwrap());
        tab.history
            .push(bowser_url::parse(&format!("{base}/redir"), None).unwrap());

        go_back(&mut tab, &jar, UA, &Arc::new(Mutex::new(FontCache::new()))).unwrap();
        assert_eq!(tab.current_url().expect("history").path(), "/page");

        go_forward(&mut tab, &jar, UA, &Arc::new(Mutex::new(FontCache::new()))).unwrap();

        assert_eq!(tab.history.index(), 1);
        assert_eq!(tab.history.len(), 2);
        assert_eq!(tab.history.entry(1).expect("entry").path(), "/final");
        assert_eq!(tab.current_url().expect("history").path(), "/final");
        assert_eq!(tab.title, "Final Page");
    }

    /// Server that accepts one connection and then drips the response body
    /// one byte at a time: each socket read returns quickly, so the net
    /// stack's per-read socket timeout never fires, but the load as a
    /// whole never completes within any reasonable deadline.
    fn spawn_drip_server() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind drip server");
        let addr = listener.local_addr().expect("drip server addr");
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
            // Drain the request head so the client is not writing into a
            // socket we never read.
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let head = "HTTP/1.1 200 OK\r\n\
                        Content-Type: text/html; charset=utf-8\r\n\
                        Content-Length: 1000000\r\n\
                        Connection: close\r\n\r\n";
            if stream.write_all(head.as_bytes()).is_err() {
                return;
            }
            // One byte every 50 ms: far under the 60 s socket read
            // timeout, so only the overall navigation deadline can fire.
            for _ in 0..400 {
                std::thread::sleep(Duration::from_millis(50));
                if stream.write_all(b"x").is_err() {
                    return;
                }
            }
        });
        format!("http://{addr}")
    }

    #[test]
    fn navigate_with_timeout_honors_custom_deadline() {
        let base = spawn_drip_server();
        let mut tab = test_tab();
        let jar = test_jar();

        let start = std::time::Instant::now();
        let err = navigate_with_timeout(
            &mut tab,
            &jar,
            &format!("{base}/drip"),
            Duration::from_millis(500),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap_err();
        let elapsed = start.elapsed();

        assert!(
            matches!(err, NavError::Timeout(_)),
            "expected a deadline timeout, got: {err:?}"
        );
        assert!(
            elapsed < Duration::from_secs(10),
            "custom deadline did not bound the load: {elapsed:?}"
        );
        // The timed-out load installed nothing: the tab is untouched.
        assert!(!tab.has_document());
        assert!(tab.history.is_empty());
    }

    #[test]
    fn navigation_deadline_fires_on_drip_feed() {
        let base = spawn_drip_server();
        let mut tab = test_tab();
        let jar = test_jar();
        let url = bowser_url::parse(&format!("{base}/drip"), None).unwrap();

        let start = std::time::Instant::now();
        let err = load_url(
            &mut tab,
            &jar,
            &url,
            Duration::from_secs(1),
            false,
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap_err();
        let elapsed = start.elapsed();

        assert!(
            matches!(err, NavError::Timeout(_)),
            "expected a deadline timeout, got: {err:?}"
        );
        assert!(
            elapsed < Duration::from_secs(10),
            "deadline did not bound the load: {elapsed:?}"
        );
        // The timed-out load installed nothing: the tab is untouched.
        assert!(!tab.has_document());
        assert!(tab.history.is_empty());
        assert_eq!(tab.title, "");
    }

    #[test]
    fn timeout_preserves_profile_cookie_jar() {
        let base = spawn_drip_server();
        let mut tab = test_tab();
        let jar = test_jar_with_cookie();
        let url = bowser_url::parse(&format!("{base}/drip"), None).unwrap();

        let err = load_url(
            &mut tab,
            &jar,
            &url,
            Duration::from_secs(1),
            false,
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap_err();
        assert!(
            matches!(err, NavError::Timeout(_)),
            "expected a deadline timeout, got: {err:?}"
        );

        // The abandoned load's cookie updates are gone with its worker,
        // but the pre-load jar was put back: a timeout must not wipe the
        // profile's cookies.
        assert_eq!(
            jar.lock().expect("jar").len(),
            1,
            "timed-out navigation wiped the profile cookie jar"
        );
        assert!(!tab.has_document());
        assert!(tab.history.is_empty());
    }

    #[test]
    fn successful_load_preserves_and_updates_profile_cookie_jar() {
        let (base, _hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar_with_cookie();

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .expect("navigate");

        // The finished load installed its (empty, the test server sets
        // none) updates without losing the pre-load cookie.
        assert_eq!(
            jar.lock().expect("jar").len(),
            1,
            "successful navigation wiped the profile cookie jar"
        );
        assert_eq!(tab.title, "Page One");
    }

    #[test]
    fn reload_refetches_without_pushing_history() {
        let (base, hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar();

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        reload(&mut tab, &jar, UA, &Arc::new(Mutex::new(FontCache::new()))).unwrap();

        assert_eq!(hit_count(&hits), 2);
        assert_eq!(tab.history.len(), 1);
        assert_eq!(tab.title, "Page One");
    }

    #[test]
    fn reload_rewrites_history_entry_to_final_url_after_redirect() {
        let (base, _hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar();

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        // Plant a pre-redirect entry (the server redirects /redir to
        // /final), the way the traversal rewrite tests do.
        tab.history
            .push(bowser_url::parse(&format!("{base}/redir"), None).unwrap());

        reload(&mut tab, &jar, UA, &Arc::new(Mutex::new(FontCache::new()))).unwrap();

        assert_eq!(tab.history.len(), 2, "reload must not push history");
        assert_eq!(
            tab.history.entry(1).expect("entry").path(),
            "/final",
            "the entry must be rewritten to the final URL"
        );
        assert_eq!(tab.current_url().expect("history").path(), "/final");
        assert_eq!(tab.title, "Final Page");
    }

    /// Traversing to a same-document (fragment-only) history entry must
    /// scroll, not refetch — and must keep the JS realm (no re-run, no
    /// lost globals).
    #[test]
    fn back_and_forward_to_fragment_entries_do_not_refetch() {
        let (base, hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar();

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/fragscripts"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        assert_eq!(hit_count(&hits), 1);

        // Fake a laid-out tree: the #frag anchor sits at y=321.
        let anchor = tab
            .document_ref()
            .expect("document")
            .get_element_by_id("frag")
            .expect("anchor element exists");
        // Fake a laid-out tree: the anchor sits at y=321 in a scrollable
        // document. Faking layout means faking it clean (a dirty root is
        // stale and ignored) and faking the max scroll `refresh` would
        // publish.
        tab.layout_root = Some(Rc::new(LayoutBox {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                w: 800.0,
                h: 600.0,
            },
            children: vec![LayoutBox {
                rect: Rect {
                    x: 0.0,
                    y: 321.0,
                    w: 800.0,
                    h: 679.0,
                },
                children: vec![],
                box_type: BoxType::Block,
                style: None,
                text: None,
                node: Some(anchor),
                fragments: vec![],
                z_index: None,
                collapsed_borders: None,
                phantom_w: false,
                phantom_h: false,
            }],
            box_type: BoxType::Block,
            style: None,
            text: None,
            node: None,
            fragments: vec![],
            z_index: None,
            collapsed_borders: None,
            phantom_w: false,
            phantom_h: false,
        }));
        tab.dirty.layout = false;
        tab.scroll().set_max_offset((0.0, 400.0));

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/fragscripts#frag"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        assert_eq!(hit_count(&hits), 1, "fragment nav must not refetch");
        assert_eq!(tab.scroll().offset(), (0.0, 321.0));
        assert_eq!(tab.history.len(), 2);

        go_back(&mut tab, &jar, UA, &Arc::new(Mutex::new(FontCache::new()))).unwrap();
        assert_eq!(
            hit_count(&hits),
            1,
            "back to a same-document entry must not refetch"
        );
        assert_eq!(tab.current_url().expect("history").fragment(), None);
        assert_eq!(tab.scroll().offset(), (0.0, 0.0), "back scrolls to the top");
        // The JS realm survived the traversal: the script did not re-run
        // and its global is intact.
        let report = tab.script_report().expect("page VM kept");
        assert_eq!(report.executed, 1, "scripts must not re-run on back");
        let marker = tab
            .page_vm_mut()
            .expect("page VM kept")
            .eval("marker")
            .expect("marker");
        assert_eq!(marker, Value::Int(7));

        go_forward(&mut tab, &jar, UA, &Arc::new(Mutex::new(FontCache::new()))).unwrap();
        assert_eq!(
            hit_count(&hits),
            1,
            "forward to a same-document entry must not refetch"
        );
        assert_eq!(tab.current_url().expect("history").fragment(), Some("frag"));
        assert_eq!(
            tab.scroll().offset(),
            (0.0, 321.0),
            "forward scrolls to the anchor"
        );
        let report = tab.script_report().expect("page VM kept");
        assert_eq!(report.executed, 1, "scripts must not re-run on forward");
    }

    #[test]
    fn failed_navigation_leaves_tab_untouched() {
        let (base, _hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar();

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        let err = navigate(
            &mut tab,
            &jar,
            &format!("{base}/missing"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap_err();
        assert!(
            matches!(err, NavError::HttpStatus(404)),
            "unexpected error: {err:?}"
        );

        assert_eq!(tab.title, "Page One");
        assert_eq!(tab.current_url().expect("history").path(), "/page");
        assert_eq!(tab.history.len(), 1);
        assert!(tab.has_document());
    }

    #[test]
    fn invalid_url_is_url_error() {
        let mut tab = test_tab();
        let jar = test_jar();

        let err = navigate(
            &mut tab,
            &jar,
            "://bad",
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap_err();
        assert!(matches!(err, NavError::Url(_)), "unexpected error: {err:?}");
        assert!(tab.history.is_empty());
        assert!(!tab.has_document());
    }

    #[test]
    fn navigate_to_about_blank_installs_welcome_page_without_fetch() {
        // The blocking path used to hand `about:blank` to the loader,
        // which would attempt a fetch and fail. It must install the
        // built-in welcome page with no network involved, exactly like
        // the windowed and headless initial-load paths.
        let mut tab = test_tab();
        let jar = test_jar();

        navigate(
            &mut tab,
            &jar,
            "about:blank",
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .expect("about:blank navigate");

        assert_eq!(tab.title, "welcome to bowser");
        assert_eq!(
            tab.current_url().expect("history").serialize(),
            "about:blank"
        );
        assert_eq!(tab.history.len(), 1);
        assert!(tab.has_document());
        assert!(tab.dirty.style && tab.dirty.layout && tab.dirty.paint);
        // No load ever ran: the profile jar was never swapped out.
        assert!(jar.lock().expect("jar").is_empty());
    }

    #[test]
    fn navigate_to_about_blank_from_a_loaded_page() {
        let (base, _hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar();

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        assert_eq!(tab.title, "Page One");

        navigate(
            &mut tab,
            &jar,
            "about:blank",
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .expect("about:blank navigate");

        assert_eq!(tab.title, "welcome to bowser");
        assert_eq!(
            tab.current_url().expect("history").serialize(),
            "about:blank"
        );
        assert_eq!(tab.history.len(), 2);
    }

    #[test]
    fn mark_all_dirty_sets_all_flags() {
        let mut tab = test_tab();
        tab.dirty = DirtyFlags::default();
        mark_all_dirty(&mut tab);
        assert!(tab.dirty.style && tab.dirty.layout && tab.dirty.paint);
    }

    #[test]
    fn full_navigation_runs_page_scripts_in_a_fresh_realm() {
        let (base, _hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar();

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/scripts"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .expect("navigate");

        // Inline and external scripts shared one realm, in document order:
        // 1, then +41 (external), then +1, then the throw, then +10.
        let report = tab.script_report().expect("page VM installed");
        assert_eq!(report.executed, 5);
        assert_eq!(report.skipped, 0);
        assert_eq!(
            report.errors.len(),
            1,
            "the throw is recorded, not fatal: {report:?}"
        );
        let counter = tab
            .page_vm_mut()
            .expect("page VM installed")
            .eval("counter")
            .expect("counter");
        assert_eq!(counter, Value::Int(53));

        // A script failure never fails the navigation.
        assert_eq!(tab.title, "Scripts");
        assert!(tab.has_document());

        // Same-document fragment navigation keeps the realm (and its
        // globals): no new PageVm, no re-run.
        navigate(
            &mut tab,
            &jar,
            &format!("{base}/scripts#frag"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .unwrap();
        let report = tab.script_report().expect("page VM kept");
        assert_eq!(
            report.executed, 5,
            "scripts must not re-run on fragment nav"
        );
        let counter = tab
            .page_vm_mut()
            .expect("page VM kept")
            .eval("counter += 1000; counter")
            .expect("counter");
        assert_eq!(counter, Value::Int(1053));

        // A new full navigation replaces the realm: the old globals are
        // gone and the new document's scripts run fresh.
        navigate(
            &mut tab,
            &jar,
            &format!("{base}/scripts2"),
            UA,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .expect("navigate");
        let report = tab.script_report().expect("page VM installed");
        assert_eq!(report.executed, 1);
        assert!(report.errors.is_empty());
        let vm = tab.page_vm_mut().expect("page VM installed");
        let counter = vm.eval("counter").expect("counter");
        assert_eq!(counter, Value::Int(100));
    }

    /// Every failure path in [`load_document_blocking`] (deadline timeout,
    /// dead worker, panic before the restore) funnels through the
    /// [`ProfileJarGuard`]'s drop without a commit: the profile must keep
    /// its pre-load cookies instead of the empty swap jar.
    #[test]
    fn jar_guard_drop_without_commit_restores_preload_jar() {
        let jar = test_jar_with_cookie();
        {
            let _guard = ProfileJarGuard::take(&jar);
            // While the load is outstanding the profile holds the fresh
            // swap jar.
            assert!(jar.lock().expect("jar").is_empty());
            // ...the load fails here; the guard drops without commit...
        }
        assert_eq!(
            jar.lock().expect("jar").len(),
            1,
            "dropped guard did not restore the pre-load cookie jar"
        );
    }

    /// A panic anywhere between the take and the commit (e.g. a thread
    /// spawn failure, or any bug in the navigation caller) unwinds through
    /// the guard: the jar must survive the unwind.
    #[test]
    fn jar_guard_restores_jar_when_caller_panics() {
        let jar = test_jar_with_cookie();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = ProfileJarGuard::take(&jar);
            panic!("simulated navigation panic");
        }));
        assert!(result.is_err());
        assert_eq!(
            jar.lock().expect("jar").len(),
            1,
            "panic in navigation wiped the profile cookie jar"
        );
    }

    /// The finished-load path commits the worker's jar (carrying the
    /// load's cookie updates); the disarmed drop must not overwrite it
    /// with the stale pre-load jar.
    #[test]
    fn jar_guard_commit_disarms_drop_restore() {
        let jar = test_jar_with_cookie();
        {
            let mut guard = ProfileJarGuard::take(&jar);
            let mut worker_jar = guard.worker_jar();
            worker_jar.set_cookie(
                "fresh=1; Path=/",
                &RequestContext {
                    host: "127.0.0.1",
                    path: "/",
                    secure: false,
                    http_api: true,
                    top_host: None,
                    top_secure: None,
                    top_level_safe_navigation: false,
                },
                1_700_000_000,
            );
            guard.commit(worker_jar);
        }
        assert_eq!(
            jar.lock().expect("jar").len(),
            2,
            "committed worker jar was clobbered by the guard's drop"
        );
    }

    /// Canned server recording the `User-Agent` header of every request.
    fn spawn_ua_server() -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("test server addr");
        let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_srv = Arc::clone(&seen);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = match stream {
                    Ok(s) => s,
                    Err(_) => break,
                };
                let seen = Arc::clone(&seen_srv);
                std::thread::spawn(move || {
                    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 4096];
                    loop {
                        match stream.read(&mut chunk) {
                            Ok(0) => break,
                            Ok(n) => {
                                buf.extend_from_slice(&chunk[..n]);
                                if buf.windows(4).any(|w| w == b"\r\n\r\n") || buf.len() > 65_536 {
                                    break;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                    let head = String::from_utf8_lossy(&buf);
                    let ua = head
                        .lines()
                        .find_map(|line| line.strip_prefix("User-Agent:"))
                        .map(|v| v.trim().to_owned())
                        .unwrap_or_default();
                    seen.lock().expect("ua seen").push(ua);
                    let body = PAGE_ONE;
                    let head = format!(
                        "HTTP/1.1 200 OK\r\n\
                         Content-Type: text/html; charset=utf-8\r\n\
                         Content-Length: {}\r\n\
                         Connection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(head.as_bytes());
                    let _ = stream.write_all(body.as_bytes());
                });
            }
        });
        (format!("http://{addr}"), seen)
    }

    #[test]
    fn navigate_sends_configured_user_agent_to_network_and_navigator() {
        let (base, seen) = spawn_ua_server();
        let mut tab = test_tab();
        let jar = test_jar();
        let ua = "CustomBrowser/1.2.3";

        navigate(
            &mut tab,
            &jar,
            &format!("{base}/page"),
            ua,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .expect("navigate");

        // Network: every request the load made carried the configured UA.
        let uas = seen.lock().expect("ua seen").clone();
        assert!(!uas.is_empty(), "server saw no requests");
        assert!(
            uas.iter().all(|u| u == ua),
            "every request must carry the configured UA, saw: {uas:?}"
        );
        // navigator.userAgent: seeded from the same UA at install.
        let reported = tab
            .page_vm_mut()
            .expect("page vm installed")
            .eval("navigator.userAgent === 'CustomBrowser/1.2.3'")
            .expect("read navigator.userAgent");
        assert!(
            matches!(reported, Value::Bool(true)),
            "navigator.userAgent must reflect the configured UA"
        );
    }

    #[test]
    fn navigate_to_same_url_reloads_in_place_without_duplicate_history() {
        let (base, hits) = spawn_server();
        let mut tab = test_tab();
        let jar = test_jar();
        let ua = bowser_net::fetch::USER_AGENT;
        let url = format!("{base}/page");

        navigate(
            &mut tab,
            &jar,
            &url,
            ua,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .expect("first navigate");
        assert_eq!(tab.history.len(), 1);
        let hits_before = hit_count(&hits);

        navigate(
            &mut tab,
            &jar,
            &url,
            ua,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .expect("same-url navigate");

        assert_eq!(
            tab.history.len(),
            1,
            "navigating to the exact current URL must not push a duplicate entry"
        );
        assert!(
            hit_count(&hits) > hits_before,
            "same-URL navigation must reload (refetch) in place"
        );
        assert_eq!(tab.title, "Page One", "reloaded document installed");
    }

    #[test]
    fn reload_about_blank_reinstalls_welcome_page_cleanly() {
        let mut tab = test_tab();
        let jar = test_jar();
        let ua = bowser_net::fetch::USER_AGENT;

        navigate(
            &mut tab,
            &jar,
            "about:blank",
            ua,
            &Arc::new(Mutex::new(FontCache::new())),
        )
        .expect("navigate about:blank");
        assert_eq!(tab.history.len(), 1);
        // Prove the reload actually reinstalls: drop the document first.
        tab.uninstall_test_document();

        reload(&mut tab, &jar, ua, &Arc::new(Mutex::new(FontCache::new())))
            .expect("reload of about:blank must succeed");

        assert!(tab.has_document(), "reload must reinstall the welcome page");
        assert_eq!(tab.history.len(), 1, "reload must not touch history");
        assert_eq!(
            tab.current_url().map(|u| u.serialize()),
            Some("about:blank".to_string())
        );
    }
}

#[cfg(test)]
mod shared_dom_tests {
    use super::*;
    use crate::tab::DirtyFlags;
    use bowser_js::Value;
    use bowser_net::fetch::FetchTrace;

    fn test_tab() -> Tab {
        Tab::new(bowser_layout::Size { w: 800.0, h: 600.0 })
    }

    /// A [`LoadedDocument`] straight from HTML, with external scripts
    /// prefetched as empty sources (no network): the unit-test stand-in
    /// for a finished load.
    fn load_from(html: &[u8]) -> LoadedDocument {
        let url = bowser_url::parse("https://example.com/", None).unwrap();
        load_from_base(html, &url)
    }

    fn load_from_base(html: &[u8], url: &Url) -> LoadedDocument {
        use crate::script::collect_script_refs;
        use crate::script_executor::prefetch_script_sources;

        let url = url.clone();
        let (document, quirks) = bowser_dom::Document::parse_html(html);
        // w77 quirks plumbing: the mode travels on the document.
        debug_assert_eq!(document.quirks_mode(), quirks);
        let script_sources =
            prefetch_script_sources(&collect_script_refs(&document), &url, |_| Ok(String::new()));
        LoadedDocument {
            url: url.clone(),
            trace: FetchTrace { hops: vec![url] },
            document,
            stylesheets: Vec::new(),
            image_sizes: std::collections::HashMap::new(),
            script_sources,
            body: html.to_vec(),
            webfonts: Vec::new(),
        }
    }

    #[test]
    fn midparse_scroll_write_rebakes_sticky_for_geometry_reads() {
        // WPT css-position/sticky/position-sticky-nested-top.html runs
        // mid-parse: each subtest writes `scroller.scrollTop`, then
        // reads `offsetTop`. The shim must invalidate its partial
        // layout on the scroll write and bake sticky constraints at
        // the written offset, or every read sees the pre-scroll bake.
        let html = br#"<!DOCTYPE html><html><head></head><body><script>
var scroller = document.createElement('div');
scroller.style.position = 'relative';
scroller.style.width = '100px';
scroller.style.height = '200px';
scroller.style.overflow = 'hidden';
var contents = document.createElement('div');
contents.style.height = '500px';
contents.style.width = '100%';
var prepadding = document.createElement('div');
prepadding.style.height = '100px';
prepadding.style.width = '100%';
var container = document.createElement('div');
container.style.height = '300px';
container.style.width = '100%';
var filler = document.createElement('div');
filler.style.height = '100px';
filler.style.width = '100%';
var sticky = document.createElement('div');
sticky.style.top = '50px';
sticky.style.position = 'sticky';
sticky.style.height = '100px';
sticky.style.width = '100%';
var innerSticky = document.createElement('div');
innerSticky.style.top = '60px';
innerSticky.style.position = 'sticky';
innerSticky.style.height = '50px';
innerSticky.style.width = '100%';
scroller.appendChild(contents);
contents.appendChild(prepadding);
contents.appendChild(container);
container.appendChild(filler);
container.appendChild(sticky);
sticky.appendChild(innerSticky);
document.body.appendChild(scroller);
var results = [];
function check(scroll) {
  scroller.scrollTop = scroll;
  results.push(sticky.offsetTop + ',' + innerSticky.offsetTop);
}
check(100);
check(145);
check(200);
check(300);
window.__r = results.join(' | ');
</script></body></html>"#;
        let mut tab = test_tab();
        install_loaded_document(
            &mut tab,
            load_from(html),
            "TestBrowser/9.9",
            &Arc::new(Mutex::new(FontCache::new())),
            &Arc::new(Mutex::new(CookieJar::new())),
        );
        let r = tab
            .page_vm_mut()
            .expect("page VM installed")
            .eval("window.__r")
            .expect("eval __r");
        // (outer offsetTop, inner offsetTop) per scroll; the WPT
        // expectations for position-sticky-nested-top.html.
        let got = match r {
            Value::Str(s) => s.as_wtf8().to_utf8_lossy().into_owned(),
            v => panic!("expected string, got {v:?}"),
        };
        assert_eq!(
            got, "200,0 | 200,5 | 250,10 | 300,50",
            "sticky must re-bake at each written scroll offset"
        );
    }

    #[test]
    fn install_carries_quirks_mode_to_js_style() {
        // w77 quirks plumbing, production path: a full-quirks document
        // installed through `install_loaded_document` must keep its
        // mode on the tab's document, and JS must observe the quirk:
        // the italic wrapper's font-style must not reach the cell
        // (§3.11 probe). (Geometry is covered by the layout-crate
        // quirks tests; a post-install eval has no layout pump, so
        // offsetHeight reads 0 here regardless of mode.)
        let html = br#"<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN"><html><head></head><body><div style="font-style:italic"><table><tr><td id="notitalic">x</td></tr></table></div></body></html>"#;
        let mut tab = test_tab();
        install_loaded_document(
            &mut tab,
            load_from(html),
            "TestBrowser/9.9",
            &Arc::new(Mutex::new(FontCache::new())),
            &Arc::new(Mutex::new(CookieJar::new())),
        );
        let mode = tab.document_ref().expect("document").quirks_mode();
        assert_eq!(mode, bowser_dom::QuirksMode::Quirks);
        // Rust-side pin: the installed document's own computed style
        // for the cell must already be `normal` before JS is involved.
        {
            let doc_ref = tab.document_ref().expect("document");
            let td = doc_ref.query_selector("#notitalic").unwrap().expect("td");
            let computed = bowser_css::compute_style(
                &doc_ref,
                td,
                &[],
                &bowser_css::default_ua_stylesheet(),
                &bowser_css::Viewport { w: 800.0, h: 600.0 },
            );
            assert_eq!(
                computed.get_str("font-style"),
                Some("normal"),
                "rust-side computed font-style"
            );
        }
        let fs = tab
            .page_vm_mut()
            .expect("page VM installed")
            .eval("window.getComputedStyle(document.querySelector('#notitalic')).fontStyle")
            .expect("eval fontStyle");
        match fs {
            Value::Str(s) => assert_eq!(
                s.as_wtf8().to_utf8_lossy().as_ref(),
                "normal",
                "font-style via JS"
            ),
            other => panic!("expected string font-style, got {other:?}"),
        }
    }

    #[test]
    fn install_slot_forces_sync_layout_for_postparse_geometry_reads() {
        // A deferred script runs in the post-parse phase of
        // `install_loaded_document`, with the production web APIs
        // installed but before any pump: its geometry read must force
        // a synchronous reflow through the initial-load
        // `LayoutEnsurer` slot instead of answering 0 (the no-layout
        // answer a pump-time getter would never see).
        let html = br#"<!DOCTYPE html><html><head></head><body><script defer>window.__w = document.documentElement.offsetWidth;</script></body></html>"#;
        let mut tab = test_tab();
        install_loaded_document(
            &mut tab,
            load_from(html),
            "TestBrowser/9.9",
            &Arc::new(Mutex::new(FontCache::new())),
            &Arc::new(Mutex::new(CookieJar::new())),
        );
        let w = tab
            .page_vm_mut()
            .expect("page VM installed")
            .eval("window.__w")
            .expect("eval __w");
        assert_eq!(
            w,
            Value::Int(800),
            "deferred script must observe laid-out geometry, got {w:?}"
        );
    }

    #[test]
    fn install_shares_one_document_between_tab_and_bindings() {
        // The page's script mutates the DOM through the mid-parse
        // document API; the mutation must be visible through the tab's
        // document — the renderer's view — because both sides share one
        // tree (no clone).
        let mut tab = test_tab();
        install_loaded_document(
            &mut tab,
            load_from(
                b"<html><head><script>document.write('<p id=\"js-added\">hi</p>')</script></head><body></body></html>",
            ),
            "TestBrowser/9.9",
            &Arc::new(Mutex::new(FontCache::new())),
            &Arc::new(Mutex::new(CookieJar::new())),
        );
        let seen = tab
            .document_ref()
            .expect("document")
            .get_element_by_id("js-added");
        assert!(seen.is_some(), "JS DOM mutation visible to renderer");
        // And the bindings still see it too: one shared tree.
        let found = tab
            .page_vm_mut()
            .expect("page VM installed")
            .eval("document.getElementById('js-added') !== null")
            .expect("eval");
        assert_eq!(found, Value::Bool(true));
    }

    #[test]
    fn install_sets_native_stack_guard_config() {
        // Configuration pin: installing a document wires the computed
        // page-realm guard (`page_realm_native_stack_guard`) through to
        // the VM — nothing more. This does NOT validate the thread-size
        // invariant, and cannot: it installs on a 2 MiB test thread,
        // where the raised guard is simply inert (no test recurses
        // anywhere near either limit). The invariant — the guard sitting
        // below the installing thread's real stack end — is enforced by
        // the `RLIMIT_STACK` probe inside `page_realm_native_stack_guard`,
        // not by this test.
        let mut tab = test_tab();
        install_loaded_document(
            &mut tab,
            load_from(b"<html><head></head><body><p>hi</p></body></html>"),
            "TestBrowser/9.9",
            &Arc::new(Mutex::new(FontCache::new())),
            &Arc::new(Mutex::new(CookieJar::new())),
        );
        let guard = tab
            .page_vm()
            .expect("page VM installed")
            .vm()
            .max_native_stack();
        assert_eq!(
            guard,
            page_realm_native_stack_guard(),
            "install applies the computed page-realm guard"
        );
    }

    #[test]
    fn native_stack_guard_clamps_below_probed_limit() {
        // The Finding 1 invariant, measured: the installed guard plus the
        // 1 MiB unwind headroom must fit inside the probed main-thread
        // stack limit, so the guard always trips before the real stack
        // end. (A sub-headroom limit clamps the guard to zero — fail
        // closed, still no abort.)
        let guard = page_realm_native_stack_guard();
        match main_thread_stack_limit() {
            Some(limit) => {
                assert_eq!(
                    guard,
                    (7 << 20).min(limit.saturating_sub(1 << 20)),
                    "guard is min(7 MiB, rlimit - headroom)"
                );
                assert!(
                    limit < (1 << 20) || guard + (1 << 20) <= limit,
                    "guard {guard} + headroom exceeds probed limit {limit}"
                );
            }
            None => assert_eq!(
                guard,
                1 << 20,
                "unprobed platform: conservative 1 MiB fallback"
            ),
        }
    }

    #[test]
    fn deep_mid_parse_recursion_runs_clean_under_raised_guard() {
        // Deep-but-finite recursion in a parser-inserted script — the
        // shape that failed every testharness.js page before the fix
        // (the script evaluates mid-parse, below the guard anchor).
        // Runs on an 8 MiB thread like the production `--check` worker.
        //
        // Wave 39 made the old peak premise stale, and the failure was
        // NOT a readyState regression: the VM's call trampoline pushes
        // sync call frames onto the dispatch loop's frame stack instead
        // of nesting a native `Interp::run` per call, so deep JS
        // recursion no longer consumes native stack per frame (see
        // `deep_recursion_is_a_clean_stack_error` in bowser_js). The
        // ~85 KiB-per-call peak this test used to aim at three quarters
        // of the guard is gone — measured ~89 KiB total now — and with
        // it the "peak must clear the old 1 MiB default" proof. What
        // remains is the behavioral core: the recursion runs clean
        // mid-parse on a production-sized thread. The peak assertion
        // pins the trampoline instead: even deep recursion must stay far
        // below the old 1 MiB default.
        let guard = page_realm_native_stack_guard();
        // The install still wires the raised guard (see
        // `install_sets_native_stack_guard_config`); on an unprobed
        // platform, or under a drastically shrunk `ulimit -s`, there is
        // no raised guard to run under — skip.
        if guard <= 2 << 20 {
            return;
        }
        let depth = 60;
        let html = format!(
            "<html><head><script>\
            function f(n){{if(n<=0)return 0;return f(n-1)+1;}}\
            f({depth});\
            </script></head><body></body></html>"
        );
        let (errors, high_water) = std::thread::Builder::new()
            .stack_size(8 << 20)
            .spawn(move || {
                let mut tab = test_tab();
                install_loaded_document(
                    &mut tab,
                    load_from(html.as_bytes()),
                    "TestBrowser/9.9",
                    &Arc::new(Mutex::new(FontCache::new())),
                    &Arc::new(Mutex::new(CookieJar::new())),
                );
                let errors = tab
                    .script_report()
                    .map(|report| report.errors.len())
                    .unwrap_or(usize::MAX);
                let high_water = tab
                    .page_vm()
                    .expect("page VM installed")
                    .vm()
                    .native_stack_high_water();
                (errors, high_water)
            })
            .expect("spawn deep-recursion thread")
            .join()
            .expect("deep-recursion thread survived");
        assert_eq!(errors, 0, "no script errors under the raised guard");
        assert!(
            high_water < 1024 * 1024,
            "peak {high_water} should stay under the old 1 MiB guard: trampolined calls consume ~no native stack per frame",
        );
    }

    #[test]
    fn script_mutation_dirt_reaches_render_flags() {
        let mut tab = test_tab();
        install_loaded_document(
            &mut tab,
            load_from(b"<html><head></head><body><p>hi</p></body></html>"),
            "TestBrowser/9.9",
            &Arc::new(Mutex::new(FontCache::new())),
            &Arc::new(Mutex::new(CookieJar::new())),
        );
        // Navigation dirties every stage; start the measurement clean.
        tab.dirty = DirtyFlags::default();

        // A later script (event handler, timer callback) mutates the DOM
        // through the bindings; folding the document's dirt must request
        // exactly the affected stages.
        {
            let vm = tab.page_vm_mut().expect("page VM installed");
            vm.eval("document.body.setAttribute('class', 'dark');")
                .expect("eval");
        }
        tab.sync_dirty_flags();
        assert!(
            tab.dirty.style && tab.dirty.layout && tab.dirty.paint,
            "class change can affect matching: all stages re-run"
        );

        // Any attribute can take part in selector matching
        // (`[data-x]`, ...), so a `data-*` write re-runs the style stage
        // too.
        tab.dirty = DirtyFlags::default();
        {
            let vm = tab.page_vm_mut().expect("page VM installed");
            vm.eval("document.body.setAttribute('data-x', '1');")
                .expect("eval");
        }
        tab.sync_dirty_flags();
        assert!(
            tab.dirty.style && tab.dirty.layout && tab.dirty.paint,
            "attribute selectors can match data-*: all stages re-run"
        );

        // A text-only edit through the tab's own handle (the same `Rc`
        // the bindings hold) skips the style stage: no selector matches
        // on character data. Scoped access: no strong handle escapes.
        tab.dirty = DirtyFlags::default();
        tab.installed()
            .expect("document")
            .with_document(|cell| {
                let text = {
                    let doc = cell.borrow();
                    let p = doc.get_elements_by_tag_name("p").item(0).expect("p");
                    doc.first_child(p).expect("text child")
                };
                cell.borrow_mut().set_text_content(text, "bye");
            })
            .expect("tab owns document");
        tab.sync_dirty_flags();
        assert!(
            !tab.dirty.style,
            "text-only edit must not re-run the style stage"
        );
        assert!(tab.dirty.layout && tab.dirty.paint);
    }

    #[test]
    fn install_shares_profile_jar_with_js_fontface_load() {
        // w77-L: production wiring for JS `FontFace.load()` credentials.
        // A local font server records the `Cookie` header of the
        // same-origin font fetch a page's `FontFace.load()` performs
        // after `install_loaded_document`. The profile jar is seeded
        // with a cookie for 127.0.0.1, so the header must carry it —
        // proving the install shared the profile's live jar with the
        // JS context, not a snapshot and not nothing. (A snapshot
        // copy would also send the seeded cookie here, so the
        // strong-count pin below is the identity half: the JS
        // context must retain this exact `Arc`, and a later
        // `Set-Cookie` from a JS load must land in this same jar —
        // both follow from sharing the handle, not a copy.)
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind font server");
        let addr = listener.local_addr().expect("font server addr");
        let seen: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen_server = Arc::clone(&seen);
        let ahem: Vec<u8> = include_bytes!("../../bowser_paint/tests/fixtures/Ahem.ttf").to_vec();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                loop {
                    match stream.read(&mut chunk) {
                        Ok(0) => break,
                        Ok(n) => {
                            buf.extend_from_slice(&chunk[..n]);
                            if buf.windows(4).any(|w| w == b"\r\n\r\n") || buf.len() > 65_536 {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
                let head = String::from_utf8_lossy(&buf);
                let cookie = head
                    .lines()
                    .find_map(|line| line.strip_prefix("Cookie:"))
                    .map(str::trim)
                    .unwrap_or("(none)")
                    .to_string();
                seen_server.lock().expect("seen").push(cookie);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: font/ttf\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    ahem.len()
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.write_all(&ahem);
            }
        });

        let base = bowser_url::parse(&format!("http://{addr}/"), None).unwrap();
        // Seeded with `session=abc123` for 127.0.0.1 (same seeding
        // as `tests::test_jar_with_cookie`, inlined: this test lives
        // in `shared_dom_tests`, which cannot see that helper).
        let jar: Arc<Mutex<CookieJar>> = Arc::new(Mutex::new(CookieJar::new()));
        jar.lock().expect("test jar").set_cookie(
            "session=abc123; Path=/",
            &bowser_net::cookies::RequestContext {
                host: "127.0.0.1",
                path: "/",
                secure: false,
                http_api: true,
                top_host: None,
                top_secure: None,
                top_level_safe_navigation: false,
            },
            0,
        );
        assert_eq!(jar.lock().expect("test jar").len(), 1);
        assert_eq!(Arc::strong_count(&jar), 1);
        let mut tab = test_tab();
        install_loaded_document(
            &mut tab,
            load_from_base(b"<html><head></head><body></body></html>", &base),
            "TestBrowser/9.9",
            &Arc::new(Mutex::new(FontCache::new())),
            &jar,
        );
        // Identity half: the JS context retained this exact `Arc`
        // (a copy would leave the count at 1).
        assert!(
            Arc::strong_count(&jar) >= 2,
            "install must share the exact profile jar Arc with the JS context"
        );
        tab.page_vm_mut()
            .expect("page VM installed")
            .eval(
                "window.__face = new FontFace('Ahem', 'url(/font.ttf)'); \
                 document.fonts.add(window.__face); \
                 window.__face.load().then(function() { window.__loaded = true; }, \
                 function(e) { window.__err = String(e); });",
            )
            .expect("start FontFace.load");
        // Pump the page event loop until the server sees the fetch
        // (the pool worker fetches asynchronously; completions arrive
        // through the pump). Bounded: a missing fetch fails the test
        // instead of hanging it.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while seen.lock().expect("seen").is_empty() && std::time::Instant::now() < deadline {
            tab.page_vm_mut()
                .expect("page VM installed")
                .pump_bounded(1000);
            std::thread::sleep(Duration::from_millis(20));
        }
        let headers = seen.lock().expect("seen").clone();
        let status = tab
            .page_vm_mut()
            .expect("page VM installed")
            .eval("[window.__loaded, window.__err].join('|')")
            .ok();
        assert!(
            headers.iter().any(|h| h.contains("session=abc123")),
            "same-origin FontFace.load() must send the profile cookie; got headers {headers:?}, status {status:?}"
        );
    }
}
