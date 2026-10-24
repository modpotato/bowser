//! Production [`ScriptExecutor`] for parser-inserted scripts.
//!
//! Mid-parse, the tree builder hands every parser-inserted script to
//! [`ParserExecutor`], which evaluates the parser-eligible classic
//! scripts immediately — in document order, through the page's own
//! [`PageVm`] — so `document.write` and mid-parse DOM reads see the
//! tree under construction (WHATWG HTML §13.2.6). `async` and `defer`
//! scripts, unresolvable `src`s, and sources the loader did not prefetch
//! decline execution; they run through the post-parse path instead
//! ([`PageVm::run`] executes the `async` batch after the classic stream
//! and the deferred scripts after that).
//! [`filter_handled_scripts`] reconciles the post-parse
//! [`collect_script_refs`](crate::script::collect_script_refs) pass with
//! what the parser already ran, by fingerprint, so nothing runs twice.

use std::collections::HashMap;

use bowser_dom::{ScriptElement, ScriptExecutor, ScriptHost};
use bowser_url::Url;

use crate::script::{
    MAX_SCRIPTS, PageVm, ScriptError, ScriptErrorKind, ScriptRef, is_executable_script_type,
};

/// What the executor decided for one parser-inserted script.
#[derive(Debug)]
enum Handling {
    /// Classic inline script: evaluate its text now.
    Inline,
    /// Classic external script with a prefetched source: evaluate the
    /// fetched source now, in document order. Boxed: the payload dwarfs
    /// the other variants.
    External(Box<ExternalScript>),
    /// Leave it for the post-parse path: `async` (the loader fired its
    /// fetch concurrently; the post-parse path runs the `async` batch
    /// after the classic stream, order-independent), `defer` (the
    /// engine's scheduling business, per the [`ScriptExecutor`]
    /// contract — the post-parse path runs deferred scripts in its third
    /// pass), an unresolvable `src`, or a source the loader did not
    /// prefetch (a failed or over-cap fetch, or markup a
    /// `document.write` added — the post-parse path records the honest
    /// skip).
    Decline,
}

/// A prefetched external script the executor will evaluate mid-parse.
#[derive(Debug)]
struct ExternalScript {
    url: Url,
    source: String,
}

/// Classify one parser-inserted script. Attribute names are already
/// lowercased by the tokenizer.
fn classify(
    attrs: &[(String, String)],
    base_url: &Url,
    prefetch: &HashMap<String, Result<String, String>>,
) -> Handling {
    let mut src: Option<&str> = None;
    let mut has_defer = false;
    let mut has_async = false;
    for (name, value) in attrs {
        match name.as_str() {
            "src" => src = Some(value),
            "defer" => has_defer = true,
            "async" => has_async = true,
            _ => {}
        }
    }
    let Some(src) = src else {
        // No `src`: `async`/`defer` are ignored per spec — a plain inline
        // classic script, evaluated mid-parse in document order.
        return Handling::Inline;
    };
    if has_async || has_defer {
        // `async` scripts never run mid-parse: the loader fired their
        // fetch concurrently and the post-parse path executes them as an
        // order-independent batch after the classic stream (before the
        // deferred pass). `defer` scripts are the post-parse path's
        // scheduling business per the `ScriptExecutor` contract.
        return Handling::Decline;
    }
    let url = match bowser_url::parse(src.trim(), Some(base_url)) {
        Ok(url) => url,
        Err(_) => return Handling::Decline,
    };
    match prefetch.get(&url.serialize()) {
        Some(Ok(source)) => Handling::External(Box::new(ExternalScript {
            url,
            source: source.clone(),
        })),
        // Failed fetch, over-cap skip, or a `document.write`-added
        // script the loader never saw: the post-parse path records the
        // skip against the element.
        _ => Handling::Decline,
    }
}

/// Fingerprint of a script the executor ran mid-parse. Used to match the
/// post-parse [`collect_script_refs`](crate::script::collect_script_refs)
/// pass against what already ran, so each ref is attributed to at most
/// one execution — even when the page repeats the same script twice, or
/// mixes `src`-identical sync/`async`/`defer` variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandledScript {
    /// Raw `src` attribute value, if present.
    pub src: Option<String>,
    /// The inline source as executed (empty for external scripts).
    pub text: String,
    /// Raw `type` attribute value, if present.
    pub script_type: Option<String>,
    /// Whether the `defer` attribute was present.
    pub defer: bool,
    /// Whether the `async` attribute was present. An `async` twin of an
    /// executed script is a different script: it was declined mid-parse
    /// and runs in the post-parse `async` batch.
    pub is_async: bool,
}

impl HandledScript {
    fn from_attrs(attrs: &[(String, String)], text: &str) -> Self {
        let mut script = HandledScript {
            src: None,
            text: text.to_string(),
            script_type: None,
            defer: false,
            is_async: false,
        };
        for (name, value) in attrs {
            match name.as_str() {
                "src" => script.src = Some(value.clone()),
                "type" => script.script_type = Some(value.clone()),
                "defer" => script.defer = true,
                "async" => script.is_async = true,
                _ => {}
            }
        }
        script
    }

    /// Whether this fingerprint identifies the collected ref. The
    /// executor only ever runs parser-eligible scripts, so a
    /// fingerprint match is proof the ref already ran — the matching
    /// does not need to re-derive the parser's eligibility rules.
    fn matches_ref(&self, r: &ScriptRef) -> bool {
        self.src == r.src
            && self.text == r.inline_text
            && self.script_type == r.script_type
            && self.defer == r.defer
            && self.is_async == r.is_async
    }
}

/// One executed script plus its mid-parse failure, if it threw.
struct HandledEntry {
    script: HandledScript,
    error: Option<ScriptFailure>,
}

/// A mid-parse evaluation failure: how it failed, and the source URL
/// (`Some` for external scripts).
struct ScriptFailure {
    kind: ScriptErrorKind,
    url: Option<Url>,
}

/// Outcome of the scripted parse: what ran, in execution (= document)
/// order, and the mid-parse failures aligned with it.
#[derive(Debug, Default)]
pub struct ParseOutcome {
    /// Fingerprints of the executed scripts, in execution order.
    pub handled: Vec<HandledScript>,
    /// Mid-parse evaluation failures as `(handled index, failure kind,
    /// source url)`, in execution order. Attributed to elements by
    /// [`filter_handled_scripts`], which has the post-parse [`ScriptRef`]s.
    pub errors: Vec<(usize, ScriptErrorKind, Option<Url>)>,
    /// Partial layout passes the parse-phase layout hook ran per
    /// executed script, in execution order (aligned with `handled`):
    /// [`bowser_js::ParseShimGuard::layout_passes`] read before the
    /// guard uninstalls. Test instrumentation for the on-demand
    /// mid-parse layout wiring — 0 for scripts that never read
    /// geometry, and for scripts whose shim install failed.
    pub layout_passes: Vec<u64>,
}

/// The production [`ScriptExecutor`]: runs parser-eligible classic
/// scripts mid-parse through the page's [`PageVm`], with the
/// parse-phase `document` shim (mid-parse DOM queries/mutation plus
/// `document.write` into the tokenizer's input stream) installed for
/// each script.
///
/// The `Vm` is `!Send` and bound to its creating thread, so this runs
/// on the installing thread — not in the loader. The loader prefetches
/// external sources into `prefetch` (keyed by serialized URL, mirroring
/// [`resolve_scripts`]' gating) so no network happens here.
pub struct ParserExecutor<'a> {
    page_vm: &'a mut PageVm,
    base_url: Url,
    prefetch: &'a HashMap<String, Result<String, String>>,
    outcome: ParseOutcome,
    /// Embedder viewport in CSS px, threaded into the parse-phase
    /// `document` shim so mid-parse partial layouts agree with
    /// post-parse layout; `None` keeps the shim's 800x600 fallback.
    viewport: Option<(u32, u32)>,
}

impl<'a> ParserExecutor<'a> {
    /// Build the executor around the page realm. `base_url` is the
    /// document base the loader used when prefetching `prefetch`, so
    /// `src` resolution here hits the same URLs the loader fetched.
    /// `viewport` is the embedder's viewport (e.g. the tab's), or
    /// `None` for the shim's 800x600 fallback.
    pub fn new(
        page_vm: &'a mut PageVm,
        base_url: &Url,
        prefetch: &'a HashMap<String, Result<String, String>>,
        viewport: Option<(u32, u32)>,
    ) -> Self {
        ParserExecutor {
            page_vm,
            base_url: base_url.clone(),
            prefetch,
            outcome: ParseOutcome::default(),
            viewport,
        }
    }

    /// The outcome of the scripted parse so far: what ran, and the
    /// mid-parse failures. Call after the parse finishes.
    pub fn outcome(&self) -> ParseOutcome {
        ParseOutcome {
            handled: self.outcome.handled.clone(),
            errors: self.outcome.errors.clone(),
            layout_passes: self.outcome.layout_passes.clone(),
        }
    }

    fn record(
        &mut self,
        attrs: &[(String, String)],
        text: &str,
        error: Option<(ScriptErrorKind, Option<Url>)>,
        passes: u64,
    ) {
        let index = self.outcome.handled.len();
        self.outcome.layout_passes.push(passes);
        self.outcome
            .handled
            .push(HandledScript::from_attrs(attrs, text));
        if let Some((kind, url)) = error {
            self.outcome.errors.push((index, kind, url));
        }
    }
}

impl ScriptExecutor for ParserExecutor<'_> {
    fn execute_script(
        &mut self,
        host: &mut ScriptHost<'_, '_, '_>,
        _el: ScriptElement,
        attrs: &[(String, String)],
        script: &str,
    ) {
        let (source, url) = match classify(attrs, &self.base_url, self.prefetch) {
            Handling::Decline => return,
            Handling::Inline => (script.to_string(), None),
            Handling::External(external) => (external.source.clone(), Some(external.url.clone())),
        };
        // Install the parse-phase `document` shim, bound to this
        // script's host. The raw pointer is sound per the contract
        // documented on `install_parse_document_shim`: the guard lives
        // across the evaluation below, and `host` (the `&mut` borrow) is
        // not touched again until the guard drops.
        // The second cast deliberately erases the borrow's lifetimes to
        // `'static`: the pointer is only dereferenced while the
        // `ParseShimGuard` below is alive, which is strictly inside the
        // `&mut ScriptHost` borrow (see `install_parse_document_shim`'s
        // contract). Clippy's `unnecessary_cast` does not see the
        // erasure, so the allow is scoped to this statement.
        #[allow(clippy::unnecessary_cast)]
        let host_ptr =
            host as *mut ScriptHost<'_, '_, '_> as *mut ScriptHost<'static, 'static, 'static>;
        // Seed the parse-phase `location` shim with this script's document
        // URL so mid-parse `location.href`/`location.search` are accurate.
        bowser_js::set_parse_shim_document_url(self.page_vm.vm_mut(), &self.base_url.serialize());
        let shim = match bowser_js::install_parse_document_shim(
            self.page_vm.vm_mut(),
            host_ptr,
            self.viewport,
        ) {
            Ok(guard) => guard,
            Err(e) => {
                // Without the shim the script would see no `document`
                // at all; record the failure and continue the parse.
                // Count the attempt, like the success path below.
                self.page_vm.record_script_executed();
                self.record(
                    attrs,
                    script,
                    Some((
                        ScriptErrorKind::Eval(format!("parse document shim: {e:?}")),
                        url,
                    )),
                    0,
                );
                return;
            }
        };
        let result = self.page_vm.eval(&source);
        // Snapshot the on-demand layout pass counter before the
        // uninstall below pops the host slot: once popped,
        // `ParseShimGuard::layout_passes` reads 0. The count travels in
        // `ParseOutcome::layout_passes` so tests can assert the mid-parse
        // geometry path actually ran (and re-ran after mutations)
        // through the app's full-page parse path, not just the
        // `bowser_js` test harness.
        let passes = shim.layout_passes();
        // Restore the parent shim on the shared `document` wrapper
        // (nested `document.write` scripts) and pop the host slot before
        // touching anything else: from here on, no native can reach
        // `host` again.
        shim.uninstall(self.page_vm.vm_mut());
        // A fuel-exhausted runaway classifies as `FuelExhausted`, like the
        // post-parse path: the page load continues with the next script.
        let error = result
            .err()
            .map(|e| (ScriptErrorKind::from_vm_error(&e), url));
        // Count the attempt like `PageVm::run_one` does — even a throw
        // counts as executed, with the failure recorded alongside.
        self.page_vm.record_script_executed();
        self.record(attrs, script, error, passes);
    }
}

/// Split the post-parse [`collect_script_refs`](crate::script::collect_script_refs)
/// pass into the refs that still need to run, and the mid-parse
/// failures as [`ScriptError`]s attributed to their elements.
///
/// Matching is by fingerprint, multiset-style: each ref consumes at
/// most one handled script, in document order, so duplicated scripts
/// each match their own execution and an `async`/`defer` twin of an
/// executed script is never mistaken for it. Refs with no matching
/// fingerprint were declined mid-parse and run through the post-parse
/// path. The errors come back in document order.
pub fn filter_handled_scripts(
    refs: Vec<ScriptRef>,
    outcome: &ParseOutcome,
) -> (Vec<ScriptRef>, Vec<ScriptError>) {
    let mut remaining: Vec<HandledEntry> = outcome
        .handled
        .iter()
        .cloned()
        .map(|script| HandledEntry {
            script,
            error: None,
        })
        .collect();
    for (index, kind, url) in &outcome.errors {
        if let Some(slot) = remaining.get_mut(*index) {
            slot.error = Some(ScriptFailure {
                kind: kind.clone(),
                url: url.clone(),
            });
        }
    }
    let mut unhandled = Vec::new();
    let mut errors = Vec::new();
    for r in refs {
        match remaining.iter().position(|e| e.script.matches_ref(&r)) {
            Some(pos) => {
                let entry = remaining.remove(pos);
                if let Some(failure) = entry.error {
                    errors.push(ScriptError {
                        node: Some(r.node),
                        url: failure.url,
                        kind: failure.kind,
                    });
                }
            }
            None => unhandled.push(r),
        }
    }
    (unhandled, errors)
}

/// How many `async` script fetches fly at once. Browsers cap parallel
/// connections per host; eight in flight is that discipline, std-only.
const ASYNC_FETCH_PARALLELISM: usize = 8;

/// Fetch every external classic script source the post-parse path may
/// need, mirroring [`resolve_scripts`]' gating: module and
/// non-executable `type`s are never fetched, unresolvable `src`s are
/// left for [`resolve_scripts`] to classify, fetching stops at
/// [`MAX_SCRIPTS`], and duplicate URLs fetch once.
///
/// `async` scripts fire first, concurrently: per spec their fetch must
/// not wait behind the document-order stream, so they run ahead of the
/// sequential pass on scoped threads. `fetch` is shared across those
/// threads directly (no serializing mutex), which is why it must be
/// `Send + Sync` — the loader's closure only shares its cookie jar
/// behind a mutex, so this holds. Everything else fetches sequentially
/// in document order, exactly as before.
///
/// The parser-inserted executor serves `src` scripts from this map
/// during the parse, so no network happens on the installing thread;
/// the post-parse path reuses the same map for the scripts the parser
/// declined.
pub fn prefetch_script_sources(
    refs: &[ScriptRef],
    base_url: &Url,
    fetch: impl Fn(&Url) -> Result<String, String> + Send + Sync,
) -> HashMap<String, Result<String, String>> {
    /// One gated external classic script worth fetching.
    struct FetchJob {
        url: Url,
        key: String,
        is_async: bool,
    }

    // Gate first, fetch second: identical gating for both passes.
    let mut jobs = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for r in refs {
        if let Some(t) = r.script_type.as_deref()
            && t.trim().eq_ignore_ascii_case("module")
        {
            continue;
        }
        if let Some(t) = r.script_type.as_deref()
            && !is_executable_script_type(t)
        {
            continue;
        }
        let Some(src) = r.src.as_deref() else {
            continue;
        };
        let url = match bowser_url::parse(src.trim(), Some(base_url)) {
            Ok(url) => url,
            Err(_) => continue,
        };
        let key = url.serialize();
        if !seen.insert(key.clone()) {
            continue;
        }
        jobs.push(FetchJob {
            url,
            key,
            is_async: r.is_async,
        });
    }

    let over_cap = || Err(format!("more than {MAX_SCRIPTS} external scripts; skipped"));
    let mut out: HashMap<String, Result<String, String>> = HashMap::with_capacity(jobs.len());
    let mut fetched = 0usize;
    let (async_jobs, sync_jobs): (Vec<FetchJob>, Vec<FetchJob>) =
        jobs.into_iter().partition(|job| job.is_async);

    // `async` first: fired immediately, concurrently, in chunks of
    // `ASYNC_FETCH_PARALLELISM`. The cap applies in dispatch order, so
    // the first `MAX_SCRIPTS` async jobs fetch and the rest record the
    // over-cap skip without touching the network.
    let mut concurrent: Vec<&FetchJob> = Vec::with_capacity(async_jobs.len());
    for job in &async_jobs {
        if fetched >= MAX_SCRIPTS {
            out.insert(job.key.clone(), over_cap());
        } else {
            fetched += 1;
            concurrent.push(job);
        }
    }
    for chunk in concurrent.chunks(ASYNC_FETCH_PARALLELISM) {
        std::thread::scope(|s| {
            let handles: Vec<(
                String,
                std::thread::ScopedJoinHandle<'_, Result<String, String>>,
            )> = chunk
                .iter()
                .map(|job| (job.key.clone(), s.spawn(|| fetch(&job.url))))
                .collect();
            for (key, handle) in handles {
                // A panicking fetch thread records the skip; the page
                // loads on regardless — a script never fails the page.
                let result = handle
                    .join()
                    .unwrap_or_else(|_| Err("async script fetch thread panicked".to_string()));
                out.insert(key, result);
            }
        });
    }

    // Everything else: sequential in document order, as before.
    for job in &sync_jobs {
        if fetched >= MAX_SCRIPTS {
            out.insert(job.key.clone(), over_cap());
            continue;
        }
        fetched += 1;
        out.insert(job.key.clone(), fetch(&job.url));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use bowser_dom::Document;

    /// A [`ScriptExecutor`] that declines everything: scripts stay in
    /// the tree for the post-parse path.
    struct NoopExecutor;
    impl ScriptExecutor for NoopExecutor {
        fn execute_script(
            &mut self,
            _host: &mut ScriptHost<'_, '_, '_>,
            _el: ScriptElement,
            _attrs: &[(String, String)],
            _script: &str,
        ) {
        }
    }

    /// Run `html` through the scripting parse with `prefetch` sources,
    /// and return the outcome plus the post-parse reconciliation
    /// (`unhandled` refs and attributed mid-parse errors).
    fn run_parse(
        html: &str,
        prefetch: &HashMap<String, Result<String, String>>,
    ) -> (ParseOutcome, Document, Vec<ScriptRef>, Vec<ScriptError>) {
        let mut page_vm = PageVm::new();
        let base = bowser_url::parse("https://example.com/", None).unwrap();
        let (document, outcome) = {
            let mut executor = ParserExecutor::new(&mut page_vm, &base, prefetch, None);
            let (document, _quirks) =
                Document::parse_html_with_executor(html.as_bytes(), true, &mut executor);
            let outcome = executor.outcome();
            (document, outcome)
        };
        let refs = crate::script::collect_script_refs(&document);
        let (unhandled, errors) = filter_handled_scripts(refs, &outcome);
        (outcome, document, unhandled, errors)
    }

    /// Rendered text of the body, skipping `<script>` subtrees (their
    /// source text is not rendered output).
    fn body_text(document: &Document) -> String {
        use bowser_dom::NodeKind;
        let body = document.query_selector("body").unwrap().unwrap();
        fn walk(document: &Document, node: bowser_dom::NodeId, out: &mut String) {
            match document.kind(node) {
                NodeKind::Text(text) => out.push_str(text),
                NodeKind::Element(data) if data.tag_name == "script" => {}
                _ => {
                    // Cursor walk, no per-node `Vec` from `Document::children`.
                    let mut cursor = document.first_child(node);
                    while let Some(child) = cursor {
                        walk(document, child, out);
                        cursor = document.next_sibling(child);
                    }
                }
            }
        }
        let mut out = String::new();
        walk(document, body, &mut out);
        out
    }

    #[test]
    fn scripts_execute_in_document_order() {
        let html = concat!(
            "<script>document.write('a')</script>",
            "<script>document.write('b')</script>",
            "<script>document.write('c')</script>",
        );
        let (outcome, document, unhandled, errors) = run_parse(html, &HashMap::new());
        assert_eq!(body_text(&document), "abc");
        assert_eq!(outcome.handled.len(), 3);
        assert!(unhandled.is_empty());
        assert!(errors.is_empty());
    }

    #[test]
    fn external_scripts_execute_in_document_order_between_inline() {
        let mut prefetch = HashMap::new();
        prefetch.insert(
            "https://example.com/x.js".to_string(),
            Ok("document.write('B')".to_string()),
        );
        let html = concat!(
            "<script>document.write('A')</script>",
            "<script src=\"/x.js\"></script>",
            "<script>document.write('C')</script>",
        );
        let (outcome, document, unhandled, errors) = run_parse(html, &prefetch);
        assert_eq!(body_text(&document), "ABC");
        assert_eq!(outcome.handled.len(), 3);
        assert!(unhandled.is_empty());
        assert!(errors.is_empty());
    }

    #[test]
    fn earlier_dom_is_visible_later_dom_is_not() {
        let html = concat!(
            "<div id=\"early\">a</div>",
            "<script>document.write(document.getElementById('early') !== null ? 'E' : 'e')</script>",
            "<script>document.write(document.getElementById('late') !== null ? 'L' : 'l')</script>",
            "<div id=\"late\">b</div>",
        );
        let (_outcome, document, _unhandled, _errors) = run_parse(html, &HashMap::new());
        // The first script sees the parsed-so-far tree; the second runs
        // before the late div is tokenized, so it is invisible.
        assert_eq!(body_text(&document), "aElb");
    }

    #[test]
    fn nested_script_written_by_document_write_executes_reentrantly() {
        let html =
            "<script>document.write('<scr' + 'ipt>document.write(\"N\")</scr' + 'ipt>')</script>";
        let (outcome, document, unhandled, errors) = run_parse(html, &HashMap::new());
        assert_eq!(body_text(&document), "N");
        assert_eq!(outcome.handled.len(), 2);
        assert!(unhandled.is_empty());
        assert!(errors.is_empty());
    }

    #[test]
    fn sequential_scripts_each_get_a_working_document() {
        // Parser-inserted scripts never nest re-entrantly: `document.write`
        // only queues input, so the written `<script>` runs after the outer
        // script's shim is uninstalled. The parse shares one `document`
        // wrapper across scripts (like the one real `Document`); each
        // install repoints it at its own host, so the second script's
        // `document` serves its own host and still sees the first
        // script's DOM writes.
        let html = concat!(
            "<script>document.write('<b id=\"a\">a</b>')</script>",
            "<script>",
            "document.write('<b id=\"b\">b</b>');",
            "document.write(document.getElementById('a') !== null ? 'F' : 'f');",
            "</script>",
        );
        let (outcome, document, unhandled, errors) = run_parse(html, &HashMap::new());
        assert!(errors.is_empty(), "unexpected script errors: {errors:?}");
        assert!(unhandled.is_empty());
        assert_eq!(outcome.handled.len(), 2);
        let text = body_text(&document);
        assert!(
            text.contains('F'),
            "second script could not see the first script's write: {text:?}"
        );
        assert!(document.query_selector("#a").unwrap().is_some());
        assert!(document.query_selector("#b").unwrap().is_some());
    }

    #[test]
    fn get_elements_by_tag_name_and_set_attribute_work_mid_parse() {
        let html = concat!(
            "<p>a</p><p>b</p><div id=\"t\"></div>",
            "<script>",
            "document.write(document.getElementsByTagName('p').length);",
            "document.getElementById('t').setAttribute('data-x', '42');",
            "</script>",
        );
        let (_outcome, document, _unhandled, _errors) = run_parse(html, &HashMap::new());
        assert!(body_text(&document).contains('2'));
        let t = document.query_selector("#t").unwrap().unwrap();
        assert_eq!(document.get_attribute(t, "data-x"), Some("42"));
    }

    #[test]
    fn async_defer_and_unfetched_scripts_decline_to_post_parse() {
        let mut prefetch = HashMap::new();
        prefetch.insert(
            "https://example.com/ok.js".to_string(),
            Ok("document.write('O')".to_string()),
        );
        prefetch.insert(
            "https://example.com/bad.js".to_string(),
            Err("404".to_string()),
        );
        let html = concat!(
            "<script>document.write('I')</script>",
            "<script src=\"/ok.js\"></script>",
            "<script src=\"/ok.js\" async></script>",
            "<script src=\"/ok.js\" defer></script>",
            "<script src=\"/bad.js\"></script>",
            "<script src=\"/missing.js\"></script>",
        );
        let (outcome, document, unhandled, errors) = run_parse(html, &prefetch);
        // Only the inline script and the sync prefetched external ran
        // mid-parse, in document order. `async` is declined mid-parse
        // (its fetch fired concurrently; the post-parse path runs it in
        // the async batch), like `defer`, the failed fetch, and the
        // never-prefetched source.
        assert_eq!(body_text(&document), "IO");
        assert_eq!(outcome.handled.len(), 2);
        assert_eq!(unhandled.len(), 4);
        assert!(errors.is_empty());
        // The declined async ref keeps its fingerprint: it is not the
        // executed sync twin.
        let async_ref = unhandled
            .iter()
            .find(|r| r.src.as_deref() == Some("/ok.js"))
            .expect("async ref");
        assert!(async_ref.is_async);
        assert!(!async_ref.defer);
    }

    #[test]
    fn mid_parse_errors_are_attributed_to_their_elements() {
        let html = concat!(
            "<script>document.write('ok')</script>",
            "<script>throw new Error('boom')</script>",
        );
        let (outcome, document, unhandled, errors) = run_parse(html, &HashMap::new());
        assert_eq!(outcome.handled.len(), 2);
        assert_eq!(outcome.errors.len(), 1);
        assert!(unhandled.is_empty());
        assert_eq!(errors.len(), 1);
        let refs = crate::script::collect_script_refs(&document);
        assert_eq!(errors[0].node, Some(refs[1].node));
        assert!(errors[0].to_string().contains("boom"));
    }

    #[test]
    fn async_twin_of_executed_sync_script_still_runs_post_parse() {
        // Same `src` twice: once sync (runs mid-parse), once async
        // (declined mid-parse). The fingerprint carries `is_async`, so
        // the async twin is never mistaken for the executed sync script.
        let mut prefetch = HashMap::new();
        prefetch.insert(
            "https://example.com/x.js".to_string(),
            Ok("document.write('X')".to_string()),
        );
        let html = concat!(
            "<script src=\"/x.js\"></script>",
            "<script src=\"/x.js\" async></script>",
        );
        let (outcome, _document, unhandled, errors) = run_parse(html, &prefetch);
        assert_eq!(outcome.handled.len(), 1);
        assert!(!outcome.handled[0].is_async);
        assert_eq!(unhandled.len(), 1);
        assert!(unhandled[0].is_async);
        assert!(errors.is_empty());
    }

    #[test]
    fn prefetch_fetches_async_scripts_on_threads() {
        use std::sync::Mutex;
        // Each fetch records its thread: with concurrent fetching, the
        // async jobs run on distinct spawned threads, not the calling
        // thread. Deterministic: one chunk of `ASYNC_FETCH_PARALLELISM`
        // spawns one OS thread per job.
        let base = bowser_url::parse("https://example.com/", None).unwrap();
        let (document, _quirks) = Document::parse_html_with_executor(
            concat!(
                "<script async src=\"/a.js\"></script>",
                "<script src=\"/s.js\"></script>",
                "<script async src=\"/b.js\"></script>",
            )
            .as_bytes(),
            true,
            &mut NoopExecutor,
        );
        let refs = crate::script::collect_script_refs(&document);
        assert!(refs[0].is_async && refs[2].is_async && !refs[1].is_async);
        let threads = Mutex::new(Vec::new());
        let fetched = prefetch_script_sources(&refs, &base, |url| {
            threads
                .lock()
                .unwrap()
                .push((url.serialize(), std::thread::current().id()));
            Ok(String::new())
        });
        assert_eq!(fetched.len(), 3);
        let threads = threads.lock().unwrap();
        let main = std::thread::current().id();
        let async_threads: Vec<_> = threads
            .iter()
            .filter(|(url, _)| url.ends_with("/a.js") || url.ends_with("/b.js"))
            .map(|(_, id)| *id)
            .collect();
        let sync_threads: Vec<_> = threads
            .iter()
            .filter(|(url, _)| url.ends_with("/s.js"))
            .map(|(_, id)| *id)
            .collect();
        // Both async fetches ran off the calling thread, on distinct
        // threads (true concurrency, not a serialized loop)...
        assert_eq!(async_threads.len(), 2);
        assert_ne!(async_threads[0], async_threads[1]);
        assert!(async_threads.iter().all(|id| *id != main));
        // ...while the sync fetch stayed on the calling thread.
        assert_eq!(sync_threads, [main]);
    }

    #[test]
    fn mid_parse_runaway_script_terminates_with_fuel_error() {
        // A non-terminating parser-inserted script burns its per-eval
        // fuel budget instead of stalling the installing thread: the
        // failure is recorded and the parse continues.
        let html = concat!(
            "<script>while (true) {}</script>",
            "<script>document.write('after')</script>",
        );
        let (outcome, document, unhandled, errors) = run_parse(html, &HashMap::new());
        assert_eq!(outcome.handled.len(), 2);
        assert_eq!(outcome.errors.len(), 1);
        assert!(unhandled.is_empty());
        assert_eq!(errors.len(), 1);
        assert_eq!(
            errors[0].kind,
            crate::script::ScriptErrorKind::FuelExhausted,
            "runaway classified as fuel exhaustion: {errors:?}"
        );
        assert!(errors[0].to_string().contains("budget"), "{}", errors[0]);
        // The script after the runaway still ran.
        assert_eq!(body_text(&document), "after");
    }

    #[test]
    fn prefetch_mirrors_resolve_scripts_gating() {
        let base = bowser_url::parse("https://example.com/", None).unwrap();
        let (document, _quirks) = Document::parse_html_with_executor(
            concat!(
                "<script src=\"/a.js\"></script>",
                "<script src=\"/a.js\"></script>",
                "<script type=\"module\" src=\"/m.js\"></script>",
                "<script type=\"application/json\" src=\"/d.js\"></script>",
                "<script src=\"http://\"></script>",
                "<script>inline</script>",
            )
            .as_bytes(),
            true,
            &mut NoopExecutor,
        );
        let refs = crate::script::collect_script_refs(&document);
        let fetched =
            prefetch_script_sources(&refs, &base, |url| Ok(format!("src:{}", url.serialize())));
        // Duplicate URLs fetch once; modules, data blocks,
        // unresolvable URLs, and inline scripts are never fetched.
        assert_eq!(fetched.len(), 1);
        assert!(fetched["https://example.com/a.js"].is_ok());
    }

    #[test]
    fn captured_shim_document_renders_results_post_parse() {
        use std::cell::RefCell;
        use std::rc::Rc;

        // testharness.js's document-capture pattern with default output
        // (no `setup({output:false})`): `Output` grabs `document` at load
        // and renders the results table through the captured reference
        // later — from a later parser-inserted script (while the parse is
        // still running) and after the parse (e.g. the harness's `load`
        // listener, adopted into the production table). The parse keeps
        // one shared `document` wrapper, so the mid-parse render resolves
        // to the live host instead of throwing `TypeError: document is
        // not available outside the parser`; the production install then
        // upgrades that same wrapper in place, so the post-parse render
        // gets the full binding surface (`createElement`, `body`, ...).
        let html = concat!(
            "<script>",
            // Captured at load, like `Output`'s `this.output_document`.
            "var __output_document = document;",
            "var __results = [];",
            "function test(fn, name) { __results.push({fn: fn, name: name}); }",
            "function done() {",
            "  var doc = __output_document;",
            "  var node = doc.getElementById('log');",
            "  if (!node) {",
            // Mid-parse there is no render target yet.
            "    if (doc.readyState === 'loading') { return; }",
            "    node = doc.createElement('section');",
            "    node.setAttribute('id', 'log');",
            "    doc.body.appendChild(node);",
            "  }",
            "  var table = doc.createElement('table');",
            "  for (var i = 0; i < __results.length; i++) {",
            "    var t = __results[i];",
            "    var status = 'PASS';",
            "    try { t.fn(); } catch (e) { status = 'FAIL'; }",
            "    var tr = doc.createElement('tr');",
            "    var td = doc.createElement('td');",
            "    td.appendChild(doc.createTextNode(t.name + ': ' + status));",
            "    tr.appendChild(td);",
            "    table.appendChild(tr);",
            "  }",
            "  node.appendChild(table);",
            "}",
            "</script>",
            // A later parser-inserted script renders mid-parse: the
            // captured wrapper must resolve to the live host (no
            // TypeError), and take the "no render target yet" return.
            "<script>",
            "test(function() {}, 'trivial');",
            "done();",
            "</script>",
        );
        let base = bowser_url::parse("https://example.com/", None).unwrap();
        let mut page_vm = PageVm::new();
        let prefetch = HashMap::new();
        let document = {
            let mut executor = ParserExecutor::new(&mut page_vm, &base, &prefetch, None);
            let (document, _quirks) =
                Document::parse_html_with_executor(html.as_bytes(), true, &mut executor);
            let outcome = executor.outcome();
            assert_eq!(outcome.handled.len(), 2);
            assert!(
                outcome.errors.is_empty(),
                "mid-parse render threw: {:?}",
                outcome.errors
            );
            document
        };
        // The production page path: install the web APIs on the same
        // realm that ran the parser-inserted scripts.
        let shared = Rc::new(RefCell::new(document));
        page_vm
            .install_document_apis(shared.clone(), &base, "BowserTest/1.0", &[])
            .expect("web API install");
        // Post-parse render through the captured wrapper (the harness's
        // `load` listener path): must not throw, and the results table
        // must land in the DOM.
        page_vm.eval("done()").expect("post-parse render threw");
        let document = shared.borrow();
        assert!(
            document.query_selector("table").unwrap().is_some(),
            "results table must render into the DOM"
        );
        assert!(
            body_text(&document).contains("trivial: PASS"),
            "unexpected body text: {:?}",
            body_text(&document)
        );
    }
}
