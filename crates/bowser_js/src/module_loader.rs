//! ES module loader: module map, linking/instantiation, dynamic `import()`,
//! and `import.meta`.
//!
//! The parser and bytecode compiler already understand module *syntax*:
//! [`compile_module`](crate::compiler::compile_module) produces a
//! [`ModuleRecord`](crate::compiler::ModuleRecord) describing a module's
//! imports and exports, while `import`/`export` statements themselves compile
//! to [`Op::Nop`](crate::opcode::Op::Nop). This loader provides the missing
//! *linking* step on top of that record:
//!
//! 1. **Module map.** [`ModuleLoader`] maps resolved URL -> module entry
//!    ([`ModuleRecord`] + evaluation state). Each URL is fetched (in-memory
//!    registration first, then `bowser_net`) and parsed at most once.
//! 2. **Linking.** Dependencies are evaluated depth-first, post-order, so
//!    every static dependency's namespace object is populated before its
//!    importers run. Cycles are tolerated: a module that (transitively)
//!    imports itself observes the partially-populated namespace, mirroring
//!    the shape of the spec's cycle handling.
//! 3. **Instantiation.** [`Vm`](crate::vm::Vm) only exposes source-level
//!    evaluation, so each module's source is rewritten before it runs:
//!    * `import {a as b} from "m"` becomes `const b =
//!      globalThis["\0bowser_module_registry"]["<m>"]["a"];` (with analogous
//!      forms for default imports, `* as ns`, and side-effect-only
//!      imports). The registry object is installed on the global by the
//!      loader and holds every evaluated module's namespace object.
//!      Unshadowed references to `b` are further rewritten to live reads
//!      through the export's defining module (see below).
//!    * `export` statements keep their declarations (they already compile to
//!      normal code); an epilogue appended to the module copies each
//!      exported binding into the module's namespace object, except for
//!      live mutable bindings (see below).
//!    * `export default <expr>` becomes `NS["default"] = (<expr>);`.
//!    * `import.meta` becomes `({ url: "<module url>" })`.
//!    * `import("<literal>")` becomes `Promise.resolve(<namespace>)`; the
//!      target is loaded eagerly at link time. A non-literal specifier
//!      keeps its `import(x)` form and runs through the VM's
//!      [`Op::DynamicImport`](crate::opcode::Op::DynamicImport), which
//!      queues a host dynamic import: the loader resolves, loads, and
//!      evaluates the target when the interpreter reaches the next
//!      microtask checkpoint, settling the returned promise with the
//!      target's namespace (or a `TypeError` rejection on failure).
//!
//! # Live bindings (common cases)
//!
//! Exported top-level `let`/`var` bindings are live: references to them
//! (in the defining module and in importers) are rewritten to reads of the
//! defining module's namespace object, which is the single source of truth.
//! The namespace is synced immediately after the declaration so reads
//! before the epilogue observe the initial value, and every assignment or
//! update (`=`, `++`, `+=`, ...) to the local writes through to the
//! namespace. Re-exports (`export { x } from "m"`, `export * from "m"`)
//! resolve through to the defining module, so chains stay live. Imported
//! bindings remain immutable: assignment to an import still throws
//! `TypeError`. A name declared in any nested scope (parameter, inner
//! `let`, function, catch, etc.) disables live rewriting for that name
//! throughout the module (conservative shadowing); the `const` prelude
//! snapshot is used instead. `with` disables live rewriting entirely.
//!
//! # Simplifications vs. the spec
//!
//! * **Live bindings cover the common cases only.** Destructured exports,
//!   `export` of a name declared only in a nested scope, and aliased
//!   re-exports of mutable bindings through multiple barrels may fall back
//!   to snapshot semantics. `const` and function exports are snapshot
//!   (they cannot be reassigned anyway).
//! * **No import maps or import assertions.** Bare specifiers resolve as
//!   relative paths; only `http(s)` URLs are fetched from the network.
//! * **Strict-mode modules only**, matching the existing parser.
//! * **`export *` conflicts** resolve first-writer-wins in source order
//!   (explicit exports always win); the spec excludes ambiguous names.
//! * A dynamic `import()` with a *literal* specifier whose target fails
//!   to load becomes `Promise.reject(...)` at link time rather than
//!   rejecting when reached. Non-literal specifiers reject at the
//!   microtask checkpoint where the host import runs instead.
//! * `import` bindings are `const` (immutable, matching the spec) but are
//!   not hoisted above preceding statements: referencing an imported binding
//!   textually before its `import` statement hits the TDZ.
//! * Namespace objects are ordinary (mutable, extensible) objects, not the
//!   spec's frozen exotic namespace objects.
//!
//! The public entry points ([`ModuleLoader`], [`ModuleError`]) are wired up
//! through `run_script_element` in `bindings.rs`: `<script type="module">`
//! executes via `ScriptEngine::evaluate_module`, which routes into
//! [`ModuleLoader::load_source`].

use std::collections::{HashMap, HashSet};
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

use crate::ast::{
    ArrayPatElem, ArrowBody, CatchClause, ClassKey, ClassMember, ExportDecl, Expr, ForInit, ForLhs,
    Literal, ModuleName, ObjectPatProp, Param, Pat, Program, PropDef, PropKey, Stmt, SwitchCase,
    TemplatePart, UnaryOp, VarDecl, VarKind,
};
use crate::builtin_object_array::get_own;
use crate::compiler::{ExportEntry, ModuleRecord};
use crate::lexer::Span;
use crate::parser;
use crate::value::{Object, ObjectKind, Property, Value};
use crate::vm::Vm;
use crate::wtf8::Wtf8;

/// Errors from module loading, linking, and evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleError {
    /// A specifier could not be resolved against its referrer.
    Resolve(String),
    /// The module source could not be fetched.
    Fetch(String),
    /// The module source failed to parse.
    Parse(String),
    /// Linking or evaluation failed (compile error, thrown value, OOM, ...).
    Eval(String),
}

impl std::fmt::Display for ModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Resolve(e) => write!(f, "cannot resolve module specifier: {e}"),
            Self::Fetch(e) => write!(f, "cannot fetch module: {e}"),
            Self::Parse(e) => write!(f, "module parse error: {e}"),
            Self::Eval(e) => write!(f, "module evaluation failed: {e}"),
        }
    }
}

impl std::error::Error for ModuleError {}

/// Evaluation state of one entry in the module map.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ModuleState {
    /// On the DFS stack: cyclic imports observe the partial namespace.
    Evaluating,
    Evaluated,
    /// Poisoned: a later `load` of the same URL re-reports the error
    /// instead of retrying the failed load. Recorded on *every* failure
    /// path — including failures that fire before an entry exists
    /// (fetch, parse, compile) — so a permanently-unloadable module
    /// costs one attempt, not one per `import()`.
    Failed(String),
}

#[derive(Debug, Clone)]
struct ModuleEntry {
    record: ModuleRecord,
    state: ModuleState,
}

/// The ES module loader.
///
/// Owns the module map (resolved URL -> [`ModuleRecord`] + state) and drives
/// linking/instantiation against a [`Vm`]. The loader holds no VM state
/// itself: namespace objects live on the VM's global under an unforgeable
/// key, so one loader can serve many evaluations as long as the same [`Vm`]
/// is used.
#[derive(Debug, Default)]
pub struct ModuleLoader {
    modules: HashMap<String, ModuleEntry>,
    /// In-memory URL -> source, checked before the network (tests, embedder
    /// pre-registration, `<script type="module">` inline text).
    mem_sources: HashMap<String, String>,
    allow_net: bool,
    /// Test instrumentation only: counts [`fetch_source`](Self::fetch_source)
    /// calls, so tests can assert a poisoned entry re-rejects without
    /// re-fetching.
    #[cfg(test)]
    fetch_count: std::cell::Cell<usize>,
}

/// Memoized parse+compile of one module: everything
/// [`ModuleLoader::load_and_evaluate`] does before running the rewritten
/// script, keyed by module URL. Plain data only (no `Vm` handles), so one
/// cache can serve many loaders and many `Vm`s.
#[derive(Debug, Clone)]
struct CachedModule {
    /// Hash of the source this entry was built from: re-registering
    /// different source under the same URL is a miss, never stale reuse.
    /// Paired with `source_hash2` (an independent FNV-1a with a different
    /// offset basis); a lookup is a hit only when *both* match, so a
    /// single-hash collision between two different sources under one URL
    /// cannot silently reuse the wrong [`ModuleRecord`].
    source_hash: u64,
    source_hash2: u64,
    /// The module record (imports/exports), needed per load for the DFS.
    record: ModuleRecord,
    /// The rewritten classic-script source: module parse, module record,
    /// and the source rewrite are done. Re-running it through
    /// [`Vm::eval`] re-parses and re-compiles it (one classic-script
    /// compile, no module pipeline) and re-executes it — which is
    /// exactly what stateful-fixture isolation needs: each load gets
    /// fresh bindings and a fresh namespace object.
    rewritten: String,
    /// URLs of dynamic-`import()` targets, resolved at compile time. A
    /// cache hit still ensures each is evaluated before the rewritten
    /// code runs (the rewrite normally does this inline while splicing
    /// the `Promise.resolve(...)`).
    dyn_deps: Vec<String>,
}

/// Cross-loader cache of module parse+compile work, keyed by URL.
///
/// A fresh [`ModuleLoader`] per test preserves stateful-fixture
/// isolation (every test re-evaluates every fixture), but re-parsing,
/// re-analyzing, and re-rewriting the same fixture sources per test is
/// pure waste: the fixtures are immutable for the run. This cache
/// memoizes everything up to (but excluding) evaluation — module parse,
/// module record, source rewrite — so a cache hit only re-runs the
/// rewritten script through [`Vm::eval`], which re-executes it against a
/// fresh namespace published by the loading [`ModuleLoader`].
///
/// This answers the soundness objection to loader reuse: reusing a
/// *loader* across tests would skip re-evaluation (unsound for stateful
/// fixtures such as `setup_FIXTURE.js`); reusing only the parse+compile
/// still evaluates every module per test through a fresh loader.
///
/// Modules whose rewrite embedded a dynamic-import *failure* are never
/// cached: the rejection string would bake in a stale evaluation
/// outcome. Bounded at [`COMPILE_CACHE_MAX_ENTRIES`]; on overflow the
/// whole map is flushed (same backstop pattern as the test262 body
/// cache) — entries are cheap to rebuild.
///
/// Entries are plain data (no `Vm` handles) keyed by URL plus a hash of
/// the source they were built from, so the single process-wide instance
/// ([`COMPILE_CACHE`]) is sound to share across loaders, threads, and
/// `Vm`s; re-registering different source under the same URL is a miss,
/// never stale reuse.
#[derive(Debug, Default)]
pub struct ModuleCompileCache {
    modules: HashMap<String, CachedModule>,
}

/// Process-wide module parse+compile cache. The loader holds the lock
/// only for short lookups/inserts, never across the recursive
/// [`ModuleLoader::ensure_evaluated`] calls.
static COMPILE_CACHE: LazyLock<Mutex<ModuleCompileCache>> =
    LazyLock::new(|| Mutex::new(ModuleCompileCache::new()));

/// Process-wide count of compile-cache hits. Test instrumentation only:
/// lets tests assert an actual HIT (not just key presence).
#[cfg(test)]
static COMPILE_CACHE_HITS: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
fn compile_cache_hits() -> usize {
    COMPILE_CACHE_HITS.load(Ordering::Relaxed)
}

/// Backstop on [`ModuleCompileCache`] growth: entry-point modules (one
/// URL per test) are cached too, so the map is flushed rather than grown
/// without bound.
const COMPILE_CACHE_MAX_ENTRIES: usize = 4096;

impl ModuleCompileCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Hash a module source for [`CachedModule::source_hash`] /
/// [`CachedModule::source_hash2`].
///
/// FNV-1a (64-bit): fully deterministic across processes, platforms, and
/// toolchains. `std`'s `DefaultHasher` is explicitly *not* guaranteed to
/// be stable across `new()` calls ("Hashers are not guaranteed to be the
/// same"), so it cannot back a cache key: a toolchain or platform change
/// could silently neuter the entire compile cache with zero test signal.
///
/// Two independent hashes (different offset bases) back each cache entry:
/// a lookup is a hit only when both match, so a 64-bit collision between
/// two different sources under one URL cannot silently reuse the wrong
/// [`ModuleRecord`].
fn hash_source(source: &str) -> u64 {
    hash_source_with_basis(source, 0xcbf2_9ce4_8422_2325)
}

/// Second, independent FNV-1a of the source: same prime, a different
/// (nonstandard) offset basis, so the two digests are uncorrelated.
fn hash_source2(source: &str) -> u64 {
    hash_source_with_basis(source, 0x9e37_79b9_7f4a_7c15)
}

fn hash_source_with_basis(source: &str, basis: u64) -> u64 {
    const FNV_PRIME: u64 = 0x0100_0000_01b3;
    let mut h = basis;
    for b in source.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Global property holding the module registry: resolved URL -> namespace
/// object. The leading NUL keeps ordinary scripts from colliding with it.
const REGISTRY_KEY: &str = "\0bowser_module_registry";

/// JS source expression evaluating to the registry object.
const REGISTRY_SRC: &str = "globalThis[\"\\0bowser_module_registry\"]";

// Public API (re-exported as `bowser_js::modules::{ModuleLoader,
// ModuleError}`). The app will drive module loading (`load_source`/
// `import`) for `<script type="module">` once it instantiates a page VM.
impl ModuleLoader {
    /// Create a loader. Network fetching is enabled by default; tests
    /// usually call [`set_allow_net`](Self::set_allow_net)`(false)`.
    pub fn new() -> Self {
        Self {
            allow_net: true,
            ..Self::default()
        }
    }

    /// Pre-register source for a URL. Checked before the network by
    /// [`load`](Self::load); overwrites any previous registration.
    pub fn add_source(&mut self, url: impl Into<String>, source: impl Into<String>) {
        self.mem_sources.insert(url.into(), source.into());
    }

    /// Enable or disable the `bowser_net` fetch fallback.
    pub fn set_allow_net(&mut self, allow: bool) {
        self.allow_net = allow;
    }

    /// Resolve `specifier` against `referrer` to a canonical,
    /// fragment-stripped URL string.
    pub fn resolve(&self, specifier: &str, referrer: &str) -> Result<String, ModuleError> {
        let spec = specifier.trim();
        // Absolute URL (or absolute path handled below via the base).
        if let Ok(abs) = bowser_url::parse(spec, None) {
            return Ok(abs.serialize_without_fragment());
        }
        let base = bowser_url::parse(referrer, None)
            .map_err(|e| ModuleError::Resolve(format!("bad referrer {referrer:?}: {e}")))?;
        bowser_url::parse(spec, Some(&base))
            .map(|u| u.serialize_without_fragment())
            .map_err(|e| {
                ModuleError::Resolve(format!("cannot resolve {spec:?} against {referrer:?}: {e}"))
            })
    }

    /// Fetch, link, and evaluate the module at `url`, returning its
    /// namespace object. Each URL is evaluated at most once per loader.
    pub fn load(&mut self, vm: &mut Vm, url: &str) -> Result<Value, ModuleError> {
        self.ensure_evaluated(vm, url)?;
        self.namespace_of(vm, url)
            .ok_or_else(|| ModuleError::Eval(format!("module {url} has no namespace object")))
    }

    /// Resolve `specifier` against `referrer`, then [`load`](Self::load) it.
    pub fn import(
        &mut self,
        vm: &mut Vm,
        referrer: &str,
        specifier: &str,
    ) -> Result<Value, ModuleError> {
        let url = self.resolve(specifier, referrer)?;
        self.load(vm, &url)
    }

    /// Evaluate already-available `source` as the module at `url`
    /// (e.g. an inline `<script type="module">` whose referrer is the
    /// document URL), returning its namespace object.
    ///
    /// Module-map semantics: the map is keyed by resolved URL and each
    /// URL is evaluated at most once per loader — **first registration
    /// wins for a given URL**. If `url` is already `Evaluated` (or
    /// poisoned as `Failed`), this returns the existing namespace (or
    /// re-reports the recorded error) and the new `source` is *not*
    /// re-evaluated. Calling this twice with the same URL but different
    /// source silently returns the stale namespace.
    pub fn load_source(
        &mut self,
        vm: &mut Vm,
        url: &str,
        source: &str,
    ) -> Result<Value, ModuleError> {
        self.add_source(url, source);
        self.load(vm, url)
    }

    /// The namespace object of an evaluated module, if present.
    pub fn namespace_of(&self, vm: &Vm, url: &str) -> Option<Value> {
        let reg = vm.get_global(REGISTRY_KEY)?;
        let Value::Object(ro) = reg else {
            return None;
        };
        get_own(vm.heap(), ro, Wtf8::from_str(url))
    }

    // -- internals ---------------------------------------------------------

    fn fetch_source(&self, url: &str) -> Result<String, ModuleError> {
        #[cfg(test)]
        self.fetch_count.set(self.fetch_count.get() + 1);
        if let Some(src) = self.mem_sources.get(url) {
            return Ok(src.clone());
        }
        if !self.allow_net {
            return Err(ModuleError::Fetch(format!(
                "no registered source for {url}"
            )));
        }
        let parsed = bowser_url::parse(url, None)
            .map_err(|e| ModuleError::Fetch(format!("bad module URL {url}: {e}")))?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(ModuleError::Fetch(format!(
                "unsupported module URL scheme {:?} for {url}",
                parsed.scheme()
            )));
        }
        let request = bowser_net::fetch::Request {
            method: "GET".to_string(),
            url: parsed,
            headers: Vec::new(),
            body: None,
        };
        bowser_net::fetch::fetch(&request)
            .map(|r| r.text())
            .map_err(|e| ModuleError::Fetch(format!("fetch of {url} failed: {e}")))
    }

    /// Depth-first, post-order evaluation with cycle tolerance.
    fn ensure_evaluated(&mut self, vm: &mut Vm, url: &str) -> Result<(), ModuleError> {
        if let Some(entry) = self.modules.get(url) {
            match &entry.state {
                ModuleState::Evaluated | ModuleState::Evaluating => return Ok(()),
                ModuleState::Failed(msg) => return Err(ModuleError::Eval(msg.clone())),
            }
        }
        let result = self.load_and_evaluate(vm, url);
        if let Err(e) = &result {
            // Poison the module map on *every* failure path — including
            // errors that fire before any entry was inserted (fetch,
            // parse, compile). Without this, every later `import()` of
            // the URL would retry the failed load instead of
            // re-rejecting: with `allow_net`, a permanently-missing
            // module would cost one HTTP request per import.
            let msg = e.to_string();
            match self.modules.get_mut(url) {
                Some(entry) => entry.state = ModuleState::Failed(msg),
                None => {
                    self.modules.insert(
                        url.to_owned(),
                        ModuleEntry {
                            record: ModuleRecord::default(),
                            state: ModuleState::Failed(msg),
                        },
                    );
                }
            }
        }
        result
    }

    fn load_and_evaluate(&mut self, vm: &mut Vm, url: &str) -> Result<(), ModuleError> {
        let source = self.fetch_source(url)?;
        // Fast path: parse + module record + rewrite memoized in the
        // shared compile cache. Evaluation still happens below, per load,
        // through this (fresh) loader with a fresh namespace — stateful
        // fixtures re-execute every time.
        if let Some(compiled) = self.lookup_compiled(url, &source) {
            return self.evaluate_compiled(vm, url, &source, &compiled);
        }
        self.evaluate_slow(vm, url, &source, true)
    }

    /// Slow path: parse, compile, rewrite, run. Factored out so the
    /// compile-cache fast path can fall back to it when a cached
    /// dynamic-import dep fails at replay (the re-rewrite splices a
    /// `Promise.reject(...)` for the dep and marks the module
    /// uncacheable, matching the uncached behavior).
    ///
    /// `republish` is true on the cache-miss path: publish a fresh
    /// namespace and evaluate the static deps. The fallback from
    /// [`evaluate_compiled`](Self::evaluate_compiled) passes false —
    /// the namespace is already published and the static deps are
    /// already evaluated, and publishing again would install a *second*
    /// namespace in the registry, orphaning the first one that cyclic
    /// static imports may already have captured via `import * as ns`
    /// (it would stay empty forever while the body populates the new
    /// one).
    fn evaluate_slow(
        &mut self,
        vm: &mut Vm,
        url: &str,
        source: &str,
        republish: bool,
    ) -> Result<(), ModuleError> {
        let program =
            parser::parse(source).map_err(|e| ModuleError::Parse(format!("{url}: {e:?}")))?;
        // Modules compile through the same VM-global brand sequence as
        // scripts and eval, so a class in one module can never share a
        // brand with a class in another module or script.
        let (_unit, record) = vm
            .compile_module(&program)
            .map_err(|e| ModuleError::Eval(format!("compile error in {url}: {e:?}")))?;
        if republish {
            self.publish_fresh_namespace(vm, url, &record)?;
        }
        // NOTE: the rewrite evaluates dynamic-import targets inline (see
        // [`rewrite`](Self::rewrite)); on the cache fast path
        // [`evaluate_compiled`](Self::evaluate_compiled) replays them
        // from [`RewriteInfo::dyn_deps`] in the same position.
        let (rewritten, info) = self.rewrite(vm, url, source, &program, &record)?;
        if info.cacheable {
            self.store_compiled(url, source, &record, rewritten.clone(), info.dyn_deps);
        }
        self.run_rewritten(vm, url, &rewritten)
    }

    /// Insert the `Evaluating` entry, publish a fresh (initially empty)
    /// namespace, then evaluate the static dependencies: cyclic imports
    /// observe the partial namespace, and the epilogue populates it when
    /// the body finishes evaluating. The order mirrors the original
    /// `load_and_evaluate` exactly (insert → resolve → publish →
    /// evaluate). Shared by the slow path and the compile-cache fast
    /// path.
    fn publish_fresh_namespace(
        &mut self,
        vm: &mut Vm,
        url: &str,
        record: &ModuleRecord,
    ) -> Result<(), ModuleError> {
        self.modules.insert(
            url.to_owned(),
            ModuleEntry {
                record: record.clone(),
                state: ModuleState::Evaluating,
            },
        );
        // Resolve the static dependency URLs before publishing anything:
        // a resolution failure reports with no namespace left behind.
        let deps = self.static_deps(url, record)?;
        self.ensure_registry(vm)?;
        self.register_namespace(vm, url)?;
        for dep in deps {
            self.ensure_evaluated(vm, &dep)?;
        }
        Ok(())
    }

    /// Run the rewritten script and mark the module evaluated.
    ///
    /// The URL is stamped onto every template of the rewritten
    /// compilation unit as its `[[ScriptOrModule]]` (see
    /// [`Vm::eval_module_source`]), so a dynamic `import()` issued from
    /// any function the module defines resolves against the importing
    /// module — during evaluation, or after it (event handlers, `.then`
    /// reactions, resumed async functions), per ES2024 §8.3.1
    /// `GetActiveScriptOrModule`. The compile-cache fast path flows
    /// through here too ([`evaluate_compiled`](Self::evaluate_compiled)
    /// calls this), so its templates are stamped the same way.
    fn run_rewritten(
        &mut self,
        vm: &mut Vm,
        url: &str,
        rewritten: &str,
    ) -> Result<(), ModuleError> {
        vm.eval_module_source(rewritten, url)
            .map_err(|e| ModuleError::Eval(format!("module {url}: {e}")))?;
        if let Some(entry) = self.modules.get_mut(url) {
            entry.state = ModuleState::Evaluated;
        }
        Ok(())
    }

    /// Evaluate a compile-cache hit: the module record and rewritten
    /// script are reused, but the namespace is fresh (published by this
    /// loader) and every dependency is evaluated anew — the same
    /// stateful-fixture isolation as the slow path.
    ///
    /// A dynamic-import dep can fail at replay even though it succeeded
    /// when the entry was cached (its source changed since: hash miss →
    /// slow re-evaluate → fetch/parse/top-level failure). The slow path
    /// turns that into a spliced `Promise.reject(...)` and still
    /// evaluates the parent, so on failure fall back to
    /// [`evaluate_slow`](Self::evaluate_slow) (re-parse + re-rewrite)
    /// instead of failing the whole parent load.
    fn evaluate_compiled(
        &mut self,
        vm: &mut Vm,
        url: &str,
        source: &str,
        compiled: &CachedModule,
    ) -> Result<(), ModuleError> {
        self.publish_fresh_namespace(vm, url, &compiled.record)?;
        // Dynamic-import targets: the cached rewrite spliced
        // `Promise.resolve(<namespace>)` for these; re-evaluate each so
        // the namespace read at run time is this load's fresh one, not a
        // previous load's. (`compiled` is an owned clone from the cache
        // lookup, so no cache borrow is held across the re-entrant
        // `ensure_evaluated` calls.)
        for dep in &compiled.dyn_deps {
            if self.ensure_evaluated(vm, dep).is_err() {
                // The namespace is already published and the static
                // deps already evaluated above; re-publishing here
                // would orphan the first namespace.
                return self.evaluate_slow(vm, url, source, false);
            }
        }
        self.run_rewritten(vm, url, &compiled.rewritten)
    }

    /// Static dependency URLs of a module record (`import` sources and
    /// `export ... from` sources), de-duplicated, in source order.
    fn static_deps(&self, url: &str, record: &ModuleRecord) -> Result<Vec<String>, ModuleError> {
        let mut deps: Vec<String> = Vec::new();
        for d in &record.imports {
            let dep = self.resolve(&d.source, url)?;
            if !deps.contains(&dep) {
                deps.push(dep);
            }
        }
        for e in &record.exports {
            let src = match e {
                ExportEntry::Named {
                    source: Some(s), ..
                }
                | ExportEntry::All { source: s, .. } => Some(s),
                _ => None,
            };
            if let Some(s) = src {
                let dep = self.resolve(s, url)?;
                if !deps.contains(&dep) {
                    deps.push(dep);
                }
            }
        }
        Ok(deps)
    }

    /// Compile-cache lookup: `Some` when the cache holds a parse+rewrite
    /// of `url` built from byte-identical source. Both independent
    /// source hashes must match: a single-hash collision between two
    /// different sources under one URL is a miss, never stale reuse.
    fn lookup_compiled(&self, url: &str, source: &str) -> Option<CachedModule> {
        let cache = COMPILE_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        let entry = cache.modules.get(url)?.clone();
        let hit =
            entry.source_hash == hash_source(source) && entry.source_hash2 == hash_source2(source);
        #[cfg(test)]
        if hit {
            COMPILE_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
        }
        hit.then_some(entry)
    }

    /// Insert a freshly rewritten module into the compile cache (see
    /// [`ModuleCompileCache`]). The lock is held only for the insert.
    fn store_compiled(
        &self,
        url: &str,
        source: &str,
        record: &ModuleRecord,
        rewritten: String,
        dyn_deps: Vec<String>,
    ) {
        let mut cache = COMPILE_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if cache.modules.len() >= COMPILE_CACHE_MAX_ENTRIES {
            cache.modules.clear();
        }
        cache.modules.insert(
            url.to_owned(),
            CachedModule {
                source_hash: hash_source(source),
                source_hash2: hash_source2(source),
                record: record.clone(),
                rewritten,
                dyn_deps,
            },
        );
    }

    /// Install the registry object on the global if it is not there yet.
    fn ensure_registry(&self, vm: &mut Vm) -> Result<(), ModuleError> {
        if vm.get_global(REGISTRY_KEY).is_some() {
            return Ok(());
        }
        let heap = vm.heap_mut();
        let obj = heap
            .alloc_object(Object {
                kind: ObjectKind::Ordinary,
                prototype: heap
                    .object_prototype
                    .map(Value::Object)
                    .unwrap_or(Value::Null),
                ..Default::default()
            })
            .map_err(|_| ModuleError::Eval("out of memory allocating module registry".into()))?;
        vm.set_global(REGISTRY_KEY, Value::Object(obj));
        Ok(())
    }

    /// Allocate this module's (initially empty) namespace object and publish
    /// it in the registry *before* the module body runs, so cyclic imports
    /// observe it.
    fn register_namespace(&self, vm: &mut Vm, url: &str) -> Result<(), ModuleError> {
        let reg = match vm.get_global(REGISTRY_KEY) {
            Some(Value::Object(ro)) => ro,
            _ => return Err(ModuleError::Eval("module registry missing".into())),
        };
        let heap = vm.heap_mut();
        let ns = heap
            .alloc_object(Object {
                kind: ObjectKind::Ordinary,
                prototype: heap
                    .object_prototype
                    .map(Value::Object)
                    .unwrap_or(Value::Null),
                ..Default::default()
            })
            .map_err(|_| ModuleError::Eval("out of memory allocating module namespace".into()))?;
        let base = heap.protect(Value::Object(ns));
        let r = heap
            .define_property(reg, Wtf8::from_str(url), Property::data(Value::Object(ns)))
            .map_err(|_| ModuleError::Eval("out of memory registering module namespace".into()));
        heap.release_temp(base);
        r
    }

    /// Provenance of every export of every loaded module: URL -> exported
    /// name -> (defining module URL, defining exported name). Reading
    /// `REG["<def_url>"]["<def_name>"]` yields the *live* value, because an
    /// exported mutable binding's defining module routes all of its own
    /// accesses through its namespace object.
    ///
    /// Computed as a monotone fixpoint: names are only ever added, so
    /// re-export cycles terminate. `export *` never re-exports `default`,
    /// per spec.
    /// The source-module name for `export { src as exported } from "s"` in
    /// the module at `url`. The [`ModuleRecord`] discards this for
    /// re-exports (`local: None`), so it is recovered from the AST.
    /// Returns `None` if the source cannot be fetched/parsed or the
    /// specifier is not found.
    fn reexport_source_name(&self, url: &str, s: &str, exported: &str) -> Option<String> {
        let source = self.fetch_source(url).ok()?;
        let program = parser::parse(&source).ok()?;
        for stmt in &program.body {
            if let Stmt::Export(ExportDecl::Named {
                specifiers,
                source: Some(src),
                ..
            }) = stmt
                && src == s
            {
                for spec in specifiers {
                    if module_name_str(&spec.exported) == exported {
                        return Some(module_name_str(&spec.local));
                    }
                }
            }
        }
        None
    }

    fn export_provenance(&self) -> HashMap<String, HashMap<String, (String, String)>> {
        let mut prov: HashMap<String, HashMap<String, (String, String)>> = HashMap::new();
        // Seed with locally declared names.
        for (u, entry) in &self.modules {
            let mut m = HashMap::new();
            for e in &entry.record.exports {
                match e {
                    ExportEntry::Named {
                        exported,
                        source: None,
                        ..
                    } => {
                        m.insert(exported.clone(), (u.clone(), exported.clone()));
                    }
                    ExportEntry::Default => {
                        m.insert("default".to_owned(), (u.clone(), "default".to_owned()));
                    }
                    ExportEntry::Decl(names) => {
                        for n in names {
                            m.insert(n.clone(), (u.clone(), n.clone()));
                        }
                    }
                    ExportEntry::All {
                        exported: Some(n), ..
                    } => {
                        m.insert(n.clone(), (u.clone(), n.clone()));
                    }
                    ExportEntry::Named { .. } | ExportEntry::All { .. } => {}
                }
            }
            prov.insert(u.clone(), m);
        }
        // Propagate re-exports to a fixpoint. Iteration order is
        // deterministic (sorted URLs, source-order exports) and each name's
        // provenance is set once, on first insertion.
        loop {
            let mut changed = false;
            let mut urls: Vec<String> = prov.keys().cloned().collect();
            urls.sort_unstable();
            for u in &urls {
                let entry = &self.modules[u.as_str()];
                let mut add = Vec::new();
                for e in &entry.record.exports {
                    match e {
                        ExportEntry::Named {
                            local,
                            exported,
                            source: Some(s),
                        } => {
                            // Re-export from another module. The record stores
                            // `local: None` here, so the source-module name comes
                            // from the AST (needed for `export { a as b } from`).
                            // Falls back to `exported` for the non-aliased case.
                            let src_name = match local {
                                Some(l) => l.clone(),
                                None => self
                                    .reexport_source_name(u, s, exported)
                                    .unwrap_or_else(|| exported.clone()),
                            };
                            if let Ok(dep) = self.resolve(s, u)
                                && let Some(dm) = prov.get(&dep)
                                && let Some(p) = dm.get(&src_name)
                            {
                                add.push((exported.clone(), p.clone()));
                            }
                        }
                        ExportEntry::All {
                            source: s,
                            exported: None,
                        } => {
                            if let Ok(dep) = self.resolve(s, u)
                                && let Some(dm) = prov.get(&dep)
                            {
                                for (n, p) in dm {
                                    if n != "default" {
                                        add.push((n.clone(), p.clone()));
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                let m = prov.get_mut(u).expect("seeded above");
                for (n, p) in add {
                    if let std::collections::hash_map::Entry::Vacant(e) = m.entry(n) {
                        e.insert(p);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        prov
    }
}

/// What [`ModuleLoader::rewrite`] reports alongside the rewritten
/// source: the information a compile-cache hit needs to replay the
/// rewrite's load-time side effects.
#[derive(Debug, Default)]
struct RewriteInfo {
    /// URLs of dynamic-`import()` targets the rewrite resolved and
    /// evaluated while splicing `Promise.resolve(<namespace>)`
    /// replacements, de-duplicated, in source order.
    dyn_deps: Vec<String>,
    /// False when any dynamic-import site embedded a rejection: the
    /// rejection string bakes in an evaluation outcome, which is not
    /// safe to replay from a cache entry.
    cacheable: bool,
}

// ---------------------------------------------------------------------------
// Source rewriting
// ---------------------------------------------------------------------------

/// A single splice: replace `source[start..end]` with `text`.
struct SourceEdit {
    start: u32,
    end: u32,
    text: String,
}

/// Apply non-overlapping edits (sorted by start; contained duplicates from
/// nested rewrites are skipped).
fn apply_edits(source: &str, mut edits: Vec<SourceEdit>) -> String {
    edits.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
    let mut out = String::with_capacity(source.len() + 256);
    let mut pos = 0u32;
    for e in edits {
        if e.start < pos || e.end < e.start || e.end as usize > source.len() {
            continue;
        }
        out.push_str(&source[pos as usize..e.start as usize]);
        out.push_str(&e.text);
        pos = e.end;
    }
    out.push_str(&source[pos as usize..]);
    out
}

/// Quote `s` as a JS string literal.
fn js_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
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

/// JS source expression evaluating to the namespace object of `url`.
fn ns_access(url: &str) -> String {
    format!("{}[{}]", REGISTRY_SRC, js_str(url))
}

fn module_name_str(n: &ModuleName) -> String {
    match n {
        ModuleName::Ident(s, _) => s.clone(),
        // Module specifiers feed URL resolution (`&str`): lossy.
        ModuleName::Str(s, _) => s.to_utf8_lossy().into_owned(),
    }
}

/// Names bound by an `export <decl>` inner declaration statement.
fn declared_names(s: &Stmt) -> Vec<String> {
    let mut out = Vec::new();
    match s {
        Stmt::VarDecl { decls, .. } => {
            for d in decls {
                pat_names(&d.id, &mut out);
            }
        }
        Stmt::FunctionDecl { name, .. } => out.push(name.clone()),
        Stmt::ClassDecl { name: Some(n), .. } => out.push(n.clone()),
        _ => {}
    }
    out
}

fn pat_names(p: &Pat, out: &mut Vec<String>) {
    match p {
        Pat::Ident(n, _) => out.push(n.clone()),
        Pat::Array(elems, _) => {
            for el in elems.iter().flatten() {
                let ArrayPatElem::Pat(p) = el;
                pat_names(p, out);
            }
        }
        Pat::Object(props, _) => {
            for pr in props {
                match pr {
                    ObjectPatProp::KeyValue { value, .. } => pat_names(value, out),
                    ObjectPatProp::Shorthand { name, .. } => out.push(name.clone()),
                    ObjectPatProp::Rest(p, _) => pat_names(p, out),
                }
            }
        }
        Pat::Rest(p, _) | Pat::Assign(p, _, _) => pat_names(p, out),
        // Member targets bind nothing.
        Pat::Member(..) => {}
    }
}

impl ModuleLoader {
    /// Rewrite module `source` into an equivalent classic script:
    /// imports become `const` reads from the registry, `export default`
    /// becomes a namespace write, `import.meta` is inlined, literal
    /// `import()` becomes `Promise.resolve(<namespace>)`, and an epilogue
    /// captures the exported bindings into the namespace object.
    fn rewrite(
        &mut self,
        vm: &mut Vm,
        url: &str,
        source: &str,
        program: &Program,
        _record: &ModuleRecord,
    ) -> Result<(String, RewriteInfo), ModuleError> {
        // 1. Find `import.meta` and literal `import()` sites anywhere in the
        //    module (nested functions included).
        let mut walker = Walker::default();
        walker.walk_program(program);

        // Dynamic imports resolve+load now; failures become rejections.
        let mut info = RewriteInfo {
            cacheable: true,
            ..RewriteInfo::default()
        };
        let mut dyn_edits: Vec<(Span, String)> = Vec::new();
        for (span, spec) in &walker.sites.imports {
            let replacement = match self.resolve(spec, url) {
                Ok(dep) => match self.ensure_evaluated(vm, &dep) {
                    Ok(()) => {
                        if !info.dyn_deps.contains(&dep) {
                            info.dyn_deps.push(dep.clone());
                        }
                        format!("(Promise.resolve({}))", ns_access(&dep))
                    }
                    Err(e) => {
                        // The rejection bakes in this load's evaluation
                        // outcome: not replayable from a cache entry.
                        info.cacheable = false;
                        format!(
                            "(Promise.reject(new TypeError({})))",
                            js_str(&format!("dynamic import of {spec:?} failed: {e}"))
                        )
                    }
                },
                Err(e) => {
                    info.cacheable = false;
                    format!(
                        "(Promise.reject(new TypeError({})))",
                        js_str(&format!("cannot resolve dynamic import {spec:?}: {e}"))
                    )
                }
            };
            dyn_edits.push((*span, replacement));
        }
        let meta_edits: Vec<(Span, String)> = walker
            .sites
            .metas
            .iter()
            .map(|sp| (*sp, format!("({{url:{}}})", js_str(url))))
            .collect();

        // 1b. Live-binding analysis. Provenance resolves every export to
        // its defining module; the shadow analysis tells which bindings
        // can be rewritten to live namespace reads.
        let prov = self.export_provenance();
        let live = !walker.has_with;
        let (nested_declared, all_declared) = walker.shadow_info();
        let ns = ns_access(url);

        // Import locals -> live read replacement (only when unshadowed).
        // Namespace imports (`* as ns`) are live by object identity and
        // need no rewriting.
        let mut import_reads: HashMap<String, String> = HashMap::new();
        let mut import_locals: HashSet<String> = HashSet::new();
        for stmt in &program.body {
            if let Stmt::Import(d) = stmt {
                let dep = self.resolve(&d.source, url)?;
                let dep_prov = prov.get(&dep);
                let target = |imported: &str| -> (String, String) {
                    dep_prov
                        .and_then(|m| m.get(imported))
                        .cloned()
                        .unwrap_or_else(|| (dep.clone(), imported.to_owned()))
                };
                if let Some((name, _)) = &d.default {
                    import_locals.insert(name.clone());
                    if live && !all_declared.contains(name) {
                        let (pu, pn) = target("default");
                        import_reads.insert(
                            name.clone(),
                            format!("({}[{}])", ns_access(&pu), js_str(&pn)),
                        );
                    }
                }
                if let Some((name, _)) = &d.namespace {
                    import_locals.insert(name.clone());
                }
                for s in &d.named {
                    import_locals.insert(s.local.clone());
                    if live && !all_declared.contains(&s.local) {
                        let imported = module_name_str(&s.imported);
                        let (pu, pn) = target(&imported);
                        import_reads.insert(
                            s.local.clone(),
                            format!("({}[{}])", ns_access(&pu), js_str(&pn)),
                        );
                    }
                }
            }
        }

        // Mutable exported locals: top-level `let`/`var` names that are
        // exported. Their references are rewritten to the module's own
        // namespace, which becomes the single source of truth.
        let mut top_let_var: HashSet<String> = HashSet::new();
        let mut decl_ends: HashMap<String, u32> = HashMap::new();
        for stmt in &program.body {
            let inner = match stmt {
                Stmt::Export(ExportDecl::Decl { inner, .. }) => inner.as_ref(),
                s => s,
            };
            if let Stmt::VarDecl { kind, decls } = inner
                && matches!(kind, VarKind::Let | VarKind::Var)
            {
                let end = decls.iter().map(|d| d.span.end).max().unwrap_or(0);
                for d in decls {
                    let mut names = Vec::new();
                    pat_names(&d.id, &mut names);
                    for n in names {
                        top_let_var.insert(n.clone());
                        decl_ends.insert(n, end);
                    }
                }
            }
        }
        // local name -> exported name (first one wins for the live slot).
        let mut export_mut: HashMap<String, String> = HashMap::new();
        for stmt in &program.body {
            if let Stmt::Export(e) = stmt {
                match e {
                    ExportDecl::Decl { inner, .. } => {
                        if let Stmt::VarDecl { kind, .. } = inner.as_ref()
                            && matches!(kind, VarKind::Let | VarKind::Var)
                        {
                            for n in declared_names(inner) {
                                if live
                                    && !nested_declared.contains(&n)
                                    && !export_mut.contains_key(&n)
                                {
                                    export_mut.insert(n.clone(), n.clone());
                                }
                            }
                        }
                    }
                    ExportDecl::Named {
                        specifiers,
                        source: None,
                        ..
                    } => {
                        for s in specifiers {
                            if let ModuleName::Ident(local, _) = &s.local
                                && live
                                && top_let_var.contains(local)
                                && !nested_declared.contains(local)
                                && !export_mut.contains_key(local)
                            {
                                export_mut.insert(local.clone(), module_name_str(&s.exported));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        let live_exported: HashSet<String> = export_mut.values().cloned().collect();
        // Sync the namespace right after each mutable declaration, so
        // reads before the epilogue observe the initial value.
        let mut sync_edits: Vec<SourceEdit> = Vec::new();
        for (local, exported) in &export_mut {
            if let Some(&end) = decl_ends.get(local) {
                sync_edits.push(SourceEdit {
                    start: end,
                    end,
                    text: format!(";{}[{}]={};", ns, js_str(exported), local),
                });
            }
        }

        // 1c. Rewrite references to live namespace reads.
        let import_reads_for_epilogue = import_reads.clone();
        let export_mut_for_epilogue = export_mut.clone();
        let mut rwr = RefRewriter {
            import_reads,
            export_mut,
            self_ns: ns.clone(),
            edits: Vec::new(),
        };
        if live {
            rwr.rw_program(program);
        }
        let ref_edits: Vec<(Span, String)> = rwr.edits;

        // 2. Top-level import/export statements.
        let mut top_edits: Vec<SourceEdit> = Vec::new();
        let mut default_spans: Vec<Span> = Vec::new();
        let mut epilogue: Vec<String> = Vec::new();
        let mut explicit: HashSet<String> = HashSet::new();
        let mut star_seen: HashSet<String> = HashSet::new();

        for stmt in &program.body {
            match stmt {
                Stmt::Import(d) => {
                    let dep = self.resolve(&d.source, url)?;
                    let dep_ns = ns_access(&dep);
                    let mut parts: Vec<String> = Vec::new();
                    if let Some((name, _)) = &d.default {
                        parts.push(format!("{name}={dep_ns}[\"default\"]"));
                    }
                    if let Some((name, _)) = &d.namespace {
                        parts.push(format!("{name}={dep_ns}"));
                    }
                    for s in &d.named {
                        let imported = module_name_str(&s.imported);
                        parts.push(format!("{}={}[{}]", s.local, dep_ns, js_str(&imported)));
                    }
                    let text = if parts.is_empty() {
                        ";".to_owned()
                    } else {
                        format!("const {};", parts.join(","))
                    };
                    top_edits.push(SourceEdit {
                        start: d.span.start,
                        end: d.span.end,
                        text,
                    });
                }
                Stmt::Export(e) => match e {
                    ExportDecl::Default { expr, span } => {
                        // Replace the whole statement; inner `import.meta` /
                        // `import()` edits are applied to the expression
                        // slice first so they survive the splice.
                        let espe = expr.span();
                        let slice = &source[espe.start as usize..espe.end as usize];
                        let inner: Vec<SourceEdit> = dyn_edits
                            .iter()
                            .chain(meta_edits.iter())
                            .chain(ref_edits.iter())
                            .filter(|(sp, _)| sp.start >= espe.start && sp.end <= espe.end)
                            .map(|(sp, text)| SourceEdit {
                                start: sp.start - espe.start,
                                end: sp.end - espe.start,
                                text: text.clone(),
                            })
                            .collect();
                        default_spans.push(*span);
                        let expr_src = apply_edits(slice, inner);
                        top_edits.push(SourceEdit {
                            start: span.start,
                            end: span.end,
                            text: format!("{ns}[\"default\"]=({expr_src});"),
                        });
                        explicit.insert("default".to_owned());
                    }
                    ExportDecl::Named {
                        specifiers,
                        source: from,
                        span,
                    } => match from {
                        None => {
                            // Strip the statement: the epilogue below
                            // publishes the bindings to the namespace.
                            // (The loader evals the rewritten source via
                            // `Vm::eval`, which parses sloppy scripts, so
                            // no `export` may remain.)
                            top_edits.push(SourceEdit {
                                start: span.start,
                                end: span.end,
                                text: ";".to_owned(),
                            });
                            for s in specifiers {
                                let exported = module_name_str(&s.exported);
                                // A non-identifier local without `from`
                                // cannot name a binding; skip (documented).
                                if let ModuleName::Ident(local, _) = &s.local {
                                    explicit.insert(exported.clone());
                                    if let Some(rep) = import_reads_for_epilogue.get(local) {
                                        // Re-export of an import: read the
                                        // live provenance, not the snapshot
                                        // `const`.
                                        epilogue.push(format!(
                                            "{}[{}]={rep};",
                                            ns,
                                            js_str(&exported)
                                        ));
                                    } else if import_locals.contains(local) {
                                        // Shadowed import: snapshot `const`
                                        // fallback.
                                        epilogue.push(format!(
                                            "{}[{}]={local};",
                                            ns,
                                            js_str(&exported)
                                        ));
                                    } else if let Some(live) = export_mut_for_epilogue.get(local) {
                                        if *live == exported {
                                            // Live via reference rewriting;
                                            // the namespace was synced at
                                            // declaration. No copy needed.
                                        } else {
                                            // Alias of a live binding:
                                            // snapshot the canonical name.
                                            epilogue.push(format!(
                                                "{}[{}]={}[{}];",
                                                ns,
                                                js_str(&exported),
                                                ns,
                                                js_str(live)
                                            ));
                                        }
                                    } else {
                                        epilogue.push(format!(
                                            "{}[{}]={local};",
                                            ns,
                                            js_str(&exported)
                                        ));
                                    }
                                }
                            }
                        }
                        Some(dep_src) => {
                            let dep = self.resolve(dep_src, url)?;
                            // Strip the statement; the epilogue publishes.
                            top_edits.push(SourceEdit {
                                start: span.start,
                                end: span.end,
                                text: ";".to_owned(),
                            });
                            for s in specifiers {
                                let imported = module_name_str(&s.local);
                                let exported = module_name_str(&s.exported);
                                explicit.insert(exported.clone());
                                // Read through to the defining module so
                                // re-export chains stay live.
                                let (pu, pn) = prov
                                    .get(&dep)
                                    .and_then(|m| m.get(&imported))
                                    .cloned()
                                    .unwrap_or_else(|| (dep.clone(), imported.clone()));
                                epilogue.push(format!(
                                    "{}[{}]={}[{}];",
                                    ns,
                                    js_str(&exported),
                                    ns_access(&pu),
                                    js_str(&pn)
                                ));
                            }
                        }
                    },
                    ExportDecl::All {
                        source: dep_src,
                        exported,
                        span,
                    } => {
                        let dep = self.resolve(dep_src, url)?;
                        // Strip the statement; the epilogue publishes.
                        top_edits.push(SourceEdit {
                            start: span.start,
                            end: span.end,
                            text: ";".to_owned(),
                        });
                        let dep_ns = ns_access(&dep);
                        match exported {
                            Some(name) => {
                                explicit.insert(name.clone());
                                epilogue.push(format!("{}[{}]={dep_ns};", ns, js_str(name)));
                            }
                            None => {
                                // `export *` never re-exports `default`;
                                // explicit and earlier star exports win.
                                // Reads go to the defining module so the
                                // chain stays live.
                                let mut names: Vec<(&String, &(String, String))> = prov
                                    .get(&dep)
                                    .map(|m| m.iter().collect())
                                    .unwrap_or_default();
                                names.sort_by(|a, b| a.0.cmp(b.0));
                                for (k, (pu, pn)) in names {
                                    if k == "default"
                                        || explicit.contains(k)
                                        || star_seen.contains(k)
                                    {
                                        continue;
                                    }
                                    star_seen.insert(k.clone());
                                    epilogue.push(format!(
                                        "{}[{}]={}[{}];",
                                        ns,
                                        js_str(k),
                                        ns_access(pu),
                                        js_str(pn)
                                    ));
                                }
                            }
                        }
                    }
                    ExportDecl::Decl { inner, span } => {
                        // Strip the `export` keyword, keeping the
                        // declaration: `export const x = 1;` -> `const x = 1;`.
                        // `span` starts at the `export` token (6 bytes).
                        top_edits.push(SourceEdit {
                            start: span.start,
                            end: span.start + 6,
                            text: String::new(),
                        });
                        for name in declared_names(inner) {
                            explicit.insert(name.clone());
                            // Live mutable bindings were rewritten to
                            // namespace reads and synced at declaration;
                            // no copy needed.
                            if !live_exported.contains(&name) {
                                epilogue.push(format!("{}[{}]={name};", ns, js_str(&name)));
                            }
                        }
                    }
                },
                _ => {}
            }
        }

        // 3. Merge: top-level edits plus walker edits outside any
        //    `export default` statement (those were incorporated above).
        let mut edits = top_edits;
        edits.extend(sync_edits);
        for (sp, text) in dyn_edits
            .iter()
            .chain(meta_edits.iter())
            .chain(ref_edits.iter())
        {
            if default_spans
                .iter()
                .any(|d| d.start <= sp.start && sp.end <= d.end)
            {
                continue;
            }
            edits.push(SourceEdit {
                start: sp.start,
                end: sp.end,
                text: text.clone(),
            });
        }
        let mut out = apply_edits(source, edits);
        if !epilogue.is_empty() {
            out.push_str("\n;");
            for stmt in &epilogue {
                out.push_str(stmt);
            }
        }
        Ok((out, info))
    }
}

// ---------------------------------------------------------------------------
// AST walker: collects `import.meta` and literal `import("...")` sites
// ---------------------------------------------------------------------------

/// Spans collected by [`Walker`].
#[derive(Default)]
struct DynamicSites {
    /// Every `import.meta` occurrence.
    metas: Vec<Span>,
    /// Every `import("<literal>")` call: span + cooked specifier.
    /// Non-literal `import(x)` calls keep their form and run through the
    /// VM's `Op::DynamicImport`, which queues a host dynamic import
    /// settled at the next microtask checkpoint.
    imports: Vec<(Span, String)>,
}

#[derive(Default)]
struct Walker {
    sites: DynamicSites,
    /// Function-scope stack for shadow analysis. `scopes[0]` is the module
    /// top-level. `import` bindings are never added: a name present in any
    /// scope is declared somewhere in the module and may not refer to the
    /// import. Block scopes are folded into the enclosing function scope
    /// (conservative: may miss a live-binding rewrite, never miscompiles).
    scopes: Vec<HashSet<String>>,
    /// Persistent accumulator: every name ever declared in a nested scope.
    /// Scopes are popped after traversal, so this retains the shadowing
    /// information that `scopes` alone would lose.
    nested_seen: HashSet<String>,
    /// `with` makes static reference resolution unsound; live-binding
    /// rewrites are skipped for such modules.
    has_with: bool,
}

impl Walker {
    fn walk_program(&mut self, program: &Program) {
        self.scopes.push(HashSet::new());
        self.walk_stmts(&program.body);
    }

    /// `(nested, all)`: names declared in function scopes nested inside the
    /// module top-level, and names declared anywhere (excluding imports).
    fn shadow_info(&self) -> (HashSet<String>, HashSet<String>) {
        // (names declared in any nested scope, names declared anywhere).
        // Conservative: any nested declaration blocks live rewriting of
        // that name throughout the module.
        let mut all = self.nested_seen.clone();
        if let Some(top) = self.scopes.first() {
            all.extend(top.iter().cloned());
        }
        (self.nested_seen.clone(), all)
    }

    fn declare(&mut self, name: &str) {
        // Persist nested declarations: scopes are popped after traversal.
        if self.scopes.len() > 1 {
            self.nested_seen.insert(name.to_owned());
        }
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_owned());
        }
    }

    fn declare_pat(&mut self, p: &Pat) {
        let mut names = Vec::new();
        pat_names(p, &mut names);
        for n in names {
            self.declare(&n);
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashSet::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn walk_stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.walk_stmt(s);
        }
    }

    fn walk_stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Expr(e) | Stmt::Throw(e) => self.walk_expr(e),
            Stmt::Block(b) => self.walk_stmts(b),
            Stmt::VarDecl { decls, .. } => {
                for d in decls {
                    self.declare_pat(&d.id);
                    self.walk_var_decl(d);
                }
            }
            Stmt::FunctionDecl {
                name, params, body, ..
            } => {
                self.declare(name);
                self.push_scope();
                for p in params {
                    self.walk_param(p);
                }
                self.walk_stmts(body);
                self.pop_scope();
            }
            Stmt::ClassDecl {
                name,
                superclass,
                body,
                ..
            } => {
                if let Some(n) = name {
                    self.declare(n);
                }
                if let Some(s) = superclass {
                    self.walk_expr(s);
                }
                for m in body {
                    self.walk_class_member(m);
                }
            }
            Stmt::Return(e) => {
                if let Some(e) = e {
                    self.walk_expr(e);
                }
            }
            Stmt::If {
                cond,
                then_b,
                else_b,
            } => {
                self.walk_expr(cond);
                self.walk_stmt(then_b);
                if let Some(e) = else_b {
                    self.walk_stmt(e);
                }
            }
            Stmt::While { cond, body } | Stmt::DoWhile { cond, body } => {
                self.walk_expr(cond);
                self.walk_stmt(body);
            }
            Stmt::For {
                init,
                cond,
                update,
                body,
            } => {
                if let Some(i) = init {
                    match i {
                        ForInit::Var(_, decls) => {
                            for d in decls {
                                self.walk_var_decl(d);
                            }
                        }
                        ForInit::Expr(e) => self.walk_expr(e),
                    }
                }
                if let Some(c) = cond {
                    self.walk_expr(c);
                }
                if let Some(u) = update {
                    self.walk_expr(u);
                }
                self.walk_stmt(body);
            }
            Stmt::ForInOf { lhs, rhs, body, .. } => {
                match lhs {
                    ForLhs::Ident(..) => {}
                    ForLhs::Pat(p) => self.walk_pat(p),
                    ForLhs::Member(e) => self.walk_expr(e),
                }
                self.walk_expr(rhs);
                self.walk_stmt(body);
            }
            Stmt::Switch { disc, cases } => {
                self.walk_expr(disc);
                for c in cases {
                    self.walk_switch_case(c);
                }
            }
            Stmt::Try {
                block,
                handler,
                finalizer,
            } => {
                self.walk_stmts(block);
                if let Some(h) = handler {
                    self.walk_catch(h);
                }
                if let Some(f) = finalizer {
                    self.walk_stmts(f);
                }
            }
            Stmt::Labeled { body, .. } => self.walk_stmt(body),
            Stmt::With { obj, body, .. } => {
                // `with` is rejected in strict-mode modules, but if one is
                // present the reference analysis below is unsound.
                self.has_with = true;
                self.walk_expr(obj);
                self.walk_stmt(body);
            }
            Stmt::Import(..) => {}
            Stmt::Export(e) => {
                // Only these forms can contain expressions; the rest are
                // handled by the statement-level rewrite. Sites found here
                // are spliced into the rewritten source (for `Default`, via
                // the expression-slice path in `rewrite`).
                match e {
                    ExportDecl::Default { expr, .. } => self.walk_expr(expr),
                    ExportDecl::Decl { inner, .. } => self.walk_stmt(inner),
                    _ => {}
                }
            }
            Stmt::Break(..) | Stmt::Continue(..) | Stmt::Debugger | Stmt::Empty => {}
        }
    }

    fn walk_var_decl(&mut self, d: &VarDecl) {
        self.walk_pat(&d.id);
        if let Some(init) = &d.init {
            self.walk_expr(init);
        }
    }

    fn walk_param(&mut self, p: &Param) {
        self.declare_pat(&p.pat);
        self.walk_pat(&p.pat);
    }

    fn walk_catch(&mut self, c: &CatchClause) {
        if let Some(p) = &c.param {
            self.declare_pat(&p.pat);
            self.walk_param(p);
        }
        self.walk_stmts(&c.body);
    }

    fn walk_switch_case(&mut self, c: &SwitchCase) {
        if let Some(t) = &c.test {
            self.walk_expr(t);
        }
        self.walk_stmts(&c.body);
    }

    fn walk_pat(&mut self, p: &Pat) {
        match p {
            Pat::Ident(..) => {}
            Pat::Array(elems, _) => {
                for el in elems.iter().flatten() {
                    let ArrayPatElem::Pat(p) = el;
                    self.walk_pat(p);
                }
            }
            Pat::Object(props, _) => {
                for pr in props {
                    match pr {
                        ObjectPatProp::KeyValue { key, value, .. } => {
                            self.walk_prop_key(key);
                            self.walk_pat(value);
                        }
                        ObjectPatProp::Shorthand { .. } => {}
                        ObjectPatProp::Rest(p, _) => self.walk_pat(p),
                    }
                }
            }
            Pat::Rest(p, _) => self.walk_pat(p),
            Pat::Assign(p, e, _) => {
                self.walk_pat(p);
                self.walk_expr(e);
            }
            // The base object and computed key are reads.
            Pat::Member(e) => self.walk_expr(e),
        }
    }

    fn walk_prop_key(&mut self, k: &PropKey) {
        if let PropKey::Computed(e) = k {
            self.walk_expr(e);
        }
    }

    fn walk_class_key(&mut self, k: &ClassKey) {
        if let ClassKey::Public(pk) = k {
            self.walk_prop_key(pk);
        }
    }

    fn walk_prop(&mut self, p: &PropDef) {
        match p {
            PropDef::KeyValue { key, value, .. } => {
                self.walk_prop_key(key);
                self.walk_expr(value);
            }
            PropDef::Shorthand { .. } => {}
            PropDef::Spread(e, _) => self.walk_expr(e),
            PropDef::Method {
                key, params, body, ..
            } => {
                self.walk_prop_key(key);
                self.push_scope();
                for p in params {
                    self.walk_param(p);
                }
                self.walk_stmts(body);
                self.pop_scope();
            }
            PropDef::Getter { key, body, .. } => {
                self.walk_prop_key(key);
                self.push_scope();
                self.walk_stmts(body);
                self.pop_scope();
            }
            PropDef::Setter {
                key, param, body, ..
            } => {
                self.walk_prop_key(key);
                self.push_scope();
                self.walk_param(param);
                self.walk_stmts(body);
                self.pop_scope();
            }
        }
    }

    fn walk_class_member(&mut self, m: &ClassMember) {
        match m {
            ClassMember::Method {
                key, params, body, ..
            } => {
                self.walk_class_key(key);
                self.push_scope();
                for p in params {
                    self.walk_param(p);
                }
                self.walk_stmts(body);
                self.pop_scope();
            }
            ClassMember::Field { key, init, .. } => {
                self.walk_class_key(key);
                if let Some(e) = init {
                    self.walk_expr(e);
                }
            }
            ClassMember::StaticBlock { body, .. } => {
                self.push_scope();
                self.walk_stmts(body);
                self.pop_scope();
            }
        }
    }

    fn walk_expr(&mut self, e: &Expr) {
        match e {
            Expr::Ident(..)
            | Expr::PrivateIdent(..)
            | Expr::This(..)
            | Expr::Super(..)
            | Expr::NewTarget(..)
            | Expr::Regex { .. }
            | Expr::Lit(..) => {}
            Expr::ImportMeta(span) => self.sites.metas.push(*span),
            Expr::Array(elems, _) => {
                for el in elems.iter().flatten() {
                    self.walk_expr(el);
                }
            }
            Expr::Object(props, _) => {
                for p in props {
                    self.walk_prop(p);
                }
            }
            Expr::Function {
                name, params, body, ..
            } => {
                self.push_scope();
                if let Some(n) = name {
                    self.declare(n);
                }
                for p in params {
                    self.walk_param(p);
                }
                self.walk_stmts(body);
                self.pop_scope();
            }
            Expr::Class {
                name,
                superclass,
                body,
                ..
            } => {
                // Class-expression names are scoped to the class itself;
                // declaring in the enclosing scope is a safe
                // over-approximation.
                if let Some(n) = name {
                    self.declare(n);
                }
                if let Some(s) = superclass {
                    self.walk_expr(s);
                }
                for m in body {
                    self.walk_class_member(m);
                }
            }
            Expr::Arrow { params, body, .. } => {
                self.push_scope();
                for p in params {
                    self.walk_param(p);
                }
                match body {
                    ArrowBody::Expr(e) => self.walk_expr(e),
                    ArrowBody::Block(b) => self.walk_stmts(b),
                }
                self.pop_scope();
            }
            Expr::Call {
                callee, args, span, ..
            } => {
                self.walk_expr(callee);
                for a in args {
                    self.walk_expr(a);
                }
                if let Expr::Ident(name, _) = callee.as_ref()
                    && name == "import"
                    && args.len() == 1
                    && let Expr::Lit(Literal::Str(spec), _) = &args[0]
                {
                    // Specifiers feed URL resolution (`&str`): degrade
                    // lossily; resolution then fails as for any bad specifier.
                    self.sites
                        .imports
                        .push((*span, spec.to_utf8_lossy().into_owned()));
                }
            }
            Expr::New { callee, args, .. } => {
                self.walk_expr(callee);
                for a in args {
                    self.walk_expr(a);
                }
            }
            Expr::Member { obj, prop, .. } => {
                self.walk_expr(obj);
                self.walk_expr(prop);
            }
            Expr::Unary { arg, .. }
            | Expr::Update { arg, .. }
            | Expr::Spread(arg, _)
            | Expr::Await(arg, _) => self.walk_expr(arg),
            Expr::Binary { left, right, .. }
            | Expr::Logical { left, right, .. }
            | Expr::Assign { left, right, .. } => {
                self.walk_expr(left);
                self.walk_expr(right);
            }
            Expr::Conditional {
                cond,
                then_e,
                else_e,
                ..
            } => {
                self.walk_expr(cond);
                self.walk_expr(then_e);
                self.walk_expr(else_e);
            }
            Expr::Sequence(exprs, _) => {
                for ex in exprs {
                    self.walk_expr(ex);
                }
            }
            Expr::Template { parts, .. } => {
                for p in parts {
                    if let TemplatePart::Expr(e) = p {
                        self.walk_expr(e);
                    }
                }
            }
            Expr::TaggedTemplate { tag, parts, .. } => {
                self.walk_expr(tag);
                for p in parts {
                    if let TemplatePart::Expr(e) = p {
                        self.walk_expr(e);
                    }
                }
            }
            Expr::Yield { arg, .. } => {
                if let Some(a) = arg {
                    self.walk_expr(a);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Live bindings: reference rewriting
// ---------------------------------------------------------------------------

/// Rewrites identifier references to live namespace reads.
///
/// The loader cannot use accessor properties (the v1 VM has none), so live
/// bindings are implemented by source rewriting:
/// * a read of an imported binding `b` becomes
///   `(globalThis["\0bowser_module_registry"]["<def-url>"]["<name>"])`,
///   where `<def-url>`/`<name>` is the export's provenance (the defining
///   module, following re-export chains);
/// * reads *and* writes of a mutable exported local become
///   `(<own-namespace>["<exported>"])`, making the namespace object the
///   single source of truth for that binding.
///
/// Shadowing is handled before this pass runs: [`Walker::shadow_info`]
/// reports every declared name, and only names declared nowhere (imports)
/// or in no nested scope (exported mutables) are rewritten. Anything
/// shadowed falls back to the snapshot `const` prelude, which is always
/// safe. Writes to imports are deliberately *not* rewritten: they stay on
/// the `const` prelude binding and throw `TypeError`, per spec.
struct RefRewriter {
    /// import local -> replacement expression for read positions.
    import_reads: HashMap<String, String>,
    /// mutable exported local -> exported name.
    export_mut: HashMap<String, String>,
    /// `ns_access` for the module being rewritten.
    self_ns: String,
    /// collected `(span, replacement)` edits.
    edits: Vec<(Span, String)>,
}

impl RefRewriter {
    fn read_rep(&self, name: &str) -> Option<String> {
        if let Some(r) = self.import_reads.get(name) {
            return Some(r.clone());
        }
        self.export_mut
            .get(name)
            .map(|e| format!("({}[{}])", self.self_ns, js_str(e)))
    }

    fn write_rep(&self, name: &str) -> Option<String> {
        self.export_mut
            .get(name)
            .map(|e| format!("({}[{}])", self.self_ns, js_str(e)))
    }

    fn rw_program(&mut self, program: &Program) {
        self.rw_stmts(&program.body);
    }

    fn rw_stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.rw_stmt(s);
        }
    }

    fn rw_stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Expr(e) | Stmt::Throw(e) => self.rw_expr(e),
            Stmt::Block(b) => self.rw_stmts(b),
            Stmt::VarDecl { decls, .. } => {
                for d in decls {
                    self.rw_pat_reads(&d.id);
                    if let Some(init) = &d.init {
                        self.rw_expr(init);
                    }
                }
            }
            Stmt::FunctionDecl { params, body, .. } => {
                for p in params {
                    self.rw_pat_reads(&p.pat);
                }
                self.rw_stmts(body);
            }
            Stmt::ClassDecl {
                superclass, body, ..
            } => {
                if let Some(s) = superclass {
                    self.rw_expr(s);
                }
                for m in body {
                    self.rw_class_member(m);
                }
            }
            Stmt::Return(e) => {
                if let Some(e) = e {
                    self.rw_expr(e);
                }
            }
            Stmt::If {
                cond,
                then_b,
                else_b,
            } => {
                self.rw_expr(cond);
                self.rw_stmt(then_b);
                if let Some(e) = else_b {
                    self.rw_stmt(e);
                }
            }
            Stmt::While { cond, body } | Stmt::DoWhile { cond, body } => {
                self.rw_expr(cond);
                self.rw_stmt(body);
            }
            Stmt::For {
                init,
                cond,
                update,
                body,
            } => {
                if let Some(i) = init {
                    match i {
                        ForInit::Var(_, decls) => {
                            for d in decls {
                                self.rw_pat_reads(&d.id);
                                if let Some(init) = &d.init {
                                    self.rw_expr(init);
                                }
                            }
                        }
                        ForInit::Expr(e) => self.rw_expr(e),
                    }
                }
                if let Some(c) = cond {
                    self.rw_expr(c);
                }
                if let Some(u) = update {
                    self.rw_expr(u);
                }
                self.rw_stmt(body);
            }
            Stmt::ForInOf { lhs, rhs, body, .. } => {
                self.rw_expr(rhs);
                match lhs {
                    ForLhs::Ident(name, span) => {
                        if let Some(r) = self.write_rep(name) {
                            self.edits.push((*span, r));
                        }
                    }
                    ForLhs::Pat(p) => self.rw_target_pat(p),
                    ForLhs::Member(e) => self.rw_target(e),
                }
                self.rw_stmt(body);
            }
            Stmt::Switch { disc, cases } => {
                self.rw_expr(disc);
                for c in cases {
                    if let Some(t) = &c.test {
                        self.rw_expr(t);
                    }
                    self.rw_stmts(&c.body);
                }
            }
            Stmt::Try {
                block,
                handler,
                finalizer,
            } => {
                self.rw_stmts(block);
                if let Some(h) = handler {
                    if let Some(p) = &h.param {
                        self.rw_pat_reads(&p.pat);
                    }
                    self.rw_stmts(&h.body);
                }
                if let Some(f) = finalizer {
                    self.rw_stmts(f);
                }
            }
            Stmt::Labeled { body, .. } => self.rw_stmt(body),
            Stmt::With { obj, body, .. } => {
                // `with` disables live rewriting for the module (see
                // `Walker::has_with`); this arm is unreachable in practice.
                self.rw_expr(obj);
                self.rw_stmt(body);
            }
            Stmt::Import(..) => {}
            Stmt::Export(e) => match e {
                ExportDecl::Default { expr, .. } => self.rw_expr(expr),
                ExportDecl::Decl { inner, .. } => self.rw_stmt(inner),
                _ => {}
            },
            Stmt::Break(..) | Stmt::Continue(..) | Stmt::Debugger | Stmt::Empty => {}
        }
    }

    /// Walk a binding pattern's default values (reads). Bound names are
    /// declarations, not references.
    fn rw_pat_reads(&mut self, p: &Pat) {
        match p {
            Pat::Ident(..) => {}
            Pat::Array(elems, _) => {
                for el in elems.iter().flatten() {
                    let ArrayPatElem::Pat(p) = el;
                    self.rw_pat_reads(p);
                }
            }
            Pat::Object(props, _) => {
                for pr in props {
                    match pr {
                        ObjectPatProp::KeyValue { key, value, .. } => {
                            self.rw_prop_key(key);
                            self.rw_pat_reads(value);
                        }
                        ObjectPatProp::Shorthand { .. } => {}
                        ObjectPatProp::Rest(p, _) => self.rw_pat_reads(p),
                    }
                }
            }
            Pat::Rest(p, _) => self.rw_pat_reads(p),
            Pat::Assign(p, e, _) => {
                self.rw_pat_reads(p);
                self.rw_expr(e);
            }
            // The base object and computed key are reads.
            Pat::Member(e) => self.rw_expr(e),
        }
    }

    /// Walk a destructuring *assignment* target: bound identifiers are
    /// writes, defaults are reads.
    fn rw_target_pat(&mut self, p: &Pat) {
        match p {
            Pat::Ident(name, span) => {
                if let Some(r) = self.write_rep(name) {
                    self.edits.push((*span, r));
                }
            }
            Pat::Array(elems, _) => {
                for el in elems.iter().flatten() {
                    let ArrayPatElem::Pat(p) = el;
                    self.rw_target_pat(p);
                }
            }
            Pat::Object(props, _) => {
                for pr in props {
                    match pr {
                        ObjectPatProp::KeyValue { key, value, .. } => {
                            self.rw_prop_key(key);
                            self.rw_target_pat(value);
                        }
                        ObjectPatProp::Shorthand { name, span } => {
                            if let Some(r) = self.write_rep(name) {
                                self.edits.push((*span, format!("{name}:{r}")));
                            }
                        }
                        ObjectPatProp::Rest(p, _) => self.rw_target_pat(p),
                    }
                }
            }
            Pat::Rest(p, _) => self.rw_target_pat(p),
            Pat::Assign(p, e, _) => {
                self.rw_target_pat(p);
                self.rw_expr(e);
            }
            // Member target: the base object and computed key are reads.
            Pat::Member(e) => self.rw_target(e),
        }
    }

    /// Walk an assignment target expression: plain identifiers are writes.
    fn rw_target(&mut self, e: &Expr) {
        match e {
            Expr::Ident(name, span) => {
                if let Some(r) = self.write_rep(name) {
                    self.edits.push((*span, r));
                }
            }
            Expr::Member {
                obj,
                prop,
                computed,
                ..
            } => {
                self.rw_expr(obj);
                if *computed {
                    self.rw_expr(prop);
                }
            }
            Expr::Object(props, _) => {
                for p in props {
                    match p {
                        PropDef::Shorthand { name, span } => {
                            if let Some(r) = self.write_rep(name) {
                                self.edits.push((*span, format!("{name}:{r}")));
                            }
                        }
                        PropDef::KeyValue { key, value, .. } => {
                            self.rw_prop_key(key);
                            self.rw_target(value);
                        }
                        PropDef::Spread(e, _) => self.rw_target(e),
                        _ => self.rw_prop(p),
                    }
                }
            }
            Expr::Array(elems, _) => {
                for el in elems.iter().flatten() {
                    self.rw_target(el);
                }
            }
            _ => self.rw_expr(e),
        }
    }

    fn rw_prop_key(&mut self, k: &PropKey) {
        if let PropKey::Computed(e) = k {
            self.rw_expr(e);
        }
    }

    fn rw_class_key(&mut self, k: &ClassKey) {
        if let ClassKey::Public(pk) = k {
            self.rw_prop_key(pk);
        }
    }

    fn rw_prop(&mut self, p: &PropDef) {
        match p {
            PropDef::KeyValue { key, value, .. } => {
                self.rw_prop_key(key);
                self.rw_expr(value);
            }
            PropDef::Shorthand { name, span } => {
                // `{x}` reads x; expand so the live value is read.
                if let Some(r) = self.read_rep(name) {
                    self.edits.push((*span, format!("{name}:{r}")));
                }
            }
            PropDef::Spread(e, _) => self.rw_expr(e),
            PropDef::Method {
                key, params, body, ..
            } => {
                self.rw_prop_key(key);
                for p in params {
                    self.rw_pat_reads(&p.pat);
                }
                self.rw_stmts(body);
            }
            PropDef::Getter { key, body, .. } => {
                self.rw_prop_key(key);
                self.rw_stmts(body);
            }
            PropDef::Setter {
                key, param, body, ..
            } => {
                self.rw_prop_key(key);
                self.rw_pat_reads(&param.pat);
                self.rw_stmts(body);
            }
        }
    }

    fn rw_class_member(&mut self, m: &ClassMember) {
        match m {
            ClassMember::Method {
                key, params, body, ..
            } => {
                self.rw_class_key(key);
                for p in params {
                    self.rw_pat_reads(&p.pat);
                }
                self.rw_stmts(body);
            }
            ClassMember::Field { key, init, .. } => {
                self.rw_class_key(key);
                if let Some(e) = init {
                    self.rw_expr(e);
                }
            }
            ClassMember::StaticBlock { body, .. } => self.rw_stmts(body),
        }
    }

    fn rw_expr(&mut self, e: &Expr) {
        match e {
            Expr::Ident(name, span) => {
                if let Some(r) = self.read_rep(name) {
                    self.edits.push((*span, r));
                }
            }
            Expr::PrivateIdent(..)
            | Expr::This(..)
            | Expr::Super(..)
            | Expr::NewTarget(..)
            | Expr::Regex { .. }
            | Expr::Lit(..) => {}
            Expr::ImportMeta(..) => {}
            Expr::Array(elems, _) => {
                for el in elems.iter().flatten() {
                    self.rw_expr(el);
                }
            }
            Expr::Object(props, _) => {
                for p in props {
                    self.rw_prop(p);
                }
            }
            Expr::Function { params, body, .. } => {
                for p in params {
                    self.rw_pat_reads(&p.pat);
                }
                self.rw_stmts(body);
            }
            Expr::Class {
                superclass, body, ..
            } => {
                if let Some(s) = superclass {
                    self.rw_expr(s);
                }
                for m in body {
                    self.rw_class_member(m);
                }
            }
            Expr::Arrow { params, body, .. } => {
                for p in params {
                    self.rw_pat_reads(&p.pat);
                }
                match body {
                    ArrowBody::Expr(e) => self.rw_expr(e),
                    ArrowBody::Block(b) => self.rw_stmts(b),
                }
            }
            Expr::Call { callee, args, .. } => {
                self.rw_expr(callee);
                for a in args {
                    self.rw_expr(a);
                }
            }
            Expr::New { callee, args, .. } => {
                self.rw_expr(callee);
                for a in args {
                    self.rw_expr(a);
                }
            }
            Expr::Member {
                obj,
                prop,
                computed,
                ..
            } => {
                self.rw_expr(obj);
                if *computed {
                    self.rw_expr(prop);
                }
                // Non-computed property names are not references.
            }
            Expr::Unary { op, arg, .. } => {
                if *op == UnaryOp::Delete && matches!(arg.as_ref(), Expr::Ident(..)) {
                    // `delete x` stays a bare-identifier delete (a
                    // strict-mode error); it must not become a property
                    // delete on the namespace.
                } else {
                    self.rw_expr(arg);
                }
            }
            Expr::Update { arg, .. } => self.rw_target(arg),
            Expr::Spread(arg, _) | Expr::Await(arg, _) => self.rw_expr(arg),
            Expr::Binary { left, right, .. } | Expr::Logical { left, right, .. } => {
                self.rw_expr(left);
                self.rw_expr(right);
            }
            Expr::Assign { left, right, .. } => {
                self.rw_expr(right);
                self.rw_target(left);
            }
            Expr::Conditional {
                cond,
                then_e,
                else_e,
                ..
            } => {
                self.rw_expr(cond);
                self.rw_expr(then_e);
                self.rw_expr(else_e);
            }
            Expr::Sequence(exprs, _) => {
                for ex in exprs {
                    self.rw_expr(ex);
                }
            }
            Expr::Template { parts, .. } => {
                for p in parts {
                    if let TemplatePart::Expr(e) = p {
                        self.rw_expr(e);
                    }
                }
            }
            Expr::TaggedTemplate { tag, parts, .. } => {
                self.rw_expr(tag);
                for p in parts {
                    if let TemplatePart::Expr(e) = p {
                        self.rw_expr(e);
                    }
                }
            }
            Expr::Yield { arg, .. } => {
                if let Some(a) = arg {
                    self.rw_expr(a);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_loader(pairs: &[(&str, &str)]) -> ModuleLoader {
        let mut loader = ModuleLoader::new();
        loader.set_allow_net(false);
        for (url, src) in pairs.iter().copied() {
            loader.add_source(url, src);
        }
        loader
    }

    fn get_int(vm: &Vm, name: &str) -> Option<i32> {
        match vm.get_global(name) {
            Some(Value::Int(i)) => Some(i),
            _ => None,
        }
    }

    fn get_str(vm: &Vm, name: &str) -> Option<String> {
        match vm.get_global(name) {
            Some(Value::Str(g)) => Some(g.as_wtf8().to_utf8_lossy().into_owned()),
            _ => None,
        }
    }

    fn ns_int(vm: &Vm, loader: &ModuleLoader, url: &str, name: &str) -> Option<i32> {
        let ns = loader.namespace_of(vm, url)?;
        let Value::Object(o) = ns else {
            return None;
        };
        match get_own(vm.heap(), o, Wtf8::from_str(name)) {
            Some(Value::Int(i)) => Some(i),
            _ => None,
        }
    }

    #[test]
    fn links_imports_and_exports() {
        let mut vm = Vm::new();
        let mut loader = test_loader(&[
            (
                "https://x.test/a.js",
                "export const x = 41;\nexport function inc(v) { return v + 1; }\n",
            ),
            (
                "https://x.test/main.js",
                "import { x, inc } from \"./a.js\";\nglobalThis.result = inc(x);\n",
            ),
        ]);
        loader
            .load(&mut vm, "https://x.test/main.js")
            .expect("load");
        assert_eq!(get_int(&vm, "result"), Some(42));
        assert_eq!(ns_int(&vm, &loader, "https://x.test/a.js", "x"), Some(41));
    }

    #[test]
    fn default_export_and_namespace_import() {
        let mut vm = Vm::new();
        let mut loader = test_loader(&[
            (
                "https://x.test/b.js",
                "const answer = 7;\nexport default answer;\n",
            ),
            (
                "https://x.test/main.js",
                "import d from \"./b.js\";\nimport * as ns from \"./b.js\";\n\
                 globalThis.r1 = d;\nglobalThis.r2 = ns.default;\n",
            ),
        ]);
        loader
            .load(&mut vm, "https://x.test/main.js")
            .expect("load");
        assert_eq!(get_int(&vm, "r1"), Some(7));
        assert_eq!(get_int(&vm, "r2"), Some(7));
        assert_eq!(
            ns_int(&vm, &loader, "https://x.test/b.js", "default"),
            Some(7)
        );
    }

    #[test]
    fn reexports_and_star_exports() {
        let mut vm = Vm::new();
        let mut loader = test_loader(&[
            ("https://x.test/c.js", "export const v = 3;\n"),
            (
                "https://x.test/d.js",
                "export { v as w } from \"./c.js\";\nexport * from \"./c.js\";\n",
            ),
            (
                "https://x.test/main.js",
                "import { w, v } from \"./d.js\";\nglobalThis.rw = w;\nglobalThis.rv = v;\n",
            ),
        ]);
        loader
            .load(&mut vm, "https://x.test/main.js")
            .expect("load");
        assert_eq!(get_int(&vm, "rw"), Some(3));
        assert_eq!(get_int(&vm, "rv"), Some(3));
    }

    #[test]
    fn explicit_export_beats_star_regardless_of_order() {
        // Explicit exports win over `export *` even when the star comes
        // first in source order.
        let mut vm = Vm::new();
        let mut loader = test_loader(&[
            ("https://x.test/c.js", "export const v = 3;\n"),
            (
                "https://x.test/d.js",
                "export * from \"./c.js\";\nexport const v = 99;\n",
            ),
            (
                "https://x.test/main.js",
                "import { v } from \"./d.js\";\nglobalThis.rv = v;\n",
            ),
        ]);
        loader
            .load(&mut vm, "https://x.test/main.js")
            .expect("load");
        assert_eq!(get_int(&vm, "rv"), Some(99));
    }

    #[test]
    fn circular_imports_see_partial_namespaces() {
        // a -> b -> a. b is evaluated while a is still on the DFS stack, so
        // b observes a's not-yet-populated namespace (undefined), while a
        // later observes b's completed exports. `const` exports are snapshot
        // (not live); only `let`/`var` exports get live bindings.
        let mut vm = Vm::new();
        let mut loader = test_loader(&[
            (
                "https://x.test/a.js",
                "import { b } from \"./b.js\";\nexport const a = 1;\n\
                 globalThis.seen_b_in_a = b;\n",
            ),
            (
                "https://x.test/b.js",
                "import { a } from \"./a.js\";\nexport const b = 2;\n\
                 globalThis.seen_a_in_b = a;\n",
            ),
        ]);
        loader.load(&mut vm, "https://x.test/a.js").expect("load");
        assert_eq!(get_int(&vm, "seen_b_in_a"), Some(2));
        assert!(
            vm.get_global("seen_a_in_b")
                .is_some_and(|v| v.is_undefined()),
            "b evaluated before a's epilogue ran"
        );
        assert_eq!(ns_int(&vm, &loader, "https://x.test/a.js", "a"), Some(1));
        assert_eq!(ns_int(&vm, &loader, "https://x.test/b.js", "b"), Some(2));
    }

    #[test]
    fn dynamic_import_resolves_to_namespace() {
        let mut vm = Vm::new();
        let mut loader = test_loader(&[
            ("https://x.test/e.js", "export const val = 99;\n"),
            (
                "https://x.test/main.js",
                "globalThis.dyn = -1;\n\
                 import(\"./e.js\").then((ns) => { globalThis.dyn = ns.val; });\n",
            ),
        ]);
        loader
            .load(&mut vm, "https://x.test/main.js")
            .expect("load");
        vm.drain_microtasks();
        assert_eq!(get_int(&vm, "dyn"), Some(99));
    }

    #[test]
    fn dynamic_import_inside_async_function() {
        let mut vm = Vm::new();
        let mut loader = test_loader(&[
            ("https://x.test/e.js", "export const val = 99;\n"),
            (
                "https://x.test/main.js",
                "globalThis.adyn = -1;\n\
                 (async () => {\n\
                 \x20 const ns = await import(\"./e.js\");\n\
                 \x20 globalThis.adyn = ns.val;\n\
                 })();\n",
            ),
        ]);
        loader
            .load(&mut vm, "https://x.test/main.js")
            .expect("load");
        vm.drain_microtasks();
        assert_eq!(get_int(&vm, "adyn"), Some(99));
    }

    #[test]
    fn dynamic_import_failure_rejects() {
        let mut vm = Vm::new();
        let mut loader = test_loader(&[(
            "https://x.test/main.js",
            "globalThis.outcome = \"pending\";\n\
             import(\"./missing.js\").then(\n\
             \x20 () => { globalThis.outcome = \"fulfilled\"; },\n\
             \x20 () => { globalThis.outcome = \"rejected\"; }\n\
             );\n",
        )]);
        loader
            .load(&mut vm, "https://x.test/main.js")
            .expect("load");
        vm.drain_microtasks();
        assert_eq!(get_str(&vm, "outcome").as_deref(), Some("rejected"));
    }

    #[test]
    fn live_binding_mutation_via_exported_function() {
        // The common case: an exported `let` mutated through an exported
        // function is observed live by importers.
        let mut vm = Vm::new();
        let mut loader = test_loader(&[
            (
                "https://x.test/counter.js",
                "export let count = 0;\nexport function inc() { count++; }\n",
            ),
            (
                "https://x.test/main.js",
                "import { count, inc } from \"./counter.js\";\n\
                 inc();\n\
                 inc();\n\
                 globalThis.c = count;\n",
            ),
        ]);
        loader
            .load(&mut vm, "https://x.test/main.js")
            .expect("load");
        assert_eq!(get_int(&vm, "c"), Some(2));
    }

    #[test]
    fn live_binding_update_operators() {
        // `++`, `+=`, and plain assignment all route through the namespace.
        let mut vm = Vm::new();
        let mut loader = test_loader(&[
            (
                "https://x.test/m.js",
                "export let x = 1;\n\
                 export function bump() {\n\
                 x++;\n\
                 x += 10;\n\
                 x = x * 2;\n\
                 return x;\n\
                 }\n",
            ),
            (
                "https://x.test/main.js",
                "import { x, bump } from \"./m.js\";\n\
                 globalThis.r = bump();\n\
                 globalThis.x = x;\n",
            ),
        ]);
        loader
            .load(&mut vm, "https://x.test/main.js")
            .expect("load");
        // ((1 + 1) + 10) * 2 = 24, observed both inside and outside.
        assert_eq!(get_int(&vm, "r"), Some(24));
        assert_eq!(get_int(&vm, "x"), Some(24));
    }

    #[test]
    fn live_binding_namespace_import() {
        // `import * as ns` is live by object identity: the exporter mutates
        // its own namespace object.
        let mut vm = Vm::new();
        let mut loader = test_loader(&[
            (
                "https://x.test/counter.js",
                "export let count = 0;\nexport function inc() { count++; }\n",
            ),
            (
                "https://x.test/main.js",
                "import * as ns from \"./counter.js\";\n\
                 ns.inc();\n\
                 globalThis.c = ns.count;\n",
            ),
        ]);
        loader
            .load(&mut vm, "https://x.test/main.js")
            .expect("load");
        assert_eq!(get_int(&vm, "c"), Some(1));
    }

    #[test]
    fn live_binding_reexport_chain() {
        // Re-exports resolve through to the defining module: a mutation in
        // the origin is observed through the barrel.
        let mut vm = Vm::new();
        let mut loader = test_loader(&[
            (
                "https://x.test/origin.js",
                "export let v = 1;\nexport function setv(n) { v = n; }\n",
            ),
            (
                "https://x.test/barrel.js",
                "export { v, setv } from \"./origin.js\";\n",
            ),
            (
                "https://x.test/main.js",
                "import { v, setv } from \"./barrel.js\";\n\
                 setv(42);\n\
                 globalThis.v = v;\n",
            ),
        ]);
        loader
            .load(&mut vm, "https://x.test/main.js")
            .expect("load");
        assert_eq!(get_int(&vm, "v"), Some(42));
    }

    #[test]
    fn live_binding_shadowed_import_falls_back() {
        // A shadowed import name is not rewritten: the inner binding wins.
        let mut vm = Vm::new();
        let mut loader = test_loader(&[
            ("https://x.test/m.js", "export const x = 7;\n"),
            (
                "https://x.test/main.js",
                "import { x } from \"./m.js\";\n\
                 function f(x) { return x; }\n\
                 globalThis.inner = f(99);\n\
                 globalThis.outer = x;\n",
            ),
        ]);
        loader
            .load(&mut vm, "https://x.test/main.js")
            .expect("load");
        assert_eq!(get_int(&vm, "inner"), Some(99));
        assert_eq!(get_int(&vm, "outer"), Some(7));
    }

    #[test]
    fn live_binding_write_to_import_throws() {
        // Writes to an imported binding stay on the `const` prelude and
        // throw TypeError.
        let mut vm = Vm::new();
        let mut loader = test_loader(&[
            ("https://x.test/m.js", "export const x = 7;\n"),
            (
                "https://x.test/main.js",
                "import { x } from \"./m.js\";\n\
                 globalThis.threw = \"no\";\n\
                 try { x = 5; } catch (e) { globalThis.threw = \"yes\"; }\n",
            ),
        ]);
        loader
            .load(&mut vm, "https://x.test/main.js")
            .expect("load");
        assert_eq!(get_str(&vm, "threw"), Some("yes".to_owned()));
    }

    #[test]
    fn live_binding_export_mutable_shadowing() {
        // A nested `let` shadowing an exported mutable keeps its own value.
        let mut vm = Vm::new();
        let mut loader = test_loader(&[
            (
                "https://x.test/m.js",
                "export let x = 1;\n\
                 export function f() { let x = 2; return x; }\n\
                 export function g() { return x; }\n",
            ),
            (
                "https://x.test/main.js",
                "import { x, f, g } from \"./m.js\";\n\
                 globalThis.fx = f();\n\
                 globalThis.gx = g();\n\
                 globalThis.x = x;\n",
            ),
        ]);
        loader
            .load(&mut vm, "https://x.test/main.js")
            .expect("load");
        assert_eq!(get_int(&vm, "fx"), Some(2));
        assert_eq!(get_int(&vm, "gx"), Some(1));
        assert_eq!(get_int(&vm, "x"), Some(1));
    }

    #[test]
    fn reexport_cycles_terminate() {
        // Named re-export cycles must not recurse forever: a re-exports x
        // from b, b re-exports x from a, and nobody defines x.
        let mut vm = Vm::new();
        let mut loader = test_loader(&[
            (
                "https://x.test/a.js",
                "export { x } from \"./b.js\";\nglobalThis.a_ran = 1;\n",
            ),
            (
                "https://x.test/b.js",
                "export { x } from \"./a.js\";\nglobalThis.b_ran = 1;\n",
            ),
        ]);
        loader.load(&mut vm, "https://x.test/a.js").expect("load");
        assert_eq!(get_int(&vm, "a_ran"), Some(1));
        assert_eq!(get_int(&vm, "b_ran"), Some(1));
    }

    #[test]
    fn star_export_cycles_terminate() {
        // a: export * from b; b: export * from a, plus const y. y must be
        // visible through a's namespace.
        let mut vm = Vm::new();
        let mut loader = test_loader(&[
            ("https://x.test/a.js", "export * from \"./b.js\";\n"),
            (
                "https://x.test/b.js",
                "export * from \"./a.js\";\nexport const y = 7;\n",
            ),
            (
                "https://x.test/main.js",
                "import { y } from \"./a.js\";\nglobalThis.y = y;\n",
            ),
        ]);
        loader
            .load(&mut vm, "https://x.test/main.js")
            .expect("load");
        assert_eq!(get_int(&vm, "y"), Some(7));
    }

    #[test]
    fn dynamic_import_nested_in_export_decl() {
        // `import()` inside an `export const` initializer: the declaration
        // stays in place, so the walker must still find and rewrite it.
        let mut vm = Vm::new();
        let mut loader = test_loader(&[
            ("https://x.test/f.js", "export const fv = 5;\n"),
            (
                "https://x.test/main.js",
                "export const p = import(\"./f.js\");\n\
                 p.then((ns) => { globalThis.fv = ns.fv; });\n",
            ),
        ]);
        loader
            .load(&mut vm, "https://x.test/main.js")
            .expect("load");
        vm.drain_microtasks();
        assert_eq!(get_int(&vm, "fv"), Some(5));
    }

    #[test]
    fn import_meta_url() {
        let mut vm = Vm::new();
        let mut loader = test_loader(&[(
            "https://x.test/m.js",
            "globalThis.meta_url = import.meta.url;\n\
             globalThis.nested = (() => import.meta.url)();\n",
        )]);
        loader.load(&mut vm, "https://x.test/m.js").expect("load");
        assert_eq!(
            get_str(&vm, "meta_url").as_deref(),
            Some("https://x.test/m.js")
        );
        assert_eq!(
            get_str(&vm, "nested").as_deref(),
            Some("https://x.test/m.js")
        );
    }

    #[test]
    fn side_effect_only_import_runs_dependency() {
        let mut vm = Vm::new();
        let mut loader = test_loader(&[
            ("https://x.test/side.js", "globalThis.side_ran = 1;\n"),
            (
                "https://x.test/main.js",
                "import \"./side.js\";\nglobalThis.main_ran = 1;\n",
            ),
        ]);
        loader
            .load(&mut vm, "https://x.test/main.js")
            .expect("load");
        assert_eq!(get_int(&vm, "side_ran"), Some(1));
        assert_eq!(get_int(&vm, "main_ran"), Some(1));
    }

    #[test]
    fn missing_module_is_an_error() {
        let mut vm = Vm::new();
        let mut loader = test_loader(&[(
            "https://x.test/main.js",
            "import { x } from \"./missing.js\";\n",
        )]);
        let err = loader.load(&mut vm, "https://x.test/main.js").unwrap_err();
        assert!(matches!(err, ModuleError::Fetch(_)), "got {err:?}");
    }

    #[test]
    fn failed_fetch_poisons_module_map_without_refetch() {
        // A fetch failure must poison the module map: the entry exists as
        // `Failed` after the first attempt, and a second `import()`
        // re-rejects from that entry without re-fetching. With
        // `allow_net`, a permanently-missing module must not cost an HTTP
        // request per import.
        let mut vm = Vm::new();
        let mut loader = test_loader(&[]);
        let url = "https://x.test/gone.js";
        let err = loader.load(&mut vm, url).unwrap_err();
        assert!(matches!(err, ModuleError::Fetch(_)), "got {err:?}");
        let entry = loader
            .modules
            .get(url)
            .expect("a failed load must leave an entry");
        assert!(
            matches!(entry.state, ModuleState::Failed(_)),
            "entry must be Failed, got {:?}",
            entry.state
        );
        let fetches = loader.fetch_count.get();
        let err2 = loader.load(&mut vm, url).unwrap_err();
        assert!(
            matches!(err2, ModuleError::Eval(_)),
            "re-rejection reports the recorded error, got {err2:?}"
        );
        assert_eq!(
            loader.fetch_count.get(),
            fetches,
            "second import must re-reject from the poisoned entry without re-fetching"
        );
    }

    #[test]
    fn each_url_evaluates_once() {
        let mut vm = Vm::new();
        let mut loader = test_loader(&[
            (
                "https://x.test/once.js",
                "globalThis.eval_count = (globalThis.eval_count || 0) + 1;\n\
                 export const v = 1;\n",
            ),
            (
                "https://x.test/main.js",
                "import \"./once.js\";\nimport \"./once.js\";\nimport { v } from \"./once.js\";\n\
                 globalThis.vv = v;\n",
            ),
        ]);
        loader
            .load(&mut vm, "https://x.test/main.js")
            .expect("load");
        assert_eq!(get_int(&vm, "eval_count"), Some(1));
        assert_eq!(get_int(&vm, "vv"), Some(1));
    }

    #[test]
    fn resolve_handles_relative_and_absolute() {
        let loader = ModuleLoader::new();
        assert_eq!(
            loader
                .resolve("./b.js", "https://x.test/a/main.js")
                .unwrap(),
            "https://x.test/a/b.js"
        );
        assert_eq!(
            loader
                .resolve("https://cdn.test/lib.js", "https://x.test/a/main.js")
                .unwrap(),
            "https://cdn.test/lib.js"
        );
        assert!(loader.resolve("./b.js", "not a url").is_err());
    }

    #[test]
    fn load_source_registers_inline_module() {
        let mut vm = Vm::new();
        let mut loader = ModuleLoader::new();
        loader.set_allow_net(false);
        let ns = loader
            .load_source(&mut vm, "https://x.test/inline.js", "export const k = 5;\n")
            .expect("load");
        assert!(matches!(ns, Value::Object(_)));
        assert_eq!(
            ns_int(&vm, &loader, "https://x.test/inline.js", "k"),
            Some(5)
        );
    }

    #[test]
    fn compile_cache_reevaluates_per_loader() {
        // The compile cache memoizes parse+rewrite across loaders, but
        // each loader still evaluates: a stateful fixture re-executes
        // per load (this is the soundness property the cache must not
        // break).
        let mut vm = Vm::new();
        let pairs = [
            (
                "https://x.test/counter.js",
                "export let n = 0; export function bump() { n++; return n; }",
            ),
            (
                "https://x.test/main.js",
                "import { bump } from './counter.js'; globalThis.cacheProbe = bump();",
            ),
        ];
        let mut loader1 = test_loader(&pairs);
        loader1.load(&mut vm, "https://x.test/main.js").unwrap();
        assert_eq!(get_int(&vm, "cacheProbe"), Some(1));
        // Second load through a fresh loader: the fixture is re-evaluated
        // (fresh namespace, fresh bindings), so the counter restarts at
        // 1 — reusing the compiled form must not reuse evaluated state.
        let mut loader2 = test_loader(&pairs);
        loader2.load(&mut vm, "https://x.test/main.js").unwrap();
        assert_eq!(get_int(&vm, "cacheProbe"), Some(1));
        // And the fixture's parse+rewrite really was cached process-wide.
        let cache = COMPILE_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        assert!(cache.modules.contains_key("https://x.test/counter.js"));
        assert!(cache.modules.contains_key("https://x.test/main.js"));
    }

    #[test]
    fn compile_cache_replays_dynamic_import_deps() {
        // A module with a dynamic import: the cache hit must still
        // evaluate the dynamic target before running the cached rewrite,
        // so the spliced `Promise.resolve(<namespace>)` reads the fresh
        // namespace rather than a previous load's.
        let mut vm = Vm::new();
        let pairs = [
            ("https://x.test/dep.js", "export const value = 42;"),
            (
                "https://x.test/dynmain.js",
                "globalThis.dynProbe = 'pending'; \
                 import('./dep.js').then(ns => { globalThis.dynProbe = ns.value; });",
            ),
        ];
        for _ in 0..2 {
            let mut loader = test_loader(&pairs);
            loader.load(&mut vm, "https://x.test/dynmain.js").unwrap();
            vm.drain_microtasks();
            assert_eq!(get_int(&vm, "dynProbe"), Some(42));
        }
        let cache = COMPILE_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        let entry = &cache.modules["https://x.test/dynmain.js"];
        assert_eq!(entry.dyn_deps, vec!["https://x.test/dep.js".to_string()]);
    }

    #[test]
    fn compile_cache_miss_on_changed_source() {
        // Re-registering different source under the same URL is a cache
        // miss (source hash differs), never stale reuse.
        let mut vm = Vm::new();
        let mut loader = test_loader(&[("https://x.test/changing.js", "export const v = 1;")]);
        loader.load(&mut vm, "https://x.test/changing.js").unwrap();
        assert_eq!(
            ns_int(&vm, &loader, "https://x.test/changing.js", "v"),
            Some(1)
        );
        let mut loader2 = test_loader(&[("https://x.test/changing.js", "export const v = 2;")]);
        loader2.load(&mut vm, "https://x.test/changing.js").unwrap();
        assert_eq!(
            ns_int(&vm, &loader2, "https://x.test/changing.js", "v"),
            Some(2)
        );
    }

    #[test]
    fn compile_cache_hit_across_loader_instances() {
        // MEDIUM-3: the source hash must be deterministic (FNV-1a, not
        // `DefaultHasher`, whose stability std does not guarantee), so an
        // identical source loaded through a second loader instance is an
        // actual cache HIT — not just key presence. The hit counter is
        // process-wide, so compare against a baseline.
        let mut vm = Vm::new();
        let pairs = [(
            "https://x.test/hitprobe.js",
            "export const v = 11;\nglobalThis.hitProbe = v;\n",
        )];
        let before = compile_cache_hits();
        let mut loader1 = test_loader(&pairs);
        loader1.load(&mut vm, "https://x.test/hitprobe.js").unwrap();
        let mut loader2 = test_loader(&pairs);
        loader2.load(&mut vm, "https://x.test/hitprobe.js").unwrap();
        assert_eq!(get_int(&vm, "hitProbe"), Some(11));
        assert!(
            compile_cache_hits() > before,
            "second identical load must be a compile-cache hit"
        );
    }

    #[test]
    fn compile_cache_dyn_dep_failure_falls_back_to_slow_path() {
        // MEDIUM-2: a cached dynamic-import dep that fails at replay (its
        // source changed since the cache entry was built) must not fail
        // the whole parent load. The fast path falls back to the slow
        // path, which splices `Promise.reject(...)` for the dep and still
        // evaluates the parent.
        let mut vm = Vm::new();
        let parent_src = "globalThis.dynOutcome = 'pending';\n\
             globalThis.parentRan = false;\n\
             import('./dyndfail-dep.js').then(\n\
             \x20 () => { globalThis.dynOutcome = 'fulfilled'; },\n\
             \x20 () => { globalThis.dynOutcome = 'rejected'; });\n\
             globalThis.parentRan = true;\n";
        // First load caches the parent's compiled form (dep succeeds).
        let mut loader1 = test_loader(&[
            (
                "https://x.test/dyndfail-dep.js",
                "export const value = 42;\n",
            ),
            ("https://x.test/dyndfail.js", parent_src),
        ]);
        loader1.load(&mut vm, "https://x.test/dyndfail.js").unwrap();
        vm.drain_microtasks();
        assert_eq!(get_str(&vm, "dynOutcome").as_deref(), Some("fulfilled"));
        // Now the dep is gone (fetch fails on re-evaluation). The
        // parent's source is byte-identical, so its cache entry hits;
        // the dep replay fails and must fall back to the slow path.
        let mut loader2 = test_loader(&[("https://x.test/dyndfail.js", parent_src)]);
        loader2
            .load(&mut vm, "https://x.test/dyndfail.js")
            .expect("parent load must survive the dep failure");
        vm.drain_microtasks();
        assert!(
            matches!(vm.get_global("parentRan"), Some(Value::Bool(true))),
            "parent must still evaluate"
        );
        assert_eq!(get_str(&vm, "dynOutcome").as_deref(), Some("rejected"));
    }

    #[test]
    fn compile_cache_lookup_requires_both_source_hashes() {
        // LOW-2: a 64-bit collision between two different sources under
        // one URL must be a cache miss, never silent reuse of the wrong
        // ModuleRecord. Simulate a collision by doctoring a cache entry
        // so only the first of the two independent hashes matches.
        let url = "https://x.test/colliding.js";
        let src = "export const v = 1;";
        {
            let mut cache = COMPILE_CACHE.lock().unwrap_or_else(|e| e.into_inner());
            cache.modules.insert(
                url.to_owned(),
                CachedModule {
                    source_hash: hash_source(src),
                    source_hash2: 0, // wrong: doctored single-hash collision
                    record: ModuleRecord::default(),
                    rewritten: String::new(),
                    dyn_deps: Vec::new(),
                },
            );
        }
        let loader = test_loader(&[]);
        assert!(
            loader.lookup_compiled(url, src).is_none(),
            "a partial hash match must be a miss"
        );
        // With both hashes correct the same entry is a hit.
        {
            let mut cache = COMPILE_CACHE.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(entry) = cache.modules.get_mut(url) {
                entry.source_hash2 = hash_source2(src);
            }
        }
        assert!(
            loader.lookup_compiled(url, src).is_some(),
            "both hashes matching must be a hit"
        );
        {
            let mut cache = COMPILE_CACHE.lock().unwrap_or_else(|e| e.into_inner());
            cache.modules.remove(url);
        }
    }

    #[test]
    fn compile_cache_fallback_reuses_published_namespace() {
        // LOW-3: on the evaluate_compiled -> evaluate_slow fallback, the
        // already-published namespace must be reused, not replaced. A
        // static dep in an import cycle that captured the first namespace
        // via `import * as ns` must observe the populated exports; a
        // second publish would orphan the first namespace (empty forever).
        let mut vm = Vm::new();
        let parent_src = "import { d } from \"./cycdep.js\";\n\
             export const mainVal = 7;\n\
             globalThis.parentRan = d;\n\
             import('./dyndfail-dep.js').then(\n\
             \x20 () => { globalThis.dynOutcome = 'fulfilled'; },\n\
             \x20 () => { globalThis.dynOutcome = 'rejected'; });\n";
        let dep_src = "import * as nsMain from \"./cycmain.js\";\n\
             export const d = 3;\n\
             globalThis.capturedNs = nsMain;\n";
        // First load caches the parent's compiled form (dyn dep
        // succeeds); the cycle captures the published namespace.
        let mut loader1 = test_loader(&[
            (
                "https://x.test/dyndfail-dep.js",
                "export const value = 42;\n",
            ),
            ("https://x.test/cycdep.js", dep_src),
            ("https://x.test/cycmain.js", parent_src),
        ]);
        loader1.load(&mut vm, "https://x.test/cycmain.js").unwrap();
        vm.drain_microtasks();
        vm.eval("globalThis.probe = globalThis.capturedNs.mainVal;")
            .unwrap();
        assert_eq!(get_int(&vm, "probe"), Some(7));
        // Second load: the parent cache entry hits, the dyn-dep replay
        // fails, and the slow-path fallback re-runs the body. The
        // namespace the cycle captured must be the one the body
        // populates — not an orphaned empty one.
        let mut loader2 = test_loader(&[
            ("https://x.test/cycdep.js", dep_src),
            ("https://x.test/cycmain.js", parent_src),
        ]);
        loader2
            .load(&mut vm, "https://x.test/cycmain.js")
            .expect("parent load must survive the dep failure");
        vm.drain_microtasks();
        assert!(
            matches!(vm.get_global("parentRan"), Some(Value::Int(3))),
            "parent must still evaluate"
        );
        assert_eq!(get_str(&vm, "dynOutcome").as_deref(), Some("rejected"));
        vm.eval("globalThis.probe = globalThis.capturedNs.mainVal;")
            .unwrap();
        assert_eq!(
            get_int(&vm, "probe"),
            Some(7),
            "the namespace captured by the cycle must be populated after the fallback"
        );
    }
}
