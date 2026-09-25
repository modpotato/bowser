//! WPT manifest reading.
//!
//! Two ways to learn which tests exist in a web-platform-tests checkout:
//!
//! 1. [`Manifest::from_json`] parses a real `MANIFEST.json` (produced by
//!    `wpt manifest`). This is the accurate path: the manifest already knows
//!    each test's type.
//! 2. [`Manifest::walk`] scans the checkout directory directly and classifies
//!    `.html`/`.htm`/`.xhtml`/`.xht` files with cheap content heuristics.
//!    Useful when no manifest has been generated yet. Non-test directories
//!    and reftest reference files are skipped.
//!
//! `MANIFEST.json` shape (v8/v9):
//! ```json
//! {"version": 9, "url_base": "/", "items": {
//!   "testharness": {"dom/x.html": ["item", ["/dom/x.html", {}]]},
//!   "reftest":     {"css/y.html": ["item", ["/css/y.html", {}]]}
//! }}
//! ```

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use crate::json::{self, Json};

/// The kind of a WPT test.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TestKind {
    /// `testharness.js`-based test: JS assertions reported programmatically.
    Testharness,
    /// Reference test: rendered output compared against a reference rendering.
    Reftest,
    /// Print reftest (paged media).
    PrintReftest,
    /// Manual test: needs a human.
    Manual,
    /// Crashtest: passes if the browser does not crash.
    Crashtest,
    /// Anything else (`wdspec`, future types, unrecognized names).
    Other(String),
}

impl TestKind {
    /// Map a `MANIFEST.json` item-type name to a [`TestKind`].
    pub fn from_manifest_type(name: &str) -> TestKind {
        match name {
            "testharness" => TestKind::Testharness,
            "reftest" => TestKind::Reftest,
            "print-reftest" => TestKind::PrintReftest,
            "manual" => TestKind::Manual,
            "crashtest" => TestKind::Crashtest,
            other => TestKind::Other(other.to_owned()),
        }
    }

    /// Whether this kind can run headless without human interaction.
    pub fn is_automated(&self) -> bool {
        !matches!(self, TestKind::Manual | TestKind::Other(_))
    }
}

impl fmt::Display for TestKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TestKind::Testharness => write!(f, "testharness"),
            TestKind::Reftest => write!(f, "reftest"),
            TestKind::PrintReftest => write!(f, "print-reftest"),
            TestKind::Manual => write!(f, "manual"),
            TestKind::Crashtest => write!(f, "crashtest"),
            TestKind::Other(name) => write!(f, "other({name})"),
        }
    }
}

/// One test in a WPT checkout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestItem {
    /// Test id, as it appears in `MANIFEST.json` (path relative to the
    /// checkout root, forward slashes).
    pub id: String,
    /// URL path the test is served at (url_base + id, roughly).
    pub url: String,
    pub kind: TestKind,
    /// Absolute filesystem path, present when the manifest came from [`Manifest::walk`].
    pub path: Option<PathBuf>,
}

/// A set of tests enumerated from a WPT checkout.
#[derive(Debug, Clone, Default)]
pub struct Manifest {
    items: Vec<TestItem>,
    /// Entries in `MANIFEST.json` whose shape we did not understand.
    pub skipped: usize,
}

impl Manifest {
    /// Parse a `MANIFEST.json` document.
    pub fn from_json(text: &str) -> Result<Manifest, ManifestError> {
        let doc = json::parse(text).map_err(|e| ManifestError::Invalid(e.to_string()))?;
        let url_base = doc.get("url_base").and_then(Json::as_str).unwrap_or("/");
        let items_obj = doc.get("items").and_then(Json::as_obj).ok_or_else(|| {
            ManifestError::Invalid("MANIFEST.json has no \"items\" object".into())
        })?;

        let mut manifest = Manifest::default();
        for (type_name, entries) in items_obj {
            let kind = TestKind::from_manifest_type(type_name);
            let entries = entries.as_obj().ok_or_else(|| {
                ManifestError::Invalid(format!("items.{type_name} is not an object"))
            })?;
            for (id, item) in entries {
                match manifest_url(item) {
                    Some(url) => manifest.items.push(TestItem {
                        id: id.clone(),
                        url: join_url(url_base, &url),
                        kind: kind.clone(),
                        path: None,
                    }),
                    None => manifest.skipped += 1,
                }
            }
        }
        // Deterministic order regardless of JSON key order.
        manifest.items.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(manifest)
    }

    /// Read and parse `<root>/MANIFEST.json`.
    pub fn from_manifest_file(root: &Path) -> Result<Manifest, ManifestError> {
        let text = std::fs::read_to_string(root.join("MANIFEST.json"))?;
        Manifest::from_json(&text)
    }

    /// Walk a WPT checkout and classify test files with content heuristics.
    ///
    /// Only `.html`/`.htm`/`.xhtml`/`.xht` files are considered. Support and
    /// infrastructure directories (`tools/`, `resources/`, `support/`,
    /// `common/`, `conformance-checkers/`, `reference/`, `infrastructure/`,
    /// `.git/`) are skipped at any depth, as are reftest reference files
    /// (`*-ref.html` and friends, mirroring wpt's `reference_file_re`).
    /// Classification:
    ///
    /// - contains a `<link rel=match|mismatch>` → [`TestKind::Reftest`]
    /// - references `testharness.js` → [`TestKind::Testharness`]
    /// - otherwise → [`TestKind::Other`]("unknown-html")
    pub fn walk(root: &Path) -> Result<Manifest, ManifestError> {
        let mut manifest = Manifest::default();
        walk_dir(root, root, &mut manifest)?;
        manifest.items.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(manifest)
    }

    pub fn items(&self) -> &[TestItem] {
        &self.items
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// All tests of one kind.
    pub fn by_kind(&self, kind: &TestKind) -> Vec<&TestItem> {
        self.items.iter().filter(|t| &t.kind == kind).collect()
    }

    /// Count tests per kind, sorted by kind name.
    pub fn kind_counts(&self) -> Vec<(String, usize)> {
        let mut counts: Vec<(String, usize)> = Vec::new();
        for item in &self.items {
            let name = item.kind.to_string();
            match counts.iter_mut().find(|(n, _)| *n == name) {
                Some((_, c)) => *c += 1,
                None => counts.push((name, 1)),
            }
        }
        counts.sort_by(|a, b| a.0.cmp(&b.0));
        counts
    }
}

/// Extract the served URL from a manifest item entry: `["item", [url, ...]]`.
fn manifest_url(item: &Json) -> Option<String> {
    let pair = item.as_arr()?;
    if pair.first()?.as_str() != Some("item") {
        return None;
    }
    let inner = pair.get(1)?.as_arr()?;
    inner.first()?.as_str().map(str::to_owned)
}

fn join_url(base: &str, url: &str) -> String {
    let base = base.strip_suffix('/').unwrap_or(base);
    if url.starts_with('/') {
        url.to_owned()
    } else {
        format!("{base}/{url}")
    }
}

/// Directories in a WPT checkout that never contain runnable tests.
///
/// Mirrors upstream wpt `tools/manifest/sourcefile.py`: `dir_non_test =
/// {"resources", "support", "tools"}`, `root_dir_non_test = {"common"}`,
/// `in_conformance_checker_dir()` (`conformance-checkers/`), and the
/// `/reference/` check in `name_is_reference`. Matched at any depth
/// (upstream only requires some of them at any depth; walk mode is
/// stricter by design since it only enumerates tests).
const SKIP_DIRS: &[&str] = &[
    ".git",
    ".github",
    "infrastructure",
    "tools",
    "resources",
    "support",
    "common",
    "conformance-checkers",
    "reference",
];

/// Filename extensions we bother classifying. Upstream wpt
/// `tools/manifest/sourcefile.py::markup_type` recognizes `html`/`htm` as
/// HTML and `xhtml`/`xht`/`xml` as XHTML; walk mode classifies the HTML
/// family (`.xml` files in a checkout are overwhelmingly resources, and
/// the content classifier only understands HTML markers).
const TEST_EXTENSIONS: &[&str] = &["html", "htm", "xhtml", "xht"];

fn walk_dir(root: &Path, dir: &Path, manifest: &mut Manifest) -> Result<(), ManifestError> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    entries.sort();
    for path in entries {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_owned();
        // Never follow symlinks: WPT has none that matter, and loops are fatal.
        let meta = std::fs::symlink_metadata(&path)?;
        if meta.is_dir() {
            if !SKIP_DIRS.contains(&name.as_str()) {
                walk_dir(root, &path, manifest)?;
            }
        } else if meta.is_file() && has_test_extension(&name) {
            // Reftest reference files are not tests (upstream `name_is_reference`).
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default();
            if is_reference_name(stem) {
                continue;
            }
            let id = path
                .strip_prefix(root)
                .map_err(|_| ManifestError::Invalid("path escapes root".into()))?
                .to_string_lossy()
                .replace('\\', "/");
            let kind = classify_file(&path)?;
            manifest.items.push(TestItem {
                id: id.clone(),
                url: format!("/{id}"),
                kind,
                path: Some(path),
            });
        }
    }
    Ok(())
}

fn has_test_extension(name: &str) -> bool {
    name.rsplit('.').next().is_some_and(|ext| {
        let ext = ext.to_ascii_lowercase();
        TEST_EXTENSIONS.iter().any(|want| *want == ext)
    })
}

/// Whether a file stem names a reftest *reference* file rather than a test.
///
/// Mirrors upstream wpt `tools/manifest/sourcefile.py`:
/// `reference_file_re = re.compile(r'(^|[\-_])(not)?ref[0-9]*([\-_]|$)')`,
/// searched (not full-matched) against the file stem. Case-sensitive, like
/// upstream.
fn is_reference_name(stem: &str) -> bool {
    let bytes = stem.as_bytes();
    // Equivalent to `re.search`: try the anchored match at every position,
    // with both alternatives of `(^|[\-_])` where they apply.
    (0..=bytes.len()).any(|start| {
        match_reference_at(bytes, start, true) || match_reference_at(bytes, start, false)
    })
}

/// Try the upstream `reference_file_re` anchored at `start`. `use_caret`
/// selects the `^` alternative of `(^|[\-_])`, otherwise the `[\-_]` one.
fn match_reference_at(bytes: &[u8], start: usize, use_caret: bool) -> bool {
    let mut i = start;
    if use_caret {
        if start != 0 {
            return false;
        }
        // `^` matches without consuming input
    } else if bytes.get(start).is_some_and(|c| *c == b'-' || *c == b'_') {
        i += 1;
    } else {
        return false;
    }
    // `(not)?ref`
    let tail = &bytes[i..];
    let tail = tail.strip_prefix(b"not").unwrap_or(tail);
    let Some(tail) = tail.strip_prefix(b"ref") else {
        return false;
    };
    // `[0-9]*([\-_]|$)`
    let mut tail = tail;
    while tail.first().is_some_and(|c| c.is_ascii_digit()) {
        tail = &tail[1..];
    }
    tail.is_empty() || tail[0] == b'-' || tail[0] == b'_'
}

/// Classify one HTML file by looking for reftest/testharness markers.
/// Only the first 128 KiB are read; markers live in `<head>`.
fn classify_file(path: &Path) -> Result<TestKind, ManifestError> {
    use std::io::Read as _;
    let mut f = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    f.by_ref().take(128 * 1024).read_to_end(&mut buf)?;
    Ok(classify_html(&String::from_utf8_lossy(&buf)))
}

fn classify_html(content: &str) -> TestKind {
    let lower = content.to_ascii_lowercase();
    // Check mismatch before match: "rel=mismatch" contains "rel=match".
    if has_rel(&lower, "mismatch") || has_rel(&lower, "match") {
        TestKind::Reftest
    } else if lower.contains("testharness.js") {
        TestKind::Testharness
    } else {
        TestKind::Other("unknown-html".into())
    }
}

/// True if the document contains a `rel=<value>` attribute (any quoting).
fn has_rel(lower: &str, value: &str) -> bool {
    for pat in [
        format!("rel={value}"),
        format!("rel=\"{value}\""),
        format!("rel='{value}'"),
    ] {
        if lower.contains(&pat) {
            return true;
        }
    }
    false
}

/// Manifest reading can fail on I/O or on a malformed `MANIFEST.json`.
#[derive(Debug)]
pub enum ManifestError {
    Io(io::Error),
    Invalid(String),
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ManifestError::Io(e) => write!(f, "manifest I/O error: {e}"),
            ManifestError::Invalid(msg) => write!(f, "invalid manifest: {msg}"),
        }
    }
}

impl std::error::Error for ManifestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ManifestError::Io(e) => Some(e),
            ManifestError::Invalid(_) => None,
        }
    }
}

impl From<io::Error> for ManifestError {
    fn from(e: io::Error) -> Self {
        ManifestError::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TREE_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// A synthetic WPT-shaped checkout on disk. Removed on drop.
    struct FixtureTree {
        root: PathBuf,
    }

    impl FixtureTree {
        fn new(files: &[(&str, &str)]) -> FixtureTree {
            let n = TREE_COUNTER.fetch_add(1, Ordering::SeqCst);
            let root =
                std::env::temp_dir().join(format!("bowser_wpt_walk_{}_{n}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            for (rel, content) in files {
                let path = root.join(rel);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(&path, content).unwrap();
            }
            FixtureTree { root }
        }
    }

    impl Drop for FixtureTree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    const TESTHARNESS_HTML: &str = r#"<!doctype html>
<script src="/resources/testharness.js"></script>
<script src="/resources/testharnessreport.js"></script>
<script>test(function() { assert_true(true); }, "smoke");</script>"#;

    const REFTHTEST_HTML: &str = r#"<!doctype html>
<link rel=match href="ref.html">
<style>div { color: green; }</style>
<div>hello</div>"#;

    const REFTHTEST_MISMATCH_HTML: &str = r#"<!doctype html>
<link rel="mismatch" href="not-ref.html">
<div>hello</div>"#;

    const PLAIN_HTML: &str = "<!doctype html><title>not a test</title>";

    #[test]
    fn from_json_reads_types_and_urls() {
        let json = r#"{
            "version": 9,
            "url_base": "/",
            "items": {
                "testharness": {
                    "dom/a.html": ["item", ["/dom/a.html", {}]],
                    "dom/b.html": ["item", ["/dom/b.html", {"timeout": "long"}]]
                },
                "reftest": {
                    "css/c.html": ["item", ["/css/c.html", {}]]
                },
                "manual": {
                    "html/d.html": ["item", ["/html/d.html", {}]]
                },
                "wdspec": {
                    "webdriver/e.py": ["item", ["/webdriver/e.py", {}]]
                },
                "testharness-broken": []
            }
        }"#;
        // "testharness-broken" is not an object of entries; parsing must fail.
        assert!(Manifest::from_json(json).is_err());

        let json = json.replace(
            r#""testharness-broken": []"#,
            r#""crashtest": {
                    "x/y.html": ["bogus-shape"]
                }"#,
        );
        let m = Manifest::from_json(&json).unwrap();
        assert_eq!(m.len(), 5);
        assert_eq!(m.skipped, 1); // the bogus-shape entry
        assert_eq!(m.by_kind(&TestKind::Testharness).len(), 2);
        assert_eq!(m.by_kind(&TestKind::Reftest).len(), 1);
        assert_eq!(m.by_kind(&TestKind::Manual).len(), 1);
        assert_eq!(m.by_kind(&TestKind::Other("wdspec".into())).len(), 1);
        let first = &m.items()[0];
        assert_eq!(first.id, "css/c.html");
        assert_eq!(first.url, "/css/c.html");
        assert_eq!(first.kind, TestKind::Reftest);
    }

    #[test]
    fn from_json_rejects_garbage() {
        assert!(Manifest::from_json("not json").is_err());
        assert!(Manifest::from_json(r#"{"version": 9}"#).is_err());
        assert!(Manifest::from_json(r#"{"items": []}"#).is_err());
    }

    #[test]
    fn from_json_applies_url_base() {
        let json = r#"{"version": 9, "url_base": "/_wpt_/", "items": {
            "testharness": {"a.html": ["item", ["a.html", {}]]}
        }}"#;
        let m = Manifest::from_json(json).unwrap();
        assert_eq!(m.items()[0].url, "/_wpt_/a.html");
    }

    #[test]
    fn walk_classifies_a_checkout() {
        let tree = FixtureTree::new(&[
            ("dom/a.html", TESTHARNESS_HTML),
            ("css/b.html", REFTHTEST_HTML),
            ("css/c.html", REFTHTEST_MISMATCH_HTML),
            ("html/plain.html", PLAIN_HTML),
            ("html/notes.txt", "not html"),
            ("tools/gen.py", "print('infra')"),
            ("resources/testharness.js", "// infra"),
            ("nested/deep/d.html", TESTHARNESS_HTML),
            ("MANIFEST.json", "{}"),
        ]);
        let m = Manifest::walk(&tree.root).unwrap();
        // 5 html files: .txt/.js/.json skipped, tools/ and resources/ skipped.
        assert_eq!(m.len(), 5);
        assert_eq!(m.by_kind(&TestKind::Testharness).len(), 2);
        assert_eq!(m.by_kind(&TestKind::Reftest).len(), 2);
        assert_eq!(m.by_kind(&TestKind::Other("unknown-html".into())).len(), 1);
        let ids: Vec<&str> = m.items().iter().map(|t| t.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "css/b.html",
                "css/c.html",
                "dom/a.html",
                "html/plain.html",
                "nested/deep/d.html",
            ]
        );
        assert!(m.items().iter().all(|t| t.path.is_some()));
    }

    #[test]
    fn walk_classifies_xht_and_skips_reference_and_support() {
        let tree = FixtureTree::new(&[
            ("css/a.xht", TESTHARNESS_HTML),          // .xht is a test extension
            ("css/a-ref.xht", REFTHTEST_HTML),        // reference name: excluded
            ("css/b-ref.html", REFTHTEST_HTML),       // *-ref.html: excluded
            ("css/b-notref.html", REFTHTEST_HTML),    // *-notref.html: excluded
            ("css/reference/c.html", REFTHTEST_HTML), // under reference/: excluded
            ("css/d.html", TESTHARNESS_HTML),
            ("css/support/e.html", TESTHARNESS_HTML), // support/ at depth: excluded
            ("a/b/resources/f.html", TESTHARNESS_HTML), // resources/ at depth: excluded
            ("conformance-checkers/g.html", TESTHARNESS_HTML), // excluded
            ("css/common/h.html", TESTHARNESS_HTML),  // common/ at depth: excluded
        ]);
        let m = Manifest::walk(&tree.root).unwrap();
        let ids: Vec<&str> = m.items().iter().map(|t| t.id.as_str()).collect();
        assert_eq!(ids, ["css/a.xht", "css/d.html"]);
        assert_eq!(m.by_kind(&TestKind::Testharness).len(), 2);
    }

    #[test]
    fn reference_name_regex_matches_upstream() {
        // Mirrors wpt's reference_file_re: (^|[\-_])(not)?ref[0-9]*([\-_]|$)
        for stem in [
            "a-ref", "a-notref", "a-ref-2", "ref", "ref2", "-ref", "_notref_", "a_ref_1",
        ] {
            assert!(is_reference_name(stem), "{stem}");
        }
        for stem in [
            "reference",
            "myref",
            "refer",
            "a-refx",
            "prefer",
            "refactor",
            "a",
        ] {
            assert!(!is_reference_name(stem), "{stem}");
        }
    }

    #[test]
    fn walk_missing_root_is_io_error() {
        let err = Manifest::walk(Path::new("/nonexistent-bowser-wpt-root")).unwrap_err();
        assert!(matches!(err, ManifestError::Io(_)));
    }

    #[test]
    fn kind_display_and_automation() {
        assert_eq!(TestKind::Testharness.to_string(), "testharness");
        assert_eq!(
            TestKind::Other("wdspec".into()).to_string(),
            "other(wdspec)"
        );
        assert!(TestKind::Reftest.is_automated());
        assert!(!TestKind::Manual.is_automated());
    }
}
