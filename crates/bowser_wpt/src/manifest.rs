//! WPT manifest reading.
//!
//! Two ways to learn which tests exist in a web-platform-tests checkout:
//!
//! 1. [`Manifest::from_json`] parses a real `MANIFEST.json` (produced by
//!    `wpt manifest`). This is the accurate path: the manifest already knows
//!    each test's type.
//! 2. [`Manifest::walk`] scans the checkout directory directly and classifies
//!    `.html`/`.htm` files with cheap content heuristics. Useful when no
//!    manifest has been generated yet.
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
    /// Only `.html`/`.htm`/`.xhtml` files are considered. Infrastructure
    /// directories (`tools/`, `resources/`, `infrastructure/`, `.git/`) are
    /// skipped. Classification (see [`classify_file`] for the scan-window
    /// policy):
    ///
    /// - a `<link>` tag with `rel` exactly `match`/`mismatch` (any quoting,
    ///   optional whitespace around `=`) → [`TestKind::Reftest`]
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
const SKIP_DIRS: &[&str] = &[".git", "tools", "resources", "infrastructure", ".github"];

/// HTML-ish extensions we bother classifying.
const TEST_EXTENSIONS: &[&str] = &["html", "htm", "xhtml"];

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

/// Classify one HTML file by looking for reftest/testharness markers.
///
/// Only the first [`CLASSIFY_WINDOW`] bytes are read. This is a deliberate
/// heuristic, not a correctness guarantee: markers for real WPT tests live
/// in `<head>`, within the first few KiB, but reading whole files is
/// unbounded (some WPT fixtures are tens of MB) and `Manifest::walk` must
/// stay cheap per file. A file whose markers sit past the window classifies
/// as `Other("unknown-html")`.
const CLASSIFY_WINDOW: usize = 1024 * 1024;

fn classify_file(path: &Path) -> Result<TestKind, ManifestError> {
    use std::io::Read as _;
    let mut f = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    f.by_ref()
        .take(CLASSIFY_WINDOW as u64)
        .read_to_end(&mut buf)?;
    Ok(classify_html(&String::from_utf8_lossy(&buf)))
}

fn classify_html(content: &str) -> TestKind {
    let lower = content.to_ascii_lowercase();
    // Reftest links are checked before testharness.js: a reftest page may
    // mention testharness helpers in prose without being a testharness test.
    if has_reftest_link(&lower) {
        TestKind::Reftest
    } else if lower.contains("testharness.js") {
        TestKind::Testharness
    } else {
        TestKind::Other("unknown-html".into())
    }
}

/// True if the document contains a `<link>` tag whose `rel` attribute is
/// exactly `match` or `mismatch`.
///
/// This is a small attribute tokenizer, not a substring search:
/// - `rel=match` inside a JS comment or prose is ignored (W10a); only a
///   real `<link ...>` tag counts,
/// - optional whitespace around `=` is accepted: `<link rel = "match">`
///   counts (W10b),
/// - the value must be exactly `match`/`mismatch`: `rel=matching` does not
///   count (W10c), and `data-rel` is a different attribute.
fn has_reftest_link(lower: &str) -> bool {
    let bytes = lower.as_bytes();
    let mut pos = 0;
    while let Some(found) = lower[pos..].find("<link") {
        let tag_start = pos + found;
        // `<link` must open a tag, not a longer tag name like `<linkfoo`.
        let boundary_ok = match bytes.get(tag_start + "<link".len()) {
            None => true, // truncated input; let the tokenizer decide
            Some(&b) => b.is_ascii_whitespace() || b == b'/' || b == b'>',
        };
        pos = tag_start + "<link".len();
        if boundary_ok && link_rel_is_reftest(bytes, pos) {
            return true;
        }
    }
    false
}

/// Scan one `<link ...>` tag's attributes, starting just after `<link`.
/// True if the tag's `rel` attribute value is exactly `match` or `mismatch`.
/// Stops at the tag's closing `>` (quote-aware) or at end of input.
fn link_rel_is_reftest(bytes: &[u8], mut i: usize) -> bool {
    loop {
        // Skip whitespace and stray self-closing slashes between attributes.
        while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b'/') {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] == b'>' {
            return false;
        }
        // Attribute name: up to `=`, whitespace, `/` or `>`.
        let name_start = i;
        while i < bytes.len()
            && !bytes[i].is_ascii_whitespace()
            && !matches!(bytes[i], b'=' | b'/' | b'>')
        {
            i += 1;
        }
        let name = &bytes[name_start..i];
        // Optional whitespace, then `=` and the value.
        let mut j = i;
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        let mut value: Option<&[u8]> = None;
        if j < bytes.len() && bytes[j] == b'=' {
            j += 1;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < bytes.len() && (bytes[j] == b'"' || bytes[j] == b'\'') {
                let quote = bytes[j];
                j += 1;
                let vstart = j;
                while j < bytes.len() && bytes[j] != quote {
                    j += 1;
                }
                value = Some(&bytes[vstart..j]);
                j += 1; // consume the closing quote, if present
            } else {
                // Unquoted value: runs to whitespace, `/` or `>`.
                let vstart = j;
                while j < bytes.len()
                    && !bytes[j].is_ascii_whitespace()
                    && !matches!(bytes[j], b'/' | b'>')
                {
                    j += 1;
                }
                value = Some(&bytes[vstart..j]);
            }
        }
        i = j;
        if name == b"rel" {
            // First `rel` attribute wins, per HTML parsing rules.
            return matches!(value, Some(v) if v == b"match" || v == b"mismatch");
        }
    }
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

    // W10a: `rel=match` in a JS comment or prose is not a reftest marker.
    #[test]
    fn classify_rel_match_in_comment_or_prose_is_not_reftest() {
        let js_comment = r#"<!doctype html>
<script src="/resources/testharness.js"></script>
<script>
// reftests use rel=match; this test does not
test(function() {}, "x");</script>"#;
        assert_eq!(classify_html(js_comment), TestKind::Testharness);

        let prose = "<!doctype html><p>to compare, add rel=match to a link tag</p>";
        assert_eq!(classify_html(prose), TestKind::Other("unknown-html".into()));

        // Only a real <link> tag counts; escaped markup and other elements
        // carrying the substring must not.
        let escaped = "<!doctype html><pre>&lt;link rel=match&gt;</pre>";
        assert_eq!(
            classify_html(escaped),
            TestKind::Other("unknown-html".into())
        );
        let div_rel = r#"<!doctype html><div data-rel="match">x</div>"#;
        assert_eq!(
            classify_html(div_rel),
            TestKind::Other("unknown-html".into())
        );
    }

    // W10b: optional whitespace around `=` is accepted.
    #[test]
    fn classify_rel_with_whitespace_around_equals() {
        for html in [
            r#"<!doctype html><link rel = "match" href="r.html">"#,
            "<!doctype html><link rel\t=\n'mismatch' href=r.html>",
            "<!doctype html><LINK REL =MATCH HREF=r.html>",
            "<!doctype html><link  rel=match  />",
        ] {
            assert_eq!(classify_html(html), TestKind::Reftest, "{html}");
        }
    }

    // W10c: the rel value must be exactly `match`/`mismatch`.
    #[test]
    fn classify_rel_value_prefix_is_not_reftest() {
        // `rel=matching` must not shadow the testharness marker.
        let prefix_unquoted = r#"<!doctype html>
<link rel=matching href="r.html">
<script src="/resources/testharness.js"></script>"#;
        assert_eq!(classify_html(prefix_unquoted), TestKind::Testharness);

        let prefix_quoted = r#"<!doctype html><link rel="matching" href="r.html"><p>x</p>"#;
        assert_eq!(
            classify_html(prefix_quoted),
            TestKind::Other("unknown-html".into())
        );

        // A longer tag name is not <link>; data-rel is not rel.
        for html in [
            r#"<!doctype html><linkfoo rel=match href="r.html">"#,
            r#"<!doctype html><link data-rel="match" href="r.html">"#,
            r#"<!doctype html><link rel href="r.html">"#,
        ] {
            assert_eq!(
                classify_html(html),
                TestKind::Other("unknown-html".into()),
                "{html}"
            );
        }
    }

    // W10d: markers past the old 128 KiB window but inside the 1 MiB window
    // classify correctly.
    #[test]
    fn walk_finds_marker_past_old_128kib_window() {
        let padding = " ".repeat(200 * 1024);
        let html = format!("<!doctype html><title>t</title>{padding}<link rel=match href=r.html>");
        let tree = FixtureTree::new(&[("css/late.html", &html)]);
        let m = Manifest::walk(&tree.root).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m.items()[0].kind, TestKind::Reftest);
    }

    // W10d: the 1 MiB window is a heuristic; a marker beyond it is missed.
    #[test]
    fn walk_marker_past_1mib_window_is_unknown() {
        let padding = " ".repeat(1024 * 1024 + 16);
        let html = format!("<!doctype html><title>t</title>{padding}<link rel=match href=r.html>");
        let tree = FixtureTree::new(&[("css/too-late.html", &html)]);
        let m = Manifest::walk(&tree.root).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m.items()[0].kind, TestKind::Other("unknown-html".into()));
    }
}
