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
    /// Directory entries dropped during [`Manifest::walk`] because reading
    /// the entry itself failed. Non-fatal by design: a single unreadable
    /// entry (transient filesystem error, vanishing file) must not abort a
    /// scan of an entire checkout. A caller that wants a strict scan should
    /// treat a non-zero value as failure.
    pub walk_errors: usize,
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
    /// skipped. Classification:
    ///
    /// - contains a `<link rel=match|mismatch>` → [`TestKind::Reftest`]
    /// - references `testharness.js` → [`TestKind::Testharness`]
    /// - otherwise → [`TestKind::Other`]("unknown-html")
    ///
    /// Robustness notes:
    /// - A symlinked `root` is refused with [`ManifestError::Invalid`]:
    ///   `read_dir` follows the root symlink, so it could point anywhere
    ///   (even `/`) and we would traverse it. Symlinks *inside* the tree
    ///   are never followed (see `walk_dir`).
    /// - Per-entry I/O errors while listing a directory are non-fatal and
    ///   counted in [`Manifest::walk_errors`]; any other I/O error aborts
    ///   the walk.
    pub fn walk(root: &Path) -> Result<Manifest, ManifestError> {
        // Guard the root itself: read_dir follows symlinks, and inner
        // symlink protection (symlink_metadata in walk_dir) does not apply
        // to the starting directory.
        if std::fs::symlink_metadata(root)?.file_type().is_symlink() {
            return Err(ManifestError::Invalid(format!(
                "refusing to walk symlinked root: {}",
                root.display()
            )));
        }
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
    let mut entries = collect_entries(std::fs::read_dir(dir)?, manifest);
    entries.sort();
    for path in entries {
        // Lossy on purpose, matching the id construction below: a non-UTF8
        // file name must still get its extension checked instead of being
        // silently dropped via `to_str().unwrap_or_default() == ""`.
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
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

/// Collect one directory's entries. Per-entry I/O errors are counted on
/// [`Manifest::walk_errors`] and skipped rather than silently dropped or
/// aborting the whole walk; see the [`Manifest::walk`] docs for the policy.
fn collect_entries(
    iter: impl Iterator<Item = io::Result<std::fs::DirEntry>>,
    manifest: &mut Manifest,
) -> Vec<PathBuf> {
    let mut entries = Vec::new();
    for entry in iter {
        match entry {
            Ok(e) => entries.push(e.path()),
            Err(_) => manifest.walk_errors += 1,
        }
    }
    entries
}

fn has_test_extension(name: &str) -> bool {
    name.rsplit('.').next().is_some_and(|ext| {
        let ext = ext.to_ascii_lowercase();
        TEST_EXTENSIONS.iter().any(|want| *want == ext)
    })
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
        assert_eq!(m.walk_errors, 0);
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
    fn collect_entries_counts_per_entry_io_errors() {
        // Deterministic coverage of the error-counting policy without
        // needing a real failing filesystem entry.
        let boom: io::Result<std::fs::DirEntry> = Err(io::Error::other("boom"));
        let mut manifest = Manifest::default();
        let entries = collect_entries(vec![boom].into_iter(), &mut manifest);
        assert!(entries.is_empty());
        assert_eq!(manifest.walk_errors, 1);
    }

    #[cfg(unix)]
    #[test]
    fn walk_keeps_non_utf8_file_names() {
        use std::os::unix::ffi::OsStringExt;

        let tree = FixtureTree::new(&[("ok.html", TESTHARNESS_HTML)]);
        let raw_name = std::ffi::OsString::from_vec(b"weird_\xff.html".to_vec());
        std::fs::write(tree.root.join(&raw_name), TESTHARNESS_HTML).unwrap();

        let m = Manifest::walk(&tree.root).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m.walk_errors, 0);
        // The id is lossy, consistent with the ids of UTF-8 names.
        let id = "weird_\u{fffd}.html";
        let item = m.items().iter().find(|t| t.id == id).unwrap();
        assert_eq!(item.kind, TestKind::Testharness);
        assert_eq!(item.url, format!("/{id}"));
    }

    #[cfg(unix)]
    #[test]
    fn walk_refuses_symlinked_root() {
        use std::os::unix::fs::symlink;

        let tree = FixtureTree::new(&[("a.html", TESTHARNESS_HTML)]);
        let link = tree.root.with_extension("link");
        let _ = std::fs::remove_file(&link);
        symlink(&tree.root, &link).unwrap();
        let err = Manifest::walk(&link).unwrap_err();
        assert!(
            matches!(err, ManifestError::Invalid(_)),
            "expected Invalid, got {err:?}"
        );
        let _ = std::fs::remove_file(&link);
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
