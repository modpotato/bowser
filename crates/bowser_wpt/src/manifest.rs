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
//! `MANIFEST.json` shape (v9, as written by `wpt manifest`):
//! ```json
//! {"version": 9, "url_base": "/", "items": {
//!   "testharness": {"dom": {"x.html": ["da39a3ee5e6b4b0d3255bfef95601890afd80709", [null, {}]]}},
//!   "reftest":     {"css": {"y.html": ["0beec7b5ea3f0c415baa777ef162b3c5a237762", ["/css/y.html", {}]]}}
//! }}
//! ```
//!
//! `items` nests one path segment per object level; each leaf is
//! `[<40-char hex file hash>, item-tuple, ...]` where a tuple is
//! `[url-or-null, ...]`. A null url means the test is served at
//! `url_base` + the key path (wpt's `URLManifestItem.to_json` emits
//! `rel_url=None` when the url equals the path). One path may carry
//! several tuples (multi-global `.any.js` variants).

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
    /// Parse a `MANIFEST.json` document (v9, as written by `wpt manifest`).
    ///
    /// A malformed item-type section or entry is skipped and counted in
    /// [`Manifest::skipped`]; only a wrong `version` or a structurally
    /// broken document is an error.
    pub fn from_json(text: &str) -> Result<Manifest, ManifestError> {
        let doc = json::parse(text).map_err(|e| ManifestError::Invalid(e.to_string()))?;
        // wpt raises ManifestVersionMismatch on anything but the current version.
        if !version_supported(&doc) {
            return Err(ManifestError::VersionMismatch(describe_value(
                doc.get("version"),
            )));
        }
        let url_base = doc.get("url_base").and_then(Json::as_str).unwrap_or("/");
        let items_obj = doc.get("items").and_then(Json::as_obj).ok_or_else(|| {
            ManifestError::Invalid("MANIFEST.json has no \"items\" object".into())
        })?;

        let mut manifest = Manifest::default();
        for (type_name, section) in items_obj {
            let kind = TestKind::from_manifest_type(type_name);
            match section.as_obj() {
                Some(tree) => {
                    let mut segments = Vec::new();
                    walk_items(tree, &mut segments, url_base, &kind, &mut manifest);
                }
                // One malformed section must not abort the whole manifest.
                None => manifest.skipped += 1,
            }
        }
        // Deterministic order regardless of JSON key order.
        manifest
            .items
            .sort_by(|a, b| a.id.cmp(&b.id).then_with(|| a.url.cmp(&b.url)));
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

/// The only `MANIFEST.json` format version this parser understands.
/// Matches wpt's `Manifest.CURRENT_VERSION`.
const MANIFEST_VERSION: f64 = 9.0;

fn version_supported(doc: &Json) -> bool {
    matches!(doc.get("version"), Some(Json::Num(n)) if *n == MANIFEST_VERSION)
}

/// Human-readable rendering of a JSON value for error messages.
fn describe_value(value: Option<&Json>) -> String {
    match value {
        None => "missing".to_owned(),
        Some(Json::Null) => "null".to_owned(),
        Some(Json::Bool(b)) => b.to_string(),
        Some(Json::Num(n)) => {
            if n.fract() == 0.0 && n.abs() < 1e15 {
                format!("{}", *n as i64)
            } else {
                n.to_string()
            }
        }
        Some(Json::Str(s)) => format!("{s:?}"),
        Some(Json::Arr(_)) => "array".to_owned(),
        Some(Json::Obj(_)) => "object".to_owned(),
    }
}

/// Walk one item-type section of a v9 manifest. Intermediate objects nest
/// one path segment per level; leaves are `[file-hash, item-tuple, ...]`
/// arrays keyed by the final segment.
fn walk_items<'a>(
    node: &'a [(String, Json)],
    segments: &mut Vec<&'a str>,
    url_base: &str,
    kind: &TestKind,
    manifest: &mut Manifest,
) {
    for (segment, value) in node {
        match value {
            Json::Obj(children) => {
                segments.push(segment);
                walk_items(children, segments, url_base, kind, manifest);
                segments.pop();
            }
            Json::Arr(leaf) => {
                let id = join_segments(segments, segment);
                parse_leaf(&id, leaf, url_base, kind, manifest);
            }
            _ => manifest.skipped += 1,
        }
    }
}

/// Join nested path segments into a manifest id (a checkout-relative path
/// with forward slashes), normalizing backslashes the way
/// [`Manifest::walk`] does.
fn join_segments(segments: &[&str], leaf: &str) -> String {
    let mut id = String::new();
    for segment in segments {
        id.push_str(segment);
        id.push('/');
    }
    id.push_str(leaf);
    id.replace('\\', "/")
}

/// Parse one manifest leaf: `[<40-char hex file hash>, item-tuple, ...]`.
/// Each tuple is `[url-or-null, ...]` and yields one [`TestItem`]; a path
/// may carry several tuples (e.g. multi-global `.any.js` variants).
fn parse_leaf(id: &str, leaf: &[Json], url_base: &str, kind: &TestKind, manifest: &mut Manifest) {
    let [Json::Str(hash), tuples @ ..] = leaf else {
        manifest.skipped += 1;
        return;
    };
    if !is_file_hash(hash) || tuples.is_empty() {
        manifest.skipped += 1;
        return;
    }
    for tuple in tuples {
        match tuple_url(tuple, id, url_base) {
            Some(url) => manifest.items.push(TestItem {
                id: id.to_owned(),
                url,
                kind: kind.clone(),
                path: None,
            }),
            None => manifest.skipped += 1,
        }
    }
}

/// The leaf's first element is the file's git blob hash: 40 hex chars.
fn is_file_hash(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Resolve the served URL for one item tuple, `[url-or-null, ...]`.
///
/// A null url element means the test is served at `url_base` + the manifest
/// key path (wpt's `URLManifestItem.to_json` emits `rel_url=None` when the
/// url equals the path). Item types with no url slot at all (e.g. crashtest
/// tuples are just `[extras]`) resolve the same way.
fn tuple_url(tuple: &Json, id: &str, url_base: &str) -> Option<String> {
    let elems = tuple.as_arr()?;
    match elems.first() {
        Some(Json::Str(url)) => Some(join_url(url_base, url)),
        Some(Json::Null) | Some(Json::Obj(_)) => Some(join_url(url_base, id)),
        _ => None,
    }
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
    /// `MANIFEST.json` has a `version` this parser does not understand.
    /// Carries a description of the version value that was found.
    VersionMismatch(String),
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ManifestError::Io(e) => write!(f, "manifest I/O error: {e}"),
            ManifestError::Invalid(msg) => write!(f, "invalid manifest: {msg}"),
            ManifestError::VersionMismatch(found) => {
                write!(f, "unsupported manifest version {found}: expected v9")
            }
        }
    }
}

impl std::error::Error for ManifestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ManifestError::Io(e) => Some(e),
            ManifestError::Invalid(_) | ManifestError::VersionMismatch(_) => None,
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

    /// 40-char hex file hashes, as v9 leaves carry.
    const H1: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const H2: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const H3: &str = "cccccccccccccccccccccccccccccccccccccccc";
    const H4: &str = "dddddddddddddddddddddddddddddddddddddddd";
    const H5: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const H6: &str = "ffffffffffffffffffffffffffffffffffffffff";
    const H7: &str = "1111111111111111111111111111111111111111";

    #[test]
    fn from_json_reads_real_v9_shape() {
        let json = format!(
            r#"{{
            "version": 9,
            "url_base": "/",
            "items": {{
                "testharness": {{
                    "dom": {{
                        "a.html": ["{H1}", [null, {{}}]],
                        "b.html": ["{H2}", ["/dom/b.html", {{"timeout": "long"}}]]
                    }},
                    "html": {{
                        "multi.any.js": [
                            "{H3}",
                            [null, {{}}, []],
                            ["/html/multi.any.worker.html", {{}}, []]
                        ]
                    }},
                    "bogus-leaf": ["not-a-hash", [null, {{}}]],
                    "bad-leaf": [["no", "hash", "here"]]
                }},
                "reftest": {{
                    "css": {{
                        "c.html": ["{H4}", [null, [["==", "/css/ref.html"]]]]
                    }}
                }},
                "manual": {{
                    "html": {{
                        "d.html": ["{H5}", [null, {{}}]]
                    }}
                }},
                "wdspec": {{
                    "webdriver": {{
                        "e.py": ["{H6}", [null, {{}}]]
                    }}
                }},
                "crashtest": {{
                    "x": {{
                        "y.html": ["{H7}", [{{}}]]
                    }}
                }},
                "broken-section": [1, 2, 3]
            }}
        }}"#
        );
        let m = Manifest::from_json(&json).unwrap();
        // 2 testharness + 2 multi-global variants + reftest + manual + wdspec + crashtest.
        assert_eq!(m.len(), 8);
        // bogus-leaf (bad hash), bad-leaf (not a leaf array), broken-section
        // (not an object): skipped and counted, never fatal.
        assert_eq!(m.skipped, 3);

        assert_eq!(m.by_kind(&TestKind::Testharness).len(), 4);
        assert_eq!(m.by_kind(&TestKind::Reftest).len(), 1);
        assert_eq!(m.by_kind(&TestKind::Manual).len(), 1);
        assert_eq!(m.by_kind(&TestKind::Other("wdspec".into())).len(), 1);
        assert_eq!(m.by_kind(&TestKind::Crashtest).len(), 1);

        // Null urls resolve to url_base + the key path.
        let a = m.items().iter().find(|t| t.id == "dom/a.html").unwrap();
        assert_eq!(a.url, "/dom/a.html");
        // Explicit urls are honored.
        let b = m.items().iter().find(|t| t.id == "dom/b.html").unwrap();
        assert_eq!(b.url, "/dom/b.html");
        // Multi-tuple paths yield one item per tuple.
        let variants: Vec<&TestItem> = m
            .items()
            .iter()
            .filter(|t| t.id == "html/multi.any.js")
            .collect();
        assert_eq!(variants.len(), 2);
        assert_eq!(variants[0].url, "/html/multi.any.js");
        assert_eq!(variants[1].url, "/html/multi.any.worker.html");
        // Crashtest tuples carry no url slot; the path is the url.
        let y = m.items().iter().find(|t| t.id == "x/y.html").unwrap();
        assert_eq!(y.url, "/x/y.html");
        assert_eq!(y.kind, TestKind::Crashtest);

        // Deterministic order regardless of JSON key order.
        let first = &m.items()[0];
        assert_eq!(first.id, "css/c.html");
        assert_eq!(first.url, "/css/c.html");
        assert_eq!(first.kind, TestKind::Reftest);
    }

    #[test]
    fn from_json_rejects_wrong_version() {
        for version in ["8", "10", "\"9\"", "null"] {
            let json = format!(r#"{{"version": {version}, "url_base": "/", "items": {{}}}}"#);
            let err = Manifest::from_json(&json).unwrap_err();
            assert!(
                matches!(err, ManifestError::VersionMismatch(_)),
                "version {version}: {err:?}"
            );
        }
        // A missing version is a mismatch too, like wpt's ManifestVersionMismatch.
        let err = Manifest::from_json(r#"{"url_base": "/", "items": {}}"#).unwrap_err();
        assert!(matches!(err, ManifestError::VersionMismatch(_)));
    }

    #[test]
    fn from_json_rejects_garbage() {
        assert!(Manifest::from_json("not json").is_err());
        // Right version, but no items object.
        let err = Manifest::from_json(r#"{"version": 9}"#).unwrap_err();
        assert!(matches!(err, ManifestError::Invalid(_)));
        assert!(Manifest::from_json(r#"{"version": 9, "items": []}"#).is_err());
    }

    #[test]
    fn from_json_applies_url_base() {
        let json = format!(
            r#"{{"version": 9, "url_base": "/_wpt_/", "items": {{
            "testharness": {{"sub": {{
                "a.html": ["{H1}", [null, {{}}]],
                "b.html": ["{H2}", ["b.html", {{}}]]
            }}}}
        }}}}"#
        );
        let m = Manifest::from_json(&json).unwrap();
        // Null url: url_base + key path.
        assert_eq!(m.items()[0].id, "sub/a.html");
        assert_eq!(m.items()[0].url, "/_wpt_/sub/a.html");
        // Relative explicit url: joined onto url_base.
        assert_eq!(m.items()[1].url, "/_wpt_/b.html");
    }

    #[test]
    fn from_json_normalizes_backslash_ids() {
        let json = format!(
            r#"{{"version": 9, "url_base": "/", "items": {{
            "testharness": {{"dom": {{"sub\\dir\\f.html": ["{H1}", [null, {{}}]]}}}}
        }}}}"#
        );
        let m = Manifest::from_json(&json).unwrap();
        assert_eq!(m.items()[0].id, "dom/sub/dir/f.html");
        assert_eq!(m.items()[0].url, "/dom/sub/dir/f.html");
    }

    #[test]
    fn from_json_handles_root_level_files() {
        let json = format!(
            r#"{{"version": 9, "url_base": "/", "items": {{
            "testharness": {{"top.html": ["{H1}", [null, {{}}]]}}
        }}}}"#
        );
        let m = Manifest::from_json(&json).unwrap();
        assert_eq!(m.items()[0].id, "top.html");
        assert_eq!(m.items()[0].url, "/top.html");
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
}
