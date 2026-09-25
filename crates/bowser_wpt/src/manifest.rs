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
//! `MANIFEST.json` shape (v9, exactly as `wpt manifest` writes it):
//! ```json
//! {"version": 9, "url_base": "/", "items": {
//!   "testharness": {"dom/a.html": [[null, {}]]},
//!   "reftest":     {"css/b.html": [[null, [["/css/b-ref.html", "=="]], {}]]}
//! }}
//! ```
//!
//! - `version` must be 9; anything else is rejected (regenerate the manifest).
//! - Each path maps to a *list* of items: one file can yield several tests
//!   (multi-global variants like `?1=1`, `?2=2`).
//! - A testharness item is `[url, extras]`; a reftest item is
//!   `[url, references, extras]`. `url` is `null` when it equals the path
//!   (wpt's `URLManifestItem.to_json` emits `rel_url=None` then), and the
//!   served URL is `url_base` + path in that case.
//! - Entries whose shape is not understood are skipped and counted in
//!   [`Manifest::skipped`]; they never fail the parse.
//!
//! Two older shapes are also accepted leniently: path segments nested one
//! object level per segment (`{"dom": {"a.html": ...}}`), and leaves with a
//! leading `<40-char hex file hash>` before the item tuples.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use crate::json::{self, Json};
use crate::reftest::{self, RefRel, Reference};

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
    /// Reftest references, parsed at manifest time from the
    /// `[[ref_url, "=="|"="], ...]` list in `MANIFEST.json` (or scraped
    /// from the file by [`Manifest::walk`]). Empty for non-reftest items.
    pub references: Vec<Reference>,
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
    /// Only `.html`/`.htm`/`.xhtml`/`.xht` files are considered. Files whose
    /// name ends in `-ref` are reference files, not tests, and are skipped.
    /// Support and infrastructure directories (`tools/`, `resources/`,
    /// `support/`, `common/`, `conformance-checkers/`, `reference/`,
    /// `infrastructure/`, `.git/`) are skipped at any depth.
    /// Classification (see [`classify_file`] for the scan-window policy):
    ///
    /// - a `<link>` tag with `rel` exactly `match`/`mismatch` (any quoting,
    ///   optional whitespace around `=`) → [`TestKind::Reftest`]
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

/// The only `MANIFEST.json` format version this parser understands.
/// Matches wpt's `Manifest.CURRENT_VERSION`.
fn version_supported(doc: &Json) -> bool {
    matches!(doc.get("version"), Some(Json::Int(9)))
}

/// Human-readable rendering of a JSON value for error messages.
fn describe_value(value: Option<&Json>) -> String {
    match value {
        None => "missing".to_owned(),
        Some(Json::Null) => "null".to_owned(),
        Some(Json::Bool(b)) => b.to_string(),
        Some(Json::Int(i)) => i.to_string(),
        Some(Json::Float(f)) => {
            if f.fract() == 0.0 && f.abs() < 1e15 {
                format!("{}", *f as i64)
            } else {
                f.to_string()
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

/// Parse one manifest leaf into [`TestItem`]s.
///
/// Two leaf shapes are accepted:
/// - `[<40-char hex file hash>, item-tuple, ...]` (hash-prefixed leaves),
/// - `[item-tuple, ...]` (a bare list of tuples, as `wpt manifest` writes).
///
/// Each tuple is `[url-or-null, ...]` and yields one [`TestItem`]; a path
/// may carry several tuples (e.g. multi-global `.any.js` variants). A
/// tuple whose shape is not understood is skipped and counted; it never
/// fails the parse.
fn parse_leaf(id: &str, leaf: &[Json], url_base: &str, kind: &TestKind, manifest: &mut Manifest) {
    let tuples: &[Json] = match leaf {
        [Json::Str(hash), rest @ ..] if is_file_hash(hash) => rest,
        // A leading string that is not a file hash: not a shape we know.
        [Json::Str(_), ..] => {
            manifest.skipped += 1;
            return;
        }
        elems => elems,
    };
    if tuples.is_empty() {
        manifest.skipped += 1;
        return;
    }
    for tuple in tuples {
        match parse_tuple(tuple, id, url_base, kind) {
            Some(item) => manifest.items.push(item),
            None => manifest.skipped += 1,
        }
    }
}

/// Parse one item tuple: `[url-or-null, ...]`.
///
/// Every element after the url slot is extras or a reference list, i.e. an
/// object or an array; anything else is not a v9 item tuple and the tuple
/// is skipped.
fn parse_tuple(tuple: &Json, id: &str, url_base: &str, kind: &TestKind) -> Option<TestItem> {
    let elems = tuple.as_arr()?;
    if !elems
        .iter()
        .skip(1)
        .all(|e| matches!(e, Json::Obj(_) | Json::Arr(_)))
    {
        return None;
    }
    Some(TestItem {
        id: id.to_owned(),
        url: tuple_url(tuple, id, url_base)?,
        kind: kind.clone(),
        path: None,
        references: manifest_references(tuple),
    })
}

/// Extract the reftest reference list from one item tuple:
/// `[url, [[ref_url, "=="|"="], ...], extras]`.
///
/// Returns an empty list for non-reftest tuples and for malformed
/// reference lists. The runner treats "no references" as a skip, so
/// leniency here is safe. Unknown relations are dropped, not fatal.
fn manifest_references(tuple: &Json) -> Vec<Reference> {
    let refs: &[Json] = tuple
        .as_arr()
        .and_then(|elems| elems.get(1))
        .and_then(Json::as_arr)
        .unwrap_or(&[]);
    let mut out = Vec::new();
    for r in refs {
        let entry = r.as_arr().unwrap_or(&[]);
        let (Some(url), Some(rel)) = (
            entry.first().and_then(Json::as_str),
            entry.get(1).and_then(Json::as_str),
        ) else {
            continue;
        };
        let rel = match rel {
            "==" => RefRel::Match,
            "!=" => RefRel::Mismatch,
            _ => continue,
        };
        out.push(Reference {
            rel,
            href: url.to_owned(),
        });
    }
    out
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
    let mut entries = collect_entries(std::fs::read_dir(dir)?, manifest);
    entries.sort();
    for path in entries {
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
            let references = if kind == TestKind::Reftest {
                // classify_file just read this file successfully, so a
                // scrape failure here is unexpected; degrade to no
                // references rather than failing the whole walk.
                reftest::extract_references(&path).unwrap_or_default()
            } else {
                Vec::new()
            };
            manifest.items.push(TestItem {
                id: id.clone(),
                url: format!("/{id}"),
                kind,
                path: Some(path),
                references,
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

/// Whether a file stem names a reftest *reference* file rather than a test.
///
/// Mirrors upstream wpt `tools/manifest/sourcefile.py`:
/// `reference_file_re = re.compile(r'(^|[\\-_])(not)?ref[0-9]*([\\-_]|$)')`,
/// searched (not full-matched) against the file stem. Case-sensitive, like
/// upstream.
fn is_reference_name(stem: &str) -> bool {
    let bytes = stem.as_bytes();
    // Equivalent to `re.search`: try the anchored match at every position,
    // with both alternatives of `(^|[\\-_])` where they apply.
    (0..=bytes.len()).any(|start| {
        match_reference_at(bytes, start, true) || match_reference_at(bytes, start, false)
    })
}

/// Try the upstream `reference_file_re` anchored at `start`. `use_caret`
/// selects the `^` alternative of `(^|[\\-_])`, otherwise the `[\\-_]`
/// one.
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
    // `[0-9]*([\\-_]|$)`
    let mut tail = tail;
    while tail.first().is_some_and(|c| c.is_ascii_digit()) {
        tail = &tail[1..];
    }
    tail.is_empty() || tail[0] == b'-' || tail[0] == b'_'
}

fn has_test_extension(name: &str) -> bool {
    name.rsplit('.').next().is_some_and(|ext| {
        let ext = ext.to_ascii_lowercase();
        TEST_EXTENSIONS.iter().any(|want| *want == ext)
    })
}

/// Classify one HTML file by looking for reftest/testharness markers.
/// Only the first 128 KiB are read; markers live in `<head>`.
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
        assert_eq!(m.walk_errors, 0);
    }

    #[test]
    fn walk_missing_root_is_io_error() {
        let err = Manifest::walk(Path::new("/nonexistent-bowser-wpt-root")).unwrap_err();
        assert!(matches!(err, ManifestError::Io(_)));
    }

    // W2: reftest references parsed from real v9 `MANIFEST.json` tuples:
    // `[url, [[ref_url, "=="|"="], ...], extras]`.
    #[test]
    fn from_json_reads_reftest_references() {
        let json = format!(
            r#"{{"version": 9, "url_base": "/", "items": {{
            "reftest": {{"css": {{
                "b.html": ["{H1}", [null, [["/css/b-ref.html", "=="]], {{}}]],
                "c.html": ["{H2}", [null, [["/css/c-ref.html", "!="]], {{}}]],
                "d.html": ["{H3}", [null, [["/css/d-ref.html", "=="], ["/css/d-ref2.html", "!="]], {{}}]],
                "e.html": ["{H4}", [null, [["/css/e-ref.html", "=="], ["ignored", "bogus"]], {{}}]],
                "f.html": ["{H5}", [null, {{}}]]
            }}}}
        }}}}"#
        );
        let m = Manifest::from_json(&json).unwrap();
        assert_eq!(m.len(), 5);
        let refs = |id: &str| {
            m.items()
                .iter()
                .find(|t| t.id == id)
                .map(|t| t.references.clone())
                .unwrap()
        };
        assert_eq!(
            refs("css/b.html"),
            vec![Reference {
                rel: RefRel::Match,
                href: "/css/b-ref.html".into(),
            }]
        );
        assert_eq!(
            refs("css/c.html"),
            vec![Reference {
                rel: RefRel::Mismatch,
                href: "/css/c-ref.html".into(),
            }]
        );
        assert_eq!(refs("css/d.html").len(), 2);
        // Unknown relations are dropped, not fatal.
        assert_eq!(refs("css/e.html").len(), 1);
        // Tuples without a reference list yield no references.
        assert!(refs("css/f.html").is_empty());
    }

    // W2: walk mode scrapes references for reftest items so a checkout
    // without MANIFEST.json still gets them.
    #[test]
    fn walk_scrapes_reftest_references() {
        let tree = FixtureTree::new(&[
            ("css/b.html", REFTHTEST_HTML),
            ("css/c.html", REFTHTEST_MISMATCH_HTML),
            ("dom/a.html", TESTHARNESS_HTML),
        ]);
        let m = Manifest::walk(&tree.root).unwrap();
        assert_eq!(m.walk_errors, 0);
        let b = m.items().iter().find(|t| t.id == "css/b.html").unwrap();
        assert_eq!(b.kind, TestKind::Reftest);
        assert_eq!(
            b.references,
            vec![Reference {
                rel: RefRel::Match,
                href: "ref.html".into(),
            }]
        );
        let c = m.items().iter().find(|t| t.id == "css/c.html").unwrap();
        assert_eq!(
            c.references,
            vec![Reference {
                rel: RefRel::Mismatch,
                href: "not-ref.html".into(),
            }]
        );
        // Non-reftest items carry no references.
        let a = m.items().iter().find(|t| t.id == "dom/a.html").unwrap();
        assert!(a.references.is_empty());
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
