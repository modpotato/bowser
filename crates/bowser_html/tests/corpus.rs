//! html5lib tree-construction conformance corpus.
//!
//! Fixtures: `tests/fixtures/html5lib/*.dat`, vendored 2026-09-25 from
//! html5lib/html5lib-tests at commit `9329e64` (the last commit carrying
//! `tree-construction/` before those tests moved to WPT; ~456 KB total).
//!
//! Each `.dat` file holds `#data` / `#errors` / `#document` cases; the
//! expected-tree format is documented in the html5lib README (element
//! lines `| <tag>` / `| <svg tag>` / `| <math tag>`, attributes sorted
//! and shown as `name="value"` child lines, text as `"..."`, comments
//! as `<!-- ... -->`, doctypes as `<!DOCTYPE name "public" "system">`,
//! template fragments under a `content` marker).
//!
//! Mapping to our [`Node`] tree:
//!
//! - Element namespace designators (`svg `/`math `) map to
//!   [`Namespace::Svg`] / [`Namespace::MathMl`]; no designator means
//!   [`Namespace::Html`]. Attribute designators (`xlink href`, `xml
//!   lang`, ...) map from our flat qualified names (`xlink:href`) for
//!   the spec's adjusted foreign attributes, on foreign elements only.
//! - Attributes are compared order-insensitively (the format sorts
//!   them; we preserve source order).
//! - Template `content` lines map to [`Element::template_contents`].
//! - Source locations are ignored. Doctype `force_quirks` is not recorded
//!   by the format, so the harness re-derives the doctype token from the
//!   input (per the WHATWG tokenizer's doctype states) and checks both
//!   the token's `force_quirks` flag and the resulting quirks mode
//!   against an independent implementation of the WHATWG 13.2.6.4.1
//!   quirks table (see [`expected_quirks`]).
//!
//! Skipped aspects (counted, not compared):
//!
//! - `#errors` sections: only the tree is compared.
//!
//! Scripting follows the WPT convention: `#script-on` runs
//! scripting-enabled only, `#script-off` runs scripting-disabled only,
//! and an absent marker runs both modes.
//!
//! The corpus gate: every runnable case must either pass or be listed
//! in [`KNOWN_DEVIATIONS`] with a reason. A listed case that starts
//! passing fails the test too, so the list cannot go stale.

use bowser_html::atom::TagName;
use bowser_html::{
    Element, Namespace, Node, QuirksMode, ScriptElement, ScriptExecutor, ScriptHost,
    parse_fragment_with_scripting, parse_with_scripting_and_executor_and_quirks,
    parse_with_scripting_and_quirks,
};

// ---------------------------------------------------------------------------
// Known deviations: (relative fixture path, input prefix, reason).
//
// These are intentional project extensions or precisely understood spec
// gaps, NOT failures swept under the rug. Each entry explains why the
// implementation differs from the corpus expectation.
//
// Currently empty: the four scripted `#script-on` cases that used to be
// listed here (scripted/adoption01.dat, scripted/ark.dat, two cases in
// scripted/webkit01.dat) now pass via the [`ScriptExecutor`] hook and the
// [`FixtureExecutor`] stand-in below.
// ---------------------------------------------------------------------------

const KNOWN_DEVIATIONS: &[(&str, &str, &str)] = &[];

// ---------------------------------------------------------------------------
// Fixture script engine
// ---------------------------------------------------------------------------

/// Minimal script-engine stand-in for the html5lib scripted fixtures.
///
/// `bowser_html` is a parser: real script evaluation belongs to
/// `bowser_js`, and the parser only defines the execution hook (see
/// [`ScriptExecutor`]). The scripted fixtures exercise a tiny, fixed
/// script surface, so this evaluates exactly that surface and ignores
/// anything else:
///
/// - `document.write(<expr>)`, where `<expr>` is `+`-concatenated string
///   literals (scripted/webkit01.dat, both cases);
/// - `document.getElementById("A").id = "B"` (scripted/adoption01.dat);
/// - `document.getElementsByTagName("font")[2].setAttribute("size", "5")`
///   (scripted/ark.dat).
///
/// The corpus gate still fails on any tree mismatch, so a harness gap
/// surfaces as a normal corpus failure rather than a silent pass.
/// `executed` counts recognized script effects; the runner asserts every
/// `#script-on` case under `scripted/` produced at least one, so the
/// four formerly-deviating scripted cases cannot silently regress to
/// no-execution (a tree that accidentally matches without running the
/// script would still trip this pin).
struct FixtureExecutor {
    executed: usize,
}

impl ScriptExecutor for FixtureExecutor {
    fn execute_script(
        &mut self,
        host: &mut ScriptHost<'_, '_, '_>,
        _el: ScriptElement,
        _attrs: &[(String, String)],
        script: &str,
    ) {
        let s = script.trim();
        // All fixture scripts are single statements with an optional
        // trailing `;`; none ends with a string literal containing `;`.
        let s = s.strip_suffix(';').unwrap_or(s).trim();
        // `document.write(<string-expr>)`
        if let Some(rest) = s.strip_prefix("document.write(")
            && let Some(arg) = rest
                .strip_suffix(')')
                .and_then(|a| eval_string_concat(a.trim()))
        {
            host.document_write(&arg);
            self.executed += 1;
            return;
        }
        // `document.getElementById("A").id = "B"`
        if let Some(rest) = s.strip_prefix("document.getElementById(")
            && let Some((id, rest)) = parse_js_string(rest.trim_start())
            && let Some(assign) = rest
                .trim_start()
                .strip_prefix(')')
                .and_then(|r| r.trim_start().strip_prefix(".id"))
                .and_then(|r| r.trim_start().strip_prefix('='))
            && let Some((value, _)) = parse_js_string(assign.trim_start())
            && let Some(el) = host.get_element_by_id(&id)
        {
            host.set_attribute(el, "id", &value);
            self.executed += 1;
            return;
        }
        // `document.getElementsByTagName("font")[2].setAttribute("size", "5")`
        if let Some(rest) = s.strip_prefix("document.getElementsByTagName(")
            && let Some((tag, rest)) = parse_js_string(rest.trim_start())
            && let Some(after) = rest.trim_start().strip_prefix(')')
            && let Some(after) = after.trim_start().strip_prefix('[')
            && let Some((idx, after)) = parse_index(after)
            && let Some(args) = after
                .trim_start()
                .strip_prefix('.')
                .and_then(|r| r.strip_prefix("setAttribute"))
                .and_then(|r| r.strip_prefix('('))
                .and_then(|r| r.strip_suffix(')'))
            && let Some((name, value)) = parse_two_strings(args)
            && let Some(el) = host.get_elements_by_tag_name(&tag).get(idx).copied()
        {
            host.set_attribute(el, &name, &value);
            self.executed += 1;
        }
        // Anything else is outside the fixture surface: ignored.
    }
}

/// Parse one JS string literal (`'...'` / `"..."`, backslash escapes) at
/// the start of `s`; returns the value and the remainder.
fn parse_js_string(s: &str) -> Option<(String, &str)> {
    let quote = match s.as_bytes().first() {
        Some(b'\'') => '\'',
        Some(b'"') => '"',
        _ => return None,
    };
    let mut rest = &s[1..];
    let mut out = String::new();
    loop {
        let (i, c) = rest.char_indices().next()?;
        if c == quote {
            return Some((out, &rest[i + c.len_utf8()..]));
        }
        if c == '\\' {
            let (_, e) = rest[i + c.len_utf8()..].char_indices().next()?;
            match e {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                _ => out.push(e),
            }
            rest = &rest[i + c.len_utf8() + e.len_utf8()..];
        } else {
            out.push(c);
            rest = &rest[i + c.len_utf8()..];
        }
    }
}

/// Evaluate a `+`-concatenation of string literals: `"a" + 'b'`.
fn eval_string_concat(mut expr: &str) -> Option<String> {
    let mut out = String::new();
    loop {
        expr = expr.trim_start();
        let (lit, rest) = parse_js_string(expr)?;
        out.push_str(&lit);
        expr = rest.trim_start();
        if expr.is_empty() {
            return Some(out);
        }
        expr = expr.strip_prefix('+')?;
    }
}

/// Parse `N]` at the start of `s` (the inside of a `getElementsByTagName`
/// index); returns the index and the remainder.
fn parse_index(s: &str) -> Option<(usize, &str)> {
    let end = s.find(']')?;
    let idx: usize = s[..end].trim().parse().ok()?;
    Some((idx, &s[end + 1..]))
}

/// Parse `"a", "b"` (a two-string argument list).
fn parse_two_strings(s: &str) -> Option<(String, String)> {
    let (a, rest) = parse_js_string(s.trim_start())?;
    let rest = rest.trim_start().strip_prefix(',')?;
    let (b, rest) = parse_js_string(rest.trim_start())?;
    if rest.trim_start().is_empty() {
        Some((a, b))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// .dat parsing
// ---------------------------------------------------------------------------

struct Case {
    input: String,
    /// `Some((namespace, tag))` for `#document-fragment` cases: the
    /// context element selecting fragment parsing (WHATWG 13.4). The
    /// context line is `svg <tag>` / `math <tag>` for foreign
    /// namespaces, otherwise an HTML tag name.
    fragment: Option<(Namespace, String)>,
    script_on: bool,
    script_off: bool,
    doc_lines: Vec<String>,
}

fn split_cases(path: &std::path::Path, content: &str) -> Vec<Case> {
    let mut cases = Vec::new();
    let mut lines = content.lines().peekable();
    while lines.peek().is_some() {
        // Skip blank separator lines between cases.
        while lines.peek().is_some_and(|l| l.is_empty()) {
            lines.next();
        }
        if lines.peek().is_none() {
            break;
        }
        let first = lines.next().unwrap();
        assert_eq!(
            first,
            "#data",
            "case must start with #data in {}",
            path.display()
        );
        let mut data = Vec::new();
        for line in lines.by_ref() {
            if line == "#errors" {
                break;
            }
            data.push(line);
        }
        // Errors section: skip until a directive or #document.
        let mut fragment: Option<(Namespace, String)> = None;
        let mut script_on = false;
        let mut script_off = false;
        let mut doc_lines = Vec::new();
        let mut done = false;
        let iter = lines.by_ref();
        while let Some(line) = iter.next() {
            match line {
                "#new-errors" => continue,
                "#document-fragment" => {
                    // Next line is the context element: a leading
                    // "svg "/"math " selects the foreign namespace.
                    let ctx = iter.next().unwrap_or("");
                    let (ns, tag) = if let Some(t) = ctx.strip_prefix("svg ") {
                        (Namespace::Svg, t.to_string())
                    } else if let Some(t) = ctx.strip_prefix("math ") {
                        (Namespace::MathMl, t.to_string())
                    } else {
                        (Namespace::Html, ctx.to_string())
                    };
                    fragment = Some((ns, tag));
                }
                "#script-on" => script_on = true,
                "#script-off" => script_off = true,
                "#document" => {
                    // A blank line ends the case — unless it continues a
                    // multi-line text node (the previous line opened a
                    // `"` that has not been closed yet).
                    let mut open_quote = false;
                    for dline in iter.by_ref() {
                        if dline.is_empty() && !open_quote {
                            break; // blank line ends the case
                        }
                        assert_ne!(dline, "#data", "#data inside #document");
                        open_quote = if let Some(rest) = dline.strip_prefix("| ") {
                            let content = rest.trim_start_matches(' ');
                            content.starts_with('"')
                                && !(content.len() > 1 && content.ends_with('"'))
                        } else {
                            // Continuation line: the quote closes iff the
                            // line ends with one.
                            open_quote && !dline.ends_with('"')
                        };
                        doc_lines.push(dline.to_string());
                    }
                    done = true;
                    break;
                }
                _ => continue, // error detail lines
            }
        }
        assert!(done, "case missing #document");
        cases.push(Case {
            // The README: data lines joined, final newline removed.
            input: data.join("\n"),
            fragment,
            script_on,
            script_off,
            doc_lines,
        });
    }
    cases
}

// ---------------------------------------------------------------------------
// Expected-tree parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Exp {
    Doc(Vec<Exp>),
    El {
        tag: String,
        ns: Namespace,
        attrs: Vec<(String, String)>,
        kids: Vec<Exp>,
        contents: Vec<Exp>,
    },
    Text(String),
    Comment(String),
    Pi(String, String),
    Doctype {
        name: String,
        public: String,
        system: String,
    },
}

struct RawLine {
    depth: usize,
    text: String,
}

/// Split `#document` lines into (depth, text), joining multi-line text
/// nodes (continuation lines carry no `| ` prefix).
fn raw_lines(doc_lines: &[String]) -> Vec<RawLine> {
    let mut out = Vec::new();
    for line in doc_lines {
        if let Some(rest) = line.strip_prefix("| ") {
            let indent = rest.len() - rest.trim_start_matches(' ').len();
            assert!(indent % 2 == 0, "odd indent in {line:?}");
            out.push(RawLine {
                depth: indent / 2,
                text: rest.trim_start().to_string(),
            });
        } else {
            // Continuation of a multi-line text node.
            let prev = out.last_mut().expect("text continuation without a line");
            prev.text.push('\n');
            prev.text.push_str(line);
        }
    }
    out
}

fn parse_quoted(s: &str) -> (String, &str) {
    // s starts just after the opening quote; returns (value, rest).
    let mut val = String::new();
    let mut chars = s.char_indices();
    for (i, c) in chars.by_ref() {
        if c == '"' {
            return (val, &s[i + 1..]);
        }
        val.push(c);
    }
    panic!("unterminated quote in {s:?}");
}

fn parse_doctype(text: &str) -> Exp {
    // "<!DOCTYPE name>" or "<!DOCTYPE name \"public\" \"system\">"
    let inner = text.strip_prefix("<!DOCTYPE").unwrap();
    let inner = inner.strip_suffix('>').unwrap_or(inner);
    let inner = inner.strip_prefix(' ').unwrap_or(inner);
    if inner.is_empty() {
        return Exp::Doctype {
            name: String::new(),
            public: String::new(),
            system: String::new(),
        };
    }
    let (name, mut rest) = match inner.find(' ') {
        Some(i) => (inner[..i].to_string(), &inner[i + 1..]),
        None => (inner.to_string(), ""),
    };
    let (public, system) = if rest.starts_with('"') {
        let (p, r) = parse_quoted(&rest[1..]);
        rest = r.trim_start();
        let s = rest
            .strip_prefix('"')
            .map(|after| {
                // The system identifier may itself contain `"` (the dump
                // does not escape it), so the closing delimiter is the
                // last quote, not the first.
                match after.rfind('"') {
                    Some(i) => after[..i].to_string(),
                    None => after.to_string(),
                }
            })
            .unwrap_or_default();
        (p, s)
    } else {
        (String::new(), String::new())
    };
    Exp::Doctype {
        name,
        public,
        system,
    }
}

fn parse_element(text: &str) -> Exp {
    // "<tag>" | "<svg tag>" | "<math tag>"
    let inner = &text[1..text.len() - 1];
    let (ns, tag) = match inner.split_once(' ') {
        Some(("svg", t)) => (Namespace::Svg, t),
        Some(("math", t)) => (Namespace::MathMl, t),
        _ => (Namespace::Html, inner),
    };
    Exp::El {
        tag: tag.to_string(),
        ns,
        attrs: Vec::new(),
        kids: Vec::new(),
        contents: Vec::new(),
    }
}

fn parse_attr(text: &str) -> (String, String) {
    let (name, val) = text.split_once('=').expect("attr without =");
    let val = val.strip_prefix('"').unwrap_or(val);
    let val = val.strip_suffix('"').unwrap_or(val);
    (name.to_string(), val.to_string())
}

/// An element line is `<tag>`, `<svg tag>`, or `<math tag>`: it starts with
/// True for `name="value"` attribute lines. Excludes element, comment,
/// doctype, PI, and text lines.
fn is_attr_line(text: &str) -> bool {
    if !text.contains('=') || text.starts_with('"') {
        return false;
    }
    if text.starts_with('<') {
        // `<tag>`, `<!-- -->`, `<? ?>` end with `>`; a `<` attribute name
        // (e.g. `<=""`) does not.
        return !text.ends_with('>');
    }
    true
}

/// Classify one non-attribute, non-`content` document line.
fn parse_node_line(text: &str) -> Exp {
    if text.starts_with("<!DOCTYPE") {
        parse_doctype(text)
    } else if text.starts_with("<!-- ") && text.ends_with(" -->") {
        Exp::Comment(text[5..text.len() - 4].to_string())
    } else if text.starts_with('"') {
        Exp::Text(text[1..text.len() - 1].to_string())
    } else if let Some(stripped) = text.strip_prefix("<?") {
        // `<?target data?>`: strip `<?` and the `?>` close delimiter, then
        // split target/data. The `?` is syntax, not data.
        let inner = stripped.strip_suffix("?>").unwrap_or(stripped);
        let inner = inner.strip_suffix('>').unwrap_or(inner);
        match inner.split_once(' ') {
            Some((t, d)) => Exp::Pi(t.to_string(), d.to_string()),
            None => Exp::Pi(inner.to_string(), String::new()),
        }
    } else if text.starts_with('<') {
        parse_element(text)
    } else {
        panic!("unrecognized document line: {text:?}");
    }
}

/// Recursive descent over raw lines at `depth`: attribute lines attach
/// to the pending element's attrs, a `content` line switches subsequent
/// siblings into its template contents, anything else is a child node.
fn parse_kids(
    lines: &[RawLine],
    pos: &mut usize,
    depth: usize,
) -> (Vec<(String, String)>, Vec<Exp>, Vec<Exp>) {
    let mut attrs = Vec::new();
    let mut kids = Vec::new();
    let mut contents: Vec<Exp> = Vec::new();
    let mut to_contents = false;
    while *pos < lines.len() && lines[*pos].depth == depth {
        let text = lines[*pos].text.clone();
        *pos += 1;
        if text == "content" {
            // Template contents marker: the deeper block that follows
            // belongs to this element's template contents.
            let (_, k, _) = parse_kids(lines, pos, depth + 1);
            contents.extend(k);
            to_contents = true;
            continue;
        }
        if is_attr_line(&text) {
            let (n, v) = parse_attr(&text);
            attrs.push((n, v));
            continue;
        }
        let mut node = parse_node_line(&text);
        if let Exp::El {
            attrs: a,
            kids: k,
            contents: c,
            ..
        } = &mut node
        {
            let (aa, kk, cc) = parse_kids(lines, pos, depth + 1);
            *a = aa;
            *k = kk;
            *c = cc;
        }
        if to_contents {
            contents.push(node);
        } else {
            kids.push(node);
        }
    }
    (attrs, kids, contents)
}

fn parse_expected(doc_lines: &[String]) -> Exp {
    let raw = raw_lines(doc_lines);
    let mut pos = 0;
    let (_attrs, kids, _contents) = parse_kids(&raw, &mut pos, 0);
    assert_eq!(pos, raw.len(), "trailing document lines");
    Exp::Doc(kids)
}

// ---------------------------------------------------------------------------
// Comparison: actual Node vs expected Exp
// ---------------------------------------------------------------------------

/// The spec's adjusted foreign attributes: our tree keeps the qualified
/// name (`xlink:href`); the corpus uses the designator form
/// (`xlink href`). Only these exact names are adjusted, on foreign
/// elements only — HTML elements keep qualified names as-is.
fn is_adjusted_foreign_attr(name: &str) -> bool {
    matches!(
        name,
        "xlink:actuate"
            | "xlink:arcrole"
            | "xlink:href"
            | "xlink:role"
            | "xlink:show"
            | "xlink:title"
            | "xml:lang"
            | "xml:space"
            | "xmlns"
            | "xmlns:xlink"
    )
}

fn designator(ns: &Namespace, name: &str) -> String {
    if matches!(ns, Namespace::Svg | Namespace::MathMl) && is_adjusted_foreign_attr(name) {
        if name == "xmlns" {
            return "xmlns xmlns".to_string();
        }
        let (prefix, local) = name.split_once(':').unwrap();
        return format!("{prefix} {local}");
    }
    name.to_string()
}

fn sorted_attrs<N: AsRef<str>>(ns: &Namespace, attrs: &[(N, String)]) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = attrs
        .iter()
        .map(|(n, val)| (designator(ns, n.as_ref()), val.clone()))
        .collect();
    v.sort();
    v
}

fn compare(a: &Node, e: &Exp, path: &mut String) -> Result<(), String> {
    match (a, e) {
        (Node::Document(ak), Exp::Doc(ek)) => compare_kids(ak, ek, path),
        (
            Node::Element(ae),
            Exp::El {
                tag,
                ns,
                attrs,
                kids,
                contents,
            },
        ) => {
            if ae.tag != *tag {
                return Err(format!("tag {:?} != {tag:?}", ae.tag));
            }
            if ae.ns != *ns {
                return Err(format!("namespace {:?} != {ns:?}", ae.ns));
            }
            let aa = sorted_attrs(&ae.ns, &ae.attrs);
            let ea = sorted_attrs(ns, attrs);
            if aa != ea {
                return Err(format!("attrs {aa:?} != {ea:?}"));
            }
            let base = path.len();
            path.push_str("/template()");
            let cr = compare_exp_kids(
                &ae.template_contents.clone().unwrap_or_default(),
                contents,
                path,
            );
            path.truncate(base);
            cr?;
            compare_kids(&ae.children, kids, path)
        }
        (Node::Text { text: at, .. }, Exp::Text(et)) => {
            if at == et {
                Ok(())
            } else {
                Err(format!("text {at:?} != {et:?}"))
            }
        }
        (Node::Comment { data: ad, .. }, Exp::Comment(ed)) => {
            if ad == ed {
                Ok(())
            } else {
                Err(format!("comment {ad:?} != {ed:?}"))
            }
        }
        (
            Node::ProcessingInstruction {
                target: at,
                data: ad,
                ..
            },
            Exp::Pi(et, ed),
        ) => {
            if at == et && ad == ed {
                Ok(())
            } else {
                Err(format!("pi {at:?} {ad:?} != {et:?} {ed:?}"))
            }
        }
        (
            Node::Doctype(ad),
            Exp::Doctype {
                name,
                public,
                system,
            },
        ) => {
            let ap = ad.public_id.clone().unwrap_or_default();
            let asys = ad.system_id.clone().unwrap_or_default();
            if ad.name == *name && ap == *public && asys == *system {
                Ok(())
            } else {
                Err(format!(
                    "doctype {:?} {:?} {:?} != {name:?} {public:?} {system:?}",
                    ad.name, ap, asys
                ))
            }
        }
        _ => Err("node kind differs".to_string()),
    }
}

fn compare_kids(a: &[Node], e: &[Exp], path: &mut String) -> Result<(), String> {
    if a.len() != e.len() {
        return Err(format!("child count {} != {}", a.len(), e.len()));
    }
    for (i, (ak, ek)) in a.iter().zip(e.iter()).enumerate() {
        let base = path.len();
        path.push('/');
        path.push_str(&kid_label(ek, i));
        let r = compare(ak, ek, path);
        path.truncate(base);
        r?;
    }
    Ok(())
}

fn compare_exp_kids(a: &[Node], e: &[Exp], path: &mut String) -> Result<(), String> {
    compare_kids(a, e, path)
}

fn kid_label(e: &Exp, i: usize) -> String {
    match e {
        Exp::Doc(_) => format!("document[{i}]"),
        Exp::El { tag, .. } => format!("{tag}[{i}]"),
        Exp::Text(_) => format!("text[{i}]"),
        Exp::Comment(_) => format!("comment[{i}]"),
        Exp::Pi(_, _) => format!("pi[{i}]"),
        Exp::Doctype { .. } => format!("doctype[{i}]"),
    }
}

// ---------------------------------------------------------------------------
// Expected quirks mode: an independent oracle for WHATWG 13.2.6.4.1.
//
// The corpus format records the doctype's name/public/system but not its
// `force_quirks` flag, so the harness re-derives the doctype token from
// the input: [`scan_doctype_token`] finds the token the "initial"
// insertion mode would see, [`parse_doctype_token`] replays the
// tokenizer's doctype states (13.2.5.52-68) to recover `force_quirks`,
// and [`expected_quirks`] applies the spec's quirks table. This is
// deliberately a second implementation: it checks the parser's
// tokenizer (`force_quirks`) and tree-construction (`handle_doctype`)
// paths against the spec text rather than against themselves.
// ---------------------------------------------------------------------------

/// A doctype token as the WHATWG tokenizer's doctype states produce it.
struct ExpectedDoctype {
    /// ASCII-lowercased name ("" when missing).
    name: String,
    public: Option<String>,
    system: Option<String>,
    /// True when the token was bogus or truncated.
    force_quirks: bool,
}

fn is_ws(b: u8) -> bool {
    matches!(b, b'\t' | b'\n' | 0x0C | b'\r' | b' ')
}

fn find_seq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Find the doctype token the "initial" insertion mode would process:
/// ASCII whitespace, comments, and bogus comments/PIs are skipped; the
/// first start tag, end tag, text character, or EOF means no doctype
/// token exists (the mode's fallback sets quirks). A `<!DOCTYPE` inside
/// a comment is skipped with the comment, and a `<script>` start tag in
/// this position ends the scan before its contents could mislead it.
fn scan_doctype_token(input: &str) -> Option<ExpectedDoctype> {
    let b = input.as_bytes();
    let mut i = 0;
    loop {
        while i < b.len() && is_ws(b[i]) {
            i += 1;
        }
        if i >= b.len() || b[i] != b'<' {
            // EOF or a character token: the "initial" fallback sets
            // quirks and no doctype token is ever produced.
            return None;
        }
        let rest = &b[i..];
        if rest.starts_with(b"<!--") {
            let k = find_seq(&b[i + 4..], b"-->")?;
            i += 4 + k + 3;
            continue;
        }
        if rest.len() >= 9 && rest[1] == b'!' && rest[2..9].eq_ignore_ascii_case(b"DOCTYPE") {
            return Some(parse_doctype_token(&b[i + 9..]));
        }
        if rest.starts_with(b"<?") || (rest.len() >= 2 && rest[1] == b'!') {
            // Bogus comment (`<?...>` or `<!...>` that is neither comment
            // nor doctype): the tokenizer consumes through the next `>`.
            // Unterminated: the rest is comment.
            let k = b[i..].iter().position(|&c| c == b'>')?;
            i += k + 1;
            continue;
        }
        // A start tag, end tag, or stray `<` in "initial" mode sets
        // quirks and leaves the mode: no later doctype can apply.
        return None;
    }
}

fn kw_at(b: &[u8], i: usize, word: &[u8]) -> bool {
    b.len() >= i + word.len() && b[i..i + word.len()].eq_ignore_ascii_case(word)
}

/// Replay the WHATWG tokenizer's doctype states (13.2.5.52-68) over the
/// bytes just after `<!DOCTYPE`, recovering the token's name (already
/// ASCII-lowercased by the tokenizer), public/system identifiers, and
/// the `force_quirks` flag. Only the fields the quirks table reads are
/// kept.
fn parse_doctype_token(input: &[u8]) -> ExpectedDoctype {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum St {
        BeforeName,
        Name,
        AfterName,
        AfterPublicKw,
        BeforePublicId,
        PublicId,
        AfterPublicId,
        AfterSystemKw,
        BeforeSystemId,
        SystemId,
        AfterSystemId,
        Bogus,
    }
    let mut t = ExpectedDoctype {
        name: String::new(),
        public: None,
        system: None,
        force_quirks: false,
    };
    let mut i = 0;
    // "DOCTYPE state" on `>` or EOF sets force-quirks (missing name);
    // every other first character reconsumes in "before DOCTYPE name",
    // which this loop's first iteration implements directly.
    let mut st = St::BeforeName;
    let mut quote = b'"';
    let mut cur = String::new();
    loop {
        let c = if i < input.len() {
            Some(input[i])
        } else {
            None
        };
        match st {
            St::BeforeName => match c {
                None => {
                    t.force_quirks = true;
                    break;
                }
                Some(b'>') => {
                    t.force_quirks = true;
                    break;
                }
                Some(c) if is_ws(c) => i += 1,
                Some(c) => {
                    t.name.push((c as char).to_ascii_lowercase());
                    i += 1;
                    st = St::Name;
                }
            },
            St::Name => match c {
                None => {
                    t.force_quirks = true;
                    break;
                }
                Some(b'>') => break,
                Some(c) if is_ws(c) => {
                    i += 1;
                    st = St::AfterName;
                }
                Some(c) => {
                    t.name.push((c as char).to_ascii_lowercase());
                    i += 1;
                }
            },
            St::AfterName => match c {
                None => {
                    t.force_quirks = true;
                    break;
                }
                Some(b'>') => break,
                Some(c) if is_ws(c) => i += 1,
                Some(_) => {
                    if kw_at(input, i, b"PUBLIC") {
                        i += 6;
                        st = St::AfterPublicKw;
                    } else if kw_at(input, i, b"SYSTEM") {
                        i += 6;
                        st = St::AfterSystemKw;
                    } else {
                        t.force_quirks = true;
                        st = St::Bogus;
                    }
                }
            },
            St::AfterPublicKw => match c {
                None => {
                    t.force_quirks = true;
                    break;
                }
                Some(b'>') => {
                    t.force_quirks = true;
                    break;
                }
                Some(c) if is_ws(c) => {
                    i += 1;
                    st = St::BeforePublicId;
                }
                Some(b'"') | Some(b'\'') => {
                    quote = c.unwrap();
                    cur.clear();
                    i += 1;
                    st = St::PublicId;
                }
                Some(_) => {
                    t.force_quirks = true;
                    st = St::Bogus;
                }
            },
            St::BeforePublicId => match c {
                None => {
                    t.force_quirks = true;
                    break;
                }
                Some(b'>') => {
                    t.force_quirks = true;
                    break;
                }
                Some(c) if is_ws(c) => i += 1,
                Some(b'"') | Some(b'\'') => {
                    quote = c.unwrap();
                    cur.clear();
                    i += 1;
                    st = St::PublicId;
                }
                Some(_) => {
                    t.force_quirks = true;
                    st = St::Bogus;
                }
            },
            St::PublicId => match c {
                None => {
                    t.force_quirks = true;
                    t.public = Some(std::mem::take(&mut cur));
                    break;
                }
                Some(q) if q == quote => {
                    t.public = Some(std::mem::take(&mut cur));
                    i += 1;
                    st = St::AfterPublicId;
                }
                Some(0) => {
                    cur.push('\u{FFFD}');
                    i += 1;
                }
                Some(c) => {
                    cur.push(c as char);
                    i += 1;
                }
            },
            St::AfterPublicId => match c {
                None => {
                    t.force_quirks = true;
                    break;
                }
                Some(b'>') => break,
                Some(c) if is_ws(c) => {
                    i += 1;
                    st = St::BeforeSystemId;
                }
                Some(b'"') | Some(b'\'') => {
                    quote = c.unwrap();
                    cur.clear();
                    i += 1;
                    st = St::SystemId;
                }
                Some(_) => {
                    t.force_quirks = true;
                    st = St::Bogus;
                }
            },
            St::AfterSystemKw => match c {
                None => {
                    t.force_quirks = true;
                    break;
                }
                Some(b'>') => {
                    t.force_quirks = true;
                    break;
                }
                Some(c) if is_ws(c) => {
                    i += 1;
                    st = St::BeforeSystemId;
                }
                Some(b'"') | Some(b'\'') => {
                    quote = c.unwrap();
                    cur.clear();
                    i += 1;
                    st = St::SystemId;
                }
                Some(_) => {
                    t.force_quirks = true;
                    st = St::Bogus;
                }
            },
            St::BeforeSystemId => match c {
                None => {
                    t.force_quirks = true;
                    break;
                }
                // `>` here ends the token with the system identifier
                // still missing: no parse error, no force-quirks.
                Some(b'>') => break,
                Some(c) if is_ws(c) => i += 1,
                Some(b'"') | Some(b'\'') => {
                    quote = c.unwrap();
                    cur.clear();
                    i += 1;
                    st = St::SystemId;
                }
                Some(_) => {
                    t.force_quirks = true;
                    st = St::Bogus;
                }
            },
            St::SystemId => match c {
                None => {
                    t.force_quirks = true;
                    t.system = Some(std::mem::take(&mut cur));
                    break;
                }
                Some(q) if q == quote => {
                    t.system = Some(std::mem::take(&mut cur));
                    i += 1;
                    st = St::AfterSystemId;
                }
                Some(0) => {
                    cur.push('\u{FFFD}');
                    i += 1;
                }
                Some(c) => {
                    cur.push(c as char);
                    i += 1;
                }
            },
            St::AfterSystemId => match c {
                None => {
                    t.force_quirks = true;
                    break;
                }
                Some(b'>') => break,
                Some(c) if is_ws(c) => i += 1,
                Some(_) => {
                    t.force_quirks = true;
                    st = St::Bogus;
                }
            },
            // "bogus DOCTYPE state": consume to `>`; force-quirks was
            // already set on entry.
            St::Bogus => match c {
                None => break,
                Some(b'>') => break,
                Some(_) => i += 1,
            },
        }
    }
    t
}

/// ASCII case-insensitive prefix check (byte-boundary safe).
fn starts_with_ci(haystack: &str, prefix: &str) -> bool {
    haystack
        .get(..prefix.len())
        .is_some_and(|s| s.eq_ignore_ascii_case(prefix))
}

/// Quirks-mode public identifier prefixes, WHATWG 13.2.6.4.1 (copied
/// from the spec text, not from the parser, so the corpus checks the
/// table instead of echoing it).
const EXPECTED_QUIRKS_PUBLIC_PREFIXES: &[&str] = &[
    "+//Silmaril//dtd html Pro v0r11 19970101//",
    "-//AS//DTD HTML 3.0 asWedit + extensions//",
    "-//AdvaSoft Ltd//DTD HTML 3.0 asWedit + extensions//",
    "-//IETF//DTD HTML 2.0 Level 1//",
    "-//IETF//DTD HTML 2.0 Level 2//",
    "-//IETF//DTD HTML 2.0 Strict Level 1//",
    "-//IETF//DTD HTML 2.0 Strict Level 2//",
    "-//IETF//DTD HTML 2.0 Strict//",
    "-//IETF//DTD HTML 2.0//",
    "-//IETF//DTD HTML 2.1E//",
    "-//IETF//DTD HTML 3.0//",
    "-//IETF//DTD HTML 3.2 Final//",
    "-//IETF//DTD HTML 3.2//",
    "-//IETF//DTD HTML 3//",
    "-//IETF//DTD HTML Level 0//",
    "-//IETF//DTD HTML Level 1//",
    "-//IETF//DTD HTML Level 2//",
    "-//IETF//DTD HTML Level 3//",
    "-//IETF//DTD HTML Strict Level 0//",
    "-//IETF//DTD HTML Strict Level 1//",
    "-//IETF//DTD HTML Strict Level 2//",
    "-//IETF//DTD HTML Strict Level 3//",
    "-//IETF//DTD HTML Strict//",
    "-//IETF//DTD HTML//",
    "-//Metrius//DTD Metrius Presentational//",
    "-//Microsoft//DTD Internet Explorer 2.0 HTML Strict//",
    "-//Microsoft//DTD Internet Explorer 2.0 HTML//",
    "-//Microsoft//DTD Internet Explorer 2.0 Tables//",
    "-//Microsoft//DTD Internet Explorer 3.0 HTML Strict//",
    "-//Microsoft//DTD Internet Explorer 3.0 HTML//",
    "-//Microsoft//DTD Internet Explorer 3.0 Tables//",
    "-//Netscape Comm. Corp.//DTD HTML//",
    "-//Netscape Comm. Corp.//DTD Strict HTML//",
    "-//O'Reilly and Associates//DTD HTML 2.0//",
    "-//O'Reilly and Associates//DTD HTML Extended 1.0//",
    "-//O'Reilly and Associates//DTD HTML Extended Relaxed 1.0//",
    "-//SQ//DTD HTML 2.0 HoTMetaL + extensions//",
    "-//SoftQuad Software//DTD HoTMetaL PRO 6.0::19990601::extensions to HTML 4.0//",
    "-//SoftQuad//DTD HoTMetaL PRO 4.0::19971010::extensions to HTML 4.0//",
    "-//Spyglass//DTD HTML 2.0 Extended//",
    "-//Sun Microsystems Corp.//DTD HotJava HTML//",
    "-//Sun Microsystems Corp.//DTD HotJava Strict HTML//",
    "-//W3C//DTD HTML 3 1995-03-24//",
    "-//W3C//DTD HTML 3.2 Draft//",
    "-//W3C//DTD HTML 3.2 Final//",
    "-//W3C//DTD HTML 3.2//",
    "-//W3C//DTD HTML 3.2S Draft//",
    "-//W3C//DTD HTML 4.0 Frameset//",
    "-//W3C//DTD HTML 4.0 Transitional//",
    "-//W3C//DTD HTML Experimental 19960712//",
    "-//W3C//DTD HTML Experimental 970421//",
    "-//W3C//DTD W3 HTML//",
    "-//W3O//DTD W3 HTML 3.0//",
    "-//WebTechs//DTD Mozilla HTML 2.0//",
    "-//WebTechs//DTD Mozilla HTML//",
];

/// The expected quirks mode for a case input, from the spec text
/// (WHATWG 13.2.6.4.1), plus the expected `force_quirks` flag (`None`
/// when the input has no doctype token). Independent of the parser's
/// own tokenizer and quirks computation by construction.
fn expected_quirks(input: &str) -> (QuirksMode, Option<bool>) {
    let Some(t) = scan_doctype_token(input) else {
        // No doctype token: the "initial" insertion mode's fallback
        // sets quirks mode.
        return (QuirksMode::Quirks, None);
    };
    if t.force_quirks {
        return (QuirksMode::Quirks, Some(true));
    }
    if t.name != "html" {
        return (QuirksMode::Quirks, Some(false));
    }
    let public = t.public.as_deref().unwrap_or("");
    let system = t.system.as_deref().unwrap_or("");
    let system_missing = t.system.is_none();
    let quirks = [
        "-//W3O//DTD W3 HTML Strict 3.0//EN//",
        "-//W3C/DTD HTML 4.0 Transitional/EN",
        "HTML",
    ]
    .iter()
    .any(|p| public.eq_ignore_ascii_case(p))
        || system
            .eq_ignore_ascii_case("http://www.ibm.com/data/dtd/v11/ibmxhtml1-transitional.dtd")
        || EXPECTED_QUIRKS_PUBLIC_PREFIXES
            .iter()
            .any(|p| starts_with_ci(public, p))
        || ((system_missing || system.is_empty())
            && (starts_with_ci(public, "-//W3C//DTD HTML 4.01 Frameset//")
                || starts_with_ci(public, "-//W3C//DTD HTML 4.01 Transitional//")));
    if quirks {
        return (QuirksMode::Quirks, Some(false));
    }
    let limited = starts_with_ci(public, "-//W3C//DTD XHTML 1.0 Frameset//")
        || starts_with_ci(public, "-//W3C//DTD XHTML 1.0 Transitional//")
        || (!system_missing
            && !system.is_empty()
            && (starts_with_ci(public, "-//W3C//DTD HTML 4.01 Frameset//")
                || starts_with_ci(public, "-//W3C//DTD HTML 4.01 Transitional//")));
    (
        if limited {
            QuirksMode::LimitedQuirks
        } else {
            QuirksMode::NoQuirks
        },
        Some(false),
    )
}

/// The parsed document's doctype node, if any.
fn doctype_node(actual: &Node) -> Option<&bowser_html::Doctype> {
    match actual {
        Node::Document(kids) => kids.iter().find_map(|k| match k {
            Node::Doctype(d) => Some(d),
            _ => None,
        }),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

fn is_known(rel: &str, input: &str) -> Option<&'static str> {
    KNOWN_DEVIATIONS
        .iter()
        .find(|(f, prefix, _)| {
            if *f != rel || !input.starts_with(prefix) {
                return false;
            }
            // The 2-char `"<?"` prefix would also match `"<?#"`, `"<?foo"`,
            // etc.; require an exact match for it.
            prefix.len() != 2 || input.len() == 2
        })
        .map(|(_, _, reason)| *reason)
}

fn fixture_files() -> Vec<std::path::PathBuf> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("html5lib");
    let mut files = Vec::new();
    let mut dirs = vec![root];
    while let Some(dir) = dirs.pop() {
        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .into_iter()
            .map(|e| e.path())
            .collect();
        entries.sort();
        for p in entries {
            if p.is_dir() {
                dirs.push(p);
            } else if p.extension().is_some_and(|x| x == "dat") {
                files.push(p);
            }
        }
    }
    files.sort();
    files
}

#[test]
fn html5lib_tree_construction() {
    let mut pass = 0usize;
    let mut fragment_cases = 0usize;
    let mut known = 0usize;
    let mut unexpected: Vec<String> = Vec::new();
    let mut stale: Vec<String> = Vec::new();

    for path in fixture_files() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("html5lib");
        let rel = path
            .strip_prefix(&root)
            .unwrap()
            .to_string_lossy()
            .to_string();
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        for (idx, case) in split_cases(&path, &content).iter().enumerate() {
            // WPT: `#script-on` runs scripting-enabled only, `#script-off`
            // runs scripting-disabled only, an absent marker runs both.
            let modes: &[bool] = if case.script_on {
                &[true]
            } else if case.script_off {
                &[false]
            } else {
                &[false, true]
            };
            for &scripting in modes {
                let id = if modes.len() == 1 {
                    format!("{rel}#{idx}")
                } else {
                    format!("{rel}#{idx}{}", if scripting { "+js" } else { "-js" })
                };
                if case.fragment.is_some() {
                    fragment_cases += 1;
                }
                let expected = parse_expected(&case.doc_lines);
                // Document parses also report the quirks mode (driven by
                // the doctype's `force_quirks` flag); fragment parses do
                // not compute one.
                let (actual, actual_quirks): (Node, Option<QuirksMode>) = match &case.fragment {
                    Some((ns, tag)) => {
                        let ctx = Element {
                            tag: TagName::from(tag.as_str()),
                            ns: ns.clone(),
                            attrs: Vec::new(),
                            children: Vec::new(),
                            template_contents: None,
                            shadow_root: None,
                            loc: None,
                        };
                        (
                            parse_fragment_with_scripting(case.input.as_bytes(), &ctx, scripting),
                            None,
                        )
                    }
                    None => {
                        // Scripting-enabled document parses run with the
                        // fixture script engine attached, so the
                        // `#script-on` cases exercise real script effects
                        // (`document.write`, DOM mutation); scripting off
                        // keeps the historical no-execution parse.
                        if scripting {
                            let mut executor = FixtureExecutor { executed: 0 };
                            let (parsed, quirks) = parse_with_scripting_and_executor_and_quirks(
                                case.input.as_bytes(),
                                true,
                                &mut executor,
                            );
                            // The scripted fixtures' expected trees all
                            // depend on script effects: a `#script-on`
                            // case under `scripted/` that executed
                            // nothing recognizable means the harness
                            // stopped covering its script surface, even
                            // if the tree happened to match.
                            let is_scripted = std::path::Path::new(&rel)
                                .components()
                                .any(|c| c.as_os_str() == "scripted");
                            if case.script_on && is_scripted && executor.executed == 0 {
                                unexpected.push(format!(
                                    "{id}: #script-on case executed no recognized script; input: {:?}",
                                    truncate(&case.input, 120)
                                ));
                            }
                            (parsed, Some(quirks))
                        } else {
                            let (parsed, quirks) =
                                parse_with_scripting_and_quirks(case.input.as_bytes(), false);
                            (parsed, Some(quirks))
                        }
                    }
                };
                let mut p = String::from("$");
                let mut result = compare(&actual, &expected, &mut p);
                // The corpus format does not record doctype `force_quirks`,
                // so the expected flag and quirks mode are re-derived from
                // the input by the independent [`expected_quirks`] oracle.
                // A mismatch fails the case exactly like a tree mismatch.
                if let Some(aq) = actual_quirks {
                    let (expected_mode, expected_force_quirks) = expected_quirks(&case.input);
                    if aq != expected_mode {
                        result = result.and(Err(format!(
                            "quirks mode {aq:?} != expected {expected_mode:?}"
                        )));
                    }
                    let actual_force_quirks = doctype_node(&actual).map(|d| d.force_quirks);
                    if actual_force_quirks != expected_force_quirks {
                        result = result.and(Err(format!(
                            "doctype force_quirks {actual_force_quirks:?} != \
                             expected {expected_force_quirks:?}"
                        )));
                    }
                }
                let known_reason = is_known(&rel, &case.input);
                match (result, known_reason) {
                    (Ok(_), None) => pass += 1,
                    (Ok(_), Some(reason)) => stale.push(format!(
                        "{id}: listed as known deviation ({reason}) but now passes; input: {:?}",
                        truncate(&case.input, 80)
                    )),
                    (Err(_), Some(_)) => known += 1,
                    (Err(diff), None) => unexpected.push(format!(
                        "{id}: at {p}: {diff}; input: {:?}",
                        truncate(&case.input, 120)
                    )),
                }
            }
        }
    }

    println!(
        "html5lib corpus: {pass} passed, {known} known deviations, \
         {fragment_cases} fragment cases"
    );
    let mut msg = String::new();
    if !stale.is_empty() {
        msg.push_str(&format!(
            "\n{} STALE known deviations (now passing):\n  ",
            stale.len()
        ));
        msg.push_str(&stale.join("\n  "));
    }
    if !unexpected.is_empty() {
        msg.push_str(&format!("\n{} UNEXPECTED failures:\n  ", unexpected.len()));
        msg.push_str(&unexpected.join("\n  "));
    }
    assert!(msg.is_empty(), "corpus failures:{msg}");
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
}
