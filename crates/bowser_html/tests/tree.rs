//! Tree-construction tests for `bowser_html`, driven against the WHATWG
//! HTML standard (13.2.6). Expected serializations were hand-traced from
//! the spec; each test names the spec section it exercises.

use bowser_html::atom::{AttrName, TagName};
use bowser_html::{Element, Namespace, Node, QuirksMode, SourceLoc, parse, parse_with_quirks};

/// Render a parsed tree as compact HTML for assertions. Template contents
/// are rendered inside the template element, after any regular children.
fn render(n: &Node, out: &mut String) {
    match n {
        Node::Document(kids) => {
            for k in kids {
                render(k, out);
            }
        }
        Node::Element(e) => {
            out.push('<');
            out.push_str(&e.tag);
            for (k, v) in &e.attrs {
                out.push(' ');
                out.push_str(k);
                out.push_str("=\"");
                out.push_str(v);
                out.push('"');
            }
            out.push('>');
            for k in &e.children {
                render(k, out);
            }
            if let Some(contents) = &e.template_contents {
                for k in contents {
                    render(k, out);
                }
            }
            out.push_str("</");
            out.push_str(&e.tag);
            out.push('>');
        }
        Node::Text { text, .. } => out.push_str(text),
        Node::Comment { data, .. } => {
            out.push_str("<!--");
            out.push_str(data);
            out.push_str("-->");
        }
        Node::ProcessingInstruction { target, data, .. } => {
            out.push_str("<?");
            out.push_str(target);
            if !data.is_empty() {
                out.push(' ');
                out.push_str(data);
            }
            out.push_str("?>");
        }
        Node::Doctype(d) => {
            out.push_str("<!DOCTYPE ");
            out.push_str(&d.name);
            out.push('>');
        }
    }
}

/// Find the first element with `tag` in document order, descending into
/// template contents as well.
fn find_element<'a>(n: &'a Node, tag: &str) -> Option<&'a Element> {
    match n {
        Node::Element(e) => {
            if e.tag == tag {
                return Some(e);
            }
            e.children
                .iter()
                .chain(e.template_contents.iter().flatten())
                .find_map(|k| find_element(k, tag))
        }
        Node::Document(kids) => kids.iter().find_map(|k| find_element(k, tag)),
        _ => None,
    }
}

fn doc(html: &[u8]) -> String {
    let mut s = String::new();
    render(&parse(html), &mut s);
    s
}

// ---------------------------------------------------------------------------
// Structural comparison
// ---------------------------------------------------------------------------

/// Build an HTML-namespace element with no attributes.
fn el(tag: &str, children: Vec<Node>) -> Node {
    Node::Element(Element {
        tag: TagName::from(tag),
        ns: Namespace::Html,
        attrs: vec![],
        children,
        template_contents: None,
        shadow_root: None,
        loc: None,
    })
}

/// Build an element in `ns` with no attributes.
fn el_ns(tag: &str, ns: Namespace, children: Vec<Node>) -> Node {
    Node::Element(Element {
        tag: TagName::from(tag),
        ns,
        attrs: vec![],
        children,
        template_contents: None,
        shadow_root: None,
        loc: None,
    })
}

/// Build an HTML-namespace element with attributes (source order).
fn el_a(tag: &str, attrs: &[(&str, &str)], children: Vec<Node>) -> Node {
    Node::Element(Element {
        tag: TagName::from(tag),
        ns: Namespace::Html,
        attrs: attrs
            .iter()
            .map(|(k, v)| (AttrName::from(*k), v.to_string()))
            .collect(),
        children,
        template_contents: None,
        shadow_root: None,
        loc: None,
    })
}

/// Build an HTML-namespace element with attributes in a foreign namespace.
fn el_a_ns(tag: &str, ns: Namespace, attrs: &[(&str, &str)], children: Vec<Node>) -> Node {
    Node::Element(Element {
        tag: TagName::from(tag),
        ns,
        attrs: attrs
            .iter()
            .map(|(k, v)| (AttrName::from(*k), v.to_string()))
            .collect(),
        children,
        template_contents: None,
        shadow_root: None,
        loc: None,
    })
}

/// Build a `template` element whose fragment is `contents`.
/// `children` must stay empty: template contents live in
/// `Element::template_contents`, never in `children`.
fn tpl(contents: Vec<Node>) -> Node {
    Node::Element(Element {
        tag: TagName::from("template"),
        ns: Namespace::Html,
        attrs: vec![],
        children: vec![],
        template_contents: Some(contents),
        shadow_root: None,
        loc: None,
    })
}

fn txt(s: &str) -> Node {
    Node::Text {
        text: s.to_string(),
        loc: None,
    }
}

fn cmt(s: &str) -> Node {
    Node::Comment {
        data: s.to_string(),
        loc: None,
    }
}

fn pi(target: &str, data: &str) -> Node {
    Node::ProcessingInstruction {
        target: target.to_string(),
        data: data.to_string(),
        loc: None,
    }
}

/// Expected `<html><head></head><body>...</body></html>` document.
fn page(body_kids: Vec<Node>) -> Node {
    Node::Document(vec![el(
        "html",
        vec![el("head", vec![]), el("body", body_kids)],
    )])
}

/// Structural assertion: compare `actual` against `expected` node by
/// node — node kind, tag name, namespace, attributes (in order), text /
/// comment / PI data, doctype identifiers, and template contents (which
/// must sit in `Element::template_contents`, never `children`). Source
/// locations are ignored. On mismatch, panics with the document path of
/// the first difference.
///
/// String serialization cannot tell a text node from a same-spelled
/// element, drops namespaces, renders void elements as `<tag></tag>`
/// pairs, and leaves text unescaped, so the behavior-critical tests
/// (foreign content, adoption agency, foster parenting, templates) use
/// this instead of [`doc`].
fn assert_tree(actual: &Node, expected: &Node) {
    let mut path = String::from("$");
    if let Some(detail) = diff_node(actual, expected, &mut path) {
        panic!(
            "structural mismatch at {path}: {detail}\nactual:   {actual:#?}\nexpected: {expected:#?}"
        );
    }
}

fn diff_node(a: &Node, e: &Node, path: &mut String) -> Option<String> {
    match (a, e) {
        (Node::Document(ak), Node::Document(ek)) => diff_kids(ak, ek, path),
        (Node::Element(ae), Node::Element(ee)) => {
            if ae.tag != ee.tag {
                return Some(format!("tag {:?} != {:?}", ae.tag, ee.tag));
            }
            if ae.ns != ee.ns {
                return Some(format!("namespace {:?} != {:?}", ae.ns, ee.ns));
            }
            if ae.attrs != ee.attrs {
                return Some(format!("attrs {:?} != {:?}", ae.attrs, ee.attrs));
            }
            match (&ae.template_contents, &ee.template_contents) {
                (None, None) => {}
                (Some(ac), Some(ec)) => {
                    let base = path.len();
                    path.push_str("/template()");
                    let r = diff_kids(ac, ec, path);
                    path.truncate(base);
                    if r.is_some() {
                        return r;
                    }
                }
                _ => {
                    return Some(format!(
                        "template_contents present: {} != {}",
                        ae.template_contents.is_some(),
                        ee.template_contents.is_some()
                    ));
                }
            }
            diff_kids(&ae.children, &ee.children, path)
        }
        (Node::Text { text: at, .. }, Node::Text { text: et, .. }) => {
            if at == et {
                None
            } else {
                Some(format!("text {at:?} != {et:?}"))
            }
        }
        (Node::Comment { data: ad, .. }, Node::Comment { data: ed, .. }) => {
            if ad == ed {
                None
            } else {
                Some(format!("comment {ad:?} != {ed:?}"))
            }
        }
        (
            Node::ProcessingInstruction {
                target: at,
                data: ad,
                ..
            },
            Node::ProcessingInstruction {
                target: et,
                data: ed,
                ..
            },
        ) => {
            if at == et && ad == ed {
                None
            } else {
                Some(format!("pi <?{at} {ad}?> != <?{et} {ed}?>"))
            }
        }
        (Node::Doctype(ad), Node::Doctype(ed)) => {
            if ad == ed {
                None
            } else {
                Some(format!("doctype {ad:?} != {ed:?}"))
            }
        }
        _ => Some("node kind differs".to_string()),
    }
}

fn diff_kids(a: &[Node], e: &[Node], path: &mut String) -> Option<String> {
    if a.len() != e.len() {
        return Some(format!("child count {} != {}", a.len(), e.len()));
    }
    for (i, (ak, ek)) in a.iter().zip(e.iter()).enumerate() {
        let base = path.len();
        path.push('/');
        path.push_str(&kid_label(ek, i));
        let r = diff_node(ak, ek, path);
        path.truncate(base);
        if r.is_some() {
            return r;
        }
    }
    None
}

fn kid_label(n: &Node, i: usize) -> String {
    match n {
        Node::Document(_) => format!("document[{i}]"),
        Node::Element(e) => format!("{}[{i}]", e.tag),
        Node::Text { .. } => format!("text[{i}]"),
        Node::Comment { .. } => format!("comment[{i}]"),
        Node::ProcessingInstruction { .. } => format!("pi[{i}]"),
        Node::Doctype(_) => format!("doctype[{i}]"),
    }
}

// ---------------------------------------------------------------------------
// Document structure: implied html/head/body, doctype, comments, decoding
// ---------------------------------------------------------------------------

#[test]
fn empty_input_gets_implied_structure() {
    assert_eq!(doc(b""), "<html><head></head><body></body></html>");
}

#[test]
fn text_gets_implied_structure() {
    assert_eq!(
        doc(b"hello"),
        "<html><head></head><body>hello</body></html>"
    );
}

#[test]
fn doctype_is_first_child_of_document() {
    assert_eq!(
        doc(b"<!DOCTYPE html><p>hi"),
        "<!DOCTYPE html><html><head></head><body><p>hi</p></body></html>"
    );
}

#[test]
fn comment_before_html_goes_to_document() {
    assert_eq!(
        doc(b"<!--c--><p>x"),
        "<!--c--><html><head></head><body><p>x</p></body></html>"
    );
}

#[test]
fn comment_inside_body() {
    assert_eq!(
        doc(b"<p>a<!--b-->c"),
        "<html><head></head><body><p>a<!--b-->c</p></body></html>"
    );
}

#[test]
fn invalid_utf8_becomes_replacement_char() {
    assert_eq!(
        doc(b"a\xffb"),
        "<html><head></head><body>a\u{FFFD}b</body></html>"
    );
}

#[test]
fn null_character_is_ignored_in_body() {
    assert_eq!(doc(b"a\x00b"), "<html><head></head><body>ab</body></html>");
}

#[test]
fn html_attributes_merge_into_existing() {
    assert_eq!(
        doc(b"<html lang=en><html dir=ltr><p>x"),
        "<html lang=\"en\" dir=\"ltr\"><head></head><body><p>x</p></body></html>"
    );
}

#[test]
fn second_body_merges_attributes() {
    assert_eq!(
        doc(b"<body class=a><body id=b>x"),
        "<html><head></head><body class=\"a\" id=\"b\">x</body></html>"
    );
}

#[test]
fn character_references_decode() {
    assert_eq!(
        doc(b"<p>&lt;&amp;&gt;"),
        "<html><head></head><body><p>&lt;&amp;&gt;</p></body></html>"
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
    );
}

// ---------------------------------------------------------------------------
// Head content and text-state elements (13.2.6.4.1-4.3)
// ---------------------------------------------------------------------------

#[test]
fn title_is_rcdata() {
    assert_eq!(
        doc(b"<title>a<b</title><p>x"),
        "<html><head><title>a<b</title></head><body><p>x</p></body></html>"
    );
}

#[test]
fn script_is_rawtext() {
    assert_eq!(
        doc(b"<script>if(a<b){x}</script><p>y"),
        "<html><head><script>if(a<b){x}</script></head><body><p>y</p></body></html>"
    );
}

#[test]
fn style_is_rawtext() {
    assert_eq!(
        doc(b"<style>a>b{}</style>"),
        "<html><head><style>a>b{}</style></head><body></body></html>"
    );
}

#[test]
fn textarea_strips_single_leading_lf() {
    assert_eq!(
        doc(b"<textarea>\nfoo</textarea>"),
        "<html><head></head><body><textarea>foo</textarea></body></html>"
    );
}

#[test]
fn pre_strips_single_leading_lf() {
    assert_eq!(
        doc(b"<pre>\nfoo</pre>"),
        "<html><head></head><body><pre>foo</pre></body></html>"
    );
}

#[test]
fn pre_keeps_second_lf() {
    assert_eq!(
        doc(b"<pre>\n\nfoo</pre>"),
        "<html><head></head><body><pre>\nfoo</pre></body></html>"
    );
}

#[test]
fn plaintext_swallows_rest_of_input() {
    assert_eq!(
        doc(b"<plaintext><p>x"),
        "<html><head></head><body><plaintext><p>x</plaintext></body></html>"
    );
}

#[test]
fn xmp_is_rawtext() {
    assert_eq!(
        doc(b"<xmp>a<b</xmp>"),
        "<html><head></head><body><xmp>a<b</xmp></body></html>"
    );
}

#[test]
fn iframe_is_rawtext() {
    assert_eq!(
        doc(b"<iframe>a<b</iframe>"),
        "<html><head></head><body><iframe>a<b</iframe></body></html>"
    );
}

#[test]
fn noscript_with_scripting_disabled_keeps_element_open() {
    assert_eq!(
        doc(b"<noscript><p>x</noscript>"),
        "<html><head><noscript></noscript></head><body><p>x</p></body></html>"
    );
}

// ---------------------------------------------------------------------------
// In-body: implied end tags, void elements, misc rules
// ---------------------------------------------------------------------------

#[test]
fn p_implied_close() {
    assert_eq!(
        doc(b"<p>a<p>b"),
        "<html><head></head><body><p>a</p><p>b</p></body></html>"
    );
}

#[test]
fn stray_end_p_creates_and_closes_paragraph() {
    assert_eq!(
        doc(b"<div></p><p>x"),
        "<html><head></head><body><div><p></p><p>x</p></div></body></html>"
    );
}

#[test]
fn li_closes_open_li() {
    assert_eq!(
        doc(b"<ul><li>a<li>b</ul>"),
        "<html><head></head><body><ul><li>a</li><li>b</li></ul></body></html>"
    );
}

#[test]
fn dd_dt_close_each_other() {
    assert_eq!(
        doc(b"<dl><dt>a<dd>b<dt>c</dl>"),
        "<html><head></head><body><dl><dt>a</dt><dd>b</dd><dt>c</dt></dl></body></html>"
    );
}

#[test]
fn headings_close_each_other() {
    assert_eq!(
        doc(b"<h1>a<h2>b</h1>"),
        "<html><head></head><body><h1>a</h1><h2>b</h2></body></html>"
    );
}

#[test]
fn br_end_tag_behaves_as_start_tag() {
    assert_eq!(
        doc(b"<p>a</br>b"),
        "<html><head></head><body><p>a<br></br>b</p></body></html>"
    );
}

#[test]
fn void_elements() {
    assert_eq!(
        doc(b"<p>a<br>b<hr><img src=x>"),
        "<html><head></head><body><p>a<br></br>b</p><hr></hr><img src=\"x\"></img></body></html>"
    );
}

#[test]
fn div_end_tag_closes_open_p() {
    assert_eq!(
        doc(b"<div><p>a</div>"),
        "<html><head></head><body><div><p>a</p></div></body></html>"
    );
}

#[test]
fn p_does_not_close_inside_button_scope() {
    // `button` is in button scope: a `p` start tag closes an open `p`.
    assert_eq!(
        doc(b"<button><p>a<p>b</button>"),
        "<html><head></head><body><button><p>a</p><p>b</p></button></body></html>"
    );
}

#[test]
fn address_div_p_do_not_stop_li_scan() {
    assert_eq!(
        doc(b"<ul><div><li>a</div><li>b</ul>"),
        "<html><head></head><body><ul><div><li>a</li></div><li>b</li></ul></body></html>"
    );
}

#[test]
fn select_options() {
    assert_eq!(
        doc(b"<select><option>a<option>b</select>"),
        "<html><head></head><body><select><option>a</option><option>b</option></select></body></html>"
    );
}

#[test]
fn select_with_optgroup() {
    assert_eq!(
        doc(b"<select><optgroup><option>a</select>"),
        "<html><head></head><body><select><optgroup><option>a</option></optgroup></select></body></html>"
    );
}

#[test]
fn ruby_rt() {
    assert_eq!(
        doc(b"<ruby>x<rt>a<rt>b</ruby>"),
        "<html><head></head><body><ruby>x<rt>a</rt><rt>b</rt></ruby></body></html>"
    );
}

#[test]
fn applet_object_marquee_use_markers() {
    assert_eq!(
        doc(b"<b>1<object>2</object>3</b>"),
        "<html><head></head><body><b>1<object>2</object>3</b></body></html>"
    );
}

#[test]
fn nested_a_runs_adoption_on_start_tag() {
    // Structural: the second `<a>` runs the adoption agency; the two
    // anchors are siblings, each keeping its own `href`.
    assert_tree(
        &parse(b"<a href=a>1<a href=b>2</a>"),
        &page(vec![
            el_a("a", &[("href", "a")], vec![txt("1")]),
            el_a("a", &[("href", "b")], vec![txt("2")]),
        ]),
    );
}

#[test]
fn nobr_start_tag_runs_adoption() {
    assert_tree(
        &parse(b"<nobr>1<nobr>2"),
        &page(vec![el("nobr", vec![txt("1")]), el("nobr", vec![txt("2")])]),
    );
}

#[test]
fn form_element() {
    assert_eq!(
        doc(b"<form action=x><input></form>"),
        "<html><head></head><body><form action=\"x\"><input></input></form></body></html>"
    );
}

// ---------------------------------------------------------------------------
// Adoption agency (13.2.6.4.7)
// ---------------------------------------------------------------------------

#[test]
fn adoption_across_formatting_elements() {
    // <p>1<b>2<i>3</b>4</i>5</p>: the `i` is closed inside `b`, then
    // reconstructed for the text after the adoption.
    assert_tree(
        &parse(b"<p>1<b>2<i>3</b>4</i>5</p>"),
        &page(vec![el(
            "p",
            vec![
                txt("1"),
                el("b", vec![txt("2"), el("i", vec![txt("3")])]),
                el("i", vec![txt("4")]),
                txt("5"),
            ],
        )]),
    );
}

#[test]
fn adoption_with_block_between() {
    // <b>1<p>2</b>3</p>: the `p` becomes the furthest block.
    assert_tree(
        &parse(b"<b>1<p>2</b>3</p>"),
        &page(vec![
            el("b", vec![txt("1")]),
            el("p", vec![el("b", vec![txt("2")]), txt("3")]),
        ]),
    );
}

#[test]
fn adoption_wraps_block_children() {
    assert_tree(
        &parse(b"<b><div>x</b>y</div>"),
        &page(vec![
            el("b", vec![]),
            el("div", vec![el("b", vec![txt("x")]), txt("y")]),
        ]),
    );
}

#[test]
fn adoption_no_furthest_block_pops_to_formatting_element() {
    assert_tree(
        &parse(b"<b>a<i>b</b>c"),
        &page(vec![
            el("b", vec![txt("a"), el("i", vec![txt("b")])]),
            el("i", vec![txt("c")]),
        ]),
    );
}

#[test]
fn stray_formatting_end_tag_with_no_element_is_ignored() {
    // `</i>` with no `i` anywhere is dropped; adjacent text merges.
    assert_tree(&parse(b"<p>a</i>b"), &page(vec![el("p", vec![txt("ab")])]));
}

// ---------------------------------------------------------------------------
// Tables and foster parenting (13.2.6.4.9)
// ---------------------------------------------------------------------------

#[test]
fn table_gets_implied_tbody() {
    assert_eq!(
        doc(b"<table><tr><td>a</td></tr></table>"),
        "<html><head></head><body><table><tbody><tr><td>a</td></tr></tbody></table></body></html>"
    );
}

#[test]
fn table_gets_implied_tbody_tr() {
    assert_eq!(
        doc(b"<table><td>a</td></table>"),
        "<html><head></head><body><table><tbody><tr><td>a</td></tr></tbody></table></body></html>"
    );
}

#[test]
fn table_caption_and_sections() {
    assert_eq!(
        doc(b"<table><caption>c</caption><thead><tr><th>h</th></tr></thead><tbody><tr><td>d</td></tr></tbody></table>"),
        "<html><head></head><body><table><caption>c</caption><thead><tr><th>h</th></tr></thead><tbody><tr><td>d</td></tr></tbody></table></body></html>"
    );
}

#[test]
fn table_colgroup() {
    assert_eq!(
        doc(b"<table><colgroup><col><col></colgroup><tr><td>x</td></tr></table>"),
        "<html><head></head><body><table><colgroup><col></col><col></col></colgroup><tbody><tr><td>x</td></tr></tbody></table></body></html>"
    );
}

#[test]
fn table_cell_closes_open_cell() {
    assert_eq!(
        doc(b"<table><tr><td>a<td>b</tr></table>"),
        "<html><head></head><body><table><tbody><tr><td>a</td><td>b</td></tr></tbody></table></body></html>"
    );
}

#[test]
fn non_whitespace_in_table_is_foster_parented() {
    // The stray `text` is foster-parented before the table element; the
    // table itself gets implied tbody/tr. Structural: placement is a
    // sibling relationship, which strings show only weakly.
    assert_tree(
        &parse(b"<table>text<tr><td>a"),
        &page(vec![
            txt("text"),
            el(
                "table",
                vec![el("tbody", vec![el("tr", vec![el("td", vec![txt("a")])])])],
            ),
        ]),
    );
}

#[test]
fn foster_parenting_with_formatting() {
    // <table><b><tr><td>aaa</td></tr>bbb</table>ccc: the foster-parented `b`
    // is popped by clear-back-to-table-context when <tr> implies <tbody>,
    // so `bbb` reconstructs a fresh `b`; after the table the surviving
    // `b` (still on the stack below the table) wraps `ccc`.
    assert_tree(
        &parse(b"<table><b><tr><td>aaa</td></tr>bbb</table>ccc"),
        &page(vec![
            el("b", vec![]),
            el("b", vec![txt("bbb")]),
            el(
                "table",
                vec![el(
                    "tbody",
                    vec![el("tr", vec![el("td", vec![txt("aaa")])])],
                )],
            ),
            el("b", vec![txt("ccc")]),
        ]),
    );
}

#[test]
fn whitespace_in_table_is_kept() {
    // Whitespace (unlike other text) stays inside the table as a text
    // node child, before the implied tbody.
    assert_tree(
        &parse(b"<table> <tr><td>a</td></tr></table>"),
        &page(vec![el(
            "table",
            vec![
                txt(" "),
                el("tbody", vec![el("tr", vec![el("td", vec![txt("a")])])]),
            ],
        )]),
    );
}

#[test]
fn form_inside_table_is_inserted_then_popped() {
    // The form element is inserted then popped; the `input` is
    // foster-parented before the table. `input` is a void element with
    // no children (strings render it as `<input></input>` either way).
    assert_tree(
        &parse(b"<table><form><input></form></table>"),
        &page(vec![
            el("input", vec![]),
            el("table", vec![el("form", vec![])]),
        ]),
    );
}

#[test]
fn select_inside_table() {
    // The select is foster-parented before the table; the option's text
    // merges into one text node.
    assert_tree(
        &parse(b"<table><select><option>a</select></table>"),
        &page(vec![
            el("select", vec![el("option", vec![txt("a")])]),
            el("table", vec![]),
        ]),
    );
}

// ---------------------------------------------------------------------------
// Templates (13.2.6.4.11)
// ---------------------------------------------------------------------------

#[test]
fn template_contents_are_separate() {
    // Template contents live in `Element::template_contents`, not in
    // `children` (WHATWG 13.2.6.4.11 "in template" insertion mode); other
    // elements have `template_contents == None`. The template appears
    // before any body content, so it lands in `head`.
    assert_tree(
        &parse(b"<template><tr><td>x</td></tr></template>"),
        &Node::Document(vec![el(
            "html",
            vec![
                el(
                    "head",
                    vec![tpl(vec![el("tr", vec![el("td", vec![txt("x")])])])],
                ),
                el("body", vec![]),
            ],
        )]),
    );
}

#[test]
fn template_in_body() {
    assert_tree(
        &parse(b"<div><template><p>x</template></div>"),
        &page(vec![el("div", vec![tpl(vec![el("p", vec![txt("x")])])])]),
    );
}

// ---------------------------------------------------------------------------
// Frameset and after-body modes
// ---------------------------------------------------------------------------

#[test]
fn frameset_replaces_body() {
    assert_eq!(
        doc(b"<frameset><frame src=a></frameset>"),
        "<html><head></head><frameset><frame src=\"a\"></frame></frameset></html>"
    );
}

#[test]
fn text_after_body_goes_to_html() {
    assert_eq!(
        doc(b"<p>x</body>y"),
        "<html><head></head><body><p>xy</p></body></html>"
    );
}

#[test]
fn comment_after_html_goes_to_document() {
    assert_eq!(
        doc(b"<p>x</body></html><!--tail-->"),
        "<html><head></head><body><p>x</p></body></html><!--tail-->"
    );
}

// ---------------------------------------------------------------------------
// Foreign content (13.2.6.5)
// ---------------------------------------------------------------------------

#[test]
fn svg_element() {
    // Structural: `circle` is a real element child of `svg` in the SVG
    // namespace (string rendering cannot show the namespace, and prints
    // the void element as a `<circle></circle>` pair either way).
    assert_tree(
        &parse(b"<p><svg><circle /></svg>"),
        &page(vec![el(
            "p",
            vec![el_ns(
                "svg",
                Namespace::Svg,
                vec![el_ns("circle", Namespace::Svg, vec![])],
            )],
        )]),
    );
}

#[test]
fn svg_tag_name_adjustment() {
    assert_tree(
        &parse(b"<svg><foreignobject></foreignobject></svg>"),
        &page(vec![el_ns(
            "svg",
            Namespace::Svg,
            vec![el_ns("foreignObject", Namespace::Svg, vec![])],
        )]),
    );
}

#[test]
fn svg_attribute_adjustment() {
    assert_tree(
        &parse(b"<svg viewbox=\"0 0 1 1\"></svg>"),
        &page(vec![el_a_ns(
            "svg",
            Namespace::Svg,
            &[("viewBox", "0 0 1 1")],
            vec![],
        )]),
    );
}

#[test]
fn mathml_element() {
    assert_tree(
        &parse(b"<math><mi>x</mi></math>"),
        &page(vec![el_ns(
            "math",
            Namespace::MathMl,
            vec![el_ns("mi", Namespace::MathMl, vec![txt("x")])],
        )]),
    );
}

#[test]
fn desc_is_html_integration_point() {
    // `desc` is an HTML integration point, so `<p>` is processed as HTML
    // inside it rather than breaking out of the foreign content. The
    // `desc` itself stays SVG; the `p` is HTML-namespace.
    assert_tree(
        &parse(b"<svg><desc><p>x</p></desc></svg>"),
        &page(vec![el_ns(
            "svg",
            Namespace::Svg,
            vec![el_ns("desc", Namespace::Svg, vec![el("p", vec![txt("x")])])],
        )]),
    );
}

#[test]
fn html_integration_point_annotation_xml() {
    assert_tree(
        &parse(
            b"<math><annotation-xml encoding=\"text/html\"><div>x</div></annotation-xml></math>",
        ),
        &page(vec![el_ns(
            "math",
            Namespace::MathMl,
            vec![el_a_ns(
                "annotation-xml",
                Namespace::MathMl,
                &[("encoding", "text/html")],
                vec![el("div", vec![txt("x")])],
            )],
        )]),
    );
}

#[test]
fn cdata_in_foreign_content() {
    // The CDATA section yields a *text* node whose data is the literal
    // `<p>` string. String rendering prints it unescaped, so it looks
    // like a `<p>` element; only the structural check pins the node
    // kind.
    assert_tree(
        &parse(b"<svg><![CDATA[<p>]]></svg>"),
        &page(vec![el_ns("svg", Namespace::Svg, vec![txt("<p>")])]),
    );
}

#[test]
fn cdata_in_html_content_becomes_comment() {
    // A CDATA section outside foreign content is a parse error and
    // becomes a bogus comment with the `[CDATA[` prefix.
    assert_tree(
        &parse(b"<div><![CDATA[x]]></div>"),
        &page(vec![el("div", vec![cmt("[CDATA[x]]")])]),
    );
}

// ---------------------------------------------------------------------------
// Namespaces (M13): the public Element carries its namespace
// ---------------------------------------------------------------------------

#[test]
fn namespace_on_svg_and_math_elements() {
    // Full structural check: namespaces ride on every element, which
    // string rendering cannot show.
    assert_tree(
        &parse(b"<div><svg><circle/></svg><math><mi>x</mi></math></div>"),
        &page(vec![el(
            "div",
            vec![
                el_ns(
                    "svg",
                    Namespace::Svg,
                    vec![el_ns("circle", Namespace::Svg, vec![])],
                ),
                el_ns(
                    "math",
                    Namespace::MathMl,
                    vec![el_ns("mi", Namespace::MathMl, vec![txt("x")])],
                ),
            ],
        )]),
    );
}

// ---------------------------------------------------------------------------
// Doctype identifiers and quirks mode (M14a, M14b)
// ---------------------------------------------------------------------------

#[test]
fn doctype_carries_public_and_system_ids() {
    let tree = parse(
        b"<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01//EN\" \"http://www.w3.org/TR/html4/strict.dtd\"><p>x",
    );
    let Node::Document(kids) = &tree else {
        panic!("expected document");
    };
    let Node::Doctype(d) = &kids[0] else {
        panic!("expected doctype first, got {:?}", kids[0]);
    };
    assert_eq!(d.name, "html");
    assert_eq!(d.public_id.as_deref(), Some("-//W3C//DTD HTML 4.01//EN"));
    assert_eq!(
        d.system_id.as_deref(),
        Some("http://www.w3.org/TR/html4/strict.dtd")
    );
    assert!(!d.force_quirks);
}

#[test]
fn doctype_without_identifiers() {
    let tree = parse(b"<!DOCTYPE html><p>x");
    let Node::Document(kids) = &tree else {
        panic!("expected document");
    };
    let Node::Doctype(d) = &kids[0] else {
        panic!("expected doctype first");
    };
    assert_eq!(d.name, "html");
    assert_eq!(d.public_id, None);
    assert_eq!(d.system_id, None);
    assert!(!d.force_quirks);
}

#[test]
fn bogus_doctype_sets_force_quirks() {
    // `PUBLIC` with no identifier is a bogus doctype: force-quirks set.
    let tree = parse(b"<!DOCTYPE html PUBLIC><p>x");
    let Node::Document(kids) = &tree else {
        panic!("expected document");
    };
    let Node::Doctype(d) = &kids[0] else {
        panic!("expected doctype first");
    };
    assert!(d.force_quirks);
    let (_, quirks) = parse_with_quirks(b"<!DOCTYPE html PUBLIC><p>x");
    assert_eq!(quirks, QuirksMode::Quirks);
}

#[test]
fn quirks_mode_no_quirks_for_html_doctype() {
    let (_, quirks) = parse_with_quirks(b"<!DOCTYPE html><p>x");
    assert_eq!(quirks, QuirksMode::NoQuirks);
}

#[test]
fn quirks_mode_quirks_without_doctype() {
    let (_, quirks) = parse_with_quirks(b"<p>x");
    assert_eq!(quirks, QuirksMode::Quirks);
}

#[test]
fn quirks_mode_limited_quirks_for_xhtml_transitional() {
    let (_, quirks) = parse_with_quirks(
        b"<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\"><p>x",
    );
    assert_eq!(quirks, QuirksMode::LimitedQuirks);
}

#[test]
fn quirks_mode_full_quirks_for_old_public_id() {
    let (_, quirks) =
        parse_with_quirks(b"<!DOCTYPE html PUBLIC \"-//IETF//DTD HTML 2.0//EN\"><p>x");
    assert_eq!(quirks, QuirksMode::Quirks);
}

#[test]
fn quirks_mode_full_quirks_for_html40_transitional_id() {
    // WHATWG 13.2.6.4.1 lists this identifier with a single `-` before
    // `/W3C` (`-/W3C/...`, unlike the `-//W3C/...` identifiers); the
    // double-slash typo used to miss it.
    let (_, quirks) =
        parse_with_quirks(b"<!DOCTYPE html PUBLIC \"-/W3C/DTD HTML 4.0 Transitional/EN\"><p>x");
    assert_eq!(quirks, QuirksMode::Quirks);
}

// ---------------------------------------------------------------------------
// Source locations (M14d)
// ---------------------------------------------------------------------------

fn loc(n: &Node) -> Option<SourceLoc> {
    match n {
        Node::Element(e) => e.loc,
        Node::Text { loc, .. } => *loc,
        Node::Comment { loc, .. } => *loc,
        _ => None,
    }
}

#[test]
fn source_locations_on_small_doc() {
    // <!--c--><div>\n<p>hi</p>\n</div>
    // line 1: `<!--c-->` cols 1-8, `<div>` cols 9-13, `\n` col 14
    // line 2: `<p>` cols 1-3, `hi` cols 4-5, `</p>` cols 6-9, `\n` col 10
    // line 3: `</div>` cols 1-6
    let tree = parse(b"<!--c--><div>\n<p>hi</p>\n</div>");
    let Node::Document(kids) = &tree else {
        panic!("expected document");
    };
    assert_eq!(
        loc(&kids[0]),
        Some(SourceLoc { line: 1, col: 1 }),
        "comment starts at 1:1"
    );
    let html = find_element(&tree, "html").expect("html");
    assert_eq!(html.loc, None, "implied html has no location");
    let head = find_element(&tree, "head").expect("head");
    assert_eq!(head.loc, None, "implied head has no location");
    let div = find_element(&tree, "div").expect("div");
    assert_eq!(div.loc, Some(SourceLoc { line: 1, col: 9 }));
    assert_eq!(div.children.len(), 3);
    let Node::Text { text, loc } = &div.children[0] else {
        panic!("expected text");
    };
    assert_eq!(text, "\n");
    assert_eq!(*loc, Some(SourceLoc { line: 1, col: 14 }));
    let p = find_element(&tree, "p").expect("p");
    assert_eq!(p.loc, Some(SourceLoc { line: 2, col: 1 }));
    let Node::Text { text, loc } = &p.children[0] else {
        panic!("expected text");
    };
    assert_eq!(text, "hi");
    assert_eq!(*loc, Some(SourceLoc { line: 2, col: 4 }));
    let Node::Text { text, loc } = &div.children[2] else {
        panic!("expected text");
    };
    assert_eq!(text, "\n");
    assert_eq!(*loc, Some(SourceLoc { line: 2, col: 10 }));
}

#[test]
fn source_location_of_merged_text_is_first_char() {
    // Adjacent character tokens merge; the node keeps the first char's loc.
    let tree = parse(b"<p>ab</p>");
    let p = find_element(&tree, "p").expect("p");
    assert_eq!(p.loc, Some(SourceLoc { line: 1, col: 1 }));
    let Node::Text { text, loc } = &p.children[0] else {
        panic!("expected text");
    };
    assert_eq!(text, "ab");
    assert_eq!(*loc, Some(SourceLoc { line: 1, col: 4 }));
}

#[test]
fn source_location_of_table_whitespace_is_first_char() {
    // `<table> <tr>...`: the whitespace run is located at the space itself
    // (1:8), not at the `<tr>` token that ends the "in table text" run.
    let tree = parse(b"<table> <tr><td>a</td></tr></table>");
    let table = find_element(&tree, "table").expect("table");
    let Node::Text { text, loc } = &table.children[0] else {
        panic!("expected whitespace text, got {:?}", table.children[0]);
    };
    assert_eq!(text, " ");
    assert_eq!(*loc, Some(SourceLoc { line: 1, col: 8 }));
}

#[test]
fn source_location_inside_template_contents() {
    // Locations are attached inside template fragments too.
    let tree = parse(b"<template><p>x</p></template>");
    let template = find_element(&tree, "template").expect("template");
    assert_eq!(template.loc, Some(SourceLoc { line: 1, col: 1 }));
    let contents = template.template_contents.as_ref().expect("contents");
    let p = find_element(&contents[0], "p").expect("p");
    assert_eq!(p.loc, Some(SourceLoc { line: 1, col: 11 }));
}

#[test]
fn doctype_public_id_multibyte_at_prefix_cut_no_panic() {
    // `<!DOCTYPE html PUBLIC "` + 42 x's + `é` + `">`: the old
    // `starts_with_ci` sliced `haystack[..43]` inside the `é` (the 43-byte
    // prefix `-//SQ//DTD HTML 2.0 HoTMetaL + extensions//`) and panicked.
    // Must parse without panicking.
    let mut input = b"<!DOCTYPE html PUBLIC \"".to_vec();
    input.extend(std::iter::repeat_n(b'x', 42));
    input.extend_from_slice("é".as_bytes());
    input.extend_from_slice(b"\">");
    let (node, quirks) = parse_with_quirks(&input);
    assert!(matches!(node, Node::Document(_)));
    // The public id matches no quirks/limited-quirks prefix, so this is
    // no-quirks; the point of the test is that it gets here at all.
    assert_eq!(quirks, QuirksMode::NoQuirks);
}

#[test]
fn deeply_nested_divs_convert_without_stack_overflow() {
    // 100k nested divs: the old recursive arena-to-Node conversion
    // aborted with a stack overflow. The iterative conversion must
    // complete and preserve nesting depth.
    const DEPTH: usize = 100_000;
    let mut input = Vec::with_capacity(DEPTH * 11);
    for _ in 0..DEPTH {
        input.extend_from_slice(b"<div>");
    }
    for _ in 0..DEPTH {
        input.extend_from_slice(b"</div>");
    }
    let tree = parse(&input);
    // The parser implies html/head/body around the divs; walk down
    // document -> html -> body -> first div iteratively.
    let doc_kids = match &tree {
        Node::Document(kids) => kids,
        _ => panic!("expected document"),
    };
    let html = doc_kids
        .iter()
        .find_map(|n| match n {
            Node::Element(e) if e.tag == "html" => Some(e),
            _ => None,
        })
        .expect("html element");
    let body = html
        .children
        .iter()
        .find_map(|n| match n {
            Node::Element(e) if e.tag == "body" => Some(e),
            _ => None,
        })
        .expect("body element");
    let mut node = body.children.first().expect("first div");
    let mut depth = 0;
    loop {
        let e = match node {
            Node::Element(e) => e,
            _ => panic!("expected div at depth {depth}"),
        };
        assert_eq!(e.tag, "div");
        depth += 1;
        if e.children.is_empty() {
            break;
        }
        assert_eq!(e.children.len(), 1);
        node = &e.children[0];
    }
    assert_eq!(depth, DEPTH);
}

#[test]
fn nested_divs_parse_scales_linearly() {
    // The old O(depth)-per-insert cycle guard plus the O(depth) "p in
    // button scope" walk made 16k nested divs take ~2.4s in release
    // (quadratic). With both O(1), 16k nested divs must finish well
    // under that.
    use std::time::{Duration, Instant};
    const DEPTH: usize = 16_000;
    let mut input = Vec::with_capacity(DEPTH * 11);
    for _ in 0..DEPTH {
        input.extend_from_slice(b"<div>");
    }
    let start = Instant::now();
    let tree = parse(&input);
    let elapsed = start.elapsed();
    // Sanity: the tree really is 16k deep (unclosed divs nest under the
    // implied html/body).
    let doc_kids = match &tree {
        Node::Document(kids) => kids,
        _ => panic!("expected document"),
    };
    let html = doc_kids
        .iter()
        .find_map(|n| match n {
            Node::Element(e) if e.tag == "html" => Some(e),
            _ => None,
        })
        .expect("html element");
    let body = html
        .children
        .iter()
        .find_map(|n| match n {
            Node::Element(e) if e.tag == "body" => Some(e),
            _ => None,
        })
        .expect("body element");
    let mut node = body.children.first().expect("first div");
    let mut depth = 0;
    loop {
        let e = match node {
            Node::Element(e) => e,
            _ => panic!("expected div at depth {depth}"),
        };
        assert_eq!(e.tag, "div");
        depth += 1;
        if e.children.is_empty() {
            break;
        }
        node = &e.children[0];
    }
    assert_eq!(depth, DEPTH);
    // Generous debug bound (debug is ~50x slower than release and the
    // machine is shared), stricter release bound. Either catches a
    // reintroduced quadratic, which would take minutes at this depth.
    #[cfg(debug_assertions)]
    let bound = Duration::from_secs(30);
    #[cfg(not(debug_assertions))]
    let bound = Duration::from_secs(5);
    assert!(
        elapsed < bound,
        "16k nested divs took {elapsed:?}, expected under {bound:?}"
    );
}

// ---------------------------------------------------------------------------
// M5-M12 regressions
// ---------------------------------------------------------------------------

#[test]
fn body_noscript_scripting_disabled_inserts_element() {
    // M5: with scripting disabled, a body-level noscript is an ordinary
    // element handled by the generic in-body rules.
    assert_eq!(
        doc(b"<body><noscript><p>x</p></noscript><div>y</div>"),
        "<html><head></head><body><noscript><p>x</p></noscript><div>y</div></body></html>"
    );
}

#[test]
fn template_table_contents_stay_in_template() {
    // M6: the table structure lands in template_contents; the following
    // paragraph is a body sibling of the template.
    assert_tree(
        &parse(b"<body><template><table><tr><td>x</td></tr></table></template><p>y</p>"),
        &page(vec![
            tpl(vec![el(
                "table",
                vec![el("tbody", vec![el("tr", vec![el("td", vec![txt("x")])])])],
            )]),
            el("p", vec![txt("y")]),
        ]),
    );
}

#[test]
fn input_pops_open_select() {
    // M7: an input start tag pops an open select before insertion.
    assert_eq!(
        doc(b"<select><input type=text></select>"),
        "<html><head></head><body><select></select><input type=\"text\"></input></body></html>"
    );
}

#[test]
fn hr_generates_implied_end_tags_in_select() {
    // M8: hr inside select generates implied end tags first, closing the
    // open option.
    assert_eq!(
        doc(b"<select><option><hr></select>"),
        "<html><head></head><body><select><option></option><hr></hr></select></body></html>"
    );
}

#[test]
fn select_scope_stops_at_mathml_boundary() {
    // M9: MathML mi/mo/mn/ms/mtext bound the default scope, so the hr
    // select-scope check does not see past mi and no implied end tags
    // are generated.
    assert_eq!(
        doc(b"<select><math><mi><hr></mi></math></select>"),
        "<html><head></head><body><select><math><mi><hr></hr></mi></math></select></body></html>"
    );
}

#[test]
fn p_scope_stops_at_mathml_boundary() {
    // M9: the div "close a p" check uses button scope, which is also
    // bounded by mi.
    assert_eq!(
        doc(b"<p><math><mi><div>x</div></mi></math>"),
        "<html><head></head><body><p><math><mi><div>x</div></mi></math></p></body></html>"
    );
}

#[test]
fn adoption_agency_inserts_into_template_contents() {
    // M11: the adoption agency's adjusted insertion location keeps the
    // template-contents flag, so the adopted node lands in
    // template_contents rather than the template's regular children.
    // The template sits in `head` (it appears before any body content).
    assert_tree(
        &parse(b"<template><b><div></b>x</div></template>"),
        &Node::Document(vec![el(
            "html",
            vec![
                el(
                    "head",
                    vec![tpl(vec![
                        el("b", vec![]),
                        el("div", vec![el("b", vec![]), txt("x")]),
                    ])],
                ),
                el("body", vec![]),
            ],
        )]),
    );
}

#[test]
fn rcdata_inappropriate_end_tag_reemits_name() {
    // M10 (corrected): WHATWG 13.2.5.11 RCDATA end tag name state,
    // "Anything else" emits `<`, `/`, then the temporary buffer's
    // characters in order.
    assert_eq!(
        doc(b"<title>a</div>b</title>"),
        "<html><head><title>a</div>b</title></head><body></body></html>"
    );
    assert_eq!(
        doc(b"<title>a</div >b</title>"),
        "<html><head><title>a</div >b</title></head><body></body></html>"
    );
    // The temporary buffer keeps the source case, not the lowercased
    // tag-name character.
    assert_eq!(
        doc(b"<title>a</DIV>b</title>"),
        "<html><head><title>a</DIV>b</title></head><body></body></html>"
    );
}

#[test]
fn pi_in_body() {
    // M12: a processing instruction inserts at the adjusted location.
    assert_eq!(
        doc(b"<div><?pi data?></div>"),
        "<html><head></head><body><div><?pi data?></div></body></html>"
    );
    // Empty data omits the separating space.
    assert_eq!(
        doc(b"<div><?a?><?b c d?></div>"),
        "<html><head></head><body><div><?a?><?b c d?></div></body></html>"
    );
}

#[test]
fn pi_before_html_goes_to_document() {
    // M12: initial/before-html PIs are document children.
    assert_eq!(
        doc(b"<?pi?><div>x</div>"),
        "<?pi?><html><head></head><body><div>x</div></body></html>"
    );
}

#[test]
fn pi_in_table_stays_in_table() {
    // M12: like comments, PIs in table mode go to the current node,
    // not through foster parenting.
    assert_eq!(
        doc(b"<table><?pi?><tr><td>x</td></tr></table>"),
        "<html><head></head><body><table><?pi?><tbody><tr><td>x</td></tr></tbody></table></body></html>"
    );
}

#[test]
fn pi_in_template_contents() {
    // M12: PIs in template mode route through in-body, landing in
    // template_contents.
    assert_eq!(
        doc(b"<template><?pi data?></template>"),
        "<html><head><template><?pi data?></template></head><body></body></html>"
    );
}

#[test]
fn pi_in_foreign_content() {
    // M12: PIs in foreign content insert at the adjusted location.
    // Structural: pins the PI node kind (strings print it inline) and
    // the SVG namespaces.
    assert_tree(
        &parse(b"<svg><?pi data?><circle/></svg>"),
        &page(vec![el_ns(
            "svg",
            Namespace::Svg,
            vec![pi("pi", "data"), el_ns("circle", Namespace::Svg, vec![])],
        )]),
    );
}

#[test]
fn pi_disallowed_target_degrades_to_comment() {
    // M12: xml/xml-stylesheet targets become bogus comments.
    assert_eq!(
        doc(b"<?xml-stylesheet href=\"x\"?>"),
        "<!--?xml-stylesheet href=\"x\"?--><html><head></head><body></body></html>"
    );
}

#[test]
fn pi_source_location() {
    // M12: PIs carry the location of their `<?`.
    let tree = parse(b"<div><?pi data?></div>");
    let div = find_element(&tree, "div").expect("div");
    let Node::ProcessingInstruction { target, data, loc } = &div.children[0] else {
        panic!("expected processing instruction");
    };
    assert_eq!(target, "pi");
    assert_eq!(data, "data");
    assert_eq!(*loc, Some(SourceLoc { line: 1, col: 6 }));
}

#[test]
fn nested_template_foster_parenting() {
    // Foster-parented text lands before the table inside the innermost
    // template contents, not outside the templates.
    let tree = parse(b"<template><template><table>Foo");
    let mut out = String::new();
    render(&tree, &mut out);
    assert_eq!(
        out,
        "<html><head><template><template>Foo<table></table></template></template></head><body></body></html>"
    );
}

#[test]
fn solidcolor_stays_lowercase() {
    // SVG tag-name adjustment: solidcolor is not mapped (SVG 1.2 only).
    let tree = parse(b"<svg><solidcolor></solidcolor></svg>");
    let svg = find_element(&tree, "svg").expect("svg");
    let Node::Element(child) = &svg.children[0] else {
        panic!("expected element");
    };
    assert_eq!(child.tag, "solidcolor");
}

#[test]
fn selectedcontent_clones_first_option() {
    // No selected attribute: first option's children are cloned.
    let tree = parse(b"<select><button><selectedcontent></selectedcontent></button><option><b>a</b></option></select>");
    let sc = find_element(&tree, "selectedcontent").expect("selectedcontent");
    let mut out = String::new();
    for k in &sc.children {
        render(k, &mut out);
    }
    assert_eq!(out, "<b>a</b>");
}

#[test]
fn selectedcontent_uses_explicitly_selected_option() {
    // The option with a selected attribute wins over the first option.
    let tree = parse(b"<select><button><selectedcontent></selectedcontent></button><option>a</option><option selected>b</option></select>");
    let sc = find_element(&tree, "selectedcontent").expect("selectedcontent");
    let mut out = String::new();
    for k in &sc.children {
        render(k, &mut out);
    }
    assert_eq!(out, "b");
}

/// Performance regression test: foster-parented insertion must scale
/// near-linearly, not quadratically.
///
/// Every `<div>` in `"<table><div>".repeat(n)` is foster-parented before
/// its table, landing in one ever-growing child list. The insertion used
/// to scan that list front-to-back for the reference table on every
/// insert (O(n) each, O(n^2) total: n=20000 took ~13s in release). The
/// reference table is the open table, which sits at the end of the list,
/// so the scan now runs back-to-front (O(1) amortized per insert).
///
/// The test asserts a scaling ratio rather than a wall-clock bound: with
/// the quadratic bug, doubling n quadruples time (ratio ~4); linear
/// scaling gives ~2. The 3x bound leaves headroom for allocator and
/// scheduler noise on loaded CI boxes, and timing robustness comes from
/// representative-size warmup, interleaved min-of-7 samples (noise only
/// ever inflates wall-clock time, so the minimum is the best estimator,
/// and interleaving means a load spike hits both sizes alike), plus a
/// release-only absolute backstop. A small-n case pins the tree shape so
/// the test cannot pass vacuously (e.g. by bailing out early).
#[test]
fn foster_parenting_insertion_scales_linearly() {
    // Pin the tree shape: each div foster-parents before its table.
    let mut out = String::new();
    render(&parse(b"<table><div><table><div>"), &mut out);
    assert_eq!(
        out,
        "<html><head></head><body><div></div><table></table>\
         <div></div><table></table></body></html>"
    );

    fn parse_ms(n: usize) -> f64 {
        let html = "<table><div>".repeat(n);
        let start = std::time::Instant::now();
        let node = parse(html.as_bytes());
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(node);
        ms
    }

    // Warmup at representative sizes: one throwaway full-size run each
    // warms the page cache and allocator, so the measured samples
    // compare steady-state performance rather than cold-start costs.
    parse_ms(10_000);
    parse_ms(20_000);
    // Interleaved samples: a scheduling spike during the test lands on
    // both sizes alike, so it cannot bias the ratio. Taking the minimum
    // of each size's samples then rejects any remaining upward outliers.
    let (mut a, mut b) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..7 {
        a = a.min(parse_ms(10_000));
        b = b.min(parse_ms(20_000));
    }
    let ratio = b / a;
    assert!(
        ratio < 3.0,
        "foster-parented insertion looks quadratic: 10k={a:.0}ms 20k={b:.0}ms ratio={ratio:.2}"
    );
    // Absolute backstop, release only: debug builds are an order of
    // magnitude slower and noisier, so only the ratio is asserted there.
    if !cfg!(debug_assertions) {
        assert!(
            b < 5_000.0,
            "foster-parented insertion too slow in release: 20k={b:.0}ms"
        );
    }
}

// ---------------------------------------------------------------------------
// Script execution hook (WHATWG HTML 13.2.6: a parser-inserted classic
// script executes as soon as the parser resumes, with the insertion point
// just before the next input character).
// ---------------------------------------------------------------------------

use bowser_html::{ScriptElement, ScriptExecutor, ScriptHost, parse_with_scripting_and_executor};

/// Canned script engine for hook tests: records executions, optionally
/// writes a fixed payload and/or mutates one element by id.
#[derive(Default)]
struct TestExecutor {
    /// Script texts executed, in order.
    ran: Vec<String>,
    /// Payload for `document.write`, when set.
    write: Option<String>,
    /// `(id, attr, value)` mutation via `getElementById`, when set.
    mutate: Option<(String, String, String)>,
    /// `getElementById` probes: ids to look up while running.
    probe: Vec<String>,
    /// Whether each probe found its element, in order.
    probe_results: Vec<bool>,
}

impl ScriptExecutor for TestExecutor {
    fn execute_script(
        &mut self,
        host: &mut ScriptHost<'_, '_, '_>,
        _el: ScriptElement,
        _attrs: &[(String, String)],
        script: &str,
    ) {
        self.ran.push(script.to_string());
        for id in &self.probe {
            self.probe_results
                .push(host.get_element_by_id(id).is_some());
        }
        if let Some(text) = &self.write {
            host.document_write(text);
        }
        if let Some((id, attr, value)) = &self.mutate
            && let Some(el) = host.get_element_by_id(id)
        {
            host.set_attribute(el, attr, value);
        }
    }
}

fn doc_with_executor(html: &[u8], scripting: bool, ex: &mut impl ScriptExecutor) -> String {
    let mut s = String::new();
    render(
        &parse_with_scripting_and_executor(html, scripting, ex),
        &mut s,
    );
    s
}

#[test]
fn script_executes_at_end_tag_when_scripting_enabled() {
    let mut ex = TestExecutor::default();
    let html = doc_with_executor(b"<p><script>hello()</script>", true, &mut ex);
    assert_eq!(ex.ran, vec!["hello()".to_string()]);
    assert!(html.contains("<script>hello()</script>"), "{html}");
}

#[test]
fn script_does_not_execute_when_scripting_disabled() {
    let mut ex = TestExecutor::default();
    let html = doc_with_executor(b"<p><script>hello()</script>", false, &mut ex);
    assert!(ex.ran.is_empty());
    assert!(html.contains("<script>hello()</script>"), "{html}");
}

#[test]
fn script_executes_before_following_tokens_are_parsed() {
    // The engine runs at `</script>`, before `<div id=x>` is tokenized:
    // the probe must miss, proving execution is not deferred to EOF.
    let mut ex = TestExecutor {
        probe: vec!["x".to_string()],
        ..Default::default()
    };
    doc_with_executor(b"<script>q()</script><div id=x>", true, &mut ex);
    assert_eq!(ex.ran, vec!["q()".to_string()]);
    assert_eq!(ex.probe_results, vec![false]);
}

#[test]
fn script_end_tag_in_title_does_not_execute() {
    // `</script>` while the current node is `title`: not a script
    // element, so no execution (and the pre-existing pop behavior is
    // unchanged).
    let mut ex = TestExecutor::default();
    doc_with_executor(b"<title>a</script>b</title>", true, &mut ex);
    assert!(ex.ran.is_empty());
}

#[test]
fn document_write_inserts_before_following_input() {
    let mut ex = TestExecutor {
        write: Some("2".to_string()),
        ..Default::default()
    };
    let html = doc_with_executor(b"1<script>w()</script>3", true, &mut ex);
    // "2" is inserted at the insertion point, ahead of "3", and the two
    // merge into a single text node.
    assert!(
        html.contains("<body>1<script>w()</script>23</body>"),
        "{html}"
    );
}

#[test]
fn document_write_markup_parses_as_markup() {
    let mut ex = TestExecutor {
        write: Some("<b>hi</b>".to_string()),
        ..Default::default()
    };
    let html = doc_with_executor(b"<script>w()</script>", true, &mut ex);
    assert!(html.contains("<b>hi</b>"), "{html}");
}

#[test]
fn document_write_nested_script_executes_in_order() {
    struct Nested {
        ran: Vec<String>,
    }
    impl ScriptExecutor for Nested {
        fn execute_script(
            &mut self,
            host: &mut ScriptHost<'_, '_, '_>,
            _el: ScriptElement,
            _attrs: &[(String, String)],
            script: &str,
        ) {
            self.ran.push(script.to_string());
            if script == "outer()" {
                host.document_write("<script>inner()</script>");
            }
        }
    }
    let mut ex = Nested { ran: Vec::new() };
    let html = doc_with_executor(b"<script>outer()</script>done", true, &mut ex);
    assert_eq!(ex.ran, vec!["outer()".to_string(), "inner()".to_string()]);
    // Both scripts ran while still in `head`; the written inner script
    // parsed as markup before the original following input.
    assert!(
        html.contains("<head><script>outer()</script><script>inner()</script></head><body>done"),
        "{html}"
    );
}

#[test]
fn document_write_normalizes_crlf() {
    let mut ex = TestExecutor {
        write: Some("a\r\nb\rc".to_string()),
        ..Default::default()
    };
    let html = doc_with_executor(b"<script>w()</script>", true, &mut ex);
    assert!(html.contains("a\nb\nc"), "{html:?}");
}

#[test]
fn script_dom_mutation_is_visible_to_later_parse() {
    // html5lib scripted/adoption01.dat shape: the live element is
    // mutated, while formatting-element reconstruction keeps using the
    // original start-tag token's attributes.
    let mut ex = TestExecutor {
        mutate: Some(("a".to_string(), "id".to_string(), "b".to_string())),
        ..Default::default()
    };
    let html = doc_with_executor(
        br#"<p><b id="a"><script>m()</script></p>TEXT</b>"#,
        true,
        &mut ex,
    );
    assert!(
        html.contains(r#"<b id="b"><script>m()</script></b>"#),
        "{html}"
    );
    assert!(html.contains(r#"<b id="a">TEXT</b>"#), "{html}");
}

#[test]
fn get_elements_by_tag_name_is_document_order_and_set_attribute_replaces() {
    struct Idx;
    impl ScriptExecutor for Idx {
        fn execute_script(
            &mut self,
            host: &mut ScriptHost<'_, '_, '_>,
            _el: ScriptElement,
            _attrs: &[(String, String)],
            _script: &str,
        ) {
            let els = host.get_elements_by_tag_name("i");
            assert_eq!(els.len(), 2);
            // Replace on the second, append on the first: document order.
            host.set_attribute(els[1], "class", "second");
            host.set_attribute(els[0], "class", "first");
        }
    }
    let mut ex = Idx;
    let html = doc_with_executor(
        br#"<i class="old">a</i><i>b</i><script>x</script>"#,
        true,
        &mut ex,
    );
    assert!(
        html.contains(r#"<i class="first">a</i><i class="second">b</i>"#),
        "{html}"
    );
}

#[test]
fn get_element_by_id_misses_unknown_id() {
    struct Probe;
    impl ScriptExecutor for Probe {
        fn execute_script(
            &mut self,
            host: &mut ScriptHost<'_, '_, '_>,
            _el: ScriptElement,
            _attrs: &[(String, String)],
            _script: &str,
        ) {
            assert!(host.get_element_by_id("nope").is_none());
            assert!(host.get_element_by_id("y").is_some());
        }
    }
    let mut ex = Probe;
    doc_with_executor(b"<div id=y><script>x</script>", true, &mut ex);
}

#[test]
fn script_type_module_does_not_execute() {
    let mut ex = TestExecutor::default();
    let html = doc_with_executor(b"<script type=module>x()</script>", true, &mut ex);
    assert!(ex.ran.is_empty());
    assert!(
        html.contains(r#"<script type="module">x()</script>"#),
        "{html}"
    );
}

#[test]
fn script_data_block_type_does_not_execute() {
    let mut ex = TestExecutor::default();
    let html = doc_with_executor(
        b"<script type=\"application/json\">{\"a\":1}</script>",
        true,
        &mut ex,
    );
    assert!(ex.ran.is_empty());
    assert!(html.contains("application/json"), "{html}");
}

#[test]
fn script_javascript_type_aliases_execute() {
    // Missing, empty, whitespace-only, and JavaScript MIME types are
    // classic; each must execute exactly once.
    for ty in [
        "",
        "   ",
        "text/javascript",
        "TEXT/JAVASCRIPT",
        "application/x-javascript",
        // WHATWG "prepare the script" matches on the MIME essence:
        // parameters after `;` are stripped.
        "text/javascript;version=1.8",
        "text/javascript ; charset=utf-8",
        "APPLICATION/JAVASCRIPT;FOO=1",
    ] {
        let mut ex = TestExecutor::default();
        let input = format!("<script type=\"{ty}\">x()</script>");
        doc_with_executor(input.as_bytes(), true, &mut ex);
        assert_eq!(ex.ran, vec!["x()".to_string()], "type={ty:?}");
    }
}

#[test]
fn script_in_template_does_not_execute() {
    let mut ex = TestExecutor::default();
    let html = doc_with_executor(b"<template><script>x()</script></template>", true, &mut ex);
    assert!(ex.ran.is_empty());
    assert!(
        html.contains("<template><script>x()</script></template>"),
        "{html}"
    );
}

#[test]
fn unclosed_script_executes_at_eof() {
    let mut ex = TestExecutor::default();
    let html = doc_with_executor(b"<p><script>hello()", true, &mut ex);
    assert_eq!(ex.ran, vec!["hello()".to_string()]);
    assert!(html.contains("<script>hello()</script>"), "{html}");
}

#[test]
fn executor_receives_element_and_attributes() {
    use std::cell::RefCell;
    struct Attrs {
        seen: RefCell<Vec<(String, String)>>,
    }
    impl ScriptExecutor for Attrs {
        fn execute_script(
            &mut self,
            host: &mut ScriptHost<'_, '_, '_>,
            el: ScriptElement,
            attrs: &[(String, String)],
            _script: &str,
        ) {
            self.seen.borrow_mut().extend(attrs.iter().cloned());
            // The element handle identifies the running script: mutating
            // through it is visible in the finished tree.
            host.set_attribute(el, "data-ran", "1");
        }
    }
    let mut ex = Attrs {
        seen: RefCell::new(Vec::new()),
    };
    let html = doc_with_executor(
        b"<script async src=\"x.js\" type=\"text/javascript\">s()</script>",
        true,
        &mut ex,
    );
    assert_eq!(
        ex.seen.borrow().as_slice(),
        &[
            ("async".to_string(), String::new()),
            ("src".to_string(), "x.js".to_string()),
            ("type".to_string(), "text/javascript".to_string()),
        ]
    );
    assert!(html.contains("data-ran=\"1\""), "{html}");
}

#[test]
fn stale_script_element_handle_fails_closed() {
    use std::cell::Cell;
    use std::rc::Rc;
    let retained: Rc<Cell<Option<ScriptElement>>> = Rc::new(Cell::new(None));
    struct First {
        retained: Rc<Cell<Option<ScriptElement>>>,
    }
    impl ScriptExecutor for First {
        fn execute_script(
            &mut self,
            _host: &mut ScriptHost<'_, '_, '_>,
            el: ScriptElement,
            _attrs: &[(String, String)],
            _script: &str,
        ) {
            self.retained.set(Some(el));
        }
    }
    let mut first = First {
        retained: retained.clone(),
    };
    doc_with_executor(b"<script>one()</script>", true, &mut first);
    let stale = retained.get().expect("first parse ran its script");

    struct Second {
        stale: ScriptElement,
    }
    impl ScriptExecutor for Second {
        fn execute_script(
            &mut self,
            host: &mut ScriptHost<'_, '_, '_>,
            el: ScriptElement,
            _attrs: &[(String, String)],
            _script: &str,
        ) {
            // The stale handle must neither panic nor mutate: no-op.
            host.set_attribute(self.stale, "data-stale", "1");
            // A fresh handle from this parse still works.
            host.set_attribute(el, "data-fresh", "1");
        }
    }
    let mut second = Second { stale };
    let html = doc_with_executor(b"<script>two()</script>", true, &mut second);
    assert!(!html.contains("data-stale"), "{html}");
    assert!(html.contains("data-fresh"), "{html}");
}

#[test]
fn get_elements_by_tag_name_wildcard() {
    struct Wildcard {
        all: usize,
        spans: usize,
    }
    impl ScriptExecutor for Wildcard {
        fn execute_script(
            &mut self,
            host: &mut ScriptHost<'_, '_, '_>,
            _el: ScriptElement,
            _attrs: &[(String, String)],
            _script: &str,
        ) {
            // html, head, body, div, template, p, script: the span inside
            // the template is not descended into.
            self.all = host.get_elements_by_tag_name("*").len();
            self.spans = host.get_elements_by_tag_name("span").len();
        }
    }
    let mut ex = Wildcard { all: 0, spans: 0 };
    doc_with_executor(
        b"<div><template><span>t</span></template><p><script>x</script></p></div>",
        true,
        &mut ex,
    );
    assert_eq!(ex.all, 7);
    assert_eq!(ex.spans, 0);
}

#[test]
fn document_write_sequential_writes_do_not_replay() {
    // Two scripts each document.write: the second insertion must not
    // replay the first insertion's (already consumed) segment.
    use std::collections::HashMap;
    struct Seq {
        ran: Vec<String>,
        payloads: HashMap<String, String>,
    }
    impl ScriptExecutor for Seq {
        fn execute_script(
            &mut self,
            host: &mut ScriptHost<'_, '_, '_>,
            _el: ScriptElement,
            _attrs: &[(String, String)],
            script: &str,
        ) {
            self.ran.push(script.to_string());
            if let Some(payload) = self.payloads.get(script).cloned() {
                host.document_write(&payload);
            }
        }
    }
    let mut ex = Seq {
        ran: Vec::new(),
        payloads: HashMap::from([
            ("a()".to_string(), "A".to_string()),
            ("b()".to_string(), "B".to_string()),
        ]),
    };
    let html = doc_with_executor(
        b"<script>a()</script><script>b()</script>tail",
        true,
        &mut ex,
    );
    assert_eq!(ex.ran, vec!["a()".to_string(), "b()".to_string()]);
    // The written text token pops `head`, so "A" lands in `body`; the
    // point is ordering and no replay: A appears exactly once, before B.
    assert!(
        html.contains("<head><script>a()</script></head><body>A<script>b()</script>Btail"),
        "{html}"
    );
}

#[test]
fn document_write_nested_two_levels() {
    // A written script that itself writes: the inner insertion nests
    // inside the still-unread outer insertion; both run in order.
    use std::collections::HashMap;
    struct Seq {
        ran: Vec<String>,
        payloads: HashMap<String, String>,
    }
    impl ScriptExecutor for Seq {
        fn execute_script(
            &mut self,
            host: &mut ScriptHost<'_, '_, '_>,
            _el: ScriptElement,
            _attrs: &[(String, String)],
            script: &str,
        ) {
            self.ran.push(script.to_string());
            if let Some(payload) = self.payloads.get(script).cloned() {
                host.document_write(&payload);
            }
        }
    }
    let mut ex = Seq {
        ran: Vec::new(),
        payloads: HashMap::from([
            ("o()".to_string(), "<script>i()</script>".to_string()),
            ("i()".to_string(), "Z".to_string()),
        ]),
    };
    let html = doc_with_executor(b"<script>o()</script>done", true, &mut ex);
    assert_eq!(ex.ran, vec!["o()".to_string(), "i()".to_string()]);
    // The inner script's written text pops `head`, so "Z" lands in
    // `body` ahead of "done".
    assert!(
        html.contains("<head><script>o()</script><script>i()</script></head><body>Zdone"),
        "{html}"
    );
}

#[test]
fn document_write_unconsume_across_insertion_boundary() {
    // The written text ends mid-character-reference; the reconsume in the
    // numeric-reference start state must step back into the inserted
    // segment, not into the base input before the insertion point.
    struct W;
    impl ScriptExecutor for W {
        fn execute_script(
            &mut self,
            host: &mut ScriptHost<'_, '_, '_>,
            _el: ScriptElement,
            _attrs: &[(String, String)],
            _script: &str,
        ) {
            host.document_write("&#y");
        }
    }
    let mut ex = W;
    let html = doc_with_executor(b"ab<script>w()</script>z;", true, &mut ex);
    // "&#y" is not a valid reference: emitted literally, then "z;".
    assert!(
        html.contains("<body>ab<script>w()</script>&#yz;</body>"),
        "{html}"
    );
}

// -- querySelector complex selectors (wave 47) -------------------------------

/// Executor that runs `querySelectorAll` for each case while the script
/// runs (mid-parse) and records `(match count, matched ids in order)`.
struct QueryProbe {
    selectors: Vec<String>,
    results: Vec<(usize, Vec<String>)>,
}

impl ScriptExecutor for QueryProbe {
    fn execute_script(
        &mut self,
        host: &mut ScriptHost<'_, '_, '_>,
        _el: ScriptElement,
        _attrs: &[(String, String)],
        _script: &str,
    ) {
        for sel in &self.selectors {
            let els = host.query_selector_all(sel);
            let ids = els
                .iter()
                .map(|el| {
                    host.get_attributes(*el)
                        .into_iter()
                        .find(|(n, _)| n.eq_ignore_ascii_case("id"))
                        .map(|(_, v)| v)
                        .unwrap_or_default()
                })
                .collect();
            self.results.push((els.len(), ids));
        }
    }
}

fn probe_query(html: &str, selectors: &[&str]) -> Vec<(usize, Vec<String>)> {
    let mut ex = QueryProbe {
        selectors: selectors.iter().map(|s| s.to_string()).collect(),
        results: Vec::new(),
    };
    // The probe script runs after the markup, so the tree is populated.
    let full = format!("{html}<script>probe()</script>");
    parse_with_scripting_and_executor(full.as_bytes(), true, &mut ex);
    ex.results
}

#[test]
fn query_selector_combinators() {
    let html = "<div id=d><p id=p1><span id=s1></span></p><p id=p2></p></div><span id=s2></span>";
    let r = probe_query(
        html,
        &[
            "div span",   // descendant: s1
            "div > span", // child: none (span is not a direct child of div)
            "div > p",    // child: p1, p2
            "p + p",      // adjacent sibling: p2
            "p ~ span",   // general sibling: none (no span after p at that level)
            "div + span", // adjacent sibling: s2
            "div ~ span", // general sibling: s2
            "body span",  // descendant across levels: s1, s2
        ],
    );
    assert_eq!(r[0], (1, vec!["s1".to_string()]));
    assert_eq!(r[1], (0, vec![]));
    assert_eq!(r[2], (2, vec!["p1".to_string(), "p2".to_string()]));
    assert_eq!(r[3], (1, vec!["p2".to_string()]));
    assert_eq!(r[4], (0, vec![]));
    assert_eq!(r[5], (1, vec!["s2".to_string()]));
    assert_eq!(r[6], (1, vec!["s2".to_string()]));
    assert_eq!(r[7], (2, vec!["s1".to_string(), "s2".to_string()]));
}

#[test]
fn query_selector_structural_pseudo_classes() {
    let html = "<ul id=u><li id=a>1</li><li id=b>2</li><li id=c>3</li></ul><p id=p></p>";
    let r = probe_query(
        html,
        &[
            "li:first-child",
            "li:last-child",
            "li:nth-child(2)",
            "li:nth-child(odd)",
            "li:nth-child(even)",
            "li:nth-child(2n+1)",
            "li:not(:first-child)",
            "ul > li:nth-of-type(3)",
            "p:empty",
            "li:empty",
        ],
    );
    assert_eq!(r[0], (1, vec!["a".to_string()]));
    assert_eq!(r[1], (1, vec!["c".to_string()]));
    assert_eq!(r[2], (1, vec!["b".to_string()]));
    assert_eq!(r[3].1, vec!["a".to_string(), "c".to_string()]);
    assert_eq!(r[4], (1, vec!["b".to_string()]));
    assert_eq!(r[5].1, vec!["a".to_string(), "c".to_string()]);
    assert_eq!(r[6].1, vec!["b".to_string(), "c".to_string()]);
    assert_eq!(r[7], (1, vec!["c".to_string()]));
    assert_eq!(r[8], (1, vec!["p".to_string()]));
    // li elements contain text, so :empty does not match.
    assert_eq!(r[9], (0, vec![]));
}

/// `document.querySelector(All)`-level host queries must not match
/// detached nodes: `createElement` without insertion, and nodes
/// removed by `removeChild`/`replaceChild`. (The `replaceChild`
/// setup loops in the css-tables WPT files keep re-querying until no
/// match remains; matching the just-detached node would spin or
/// throw.)
struct DetachProbe {
    results: Vec<bool>,
}

impl ScriptExecutor for DetachProbe {
    fn execute_script(
        &mut self,
        host: &mut ScriptHost<'_, '_, '_>,
        _el: ScriptElement,
        _attrs: &[(String, String)],
        _script: &str,
    ) {
        // A created-but-never-inserted element is not in the document.
        let fresh = host.create_element("span");
        host.set_attribute(fresh, "id", "detached");
        self.results
            .push(host.query_selector("#detached").is_none());
        // Removing an element takes it out of `querySelector(All)`.
        let target = host.get_element_by_id("victim").expect("victim present");
        let parent = host.parent_element(target).expect("victim parent");
        assert!(host.remove_child(parent, target));
        self.results.push(host.query_selector("#victim").is_none());
        self.results.push(host.query_selector_all("i").is_empty());
        // The detached node is also gone from its old parent's subtree.
        self.results
            .push(host.query_selector_in(parent, "#victim").is_none());
        // `getElementById` agrees (it always walked the document).
        self.results
            .push(host.get_element_by_id("victim").is_none());
    }
}

#[test]
fn query_selector_skips_detached_nodes() {
    let mut ex = DetachProbe {
        results: Vec::new(),
    };
    parse_with_scripting_and_executor(
        b"<div><i id=\"victim\">x</i></div><script>probe()</script>",
        true,
        &mut ex,
    );
    assert_eq!(ex.results, vec![true; 5]);
}

#[test]
fn query_selector_attribute_operators() {
    let html = "<div id=d1 data-x=\"foobar\" class=\"a b\"></div>\
                <div id=d2 data-x=\"foo bar\"></div>\
                <div id=d3 data-x=\"en-US\"></div>";
    let r = probe_query(
        html,
        &[
            "div[data-x]",
            "[data-x=\"foobar\"]",
            "[data-x^=\"foo\"]",
            "[data-x$=\"bar\"]",
            "[data-x*=\"ob\"]",
            "[data-x~=\"bar\"]",
            "[data-x|=\"en\"]",
            ".a.b",
            "div:not([data-x])",
        ],
    );
    assert_eq!(r[0].0, 3);
    assert_eq!(r[1], (1, vec!["d1".to_string()]));
    assert_eq!(r[2].1, vec!["d1".to_string(), "d2".to_string()]);
    assert_eq!(r[3].1, vec!["d1".to_string(), "d2".to_string()]);
    assert_eq!(r[4], (1, vec!["d1".to_string()]));
    assert_eq!(r[5], (1, vec!["d2".to_string()]));
    assert_eq!(r[6], (1, vec!["d3".to_string()]));
    assert_eq!(r[7], (1, vec!["d1".to_string()]));
    assert_eq!(r[8], (0, vec![]));
}

#[test]
fn query_selector_wpt_table_shapes() {
    // Shapes taken from real css-tables testharness selectors.
    let html = "<main id=m><div id=d1>\
                <x-table id=t1><x-tr id=r1><x-td id=c1></x-td></x-tr></x-table>\
                <x-table id=t2></x-table></div>\
                <div id=d2></div></main>";
    let r = probe_query(
        html,
        &[
            "x-table:nth-of-type(1)",
            "main > div:nth-of-type(2)",
            "x-table:nth-of-type(1) > x-tr:first-child > x-td:first-child",
            "main table",
            "table *",
            "x-td[rowspan], x-td[colspan]",
        ],
    );
    assert_eq!(r[0], (1, vec!["t1".to_string()]));
    assert_eq!(r[1], (1, vec!["d2".to_string()]));
    assert_eq!(r[2], (1, vec!["c1".to_string()]));
    // `table` does not match `x-table`.
    assert_eq!(r[3], (0, vec![]));
    assert_eq!(r[4], (0, vec![]));
    assert_eq!(r[5], (0, vec![]));
}

#[test]
fn query_selector_unsupported_still_null() {
    // Unsupported selectors keep the old fail-closed behavior (no match),
    // rather than throwing or mis-matching.
    let html = "<div id=d><p id=p></p></div>";
    let r = probe_query(
        html,
        &[
            "div:hover",
            "div::before",
            "div:has(p)",
            "div >",
            ":not(div > p)",
        ],
    );
    for (i, sel) in [
        "div:hover",
        "div::before",
        "div:has(p)",
        "div >",
        ":not(div > p)",
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(r[i].0, 0, "selector {sel} should not match");
    }
}
