use super::*;

fn tag(doc: &Document, n: NodeId) -> &str {
    match doc.kind(n) {
        NodeKind::Element(d) => &d.tag_name,
        _ => panic!("expected element"),
    }
}

// -- tree structure --------------------------------------------------------

#[test]
fn root_is_document_node() {
    let doc = Document::new();
    assert!(matches!(doc.kind(doc.root()), NodeKind::Document));
    assert_eq!(doc.parent(doc.root()), None);
    assert!(doc.children(doc.root()).is_empty());
    assert_eq!(doc.document_element(), None);
}

#[test]
fn build_tree_and_navigate() {
    let mut doc = Document::new();
    let root = doc.root();
    let html = doc.create_element("HTML"); // createElement lowercases
    assert_eq!(tag(&doc, html), "html");
    let head = doc.create_element("head");
    let body = doc.create_element("body");
    let p1 = doc.create_element("p");
    let p2 = doc.create_element("p");
    let t = doc.create_text_node("hello");
    doc.append_child(root, html);
    doc.append_child(html, head);
    doc.append_child(html, body);
    doc.append_child(body, p1);
    doc.append_child(p1, t);
    doc.append_child(body, p2);

    assert_eq!(doc.document_element(), Some(html));
    assert_eq!(doc.parent(html), Some(root));
    assert_eq!(doc.children(html), vec![head, body]);
    assert_eq!(doc.children(body), vec![p1, p2]);
    assert_eq!(doc.next_sibling(p1), Some(p2));
    assert_eq!(doc.previous_sibling(p2), Some(p1));
    assert_eq!(doc.next_sibling(p2), None);
    assert_eq!(doc.previous_sibling(p1), None);
    assert_eq!(doc.children(p1), vec![t]);
    assert_eq!(doc.ancestors(t), vec![p1, body, html, root]);
}

#[test]
fn append_moves_node_from_old_parent() {
    let mut doc = Document::new();
    let a = doc.create_element("div");
    let b = doc.create_element("div");
    let c = doc.create_element("span");
    doc.append_child(doc.root(), a);
    doc.append_child(doc.root(), b);
    doc.append_child(a, c);
    doc.append_child(b, c); // moves c from a to b
    assert!(doc.children(a).is_empty());
    assert_eq!(doc.children(b), vec![c]);
    assert_eq!(doc.parent(c), Some(b));
}

#[test]
fn remove_child_repairs_links() {
    let mut doc = Document::new();
    let p = doc.create_element("div");
    let kids: Vec<NodeId> = (0..4).map(|_| doc.create_element("span")).collect();
    doc.append_child(doc.root(), p);
    for &k in &kids {
        doc.append_child(p, k);
    }
    doc.remove_child(p, kids[1]);
    assert_eq!(doc.children(p), vec![kids[0], kids[2], kids[3]]);
    assert_eq!(doc.next_sibling(kids[0]), Some(kids[2]));
    assert_eq!(doc.previous_sibling(kids[2]), Some(kids[0]));
    assert_eq!(doc.parent(kids[1]), None);
    // removing head and tail
    doc.remove_child(p, kids[0]);
    doc.remove_child(p, kids[3]);
    assert_eq!(doc.children(p), vec![kids[2]]);
    assert_eq!(doc.next_sibling(kids[2]), None);
    assert_eq!(doc.previous_sibling(kids[2]), None);
}

#[test]
fn remove_child_wrong_parent_is_noop() {
    let mut doc = Document::new();
    let a = doc.create_element("div");
    let b = doc.create_element("div");
    let c = doc.create_element("span");
    doc.append_child(doc.root(), a);
    doc.append_child(doc.root(), b);
    doc.append_child(a, c);
    doc.remove_child(b, c); // c is not b's child
    assert_eq!(doc.children(a), vec![c]);
    assert_eq!(doc.parent(c), Some(a));
}

#[test]
fn append_cycle_guards() {
    let mut doc = Document::new();
    let a = doc.create_element("div");
    let b = doc.create_element("div");
    doc.append_child(doc.root(), a);
    doc.append_child(a, b);
    doc.append_child(a, a); // self-append: no-op
    doc.append_child(b, a); // ancestor append: no-op
    assert_eq!(doc.children(a), vec![b]);
    assert_eq!(doc.parent(a), Some(doc.root()));
}

#[test]
fn append_child_to_text_node_is_noop() {
    let mut doc = Document::new();
    let t = doc.create_text_node("x");
    let e = doc.create_element("div");
    doc.append_child(t, e); // Text nodes cannot have children: no-op
    assert_eq!(doc.parent(e), None);
    assert_eq!(doc.children(t), Vec::new());
}

#[test]
fn append_child_to_doctype_node_is_noop() {
    let mut doc = Document::new();
    let dt = doc.alloc(NodeKind::DocumentType(DocumentTypeData {
        name: "html".into(),
        public_id: None,
        system_id: None,
    }));
    let e = doc.create_element("div");
    doc.append_child(dt, e); // DocumentType nodes cannot have children: no-op
    assert_eq!(doc.parent(e), None);
}

#[test]
fn node_ids_stable_across_appends() {
    let mut doc = Document::new();
    let root = doc.create_element("div");
    doc.append_child(doc.root(), root);
    let first = doc.create_element("div");
    doc.append_child(root, first);
    doc.set_attribute(first, "id", "keep");
    for i in 0..5000 {
        let e = doc.create_element("span");
        doc.set_attribute(e, "data-n", &i.to_string());
        doc.append_child(root, e);
    }
    // The first id still addresses the same node with its attribute.
    assert_eq!(doc.get_attribute(first, "id"), Some("keep"));
    assert_eq!(doc.parent(first), Some(root));
    assert_eq!(doc.children(root).len(), 5001);
}

#[test]
fn removed_nodes_stay_addressable() {
    let mut doc = Document::new();
    let p = doc.create_element("div");
    let c = doc.create_element("span");
    doc.append_child(doc.root(), p);
    doc.append_child(p, c);
    doc.set_attribute(c, "id", "x");
    doc.remove_child(p, c);
    assert_eq!(doc.get_attribute(c, "id"), Some("x"));
    assert!(matches!(doc.kind(c), NodeKind::Element(_)));
}

// -- attributes ------------------------------------------------------------

#[test]
fn attributes_set_get_remove() {
    let mut doc = Document::new();
    let e = doc.create_element("div");
    assert_eq!(doc.get_attribute(e, "id"), None);
    doc.set_attribute(e, "id", "a");
    doc.set_attribute(e, "CLASS", "x y"); // HTML: stored lowercased
    assert_eq!(doc.get_attribute(e, "id"), Some("a"));
    assert_eq!(doc.get_attribute(e, "class"), Some("x y"));
    assert_eq!(doc.get_attribute(e, "CLASS"), Some("x y")); // lookup CI too
    doc.set_attribute(e, "id", "b"); // replace, not duplicate
    assert_eq!(doc.get_attribute(e, "id"), Some("b"));
    if let NodeKind::Element(d) = doc.kind(e) {
        assert_eq!(d.attributes.len(), 2);
    } else {
        panic!("expected element");
    }
    doc.remove_attribute(e, "ID");
    assert_eq!(doc.get_attribute(e, "id"), None);
    doc.remove_attribute(e, "nonexistent"); // no-op
}

#[test]
fn foreign_attributes_are_case_sensitive() {
    let mut doc = Document::new();
    let e = doc.create_element_ns(Namespace::Svg, "foreignObject");
    doc.set_attribute(e, "viewBox", "0 0 1 1");
    assert_eq!(doc.get_attribute(e, "viewBox"), Some("0 0 1 1"));
    assert_eq!(doc.get_attribute(e, "viewbox"), None);
}

#[test]
fn attribute_ops_on_non_elements_are_noops() {
    let mut doc = Document::new();
    let t = doc.create_text_node("x");
    assert_eq!(doc.get_attribute(t, "id"), None);
    doc.set_attribute(t, "id", "x");
    doc.remove_attribute(t, "id");
    assert!(matches!(doc.kind(t), NodeKind::Text(_)));
}

// -- lookup -----------------------------------------------------------------

fn lookup_fixture() -> Document {
    let mut doc = Document::new();
    let root = doc.root();
    let html = doc.create_element("html");
    let body = doc.create_element("body");
    doc.append_child(root, html);
    doc.append_child(html, body);
    let d1 = doc.create_element("div");
    doc.set_attribute(d1, "id", "main");
    doc.set_attribute(d1, "class", "a b");
    let d2 = doc.create_element("div");
    doc.set_attribute(d2, "class", "b");
    let s = doc.create_element("span");
    doc.set_attribute(s, "id", "main"); // duplicate id; first in order wins
    doc.append_child(body, d1);
    doc.append_child(body, d2);
    doc.append_child(d2, s);
    doc
}

#[test]
fn get_element_by_id_first_in_document_order() {
    let doc = lookup_fixture();
    let found = doc.get_element_by_id("main").unwrap();
    assert_eq!(tag(&doc, found), "div");
    assert_eq!(doc.get_element_by_id("nope"), None);
}

#[test]
fn get_elements_by_tag_name() {
    let doc = lookup_fixture();
    let divs = doc.get_elements_by_tag_name("div");
    assert_eq!(divs.len(), 2);
    assert_eq!(tag(&doc, divs.item(0).unwrap()), "div");
    let all = doc.get_elements_by_tag_name("*");
    // html, body, div, div, span
    assert_eq!(all.len(), 5);
    assert!(doc.get_elements_by_tag_name("table").is_empty());
    // case-insensitive for HTML elements
    assert_eq!(doc.get_elements_by_tag_name("DIV").len(), 2);
}

#[test]
fn get_elements_by_tag_name_matches_non_canonical_html_tag() {
    // `createElementNS(Html, "DIV")` keeps the tag's case (the name
    // spills off the atom table); the DOM spec still matches ASCII
    // case-insensitively in HTML documents, so "div" must find it.
    let mut doc = Document::new();
    let root = doc.root();
    let html = doc.create_element("html");
    doc.append_child(root, html);
    let upper = doc.create_element_ns(Namespace::Html, "DIV");
    doc.append_child(html, upper);
    assert_eq!(doc.get_elements_by_tag_name("div").len(), 1);
    assert_eq!(doc.get_elements_by_tag_name("DIV").len(), 1);
    assert_eq!(doc.get_elements_by_tag_name("span").len(), 0);
    // foreign namespaces stay case-sensitive: spilled "DIV" in SVG
    // must NOT match "div".
    let svg_div = doc.create_element_ns(Namespace::Svg, "DIV");
    doc.append_child(html, svg_div);
    assert_eq!(doc.get_elements_by_tag_name("div").len(), 1);
    assert_eq!(doc.get_elements_by_tag_name("DIV").len(), 2);
}

// -- selectors ----------------------------------------------------------------

fn selector_fixture() -> Document {
    // <div id="app" class="root"><section class="a"><p id="x" class="a b">t</p></section><p class="b">u</p></div>
    let mut doc = Document::new();
    let root = doc.root();
    let div = doc.create_element("div");
    doc.set_attribute(div, "id", "app");
    doc.set_attribute(div, "class", "root");
    let section = doc.create_element("section");
    doc.set_attribute(section, "class", "a");
    let p1 = doc.create_element("p");
    doc.set_attribute(p1, "id", "x");
    doc.set_attribute(p1, "class", "a b");
    let t1 = doc.create_text_node("t");
    doc.append_child(p1, t1);
    let p2 = doc.create_element("p");
    doc.set_attribute(p2, "class", "b");
    let t2 = doc.create_text_node("u");
    doc.append_child(p2, t2);
    doc.append_child(root, div);
    doc.append_child(div, section);
    doc.append_child(section, p1);
    doc.append_child(div, p2);
    doc
}

fn qs_text(doc: &Document, sel: &str) -> Vec<String> {
    doc.query_selector_all(sel)
        .expect("valid selector")
        .into_iter()
        .map(|n| doc.text_content(n))
        .collect()
}

#[test]
fn selector_type_class_id() {
    let doc = selector_fixture();
    assert_eq!(qs_text(&doc, "p"), vec!["t", "u"]);
    assert_eq!(qs_text(&doc, ".a"), vec!["t", "t"]); // section and p both match
    assert_eq!(qs_text(&doc, ".b"), vec!["t", "u"]);
    assert_eq!(qs_text(&doc, "#x"), vec!["t"]);
    assert_eq!(qs_text(&doc, "p#x"), vec!["t"]);
    assert_eq!(qs_text(&doc, "p.b"), vec!["t", "u"]);
    assert_eq!(qs_text(&doc, "p.a.b"), vec!["t"]);
    assert_eq!(qs_text(&doc, ".a.b"), vec!["t"]);
    assert_eq!(qs_text(&doc, "#app.root"), vec!["tu"]);
    assert_eq!(qs_text(&doc, "*"), vec!["tu", "t", "t", "u"]);
    assert!(
        doc.query_selector("span")
            .expect("valid selector")
            .is_none()
    );
    assert!(
        doc.query_selector("#missing")
            .expect("valid selector")
            .is_none()
    );
}

#[test]
fn selector_combinators() {
    let doc = selector_fixture();
    assert_eq!(qs_text(&doc, "div p"), vec!["t", "u"]);
    assert_eq!(qs_text(&doc, "div > p"), vec!["u"]);
    assert_eq!(qs_text(&doc, "section > p"), vec!["t"]);
    assert_eq!(qs_text(&doc, "div section p"), vec!["t"]);
    assert_eq!(qs_text(&doc, "div > section > p"), vec!["t"]);
    assert_eq!(qs_text(&doc, "section p.b"), vec!["t"]);
    assert!(
        doc.query_selector("p > div")
            .expect("valid selector")
            .is_none()
    );
    assert!(
        doc.query_selector("section > #app")
            .expect("valid selector")
            .is_none()
    );
}

#[test]
fn selector_groups_and_malformed() {
    let doc = selector_fixture();
    assert_eq!(qs_text(&doc, "#x, .b"), vec!["t", "u"]);
    assert_eq!(qs_text(&doc, "span, p"), vec!["t", "u"]);
    // Malformed groups are syntax errors (the DOM `SyntaxError`
    // analogue): one bad group fails the whole call, even when other
    // groups are valid — no silent never-match.
    assert!(doc.query_selector_all(">>>, p").is_err());
    assert!(doc.query_selector_all("p, >>>").is_err());
    assert!(doc.query_selector("").is_err());
    assert!(doc.query_selector("   ").is_err());
    assert!(doc.query_selector("div >").is_err());
    assert!(doc.query_selector("> div").is_err());
    assert!(doc.query_selector("div..a").is_err());
    assert!(doc.query_selector("div#").is_err());
    assert!(doc.query_selector("p#a#b").is_err());
    assert!(doc.query_selector("p:first-child").is_err());
    assert!(doc.query_selector("div[").is_err());
    assert!(doc.query_selector("[data-x=\"unterminated").is_err());
    assert!(doc.query_selector("[data-x~= \"a b\"]").is_err());
    assert!(doc.query_selector("[data-x^= \"\"]").is_err());
    // query_selector returns the first in document order
    let first = doc.query_selector("p").expect("valid selector").unwrap();
    assert_eq!(doc.text_content(first), "t");
}

#[test]
fn query_selector_first_match_in_document_order() {
    // `query_selector` agrees with the first hit of `query_selector_all`
    // across combinators, groups, and malformed groups — it just stops
    // the walk at the first match instead of collecting every match.
    let doc = selector_fixture();
    for sel in [
        "p",
        ".a",
        ".b",
        "#x",
        "p#x",
        "div p",
        "div > p",
        "section > p",
        "div section p",
        "div > section > p",
        "*",
        "#x, .b",
        "span, p",
        ">>>, p",
        "p, >>>",
        "span",
        "#missing",
        "",
        "div >",
        "section p + p",
        "div ~ p",
        "[class]",
        "p[class=\"a b\"]",
    ] {
        let all = doc.query_selector_all(sel);
        // Both sides agree: same first hit, or the same syntax error.
        assert_eq!(
            doc.query_selector(sel),
            all.map(|v| v.into_iter().next()),
            "mismatch for {sel:?}"
        );
    }
    // First document-order match, not first group.
    let first = doc
        .query_selector(".b, #x")
        .expect("valid selector")
        .expect("match");
    assert_eq!(doc.text_content(first), "t");
}

#[test]
fn descendant_combinator_scales_linearly() {
    // Regression guard: right-to-left matching walked the full ancestor
    // chain for every candidate, so `span div` over a deep div chain was
    // quadratic (2.5s at depth 8k, 21.2s at depth 32k in release).
    // Matching is now left-to-right at O(nodes x compounds): a 32k-deep
    // chain completes in well under 1s in release.
    //
    // Depth is bounded so this stays fast under debug builds (override
    // with BOWSER_DOM_PERF_DEPTH, e.g. 32000 for the release check).
    let depth: usize = std::env::var("BOWSER_DOM_PERF_DEPTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8_000);
    let mut html = String::from("<html><body>");
    for _ in 0..depth {
        html.push_str("<div>");
    }
    for _ in 0..depth {
        html.push_str("</div>");
    }
    html.push_str("</body></html>");
    let (doc, _) = Document::parse_html(html.as_bytes());
    // sanity: nested sources still match through the descendant step
    assert_eq!(
        doc.query_selector_all("div div")
            .expect("valid selector")
            .len(),
        depth - 1
    );
    let start = std::time::Instant::now();
    let found = doc.query_selector_all("span div").expect("valid selector");
    let elapsed = start.elapsed();
    assert!(found.is_empty());
    assert!(
        elapsed.as_secs() < 30,
        "descendant matching regressed to superlinear: {elapsed:?} at depth {depth}"
    );
}

// -- parse_html ---------------------------------------------------------------

#[test]
fn parse_simple_document() {
    let html = b"<!DOCTYPE html><html><head><title>Hi</title></head><body><p id=\"a\">Hello <b>world</b></p><!-- c --></body></html>";
    let (doc, quirks) = Document::parse_html(html);
    assert_eq!(quirks, QuirksMode::NoQuirks);
    let html_el = doc.document_element().unwrap();
    assert_eq!(tag(&doc, html_el), "html");
    // doctype preserved
    let kids = doc.children(doc.root());
    assert!(matches!(doc.kind(kids[0]), NodeKind::DocumentType(_)));
    if let NodeKind::DocumentType(d) = doc.kind(kids[0]) {
        assert_eq!(d.name, "html");
    }
    // lookup + text
    let p = doc.get_element_by_id("a").unwrap();
    assert_eq!(doc.text_content(p), "Hello world");
    assert_eq!(doc.serialize_node(p), "<p id=\"a\">Hello <b>world</b></p>");
    // comment preserved
    let body = doc.get_elements_by_tag_name("body").item(0).unwrap();
    let body_kids = doc.children(body);
    assert!(matches!(
        doc.kind(*body_kids.last().unwrap()),
        NodeKind::Comment(_)
    ));
}

#[test]
fn parse_html_name_round_trip() {
    // Pins name behavior only, not the zero-alloc property of the
    // atomized path: well-known names must land as pure atoms (no heap
    // spill); unknown names (custom elements, `data-*`) spill but
    // round-trip exactly; the id index is still fed. The allocation
    // property itself is pinned by `tests/atom_alloc.rs`.
    let html = br#"<div id="main" class="c" data-x="1" custom-attr="v"><my-element foo="bar"></my-element></div>"#;
    let (doc, _) = Document::parse_html(html);
    let div = doc.get_element_by_id("main").unwrap();
    let NodeKind::Element(d) = doc.kind(div) else {
        panic!("expected element");
    };
    assert_ne!(d.tag_name.atom(), NameAtom::UNKNOWN);
    assert_eq!(d.tag_name.as_str(), "div");
    let by_name = |n: &str| d.attributes.iter().find(|a| a.name.as_str() == n).unwrap();
    assert_ne!(by_name("class").name.atom(), NameAtom::UNKNOWN);
    assert_eq!(by_name("class").value, "c");
    for unknown in ["data-x", "custom-attr"] {
        let a = by_name(unknown);
        assert_eq!(a.name.atom(), NameAtom::UNKNOWN, "{unknown} should spill");
        assert_eq!(a.name.as_str(), unknown);
    }
    // Custom element tag name and its unknown attribute round-trip
    // exactly, spilling onto the heap sidecar.
    let mut custom = None;
    let mut custom_foo = None;
    for child in doc.children(div) {
        if let NodeKind::Element(e) = doc.kind(child)
            && e.tag_name.as_str() == "my-element"
        {
            custom = Some(e.tag_name.atom());
            custom_foo = e.attributes.iter().find(|a| a.name.as_str() == "foo");
        }
    }
    assert_eq!(custom, Some(NameAtom::UNKNOWN));
    let foo = custom_foo.unwrap();
    assert_eq!(foo.name.atom(), NameAtom::UNKNOWN);
    assert_eq!(foo.value, "bar");
}

#[test]
fn string_start_element_indexes_uppercase_id() {
    // Regression test for the atomization refactor: the `String`
    // `start_element` fallback must preserve the old HTML
    // case-insensitivity contract (`attr_name_eq(ns, name, "id")`), so a
    // driver passing `("ID", …)` still feeds the id index.
    let mut doc = Document::new();
    let mut sink = DomSink {
        doc: &mut doc,
        stack: Vec::new(),
        template_depth: 0,
        index_ids: true,
        shadow_stack: Vec::new(),
    };
    TreeSink::start_element(
        &mut sink,
        "div".to_string(),
        Namespace::Html,
        vec![("ID".to_string(), "upper".to_string())],
        None,
    );
    drop(sink);
    let el = doc
        .get_element_by_id("upper")
        .expect("uppercase ID attribute must populate the id index");
    let NodeKind::Element(d) = doc.kind(el) else {
        panic!("expected element");
    };
    assert_eq!(d.tag_name.as_str(), "div");
    // The stored attribute name is the ASCII-lowercased HTML name.
    assert_eq!(d.attributes[0].name.as_str(), "id");
}

#[test]
fn parse_quirks_modes() {
    let (_, q) = Document::parse_html(b"<html><body><p>x</p></body></html>");
    assert_eq!(q, QuirksMode::Quirks); // missing doctype
    let (_, q) = Document::parse_html(
        b"<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\"><html></html>",
    );
    assert_eq!(q, QuirksMode::LimitedQuirks);
}

#[test]
fn parse_preserves_namespaces() {
    let (doc, _) = Document::parse_html(
        b"<html><body><svg viewBox=\"0 0 1 1\"><foreignObject><p>x</p></foreignObject></svg><math><mi>y</mi></math></body></html>",
    );
    let svg = doc.get_elements_by_tag_name("svg").item(0).unwrap();
    assert!(matches!(
        doc.kind(svg),
        NodeKind::Element(d) if d.namespace == Namespace::Svg
    ));
    let fo = doc
        .get_elements_by_tag_name("foreignObject")
        .item(0)
        .unwrap();
    assert!(matches!(
        doc.kind(fo),
        NodeKind::Element(d) if d.namespace == Namespace::Svg
    ));
    // HTML integration point: <p> inside foreignObject is HTML again
    let p = doc.get_elements_by_tag_name("p").item(0).unwrap();
    assert!(matches!(
        doc.kind(p),
        NodeKind::Element(d) if d.namespace == Namespace::Html
    ));
    let mi = doc.get_elements_by_tag_name("mi").item(0).unwrap();
    assert!(matches!(
        doc.kind(mi),
        NodeKind::Element(d) if d.namespace == Namespace::MathMl
    ));
    // namespaced lookup by tag still works
    assert_eq!(
        doc.query_selector_all("svg").expect("valid selector").len(),
        1
    );
}

#[test]
fn parse_template_contents_become_fragment_child() {
    let (doc, _) =
        Document::parse_html(b"<html><body><template><p id=\"t\">in</p></template></body></html>");
    let template = doc.get_elements_by_tag_name("template").item(0).unwrap();
    // General traversal skips the contents fragment; reach it directly.
    assert!(doc.children(template).is_empty());
    let frag = doc.template_contents(template).unwrap();
    assert!(matches!(doc.kind(frag), NodeKind::DocumentFragment));
    let frag_kids = doc.children(frag);
    assert_eq!(frag_kids.len(), 1);
    assert_eq!(tag(&doc, frag_kids[0]), "p");
    // get_element_by_id does not see into template contents
    assert_eq!(doc.get_element_by_id("t"), None);
    // template serializes with its contents inline
    assert_eq!(
        doc.serialize_node(template),
        "<template><p id=\"t\">in</p></template>"
    );
}

#[test]
fn cursor_walk_matches_children_vec_over_template() {
    // `first_child` + `next_sibling` must visit exactly the nodes
    // `children()` returns, with the template-contents fragment filtered
    // however it is positioned among its siblings.
    let (mut doc, _) = Document::parse_html(b"<template>hello</template>");
    let template = doc.get_elements_by_tag_name("template").item(0).unwrap();
    let mut via_cursor = Vec::new();
    let mut c = doc.first_child(template);
    while let Some(n) = c {
        via_cursor.push(n);
        c = doc.next_sibling(n);
    }
    assert_eq!(via_cursor, doc.children(template));
    assert!(via_cursor.is_empty());

    // Fragment in the middle: a regular child linked before it.
    let frag = doc.template_contents(template).unwrap();
    let div = doc.create_element("div");
    doc.insert_before(template, div, Some(frag));
    let mut via_cursor = Vec::new();
    let mut c = doc.first_child(template);
    while let Some(n) = c {
        via_cursor.push(n);
        c = doc.next_sibling(n);
    }
    assert_eq!(via_cursor, vec![div]);
    assert_eq!(via_cursor, doc.children(template));
}

#[test]
fn parse_processing_instruction() {
    let (doc, _) = Document::parse_html(b"<html><head><?php echo 1; ?></head><body></body></html>");
    let head = doc.get_elements_by_tag_name("head").item(0).unwrap();
    let pi = doc
        .children(head)
        .into_iter()
        .find(|&n| matches!(doc.kind(n), NodeKind::ProcessingInstruction { .. }))
        .unwrap();
    if let NodeKind::ProcessingInstruction { target, data } = doc.kind(pi) {
        assert_eq!(target, "php");
        assert!(data.contains("echo"));
    } else {
        panic!("expected PI");
    }
    assert!(doc.serialize_node(pi).starts_with("<?php"));
}

#[test]
fn parse_real_worldish_document() {
    let html = br#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Example &amp; Test</title>
<style>.a > .b { content: "&"; }</style>
</head>
<body class="home">
<header><nav><a href="/">Home</a><a href="/about">About</a></nav></header>
<main>
<article id="post-1" class="post featured">
<h1>Title &lt;here&gt;</h1>
<p>First paragraph with <em>emphasis</em> and a <a href="https://example.com/?a=1&amp;b=2">link</a>.</p>
<img src="pic.png" alt="a &quot;pic&quot;">
<table><tr><td>cell 1</td><td>cell 2</td></tr></table>
</article>
</main>
<script>if (a < b && ready) { init(); }</script>
<footer><!-- footer comment --></footer>
</body>
</html>"#;
    let (doc, quirks) = Document::parse_html(html);
    assert_eq!(quirks, QuirksMode::NoQuirks);
    // structure
    assert_eq!(doc.get_elements_by_tag_name("a").len(), 3);
    let h1 = doc.get_elements_by_tag_name("h1").item(0).unwrap();
    assert_eq!(doc.text_content(h1), "Title <here>");
    // entity decoding happened in the parser
    let title = doc.get_elements_by_tag_name("title").item(0).unwrap();
    assert_eq!(doc.text_content(title), "Example & Test");
    // selectors over parsed doc
    assert_eq!(
        doc.query_selector_all("main article.post")
            .expect("valid selector")
            .len(),
        1
    );
    assert_eq!(
        doc.query_selector_all("header nav > a")
            .expect("valid selector")
            .len(),
        2
    );
    assert_eq!(
        doc.query_selector_all("article#post-1.featured")
            .expect("valid selector")
            .len(),
        1
    );
    assert_eq!(
        doc.query_selector("table td")
            .expect("valid selector")
            .map(|n| doc.text_content(n)),
        Some("cell 1".into())
    );
    // script content kept verbatim (raw text, no entity decoding)
    let script = doc.get_elements_by_tag_name("script").item(0).unwrap();
    assert_eq!(doc.text_content(script), "if (a < b && ready) { init(); }");
    // style content kept verbatim too
    let style = doc.get_elements_by_tag_name("style").item(0).unwrap();
    assert_eq!(doc.text_content(style), ".a > .b { content: \"&\"; }");
    // serialization round-trips
    let s1 = doc.serialize();
    let (doc2, _) = Document::parse_html(s1.as_bytes());
    assert_eq!(doc2.serialize(), s1);
    // spot-check escaping in output
    assert!(s1.contains("Title &lt;here&gt;"));
    assert!(s1.contains("alt=\"a &quot;pic&quot;\""));
    assert!(s1.contains("<img src=\"pic.png\" alt=\"a &quot;pic&quot;\">"));
}

// -- serialization ---------------------------------------------------------------

#[test]
fn serialize_constructed_tree() {
    let mut doc = Document::new();
    let root = doc.root();
    let html = doc.create_element("html");
    let body = doc.create_element("body");
    let p = doc.create_element("p");
    doc.set_attribute(p, "class", "x");
    let txt = doc.create_text_node("a&b<c");
    doc.append_child(p, txt);
    let br = doc.create_element("br");
    doc.append_child(root, html);
    doc.append_child(html, body);
    doc.append_child(body, p);
    doc.append_child(body, br);
    assert_eq!(
        doc.serialize(),
        "<html><body><p class=\"x\">a&amp;b&lt;c</p><br></body></html>"
    );
    // subtree serialization
    assert_eq!(doc.serialize_node(p), "<p class=\"x\">a&amp;b&lt;c</p>");
    assert_eq!(doc.serialize_node(br), "<br>");
}

#[test]
fn serialize_raw_text_elements_verbatim() {
    // Raw-text elements must serialize their text children literally;
    // entity-escaping them corrupts the content on re-parse.
    let (doc, _) = Document::parse_html(
        b"<html><head><script>if (a<b && c) { x(); }</script><style>.a>.b{content:\"&\"}</style></head><body><p>a &lt; b</p></body></html>",
    );
    let script = doc.get_elements_by_tag_name("script").item(0).unwrap();
    assert_eq!(doc.text_content(script), "if (a<b && c) { x(); }");
    let s1 = doc.serialize();
    assert!(s1.contains("<script>if (a<b && c) { x(); }</script>"));
    assert!(s1.contains("<style>.a>.b{content:\"&\"}</style>"));
    // ordinary text is still escaped
    assert!(s1.contains("<p>a &lt; b</p>"));
    // full round-trip is stable
    let (doc2, _) = Document::parse_html(s1.as_bytes());
    assert_eq!(doc2.serialize(), s1);
}

#[test]
fn serialize_doctype_comment_pi() {
    let mut doc = Document::new();
    let root = doc.root();
    let dt = doc.alloc(NodeKind::DocumentType(DocumentTypeData {
        name: "html".into(),
        public_id: Some("-//W3C//DTD HTML 4.01//EN".into()),
        system_id: None,
    }));
    let c = doc.alloc(NodeKind::Comment(" hi ".into()));
    let pi = doc.alloc(NodeKind::ProcessingInstruction {
        target: "php".into(),
        data: "echo 1;".into(),
    });
    doc.append_child(root, dt);
    doc.append_child(root, c);
    doc.append_child(root, pi);
    assert_eq!(
        doc.serialize(),
        "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01//EN\"><!-- hi --><?php echo 1;?>"
    );
}

#[test]
fn kind_mut_allows_payload_edit() {
    let mut doc = Document::new();
    let t = doc.create_text_node("old");
    if let NodeKind::Text(s) = doc.kind_mut(t) {
        *s = Arc::from(format!("{s} new"));
    }
    assert_eq!(doc.text_content(t), "old new");
    let e = doc.create_element("div");
    if let NodeKind::Element(d) = doc.kind_mut(e) {
        d.attributes.push(Attribute::plain("id".into(), "z".into()));
    }
    assert_eq!(doc.get_attribute(e, "id"), Some("z"));
}

#[test]
fn descendants_document_order_and_ancestors() {
    let (doc, _) = Document::parse_html(b"<html><body><div><p>a</p><p>b</p></div></body></html>");
    let div = doc.get_elements_by_tag_name("div").item(0).unwrap();
    let tags: Vec<String> = doc
        .descendants(div)
        .into_iter()
        .map(|n| match doc.kind(n) {
            NodeKind::Element(d) => d.tag_name.to_string(),
            NodeKind::Text(t) => format!("text:{t}"),
            _ => "?".into(),
        })
        .collect();
    assert_eq!(tags, vec!["p", "text:a", "p", "text:b"]);
    let p2 = doc.get_elements_by_tag_name("p").item(1).unwrap();
    let anc_tags: Vec<String> = doc
        .ancestors(p2)
        .into_iter()
        .map(|n| match doc.kind(n) {
            NodeKind::Element(d) => d.tag_name.to_string(),
            NodeKind::Document => "#document".into(),
            _ => "?".into(),
        })
        .collect();
    assert_eq!(anc_tags, vec!["div", "body", "html", "#document"]);
}

#[test]
fn deep_tree_does_not_overflow_stack() {
    // 50k nested divs: every traversal here is iterative.
    let mut html = String::from("<html><body>");
    for _ in 0..50_000 {
        html.push_str("<div>");
    }
    html.push('x');
    for _ in 0..50_000 {
        html.push_str("</div>");
    }
    html.push_str("</body></html>");
    let (doc, _) = Document::parse_html(html.as_bytes());
    let divs = doc.get_elements_by_tag_name("div");
    assert_eq!(divs.len(), 50_000);
    let deepest = divs.item(divs.len() - 1).unwrap();
    assert_eq!(doc.text_content(deepest), "x");
    assert_eq!(doc.descendants(doc.root()).len(), 50_004); // 50k divs + html + head + body + text
    // ancestors() is iterative too: full chain from the deepest node.
    let anc = doc.ancestors(deepest);
    assert_eq!(anc.len(), 50_002); // 49_999 divs + body + html + #document
    assert_eq!(anc[0], divs.item(divs.len() - 2).unwrap()); // nearest first
    assert_eq!(*anc.last().unwrap(), doc.root());
    let s = doc.serialize();
    assert!(s.contains(">x</div>"));
    drop(doc);
    drop(s);
}

// -- mutation ---------------------------------------------------------------

#[test]
fn insert_before_orders_children() {
    let mut doc = Document::new();
    let root = doc.create_element("div");
    doc.append_child(doc.root(), root);
    let a = doc.create_element("div");
    let b = doc.create_element("div");
    let c = doc.create_element("div");
    doc.append_child(root, a);
    doc.append_child(root, c);
    doc.insert_before(root, b, Some(c));
    assert_eq!(doc.children(root), vec![a, b, c]);
    // None appends at the end.
    let d = doc.create_element("div");
    doc.insert_before(root, d, None);
    assert_eq!(doc.children(root), vec![a, b, c, d]);
    // Before the first child.
    let z = doc.create_element("div");
    doc.insert_before(root, z, Some(a));
    assert_eq!(doc.children(root), vec![z, a, b, c, d]);
    assert_eq!(doc.parent(z), Some(root));
    assert_eq!(doc.previous_sibling(a), Some(z));
    assert_eq!(doc.next_sibling(z), Some(a));
    // Sibling chain is consistent end to end.
    assert_eq!(doc.next_sibling(d), None);
    assert_eq!(doc.previous_sibling(z), None);
}

#[test]
fn insert_before_moves_node_from_old_parent() {
    let mut doc = Document::new();
    let p1 = doc.create_element("div");
    let p2 = doc.create_element("div");
    let kid = doc.create_element("span");
    let anchor = doc.create_element("span");
    doc.append_child(doc.root(), p1);
    doc.append_child(doc.root(), p2);
    doc.append_child(p1, kid);
    doc.append_child(p2, anchor);
    doc.insert_before(p2, kid, Some(anchor)); // moves kid from p1 to p2
    assert_eq!(doc.children(p1), vec![]);
    assert_eq!(doc.children(p2), vec![kid, anchor]);
    assert_eq!(doc.parent(kid), Some(p2));
}

#[test]
fn insert_before_self_reference_keeps_position() {
    let mut doc = Document::new();
    let root = doc.create_element("div");
    doc.append_child(doc.root(), root);
    let a = doc.create_element("div");
    let b = doc.create_element("div");
    let c = doc.create_element("div");
    doc.append_child(root, a);
    doc.append_child(root, b);
    doc.append_child(root, c);
    // Per the DOM algorithm the reference re-targets to the next sibling
    // after detaching, so the node lands back in the same slot.
    doc.insert_before(root, b, Some(b));
    assert_eq!(doc.children(root), vec![a, b, c]);
}

#[test]
fn insert_before_cycle_is_noop() {
    let mut doc = Document::new();
    let root = doc.root();
    let outer = doc.create_element("div");
    let inner = doc.create_element("div");
    doc.append_child(root, outer);
    doc.append_child(outer, inner);
    // Adopting an ancestor (or self) would create a cycle: no-op.
    doc.insert_before(inner, outer, None);
    doc.insert_before(outer, outer, None);
    assert_eq!(doc.children(root), vec![outer]);
    assert_eq!(doc.children(outer), vec![inner]);
    assert_eq!(doc.parent(outer), Some(root));
}

#[test]
fn insert_before_text_parent_is_noop() {
    let mut doc = Document::new();
    let t = doc.create_text_node("x");
    let el = doc.create_element("div");
    doc.insert_before(t, el, None); // no-op: text cannot have children
    assert_eq!(doc.parent(el), None);
}

#[test]
fn insert_before_foreign_reference_is_noop() {
    let mut doc = Document::new();
    let root = doc.root();
    let other = doc.create_element("div");
    let el = doc.create_element("span");
    // `other` is detached, not a child: fail-closed no-op, no panic.
    doc.insert_before(root, el, Some(other));
    assert_eq!(doc.children(root), Vec::new());
    assert_eq!(doc.parent(el), None);
}

#[test]
fn replace_child_swaps_in_place() {
    let mut doc = Document::new();
    let root = doc.create_element("div");
    doc.append_child(doc.root(), root);
    let a = doc.create_element("div");
    let b = doc.create_element("div");
    let c = doc.create_element("div");
    let n = doc.create_element("span");
    doc.append_child(root, a);
    doc.append_child(root, b);
    doc.append_child(root, c);
    doc.replace_child(root, n, b);
    assert_eq!(doc.children(root), vec![a, n, c]);
    // Old child is detached but alive.
    assert_eq!(doc.parent(b), None);
    assert!(matches!(doc.kind(b), NodeKind::Element(_)));
    assert_eq!(tag(&doc, b), "div");
    // Sibling links repaired.
    assert_eq!(doc.next_sibling(a), Some(n));
    assert_eq!(doc.previous_sibling(n), Some(a));
    assert_eq!(doc.next_sibling(n), Some(c));
    assert_eq!(doc.previous_sibling(c), Some(n));
}

#[test]
fn replace_child_at_ends() {
    let mut doc = Document::new();
    let root = doc.create_element("div");
    doc.append_child(doc.root(), root);
    let a = doc.create_element("div");
    let b = doc.create_element("div");
    let first = doc.create_element("span");
    let last = doc.create_element("span");
    doc.append_child(root, a);
    doc.append_child(root, b);
    doc.replace_child(root, first, a);
    doc.replace_child(root, last, b);
    assert_eq!(doc.children(root), vec![first, last]);
    assert_eq!(doc.previous_sibling(first), None);
    assert_eq!(doc.next_sibling(last), None);
}

#[test]
fn replace_child_moves_new_child() {
    let mut doc = Document::new();
    let root = doc.root();
    let p1 = doc.create_element("div");
    let p2 = doc.create_element("div");
    let old = doc.create_element("span");
    let fresh = doc.create_element("em");
    doc.append_child(root, p1);
    doc.append_child(root, p2);
    doc.append_child(p1, fresh);
    doc.append_child(p2, old);
    doc.replace_child(p2, fresh, old);
    assert_eq!(doc.children(p1), vec![]);
    assert_eq!(doc.children(p2), vec![fresh]);
    assert_eq!(doc.parent(old), None);
}

#[test]
fn replace_child_self_is_noop() {
    let mut doc = Document::new();
    let root = doc.create_element("div");
    doc.append_child(doc.root(), root);
    let a = doc.create_element("div");
    let b = doc.create_element("div");
    doc.append_child(root, a);
    doc.append_child(root, b);
    doc.replace_child(root, a, a);
    assert_eq!(doc.children(root), vec![a, b]);
}

#[test]
fn replace_child_cycle_is_noop() {
    let mut doc = Document::new();
    let root = doc.root();
    let outer = doc.create_element("div");
    let inner = doc.create_element("div");
    let other = doc.create_element("div");
    doc.append_child(root, outer);
    doc.append_child(outer, inner);
    doc.append_child(outer, other);
    // Replacing `inner` with its own ancestor would create a cycle: no-op.
    doc.replace_child(outer, outer, inner);
    doc.replace_child(outer, root, other);
    assert_eq!(doc.children(outer), vec![inner, other]);
}

#[test]
fn replace_child_old_not_child_is_noop() {
    let mut doc = Document::new();
    let root = doc.root();
    let a = doc.create_element("div");
    let stray = doc.create_element("span");
    let n = doc.create_element("em");
    doc.append_child(root, a);
    doc.replace_child(root, n, stray); // `stray` is not a child: no-op
    assert_eq!(doc.children(root), vec![a]);
    assert_eq!(doc.parent(n), None);
}

#[test]
fn replace_child_text_parent_is_noop() {
    let mut doc = Document::new();
    let t = doc.create_text_node("x");
    let a = doc.create_element("div");
    let b = doc.create_element("span");
    doc.replace_child(t, a, b); // no-op: text cannot have children
    assert_eq!(doc.parent(a), None);
}

#[test]
fn remove_detaches_but_keeps_node_alive() {
    let mut doc = Document::new();
    let root = doc.create_element("div");
    doc.append_child(doc.root(), root);
    let a = doc.create_element("div");
    let b = doc.create_element("div");
    let c = doc.create_element("div");
    doc.append_child(root, a);
    doc.append_child(root, b);
    doc.append_child(root, c);
    doc.remove(b);
    assert_eq!(doc.children(root), vec![a, c]);
    assert_eq!(doc.parent(b), None);
    // Node stays alive: kind, attributes, and re-insertion all work.
    assert_eq!(tag(&doc, b), "div");
    doc.set_attribute(b, "id", "revived");
    doc.append_child(root, b);
    assert_eq!(doc.children(root), vec![a, c, b]);
    assert_eq!(doc.get_attribute(b, "id"), Some("revived"));
}

#[test]
fn remove_keeps_subtree_intact() {
    let mut doc = Document::new();
    let root = doc.root();
    let outer = doc.create_element("div");
    let inner = doc.create_element("span");
    let t = doc.create_text_node("deep");
    doc.append_child(root, outer);
    doc.append_child(outer, inner);
    doc.append_child(inner, t);
    doc.remove(outer);
    assert_eq!(doc.children(root), vec![]);
    assert_eq!(doc.children(outer), vec![inner]);
    assert_eq!(doc.text_content(outer), "deep");
}

#[test]
fn remove_detached_and_root_are_noops() {
    let mut doc = Document::new();
    let stray = doc.create_element("div");
    doc.remove(stray); // detached: no-op
    assert_eq!(doc.parent(stray), None);
    doc.remove(doc.root()); // root has no parent: no-op
    assert_eq!(doc.parent(doc.root()), None);
}

#[test]
fn clone_node_shallow_copies_payload_only() {
    let mut doc = Document::new();
    let el = doc.create_element("div");
    doc.set_attribute(el, "class", "a b");
    doc.set_attribute(el, "id", "x");
    let t = doc.create_text_node("hi");
    doc.append_child(el, t);

    let shallow = doc.clone_node(el, false);
    assert_ne!(shallow, el);
    assert_eq!(doc.parent(shallow), None); // detached
    assert_eq!(doc.children(shallow), vec![]);
    assert_eq!(doc.get_attribute(shallow, "class"), Some("a b"));
    assert_eq!(doc.get_attribute(shallow, "id"), Some("x"));
    // Original untouched.
    assert_eq!(doc.children(el), vec![t]);

    let tclone = doc.clone_node(t, false);
    assert!(matches!(doc.kind(tclone), NodeKind::Text(s) if s.as_ref() == "hi"));
    assert_eq!(doc.parent(tclone), None);
}

#[test]
fn clone_node_deep_copies_subtree() {
    let (mut doc, _) = Document::parse_html(b"<div id=\"o\"><p>one<!--note--></p>two</div>");
    let outer = doc
        .get_elements_by_tag_name("div")
        .into_iter()
        .next()
        .expect("div");
    let mid = doc
        .get_elements_by_tag_name("p")
        .into_iter()
        .next()
        .expect("p");
    // Detach `outer` from the parsed document so we can assert the clone
    // of an attached node comes out detached.
    doc.remove(outer);

    let deep = doc.clone_node(outer, true);
    assert_ne!(deep, outer);
    assert_eq!(doc.parent(deep), None); // detached even though `outer` is attached
    let kids = doc.children(deep);
    assert_eq!(kids.len(), 2);
    let (dmid, dt2) = (kids[0], kids[1]);
    assert_eq!(tag(&doc, dmid), "p");
    assert!(matches!(doc.kind(dt2), NodeKind::Text(s) if s.as_ref() == "two"));
    assert_eq!(doc.get_attribute(deep, "id"), Some("o"));
    let gkids = doc.children(dmid);
    assert_eq!(gkids.len(), 2);
    assert!(matches!(doc.kind(gkids[0]), NodeKind::Text(s) if s.as_ref() == "one"));
    assert!(matches!(doc.kind(gkids[1]), NodeKind::Comment(s) if s == "note"));
    // Structure matches the original exactly.
    assert_eq!(doc.serialize_node(deep), doc.serialize_node(outer));
    // Clones are independent: mutating the clone leaves the original alone.
    doc.set_attribute(deep, "id", "changed");
    doc.set_attribute(dmid, "data-x", "1");
    assert_eq!(doc.get_attribute(outer, "id"), Some("o"));
    assert_eq!(doc.get_attribute(mid, "data-x"), None);
    assert_eq!(
        doc.serialize_node(deep),
        "<div id=\"changed\"><p data-x=\"1\">one<!--note--></p>two</div>"
    );
}

#[test]
fn clone_node_deep_is_iterative() {
    // 20k-deep chain: the clone walk must not recurse.
    let mut doc = Document::new();
    let mut parent = doc.root();
    for _ in 0..20_000 {
        let d = doc.create_element("div");
        doc.append_child(parent, d);
        parent = d;
    }
    let deep = doc.clone_node(doc.root(), true);
    assert_eq!(doc.descendants(deep).len(), 20_000);
    assert_eq!(doc.parent(deep), None);
}

#[test]
fn deep_chain_build_stays_linear() {
    // Regression: `append_child` used to do ~3 O(depth) parent-chain
    // walks per call (`is_attached`, `in_template_contents` via
    // `mark_dirty` / `note_attach_if_attached`), making a 20k-deep chain
    // build quadratic. The flags are now maintained incrementally in the
    // link primitives, so the whole build is O(n). The bound is generous
    // on purpose (slow CI): the new code finishes in well under a
    // second; the old walks needed tens of seconds in debug builds.
    let start = std::time::Instant::now();
    let mut doc = Document::new();
    let mut parent = doc.root();
    let mut deepest = parent;
    for _ in 0..20_000 {
        let d = doc.create_element("div");
        doc.append_child(parent, d);
        parent = d;
        deepest = d;
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs() < 30,
        "20k-deep chain build took {elapsed:?}; append_child looks O(depth) again"
    );
    // The flags are correct on the deep chain too.
    assert!(doc.is_attached(deepest));
    assert!(!doc.in_template_contents(deepest));
    let fresh = doc.create_element("span");
    assert!(!doc.is_attached(fresh));
    assert_eq!(doc.descendants(doc.root()).len(), 20_000);
}

#[test]
fn attached_flags_track_link_and_unlink() {
    let mut doc = Document::new();
    let root = doc.root();
    assert!(doc.is_attached(root));
    // `wrap` is the document's single element child (document-validity
    // allows only one); everything else nests under it.
    let wrap = doc.create_element("div");
    doc.append_child(root, wrap);
    let a = doc.create_element("div");
    assert!(!doc.is_attached(a)); // fresh alloc: detached
    let b = doc.create_element("span");
    doc.append_child(a, b); // linking detached under detached stays detached
    assert!(!doc.is_attached(a));
    assert!(!doc.is_attached(b));
    doc.append_child(wrap, a);
    assert!(doc.is_attached(a));
    assert!(doc.is_attached(b));
    // Moving an attached node keeps the subtree attached.
    let c = doc.create_element("p");
    doc.append_child(wrap, c);
    doc.append_child(c, a);
    assert!(doc.is_attached(a));
    assert!(doc.is_attached(b));
    assert!(doc.is_attached(c));
    // Detach clears the whole subtree.
    doc.remove_child(c, a);
    assert!(!doc.is_attached(a));
    assert!(!doc.is_attached(b));
    assert!(doc.is_attached(c));
    // Re-attach restores the flags.
    doc.append_child(wrap, a);
    assert!(doc.is_attached(a));
    assert!(doc.is_attached(b));
    // insert_before maintains them as well.
    let d = doc.create_element("em");
    doc.insert_before(wrap, d, Some(a));
    assert!(doc.is_attached(d));
    doc.remove(d);
    assert!(!doc.is_attached(d));
}

#[test]
fn template_contents_flags_match_walk_semantics() {
    let (mut doc, _) = Document::parse_html(
        b"<template><div id=\"in-tpl\"><span></span></div></template><div id=\"out\"></div>",
    );
    let template = doc
        .get_elements_by_tag_name("template")
        .item(0)
        .expect("template");
    let frag = doc.template_contents(template).expect("contents fragment");
    let inner = doc.first_child(frag).expect("inner div");
    let span = doc.first_child(inner).expect("span");
    let out = doc.query_selector("#out").unwrap().expect("outer div");

    assert!(doc.is_attached(template));
    assert!(doc.is_attached(span));
    assert!(!doc.in_template_contents(template));
    assert!(doc.in_template_contents(frag));
    assert!(doc.in_template_contents(inner));
    assert!(doc.in_template_contents(span));
    assert!(!doc.in_template_contents(out));
    // Template contents stay out of reach of the id index, via the flags.
    assert_eq!(doc.get_element_by_id("in-tpl"), None);
    assert_eq!(doc.get_element_by_id("out"), Some(out));

    let tpl_parent = doc.parent(template).expect("template parent");
    // Detaching the template clears attachment for the whole subtree but
    // keeps the *relative* template-contents flags (the old parent-chain
    // walk still crossed the template-to-fragment edge inside the
    // detached subtree).
    doc.remove(template);
    assert!(!doc.is_attached(template));
    assert!(!doc.is_attached(span));
    assert!(!doc.in_template_contents(template));
    assert!(doc.in_template_contents(frag));
    assert!(doc.in_template_contents(span));

    // Re-attaching restores both flags exactly.
    doc.append_child(tpl_parent, template);
    assert!(doc.is_attached(span));
    assert!(!doc.in_template_contents(template));
    assert!(doc.in_template_contents(span));
    assert_eq!(doc.get_element_by_id("in-tpl"), None);
}

#[test]
fn dirty_tracking_uses_attached_flags() {
    let mut doc = Document::new();
    let root = doc.root();
    // Mutations on detached subtrees record nothing...
    let d = doc.create_element("div");
    doc.set_attribute(d, "id", "x");
    assert!(doc.take_dirty().is_empty());
    // ...but the same mutation once attached does.
    doc.append_child(root, d);
    assert!(!doc.take_dirty().is_empty());
    doc.set_attribute(d, "class", "y");
    let report = doc.take_dirty();
    assert!(report.style);
    assert_eq!(report.subtrees, vec![d]);
    // Detaching silences it again.
    doc.remove_child(root, d);
    let _ = doc.take_dirty();
    doc.set_attribute(d, "class", "z");
    assert!(doc.take_dirty().is_empty());
}

#[test]
fn dirty_report_records_style_localized_attribute_writes() {
    let (mut doc, _) = Document::parse_html(b"<div id=\"a\"><p>hi</p></div>");
    let div = doc.get_element_by_id("a").expect("div");
    let p = doc.first_child(div).expect("p");
    let text = doc.first_child(p).expect("text");
    assert!(doc.take_dirty().is_empty());

    // Attribute write on an attached element: style dirt plus a
    // subtree-localized restyle root.
    doc.set_attribute(div, "class", "x");
    let report = doc.take_dirty();
    assert!(report.style);
    assert_eq!(report.subtrees, vec![div]);
    assert_eq!(report.style_subtrees, vec![div]);

    // Removal records the same way.
    doc.remove_attribute(div, "class");
    let report = doc.take_dirty();
    assert!(report.style);
    assert_eq!(report.style_subtrees, vec![div]);

    // A no-op write (same value) records nothing.
    doc.set_attribute(div, "id", "a");
    assert!(doc.take_dirty().is_empty());

    // Plain text edit: layout/paint dirt only, no style localization.
    doc.set_text_content(text, "bye");
    let report = doc.take_dirty();
    assert!(!report.style);
    assert!(report.style_subtrees.is_empty());

    // Structural change: style dirt, but deliberately NOT localized
    // (fail-closed: the old full-path behavior).
    let span = doc.create_element("span");
    doc.append_child(div, span);
    let report = doc.take_dirty();
    assert!(report.style);
    assert!(report.style_subtrees.is_empty());

    // Writes on a shadow host or inside its shadow tree are not
    // localized: shadow stylesheets are outside the embedder's audited
    // set.
    let host = doc.create_element("section");
    doc.append_child(div, host);
    let shadow = doc
        .attach_shadow_root(host, ShadowRootMode::Open)
        .expect("shadow root");
    let inner = doc.create_element("b");
    doc.append_child(shadow, inner);
    let _ = doc.take_dirty();
    doc.set_attribute(host, "class", "h");
    assert!(doc.take_dirty().style_subtrees.is_empty());
    doc.set_attribute(inner, "class", "i");
    let report = doc.take_dirty();
    assert!(report.style);
    assert!(report.style_subtrees.is_empty());
}

#[test]
fn document_stays_send_sync() {
    // The O(1) flags and the selector scratch must not smuggle interior
    // mutability into `Document`: it stays shareable across threads.
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Document>();
}

#[test]
fn selector_scratch_variants_match_plain_queries() {
    let (doc, _) = Document::parse_html(
        b"<div id=\"a\" class=\"x\"><p class=\"y\">one</p><p class=\"y\">two</p></div><span class=\"x\">three</span>",
    );
    let mut scratch = SelectorScratch::new();
    for sel in [
        "p.y",
        "div > p",
        ".x",
        "#a .y",
        "div, span",
        "p + p",
        "div ~ span",
        "div p",
    ] {
        assert_eq!(
            doc.query_selector_all_with_scratch(sel, &mut scratch)
                .unwrap(),
            doc.query_selector_all(sel).unwrap(),
            "query_selector_all mismatch for {sel}"
        );
        assert_eq!(
            doc.query_selector_with_scratch(sel, &mut scratch).unwrap(),
            doc.query_selector(sel).unwrap(),
            "query_selector mismatch for {sel}"
        );
    }
    let p = doc.query_selector("p.y").unwrap().expect("p");
    assert_eq!(
        doc.matches_with_scratch(p, "p.y", &mut scratch).unwrap(),
        doc.matches(p, "p.y").unwrap()
    );
    assert_eq!(
        doc.closest_with_scratch(p, "div", &mut scratch).unwrap(),
        doc.closest(p, "div").unwrap()
    );
    // Parse errors are cached too, and stay errors on repeat queries.
    let err1 = doc
        .query_selector_all_with_scratch("div > > p", &mut scratch)
        .unwrap_err();
    let err2 = doc.query_selector_all("div > > p").unwrap_err();
    assert_eq!(err1.message(), err2.message());
    assert!(
        doc.matches_with_scratch(p, "div > > p", &mut scratch)
            .is_err()
    );
    // One scratch survives many queries and still agrees with fresh ones.
    for _ in 0..100 {
        assert_eq!(
            doc.query_selector_all_with_scratch("p.y", &mut scratch)
                .unwrap()
                .len(),
            2
        );
    }
    assert_eq!(
        doc.query_selector_all_with_scratch("div p", &mut scratch)
            .unwrap(),
        doc.query_selector_all("div p").unwrap()
    );
}

#[test]
fn set_text_content_replaces_children() {
    let mut doc = Document::new();
    let root = doc.root();
    let div = doc.create_element("div");
    let p = doc.create_element("p");
    let t = doc.create_text_node("old");
    doc.append_child(root, div);
    doc.append_child(div, p);
    doc.append_child(div, t);
    doc.set_text_content(div, "new");
    let kids = doc.children(div);
    assert_eq!(kids.len(), 1);
    assert!(matches!(doc.kind(kids[0]), NodeKind::Text(s) if s.as_ref() == "new"));
    assert_eq!(doc.text_content(div), "new");
    // The evicted subtree stays alive (detached), like DOM replace-all.
    assert_eq!(doc.parent(p), None);
    assert_eq!(doc.parent(t), None);
}

#[test]
fn set_text_content_empty_string_clears() {
    let mut doc = Document::new();
    let div = doc.create_element("div");
    let t = doc.create_text_node("old");
    doc.append_child(doc.root(), div);
    doc.append_child(div, t);
    doc.set_text_content(div, "");
    assert_eq!(doc.children(div), vec![]);
    assert_eq!(doc.text_content(div), "");
}

#[test]
fn set_text_content_on_character_data() {
    let mut doc = Document::new();
    let t = doc.create_text_node("a");
    doc.set_text_content(t, "b");
    assert!(matches!(doc.kind(t), NodeKind::Text(s) if s.as_ref() == "b"));

    let (mut doc2, _) = Document::parse_html(b"<!--a-->");
    let c = doc2
        .children(doc2.root())
        .into_iter()
        .find(|&n| matches!(doc2.kind(n), NodeKind::Comment(_)))
        .expect("comment node");
    doc2.set_text_content(c, "b");
    assert!(matches!(doc2.kind(c), NodeKind::Comment(s) if s == "b"));

    // ProcessingInstruction has no constructor yet; set the payload via
    // kind_mut and verify set_text_content rewrites the data in place.
    let pi = doc.create_text_node("x");
    *doc.kind_mut(pi) = NodeKind::ProcessingInstruction {
        target: "php".to_string(),
        data: "old".to_string(),
    };
    doc.set_text_content(pi, "new");
    assert!(matches!(
        doc.kind(pi),
        NodeKind::ProcessingInstruction { target, data }
            if target == "php" && data == "new"
    ));
}

#[test]
fn set_text_content_on_document_and_doctype_is_noop() {
    let mut doc = Document::new();
    let root = doc.root();
    let el = doc.create_element("div");
    doc.append_child(root, el);
    // Document: no-op (matches the DOM textContent setter).
    doc.set_text_content(root, "ignored");
    assert_eq!(doc.children(root), vec![el]);

    let (mut doc2, _) = Document::parse_html(b"<!DOCTYPE html><p>x</p>");
    let doctype = doc2
        .children(doc2.root())
        .into_iter()
        .find(|&n| matches!(doc2.kind(n), NodeKind::DocumentType(_)))
        .expect("doctype node");
    doc2.set_text_content(doctype, "ignored");
    assert!(matches!(
        doc2.kind(doctype),
        NodeKind::DocumentType(d) if d.name == "html"
    ));
}

// -- events ------------------------------------------------------------------

#[test]
fn event_fields_and_prevent_default() {
    let mut e = Event::new("click", true, true);
    assert_eq!(e.event_type, "click");
    assert!(e.bubbles);
    assert!(e.cancelable);
    assert!(!e.default_prevented);
    e.prevent_default();
    assert!(e.default_prevented);

    // Per DOM preventDefault(): a non-cancelable event ignores the call;
    // defaultPrevented stays false.
    let mut nc = Event::new("mousemove", false, false);
    nc.prevent_default();
    assert!(!nc.default_prevented);
    assert!(!nc.bubbles);
}

#[test]
fn propagation_path_is_target_first_to_root() {
    let mut doc = Document::new();
    let root = doc.root();
    let html = doc.create_element("html");
    let body = doc.create_element("body");
    let div = doc.create_element("div");
    doc.append_child(root, html);
    doc.append_child(html, body);
    doc.append_child(body, div);
    assert_eq!(doc.propagation_path(div), vec![div, body, html, root]);
    // Capture phase walks this in reverse.
    let capture: Vec<NodeId> = doc.propagation_path(div).into_iter().rev().collect();
    assert_eq!(capture, vec![root, html, body, div]);
    // Detached node: single-element path.
    let stray = doc.create_element("span");
    assert_eq!(doc.propagation_path(stray), vec![stray]);
    // Root: just itself.
    assert_eq!(doc.propagation_path(root), vec![root]);
}

// -- id index ---------------------------------------------------------------

#[test]
fn id_index_tracks_set_and_remove_attribute() {
    let mut doc = Document::new();
    let div = doc.create_element("div");
    doc.append_child(doc.root(), div);
    assert_eq!(doc.get_element_by_id("a"), None);

    doc.set_attribute(div, "id", "a");
    assert_eq!(doc.get_element_by_id("a"), Some(div));

    // Same value: no-op, still found.
    doc.set_attribute(div, "id", "a");
    assert_eq!(doc.get_element_by_id("a"), Some(div));

    // Rename: old id gone, new id found.
    doc.set_attribute(div, "id", "b");
    assert_eq!(doc.get_element_by_id("a"), None);
    assert_eq!(doc.get_element_by_id("b"), Some(div));

    // Case-insensitive attribute name on HTML elements.
    doc.set_attribute(div, "ID", "c");
    assert_eq!(doc.get_element_by_id("b"), None);
    assert_eq!(doc.get_element_by_id("c"), Some(div));

    doc.remove_attribute(div, "id");
    assert_eq!(doc.get_element_by_id("c"), None);

    // Removing a missing attribute is a no-op.
    doc.remove_attribute(div, "id");
    assert_eq!(doc.get_element_by_id("c"), None);

    // set_attribute on a non-element is a no-op (existing behavior).
    let t = doc.create_text_node("x");
    doc.set_attribute(t, "id", "t");
    assert_eq!(doc.get_element_by_id("t"), None);
}

#[test]
fn id_index_ignores_detached_subtrees() {
    let mut doc = Document::new();
    let root = doc.create_element("div");
    doc.append_child(doc.root(), root);
    // Id set while detached: invisible until attached.
    let div = doc.create_element("div");
    doc.set_attribute(div, "id", "a");
    assert_eq!(doc.get_element_by_id("a"), None);

    doc.append_child(root, div);
    assert_eq!(doc.get_element_by_id("a"), Some(div));

    doc.remove(div);
    assert_eq!(doc.get_element_by_id("a"), None);

    // Rename while detached: still invisible, new id picked up on attach.
    doc.set_attribute(div, "id", "b");
    assert_eq!(doc.get_element_by_id("b"), None);
    doc.append_child(root, div);
    assert_eq!(doc.get_element_by_id("a"), None);
    assert_eq!(doc.get_element_by_id("b"), Some(div));

    // Nested detached subtree: ids inside are picked up on attach.
    let outer = doc.create_element("section");
    let inner = doc.create_element("p");
    doc.set_attribute(inner, "id", "deep");
    doc.append_child(outer, inner); // outer detached: no indexing
    assert_eq!(doc.get_element_by_id("deep"), None);
    doc.append_child(root, outer);
    assert_eq!(doc.get_element_by_id("deep"), Some(inner));
    doc.remove_child(root, outer);
    assert_eq!(doc.get_element_by_id("deep"), None);
}

#[test]
fn id_index_duplicate_ids_first_in_document_order() {
    let mut doc = Document::new();
    let root = doc.create_element("div");
    doc.append_child(doc.root(), root);
    let d1 = doc.create_element("div");
    doc.set_attribute(d1, "id", "x");
    let d2 = doc.create_element("div");
    doc.set_attribute(d2, "id", "x");
    doc.append_child(root, d1);
    doc.append_child(root, d2);
    // First in document order wins.
    assert_eq!(doc.get_element_by_id("x"), Some(d1));

    // Removing the winner promotes the next in document order.
    doc.remove(d1);
    assert_eq!(doc.get_element_by_id("x"), Some(d2));

    // Moving d2 before a re-attached d1 flips the winner.
    doc.append_child(root, d1);
    assert_eq!(doc.get_element_by_id("x"), Some(d2));
    doc.insert_before(root, d2, Some(d1));
    assert_eq!(doc.get_element_by_id("x"), Some(d2));
    doc.insert_before(root, d1, Some(d2));
    assert_eq!(doc.get_element_by_id("x"), Some(d1));

    // Renaming the winner away promotes the other duplicate.
    doc.set_attribute(d1, "id", "y");
    assert_eq!(doc.get_element_by_id("x"), Some(d2));
    assert_eq!(doc.get_element_by_id("y"), Some(d1));
}

#[test]
fn id_index_maintained_across_structural_ops() {
    let mut doc = Document::new();
    let root = doc.root();
    let a = doc.create_element("div");
    doc.set_attribute(a, "id", "a");
    let b = doc.create_element("div");
    doc.set_attribute(b, "id", "b");
    doc.append_child(root, a);
    doc.append_child(root, b);

    // replace_child swaps the indexed nodes.
    let c = doc.create_element("span");
    doc.set_attribute(c, "id", "c");
    doc.replace_child(root, c, a);
    assert_eq!(doc.get_element_by_id("a"), None);
    assert_eq!(doc.get_element_by_id("c"), Some(c));

    // Move within the document keeps the index correct.
    doc.append_child(c, b); // moves b under c
    assert_eq!(doc.get_element_by_id("b"), Some(b));
    assert_eq!(doc.parent(b), Some(c));

    // clone_node produces a detached clone: not indexed, original intact.
    let b2 = doc.clone_node(b, true);
    assert_eq!(doc.get_element_by_id("b"), Some(b));
    assert_eq!(doc.parent(b2), None);

    // set_text_content drops ids of cleared children.
    let p = doc.create_element("p");
    doc.set_attribute(p, "id", "p");
    doc.append_child(c, p);
    assert_eq!(doc.get_element_by_id("p"), Some(p));
    doc.set_text_content(c, "flat");
    assert_eq!(doc.get_element_by_id("p"), None);
    assert_eq!(doc.get_element_by_id("b"), None);
    assert_eq!(doc.get_element_by_id("c"), Some(c));
    assert_eq!(doc.text_content(c), "flat");
}

#[test]
fn id_index_parse_html() {
    let (doc, _) = Document::parse_html(b"<div id=a><p id=b>x</p></div><span id=c></span>");
    let div = doc.get_element_by_id("a").unwrap();
    assert_eq!(tag(&doc, div), "div");
    let p = doc.get_element_by_id("b").unwrap();
    assert_eq!(tag(&doc, p), "p");
    assert_eq!(
        doc.get_element_by_id("c").map(|n| tag(&doc, n)),
        Some("span")
    );
    assert_eq!(doc.get_element_by_id("missing"), None);
    // Duplicate ids in parsed input: first in document order wins.
    let (doc2, _) = Document::parse_html(b"<i id=d>1</i><i id=d>2</i>");
    let first = doc2.get_element_by_id("d").unwrap();
    assert_eq!(doc2.text_content(first), "1");
}

#[test]
fn id_index_is_per_document() {
    let mut da = Document::new();
    let mut db = Document::new();
    let a = da.create_element("div");
    da.set_attribute(a, "id", "x");
    da.append_child(da.root(), a);
    let b = db.create_element("div");
    db.set_attribute(b, "id", "x");
    db.append_child(db.root(), b);
    // Same id string, separate indexes.
    assert_eq!(da.get_element_by_id("x"), Some(a));
    assert_eq!(db.get_element_by_id("x"), Some(b));
    assert_ne!(da.get_element_by_id("x"), db.get_element_by_id("x"));
}

// -- innerHTML ----------------------------------------------------------------

#[test]
fn set_inner_html_replaces_children() {
    let mut doc = Document::new();
    let div = doc.create_element("div");
    doc.set_attribute(div, "id", "old");
    let stale = doc.create_element("p");
    doc.append_child(div, stale);
    doc.append_child(doc.root(), div);

    doc.set_inner_html(div, "<p>one</p>two<p>three</p>");
    let kids = doc.children(div);
    assert_eq!(kids.len(), 3);
    assert_eq!(tag(&doc, kids[0]), "p");
    assert!(matches!(doc.kind(kids[1]), NodeKind::Text(_)));
    assert_eq!(tag(&doc, kids[2]), "p");
    assert_eq!(doc.text_content(div), "onetwothree");
    // Old children are detached but alive.
    assert_eq!(doc.parent(stale), None);
    assert!(matches!(doc.kind(stale), NodeKind::Element(_)));
    // The element itself is untouched.
    assert_eq!(doc.get_attribute(div, "id"), Some("old"));
    assert_eq!(doc.get_element_by_id("old"), Some(div));
}

#[test]
fn set_inner_html_round_trip() {
    // Direct-sink fragment parse: no intermediate owned tree, same tree
    // as a fresh parse of the fragment.
    let mut doc = Document::new();
    let div = doc.create_element("div");
    doc.append_child(doc.root(), div);
    doc.set_inner_html(div, "<p>a<b>c</b></p>");
    assert_eq!(doc.serialize_node(div), "<div><p>a<b>c</b></p></div>");
    let kids = doc.children(div);
    assert_eq!(kids.len(), 1);
    assert_eq!(tag(&doc, kids[0]), "p");
    assert_eq!(doc.text_content(div), "ac");

    // Ids in the new subtree are indexed, with document-order duplicate
    // resolution against ids already present elsewhere in the document.
    let target = doc.create_element("section");
    doc.append_child(div, target);
    let other_wrap = doc.create_element("aside");
    doc.append_child(div, other_wrap);
    let other = doc.create_element("span");
    doc.set_attribute(other, "id", "dup");
    doc.append_child(other_wrap, other);
    // The new element is earlier in document order, so it wins the id.
    doc.set_inner_html(target, "<p id=\"dup\">new</p>");
    let p = doc.children(target)[0];
    assert_eq!(tag(&doc, p), "p");
    assert_eq!(doc.get_element_by_id("dup"), Some(p));
    // A pre-existing earlier duplicate keeps winning over a new later
    // one, while genuinely new ids are indexed.
    doc.set_inner_html(other_wrap, "<i id=\"dup\">x</i><b id=\"fresh\">y</b>");
    assert_eq!(doc.get_element_by_id("dup"), Some(p));
    assert_eq!(
        doc.get_element_by_id("fresh").map(|n| tag(&doc, n)),
        Some("b")
    );
}

#[test]
fn set_inner_html_fragment_parsing_rules() {
    let mut doc = Document::new();
    let root = doc.create_element("div");
    doc.append_child(doc.root(), root);
    let div = doc.create_element("div");
    doc.append_child(root, div);

    // Unclosed elements auto-close like a fragment parse.
    doc.set_inner_html(div, "<p>one<p>two");
    let kids = doc.children(div);
    assert_eq!(kids.len(), 2);
    assert!(kids.iter().all(|&k| tag(&doc, k) == "p"));
    assert_eq!(doc.serialize_node(div), "<div><p>one</p><p>two</p></div>");

    // Entities decode, attributes survive, comments preserved.
    doc.set_inner_html(div, "<a href=\"u&amp;v\" id=lnk>x</a><!--c-->");
    assert_eq!(
        doc.serialize_node(div),
        "<div><a href=\"u&amp;v\" id=\"lnk\">x</a><!--c--></div>"
    );
    assert_eq!(
        doc.get_element_by_id("lnk").map(|n| tag(&doc, n)),
        Some("a")
    );

    // Context-aware fragment parsing: table parts outside a table are
    // dropped, but stay put when the context element is a table.
    doc.set_inner_html(div, "<td>x</td>");
    assert_eq!(doc.text_content(div), "x");
    assert!(doc.get_elements_by_tag_name("td").is_empty());
    let table = doc.create_element("table");
    doc.append_child(root, table);
    doc.set_inner_html(table, "<tr><td>y</td></tr>");
    assert_eq!(doc.get_elements_by_tag_name("td").len(), 1);
    assert_eq!(doc.text_content(table), "y");

    // Nested markup and foreign content.
    doc.set_inner_html(
        div,
        "<div><span>de<i>ep</i></span></div><svg><circle/></svg>",
    );
    assert_eq!(doc.get_elements_by_tag_name("circle").len(), 1);
    assert_eq!(doc.text_content(div), "deep");
}

#[test]
fn set_inner_html_empty_clears_and_reindexes() {
    let mut doc = Document::new();
    let div = doc.create_element("div");
    doc.append_child(doc.root(), div);
    doc.set_inner_html(div, "<p id=gone>x</p>");
    assert_eq!(
        doc.get_element_by_id("gone").map(|n| tag(&doc, n)),
        Some("p")
    );

    doc.set_inner_html(div, "");
    assert!(doc.children(div).is_empty());
    assert_eq!(doc.get_element_by_id("gone"), None);

    // Replacing swaps the indexed ids.
    doc.set_inner_html(div, "<b id=new>y</b>");
    assert_eq!(doc.get_element_by_id("gone"), None);
    assert_eq!(
        doc.get_element_by_id("new").map(|n| tag(&doc, n)),
        Some("b")
    );
}

#[test]
fn set_inner_html_template_fragment_convention() {
    let mut doc = Document::new();
    let div = doc.create_element("div");
    doc.append_child(doc.root(), div);
    doc.set_inner_html(div, "<template><p id=t>in</p></template>");
    let template = doc.get_elements_by_tag_name("template").item(0).unwrap();
    assert!(doc.children(template).is_empty());
    let frag = doc.template_contents(template).unwrap();
    assert!(matches!(doc.kind(frag), NodeKind::DocumentFragment));
    // Ids inside template contents are not indexed.
    assert_eq!(doc.get_element_by_id("t"), None);
    assert_eq!(
        doc.serialize_node(div),
        "<div><template><p id=\"t\">in</p></template></div>"
    );
}

#[test]
fn set_inner_html_template_context_uses_contents_fragment() {
    let mut doc = Document::new();
    let tmpl = doc.create_element("template");
    doc.append_child(doc.root(), tmpl);
    doc.set_inner_html(tmpl, "<p id=t>hi</p>tail");
    // Parsed nodes land in a DocumentFragment reachable via
    // `template_contents`, not as traversal-visible children of the
    // template element.
    assert!(doc.children(tmpl).is_empty());
    let frag = doc.template_contents(tmpl).unwrap();
    assert!(matches!(doc.kind(frag), NodeKind::DocumentFragment));
    let inner = doc.children(frag);
    assert_eq!(inner.len(), 2);
    assert_eq!(tag(&doc, inner[0]), "p");
    assert!(matches!(doc.kind(inner[1]), NodeKind::Text(_)));
    // Ids inside the contents fragment are not indexed.
    assert_eq!(doc.get_element_by_id("t"), None);
    assert_eq!(
        doc.serialize_node(tmpl),
        "<template><p id=\"t\">hi</p>tail</template>"
    );
}

#[test]
fn set_inner_html_template_clears_and_refills_contents_fragment() {
    // Build a template with a pre-existing contents fragment via parsing.
    let (mut doc, _) = Document::parse_html(b"<template><p id=old>x</p></template>");
    let tmpl = doc.get_elements_by_tag_name("template").item(0).unwrap();
    let frag_before = doc.template_contents(tmpl).unwrap();
    assert!(matches!(doc.kind(frag_before), NodeKind::DocumentFragment));
    let old_p = doc.children(frag_before)[0];
    // Template contents are out of reach of the id index.
    assert_eq!(doc.get_element_by_id("old"), None);

    doc.set_inner_html(tmpl, "<div id=new>y</div>");
    // The same fragment node is cleared and refilled, never detached.
    assert_eq!(doc.template_contents(tmpl), Some(frag_before));
    let inner = doc.children(frag_before);
    assert_eq!(inner.len(), 1);
    assert_eq!(tag(&doc, inner[0]), "div");
    // The old contents are detached but alive; the id index stays empty.
    assert_eq!(doc.parent(old_p), None);
    assert_eq!(doc.get_element_by_id("old"), None);
    assert_eq!(doc.get_element_by_id("new"), None);
    assert_eq!(
        doc.serialize_node(tmpl),
        "<template><div id=\"new\">y</div></template>"
    );
}

#[test]
fn set_inner_html_nested_template() {
    let mut doc = Document::new();
    let tmpl = doc.create_element("template");
    doc.append_child(doc.root(), tmpl);
    doc.set_inner_html(tmpl, "<template><p>deep</p></template><b>out</b>");
    let outer_frag = doc.template_contents(tmpl).unwrap();
    assert!(matches!(doc.kind(outer_frag), NodeKind::DocumentFragment));
    let inner = doc.children(outer_frag);
    assert_eq!(inner.len(), 2);
    assert_eq!(tag(&doc, inner[0]), "template");
    assert_eq!(tag(&doc, inner[1]), "b");
    // The nested template got its own contents fragment.
    let nested_frag = doc.template_contents(inner[0]).unwrap();
    assert!(matches!(doc.kind(nested_frag), NodeKind::DocumentFragment));
    let deep = doc.children(nested_frag);
    assert_eq!(deep.len(), 1);
    assert_eq!(tag(&doc, deep[0]), "p");
    // Template contents are out of reach of text collection; the
    // fragment itself still reads fine (nested template contents stay
    // out of reach too).
    assert_eq!(doc.text_content(tmpl), "");
    assert_eq!(doc.text_content(outer_frag), "out");
    assert_eq!(doc.text_content(nested_frag), "deep");
}

#[test]
fn set_inner_html_passes_context_attributes() {
    let mut doc = Document::new();
    let ax = doc.create_element_ns(Namespace::MathMl, "annotation-xml");
    doc.set_attribute(ax, "encoding", "text/html");
    doc.append_child(doc.root(), ax);
    doc.set_inner_html(ax, "<div>x</div>");
    let kids = doc.children(ax);
    assert_eq!(kids.len(), 1);
    // With the encoding attribute visible to the parser, annotation-xml
    // is an HTML integration point, so the div parses in the HTML
    // namespace. (When the context attributes were dropped, the div
    // parsed as MathML content instead.)
    match doc.kind(kids[0]) {
        NodeKind::Element(d) => assert_eq!(d.namespace, Namespace::Html),
        other => panic!("expected element, got {other:?}"),
    }
    assert_eq!(
        doc.serialize_node(ax),
        "<annotation-xml encoding=\"text/html\"><div>x</div></annotation-xml>"
    );
}

#[test]
fn set_inner_html_on_text_node_is_noop() {
    let mut doc = Document::new();
    let t = doc.create_text_node("x");
    doc.append_child(doc.root(), t);
    doc.set_inner_html(t, "<p>y</p>"); // no-op: text cannot have children
    assert_eq!(doc.text_content(t), "x");
}

#[test]
fn set_inner_html_on_detached_element() {
    let mut doc = Document::new();
    let div = doc.create_element("div"); // detached
    doc.set_inner_html(div, "<p id=late>x</p>");
    // Detached: ids stay out of the index until attached.
    assert_eq!(doc.get_element_by_id("late"), None);
    doc.append_child(doc.root(), div);
    assert_eq!(
        doc.get_element_by_id("late").map(|n| tag(&doc, n)),
        Some("p")
    );
}

// -- document identity --------------------------------------------------------

#[test]
fn node_ids_carry_document_identity() {
    let da = Document::new();
    let db = Document::new();
    // Same arena index, different documents: never equal.
    assert_ne!(da.root(), db.root());
    assert_eq!(da.root(), da.root());
    let mut dc = Document::new();
    let ea = dc.create_element("div");
    assert_ne!(ea, da.root());
}

#[test]
fn cross_document_nodeid_fails_closed() {
    let da = Document::new();
    let mut db = Document::new();
    let foreign = db.create_element("div");
    // Fail-closed instead of panicking: a linkless empty comment, so
    // structural queries on it stay neutral.
    assert!(matches!(da.kind(foreign), NodeKind::Comment(c) if c.is_empty()));
    assert_eq!(da.parent(foreign), None);
    assert_eq!(da.children(foreign), Vec::new());
    assert_eq!(da.text_content(foreign), "");
}

#[test]
fn cross_document_nodeid_mutation_fails_closed() {
    let mut da = Document::new();
    let mut db = Document::new();
    let foreign = db.create_element("div");
    // Mutating accessors fail closed too: no panic, no effect.
    da.append_child(da.root(), foreign);
    da.insert_before(da.root(), foreign, None);
    da.remove(foreign);
    da.set_text_content(foreign, "x");
    assert_eq!(da.children(da.root()), Vec::new());
    // The foreign node is untouched in its own document.
    assert_eq!(db.parent(foreign), None);
}

// -- document fragments and document validity --------------------------------

fn make_fragment(doc: &mut Document, tags: &[&str]) -> NodeId {
    let frag = doc.alloc(NodeKind::DocumentFragment);
    for t in tags {
        let el = doc.create_element(t);
        doc.append_child(frag, el);
    }
    frag
}

#[test]
fn append_child_splices_fragment_in_order() {
    let mut doc = Document::new();
    let root = doc.create_element("div");
    doc.append_child(doc.root(), root);
    let frag = make_fragment(&mut doc, &["a", "b", "c"]);
    doc.append_child(root, frag);
    let kids = doc.children(root);
    assert_eq!(kids.len(), 3);
    assert_eq!(tag(&doc, kids[0]), "a");
    assert_eq!(tag(&doc, kids[1]), "b");
    assert_eq!(tag(&doc, kids[2]), "c");
    // The fragment is emptied; the fragment node itself is not inserted.
    assert!(doc.children(frag).is_empty());
    assert_eq!(doc.parent(frag), None);
}

#[test]
fn insert_before_splices_fragment() {
    let mut doc = Document::new();
    let root = doc.create_element("div");
    doc.append_child(doc.root(), root);
    let tail = doc.create_element("span");
    doc.append_child(root, tail);
    let frag = make_fragment(&mut doc, &["a", "b"]);
    doc.insert_before(root, frag, Some(tail));
    let kids = doc.children(root);
    assert_eq!(kids.len(), 3);
    assert_eq!(tag(&doc, kids[0]), "a");
    assert_eq!(tag(&doc, kids[1]), "b");
    assert_eq!(kids[2], tail);
    assert!(doc.children(frag).is_empty());
}

#[test]
fn replace_child_splices_fragment() {
    let mut doc = Document::new();
    let root = doc.create_element("div");
    doc.append_child(doc.root(), root);
    let old = doc.create_element("span");
    doc.append_child(root, old);
    let frag = make_fragment(&mut doc, &["a", "b"]);
    doc.replace_child(root, frag, old);
    let kids = doc.children(root);
    assert_eq!(kids.len(), 2);
    assert_eq!(tag(&doc, kids[0]), "a");
    assert_eq!(tag(&doc, kids[1]), "b");
    assert_eq!(doc.parent(old), None);
    assert!(doc.children(frag).is_empty());
}

#[test]
fn replace_child_empty_fragment_removes_old() {
    let mut doc = Document::new();
    let root = doc.create_element("div");
    doc.append_child(doc.root(), root);
    let old = doc.create_element("span");
    doc.append_child(root, old);
    let frag = doc.alloc(NodeKind::DocumentFragment);
    doc.replace_child(root, frag, old);
    assert!(doc.children(root).is_empty());
    assert_eq!(doc.parent(old), None);
}

#[test]
fn document_rejects_second_element_and_text() {
    let mut doc = Document::new();
    let root = doc.root();
    let html = doc.create_element("html");
    doc.append_child(root, html);
    // A second element is a no-op.
    let extra = doc.create_element("div");
    doc.append_child(root, extra);
    assert_eq!(doc.children(root), vec![html]);
    assert_eq!(doc.parent(extra), None);
    // Text directly under the document is a no-op.
    let t = doc.create_text_node("x");
    doc.append_child(root, t);
    assert_eq!(doc.children(root), vec![html]);
    // Comments are still fine.
    let c = doc.alloc(NodeKind::Comment("c".into()));
    doc.append_child(root, c);
    assert_eq!(doc.children(root), vec![html, c]);
}

#[test]
fn document_doctype_ordering() {
    let mut doc = Document::new();
    let root = doc.root();
    let dt = doc.alloc(NodeKind::DocumentType(DocumentTypeData {
        name: "html".into(),
        public_id: None,
        system_id: None,
    }));
    let html = doc.create_element("html");
    doc.append_child(root, dt);
    doc.append_child(root, html);
    assert_eq!(doc.children(root), vec![dt, html]);
    // A second doctype is rejected.
    let dt2 = doc.alloc(NodeKind::DocumentType(DocumentTypeData {
        name: "html".into(),
        public_id: None,
        system_id: None,
    }));
    doc.append_child(root, dt2);
    assert_eq!(doc.children(root), vec![dt, html]);
    // An element before the doctype is rejected (doctype must come first).
    let mut doc2 = Document::new();
    let r2 = doc2.root();
    let h2 = doc2.create_element("html");
    doc2.append_child(r2, h2);
    let d2 = doc2.alloc(NodeKind::DocumentType(DocumentTypeData {
        name: "html".into(),
        public_id: None,
        system_id: None,
    }));
    doc2.append_child(r2, d2);
    assert_eq!(doc2.children(r2), vec![h2]);
}

#[test]
fn document_fragment_validity() {
    let mut doc = Document::new();
    let root = doc.root();
    // A fragment with two elements cannot go under the document.
    let frag = make_fragment(&mut doc, &["a", "b"]);
    doc.append_child(root, frag);
    assert!(doc.children(root).is_empty());
    // A fragment with a text child cannot go under the document.
    let frag2 = doc.alloc(NodeKind::DocumentFragment);
    let t = doc.create_text_node("x");
    doc.append_child(frag2, t);
    doc.append_child(root, frag2);
    assert!(doc.children(root).is_empty());
    // A single-element fragment is fine.
    let frag3 = make_fragment(&mut doc, &["only"]);
    doc.append_child(root, frag3);
    assert_eq!(doc.children(root).len(), 1);
}

#[test]
fn document_same_parent_move_and_replace_ok() {
    let mut doc = Document::new();
    let root = doc.root();
    let dt = doc.alloc(NodeKind::DocumentType(DocumentTypeData {
        name: "html".into(),
        public_id: None,
        system_id: None,
    }));
    let html = doc.create_element("html");
    doc.append_child(root, dt);
    doc.append_child(root, html);
    // Moving the existing element within the document is allowed: it is
    // excluded from the one-element count.
    doc.append_child(root, html);
    assert_eq!(doc.children(root), vec![dt, html]);
    // Replacing the element with a new one is allowed: the old one is
    // excluded since it is removed.
    let html2 = doc.create_element("html");
    doc.replace_child(root, html2, html);
    assert_eq!(doc.children(root), vec![dt, html2]);
    assert_eq!(doc.parent(html), None);
}

// -- template contents traversal --------------------------------------------

fn parse_template_doc() -> (Document, NodeId, NodeId) {
    let (doc, _) = Document::parse_html(
        b"<html><body><template><div id=tin><p>hi</p></div></template><div id=tout></div></body></html>",
    );
    let template = doc.query_selector_all("template").expect("valid selector")[0];
    let frag = doc.template_contents(template).unwrap();
    (doc, template, frag)
}

#[test]
fn template_traversal_skips_contents() {
    let (doc, template, frag) = parse_template_doc();
    // General traversal does not cross into the contents fragment.
    assert!(doc.children(template).is_empty());
    assert!(doc.descendants(template).is_empty());
    assert_eq!(doc.text_content(template), "");
    assert!(
        doc.get_elements_by_tag_name("div")
            .iter()
            .all(|n| { doc.get_attribute(n, "id") != Some("tin") })
    );
    assert_eq!(
        doc.get_elements_by_tag_name("div")
            .iter()
            .filter_map(|n| doc.get_attribute(n, "id"))
            .collect::<Vec<_>>(),
        vec!["tout"]
    );
    assert!(
        doc.query_selector_all("#tin")
            .expect("valid selector")
            .is_empty()
    );
    assert!(
        doc.query_selector_all("template div")
            .expect("valid selector")
            .is_empty()
    );
    assert_eq!(doc.get_element_by_id("tin"), None);
    assert!(doc.get_element_by_id("tout").is_some());
    // Direct traversal from the fragment works.
    assert_eq!(doc.children(frag).len(), 1);
    assert_eq!(doc.descendants(frag).len(), 3);
    assert_eq!(doc.text_content(frag), "hi");
    assert_eq!(doc.get_attribute(doc.children(frag)[0], "id"), Some("tin"));
    // Serialization still inlines the contents.
    let html = doc.serialize_node(template);
    assert!(html.contains("<div id=\"tin\"><p>hi</p></div>"), "{html}");
}

#[test]
fn template_contents_id_indexing() {
    let (mut doc, template, frag) = parse_template_doc();
    // Setting an id on a template-contents element does not index it.
    let inner = doc.children(frag)[0];
    doc.set_attribute(inner, "id", "newid");
    assert_eq!(doc.get_element_by_id("newid"), None);
    // Removing the id is a no-op for the index.
    doc.remove_attribute(inner, "id");
    assert_eq!(doc.get_element_by_id("tin"), None);
    // The template element itself is a normal indexed element.
    doc.set_attribute(template, "id", "tpl");
    assert_eq!(doc.get_element_by_id("tpl"), Some(template));
}

#[test]
fn template_contents_accessor() {
    let (doc, template, frag) = parse_template_doc();
    let div = doc.query_selector_all("div").expect("valid selector")[0];
    assert_eq!(doc.template_contents(div), None);
    assert_eq!(doc.template_contents(frag), None);
    // A template without parsed contents has no fragment yet.
    let mut doc2 = Document::new();
    let t = doc2.create_element("template");
    assert_eq!(doc2.template_contents(t), None);
    let _ = (template, frag);
    let _ = doc;
}

// -- foreign attributes -----------------------------------------------------

#[test]
fn foreign_attributes_get_metadata() {
    let (doc, _) = Document::parse_html(
        b"<html><body><svg><a xlink:href=\"u\" xml:lang=\"en\" xmlns:xlink=\"x\" id=s>hi</a></svg></body></html>",
    );
    let a = doc.query_selector_all("a").expect("valid selector")[0];
    let attrs = match doc.kind(a) {
        NodeKind::Element(d) => d.attributes.clone(),
        _ => panic!("expected element"),
    };
    let by_name = |n: &str| attrs.iter().find(|a| a.name == n).unwrap();
    let href = by_name("xlink:href");
    assert_eq!(href.namespace, Some("http://www.w3.org/1999/xlink"));
    assert_eq!(href.prefix(), Some("xlink"));
    assert_eq!(href.local_name(), "href");
    assert_eq!(href.value, "u");
    let lang = by_name("xml:lang");
    assert_eq!(lang.namespace, Some("http://www.w3.org/XML/1998/namespace"));
    assert_eq!(lang.prefix(), Some("xml"));
    assert_eq!(lang.local_name(), "lang");
    let xmlns = by_name("xmlns:xlink");
    assert_eq!(xmlns.namespace, Some("http://www.w3.org/2000/xmlns/"));
    assert_eq!(xmlns.prefix(), Some("xmlns"));
    assert_eq!(xmlns.local_name(), "xlink");
    // Ordinary attributes stay unnamespaced.
    let id = by_name("id");
    assert_eq!(id.namespace, None);
    assert_eq!(id.prefix(), None);
    assert_eq!(id.local_name(), "id");
}

#[test]
fn html_element_foreign_looking_attrs_stay_plain() {
    let (doc, _) = Document::parse_html(
        b"<html><body><div xlink:href=\"u\" xmlns=\"y\">x</div></body></html>",
    );
    let div = doc.query_selector_all("div").expect("valid selector")[0];
    let attrs = match doc.kind(div) {
        NodeKind::Element(d) => d.attributes.clone(),
        _ => panic!("expected element"),
    };
    // On HTML elements there is no foreign adjustment: no namespaces,
    // but the qualified name still splits mechanically.
    for a in &attrs {
        assert_eq!(a.namespace, None);
    }
    let href = attrs.iter().find(|a| a.name == "xlink:href").unwrap();
    assert_eq!(href.prefix(), Some("xlink"));
    assert_eq!(href.local_name(), "href");
}

#[test]
fn set_attribute_stays_unnamespaced() {
    let mut doc = Document::new();
    let svg = doc.create_element_ns(Namespace::Svg, "a");
    doc.set_attribute(svg, "xlink:href", "u");
    let attrs = match doc.kind(svg) {
        NodeKind::Element(d) => d.attributes.clone(),
        _ => panic!("expected element"),
    };
    let href = attrs.iter().find(|a| a.name == "xlink:href").unwrap();
    assert_eq!(href.namespace, None);
    assert_eq!(href.prefix(), Some("xlink"));
    assert_eq!(href.local_name(), "href");
}

#[test]
fn replace_child_with_next_sibling() {
    let mut doc = Document::new();
    let root = doc.create_element("div");
    doc.append_child(doc.root(), root);
    let a = doc.create_element("a");
    let b = doc.create_element("b");
    let c = doc.create_element("c");
    doc.append_child(root, a);
    doc.append_child(root, b);
    doc.append_child(root, c);
    // Replacing `a` with its own next sibling `b`: the reference is
    // retargeted past `b` (per the DOM replace algorithm) instead of
    // linking before a detached node.
    doc.replace_child(root, b, a);
    assert_eq!(doc.children(root), vec![b, c]);
    assert_eq!(doc.parent(a), None);
    assert_eq!(doc.next_sibling(b), Some(c));
    assert_eq!(doc.previous_sibling(c), Some(b));
}

// -- first_child traversal cursor -------------------------------------------

/// `first_child` + `next_sibling` walks must visit exactly the nodes
/// `children()` returns, node by node over a mixed tree.
#[test]
fn first_child_cursor_matches_children_everywhere() {
    let mut doc = Document::new();
    let root = doc.root();
    let div = doc.create_element("div");
    doc.append_child(root, div);
    let p = doc.create_element("p");
    doc.append_child(div, p);
    let t = doc.create_text_node("x");
    doc.append_child(p, t);
    let span = doc.create_element("span");
    doc.append_child(div, span);
    let em = doc.create_element("em");
    doc.append_child(span, em);
    for node in [root, div, p, span, em] {
        let expected = doc.children(node);
        let mut got = Vec::new();
        let mut cursor = doc.first_child(node);
        while let Some(c) = cursor {
            got.push(c);
            cursor = doc.next_sibling(c);
        }
        assert_eq!(got, expected, "cursor walk differs for {node:?}");
    }
    // Leaf with no children: cursor starts empty.
    assert_eq!(doc.first_child(t), None);
    assert_eq!(doc.first_child(em), None);
}

/// `first_child` skips the template-contents edge just like `children()`.
#[test]
fn first_child_skips_template_contents() {
    let (doc, _) =
        Document::parse_html(b"<html><body><template><p id=\"t\">in</p></template></body></html>");
    let template = doc.get_elements_by_tag_name("template").item(0).unwrap();
    assert!(doc.children(template).is_empty());
    assert_eq!(doc.first_child(template), None);
    // The fragment is still reachable directly (its ids are not indexed,
    // like children() of the fragment itself).
    let frag = doc.template_contents(template).unwrap();
    let p = doc.first_child(frag).unwrap();
    assert!(matches!(doc.kind(p), NodeKind::Element(d) if d.tag_name == "p"));
    assert_eq!(doc.children(frag), vec![p]);
    // Fail-closed on a NodeId from another document: no allocation, no panic.
    let other = Document::new();
    let foreign = other.root();
    assert_eq!(doc.first_child(foreign), None);
    assert_eq!(doc.next_sibling(foreign), None);
}

// -- dirty tracking --------------------------------------------------------

#[test]
fn parse_starts_clean() {
    let (mut doc, _) = Document::parse_html(b"<html><body><p>hi</p></body></html>");
    assert!(doc.take_dirty().is_empty());
}

#[test]
fn append_child_marks_parent_style_affecting() {
    let (mut doc, _) = Document::parse_html(b"<html><body></body></html>");
    assert!(doc.take_dirty().is_empty());
    let body = doc.get_elements_by_tag_name("body").item(0).unwrap();
    let p = doc.create_element("p");
    // Detached construction records nothing.
    assert!(doc.take_dirty().is_empty());
    doc.append_child(body, p);
    let report = doc.take_dirty();
    assert!(report.style, "new subtree needs selector matching");
    assert_eq!(report.subtrees, vec![body]);
    // The drain is one-shot.
    assert!(doc.take_dirty().is_empty());
}

#[test]
fn text_node_emptiness_toggle_marks_style() {
    // `:empty` (and through it `:not(:empty)` / `:has()`) toggles on
    // exactly the empty <-> non-empty transition of a text child, so
    // the style stage must re-run when a CharacterData write crosses
    // it.
    let (mut doc, _) = Document::parse_html(b"<html><body><p>hi</p></body></html>");
    assert!(doc.take_dirty().is_empty());
    let p = doc.get_elements_by_tag_name("p").item(0).unwrap();
    let text = doc.first_child(p).expect("text child");
    doc.set_text_content(text, "");
    let report = doc.take_dirty();
    assert!(report.style, "emptying a text node can toggle :empty");
    assert_eq!(report.subtrees, vec![text]);
    // Refilling toggles back: style again.
    doc.set_text_content(text, "hi");
    assert!(
        doc.take_dirty().style,
        "refilling a text node toggles :empty back"
    );
}

#[test]
fn text_node_nonempty_rewrite_skips_style() {
    // The cheap path: no emptiness change, layout/paint only.
    let (mut doc, _) = Document::parse_html(b"<html><body><p>hi</p></body></html>");
    assert!(doc.take_dirty().is_empty());
    let p = doc.get_elements_by_tag_name("p").item(0).unwrap();
    let text = doc.first_child(p).expect("text child");
    doc.set_text_content(text, "hello");
    let report = doc.take_dirty();
    assert!(!report.style, "non-empty rewrite keeps :empty matching");
    assert_eq!(report.subtrees, vec![text]);
    // Writing the same text again records nothing.
    doc.set_text_content(text, "hello");
    assert!(doc.take_dirty().is_empty());
}

#[test]
fn element_text_content_marks_style() {
    // Element-level `textContent` was already style-affecting
    // (structural: children are replaced, `:has()` on ancestors can
    // change); pin it stays that way.
    let (mut doc, _) = Document::parse_html(b"<html><body><p>hi</p></body></html>");
    assert!(doc.take_dirty().is_empty());
    let p = doc.get_elements_by_tag_name("p").item(0).unwrap();
    doc.set_text_content(p, "hello");
    let report = doc.take_dirty();
    assert!(report.style, "element textContent is style-affecting");
    assert_eq!(report.subtrees, vec![p]);
}

#[test]
fn comment_data_write_skips_style() {
    // Comments never affect `:empty` (Selectors 4 ignores them), so
    // comment data writes stay on the cheap path.
    let (mut doc, _) = Document::parse_html(b"<html><body><p>hi</p></body></html>");
    assert!(doc.take_dirty().is_empty());
    let body = doc.get_elements_by_tag_name("body").item(0).unwrap();
    let comment = doc.create_comment("note");
    doc.append_child(body, comment);
    assert!(
        doc.take_dirty().style,
        "structural attach is style-affecting"
    );
    doc.set_text_content(comment, "edited");
    let report = doc.take_dirty();
    assert!(!report.style, "comment data cannot toggle :empty");
    assert_eq!(report.subtrees, vec![comment]);
}

#[test]
fn any_attribute_write_marks_style() {
    let (mut doc, _) = Document::parse_html(b"<html><body><p>hi</p></body></html>");
    assert!(doc.take_dirty().is_empty());
    let p = doc.get_elements_by_tag_name("p").item(0).unwrap();
    doc.set_attribute(p, "class", "a");
    assert!(doc.take_dirty().style);
    // Attribute selectors can match any attribute, so even `data-*`
    // writes can change matching: the style stage re-runs.
    doc.set_attribute(p, "data-x", "1");
    let report = doc.take_dirty();
    assert!(report.style, "data-* can affect matching");
    assert_eq!(report.subtrees, vec![p]);
    // Unchanged value records nothing.
    doc.set_attribute(p, "data-x", "1");
    assert!(doc.take_dirty().is_empty());
    doc.remove_attribute(p, "data-x");
    assert!(doc.take_dirty().style);
    // Removing a missing attribute records nothing.
    doc.remove_attribute(p, "missing");
    assert!(doc.take_dirty().is_empty());
}

#[test]
fn move_marks_old_and_new_parent() {
    let (mut doc, _) = Document::parse_html(
        b"<html><body><div id=a><p>hi</p></div><div id=b></div></body></html>",
    );
    assert!(doc.take_dirty().is_empty());
    let a = doc.get_element_by_id("a").expect("div#a");
    let b = doc.get_element_by_id("b").expect("div#b");
    let p = doc.get_elements_by_tag_name("p").item(0).unwrap();
    // Moving an attached node dirties both the old and the new parent:
    // both subtrees' boxes and matching change.
    doc.append_child(b, p);
    let report = doc.take_dirty();
    assert!(report.style);
    assert_eq!(report.subtrees.len(), 2);
    assert!(report.subtrees.contains(&a), "old parent must be dirtied");
    assert!(report.subtrees.contains(&b), "new parent must be dirtied");
    // Reordering within one parent dirties it exactly once.
    doc.append_child(b, p);
    let report = doc.take_dirty();
    assert_eq!(report.subtrees, vec![b]);
}

#[test]
fn structural_removals_mark_parent() {
    let (mut doc, _) = Document::parse_html(b"<html><body><p>hi</p></body></html>");
    assert!(doc.take_dirty().is_empty());
    let body = doc.get_elements_by_tag_name("body").item(0).unwrap();
    let p = doc.get_elements_by_tag_name("p").item(0).unwrap();
    doc.remove_child(body, p);
    let report = doc.take_dirty();
    assert!(report.style);
    assert_eq!(report.subtrees, vec![body]);
    // Re-attach, then detach via ChildNode.remove().
    doc.append_child(body, p);
    assert!(doc.take_dirty().style);
    doc.remove(p);
    let report = doc.take_dirty();
    assert_eq!(report.subtrees, vec![body]);
}

#[test]
fn detached_mutations_record_nothing() {
    let mut doc = Document::new();
    let el = doc.create_element("div");
    doc.set_attribute(el, "class", "a");
    doc.set_text_content(el, "hi");
    assert!(
        doc.take_dirty().is_empty(),
        "detached work is invisible to the renderer"
    );
    // Linking the subtree in marks the new parent exactly once.
    let root = doc.root();
    doc.append_child(root, el);
    let report = doc.take_dirty();
    assert_eq!(report.subtrees, vec![root]);
}

#[test]
fn set_inner_html_marks_element() {
    let (mut doc, _) = Document::parse_html(b"<html><body><div>old</div></body></html>");
    assert!(doc.take_dirty().is_empty());
    let div = doc.get_elements_by_tag_name("div").item(0).unwrap();
    doc.set_inner_html(div, "<p>new</p>");
    let report = doc.take_dirty();
    assert!(report.style);
    assert_eq!(report.subtrees, vec![div]);
}

#[test]
fn insert_before_marks_parent_style_affecting() {
    let (mut doc, _) = Document::parse_html(b"<html><body><p>one</p><p>two</p></body></html>");
    assert!(doc.take_dirty().is_empty());
    let body = doc.get_elements_by_tag_name("body").item(0).unwrap();
    let second = doc.get_elements_by_tag_name("p").item(1).unwrap();
    let span = doc.create_element("span");
    doc.insert_before(body, span, Some(second));
    let report = doc.take_dirty();
    assert!(report.style, "new subtree needs selector matching");
    assert_eq!(report.subtrees, vec![body]);
    assert!(doc.take_dirty().is_empty());
}

#[test]
fn replace_child_marks_parent_style_affecting() {
    let (mut doc, _) = Document::parse_html(b"<html><body><p>one</p><p>two</p></body></html>");
    assert!(doc.take_dirty().is_empty());
    let body = doc.get_elements_by_tag_name("body").item(0).unwrap();
    let first = doc.get_elements_by_tag_name("p").item(0).unwrap();
    let span = doc.create_element("span");
    doc.replace_child(body, span, first);
    let report = doc.take_dirty();
    assert!(report.style, "replaced subtree needs selector matching");
    assert_eq!(report.subtrees, vec![body]);
    assert!(doc.take_dirty().is_empty());
}

#[test]
fn shadow_nested_descendant_attribute_dirties_host() {
    // The shadow fragment is unlinked, so plain `mark_dirty` on a
    // shadow descendant records nothing; the host (whose boxes render
    // the shadow children) must be dirtied instead.
    let (mut doc, _) = Document::parse_html(b"<html><body><div id=h></div></body></html>");
    let host = doc.get_element_by_id("h").expect("div#h");
    let frag = doc
        .attach_shadow_root(host, ShadowRootMode::Open)
        .expect("shadow root");
    let outer = doc.create_element("div");
    doc.append_child(frag, outer);
    let inner = doc.create_element("div");
    doc.append_child(outer, inner);
    assert!(!doc.take_dirty().is_empty());
    // Attribute writes on the nested descendant must dirty the host.
    doc.set_attribute(inner, "class", "x");
    let report = doc.take_dirty();
    assert!(report.style, "attribute writes affect selector matching");
    assert_eq!(report.subtrees, vec![host]);
    doc.remove_attribute(inner, "class");
    let report = doc.take_dirty();
    assert!(report.style);
    assert_eq!(report.subtrees, vec![host]);
}

#[test]
fn shadow_descendant_text_content_dirties_host() {
    // `set_text_content` (element and text-node paths) inside a shadow
    // tree must dirty the host.
    let (mut doc, _) = Document::parse_html(b"<html><body><div id=h></div></body></html>");
    let host = doc.get_element_by_id("h").expect("div#h");
    let frag = doc
        .attach_shadow_root(host, ShadowRootMode::Open)
        .expect("shadow root");
    let outer = doc.create_element("div");
    doc.append_child(frag, outer);
    let p = doc.create_element("p");
    doc.set_text_content(p, "hi");
    doc.append_child(outer, p);
    assert!(!doc.take_dirty().is_empty());
    // Element textContent replaces children: style-affecting.
    doc.set_text_content(p, "bye");
    let report = doc.take_dirty();
    assert!(report.style, "textContent can toggle :empty/:has matching");
    assert_eq!(report.subtrees, vec![host]);
    // Text-node data writes route through the shared private path.
    let text = doc.first_child(p).expect("text child");
    doc.set_text_content(text, "hello");
    let report = doc.take_dirty();
    assert!(!report.style, "non-empty rewrite keeps :empty matching");
    assert_eq!(report.subtrees, vec![host]);
}

#[test]
fn shadow_descendant_remove_dirties_host() {
    // `ChildNode.remove()` on a shadow descendant must dirty the host:
    // the old parent lives in the unlinked shadow tree, so plain
    // `mark_dirty` records nothing there.
    let (mut doc, _) = Document::parse_html(b"<html><body><div id=h></div></body></html>");
    let host = doc.get_element_by_id("h").expect("div#h");
    let frag = doc
        .attach_shadow_root(host, ShadowRootMode::Open)
        .expect("shadow root");
    let outer = doc.create_element("div");
    doc.append_child(frag, outer);
    let p = doc.create_element("p");
    doc.set_text_content(p, "hi");
    doc.append_child(outer, p);
    assert!(!doc.take_dirty().is_empty());
    doc.remove(p);
    let report = doc.take_dirty();
    assert!(report.style);
    assert_eq!(report.subtrees, vec![host]);
}

#[test]
fn shadow_fragment_inner_html_dirties_host() {
    // `set_inner_html` on the shadow fragment itself (the DOM
    // `ShadowRoot.innerHTML` setter) must dirty the host.
    let (mut doc, _) = Document::parse_html(b"<html><body><div id=h></div></body></html>");
    let host = doc.get_element_by_id("h").expect("div#h");
    let frag = doc
        .attach_shadow_root(host, ShadowRootMode::Open)
        .expect("shadow root");
    assert!(!doc.take_dirty().is_empty());
    doc.set_inner_html(frag, "<p>new</p>");
    let report = doc.take_dirty();
    assert!(report.style, "wholesale rebuild changes matching and boxes");
    assert_eq!(report.subtrees, vec![host]);
    // A nested shadow descendant element works the same way.
    let p = doc.first_child(frag).expect("shadow <p>");
    doc.set_inner_html(p, "<b>bold</b>");
    let report = doc.take_dirty();
    assert!(report.style);
    assert_eq!(report.subtrees, vec![host]);
}

#[test]
fn set_outer_html_marks_parent_style_affecting() {
    let (mut doc, _) = Document::parse_html(b"<html><body><p>one</p></body></html>");
    assert!(doc.take_dirty().is_empty());
    let body = doc.get_elements_by_tag_name("body").item(0).unwrap();
    let p = doc.get_elements_by_tag_name("p").item(0).unwrap();
    doc.set_outer_html(p, "<span>new</span>");
    let report = doc.take_dirty();
    assert!(report.style, "replaced node needs selector matching");
    assert_eq!(report.subtrees, vec![body]);
    // Replacing with an empty string still removes the node: dirt must
    // be recorded even though nothing is inserted.
    let span = doc.get_elements_by_tag_name("span").item(0).unwrap();
    doc.set_outer_html(span, "");
    let report = doc.take_dirty();
    assert!(report.style);
    assert_eq!(report.subtrees, vec![body]);
    assert!(doc.get_elements_by_tag_name("span").is_empty());
    assert!(doc.take_dirty().is_empty());
}

#[test]
fn element_text_content_marks_style_affecting() {
    let (mut doc, _) = Document::parse_html(b"<html><body><p>hi</p></body></html>");
    assert!(doc.take_dirty().is_empty());
    let p = doc.get_elements_by_tag_name("p").item(0).unwrap();
    doc.set_text_content(p, "hello");
    let report = doc.take_dirty();
    assert!(
        report.style,
        "children went away: :has() matching on ancestors can change"
    );
    assert_eq!(report.subtrees, vec![p]);
    assert!(doc.take_dirty().is_empty());
}

#[test]
fn clone_node_records_nothing() {
    let (mut doc, _) = Document::parse_html(b"<html><body><p>hi</p></body></html>");
    assert!(doc.take_dirty().is_empty());
    let p = doc.get_elements_by_tag_name("p").item(0).unwrap();
    let _ = doc.clone_node(p, true);
    assert!(
        doc.take_dirty().is_empty(),
        "the clone is detached: invisible to the renderer"
    );
}

#[test]
fn remove_detached_records_nothing() {
    let mut doc = Document::new();
    let el = doc.create_element("div");
    doc.remove(el);
    assert!(doc.take_dirty().is_empty(), "no parent, no mutation");
}

#[test]
fn take_dirty_reports_later_mutations_exactly_once() {
    let (mut doc, _) = Document::parse_html(b"<html><body></body></html>");
    assert!(doc.take_dirty().is_empty());
    let body = doc.get_elements_by_tag_name("body").item(0).unwrap();
    let a = doc.create_element("p");
    doc.append_child(body, a);
    let first = doc.take_dirty();
    assert_eq!(first.subtrees, vec![body]);
    // The drain is one-shot: no replay on the next take.
    assert!(doc.take_dirty().is_empty());
    // A mutation after a take is reported by the next take, exactly
    // once: no lost updates, no duplication.
    doc.set_attribute(a, "class", "x");
    let second = doc.take_dirty();
    assert!(second.style);
    assert_eq!(second.subtrees, vec![a]);
    assert!(doc.take_dirty().is_empty());
}

// -- web-parity DOM API --------------------------------------------------------

#[test]
fn last_child_cursor() {
    let (doc, _) = Document::parse_html(b"<div><p>a</p>tail</div>");
    let div = doc.get_elements_by_tag_name("div").item(0).unwrap();
    let last = doc.last_child(div).unwrap();
    assert!(matches!(doc.kind(last), NodeKind::Text(t) if t.as_ref() == "tail"));
    // A leaf's last_child is None; walking last_child + previous_sibling
    // visits every child in reverse.
    assert_eq!(doc.last_child(last), None);
    let mut rev = Vec::new();
    let mut cursor = doc.last_child(div);
    while let Some(c) = cursor {
        rev.push(c);
        cursor = doc.previous_sibling(c);
    }
    assert_eq!(rev.len(), 2);
    // Template contents stay out of reach of last_child too.
    let (doc2, _) = Document::parse_html(b"<template><p>t</p></template>");
    let template = doc2.get_elements_by_tag_name("template").item(0).unwrap();
    assert_eq!(doc2.last_child(template), None);
}

#[test]
fn element_sibling_navigation() {
    let (doc, _) = Document::parse_html(b"<div>text<p>a</p>mid<span>b</span></div>");
    let div = doc.get_elements_by_tag_name("div").item(0).unwrap();
    // first_element_child skips the leading text node.
    let p = doc.first_element_child(div).unwrap();
    assert_eq!(tag(&doc, p), "p");
    // next/previous skip the "mid" text node.
    let span = doc.next_element_sibling(p).unwrap();
    assert_eq!(tag(&doc, span), "span");
    assert_eq!(doc.next_element_sibling(span), None);
    assert_eq!(doc.previous_element_sibling(span), Some(p));
    assert_eq!(doc.previous_element_sibling(p), None);
    assert_eq!(doc.child_element_count(div), 2);
    // Childless nodes: no element children or siblings.
    let text = doc.first_child(div).unwrap();
    assert_eq!(doc.first_element_child(text), None);
    assert_eq!(doc.child_element_count(text), 0);
    // Foreign ids fail closed.
    let other = Document::new();
    assert_eq!(other.first_element_child(div), None);
    assert_eq!(other.next_element_sibling(p), None);
    assert_eq!(other.previous_element_sibling(p), None);
    assert_eq!(other.child_element_count(div), 0);
}

#[test]
fn has_attribute_cases() {
    let mut doc = Document::new();
    let el = doc.create_element("div");
    doc.set_attribute(el, "id", "a");
    assert!(doc.has_attribute(el, "id"));
    assert!(doc.has_attribute(el, "ID")); // HTML: ASCII case-insensitive
    assert!(!doc.has_attribute(el, "class"));
    doc.set_attribute(el, "data-x", "");
    assert!(doc.has_attribute(el, "data-x")); // present but empty
    let t = doc.create_text_node("x");
    assert!(!doc.has_attribute(t, "id"));
    // Foreign-namespace elements match attribute names exactly.
    let svg = doc.create_element_ns(Namespace::Svg, "svg");
    doc.set_attribute(svg, "viewBox", "0 0 1 1");
    assert!(doc.has_attribute(svg, "viewBox"));
    assert!(!doc.has_attribute(svg, "viewbox"));
}

#[test]
fn create_comment_and_fragment() {
    let mut doc = Document::new();
    let c = doc.create_comment("hi");
    assert!(matches!(doc.kind(c), NodeKind::Comment(s) if s == "hi"));
    let f = doc.create_document_fragment();
    assert!(matches!(doc.kind(f), NodeKind::DocumentFragment));
    assert!(doc.children(f).is_empty());
    // Fragments splice into the tree like DOM appendChild.
    let p = doc.create_element("p");
    doc.append_child(f, p);
    doc.append_child(doc.root(), f);
    assert_eq!(doc.inner_html(doc.root()), "<p></p>");
}

#[test]
fn inner_html_getter() {
    let (doc, _) = Document::parse_html(b"<div id=a>hi <b>x</b><!--c--></div>");
    let div = doc.get_element_by_id("a").unwrap();
    assert_eq!(doc.inner_html(div), "hi <b>x</b><!--c-->");
    // Entities re-escape on serialization.
    let (doc2, _) = Document::parse_html(b"<p>a &amp; b</p>");
    let p = doc2.get_elements_by_tag_name("p").item(0).unwrap();
    assert_eq!(doc2.inner_html(p), "a &amp; b");
    // A template's innerHTML is its contents.
    let (doc3, _) = Document::parse_html(b"<template><p>t</p></template>");
    let template = doc3.get_elements_by_tag_name("template").item(0).unwrap();
    assert_eq!(doc3.inner_html(template), "<p>t</p>");
    // Foreign ids fail closed.
    let other = Document::new();
    assert_eq!(other.inner_html(div), "");
}

#[test]
fn outer_html_getter_and_setter() {
    let (mut doc, _) = Document::parse_html(b"<div><p id=a>old</p><span>s</span></div>");
    let p = doc.get_element_by_id("a").unwrap();
    assert_eq!(doc.outer_html(p), "<p id=\"a\">old</p>");
    // The parsed nodes take the element's exact position.
    doc.set_outer_html(p, "<b>new</b><i>i</i>");
    let div = doc.get_elements_by_tag_name("div").item(0).unwrap();
    assert_eq!(doc.inner_html(div), "<b>new</b><i>i</i><span>s</span>");
    // The replaced node is detached but alive; the id index dropped it.
    assert_eq!(doc.get_element_by_id("a"), None);
    assert_eq!(doc.parent(p), None);
    assert_eq!(doc.outer_html(p), "<p id=\"a\">old</p>");
    // New ids are indexed.
    let b = doc.get_elements_by_tag_name("b").item(0).unwrap();
    doc.set_outer_html(b, "<section id=nb>n</section>");
    assert_eq!(
        doc.get_element_by_id("nb").map(|n| doc.text_content(n)),
        Some("n".to_string())
    );
    // Empty html removes the node.
    let i = doc.get_elements_by_tag_name("i").item(0).unwrap();
    doc.set_outer_html(i, "");
    assert_eq!(doc.get_elements_by_tag_name("i").len(), 0);
    // Fail-closed: no parent, or the document as parent.
    let orphan = doc.create_element("p");
    doc.set_outer_html(orphan, "<b>x</b>");
    assert_eq!(doc.get_elements_by_tag_name("b").len(), 0);
    let html = doc.document_element().unwrap();
    let before = doc.serialize();
    doc.set_outer_html(html, "<p>x</p>");
    assert_eq!(doc.serialize(), before);
}

#[test]
fn matches_api() {
    let (doc, _) = Document::parse_html(b"<div id=a class=x><p id=b class='x y'>t</p></div>");
    let p = doc.get_element_by_id("b").unwrap();
    assert!(doc.matches(p, "p").unwrap());
    assert!(doc.matches(p, "p.x.y").unwrap());
    assert!(doc.matches(p, "div p").unwrap());
    assert!(doc.matches(p, "#a > p").unwrap());
    assert!(doc.matches(p, "p + p, p.y").unwrap());
    assert!(!doc.matches(p, "span").unwrap());
    assert!(!doc.matches(p, "div").unwrap());
    // Non-elements never match; foreign ids fail closed.
    let text = doc.first_child(p).unwrap();
    assert!(!doc.matches(text, "*").unwrap());
    let other = Document::new();
    assert!(!other.matches(p, "*").unwrap());
    // Syntax errors surface.
    assert!(doc.matches(p, "p >").is_err());
    assert!(doc.matches(p, "").is_err());
}

#[test]
fn closest_api() {
    let (doc, _) = Document::parse_html(
        b"<div id=a class=x><section><p id=b class='x y'>t</p></section></div>",
    );
    let p = doc.get_element_by_id("b").unwrap();
    // Includes the element itself.
    assert_eq!(doc.closest(p, "p").unwrap(), Some(p));
    assert_eq!(doc.closest(p, ".y").unwrap(), Some(p));
    let section = doc.get_elements_by_tag_name("section").item(0).unwrap();
    assert_eq!(doc.closest(p, "section").unwrap(), Some(section));
    let div = doc.get_element_by_id("a").unwrap();
    assert_eq!(doc.closest(p, "div.x").unwrap(), Some(div));
    assert_eq!(doc.closest(p, "span").unwrap(), None);
    // From a text node: ancestors only.
    let text = doc.first_child(p).unwrap();
    assert_eq!(doc.closest(text, "p").unwrap(), Some(p));
    // Template contents are out of reach: the walk stops at the contents
    // fragment instead of crossing to the host element.
    let (doc2, _) = Document::parse_html(b"<div id=out><template><p id=in>t</p></template></div>");
    let template = doc2.get_elements_by_tag_name("template").item(0).unwrap();
    let frag = doc2.template_contents(template).unwrap();
    let pin = doc2.first_child(frag).unwrap();
    assert_eq!(doc2.closest(pin, "div").unwrap(), None);
    assert_eq!(doc2.closest(pin, "p").unwrap(), Some(pin));
    // Syntax errors surface.
    assert!(doc.closest(p, ">>").is_err());
}

#[test]
fn get_elements_by_class_name_basics() {
    let (doc, _) = Document::parse_html(
        b"<div class='a b'><p class=b>x</p><span class=c>y</span><i class='a b c'>z</i></div>",
    );
    // Every listed class must be present.
    assert_eq!(doc.get_elements_by_class_name("b").len(), 3);
    assert_eq!(doc.get_elements_by_class_name("a b").len(), 2);
    assert_eq!(doc.get_elements_by_class_name("a b c").len(), 1);
    assert_eq!(doc.get_elements_by_class_name("z").len(), 0);
    // Empty or blank queries match nothing.
    assert!(doc.get_elements_by_class_name("").is_empty());
    assert!(doc.get_elements_by_class_name("   ").is_empty());
    // Document order.
    let tags: Vec<String> = doc
        .get_elements_by_class_name("b")
        .iter()
        .map(|n| tag(&doc, n).to_string())
        .collect();
    assert_eq!(tags, vec!["div", "p", "i"]);
    // item() indexes into document order.
    let div = doc.get_elements_by_tag_name("div").item(0).unwrap();
    assert_eq!(doc.get_elements_by_class_name("b").item(0), Some(div));
    assert_eq!(doc.get_elements_by_class_name("b").item(99), None);
}

#[test]
fn live_collections_requery_on_access() {
    let (mut doc, _) = Document::parse_html(b"<div><p class=a>x</p></div>");
    // The view is live: each access re-runs the query, so mutations made
    // after a view was created are visible on the next access. Views are
    // short-lived borrows here (a live `HtmlCollection` borrows the
    // document, so the query is re-issued per assertion instead of held
    // across mutations).
    assert_eq!(doc.get_elements_by_class_name("a").len(), 1);
    assert_eq!(doc.get_elements_by_tag_name("p").len(), 1);
    let div = doc.get_elements_by_tag_name("div").item(0).unwrap();
    let q = doc.create_element("p");
    doc.set_attribute(q, "class", "a");
    doc.append_child(div, q);
    assert_eq!(doc.get_elements_by_class_name("a").len(), 2);
    assert_eq!(doc.get_elements_by_tag_name("p").len(), 2);
    assert_eq!(doc.get_elements_by_class_name("a").item(1), Some(q));
    // Removals are visible too.
    let first_p = doc.get_elements_by_tag_name("p").item(0).unwrap();
    doc.remove_child(div, first_p);
    assert_eq!(doc.get_elements_by_tag_name("p").len(), 1);
    assert_eq!(doc.get_elements_by_tag_name("p").item(0), Some(q));
    assert_eq!(doc.get_elements_by_class_name("a").len(), 1);
    // Class changes via set_attribute are picked up as well.
    doc.set_attribute(q, "class", "b");
    assert!(doc.get_elements_by_class_name("a").is_empty());
    assert_eq!(doc.get_elements_by_class_name("b").len(), 1);
}

#[test]
fn selector_attribute_selectors() {
    let (doc, _) = Document::parse_html(
        b"<div id=d data-x=1 data-y='a b c' data-z=a-b data-pre=hello data-suf=world data-sub='xxhelloyy' data-empty='' lang=en-US data-gt='a>b'></div>",
    );
    let hits = |sel: &str| doc.query_selector_all(sel).expect("valid selector").len();
    assert_eq!(hits("[data-x]"), 1);
    assert_eq!(hits("[data-missing]"), 0);
    assert_eq!(hits("[data-x=\"1\"]"), 1);
    assert_eq!(hits("[data-x='1']"), 1);
    assert_eq!(hits("[data-x=1]"), 1);
    assert_eq!(hits("[data-x=\"2\"]"), 0);
    assert_eq!(hits("[data-empty=\"\"]"), 1);
    // ~= : whitespace-separated word.
    assert_eq!(hits("[data-y~=\"b\"]"), 1);
    assert_eq!(hits("[data-y~=\"bc\"]"), 0);
    // |= : exact or dash-prefixed.
    assert_eq!(hits("[lang|=\"en\"]"), 1);
    assert_eq!(hits("[lang|=\"en-US\"]"), 1);
    assert_eq!(hits("[lang|=\"en-U\"]"), 0);
    assert_eq!(hits("[data-z|=\"a\"]"), 1);
    // ^= $= *= : affix / substring.
    assert_eq!(hits("[data-pre^=\"hell\"]"), 1);
    assert_eq!(hits("[data-pre^=\"ello\"]"), 0);
    assert_eq!(hits("[data-suf$=\"rld\"]"), 1);
    assert_eq!(hits("[data-suf$=\"wor\"]"), 0);
    assert_eq!(hits("[data-sub*=\"ello\"]"), 1);
    assert_eq!(hits("[data-sub*=\"zz\"]"), 0);
    // Combined with type / class / id in one compound.
    assert_eq!(hits("div#d[data-x=\"1\"]"), 1);
    assert_eq!(hits("span[data-x]"), 0);
    // Whitespace around the operator and quoted combinator chars.
    assert_eq!(hits("[data-x = \"1\"]"), 1);
    assert_eq!(hits("[ data-gt = 'a>b' ]"), 1);
    // HTML attribute names are ASCII case-insensitive.
    assert_eq!(hits("[DATA-X]"), 1);
    assert_eq!(hits("[Data-X=\"1\"]"), 1);
}

#[test]
fn selector_attribute_selector_errors() {
    let (doc, _) = Document::parse_html(b"<div data-x=1></div>");
    let bad = [
        "[data-x",           // unterminated bracket
        "[data-x=\"1",       // unterminated string
        "[]",                // missing name
        "[=1]",              // missing name
        "[data-x!=\"1\"]",   // bad operator
        "[data-x~=\"a b\"]", // ~= value with whitespace
        "[data-x~=\"\"]",    // ~= empty value
        "[data-x^=\"\"]",    // ^= empty value
        "[data-x$=\"\"]",    // $= empty value
        "[data-x*=\"\"]",    // *= empty value
        "[data-x|=]",        // |= empty value
        "div]",              // stray bracket
    ];
    for sel in bad {
        assert!(
            doc.query_selector_all(sel).is_err(),
            "{sel:?} should be a syntax error"
        );
    }
}

#[test]
fn selector_sibling_combinators() {
    // <div><p class=a>1</p> text <p class=b>2</p><span>3</span><p class=b>4</p></div>
    let (doc, _) = Document::parse_html(
        b"<div><p class=a>1</p> text <p class=b>2</p><span>3</span><p class=b>4</p></div>",
    );
    // Adjacent: text nodes between elements do not break adjacency.
    assert_eq!(qs_text(&doc, "p.a + p"), vec!["2"]);
    assert_eq!(qs_text(&doc, ".a + .b"), vec!["2"]);
    assert_eq!(qs_text(&doc, "p.b + span"), vec!["3"]);
    assert_eq!(qs_text(&doc, "span + p"), vec!["4"]);
    assert_eq!(qs_text(&doc, "p.b + p.b"), Vec::<String>::new());
    assert_eq!(qs_text(&doc, "div + p"), Vec::<String>::new());
    // General sibling.
    assert_eq!(qs_text(&doc, "p.a ~ p"), vec!["2", "4"]);
    assert_eq!(qs_text(&doc, "p.a ~ span"), vec!["3"]);
    assert_eq!(qs_text(&doc, "p.b ~ p"), vec!["4"]);
    assert_eq!(qs_text(&doc, "span ~ p.a"), Vec::<String>::new());
    // Mixed with descendant and child combinators.
    assert_eq!(qs_text(&doc, "div p.a ~ p.b"), vec!["2", "4"]);
    assert_eq!(qs_text(&doc, "div > p.a + p"), vec!["2"]);
    assert_eq!(qs_text(&doc, "div > p + span"), vec!["3"]);
    // Chains of sibling combinators.
    assert_eq!(qs_text(&doc, "p.a + p.b + span"), vec!["3"]);
    assert_eq!(qs_text(&doc, "p.a ~ p.b + span"), vec!["3"]);
    // query_selector (right-to-left) agrees with query_selector_all.
    let first = doc
        .query_selector("p.a ~ p")
        .expect("valid selector")
        .unwrap();
    assert_eq!(doc.text_content(first), "2");
    let adj = doc
        .query_selector("span + p.b")
        .expect("valid selector")
        .unwrap();
    assert_eq!(doc.text_content(adj), "4");
}

#[test]
fn selector_error_type() {
    let (doc, _) = Document::parse_html(b"<div></div>");
    let err = doc.query_selector("div >").unwrap_err();
    assert_eq!(err.message(), "trailing combinator");
    assert_eq!(err.to_string(), "invalid selector: trailing combinator");
    // One bad group fails the whole list, even with valid groups around it.
    let err = doc
        .query_selector_all("div, [bad~=\"a b\"], p")
        .unwrap_err();
    assert!(err.message().contains("~="));
}

// -- foreign attribute namespaces (WHATWG HTML 13.2.6.1) --------------------

/// First descendant element with the given tag (depth-first).
fn descendant(doc: &Document, root: NodeId, tag_name: &str) -> Option<NodeId> {
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        if matches!(doc.kind(n), NodeKind::Element(_)) && tag(doc, n) == tag_name {
            return Some(n);
        }
        stack.extend(doc.children(n).into_iter().rev());
    }
    None
}

#[test]
fn foreign_attributes_gain_namespaces() {
    // The spec's "adjust foreign attributes" table, on an SVG element.
    let (doc, _) = Document::parse_html(
        b"<svg><a id=\"l\" xlink:href=\"/u\" xlink:title=\"t\" xml:lang=\"en\" xmlns=\"http://example.com\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" xmlns:foo=\"bar\" data-z=\"1\">x</a></svg>",
    );
    let a = doc.get_element_by_id("l").expect("a");
    assert_eq!(
        doc.get_attribute_ns(a, Some(XLINK_NAMESPACE), "href"),
        Some("/u")
    );
    assert_eq!(
        doc.get_attribute_ns(a, Some(XLINK_NAMESPACE), "title"),
        Some("t")
    );
    assert_eq!(
        doc.get_attribute_ns(a, Some(XML_NAMESPACE), "lang"),
        Some("en")
    );
    assert_eq!(
        doc.get_attribute_ns(a, Some(XMLNS_NAMESPACE), "xmlns"),
        Some("http://example.com")
    );
    assert_eq!(
        doc.get_attribute_ns(a, Some(XMLNS_NAMESPACE), "xlink"),
        Some("http://www.w3.org/1999/xlink")
    );
    // Outside the table, `xmlns:foo` stays unnamespaced.
    assert_eq!(doc.get_attribute_ns(a, None, "foo"), Some("bar"));
    assert_eq!(doc.get_attribute_ns(a, Some(XMLNS_NAMESPACE), "foo"), None);
    // Ordinary attributes are unnamespaced.
    assert_eq!(doc.get_attribute_ns(a, None, "data-z"), Some("1"));
    // The qualified name still resolves via getAttribute.
    assert_eq!(doc.get_attribute(a, "xlink:href"), Some("/u"));
}

#[test]
fn foreign_attribute_adjustment_is_foreign_only() {
    // `xlink:href` on an HTML element is NOT adjusted.
    let (doc, _) = Document::parse_html(b"<div id=\"h\" xlink:href=\"/u\"></div>");
    let div = doc.get_element_by_id("h").expect("div");
    assert_eq!(
        doc.get_attribute_ns(div, Some(XLINK_NAMESPACE), "href"),
        None
    );
    assert_eq!(doc.get_attribute_ns(div, None, "href"), Some("/u"));
}

#[test]
fn foreign_attributes_on_mathml() {
    let (doc, _) = Document::parse_html(b"<math><mi id=\"m\" xml:space=\"preserve\">x</mi></math>");
    let mi = doc.get_element_by_id("m").expect("mi");
    assert_eq!(
        doc.get_attribute_ns(mi, Some(XML_NAMESPACE), "space"),
        Some("preserve")
    );
}

#[test]
fn parsed_attr_namespace_vs_set_attribute() {
    // DOM `setAttribute` never assigns a namespace, even for qualified
    // names that the parser would adjust.
    let (doc, _) = Document::parse_html(b"<svg><a id=\"l\">x</a></svg>");
    let mut doc = doc;
    let a = doc.get_element_by_id("l").expect("a");
    doc.set_attribute(a, "xlink:href", "/v");
    assert_eq!(doc.get_attribute_ns(a, Some(XLINK_NAMESPACE), "href"), None);
    assert_eq!(doc.get_attribute_ns(a, None, "href"), Some("/v"));
}

#[test]
fn attribute_prefix_and_local_name() {
    let (doc, _) = Document::parse_html(b"<svg><a id=\"l\" xlink:href=\"/u\">x</a></svg>");
    let a = doc.get_element_by_id("l").expect("a");
    let attrs = match doc.kind(a) {
        NodeKind::Element(d) => &d.attributes,
        _ => panic!("expected element"),
    };
    let attr = attrs.iter().find(|x| x.name == "xlink:href").expect("attr");
    assert_eq!(attr.namespace, Some(XLINK_NAMESPACE));
    assert_eq!(attr.prefix(), Some("xlink"));
    assert_eq!(attr.local_name(), "href");
}

// -- declarative shadow roots (WHATWG HTML 4.12.3 / 13.2.6.4.7) -------------

#[test]
fn dsr_open_shadow_root() {
    let (doc, _) = Document::parse_html(
        b"<div id=\"h\"><template shadowrootmode=\"open\"><p>hi</p></template><b>light</b></div>",
    );
    let host = doc.get_element_by_id("h").expect("host");
    let shadow = doc.shadow_root(host).expect("shadow root");
    assert_eq!(shadow.mode, ShadowRootMode::Open);
    assert!(!shadow.delegates_focus);
    assert!(!shadow.clonable);
    // Shadow children live under the fragment.
    let kids = doc.children(shadow.root);
    assert_eq!(kids.len(), 1);
    assert_eq!(tag(&doc, kids[0]), "p");
    assert_eq!(doc.text_content(kids[0]), "hi");
    // The template element is gone from the light DOM; light children stay.
    let light: Vec<&str> = doc
        .children(host)
        .iter()
        .filter(|&&n| matches!(doc.kind(n), NodeKind::Element(_)))
        .map(|&n| tag(&doc, n))
        .collect();
    assert_eq!(light, vec!["b"]);
    assert!(descendant(&doc, host, "template").is_none());
}

#[test]
fn dsr_closed_with_flags() {
    let (doc, _) = Document::parse_html(
        b"<div id=\"h\"><template shadowrootmode=\"closed\" shadowrootdelegatesfocus shadowrootclonable><span>x</span></template></div>",
    );
    let host = doc.get_element_by_id("h").expect("host");
    let shadow = doc.shadow_root(host).expect("shadow root");
    assert_eq!(shadow.mode, ShadowRootMode::Closed);
    assert!(shadow.delegates_focus);
    assert!(shadow.clonable);
}

#[test]
fn dsr_ids_inside_shadow_are_unindexed() {
    // `getElementById` does not pierce shadow roots.
    let (doc, _) = Document::parse_html(
        b"<div id=\"h\"><template shadowrootmode=\"open\"><p id=\"inshadow\">x</p></template></div>",
    );
    assert!(doc.get_element_by_id("inshadow").is_none());
    assert!(doc.get_element_by_id("h").is_some());
}

#[test]
fn dsr_clone_does_not_alias_shadow() {
    let (doc, _) = Document::parse_html(
        b"<div id=\"h\"><template shadowrootmode=\"open\"><p>x</p></template></div>",
    );
    let mut doc = doc;
    let host = doc.get_element_by_id("h").expect("host");
    let before = doc.shadow_root(host).expect("shadow root").root;
    let copy = doc.clone_node(host, true);
    // The clone carries no shadow root (no aliased fragment).
    assert!(doc.shadow_root(copy).is_none());
    // The original is untouched.
    assert_eq!(doc.shadow_root(host).expect("shadow root").root, before);

    // A nested shadow host deep in the cloned subtree must not alias the
    // original's shadow fragment either.
    let (doc, _) = Document::parse_html(
        b"<div id=\"o\"><div id=\"i\"><template shadowrootmode=\"open\"><p>y</p></template></div></div>",
    );
    let mut doc = doc;
    let outer = doc.get_element_by_id("o").expect("outer");
    let inner = doc.get_element_by_id("i").expect("inner");
    let inner_before = doc.shadow_root(inner).expect("inner shadow root").root;
    let copy = doc.clone_node(outer, true);
    let inner_copy = descendant(&doc, copy, "div").expect("inner copy");
    assert_ne!(inner_copy, inner);
    assert!(
        doc.shadow_root(inner_copy).is_none(),
        "cloned nested host must not alias the original shadow fragment"
    );
    // The original is untouched.
    assert_eq!(
        doc.shadow_root(inner).expect("inner shadow root").root,
        inner_before
    );
}

#[test]
fn dsr_clone_shallow_copies_clonable_shadow() {
    // Per WHATWG DOM "clone a node", the shadow-root copy step is not
    // conditioned on `deep`: even `cloneNode(false)` copies a
    // `clonable` shadow tree.
    let (mut doc, _) = Document::parse_html(
        b"<div id=\"h\"><template shadowrootmode=\"closed\" shadowrootdelegatesfocus shadowrootclonable><p>hi</p></template><b>light</b></div>",
    );
    let host = doc.get_element_by_id("h").expect("host");
    let orig_frag = doc.shadow_root(host).expect("shadow root").root;

    let shallow = doc.clone_node(host, false);
    assert_ne!(shallow, host);
    assert_eq!(doc.parent(shallow), None); // detached
    assert_eq!(doc.get_attribute(shallow, "id"), Some("h"));
    // Shallow: light children are not copied...
    assert!(doc.children(shallow).is_empty());

    // ...but the clonable shadow tree is.
    let shadow = doc.shadow_root(shallow).expect("cloned shadow root");
    assert_ne!(shadow.root, orig_frag, "no aliased shadow fragment");
    assert_eq!(shadow.mode, ShadowRootMode::Closed);
    assert!(shadow.delegates_focus);
    assert!(shadow.clonable);
    let kids = doc.children(shadow.root);
    assert_eq!(kids.len(), 1);
    assert_eq!(tag(&doc, kids[0]), "p");
    assert_eq!(doc.text_content(kids[0]), "hi");

    // Shadow content is invisible to light-DOM traversal on the clone.
    assert!(doc.descendants(shallow).is_empty());
    assert_eq!(doc.text_content(shallow), "");
    assert!(
        doc.query_selector_all("p")
            .expect("selector parses")
            .is_empty(),
        "selectors must not pierce shadow roots (original or clone)"
    );
    // The original is untouched.
    assert_eq!(doc.shadow_root(host).expect("shadow root").root, orig_frag);
}

#[test]
fn dsr_clone_deep_copies_nested_clonable_hosts() {
    let (mut doc, _) = Document::parse_html(
        b"<div id=\"o\"><template shadowrootmode=\"open\" shadowrootclonable><section></section></template><p>light</p></div>",
    );
    let outer = doc.get_element_by_id("o").expect("outer");
    let outer_frag = doc.shadow_root(outer).expect("outer shadow").root;
    let inner = doc
        .children(outer_frag)
        .into_iter()
        .next()
        .expect("inner host");
    assert_eq!(tag(&doc, inner), "section");

    // Manually attach a nested clonable shadow root (the parser keeps
    // nested `<template shadowrootmode>` as plain template content).
    let inner_frag = doc.alloc(NodeKind::DocumentFragment);
    let span = doc.create_element("span");
    doc.set_attribute(span, "id", "nested");
    doc.append_child(inner_frag, span);
    doc.element_data_mut(inner).expect("inner data").shadow_root = Some(ShadowRootData {
        mode: ShadowRootMode::Open,
        delegates_focus: true,
        clonable: true,
        root: inner_frag,
    });

    let copy = doc.clone_node(outer, true);
    assert_ne!(copy, outer);
    assert_eq!(doc.parent(copy), None); // detached
    // Light DOM deep-copied exactly.
    assert_eq!(doc.serialize_node(copy), doc.serialize_node(outer));
    // Outer shadow copied, not aliased.
    let copy_shadow = doc.shadow_root(copy).expect("cloned outer shadow");
    assert_ne!(copy_shadow.root, outer_frag);
    assert_eq!(copy_shadow.mode, ShadowRootMode::Open);
    assert!(copy_shadow.clonable);
    // The nested host inside the copied shadow tree got its own copied
    // shadow root, with copied content.
    let inner_copy = doc
        .children(copy_shadow.root)
        .into_iter()
        .next()
        .expect("inner copy");
    assert_ne!(inner_copy, inner);
    assert_eq!(tag(&doc, inner_copy), "section");
    let inner_shadow = doc.shadow_root(inner_copy).expect("cloned inner shadow");
    assert_ne!(inner_shadow.root, inner_frag);
    assert_eq!(inner_shadow.mode, ShadowRootMode::Open);
    assert!(inner_shadow.delegates_focus);
    assert!(inner_shadow.clonable);
    let span_copy = doc
        .children(inner_shadow.root)
        .into_iter()
        .next()
        .expect("span copy");
    assert_ne!(span_copy, span);
    assert_eq!(tag(&doc, span_copy), "span");
    assert_eq!(doc.get_attribute(span_copy, "id"), Some("nested"));
    // Shadow ids stay out of the id index.
    assert!(doc.get_element_by_id("nested").is_none());
    // The original is untouched.
    assert_eq!(
        doc.shadow_root(outer).expect("outer shadow").root,
        outer_frag
    );
    assert_eq!(
        doc.shadow_root(inner).expect("inner shadow").root,
        inner_frag
    );
}

#[test]
fn dsr_clone_drops_non_clonable_shadow_shallow_too() {
    // A non-clonable shadow root is dropped by both shallow and deep
    // clones (the deep case is also covered by
    // `dsr_clone_does_not_alias_shadow`).
    let (mut doc, _) = Document::parse_html(
        b"<div id=\"h\"><template shadowrootmode=\"open\"><p>x</p></template></div>",
    );
    let host = doc.get_element_by_id("h").expect("host");
    let shallow = doc.clone_node(host, false);
    let deep = doc.clone_node(host, true);
    assert!(doc.shadow_root(shallow).is_none());
    assert!(doc.shadow_root(deep).is_none());
    assert!(doc.shadow_root(host).is_some(), "original untouched");
}

#[test]
fn dsr_clone_shadow_is_independent_of_original() {
    let (mut doc, _) = Document::parse_html(
        b"<div id=\"h\"><template shadowrootmode=\"open\" shadowrootclonable><p>hi</p></template></div>",
    );
    let host = doc.get_element_by_id("h").expect("host");
    let copy = doc.clone_node(host, true);
    let copy_p = doc
        .children(doc.shadow_root(copy).expect("cloned shadow").root)
        .into_iter()
        .next()
        .expect("cloned p");
    let orig_p = doc
        .children(doc.shadow_root(host).expect("shadow").root)
        .into_iter()
        .next()
        .expect("original p");
    // Mutating the clone's shadow leaves the original intact.
    doc.set_text_content(copy_p, "changed");
    assert_eq!(doc.text_content(copy_p), "changed");
    assert_eq!(doc.text_content(orig_p), "hi");
    doc.set_attribute(copy_p, "class", "c");
    assert_eq!(doc.get_attribute(orig_p, "class"), None);
}

#[test]
fn dsr_cycle_guard_rejects_host_inside_own_shadow() {
    // Regression: `shadow_root()` exposes the unlinked fragment, so a
    // consumer can `append_child(shadow_root(host).root, host)` and build
    // a host->shadow->host cycle. The insert-path cycle guard must see
    // through shadow edges and make that a no-op; otherwise
    // `clone_node(host, true)` loops forever (fresh allocs every lap) and
    // `serialize_node` overflows the stack. Before the guard fix this
    // test's `clone_node` call never returned.
    let mut doc = Document::new();
    let host = doc.create_element("div");
    let frag = doc.alloc(NodeKind::DocumentFragment);
    doc.element_data_mut(host).expect("host data").shadow_root = Some(ShadowRootData {
        mode: ShadowRootMode::Open,
        delegates_focus: false,
        clonable: true,
        root: frag,
    });
    let shadow = doc.shadow_root(host).expect("shadow root").root;
    // The hostile insert is a no-op: the host stays unparented and the
    // fragment stays empty.
    doc.append_child(shadow, host);
    assert_eq!(doc.parent(host), None);
    assert!(doc.children(shadow).is_empty());
    // Both traversal victims terminate on the (uncycled) tree.
    let copy = doc.clone_node(host, true);
    assert_ne!(copy, host);
    assert!(doc.shadow_root(copy).is_some());
    let _ = doc.serialize_node(host);
    // Sanity: ordinary inserts into the shadow fragment still work.
    let span = doc.create_element("span");
    doc.append_child(shadow, span);
    assert_eq!(doc.children(shadow), vec![span]);
}

#[test]
fn dsr_nested_template_stays_plain() {
    // The nested-declarative-shadow-root input through `DomSink`: the
    // inner template must not produce a nested shadow bracket (which used
    // to record the inner end on the outer host and orphan a fragment).
    let (doc, _) = Document::parse_html(
        b"<div id=\"h\"><template shadowrootmode=\"open\">A<template shadowrootmode=\"open\">B</template>C</template></div>",
    );
    let host = doc.get_element_by_id("h").expect("host");
    // Nothing leaked into the light DOM.
    assert!(doc.children(host).is_empty());
    let shadow = doc.shadow_root(host).expect("shadow root");
    assert_eq!(shadow.mode, ShadowRootMode::Open);
    let kids = doc.children(shadow.root);
    assert_eq!(kids.len(), 3);
    assert_eq!(doc.text_content(kids[0]), "A");
    // The inner template is a plain template element of the shadow tree.
    assert_eq!(tag(&doc, kids[1]), "template");
    assert!(doc.shadow_root(kids[1]).is_none());
    let frag = doc.template_contents(kids[1]).expect("template contents");
    assert_eq!(doc.text_content(frag), "B");
    assert_eq!(doc.text_content(kids[2]), "C");
}

#[test]
fn dsr_serialize_round_trip() {
    // `outer_html` re-emits a declarative shadow root as
    // `<template shadowrootmode>` (WHATWG HTML 13.3 step 4), first inside
    // the host, so parse→serialize→parse round-trips instead of silently
    // dropping the root.
    let (doc, _) = Document::parse_html(
        b"<div id=\"h\"><template shadowrootmode=\"open\" shadowrootdelegatesfocus shadowrootclonable><p>hi</p></template><b>light</b></div>",
    );
    let host = doc.get_element_by_id("h").expect("host");
    let html = doc.outer_html(host);
    assert_eq!(
        html,
        "<div id=\"h\"><template shadowrootmode=\"open\" shadowrootdelegatesfocus=\"\" shadowrootclonable=\"\"><p>hi</p></template><b>light</b></div>"
    );
    assert_eq!(
        doc.inner_html(host),
        "<template shadowrootmode=\"open\" shadowrootdelegatesfocus=\"\" shadowrootclonable=\"\"><p>hi</p></template><b>light</b>"
    );
    // Re-parsing reattaches the declarative shadow root with its flags.
    let (doc2, _) = Document::parse_html(html.as_bytes());
    let host2 = doc2.get_element_by_id("h").expect("host");
    let shadow = doc2.shadow_root(host2).expect("shadow root");
    assert_eq!(shadow.mode, ShadowRootMode::Open);
    assert!(shadow.delegates_focus);
    assert!(shadow.clonable);
    let kids = doc2.children(shadow.root);
    assert_eq!(kids.len(), 1);
    assert_eq!(tag(&doc2, kids[0]), "p");
    assert_eq!(doc2.text_content(kids[0]), "hi");
    let light: Vec<&str> = doc2
        .children(host2)
        .iter()
        .filter(|&&n| matches!(doc2.kind(n), NodeKind::Element(_)))
        .map(|&n| tag(&doc2, n))
        .collect();
    assert_eq!(light, vec!["b"]);
}

#[test]
fn dsr_serialize_closed_mode_round_trip() {
    let (doc, _) = Document::parse_html(
        b"<div id=\"h\"><template shadowrootmode=\"closed\"><span>x</span></template></div>",
    );
    let host = doc.get_element_by_id("h").expect("host");
    let html = doc.outer_html(host);
    assert_eq!(
        html,
        "<div id=\"h\"><template shadowrootmode=\"closed\"><span>x</span></template></div>"
    );
    let (doc2, _) = Document::parse_html(html.as_bytes());
    let host2 = doc2.get_element_by_id("h").expect("host");
    let shadow = doc2.shadow_root(host2).expect("shadow root");
    assert_eq!(shadow.mode, ShadowRootMode::Closed);
    assert_eq!(doc2.text_content(doc2.children(shadow.root)[0]), "x");
}

#[test]
fn dsr_no_shadow_without_valid_mode() {
    let (doc, _) =
        Document::parse_html(b"<div id=\"h\"><template shadowrootmode=\"bogus\"></template></div>");
    let host = doc.get_element_by_id("h").expect("host");
    assert!(doc.shadow_root(host).is_none());
    assert!(descendant(&doc, host, "template").is_some());
}

#[test]
fn dsr_clone_shadow_content_not_attached() {
    // Cloned shadow content stays unlinked and its children keep
    // `attached = false`, exactly like the parse-time shadow fragment:
    // it is out of reach of light-DOM traversal and the id index.
    let (mut doc, _) = Document::parse_html(
        b"<div id=\"h\"><template shadowrootmode=\"open\" shadowrootclonable><p><b>hi</b></p></template></div>",
    );
    let host = doc.get_element_by_id("h").expect("host");
    for deep in [false, true] {
        let copy = doc.clone_node(host, deep);
        assert!(!doc.is_attached(copy));
        let frag = doc.shadow_root(copy).expect("cloned shadow").root;
        assert!(!doc.is_attached(frag));
        let p = doc.children(frag).into_iter().next().expect("cloned p");
        assert!(!doc.is_attached(p));
        let b = doc.children(p).into_iter().next().expect("cloned b");
        assert!(!doc.is_attached(b));
    }
}

#[test]
fn dsr_clone_shallow_deep_copies_nested_clonable_host() {
    // Per WHATWG DOM "clone a node", the shadow-root copy step is not
    // conditioned on `deep`: a *shallow* clone of a host copies the whole
    // shadow tree, including a nested clonable host's shadow.
    let (mut doc, _) = Document::parse_html(
        b"<div id=\"o\"><template shadowrootmode=\"open\" shadowrootclonable><section></section></template><p>light</p></div>",
    );
    let outer = doc.get_element_by_id("o").expect("outer");
    let outer_frag = doc.shadow_root(outer).expect("outer shadow").root;
    let inner = doc
        .children(outer_frag)
        .into_iter()
        .next()
        .expect("inner host");

    // Manually attach a nested clonable shadow root (the parser keeps
    // nested `<template shadowrootmode>` as plain template content).
    let inner_frag = doc.alloc(NodeKind::DocumentFragment);
    let span = doc.create_element("span");
    doc.set_attribute(span, "id", "nested");
    doc.append_child(inner_frag, span);
    doc.element_data_mut(inner).expect("inner data").shadow_root = Some(ShadowRootData {
        mode: ShadowRootMode::Open,
        delegates_focus: false,
        clonable: true,
        root: inner_frag,
    });

    let shallow = doc.clone_node(outer, false);
    // The light-DOM child is not copied by a shallow clone...
    assert!(doc.children(shallow).is_empty());
    // ...but the shadow tree is always deep-copied.
    let copy_shadow = doc.shadow_root(shallow).expect("cloned outer shadow");
    assert_ne!(copy_shadow.root, outer_frag);
    assert_eq!(copy_shadow.mode, ShadowRootMode::Open);
    assert!(copy_shadow.clonable);
    let inner_copy = doc
        .children(copy_shadow.root)
        .into_iter()
        .next()
        .expect("inner copy");
    assert_ne!(inner_copy, inner);
    let inner_shadow = doc.shadow_root(inner_copy).expect("cloned inner shadow");
    assert_ne!(inner_shadow.root, inner_frag);
    assert!(inner_shadow.clonable);
    let span_copy = doc
        .children(inner_shadow.root)
        .into_iter()
        .next()
        .expect("span copy");
    assert_eq!(doc.get_attribute(span_copy, "id"), Some("nested"));
    assert!(!doc.is_attached(span_copy));
    // Shadow ids stay out of the id index.
    assert!(doc.get_element_by_id("nested").is_none());
    // The original is untouched.
    assert_eq!(
        doc.shadow_root(inner).expect("inner shadow").root,
        inner_frag
    );
}

#[test]
fn dsr_clone_node_inside_shadow_tree() {
    // `clone_node` works on a node that itself lives inside a shadow
    // tree (shallow and deep); the clone is detached, independent, and
    // carries the node's own clonable shadow root along.
    let (mut doc, _) = Document::parse_html(
        b"<div id=\"h\"><template shadowrootmode=\"open\" shadowrootclonable><p id=\"inner\"><b>x</b></p></template></div>",
    );
    let host = doc.get_element_by_id("h").expect("host");
    let frag = doc.shadow_root(host).expect("shadow").root;
    let p = doc.children(frag).into_iter().next().expect("shadow p");
    assert!(!doc.is_attached(p));

    // Give the shadow-resident node its own nested clonable shadow root.
    let p_frag = doc.alloc(NodeKind::DocumentFragment);
    let i = doc.create_element("i");
    doc.append_child(p_frag, i);
    let y = doc.create_text_node("y");
    doc.append_child(i, y);
    doc.element_data_mut(p).expect("p data").shadow_root = Some(ShadowRootData {
        mode: ShadowRootMode::Open,
        delegates_focus: false,
        clonable: true,
        root: p_frag,
    });

    let shallow = doc.clone_node(p, false);
    assert_ne!(shallow, p);
    assert_eq!(tag(&doc, shallow), "p");
    assert_eq!(doc.get_attribute(shallow, "id"), Some("inner"));
    assert!(doc.children(shallow).is_empty());
    assert!(!doc.is_attached(shallow));
    // The shadow-root copy step applies even to the shallow clone.
    let shallow_shadow = doc.shadow_root(shallow).expect("shallow shadow");
    assert_ne!(shallow_shadow.root, p_frag);
    assert_eq!(doc.text_content(shallow_shadow.root), "y");

    let deep = doc.clone_node(p, true);
    assert_eq!(doc.get_attribute(deep, "id"), Some("inner"));
    assert_eq!(doc.text_content(deep), "x");
    let b_copy = doc.children(deep).into_iter().next().expect("b copy");
    assert_ne!(b_copy, doc.children(p).into_iter().next().unwrap());
    // Mutating the deep clone's child leaves the shadow original intact.
    doc.set_text_content(b_copy, "changed");
    assert_eq!(doc.text_content(deep), "changed");
    assert_eq!(
        doc.text_content(doc.children(p).into_iter().next().unwrap()),
        "x"
    );
}

#[test]
fn dsr_clone_attach_keeps_shadow_ids_out_of_index() {
    // Attaching a clone into the document must not index the cloned
    // shadow tree's ids: shadow content is out of reach of the id index.
    let (mut doc, _) = Document::parse_html(
        b"<div id=\"h\"><template shadowrootmode=\"open\" shadowrootclonable><p id=\"sh\">x</p></template></div>",
    );
    let host = doc.get_element_by_id("h").expect("host");
    let copy = doc.clone_node(host, true);
    assert!(doc.get_element_by_id("sh").is_none());
    // A second element under the Document node would violate DOM
    // pre-insert validity, so attach under <html> instead.
    let html = doc.document_element().expect("html");
    doc.append_child(html, copy);
    assert!(doc.is_attached(copy));
    assert!(
        doc.get_element_by_id("sh").is_none(),
        "shadow ids stay unindexed after the clone is attached"
    );
}

#[test]
fn dsr_clone_does_not_steal_id_index() {
    // The clone carries `id="h"` too, but it is detached, so
    // `get_element_by_id("h")` keeps resolving to the original host.
    let (mut doc, _) = Document::parse_html(
        b"<div id=\"h\"><template shadowrootmode=\"open\" shadowrootclonable><p>hi</p></template></div>",
    );
    let host = doc.get_element_by_id("h").expect("host");
    let copy = doc.clone_node(host, true);
    assert_ne!(copy, host);
    assert_eq!(doc.get_element_by_id("h"), Some(host));
    assert_eq!(doc.get_attribute(copy, "id"), Some("h"));
}

// -- touch events ------------------------------------------------------------

#[test]
fn touch_phase_maps_to_event_types() {
    assert_eq!(TouchPhase::Start.event_type(), "touchstart");
    assert_eq!(TouchPhase::Move.event_type(), "touchmove");
    assert_eq!(TouchPhase::End.event_type(), "touchend");
}

#[test]
fn touch_list_item_and_len() {
    let mut doc = Document::new();
    let div = doc.create_element("div");
    let mut list = TouchList::new();
    assert!(list.is_empty());
    assert_eq!(list.len(), 0);
    assert_eq!(list.item(0), None);
    list.push(Touch::new(7, Some(div), 10.0, 20.0, 11.0, 21.0, 12.0, 22.0));
    list.push(Touch::new(8, None, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0));
    assert_eq!(list.len(), 2);
    assert!(!list.is_empty());
    let first = list.item(0).unwrap();
    assert_eq!(first.identifier, 7);
    assert_eq!(first.target, Some(div));
    assert_eq!(first.client_x, 10.0);
    assert_eq!(first.client_y, 20.0);
    assert_eq!(first.screen_x, 11.0);
    assert_eq!(first.screen_y, 21.0);
    assert_eq!(first.page_x, 12.0);
    assert_eq!(first.page_y, 22.0);
    assert_eq!(list.item(1).unwrap().identifier, 8);
    assert_eq!(list.item(2), None);
    assert_eq!(list.iter().count(), 2);
}

#[test]
fn touch_event_fields_bubble_cancelable_and_prevent_default() {
    let changed = TouchList::from_touches([Touch::new(1, None, 5.0, 6.0, 0.0, 0.0, 5.0, 6.0)]);
    let mut e = TouchEvent::new(
        TouchPhase::Start,
        TouchList::new(),
        TouchList::new(),
        changed,
        true,
        false,
        false,
        true,
    );
    // Per the Touch Events spec every touch event bubbles and is
    // cancelable: preventDefault on touchstart cancels the default pan.
    assert_eq!(e.event.event_type, "touchstart");
    assert!(e.event.bubbles);
    assert!(e.event.cancelable);
    assert!(!e.event.default_prevented);
    assert!(e.touches.is_empty());
    assert_eq!(e.changed_touches.len(), 1);
    assert!(e.alt_key);
    assert!(!e.ctrl_key);
    assert!(!e.meta_key);
    assert!(e.shift_key);
    e.event.prevent_default();
    assert!(e.event.default_prevented);
}

// -- pointer / mouse events --------------------------------------------------

#[test]
fn pointer_type_strings_round_trip() {
    assert_eq!(PointerType::Mouse.as_str(), "mouse");
    assert_eq!(PointerType::Pen.as_str(), "pen");
    assert_eq!(PointerType::Touch.as_str(), "touch");
    assert_eq!(PointerType::parse("mouse"), PointerType::Mouse);
    assert_eq!(PointerType::parse("pen"), PointerType::Pen);
    assert_eq!(PointerType::parse("touch"), PointerType::Touch);
    // Unknown devices fall back to the mouse default.
    assert_eq!(PointerType::parse(""), PointerType::Mouse);
    assert_eq!(PointerType::default(), PointerType::Mouse);
}

#[test]
fn pointer_data_mouse_at_and_event_attachment() {
    let p = PointerData::mouse_at(12.5, -3.0);
    assert_eq!(p.client_x, 12.5);
    assert_eq!(p.client_y, -3.0);
    assert_eq!(p.screen_x, 12.5);
    assert_eq!(p.page_y, -3.0);
    assert_eq!(p.button, -1);
    assert_eq!(p.buttons, 0);
    assert!(p.is_primary);
    assert_eq!(p.width, 1.0);
    assert_eq!(p.height, 1.0);
    assert_eq!(p.pointer_type, PointerType::Mouse);

    let bare = Event::new("mousedown", true, true);
    assert_eq!(bare.pointer, None);
    let with = Event::new("mousedown", true, true).with_pointer(p);
    assert_eq!(with.pointer, Some(p));
    // The zero state: a default PointerData reads as the spec zeros.
    let d = PointerData::default();
    assert_eq!(d.client_x, 0.0);
    assert_eq!(d.pointer_id, 0);
    assert!(!d.is_primary);
}

// -- wave 36: per-node allocation trimming -----------------------------------

#[test]
fn repeated_short_text_runs_share_one_allocation() {
    let (doc, _) = Document::parse_html(
        b"<html>\n  <head>\n    <title>t</title>\n  </head>\n  <body>\n    <p>a</p>\n    <p>b</p>\n  </body>\n</html>",
    );
    let mut payloads: Vec<Arc<str>> = Vec::new();
    for n in doc.descendants(doc.root()) {
        if let NodeKind::Text(t) = doc.kind(n) {
            payloads.push(t.clone());
        }
    }
    assert!(payloads.len() > 4, "expected several text runs");
    // Every repeated run must be pointer-identical: one allocation per
    // distinct value, not one per node.
    for i in 0..payloads.len() {
        for j in (i + 1)..payloads.len() {
            if payloads[i].as_ref() == payloads[j].as_ref() {
                assert!(
                    Arc::ptr_eq(&payloads[i], &payloads[j]),
                    "repeated text {:?} should share one allocation",
                    payloads[i].as_ref()
                );
            }
        }
    }
    // Sanity: the document really does contain repeated whitespace runs.
    let indent2 = payloads.iter().filter(|t| t.as_ref() == "\n  ").count();
    assert!(indent2 >= 2, "expected repeated indent runs");
}

#[test]
fn create_text_node_interns_short_payloads() {
    let mut doc = Document::new();
    let a = doc.create_text_node("hello");
    let b = doc.create_text_node("hello");
    let (pa, pb) = match (doc.kind(a), doc.kind(b)) {
        (NodeKind::Text(x), NodeKind::Text(y)) => (x.clone(), y.clone()),
        _ => panic!("expected text nodes"),
    };
    assert!(Arc::ptr_eq(&pa, &pb));
    // Long payloads skip the cache (bounded hash cost and entry size)
    // but still compare equal.
    let long = "x".repeat(128);
    let c = doc.create_text_node(&long);
    let d = doc.create_text_node(&long);
    let (pc, pd) = match (doc.kind(c), doc.kind(d)) {
        (NodeKind::Text(x), NodeKind::Text(y)) => (x.clone(), y.clone()),
        _ => panic!("expected text nodes"),
    };
    assert!(!Arc::ptr_eq(&pc, &pd));
    assert_eq!(pc.as_ref(), pd.as_ref());
}

#[test]
fn set_text_content_shares_interned_payload() {
    let mut doc = Document::new();
    let t = doc.create_text_node("old");
    doc.set_text_content(t, "new");
    let u = doc.create_text_node("new");
    let (pt, pu) = match (doc.kind(t), doc.kind(u)) {
        (NodeKind::Text(x), NodeKind::Text(y)) => (x.clone(), y.clone()),
        _ => panic!("expected text nodes"),
    };
    assert!(Arc::ptr_eq(&pt, &pu));
    assert_eq!(doc.text_content(t), "new");
    // Setting identical text is a no-op: the payload (and its sharing)
    // is untouched.
    doc.set_text_content(t, "new");
    let after = match doc.kind(t) {
        NodeKind::Text(x) => x.clone(),
        _ => panic!("expected text node"),
    };
    assert!(Arc::ptr_eq(&pt, &after));
}

#[test]
fn clone_node_shares_text_payload() {
    let mut doc = Document::new();
    let t = doc.create_text_node("shared");
    let c = doc.clone_node(t, false);
    let (pt, pc) = match (doc.kind(t), doc.kind(c)) {
        (NodeKind::Text(x), NodeKind::Text(y)) => (x.clone(), y.clone()),
        _ => panic!("expected text nodes"),
    };
    // Cloning a text node is a refcount bump, not a string copy.
    assert!(Arc::ptr_eq(&pt, &pc));
}

#[test]
fn attribute_stores_no_prefix_or_local_name_strings() {
    // prefix/local_name are derived from the interned qualified name, so
    // the struct lost both stored Strings (24 B each on 64-bit).
    assert!(std::mem::size_of::<Attribute>() <= 64);
    let a = Attribute::plain(AttrName::from("xlink:href"), "u".into());
    assert_eq!(a.namespace, None);
    assert_eq!(a.prefix(), Some("xlink"));
    assert_eq!(a.local_name(), "href");
    let b = Attribute::plain(AttrName::from("id"), "v".into());
    assert_eq!(b.prefix(), None);
    assert_eq!(b.local_name(), "id");
}

#[test]
fn text_intern_cache_stays_bounded_and_correct() {
    let mut doc = Document::new();
    // More distinct short texts than the cache cap: the cache clears
    // instead of growing without bound, and every node still reads back
    // its own text.
    let nodes: Vec<NodeId> = (0..3000)
        .map(|i| doc.create_text_node(&format!("item-{i:04}")))
        .collect();
    for (i, n) in nodes.iter().enumerate() {
        let expect = format!("item-{i:04}");
        match doc.kind(*n) {
            NodeKind::Text(t) => assert_eq!(t.as_ref(), expect),
            _ => panic!("expected text node"),
        }
    }
    // A value still resident in the cache shares on re-intern.
    let a = doc.create_text_node("item-2999");
    let (pa, pb) = match (doc.kind(a), doc.kind(nodes[2999])) {
        (NodeKind::Text(x), NodeKind::Text(y)) => (x.clone(), y.clone()),
        _ => panic!("expected text nodes"),
    };
    assert!(Arc::ptr_eq(&pa, &pb));
}

// -- imperative shadow roots (attachShadow) --------------------------------

#[test]
fn attach_shadow_root_open_and_closed() {
    let (mut doc, _) = Document::parse_html(
        b"<html><body><div id=\"h1\"></div><div id=\"h2\"></div></body></html>",
    );
    let h1 = doc.get_element_by_id("h1").expect("h1");
    let h2 = doc.get_element_by_id("h2").expect("h2");
    // Open mode.
    let frag = doc
        .attach_shadow_root(h1, ShadowRootMode::Open)
        .expect("attach open");
    assert!(matches!(doc.kind(frag), NodeKind::DocumentFragment));
    let data = doc.shadow_root(h1).expect("shadow data");
    assert_eq!(data.mode, ShadowRootMode::Open);
    assert_eq!(data.root, frag);
    // The fragment is unlinked: not a light-DOM child of the host.
    assert_eq!(doc.parent(frag), None);
    assert!(doc.children(h1).is_empty());
    // Closed mode.
    let frag2 = doc
        .attach_shadow_root(h2, ShadowRootMode::Closed)
        .expect("attach closed");
    assert_eq!(
        doc.shadow_root(h2).expect("shadow data").mode,
        ShadowRootMode::Closed
    );
    assert_ne!(frag2, frag);
}

#[test]
fn attach_shadow_root_rejects_non_element_and_double_attach() {
    let (mut doc, _) = Document::parse_html(b"<html><body><div id=\"h\"></div>text</body></html>");
    let h = doc.get_element_by_id("h").expect("h");
    // Non-elements (document node, text node) are rejected.
    let root = doc.root();
    assert_eq!(doc.attach_shadow_root(root, ShadowRootMode::Open), None);
    let text = doc
        .children(doc.get_elements_by_tag_name("body").item(0).expect("body"))
        .into_iter()
        .find(|&n| matches!(doc.kind(n), NodeKind::Text(_)))
        .expect("text");
    assert_eq!(doc.attach_shadow_root(text, ShadowRootMode::Open), None);
    // A host can only have one shadow root.
    assert!(doc.attach_shadow_root(h, ShadowRootMode::Open).is_some());
    assert_eq!(doc.attach_shadow_root(h, ShadowRootMode::Closed), None);
    // The first root survives the rejected second attach.
    assert_eq!(
        doc.shadow_root(h).expect("shadow").mode,
        ShadowRootMode::Open
    );
}

#[test]
fn shadow_host_for_retargets_descendants() {
    let (mut doc, _) = Document::parse_html(b"<html><body><div id=\"h\"></div></body></html>");
    let h = doc.get_element_by_id("h").expect("h");
    let frag = doc
        .attach_shadow_root(h, ShadowRootMode::Closed)
        .expect("attach");
    // Plain light-DOM nodes have no host.
    assert_eq!(doc.shadow_host_for(h), None);
    // The fragment itself maps to the host.
    assert_eq!(doc.shadow_host_for(frag), Some(h));
    // Descendants at any depth retarget to the host.
    let outer = doc.create_element("section");
    let inner = doc.create_element("span");
    doc.append_child(outer, inner);
    doc.append_child(frag, outer);
    assert_eq!(doc.shadow_host_for(outer), Some(h));
    assert_eq!(doc.shadow_host_for(inner), Some(h));
    // A node in a different (unrelated) tree still has no host.
    let other = doc.create_element("div");
    assert_eq!(doc.shadow_host_for(other), None);
}

#[test]
fn shadow_tree_mutation_dirties_host() {
    let (mut doc, _) = Document::parse_html(b"<html><body><div id=\"h\"></div></body></html>");
    let h = doc.get_element_by_id("h").expect("h");
    let frag = doc
        .attach_shadow_root(h, ShadowRootMode::Open)
        .expect("attach");
    // Drain the attach dirt.
    let _ = doc.take_dirty();
    // Appending into the shadow fragment dirties the host (the fragment
    // itself is unlinked, so plain mark_dirty would record nothing).
    let kid = doc.create_element("span");
    doc.append_child(frag, kid);
    let report = doc.take_dirty();
    assert!(
        report.subtrees.contains(&h),
        "host dirtied by shadow append"
    );
    // Removal too.
    let _ = doc.take_dirty();
    doc.remove_child(frag, kid);
    let report = doc.take_dirty();
    assert!(
        report.subtrees.contains(&h),
        "host dirtied by shadow remove"
    );
}

#[test]
fn clone_registers_cloned_shadow_root() {
    let (mut doc, _) = Document::parse_html(b"<html><body><div id=\"h\"></div></body></html>");
    let h = doc.get_element_by_id("h").expect("h");
    // attach_shadow_root makes a non-clonable root; flip clonable on for
    // the clone exercise (mirrors the parser's shadowrootclonable).
    let frag = doc
        .attach_shadow_root(h, ShadowRootMode::Open)
        .expect("attach");
    doc.element_data_mut(h)
        .expect("data")
        .shadow_root
        .as_mut()
        .expect("shadow")
        .clonable = true;
    let kid = doc.create_element("b");
    doc.append_child(frag, kid);
    let copy = doc.clone_node(h, true);
    assert_ne!(copy, h);
    let copy_frag = doc.shadow_root(copy).expect("cloned shadow").root;
    assert_ne!(copy_frag, frag);
    // The cloned fragment is registered: retargeting works on the copy.
    assert_eq!(doc.shadow_host_for(copy_frag), Some(copy));
    let kid_copy = doc
        .children(copy_frag)
        .into_iter()
        .next()
        .expect("kid copy");
    assert_eq!(doc.shadow_host_for(kid_copy), Some(copy));
    // The original mapping is untouched.
    assert_eq!(doc.shadow_host_for(frag), Some(h));
}

#[test]
fn document_insert_rejects_live_shadow_root_fragment() {
    // A raw `Document` caller could pass a live shadow-root fragment to
    // the insertion methods; without a guard the fragment would be
    // spliced (emptied) while `shadow_hosts` still maps it. The
    // insertion methods no-op instead, like the other validity
    // failures.
    let (mut doc, _) = Document::parse_html(b"<html><body><div id=\"h\"></div></body></html>");
    let h = doc.get_element_by_id("h").expect("h");
    let frag = doc
        .attach_shadow_root(h, ShadowRootMode::Open)
        .expect("attach");
    assert!(doc.is_shadow_root(frag));
    // A shadow child makes a splice observable.
    let kid = doc.create_element("b");
    doc.append_child(frag, kid);
    assert_eq!(doc.children(frag), vec![kid]);

    let body = doc.get_elements_by_tag_name("body").item(0).expect("body");

    // append_child: no-op; the shadow tree is untouched.
    doc.append_child(body, frag);
    assert_eq!(doc.children(frag), vec![kid]);
    assert!(doc.is_shadow_root(frag));
    assert!(!doc.children(body).contains(&frag));

    // insert_before: no-op.
    let first = doc.children(body)[0];
    doc.insert_before(body, frag, Some(first));
    assert_eq!(doc.children(frag), vec![kid]);
    assert!(!doc.children(body).contains(&frag));

    // replace_child: no-op; the old child survives in place.
    let old = doc.children(body)[0];
    doc.replace_child(body, frag, old);
    assert_eq!(doc.children(frag), vec![kid]);
    assert!(doc.children(body).contains(&old));
    assert!(!doc.children(body).contains(&frag));

    // The still-live shadow root keeps accepting real inserts.
    let kid2 = doc.create_element("i");
    doc.append_child(frag, kid2);
    assert_eq!(doc.children(frag), vec![kid, kid2]);
}
