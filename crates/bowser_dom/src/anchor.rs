//! `HTMLAnchorElement` URL decomposition per WHATWG HTML
//! (`HTMLHyperlinkElementUtils`).
//!
//! Pure DOM-level helpers over [`Document`]/[`NodeId`]: the `href` content
//! attribute is the source of truth (the `href` IDL setter reflects it),
//! and the URL-decomposition getters parse it against the document base
//! URL. The JS bindings (`bowser_js`) install the `href`/`search`
//! accessors on `<a>` element wrappers and supply the base URL.
//!
//! Query percent-encoding uses the URL Standard's encoding override: the
//! document encoding. The query text is converted to bytes with the
//! document encoding's encoder ([`bowser_encoding::encode`]) before
//! percent-encoding ("percent-encode after encoding"); encodings without
//! an encoder fall back to UTF-8.

use bowser_encoding::Encoding;
use bowser_url::Url;

use crate::{Document, Namespace, NodeId, NodeKind};

/// The URL Standard's "percent-encode after encoding" for `encoding`:
/// convert query text to bytes with the encoding's encoder, falling back
/// to UTF-8 when the encoding has no encoder.
fn query_encoder(encoding: Encoding) -> impl Fn(&str, &mut Vec<u8>) {
    move |s: &str, out: &mut Vec<u8>| {
        if !bowser_encoding::encode_into(s, encoding, out) {
            out.extend_from_slice(s.as_bytes());
        }
    }
}

/// Whether `node` is an HTML `<a>` element: the only element these
/// helpers apply to. (`HTMLAreaElement` shares the spec mixin but has no
/// binding yet.)
pub fn is_anchor_element(doc: &Document, node: NodeId) -> bool {
    matches!(
        doc.kind(node),
        NodeKind::Element(d) if d.namespace == Namespace::Html && d.tag_name.as_str() == "a"
    )
}

/// The `href` content attribute value, if the element carries one.
pub fn anchor_href(doc: &Document, node: NodeId) -> Option<&str> {
    doc.get_attribute(node, "href")
}

/// The `href` IDL setter: reflects to the content attribute (WHATWG HTML
/// sets the attribute; the URL is re-parsed from it on demand).
pub fn set_anchor_href(doc: &mut Document, node: NodeId, value: &str) {
    doc.set_attribute(node, "href", value);
}

/// The `href` IDL getter: the serialization of the element's URL, or the
/// raw content-attribute value when it does not parse (missing attribute
/// reads as the empty string).
pub fn anchor_href_serialized(
    doc: &Document,
    node: NodeId,
    base: &Url,
    encoding: Encoding,
) -> String {
    match anchor_url(doc, node, base, encoding) {
        Some(url) => url.serialize(),
        None => anchor_href(doc, node).unwrap_or("").to_owned(),
    }
}

/// The element's URL: the `href` content attribute parsed against `base`.
/// `None` when the attribute is absent or fails to parse (the spec's
/// "element does not have a URL").
fn anchor_url(doc: &Document, node: NodeId, base: &Url, encoding: Encoding) -> Option<Url> {
    let href = anchor_href(doc, node)?;
    bowser_url::parse_with_query_encoding(href, Some(base), &query_encoder(encoding)).ok()
}

/// The `search` IDL getter (WHATWG HTML): the query with its leading `?`,
/// or the empty string when the element has no URL or the URL has no
/// query. A present-but-empty query (`...?`) reads back as `"?"`.
pub fn anchor_search(doc: &Document, node: NodeId, base: &Url, encoding: Encoding) -> String {
    match anchor_url(doc, node, base, encoding).and_then(|u| u.query().map(str::to_owned)) {
        Some(query) => format!("?{query}"),
        None => String::new(),
    }
}

/// The `search` IDL setter (WHATWG HTML `HTMLHyperlinkElementUtils`):
/// when the element has a URL, basic-URL-parse `value` with the query
/// state override and write the result back to the `href` content
/// attribute. No-op when the element has no URL — that is the mixin's
/// only early return. Opaque-path (cannot-be-a-base) URLs are *not*
/// excluded: the URL Standard's `search` setter applies the state
/// override to them too, and [`Url::set_search`] matches that (unlike
/// the `host`/`hostname` setters, which do return early on opaque
/// paths).
pub fn set_anchor_search(
    doc: &mut Document,
    node: NodeId,
    base: &Url,
    encoding: Encoding,
    value: &str,
) {
    if let Some(mut url) = anchor_url(doc, node, base, encoding) {
        url.set_search_with_encoding(value, &query_encoder(encoding));
        let serialized = url.serialize();
        set_anchor_href(doc, node, &serialized);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Url {
        bowser_url::parse("https://example.com/dir/page.html", None).unwrap()
    }

    fn anchor(doc: &mut Document, href: Option<&str>) -> NodeId {
        let el = doc.create_element("a");
        if let Some(h) = href {
            doc.set_attribute(el, "href", h);
        }
        el
    }

    #[test]
    fn search_getter_returns_query_with_leading_question_mark() {
        let mut doc = Document::new();
        let el = anchor(&mut doc, Some("https://example.com/?x=1&y=2"));
        assert_eq!(anchor_search(&doc, el, &base(), Encoding::Utf8), "?x=1&y=2");
    }

    #[test]
    fn search_getter_empty_without_query() {
        let mut doc = Document::new();
        let el = anchor(&mut doc, Some("https://example.com/path"));
        assert_eq!(anchor_search(&doc, el, &base(), Encoding::Utf8), "");
    }

    #[test]
    fn search_getter_empty_without_href() {
        let mut doc = Document::new();
        let el = anchor(&mut doc, None);
        assert_eq!(anchor_search(&doc, el, &base(), Encoding::Utf8), "");
    }

    #[test]
    fn search_getter_empty_on_unparseable_href() {
        let mut doc = Document::new();
        let el = anchor(&mut doc, Some("https://exa mple.com/?x=1"));
        assert_eq!(anchor_search(&doc, el, &base(), Encoding::Utf8), "");
    }

    #[test]
    fn search_getter_resolves_relative_href_against_base() {
        let mut doc = Document::new();
        let el = anchor(&mut doc, Some("other?z=9"));
        assert_eq!(anchor_search(&doc, el, &base(), Encoding::Utf8), "?z=9");
    }

    #[test]
    fn search_getter_keeps_bare_question_mark() {
        // `?` with an empty query round-trips as `"?"` (query is
        // present-but-empty, not absent).
        let mut doc = Document::new();
        let el = anchor(&mut doc, Some("https://example.com/?"));
        assert_eq!(anchor_search(&doc, el, &base(), Encoding::Utf8), "?");
    }

    #[test]
    fn search_setter_replaces_query() {
        let mut doc = Document::new();
        let el = anchor(&mut doc, Some("https://example.com/a?x=1#frag"));
        set_anchor_search(&mut doc, el, &base(), Encoding::Utf8, "y=2");
        assert_eq!(anchor_search(&doc, el, &base(), Encoding::Utf8), "?y=2");
        // Path and fragment survive; the href attribute holds the update.
        assert_eq!(
            anchor_href(&doc, el),
            Some("https://example.com/a?y=2#frag")
        );
    }

    #[test]
    fn search_setter_strips_leading_question_mark() {
        let mut doc = Document::new();
        let el = anchor(&mut doc, Some("https://example.com/a?x=1"));
        set_anchor_search(&mut doc, el, &base(), Encoding::Utf8, "?y=2");
        assert_eq!(anchor_search(&doc, el, &base(), Encoding::Utf8), "?y=2");
    }

    #[test]
    fn search_setter_noop_without_url() {
        let mut doc = Document::new();
        let el = anchor(&mut doc, None);
        set_anchor_search(&mut doc, el, &base(), Encoding::Utf8, "y=2");
        assert_eq!(anchor_href(&doc, el), None);
        assert_eq!(anchor_search(&doc, el, &base(), Encoding::Utf8), "");
    }

    #[test]
    fn href_reflects_content_attribute() {
        let mut doc = Document::new();
        let el = anchor(&mut doc, None);
        set_anchor_href(&mut doc, el, "https://example.com/?q=1");
        assert_eq!(anchor_href(&doc, el), Some("https://example.com/?q=1"));
        assert_eq!(
            anchor_href_serialized(&doc, el, &base(), Encoding::Utf8),
            "https://example.com/?q=1"
        );
    }

    #[test]
    fn href_getter_falls_back_to_raw_attribute() {
        let mut doc = Document::new();
        // Unparseable even against a base (bad IPv6 host): the getter
        // returns the raw attribute value.
        let el = anchor(&mut doc, Some("https://[::1/?x=1"));
        assert_eq!(
            anchor_href_serialized(&doc, el, &base(), Encoding::Utf8),
            "https://[::1/?x=1"
        );
    }

    #[test]
    fn is_anchor_element_only_matches_html_a() {
        let mut doc = Document::new();
        let a = doc.create_element("a");
        let div = doc.create_element("div");
        assert!(is_anchor_element(&doc, a));
        assert!(!is_anchor_element(&doc, div));
    }

    #[test]
    fn big5_document_encodes_query_as_big5() {
        // WPT encoding/big5-encoder.html: the document encoding is the
        // URL Standard's encoding override for the query.
        let mut doc = Document::new();
        doc.set_encoding(Encoding::Big5);
        // U+3000 -> A1 40 ('@' is query-allowed); U+2603 unmapped -> NCR.
        let el = anchor(&mut doc, Some("https://example.com/?X\u{3000}X"));
        assert_eq!(anchor_search(&doc, el, &base(), Encoding::Big5), "?X%A1@X");
        let el = anchor(&mut doc, Some("https://example.com/?\u{2603}"));
        assert_eq!(
            anchor_search(&doc, el, &base(), Encoding::Big5),
            "?%26%239731%3B"
        );
        // The search setter encodes with the override too.
        let el = anchor(&mut doc, Some("https://example.com/"));
        set_anchor_search(&mut doc, el, &base(), Encoding::Big5, "\u{3000}");
        assert_eq!(anchor_search(&doc, el, &base(), Encoding::Big5), "?%A1@");
    }

    #[test]
    fn parse_html_records_document_encoding() {
        let (doc, _) = Document::parse_html(
            b"<meta charset=big5><a href=\"https://example.com/?\xE3\x80\x80\">x</a>",
        );
        assert_eq!(doc.encoding(), Encoding::Big5);
        let (doc, _) = Document::parse_html(b"<meta charset=\"windows-1252\">");
        assert_eq!(doc.encoding(), Encoding::Windows1252);
        let (doc, _) = Document::parse_html(b"<p>plain utf-8: \xC3\xA9</p>");
        assert_eq!(doc.encoding(), Encoding::Utf8);
    }
}
