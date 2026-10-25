//! Tree construction stage (WHATWG HTML 13.2.6).
//!
//! Implements the insertion modes, the adoption agency algorithm, foster
//! parenting, the list of active formatting elements (with reconstruction),
//! template contents, and foreign (MathML/SVG) content, and drives the
//! finished tree into a [`TreeSink`](crate::sink::TreeSink) in a single
//! pass: string payloads move out of the builder's working arena straight
//! into the sink, with no intermediate owned tree.
//!
//! The simple owned [`Node`] tree is one such sink (see
//! [`NodeSink`](crate::sink::NodeSink)); it exists for the test corpus and
//! for embedders that want a plain tree.
//!
//! Scripting is disabled, so `noscript`/`noframes` parse as fallback content
//! and `script` elements are inert.
//!
//! `template` contents are kept in a separate fragment: the public
//! [`Element::template_contents`] slot, per WHATWG 13.2.6.4.11. They are
//! not flattened into `children`.

use crate::atom::{AttrName, NameAtom, TagName};
use crate::sink::TreeSink;
use crate::token::{DoctypeToken, Tag, Token};
use crate::tokenizer::{State, Tokenizer};
use bowser_encoding::{
    Encoding, decode, decode_with, output_encoding, prescan_encoding, sniff_bom,
};
use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A node in the parsed tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    Document(Vec<Node>),
    Element(Element),
    Text {
        text: String,
        loc: Option<SourceLoc>,
    },
    Comment {
        data: String,
        loc: Option<SourceLoc>,
    },
    ProcessingInstruction {
        target: String,
        data: String,
        loc: Option<SourceLoc>,
    },
    Doctype(Doctype),
}

impl Drop for Node {
    /// Iterative destruction. Adversarial input can nest elements 100k+
    /// deep, and the default recursive drop would overflow the stack when
    /// such a tree is freed. Each node's child vectors are taken onto an
    /// explicit stack first, so every individual `Node` drops with empty
    /// children: O(1) per node, no recursion.
    fn drop(&mut self) {
        let mut stack = Vec::new();
        Self::take_children(self, &mut stack);
        while let Some(mut node) = stack.pop() {
            Self::take_children(&mut node, &mut stack);
            // `node` drops here with no children left.
        }
        // `self` drops here with no children left.
    }
}

impl Node {
    /// Move a node's child vectors (including template contents) onto
    /// `stack`, leaving the node childless.
    fn take_children(node: &mut Node, stack: &mut Vec<Node>) {
        match node {
            Node::Document(kids) => stack.extend(std::mem::take(kids)),
            Node::Element(e) => {
                stack.extend(std::mem::take(&mut e.children));
                if let Some(tc) = e.template_contents.as_mut() {
                    stack.extend(std::mem::take(tc));
                }
            }
            Node::Text { .. }
            | Node::Comment { .. }
            | Node::ProcessingInstruction { .. }
            | Node::Doctype(_) => {}
        }
    }
}

/// An element node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Element {
    /// Interned tag name: lowercased for HTML elements (the tokenizer
    /// lowercases), case-adjusted for SVG/MathML (e.g. `foreignObject`).
    /// Well-known names are pure atoms (no heap); anything else spills
    /// onto [`TagName`]'s heap sidecar and round-trips exactly.
    pub tag: TagName,
    /// The element's namespace.
    pub ns: Namespace,
    /// Interned attribute names with their (uninterned) values.
    pub attrs: Vec<(AttrName, String)>,
    pub children: Vec<Node>,
    /// For `template` elements, the template contents fragment
    /// (WHATWG 13.2.6.4.11). `None` for every other element. Template
    /// contents are *not* flattened into `children`.
    pub template_contents: Option<Vec<Node>>,
    /// Declarative shadow root attached by the parser from
    /// `<template shadowrootmode>` (WHATWG HTML 13.2.6.4.7). The template
    /// element itself is not in the tree; its contents became the shadow
    /// root's children. `None` for every other element, and for templates
    /// without a valid `shadowrootmode`.
    pub shadow_root: Option<ShadowRoot>,
    /// Position of the `<` that opened this element, when the element came
    /// from a real token; `None` for implied or otherwise synthesized
    /// elements (parser-added `html`/`head`/`body`, elements created by the
    /// adoption agency algorithm, ...).
    pub loc: Option<SourceLoc>,
}

/// Mode of a declarative shadow root (WHATWG HTML 4.12.3: the states of
/// the `shadowrootmode` enumerated content attribute).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowRootMode {
    Open,
    Closed,
}

/// A declarative shadow root attached to an element by the parser
/// (`<template shadowrootmode>`; WHATWG HTML 13.2.6.4.7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShadowRoot {
    /// From `shadowrootmode="open" | "closed"` (ASCII case-insensitive).
    pub mode: ShadowRootMode,
    /// From the `shadowrootdelegatesfocus` boolean attribute (presence).
    pub delegates_focus: bool,
    /// From the `shadowrootclonable` boolean attribute (presence).
    pub clonable: bool,
    /// The template's contents, parsed directly into the shadow root.
    pub children: Vec<Node>,
}

/// A 1-based source position.
///
/// Columns count Unicode scalar values (chars), not bytes, after the
/// tokenizer's input preprocessing (one leading BOM stripped, CR/CRLF
/// normalized to LF).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceLoc {
    pub line: u32,
    pub col: u32,
}

/// Namespace of an element. The HTML parser only ever produces the three
/// well-known namespaces; [`Namespace::Other`] arises from
/// `document.createElementNS` with an arbitrary namespace URI (DOM
/// "validate and extract" accepts any URI — only prefix rules are
/// validated), so the element is never silently mis-namespaced as
/// HTML/SVG/MathML.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Namespace {
    Html,
    MathMl,
    Svg,
    /// Any other namespace URI, carried as written.
    Other(Box<str>),
}

impl Namespace {
    /// The namespace URI: the well-known URI for the three built-in
    /// namespaces, the custom URI for [`Namespace::Other`].
    pub fn uri(&self) -> &str {
        match self {
            Namespace::Html => "http://www.w3.org/1999/xhtml",
            Namespace::MathMl => "http://www.w3.org/1998/Math/MathML",
            Namespace::Svg => "http://www.w3.org/2000/svg",
            Namespace::Other(u) => u,
        }
    }
}

/// A `<!DOCTYPE ...>` node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Doctype {
    /// Doctype name, lowercased by the tokenizer (`"html"` normally).
    pub name: String,
    /// The public identifier, if present.
    pub public_id: Option<String>,
    /// The system identifier, if present.
    pub system_id: Option<String>,
    /// True when the doctype token was bogus or truncated; forces quirks
    /// mode regardless of the identifiers.
    pub force_quirks: bool,
}

/// Quirks mode of a parsed document (WHATWG 13.2.6.4.1), computed from the
/// doctype (or its absence).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuirksMode {
    NoQuirks,
    LimitedQuirks,
    Quirks,
}

/// Parse `html` into a [`Node::Document`] tree.
///
/// The input bytes are decoded per WHATWG "decode" (BOM sniff, meta
/// prescan, UTF-8 fallback); see [`decode_input`].
///
/// Thin wrapper over [`parse_into`] with [`NodeSink`](crate::sink::NodeSink).
pub fn parse(html: &[u8]) -> Node {
    parse_with_quirks(html).0
}

/// Parse `html` like [`parse`], additionally returning the document's
/// [`QuirksMode`] as computed from its doctype.
///
/// Thin wrapper over [`parse_into`] with [`NodeSink`](crate::sink::NodeSink).
pub fn parse_with_quirks(html: &[u8]) -> (Node, QuirksMode) {
    let mut sink = crate::sink::NodeSink::new();
    let quirks = parse_into(html, &mut sink);
    (sink.into_node(), quirks)
}

/// Parse `html` into a [`Node::Document`] tree, driving `sink` directly
/// with the tree-construction output.
///
/// The input bytes are decoded per WHATWG "decode" (BOM sniff, meta
/// prescan, UTF-8 fallback); see [`decode_input`]. The input is borrowed,
/// not copied: when the decoded text needs no preprocessing, the
/// tokenizer works directly on it. A copy happens only when preprocessing
/// rewrites the input (CR/CRLF normalization or UTF-8 repair allocates).
///
/// This is the single-pass entry point: the tokenizer and insertion-mode
/// stages run as usual, but instead of materializing the intermediate owned
/// [`Node`] tree, string payloads are moved out of the parser straight into
/// the sink. Returns the document's [`QuirksMode`] as computed from its
/// doctype (WHATWG 13.2.6.4.1).
///
/// A pure streaming (SAX-style) parser is not possible here: the WHATWG
/// tree-construction stage reparents already-inserted nodes (adoption
/// agency algorithm, foster parenting), so the builder keeps its working
/// arena for the duration of the parse. What this removes is every
/// *additional* materialization: previously the arena was converted into
/// the owned [`Node`] tree (cloning every string), which downstream crates
/// then converted again into their own arenas (cloning every string a
/// second time).
pub fn parse_into(html: &[u8], sink: &mut impl TreeSink) -> QuirksMode {
    parse_into_with_errors(html, sink).0
}

/// [`parse_into`] plus error reporting. Returns the document's
/// [`QuirksMode`] and the [`ParseError`]s recorded during tokenization and
/// tree construction, in source order.
fn parse_into_with_errors(html: &[u8], sink: &mut impl TreeSink) -> (QuirksMode, Vec<ParseError>) {
    parse_into_with_errors_and_scripting(
        html,
        sink,
        // Process-wide default; snapshotted once per parse so concurrent
        // parses never cross-contaminate.
        !SCRIPTING_DISABLED.load(Ordering::Relaxed),
    )
}

/// The document encoding for `html` per WHATWG "decode": BOM sniffing
/// first (a BOM overrides everything), then the meta prescan over the
/// first 1024 bytes, then UTF-8 as the fallback. Mirrors
/// [`decode_input`]'s detection so [`Document::encoding`](bowser_dom::Document::encoding)
/// (the URL Standard's encoding override for query percent-encoding)
/// agrees with the decoder that produced the document's text.
pub fn detect_encoding(html: &[u8]) -> Encoding {
    // WHATWG "get an output encoding": replacement and UTF-16BE/LE become
    // UTF-8 (the same mapping `decode_input` applies — a meta-declared
    // UTF-16 is bogus because finding ASCII "charset" text proves the
    // bytes are ASCII-compatible, and `decode` strips a UTF-16 BOM then
    // decodes the rest as UTF-8).
    if let Some(enc) = sniff_bom(html) {
        return output_encoding(enc);
    }
    let head = &html[..html.len().min(1024)];
    prescan_encoding(head)
        .map(output_encoding)
        .unwrap_or(Encoding::Utf8)
}

/// Decode parser input bytes per WHATWG "decode": BOM sniffing first (a
/// BOM overrides everything and is stripped), then the meta prescan over
/// the first 1024 bytes, then UTF-8 as the fallback. Undecodable bytes
/// become U+FFFD, mirroring the spec's error handling.
///
/// The common case (no BOM, no meta-declared encoding, valid UTF-8)
/// borrows the caller's buffer with no copy, like the old
/// `String::from_utf8_lossy` fast path.
fn decode_input(html: &[u8]) -> Cow<'_, str> {
    if sniff_bom(html).is_none() {
        let head = &html[..html.len().min(1024)];
        let enc = prescan_encoding(head).map(|e| match e {
            // A meta-declared UTF-16 is remapped to UTF-8: finding ASCII
            // "charset" text proves the bytes are ASCII-compatible, so the
            // label is bogus (WHATWG "prescan a byte stream").
            Encoding::Utf16Be | Encoding::Utf16Le => Encoding::Utf8,
            e => e,
        });
        if enc.is_none_or(|e| e == Encoding::Utf8) {
            if let Ok(text) = std::str::from_utf8(html) {
                return Cow::Borrowed(text);
            }
        } else if let Some(e) = enc
            && let Ok((text, _)) = decode_with(html, e)
        {
            return Cow::Owned(text);
        }
    }
    // BOM present, or invalid UTF-8 without a usable declared encoding:
    // `decode` strips the BOM when present and substitutes U+FFFD.
    Cow::Owned(
        decode(html, Encoding::Utf8)
            .map(|r| r.text)
            .unwrap_or_else(|_| String::from_utf8_lossy(html).into_owned()),
    )
}

/// [`parse_into_with_errors`] with an explicit parser-local scripting flag.
fn parse_into_with_errors_and_scripting(
    html: &[u8],
    sink: &mut impl TreeSink,
    scripting: bool,
) -> (QuirksMode, Vec<ParseError>) {
    parse_into_with_errors_and_scripting_and_executor(html, sink, scripting, None)
}

/// [`parse_into_with_errors_and_scripting`] with a script engine attached.
///
/// When `scripting` is true, each eligible parser-inserted classic
/// `<script>` element (classic `type`, not in a `template`) is handed to
/// `executor` — element, attributes, and text — as soon as its end tag is
/// processed, before the next token is tokenized (see
/// [`ScriptExecutor`]).
fn parse_into_with_errors_and_scripting_and_executor(
    html: &[u8],
    sink: &mut impl TreeSink,
    scripting: bool,
    executor: Option<&mut dyn ScriptExecutor>,
) -> (QuirksMode, Vec<ParseError>) {
    let text = decode_input(html);
    let mut tb = TreeBuilder::new(&text, scripting, executor);
    tb.run();
    let errors = std::mem::take(&mut tb.errors);
    let quirks = tb.quirks_mode();
    tb.fixup_selectedcontent();
    tb.drive_sink(sink);
    (quirks, errors)
}

/// Parse `html` like [`crate::parse`], additionally returning the [`ParseError`]s
/// recorded during tokenization and tree construction, in source order.
///
/// The tokenizer is instrumented: tokenizer errors carry WHATWG 13.2.2
/// codes and source locations (the DOCTYPE family collapses into
/// [`ParseErrorCode::InvalidDoctype`]). Tree-construction errors are
/// reported at the insertion-mode rule that flags them.
pub fn parse_with_errors(html: &[u8]) -> (Node, Vec<ParseError>) {
    let mut sink = crate::sink::NodeSink::new();
    let (_, errors) = parse_into_with_errors(html, &mut sink);
    (sink.into_node(), errors)
}

/// [`parse_with_errors`] with an explicit parser-local scripting flag.
///
/// When `scripting` is true, `<noscript>` parses its content as raw text
/// per WHATWG 13.2.6 instead of as an ordinary element. This is the
/// per-parse alternative to the process-wide [`set_scripting_enabled`]
/// default; concurrent parses with different flags never interfere.
pub fn parse_with_scripting(html: &[u8], scripting: bool) -> Node {
    let mut sink = crate::sink::NodeSink::new();
    parse_into_with_errors_and_scripting(html, &mut sink, scripting);
    sink.into_node()
}

/// [`parse_with_scripting`] with a script engine attached.
///
/// When `scripting` is true, each eligible parser-inserted classic
/// `<script>` element (classic `type` per WHATWG "prepare the script",
/// not inside a `template`) is handed to `executor` — element,
/// attributes, and text — as soon as its end tag is processed, before the
/// next token is tokenized (WHATWG HTML 13.2.6: "if there is a script that
/// will execute as soon as the parser resumes ... execute the script",
/// with the insertion point just before the next input character). An
/// unclosed `<script>` at EOF executes the same way. The executor's
/// [`document.write`](ScriptHost::document_write) calls insert into the
/// tokenizer's input stream at that point, and its DOM mutations are
/// visible to the still-running parse. With `scripting` false (or no
/// executor), scripts never run, exactly like [`parse_with_scripting`].
pub fn parse_with_scripting_and_executor(
    html: &[u8],
    scripting: bool,
    executor: &mut dyn ScriptExecutor,
) -> Node {
    let mut sink = crate::sink::NodeSink::new();
    parse_into_with_errors_and_scripting_and_executor(html, &mut sink, scripting, Some(executor));
    sink.into_node()
}

/// [`parse_with_scripting`] plus the document's [`QuirksMode`] as computed
/// from its doctype (WHATWG 13.2.6.4.1).
///
/// Thin wrapper over [`parse_into`] with [`NodeSink`](crate::sink::NodeSink).
pub fn parse_with_scripting_and_quirks(html: &[u8], scripting: bool) -> (Node, QuirksMode) {
    let mut sink = crate::sink::NodeSink::new();
    let (quirks, _) = parse_into_with_errors_and_scripting(html, &mut sink, scripting);
    (sink.into_node(), quirks)
}

/// [`parse_with_scripting_and_executor`] plus the document's [`QuirksMode`]
/// as computed from its doctype (WHATWG 13.2.6.4.1).
///
/// Thin wrapper over [`parse_into`] with [`NodeSink`](crate::sink::NodeSink).
pub fn parse_with_scripting_and_executor_and_quirks(
    html: &[u8],
    scripting: bool,
    executor: &mut dyn ScriptExecutor,
) -> (Node, QuirksMode) {
    let mut sink = crate::sink::NodeSink::new();
    let (quirks, _) = parse_into_with_errors_and_scripting_and_executor(
        html,
        &mut sink,
        scripting,
        Some(executor),
    );
    (sink.into_node(), quirks)
}

/// [`parse_with_scripting_and_executor`] driving a caller-chosen
/// [`TreeSink`] instead of the owned [`Node`] tree.
///
/// This is the entry point for embedders that build their own arena
/// straight from the parser (e.g. `bowser_dom`'s `DomSink`): the
/// executor hook behaves exactly as documented on
/// [`parse_with_scripting_and_executor`], but the finished tree is the
/// sink's, not a [`Node`].
pub fn parse_into_with_scripting_and_executor(
    html: &[u8],
    sink: &mut impl TreeSink,
    scripting: bool,
    executor: &mut dyn ScriptExecutor,
) -> QuirksMode {
    parse_into_with_errors_and_scripting_and_executor(html, sink, scripting, Some(executor)).0
}

/// A parse error recorded while parsing (WHATWG 13.2.2 "parse errors").
///
/// Tree-construction errors are reported at the insertion-mode rule that
/// flags them. Tokenizer errors are reported by the tokenizer itself, at
/// the offending input, with WHATWG 13.2.2 codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// Machine-readable error code.
    pub code: ParseErrorCode,
    /// Source location of the offending token, when known.
    pub loc: Option<SourceLoc>,
}

/// Parse error codes. Tokenizer codes use the WHATWG 13.2.2 names;
/// tree-construction codes are generic, since the spec assigns them none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ParseErrorCode {
    /// U+0000 in the input (WHATWG `unexpected-null-character`).
    UnexpectedNullCharacter,
    /// Truncated or malformed DOCTYPE (the WHATWG `missing-doctype-name` /
    /// `abrupt-doctype-public-identifier` / `abrupt-doctype-system-identifier`
    /// / `eof-in-doctype` family).
    InvalidDoctype,
    /// EOF after `<` (WHATWG `eof-before-tag-name`).
    EofBeforeTagName,
    /// `</>` (WHATWG `missing-end-tag-name`).
    MissingEndTagName,
    /// e.g. `<1` (WHATWG `invalid-first-character-of-tag-name`).
    InvalidFirstCharacterOfTagName,
    /// EOF inside a tag (WHATWG `eof-in-tag`).
    EofInTag,
    /// `<div =>` (WHATWG `unexpected-equals-sign-before-attribute-name`).
    UnexpectedEqualsSignBeforeAttributeName,
    /// `"`, `'` or `<` in an attribute name (WHATWG
    /// `unexpected-character-in-attribute-name`).
    UnexpectedCharacterInAttributeName,
    /// e.g. `<div a="1"b="2">` (WHATWG
    /// `missing-whitespace-between-attributes`).
    MissingWhitespaceBetweenAttributes,
    /// `<div a=>` (WHATWG `missing-attribute-value`).
    MissingAttributeValue,
    /// `"`, `'`, `<`, `=` or backtick in an unquoted attribute value
    /// (WHATWG `unexpected-character-in-unquoted-attribute-value`).
    UnexpectedCharacterInUnquotedAttributeValue,
    /// `<div /x>` (WHATWG `unexpected-solidus-in-tag`).
    UnexpectedSolidusInTag,
    /// `<!x` (WHATWG `incorrectly-opened-comment`).
    IncorrectlyOpenedComment,
    /// `<!-->` / `<!--->` (WHATWG `abrupt-closing-of-empty-comment`).
    AbruptClosingOfEmptyComment,
    /// `<![CDATA[` in HTML content (WHATWG `cdata-in-html-content`).
    CdataInHtmlContent,
    /// EOF inside a comment (WHATWG `eof-in-comment`).
    EofInComment,
    /// `&#0;` (WHATWG `null-character-reference`).
    NullCharacterReference,
    /// `&#x110000;` (WHATWG `character-reference-outside-unicode-range`).
    CharacterReferenceOutsideUnicodeRange,
    /// `&#xD800;` (WHATWG `surrogate-character-reference`).
    SurrogateCharacterReference,
    /// `&#xFDD0;` (WHATWG `noncharacter-character-reference`).
    NoncharacterCharacterReference,
    /// `&#x80;` (WHATWG `control-character-reference`; the value is still
    /// remapped, e.g. `&#x80;` to U+20AC).
    ControlCharacterReference,
    /// `&#65x` or `&ampx` (WHATWG
    /// `missing-semicolon-after-character-reference`).
    MissingSemicolonAfterCharacterReference,
    /// `&#x;` (WHATWG `absence-of-digits-in-numeric-character-reference`).
    AbsenceOfDigitsInNumericCharacterReference,
    /// `&` followed by ASCII alphanumerics that match no named character
    /// reference (WHATWG `unknown-named-character-reference`).
    UnknownNamedCharacterReference,
    /// A token the insertion mode handled via error recovery (dropped,
    /// or processed after popping open elements).
    UnexpectedToken,
    /// Content foster-parented out of a table context (WHATWG parse error;
    /// html5lib: `foster-parenting-character-in-table`).
    FosterParented,
    /// An end tag with no matching open element; ignored.
    StrayEndTag,
    /// A foreign-content end tag whose name does not match the current
    /// node (WHATWG 13.2.6.5 "any other end tag", step 2).
    MismatchedEndTag,
    /// EOF with elements other than the implied-end-tag set (plus
    /// `body`/`html`) still open (WHATWG 13.2.7 "the end").
    UnclosedElements,
    /// The adoption agency algorithm lost track of a node it had just
    /// removed from the stack of open elements. Unreachable in a correct
    /// run (the removed node is always tracked); the parser records this
    /// and bails out of the algorithm instead of panicking.
    AdoptionAgencyLostNode,
    /// An internal tree-builder invariant failed (empty stack, node missing
    /// from the stack of open elements where the algorithm requires it,
    /// etc.). Unreachable in a correct run; the parser records this and
    /// fails closed (best-effort tree, current operation abandoned) instead
    /// of panicking on adversarial input.
    TreeInvariant,
}

/// Parse `html` as an HTML fragment with the given context element
/// (WHATWG 13.4 "parsing HTML fragments"), returning the fragment's nodes
/// as a [`Node::Document`].
///
/// The context element's tag name (ASCII-lowercased for HTML contexts)
/// selects the tokenizer's initial state (RCDATA for `title`/`textarea`,
/// RAWTEXT for `style`/`xmp`/`iframe`/`noembed`/`noframes`, script data for
/// `script`, plaintext for `plaintext`, RAWTEXT for `noscript` when
/// scripting is enabled) and the initial insertion mode via "reset the
/// insertion mode appropriately". Foreign-namespace contexts keep their
/// tag case and always start the tokenizer in the data state. The
/// context's namespace drives foreign-content dispatch through the
/// "adjusted current node", so e.g. an `svg` context parses its content
/// as SVG.
///
/// The synthetic `html` root the algorithm requires is never exposed: the
/// returned document holds exactly the fragment's children.
///
/// Form element pointer (WHATWG 13.4, "set the form element pointer" to
/// the context element's nearest `form` ancestor-or-self): a bare
/// [`Element`] carries no ancestors, so this defaults to the context
/// element itself when the context is an HTML `form` and to null
/// otherwise. Pass the context's ancestor chain to
/// [`parse_fragment_with_ancestors`] for the full lookup.
pub fn parse_fragment(html: &[u8], context_element: &Element) -> Node {
    parse_fragment_with_errors(html, context_element).0
}

/// [`parse_fragment`] with the context element's ancestor chain,
/// nearest first and *not* including the context element itself.
///
/// This completes the WHATWG 13.4 form-element-pointer initialization:
/// when the context element is not itself an HTML `form`, an HTML
/// `form` anywhere in `ancestors` puts the form element pointer in
/// scope, so a `<form>` start tag in the fragment is ignored exactly as
/// if the pointer named that ancestor. The pointer still names the
/// synthetic context copy (which is never on the open-element stack),
/// which is behaviorally identical to naming the real ancestor: the
/// ancestor sits above the fragment root and can never be in scope for
/// an end-tag check.
pub fn parse_fragment_with_ancestors(
    html: &[u8],
    context_element: &Element,
    ancestors: &[&Element],
) -> Node {
    parse_fragment_with_errors_and_scripting(
        html,
        context_element,
        // Process-wide default; snapshotted once per parse so concurrent
        // parses never cross-contaminate.
        !SCRIPTING_DISABLED.load(Ordering::Relaxed),
        ancestors,
    )
    .0
}

/// [`parse_fragment`] plus error reporting. Returns the fragment and the
/// recorded [`ParseError`]s, in source order.
pub fn parse_fragment_with_errors(
    html: &[u8],
    context_element: &Element,
) -> (Node, Vec<ParseError>) {
    parse_fragment_with_errors_and_scripting(
        html,
        context_element,
        // Process-wide default; snapshotted once per parse so concurrent
        // parses never cross-contaminate.
        !SCRIPTING_DISABLED.load(Ordering::Relaxed),
        &[],
    )
}

/// [`parse_fragment_with_errors`] with an explicit parser-local scripting
/// flag. See [`parse_with_scripting`].
pub fn parse_fragment_with_scripting(
    html: &[u8],
    context_element: &Element,
    scripting: bool,
) -> Node {
    parse_fragment_with_errors_and_scripting(html, context_element, scripting, &[]).0
}

/// Parse `html` as an HTML fragment like [`parse_fragment`], driving
/// `sink` directly with the tree-construction output (no intermediate
/// owned [`Node`] tree). See [`parse_into`] for the single-pass contract.
///
/// The input bytes are decoded per WHATWG "decode" (BOM sniff, meta
/// prescan, UTF-8 fallback); see [`decode_input`].
pub fn parse_fragment_into(html: &[u8], context_element: &Element, sink: &mut impl TreeSink) {
    parse_fragment_into_with_errors_and_scripting(
        html,
        context_element,
        // Process-wide default; snapshotted once per parse so concurrent
        // parses never cross-contaminate.
        !SCRIPTING_DISABLED.load(Ordering::Relaxed),
        &[],
        sink,
    );
}

/// [`parse_fragment_into`] with the context element's ancestor chain;
/// see [`parse_fragment_with_ancestors`].
pub fn parse_fragment_into_with_ancestors(
    html: &[u8],
    context_element: &Element,
    ancestors: &[&Element],
    sink: &mut impl TreeSink,
) {
    parse_fragment_into_with_errors_and_scripting(
        html,
        context_element,
        // Process-wide default; snapshotted once per parse so concurrent
        // parses never cross-contaminate.
        !SCRIPTING_DISABLED.load(Ordering::Relaxed),
        ancestors,
        sink,
    );
}

/// [`parse_fragment_with_errors`] with an explicit parser-local scripting
/// flag, plus error reporting.
fn parse_fragment_with_errors_and_scripting(
    html: &[u8],
    context_element: &Element,
    scripting: bool,
    ancestors: &[&Element],
) -> (Node, Vec<ParseError>) {
    let mut sink = crate::sink::NodeSink::new();
    let errors = parse_fragment_into_with_errors_and_scripting(
        html,
        context_element,
        scripting,
        ancestors,
        &mut sink,
    );
    (sink.into_node(), errors)
}

/// [`parse_fragment_into`] with an explicit parser-local scripting flag,
/// plus error reporting.
fn parse_fragment_into_with_errors_and_scripting(
    html: &[u8],
    context_element: &Element,
    scripting: bool,
    ancestors: &[&Element],
    sink: &mut impl TreeSink,
) -> Vec<ParseError> {
    let text = decode_input(html);
    let mut tb = TreeBuilder::new(&text, scripting, None);
    tb.init_fragment(context_element, ancestors);
    tb.run();
    let errors = std::mem::take(&mut tb.errors);
    tb.fixup_selectedcontent();
    tb.drive_sink(sink);
    errors
}

// ---------------------------------------------------------------------------
// Post-parse fixup (runs on the builder arena, before the sink walk)
// ---------------------------------------------------------------------------

impl<'a, 'x> TreeBuilder<'a, 'x> {
    /// Post-parse fixup for `<selectedcontent>` (WHATWG 4.10.17 "update a
    /// select's selectedcontent"): for each `select` that is the nearest
    /// `select` ancestor of an enabled `selectedcontent` (and has no
    /// `multiple` attribute), the primary `selectedcontent` gets its
    /// children replaced with a clone of the selected `option`'s children
    /// (or is cleared when the select has no options), and the select's
    /// non-primary `selectedcontent` elements are cleared.
    ///
    /// Runs on the internal arena before the sink walk, so every sink (the
    /// owned [`Node`] tree and direct-to-DOM alike) sees the identical tree.
    /// This batches what the spec runs incrementally from each
    /// `selectedcontent`'s post-connection steps; the per-select
    /// update-then-clear order reproduces the incremental end state (a
    /// nested select's `selectedcontent` is always disabled, so an outer
    /// select's clearing pass can never clobber an inner select's filled
    /// primary). Iterative to avoid stack overflow on deeply nested input.
    fn fixup_selectedcontent(&mut self) {
        // Collect the select elements first: the update step mutates the
        // arena, so selection and mutation stay in separate passes.
        let mut selects = Vec::new();
        let mut stack: Vec<El> = self
            .document
            .iter()
            .filter_map(|c| match c {
                Child::Elem(el) => Some(*el),
                _ => None,
            })
            .collect();
        while let Some(el) = stack.pop() {
            let node = &self.arena[el];
            if node.ns == Namespace::Html && node.tag.atom() == NameAtom::SELECT {
                selects.push(el);
            }
            // Descend into children and template contents alike, mirroring
            // the old owned-tree walk.
            for c in node.children.iter().rev() {
                if let Child::Elem(e) = c {
                    stack.push(*e);
                }
            }
            if let Some(t) = node.template_contents.as_ref() {
                for c in t.iter().rev() {
                    if let Child::Elem(e) = c {
                        stack.push(*e);
                    }
                }
            }
        }
        for select in selects {
            if self.select_needs_selectedcontent_fixup(select) {
                self.update_select_selectedcontent(select);
                self.clear_non_primary_selectedcontent(select);
            }
        }
    }

    /// Whether `select` gets the update/clear treatment: it has no
    /// `multiple` attribute, and it is the nearest `select` ancestor of at
    /// least one enabled `selectedcontent` descendant. Mirrors the early
    /// return in the spec's `selectedcontent` post-connection steps.
    fn select_needs_selectedcontent_fixup(&self, select: El) -> bool {
        if self.arena[select]
            .attrs
            .iter()
            .any(|(name, _)| name.atom() == NameAtom::MULTIPLE)
        {
            return false;
        }
        self.selectedcontent_descendants(select)
            .into_iter()
            .any(|sc| {
                let (disabled, nearest) = self.selectedcontent_state(sc);
                !disabled && nearest == Some(select)
            })
    }

    /// All `selectedcontent` (HTML namespace) descendants of `select` in
    /// tree order. Iterative.
    fn selectedcontent_descendants(&self, select: El) -> Vec<El> {
        let mut out = Vec::new();
        let mut stack: Vec<El> = vec![select];
        while let Some(el) = stack.pop() {
            let mut kids: Vec<El> = self.arena[el]
                .children
                .iter()
                .filter_map(|c| match c {
                    Child::Elem(e) => Some(*e),
                    _ => None,
                })
                .collect();
            kids.reverse();
            for child in kids {
                if self.arena[child].ns == Namespace::Html
                    && self.arena[child].tag.atom() == NameAtom::SELECTEDCONTENT
                {
                    out.push(child);
                }
                stack.push(child);
            }
        }
        out
    }

    /// A `selectedcontent` element's `disabled` flag and nearest `select`
    /// ancestor (WHATWG 4.10.17 post-connection steps 2-3): disabled when an
    /// `option` or `selectedcontent` element, or a second `select` element,
    /// appears among its ancestors.
    fn selectedcontent_state(&self, sc: El) -> (bool, Option<El>) {
        let mut nearest = None;
        let mut parent = self.arena[sc].parent;
        while let Some(p) = parent {
            let el = match p {
                Parent::Elem(el) => el,
                Parent::Doc => break,
            };
            let n = &self.arena[el];
            if n.ns == Namespace::Html {
                if n.tag.atom() == NameAtom::SELECT {
                    if nearest.is_some() {
                        return (true, nearest);
                    }
                    nearest = Some(el);
                } else if n.tag.atom() == NameAtom::OPTION
                    || n.tag.atom() == NameAtom::SELECTEDCONTENT
                {
                    return (true, nearest);
                }
            }
            parent = n.parent;
        }
        (false, nearest)
    }

    /// "Get a select's enabled selectedcontent" (WHATWG 4.10.17): null when
    /// the select has a `multiple` attribute, when it has no
    /// `selectedcontent` descendant, or when the first such descendant is
    /// disabled.
    fn enabled_selectedcontent(&self, select: El) -> Option<El> {
        if self.arena[select]
            .attrs
            .iter()
            .any(|(name, _)| name.atom() == NameAtom::MULTIPLE)
        {
            return None;
        }
        let first = self
            .selectedcontent_descendants(select)
            .into_iter()
            .next()?;
        if self.selectedcontent_state(first).0 {
            return None;
        }
        Some(first)
    }

    /// The select's "list of options" (WHATWG 4.10.7): `option` element
    /// children of the select plus `option` children of its `optgroup`
    /// children, in tree order. Descendants of nested `select`, `hr`,
    /// `option`, `datalist`, and nested-`optgroup` elements are skipped.
    /// Iterative.
    fn list_of_options(&self, select: El) -> Vec<El> {
        let mut options = Vec::new();
        let mut stack: Vec<El> = vec![select];
        while let Some(el) = stack.pop() {
            let n = &self.arena[el];
            let is_html = n.ns == Namespace::Html;
            if el != select {
                if is_html && n.tag.atom() == NameAtom::OPTION {
                    options.push(el);
                }
                // Skip this subtree when the spec's walk would jump past
                // it ("next descendant ... excluding node's descendants").
                if is_html
                    && (n.tag.atom() == NameAtom::SELECT
                        || n.tag.atom() == NameAtom::HR
                        || n.tag.atom() == NameAtom::OPTION
                        || n.tag.atom() == NameAtom::DATALIST
                        || (n.tag.atom() == NameAtom::OPTGROUP
                            && self.in_nested_optgroup(el, select)))
                {
                    continue;
                }
            }
            let mut kids: Vec<El> = n
                .children
                .iter()
                .filter_map(|c| match c {
                    Child::Elem(e) => Some(*e),
                    _ => None,
                })
                .collect();
            kids.reverse();
            stack.extend(kids);
        }
        options
    }

    /// Whether `el` (an `optgroup` element) has an `optgroup` ancestor
    /// strictly between itself and `select`.
    fn in_nested_optgroup(&self, el: El, select: El) -> bool {
        let mut parent = self.arena[el].parent;
        while let Some(p) = parent {
            match p {
                Parent::Elem(e) if e == select => return false,
                Parent::Elem(e) => {
                    let n = &self.arena[e];
                    if n.ns == Namespace::Html && n.tag.atom() == NameAtom::OPTGROUP {
                        return true;
                    }
                    parent = n.parent;
                }
                Parent::Doc => return false,
            }
        }
        false
    }

    /// "Update a select's selectedcontent" (WHATWG 4.10.17): fill the
    /// primary `selectedcontent` with a clone of the first selected
    /// option's children (else the first option's children); clear the
    /// primary when the select has no options.
    ///
    /// The primary is collected before any fill: filling mutates the arena,
    /// and the filled content can itself contain `selectedcontent` (when
    /// the source option does), which a live walk would chase forever.
    /// Each fill gets an independent deep copy (like the old per-target
    /// clone), which also breaks any ancestor cycle between the source
    /// option and a target nested inside it.
    fn update_select_selectedcontent(&mut self, select: El) {
        let Some(primary) = self.enabled_selectedcontent(select) else {
            return;
        };
        let options = self.list_of_options(select);
        let chosen = options
            .iter()
            .find(|el| {
                self.arena[**el]
                    .attrs
                    .iter()
                    .any(|(name, _)| name.atom() == NameAtom::SELECTED)
            })
            .or(options.first());
        match chosen {
            Some(option) => {
                let children = self.arena[*option].children.clone();
                let cloned = self.clone_subtree(&children, Parent::Elem(primary));
                self.arena[primary].children = cloned;
            }
            None => {
                // "Clear a selectedcontent": replace all with null.
                self.arena[primary].children.clear();
            }
        }
        self.bump_mutation();
    }

    /// "Clear a select's non-primary selectedcontent elements" (WHATWG
    /// 4.10.17): every `selectedcontent` descendant after the first in tree
    /// order is emptied. Runs after the fill, on the live tree, like the
    /// spec's post-connection steps.
    fn clear_non_primary_selectedcontent(&mut self, select: El) {
        let descendants = self.selectedcontent_descendants(select);
        let mut cleared = false;
        for sc in descendants.into_iter().skip(1) {
            if !self.arena[sc].children.is_empty() {
                self.arena[sc].children.clear();
                cleared = true;
            }
        }
        if cleared {
            self.bump_mutation();
        }
    }

    /// Deep-copy an arena child list into fresh arena nodes parented under
    /// `parent`, returning the copied list.
    ///
    /// Element subtrees are duplicated node by node (tag, namespace,
    /// attributes, locations, and template-contents structure are
    /// preserved); leaf payloads are cloned. Iterative over an explicit
    /// stack: the source subtree can be adversarially deep.
    fn clone_subtree(&mut self, src: &[Child], parent: Parent) -> Vec<Child> {
        let mut out = Vec::new();
        // (source element, fresh copy) pairs whose child lists still need
        // copying.
        let mut pending: Vec<(El, El)> = Vec::new();
        copy_child_list(&mut self.arena, src, parent, &mut out, &mut pending);
        self.drain_clone_pending(&mut pending);
        out
    }

    /// Finish a deep element copy: drain `(source, copy)` pairs,
    /// duplicating each source element's `children` and
    /// `template_contents` into its copy. Nested template contents live
    /// in `template_contents`, not `children` (the tree builder
    /// invariant), so both lists are copied. Iterative over an explicit
    /// stack: the source subtree can be adversarially deep. Shared by
    /// [`TreeBuilder::clone_subtree`] and the [`ScriptHost`] deep-clone
    /// paths (`cloneNode(deep)`, `template.content`).
    fn drain_clone_pending(&mut self, pending: &mut Vec<(El, El)>) {
        while let Some((src_el, new_el)) = pending.pop() {
            // Clone the source lists before mutating the arena: pushing
            // fresh nodes may reallocate it.
            let (kids, tmpl) = {
                let s = &self.arena[src_el];
                (s.children.clone(), s.template_contents.clone())
            };
            let mut new_kids = Vec::new();
            copy_child_list(
                &mut self.arena,
                &kids,
                Parent::Elem(new_el),
                &mut new_kids,
                pending,
            );
            self.arena[new_el].children = new_kids;
            if let Some(t) = tmpl {
                let mut new_tmpl = Vec::new();
                copy_child_list(
                    &mut self.arena,
                    &t,
                    Parent::Elem(new_el),
                    &mut new_tmpl,
                    pending,
                );
                self.arena[new_el].template_contents = Some(new_tmpl);
            }
        }
    }

    /// Drive `sink` with the finished tree, moving string payloads out of
    /// the internal arena (no clones).
    ///
    /// Iterative pre-order walk over an explicit job stack: adversarial
    /// input can nest elements 100k+ deep, which would overflow the call
    /// stack if this recursed. Template contents are delivered bracketed by
    /// `start_template_contents` / `end_template_contents`, after the
    /// element's own children (matching the DOM convention where the
    /// template fragment is the template element's last child). A
    /// `<template shadowrootmode>` element is replaced by a
    /// `start_shadow_root` / `end_shadow_root` bracket on its parent,
    /// delivered at the template's position among the host's children.
    fn drive_sink(self, sink: &mut impl TreeSink) {
        enum Job {
            Visit(Child),
            EndElement,
            StartTemplate,
            EndTemplate,
            StartShadowRoot(DsrParams),
            EndShadowRoot,
        }
        let TreeBuilder {
            mut arena,
            document,
            ..
        } = self;
        let mut jobs: Vec<Job> = document.into_iter().rev().map(Job::Visit).collect();
        while let Some(job) = jobs.pop() {
            match job {
                Job::Visit(Child::Elem(el)) => {
                    // Move the element's payloads out of the arena; the
                    // node itself is never visited again.
                    let node = &mut arena[el];
                    let tag = std::mem::take(&mut node.tag);
                    let ns = std::mem::replace(&mut node.ns, Namespace::Html);
                    let attrs = std::mem::take(&mut node.attrs);
                    let loc = node.loc;
                    let children = std::mem::take(&mut node.children);
                    let template_contents = node.template_contents.take();
                    let dsr = node.dsr.take();
                    if let Some(dsr) = dsr {
                        // Declarative shadow root (WHATWG HTML 13.2.6.4.7):
                        // the `<template shadowrootmode>` element is
                        // replaced by its shadow root. No element brackets
                        // are emitted; the template's contents become the
                        // shadow root's children, bracketed at the
                        // template's position among the host's children.
                        //
                        // `children` is always empty for a template (the
                        // appropriate-place logic routes template children
                        // into `template_contents`); the assert guards the
                        // invariant in debug builds.
                        debug_assert!(children.is_empty());
                        jobs.push(Job::EndShadowRoot);
                        if let Some(tmpl) = template_contents {
                            jobs.extend(tmpl.into_iter().rev().map(Job::Visit));
                        }
                        jobs.push(Job::StartShadowRoot(dsr));
                        continue;
                    }
                    sink.start_element_atomized(tag, ns, attrs, loc);
                    // Push in reverse execution order: children first,
                    // then the bracketed template contents, then the
                    // element end.
                    jobs.push(Job::EndElement);
                    if let Some(tmpl) = template_contents {
                        jobs.push(Job::EndTemplate);
                        jobs.extend(tmpl.into_iter().rev().map(Job::Visit));
                        jobs.push(Job::StartTemplate);
                    }
                    jobs.extend(children.into_iter().rev().map(Job::Visit));
                }
                Job::Visit(Child::Text(text, loc)) => sink.text(text, loc),
                Job::Visit(Child::Comment(data, loc)) => sink.comment(data, loc),
                Job::Visit(Child::ProcessingInstruction { target, data, loc }) => {
                    sink.processing_instruction(target, data, loc)
                }
                Job::Visit(Child::Doctype(dt)) => sink.doctype(Doctype {
                    name: dt.name.unwrap_or_default(),
                    public_id: dt.public_id,
                    system_id: dt.system_id,
                    force_quirks: dt.force_quirks,
                }),
                Job::EndElement => sink.end_element(),
                Job::StartTemplate => sink.start_template_contents(),
                Job::EndTemplate => sink.end_template_contents(),
                Job::StartShadowRoot(dsr) => {
                    sink.start_shadow_root(dsr.mode, dsr.delegates_focus, dsr.clonable)
                }
                Job::EndShadowRoot => sink.end_shadow_root(),
            }
        }
    }

    /// Record one tree mutation for [`mutation_count`](Self::mutation_count).
    /// Every insertion, removal, and attribute/text change funnels through
    /// here, so a geometry cache keyed on the count can never go stale.
    fn bump_mutation(&mut self) {
        self.mutation_count = self.mutation_count.wrapping_add(1);
    }

    /// Number of tree mutations applied to the working tree since the
    /// builder was created. Mid-parse geometry snapshots are cached
    /// against this count: any structural, text, or attribute change bumps
    /// it, so an unchanged count proves the snapshot is still valid.
    pub fn mutation_count(&self) -> u64 {
        self.mutation_count
    }

    /// Drive `sink` with the tree built so far, *borrowing* the arena: a
    /// borrowing port of [`drive_sink`](Self::drive_sink). String payloads
    /// are cloned (tag/attribute names are interned atoms, so only
    /// attribute values and text allocate); the builder keeps parsing
    /// afterwards, undisturbed.
    ///
    /// Buffered state is flushed first —
    /// [`pending_text`](Self::pending_text),
    /// [`pending_table_chars`](Self::pending_table_chars), and
    /// [`foster_buf`](Self::foster_buf) — so no text run or
    /// foster-parented node goes missing from the snapshot (missing text
    /// would measure wrong widths). The table-chars flush can refill
    /// `pending_text` through the foster path, so text flushes after it
    /// and the foster buffer splices last.
    ///
    /// Returns the arena index of every emitted element, in
    /// `start_element` emission order, so the caller can zip emission
    /// order back to nodes. Template elements replaced by a shadow-root
    /// bracket are not elements and contribute no entry.
    pub fn snapshot_into(&mut self, sink: &mut impl TreeSink) -> Vec<El> {
        self.flush_pending_table_chars();
        self.flush_text();
        self.flush_foster_buf();

        enum Job {
            Visit(Child),
            EndElement,
            StartTemplate,
            EndTemplate,
            StartShadowRoot(DsrParams),
            EndShadowRoot,
        }
        let mut order = Vec::new();
        let mut jobs: Vec<Job> = self
            .document
            .iter()
            .rev()
            .map(|c| Job::Visit(c.clone()))
            .collect();
        while let Some(job) = jobs.pop() {
            match job {
                Job::Visit(Child::Elem(el)) => {
                    let node = &self.arena[el];
                    if let Some(dsr) = node.dsr {
                        // Declarative shadow root: same bracket treatment
                        // as `drive_sink` (no element brackets, no order
                        // entry); the template's contents become the
                        // shadow root's children.
                        debug_assert!(node.children.is_empty());
                        jobs.push(Job::EndShadowRoot);
                        if let Some(tmpl) = node.template_contents.clone() {
                            jobs.extend(tmpl.into_iter().rev().map(Job::Visit));
                        }
                        jobs.push(Job::StartShadowRoot(dsr));
                        continue;
                    }
                    let tag = node.tag.clone();
                    let ns = node.ns.clone();
                    let attrs = node.attrs.clone();
                    let loc = node.loc;
                    let children = node.children.clone();
                    let template_contents = node.template_contents.clone();
                    order.push(el);
                    sink.start_element_atomized(tag, ns, attrs, loc);
                    // Push in reverse execution order: children first,
                    // then the bracketed template contents, then the
                    // element end.
                    jobs.push(Job::EndElement);
                    if let Some(tmpl) = template_contents {
                        jobs.push(Job::EndTemplate);
                        jobs.extend(tmpl.into_iter().rev().map(Job::Visit));
                        jobs.push(Job::StartTemplate);
                    }
                    jobs.extend(children.into_iter().rev().map(Job::Visit));
                }
                Job::Visit(Child::Text(text, loc)) => sink.text(text, loc),
                Job::Visit(Child::Comment(data, loc)) => sink.comment(data, loc),
                Job::Visit(Child::ProcessingInstruction { target, data, loc }) => {
                    sink.processing_instruction(target, data, loc)
                }
                Job::Visit(Child::Doctype(dt)) => sink.doctype(Doctype {
                    name: dt.name.clone().unwrap_or_default(),
                    public_id: dt.public_id.clone(),
                    system_id: dt.system_id.clone(),
                    force_quirks: dt.force_quirks,
                }),
                Job::EndElement => sink.end_element(),
                Job::StartTemplate => sink.start_template_contents(),
                Job::EndTemplate => sink.end_template_contents(),
                Job::StartShadowRoot(dsr) => {
                    sink.start_shadow_root(dsr.mode, dsr.delegates_focus, dsr.clonable)
                }
                Job::EndShadowRoot => sink.end_shadow_root(),
            }
        }
        order
    }
}

/// Copy one arena child list into `out`, duplicating element children into
/// fresh arena nodes (with empty child lists) and queueing
/// `(source, copy)` pairs on `pending` for [`TreeBuilder::clone_subtree`].
/// Leaf payloads are cloned. Free function so the arena borrow can be held
/// across the call.
fn copy_child_list(
    arena: &mut Vec<ArenaNode>,
    src: &[Child],
    parent: Parent,
    out: &mut Vec<Child>,
    pending: &mut Vec<(El, El)>,
) {
    out.reserve(src.len());
    for c in src {
        match c {
            Child::Elem(e) => {
                let s = &arena[*e];
                let new_el = arena.len();
                arena.push(ArenaNode {
                    tag: s.tag.clone(),
                    ns: s.ns.clone(),
                    attrs: s.attrs.clone(),
                    children: Vec::new(),
                    template_contents: s.template_contents.as_ref().map(|_| Vec::new()),
                    dsr: s.dsr,
                    parent: Some(parent),
                    loc: s.loc,
                    annotation_xml_html: s.annotation_xml_html,
                });
                pending.push((*e, new_el));
                out.push(Child::Elem(new_el));
            }
            Child::Text(text, loc) => out.push(Child::Text(text.clone(), *loc)),
            Child::Comment(data, loc) => out.push(Child::Comment(data.clone(), *loc)),
            Child::ProcessingInstruction { target, data, loc } => {
                out.push(Child::ProcessingInstruction {
                    target: target.clone(),
                    data: data.clone(),
                    loc: *loc,
                })
            }
            Child::Doctype(dt) => out.push(Child::Doctype(dt.clone())),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal representation
// ---------------------------------------------------------------------------

/// Where an arena element's parent lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Parent {
    Elem(El),
    Doc,
}

/// Arena element id. Index 0 of the stack is the topmost (`html`) element;
/// the last stack entry is the current node.
type El = usize;

/// A child slot: either an arena element or leaf data (with the source
/// location of the token that produced it, when known).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Child {
    Elem(El),
    Text(String, Option<SourceLoc>),
    Comment(String, Option<SourceLoc>),
    ProcessingInstruction {
        target: String,
        data: String,
        loc: Option<SourceLoc>,
    },
    Doctype(DoctypeToken),
}

#[derive(Debug)]
struct ArenaNode {
    tag: TagName,
    ns: Namespace,
    attrs: Vec<(AttrName, String)>,
    children: Vec<Child>,
    /// Separate content fragment for `template` elements.
    template_contents: Option<Vec<Child>>,
    /// Declarative shadow root params when this node is a
    /// `<template shadowrootmode>` whose shadow root attached
    /// successfully (WHATWG HTML 13.2.6.4.7). The template's
    /// `template_contents` then hold the shadow root's children, and the
    /// sink driver emits a shadow-root bracket instead of the element.
    dsr: Option<DsrParams>,
    parent: Option<Parent>,
    /// Source position of the `<` that opened this element, if it came from
    /// a real token.
    loc: Option<SourceLoc>,
    /// For MathML `annotation-xml`: whether its `encoding` attribute was
    /// `text/html` or `application/xhtml+xml` (HTML integration point).
    annotation_xml_html: bool,
}

/// An entry in the list of active formatting elements.
#[derive(Debug, Clone)]
enum AfeEntry {
    Marker,
    Element {
        element: El,
        /// Tag name and attributes of the token the element was created for
        /// (needed by the adoption agency algorithm and reconstruction).
        token_name: TagName,
        attrs: Vec<(AttrName, String)>,
    },
}

impl AfeEntry {
    fn is_marker(&self) -> bool {
        matches!(self, AfeEntry::Marker)
    }
    fn element(&self) -> Option<El> {
        match self {
            AfeEntry::Marker => None,
            AfeEntry::Element { element, .. } => Some(*element),
        }
    }
}

/// Insertion modes (WHATWG 13.2.6.4). There is no "in select" mode in the
/// current standard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Initial,
    BeforeHtml,
    BeforeHead,
    InHead,
    InHeadNoscript,
    AfterHead,
    InBody,
    Text,
    InTable,
    InTableText,
    InCaption,
    InColumnGroup,
    InTableBody,
    InRow,
    InCell,
    InTemplate,
    AfterBody,
    InFrameset,
    AfterFrameset,
    AfterAfterBody,
    AfterAfterFrameset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Quirks {
    None,
    Limited,
    Full,
}

/// Whether the HTML parser treats scripting as disabled.
///
/// Process-wide *default*, snapshotted once per parse into the builder's
/// parser-local flag: concurrent parses never cross-contaminate, and
/// flipping this mid-parse does not affect a parse already in flight. For
/// a per-parse choice without touching global state, use
/// [`parse_with_scripting`] / [`parse_fragment_with_scripting`].
///
/// This flag exists behind the `bowser_js` integration: it defaults to
/// `true` (disabled), preserving the historical parse where `<noscript>`
/// is an ordinary element whose children parse as markup. The JS
/// integration flips it via [`set_scripting_enabled`] before parsing
/// documents whose parser-inserted scripts will be executed.
///
/// Behavior change when scripting is enabled: `<noscript>` parses its
/// content as raw text (head: the "in head noscript" RAWTEXT handling;
/// body: raw-text element), per WHATWG 13.2.6, instead of as an ordinary
/// element.
pub static SCRIPTING_DISABLED: AtomicBool = AtomicBool::new(true);

/// Enable or disable scripting as the default for subsequently parsed
/// documents.
///
/// Called by the `bowser_js` integration (`WebBindings::install_with_clock`
/// enables it); the parser default stays disabled so existing parse trees
/// are unaffected. See [`SCRIPTING_DISABLED`] for the `<noscript>`
/// behavior change this causes. Prefer [`parse_with_scripting`] for a
/// per-parse choice.
///
/// Reserved for the future `bowser_js` integration: no workspace caller
/// yet, so no public re-export (see `lib.rs`).
#[expect(dead_code)]
pub fn set_scripting_enabled(enabled: bool) {
    SCRIPTING_DISABLED.store(!enabled, Ordering::Relaxed);
}

/// Read the current scripting default. Reserved for the `bowser_js`
/// integration's diagnostics; the parser itself snapshots the default
/// into a parser-local flag at parse start. No workspace caller yet, so
/// no public re-export (see `lib.rs`).
#[expect(dead_code)]
pub fn scripting_enabled() -> bool {
    !SCRIPTING_DISABLED.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Script execution hook
// ---------------------------------------------------------------------------

/// Opaque handle to an element of the document being parsed, handed to a
/// [`ScriptExecutor`] for DOM queries and mutation.
///
/// The handle stays valid for the whole parse: arena slots are never
/// reused or reordered, so an element fetched before a
/// [`document.write`](ScriptHost::document_write) still denotes the same
/// element afterwards.
///
/// Handles are tagged with their parse's generation: a handle retained
/// past its parse and used in a later one fails validation instead of
/// addressing a stale or wrong arena slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScriptElement {
    idx: usize,
    generation: u64,
}

impl ScriptElement {
    fn from_el(el: El, generation: u64) -> Self {
        ScriptElement {
            idx: el,
            generation,
        }
    }

    fn to_el(self) -> El {
        self.idx
    }

    /// The arena index this handle addresses. Public so later stages can
    /// zip a `TreeBuilder` snapshot's emission order back to nodes. Only
    /// meaningful for the parse generation the handle was issued in; the
    /// generation tag is still validated separately.
    pub fn arena_index(&self) -> usize {
        self.idx
    }
}

/// Process-wide parse generation counter. Every parse takes the next
/// value for [`TreeBuilder::generation`], which tags the
/// [`ScriptElement`] handles handed out during that parse; [`ScriptHost`]
/// validates the tag so a handle retained across parses cannot address a
/// stale or wrong arena slot.
static PARSE_GENERATION: AtomicU64 = AtomicU64::new(0);

/// The script-engine side of parser-driven script execution.
///
/// `bowser_html` tokenizes and builds the tree but never evaluates script
/// source. When scripting is enabled and an executor is installed (see
/// [`parse_with_scripting_and_executor`]), the tree builder calls
/// [`execute_script`](Self::execute_script) as soon as an eligible
/// parser-inserted classic `<script>` element's end tag has been
/// processed, before the next token is tokenized (WHATWG HTML 13.2.6: "if
/// there is a script that will execute as soon as the parser resumes ...
/// execute the script"). `src`/`async`/`defer` fetching and scheduling are
/// the engine's business; the parser reports the element, its attributes,
/// and its text.
///
/// This is the seam the `bowser_js` integration will implement; the
/// html5lib corpus stands one in with a minimal evaluator covering
/// exactly the script surface the scripted fixtures use.
pub trait ScriptExecutor {
    /// Execute a parser-inserted classic script.
    ///
    /// `el` identifies the script element and `attrs` carries its
    /// attributes (lowercased names), so the engine can implement
    /// `src`/`async`/`defer` fetching and scheduling; `script` is the
    /// element's text content. `host` exposes the two parser-visible
    /// effects a running script can have mid-parse: DOM queries and
    /// mutation, and [`document.write`](ScriptHost::document_write) into
    /// the input stream. Mutations are visible to the still-running parse,
    /// exactly as in a browser.
    fn execute_script(
        &mut self,
        host: &mut ScriptHost<'_, '_, '_>,
        el: ScriptElement,
        attrs: &[(String, String)],
        script: &str,
    );
}

/// Parser services exposed to a [`ScriptExecutor`] while a script runs.
///
/// These mirror the WHATWG integration points the tree-construction stage
/// offers script execution: the live document (for `document.*` DOM APIs)
/// and the input-stream insertion point (for `document.write`).
pub struct ScriptHost<'a, 'x, 'h> {
    tb: &'h mut TreeBuilder<'a, 'x>,
}

impl<'a, 'x, 'h> ScriptHost<'a, 'x, 'h> {
    /// `document.write(text)`: insert `text` into the tokenizer's input
    /// stream just before the next input character (WHATWG HTML 13.2.6:
    /// "let the insertion point be just before the next input character").
    /// The inserted text tokenizes as if it had always been there, so
    /// markup written by a script parses before the original following
    /// input; nested `document.write` calls nest naturally.
    pub fn document_write(&mut self, text: &str) {
        self.tb.tok.insert_input(text);
    }

    /// Number of tree mutations applied since the parse started. A script
    /// measuring geometry mid-parse snapshots the tree and caches the
    /// layout against this count: any DOM mutation — including
    /// `set_attribute`, which the classic forced-sync-layout pattern
    /// (`el.style.width = ...; el.offsetWidth`) relies on — bumps it, so
    /// the cache can never serve a stale layout.
    pub fn mutation_count(&self) -> u64 {
        self.tb.mutation_count()
    }

    /// The document's quirks mode as computed from its doctype
    /// (WHATWG 13.2.6.4.1). Live during script execution: the mode
    /// is set when the doctype token processes, before any body
    /// script runs, so a mid-parse measurement can read it even
    /// though no `Document` exists yet — the parse shim stamps it
    /// onto the documents it rebuilds from the snapshot.
    pub fn quirks_mode(&self) -> QuirksMode {
        self.tb.quirks_mode()
    }

    /// Snapshot the tree built so far into `sink`, borrowing the arena:
    /// string payloads are cloned and the parse continues afterwards,
    /// undisturbed. Buffered text runs, table text, and foster-parented
    /// nodes are flushed first so the snapshot (and therefore measured
    /// geometry) is complete. Returns the arena index of every emitted
    /// element in `start_element` emission order — pair with
    /// [`ScriptElement::arena_index`] to zip emission order back to nodes.
    pub fn snapshot_document(&mut self, sink: &mut impl TreeSink) -> Vec<usize> {
        self.tb.snapshot_into(sink)
    }

    /// `document.getElementById(id)`: the first element in document order
    /// whose `id` attribute equals `id`, or `None` when there is none.
    pub fn get_element_by_id(&self, id: &str) -> Option<ScriptElement> {
        let generation = self.tb.generation;
        self.tb
            .find_element_by_id(id)
            .map(|el| ScriptElement::from_el(el, generation))
    }

    /// `document.getElementsByTagName(tag)`: the HTML-namespace elements
    /// with tag name `tag` (ASCII case-insensitive, as the parser
    /// lowercases HTML tag names), in document order. `"*"` matches every
    /// HTML-namespace element. Template contents are not descended into,
    /// per DOM.
    pub fn get_elements_by_tag_name(&self, tag: &str) -> Vec<ScriptElement> {
        let generation = self.tb.generation;
        self.tb
            .elements_by_tag_name(tag)
            .into_iter()
            .map(|el| ScriptElement::from_el(el, generation))
            .collect()
    }

    /// `document.querySelector(sel)`: the first element in document order
    /// matching the selector, or `None`. Supports compounds of tag,
    /// `#id`, `.class`, `[attr]` (with `=`, `~=`, `|=`, `^=`, `$=`, `*=`
    /// operators), the structural pseudo-classes `:first-child`,
    /// `:last-child`, `:only-child`, `:nth-child()`, `:nth-last-child()`,
    /// `:first-of-type`, `:last-of-type`, `:only-of-type`,
    /// `:nth-of-type()`, `:nth-last-of-type()`, `:empty`, `:root`, and
    /// `:not()`, combined with the descendant (` `), child (`>`),
    /// adjacent-sibling (`+`), and general-sibling (`~`) combinators.
    /// Comma-separated groups are unioned: the earliest element in
    /// document order matching any group wins. Anything else (other
    /// pseudo-classes/elements, namespaces, `:has()`, ...) is
    /// unsupported and yields `None`.
    pub fn query_selector(&self, sel: &str) -> Option<ScriptElement> {
        self.query_selector_all(sel).into_iter().next()
    }

    /// `document.querySelectorAll(sel)`: all elements in document order
    /// matching the selector. Same selector support as
    /// [`query_selector`](Self::query_selector). Comma-separated groups
    /// are unioned: an element matching two groups appears once, and
    /// interleaved groups come out in document order, not group order
    /// — mirroring the production DOM's matched-set + preorder walk.
    /// Arena order is creation order, NOT document order: moves
    /// (`appendChild` / `insertBefore`) and foster parenting reorder
    /// the tree without touching arena slots, so order is taken from
    /// the preorder walk from the document root (the same walk
    /// `getElementById` / `getElementsByTagName` use). That walk only
    /// reaches connected nodes, so detached nodes — e.g. an element
    /// removed by `removeChild` / `replaceChild`, or created by
    /// `createElement` but never inserted — never match (a setup loop
    /// like `while (xtd = querySelector('x-td[rowspan]')) {
    /// ...replaceChild(td, xtd); }` would otherwise keep finding the
    /// just-detached node forever). Template contents are likewise
    /// never walked, per DOM.
    pub fn query_selector_all(&self, sel: &str) -> Vec<ScriptElement> {
        let generation = self.tb.generation;
        let complexes: Vec<_> = split_selector_groups(sel)
            .into_iter()
            .filter(|group| !group.is_empty())
            .filter_map(|group| parse_complex_selector(&group))
            .collect();
        if complexes.is_empty() {
            return Vec::new();
        }
        let order = self.document_order_elements();
        let mut matched = vec![false; self.tb.arena.len()];
        for complex in &complexes {
            for &idx in &order {
                if self.complex_matches(complex, idx) {
                    matched[idx] = true;
                }
            }
        }
        order
            .into_iter()
            .filter(|idx| matched[*idx])
            .map(|idx| ScriptElement::from_el(idx, generation))
            .collect()
    }

    /// Element indices in document order: the preorder walk from the
    /// document root, following `children` lists. This — not arena
    /// index order — is document order after moves and foster
    /// parenting, and it excludes detached nodes and template
    /// contents by construction.
    fn document_order_elements(&self) -> Vec<El> {
        let mut out = Vec::new();
        self.tb.each_element(|el| {
            out.push(el);
            true
        });
        out
    }

    /// Right-to-left match of a complex selector against arena element
    /// `idx`.
    fn complex_matches(&self, sel: &ComplexSelector, idx: usize) -> bool {
        let n = sel.compounds.len();
        if !self.compound_matches(&sel.compounds[n - 1], idx) {
            return false;
        }
        let mut current = idx;
        for i in (0..n - 1).rev() {
            match self.step_combinator(sel.combinators[i], &sel.compounds[i], current) {
                Some(next) => current = next,
                None => return false,
            }
        }
        true
    }

    /// Walk one combinator leftwards from `idx`, returning the element the
    /// compound must match, if any.
    fn step_combinator(
        &self,
        comb: Combinator,
        compound: &CompoundSelector,
        idx: usize,
    ) -> Option<usize> {
        match comb {
            Combinator::Child => {
                let p = self.parent_element_idx(idx)?;
                self.compound_matches(compound, p).then_some(p)
            }
            Combinator::Descendant => {
                let mut p = self.parent_element_idx(idx);
                while let Some(a) = p {
                    if self.compound_matches(compound, a) {
                        return Some(a);
                    }
                    p = self.parent_element_idx(a);
                }
                None
            }
            Combinator::NextSibling => {
                let s = self.prev_element_sibling(idx)?;
                self.compound_matches(compound, s).then_some(s)
            }
            Combinator::LaterSibling => {
                let mut s = self.prev_element_sibling(idx);
                while let Some(cand) = s {
                    if self.compound_matches(compound, cand) {
                        return Some(cand);
                    }
                    s = self.prev_element_sibling(cand);
                }
                None
            }
        }
    }

    /// Arena index of the parent element, if any.
    fn parent_element_idx(&self, idx: usize) -> Option<usize> {
        match self.tb.arena.get(idx)?.parent {
            Some(Parent::Elem(p)) => Some(p),
            _ => None,
        }
    }

    /// Arena index of the nearest preceding element sibling, if any.
    fn prev_element_sibling(&self, idx: usize) -> Option<usize> {
        let parent = self.parent_element_idx(idx)?;
        let siblings = &self.tb.arena.get(parent)?.children;
        let mut prev: Option<usize> = None;
        for child in siblings {
            match child {
                Child::Elem(e) if *e == idx => return prev,
                Child::Elem(e) => prev = Some(*e),
                _ => {}
            }
        }
        None
    }

    /// 1-based position of `idx` among its element siblings (itself
    /// included), and the total element-sibling count.
    fn sibling_position(&self, idx: usize) -> Option<(usize, usize)> {
        let parent = self.parent_element_idx(idx)?;
        let siblings = &self.tb.arena.get(parent)?.children;
        let mut pos = 0;
        let mut total = 0;
        for child in siblings {
            if let Child::Elem(e) = child {
                total += 1;
                if *e == idx {
                    pos = total;
                }
            }
        }
        if pos == 0 {
            return None;
        }
        Some((pos, total))
    }

    /// 1-based position of `idx` among element siblings with the same tag
    /// name, and their total count.
    fn type_position(&self, idx: usize) -> Option<(usize, usize)> {
        let parent = self.parent_element_idx(idx)?;
        let tag = self.tb.arena.get(idx)?.tag.clone();
        let siblings = &self.tb.arena.get(parent)?.children;
        let mut pos = 0;
        let mut total = 0;
        for child in siblings {
            if let Child::Elem(e) = child
                && self.tb.arena.get(*e).is_some_and(|n| n.tag == tag)
            {
                total += 1;
                if *e == idx {
                    pos = total;
                }
            }
        }
        if pos == 0 {
            return None;
        }
        Some((pos, total))
    }

    /// Check one compound selector against arena element `idx`.
    fn compound_matches(&self, compound: &CompoundSelector, idx: usize) -> bool {
        let node = match self.tb.arena.get(idx) {
            Some(n) => n,
            None => return false,
        };
        // Tag name match (ASCII case-insensitive; parser lowercases HTML tags).
        if let Some(t) = &compound.tag
            && node.tag.to_string().to_ascii_lowercase() != *t
        {
            return false;
        }
        let attr_val = |want: &str| {
            node.attrs.iter().find_map(|(name, value)| {
                name.to_string()
                    .eq_ignore_ascii_case(want)
                    .then_some(value.as_str())
            })
        };
        // ID match.
        if let Some(want_id) = &compound.id
            && attr_val("id") != Some(want_id.as_str())
        {
            return false;
        }
        // Class matches (all must be present).
        for class in &compound.classes {
            let has_class =
                attr_val("class").is_some_and(|v| v.split_whitespace().any(|c| c == class));
            if !has_class {
                return false;
            }
        }
        // Attribute matches.
        for attr in &compound.attrs {
            let Some(v) = attr_val(&attr.name) else {
                return false;
            };
            let ok = match attr.op {
                AttrOp::Exists => true,
                AttrOp::Eq => v == attr.value,
                AttrOp::Includes => v.split_whitespace().any(|w| w == attr.value),
                AttrOp::Dash => v == attr.value || v.starts_with(&format!("{}-", attr.value)),
                AttrOp::Prefix => v.starts_with(attr.value.as_str()),
                AttrOp::Suffix => v.ends_with(attr.value.as_str()),
                AttrOp::Substr => v.contains(attr.value.as_str()),
            };
            if !ok {
                return false;
            }
        }
        // Pseudo-classes.
        for pseudo in &compound.pseudos {
            if !self.pseudo_matches(pseudo, idx) {
                return false;
            }
        }
        true
    }

    /// Check one structural pseudo-class against arena element `idx`.
    fn pseudo_matches(&self, pseudo: &PseudoClass, idx: usize) -> bool {
        match pseudo {
            PseudoClass::FirstChild => self.sibling_position(idx).is_some_and(|(p, _)| p == 1),
            PseudoClass::LastChild => self.sibling_position(idx).is_some_and(|(p, t)| p == t),
            PseudoClass::OnlyChild => self.sibling_position(idx).is_some_and(|(_, t)| t == 1),
            PseudoClass::Empty => self.tb.arena.get(idx).is_some_and(|n| {
                n.children
                    .iter()
                    .all(|c| !matches!(c, Child::Elem(_) | Child::Text(_, _)))
            }),
            PseudoClass::Root => matches!(
                self.tb.arena.get(idx).and_then(|n| n.parent),
                Some(Parent::Doc)
            ),
            PseudoClass::FirstOfType => self.type_position(idx).is_some_and(|(p, _)| p == 1),
            PseudoClass::LastOfType => self.type_position(idx).is_some_and(|(p, t)| p == t),
            PseudoClass::OnlyOfType => self.type_position(idx).is_some_and(|(_, t)| t == 1),
            PseudoClass::NthChild(nth) => self
                .sibling_position(idx)
                .is_some_and(|(p, _)| nth.matches(p)),
            PseudoClass::NthLastChild(nth) => self
                .sibling_position(idx)
                .is_some_and(|(p, t)| nth.matches(t - p + 1)),
            PseudoClass::NthOfType(nth) => {
                self.type_position(idx).is_some_and(|(p, _)| nth.matches(p))
            }
            PseudoClass::NthLastOfType(nth) => self
                .type_position(idx)
                .is_some_and(|(p, t)| nth.matches(t - p + 1)),
            PseudoClass::Not(inner) => !self.compound_matches(inner, idx),
        }
    }

    /// `setAttribute(name, value)` on `el`: replaces the attribute's value
    /// when present, appends it otherwise. The name is ASCII-lowercased
    /// for HTML-namespace elements, per DOM.
    ///
    /// A handle retained from an earlier parse fails closed: the parse
    /// generation tag no longer matches, so the call is a no-op instead
    /// of mutating a stale or wrong arena slot.
    pub fn set_attribute(&mut self, el: ScriptElement, name: &str, value: &str) {
        if el.generation != self.tb.generation {
            return;
        }
        self.tb.set_attr(el.to_el(), name, value);
    }

    /// `document.createElement(tag)`: creates a new HTML-namespace
    /// element with the given tag name (ASCII-lowercased, per DOM for
    /// HTML documents), not yet attached to the document. The returned
    /// handle is valid for the current parse generation.
    pub fn create_element(&mut self, tag: &str) -> ScriptElement {
        let generation = self.tb.generation;
        let lower = tag.to_ascii_lowercase();
        let el = self.tb.create_element(
            TagName::from(lower.as_str()),
            Namespace::Html,
            Vec::new(),
            None,
        );
        ScriptElement::from_el(el, generation)
    }

    /// Get all attributes of an element as (name, value) pairs.
    /// Returns empty vec for stale handles.
    pub fn get_attributes(&self, el: ScriptElement) -> Vec<(String, String)> {
        if el.generation != self.tb.generation {
            return Vec::new();
        }
        let idx = el.to_el();
        if idx >= self.tb.arena.len() {
            return Vec::new();
        }
        self.tb.arena[idx]
            .attrs
            .iter()
            .map(|(name, value)| (name.to_string(), value.clone()))
            .collect()
    }

    /// Get the parent element of an element, or None if it's the root
    /// or the handle is stale.
    pub fn parent_element(&self, el: ScriptElement) -> Option<ScriptElement> {
        if el.generation != self.tb.generation {
            return None;
        }
        let idx = el.to_el();
        if idx >= self.tb.arena.len() {
            return None;
        }
        // The arena nodes have a `parent` field.
        match self.tb.arena[idx].parent {
            Some(Parent::Elem(parent_idx)) => {
                Some(ScriptElement::from_el(parent_idx, el.generation))
            }
            _ => None,
        }
    }

    /// Validate a handle: generation matches and the index is in range.
    fn check_el(&self, el: ScriptElement) -> Option<El> {
        if el.generation != self.tb.generation {
            return None;
        }
        let idx = el.to_el();
        if idx >= self.tb.arena.len() {
            return None;
        }
        Some(idx)
    }

    /// `childNodes`: element and text children in order (comments and
    /// processing instructions are skipped, per the DOM `childNodes`
    /// view used by scripts).
    pub fn child_nodes(&self, el: ScriptElement) -> Vec<ScriptChild> {
        let Some(idx) = self.check_el(el) else {
            return Vec::new();
        };
        self.tb.arena[idx]
            .children
            .iter()
            .filter_map(|c| match c {
                Child::Elem(e) => Some(ScriptChild::Element(ScriptElement::from_el(
                    *e,
                    el.generation,
                ))),
                Child::Text(t, _) => Some(ScriptChild::Text(t.clone())),
                _ => None,
            })
            .collect()
    }

    /// `children`: element children only, in order.
    pub fn child_elements(&self, el: ScriptElement) -> Vec<ScriptElement> {
        let Some(idx) = self.check_el(el) else {
            return Vec::new();
        };
        self.tb.arena[idx]
            .children
            .iter()
            .filter_map(|c| match c {
                Child::Elem(e) => Some(ScriptElement::from_el(*e, el.generation)),
                _ => None,
            })
            .collect()
    }

    /// `firstChild` / `lastChild`.
    pub fn first_child(&self, el: ScriptElement) -> Option<ScriptChild> {
        self.child_nodes(el).into_iter().next()
    }

    pub fn last_child(&self, el: ScriptElement) -> Option<ScriptChild> {
        // `child_nodes` is small in practice; keep the simple version.
        let nodes = self.child_nodes(el);
        nodes.into_iter().last()
    }

    /// Descendant element indices of `idx` in document order (iterative:
    /// adversarial nesting must not overflow the stack).
    fn descendant_elements(&self, idx: El) -> Vec<El> {
        let mut out = Vec::new();
        let mut stack = vec![idx];
        while let Some(cur) = stack.pop() {
            if cur != idx {
                out.push(cur);
            }
            if let Some(node) = self.tb.arena.get(cur) {
                for child in node.children.iter().rev() {
                    if let Child::Elem(e) = child {
                        stack.push(*e);
                    }
                }
            }
        }
        out
    }

    /// `el.querySelector(sel)`: first descendant of `el` (not `el`
    /// itself) matching the selector, in document order — the earliest
    /// descendant in document order matching any comma-separated group,
    /// not the first group's first match.
    pub fn query_selector_in(&self, el: ScriptElement, sel: &str) -> Option<ScriptElement> {
        self.query_selector_all_in(el, sel).into_iter().next()
    }

    /// `el.querySelectorAll(sel)`: all descendants of `el` matching the
    /// selector, in document order. Comma-separated groups are unioned
    /// like [`query_selector_all`](Self::query_selector_all): each
    /// descendant appears once, in document order, not group order.
    /// (`descendant_elements` already walks `children` lists, so it is
    /// document order even after moves; no dedup scan is needed because
    /// each descendant is visited exactly once.)
    pub fn query_selector_all_in(&self, el: ScriptElement, sel: &str) -> Vec<ScriptElement> {
        let Some(idx) = self.check_el(el) else {
            return Vec::new();
        };
        let generation = el.generation;
        let complexes: Vec<_> = split_selector_groups(sel)
            .into_iter()
            .filter(|group| !group.is_empty())
            .filter_map(|group| parse_complex_selector(&group))
            .collect();
        if complexes.is_empty() {
            return Vec::new();
        }
        self.descendant_elements(idx)
            .into_iter()
            .filter(|desc| {
                complexes
                    .iter()
                    .any(|complex| self.complex_matches(complex, *desc))
            })
            .map(|desc| ScriptElement::from_el(desc, generation))
            .collect()
    }

    /// Whether `anc` is an ancestor of (or equal to) `idx`.
    fn is_ancestor_or_self(&self, anc: El, idx: El) -> bool {
        let mut cur = idx;
        // Bound the walk by the arena length:
        // parent links form a tree, so a longer walk means a cycle
        // (unreachable through the guarded mutation APIs) — fail closed
        // instead of hanging.
        for _ in 0..self.tb.arena.len() {
            if cur == anc {
                return true;
            }
            match self.tb.arena.get(cur).and_then(|n| n.parent) {
                Some(Parent::Elem(p)) => cur = p,
                _ => return false,
            }
        }
        false
    }

    /// Detach `idx` from its parent, if attached.
    fn detach(&mut self, idx: El) {
        let parent = match self.tb.arena.get(idx).and_then(|n| n.parent) {
            Some(Parent::Elem(p)) => p,
            _ => {
                if let Some(node) = self.tb.arena.get_mut(idx) {
                    node.parent = None;
                }
                return;
            }
        };
        if let Some(pnode) = self.tb.arena.get_mut(parent) {
            pnode
                .children
                .retain(|c| !matches!(c, Child::Elem(e) if *e == idx));
        }
        if let Some(node) = self.tb.arena.get_mut(idx) {
            node.parent = None;
        }
        self.tb.bump_mutation();
    }

    /// `parent.appendChild(child)`: attach `child` (detached, or moved
    /// from its current parent per DOM) as the last child. Fails closed
    /// (`false`) on stale handles or hierarchy cycles.
    pub fn append_child(&mut self, parent: ScriptElement, child: ScriptElement) -> bool {
        let (Some(p), Some(c)) = (self.check_el(parent), self.check_el(child)) else {
            return false;
        };
        if p == c || self.is_ancestor_or_self(c, p) {
            return false;
        }
        self.detach(c);
        self.tb.arena[c].parent = Some(Parent::Elem(p));
        self.tb.arena[p].children.push(Child::Elem(c));
        self.tb.bump_mutation();
        true
    }

    /// `parent.removeChild(child)`: detach `child` when its parent is
    /// `parent`.
    pub fn remove_child(&mut self, parent: ScriptElement, child: ScriptElement) -> bool {
        let (Some(p), Some(c)) = (self.check_el(parent), self.check_el(child)) else {
            return false;
        };
        if !matches!(self.tb.arena[c].parent, Some(Parent::Elem(pp)) if pp == p) {
            return false;
        }
        self.detach(c);
        true
    }

    /// `parent.insertBefore(new, reference)`: insert `new` before
    /// `reference` (a child of `parent`); `None` reference appends.
    pub fn insert_before(
        &mut self,
        parent: ScriptElement,
        new: ScriptElement,
        reference: Option<ScriptElement>,
    ) -> bool {
        let (Some(p), Some(n)) = (self.check_el(parent), self.check_el(new)) else {
            return false;
        };
        if p == n || self.is_ancestor_or_self(n, p) {
            return false;
        }
        let r = match reference {
            Some(rr) => {
                let Some(ri) = self.check_el(rr) else {
                    return false;
                };
                if !matches!(self.tb.arena[ri].parent, Some(Parent::Elem(pp)) if pp == p) {
                    return false;
                }
                Some(ri)
            }
            None => None,
        };
        self.detach(n);
        self.tb.arena[n].parent = Some(Parent::Elem(p));
        let children = &mut self.tb.arena[p].children;
        match r {
            Some(ri) => {
                if let Some(pos) = children
                    .iter()
                    .position(|c| matches!(c, Child::Elem(e) if *e == ri))
                {
                    children.insert(pos, Child::Elem(n));
                } else {
                    children.push(Child::Elem(n));
                }
            }
            None => children.push(Child::Elem(n)),
        }
        self.tb.bump_mutation();
        true
    }

    /// `document.documentElement`: the `<html>` element.
    pub fn document_element(&self) -> Option<ScriptElement> {
        let generation = self.tb.generation;
        self.tb.arena.iter().enumerate().find_map(|(i, n)| {
            (n.parent == Some(Parent::Doc) && n.tag.to_string().eq_ignore_ascii_case("html"))
                .then(|| ScriptElement::from_el(i, generation))
        })
    }

    /// `document.body`: the first `<body>` element.
    pub fn body_element(&self) -> Option<ScriptElement> {
        let generation = self.tb.generation;
        self.tb.arena.iter().enumerate().find_map(|(i, n)| {
            n.tag
                .to_string()
                .eq_ignore_ascii_case("body")
                .then(|| ScriptElement::from_el(i, generation))
        })
    }

    /// `textContent`: concatenated descendant text, in document order.
    pub fn text_content(&self, el: ScriptElement) -> String {
        let Some(root) = self.check_el(el) else {
            return String::new();
        };
        let mut out = String::new();
        // Stack of (element, next child index): iterative pre-order.
        let mut stack: Vec<(El, usize)> = vec![(root, 0)];
        while let Some((cur, idx)) = stack.pop() {
            let Some(node) = self.tb.arena.get(cur) else {
                continue;
            };
            if idx < node.children.len() {
                stack.push((cur, idx + 1));
                match &node.children[idx] {
                    Child::Elem(e) => stack.push((*e, 0)),
                    Child::Text(t, _) => out.push_str(t),
                    _ => {}
                }
            }
        }
        out
    }

    /// `textContent = text`: replace children with a single text node.
    pub fn set_text_content(&mut self, el: ScriptElement, text: &str) {
        let Some(idx) = self.check_el(el) else {
            return;
        };
        let node = &mut self.tb.arena[idx];
        node.children.clear();
        if !text.is_empty() {
            node.children.push(Child::Text(text.to_string(), None));
        }
        self.tb.bump_mutation();
    }

    /// Serialize one arena element (open tag, children, close tag).
    fn write_element(&self, el: El, out: &mut String) {
        enum Job {
            Enter(El),
            Exit(String),
            Text(String),
            Comment(String),
        }
        let mut stack = vec![Job::Enter(el)];
        while let Some(job) = stack.pop() {
            match job {
                Job::Enter(e) => {
                    let Some(node) = self.tb.arena.get(e) else {
                        continue;
                    };
                    let tag = node.tag.to_string();
                    out.push('<');
                    out.push_str(&tag);
                    for (k, v) in &node.attrs {
                        out.push(' ');
                        out.push_str(k.as_ref());
                        out.push_str("=\"");
                        out.push_str(&v.replace('&', "&amp;").replace('"', "&quot;"));
                        out.push('"');
                    }
                    if is_void_element(&tag) {
                        out.push('>');
                    } else {
                        out.push('>');
                        stack.push(Job::Exit(tag));
                        for child in node.children.iter().rev() {
                            match child {
                                Child::Elem(c) => stack.push(Job::Enter(*c)),
                                Child::Text(t, _) => stack.push(Job::Text(t.clone())),
                                Child::Comment(d, _) => stack.push(Job::Comment(d.clone())),
                                _ => {}
                            }
                        }
                    }
                }
                Job::Exit(tag) => {
                    out.push_str("</");
                    out.push_str(&tag);
                    out.push('>');
                }
                Job::Text(t) => {
                    out.push_str(&t.replace('&', "&amp;").replace('<', "&lt;"));
                }
                Job::Comment(d) => {
                    out.push_str("<!--");
                    out.push_str(&d);
                    out.push_str("-->");
                }
            }
        }
    }

    /// `innerHTML`: serialized children of `el`.
    pub fn inner_html(&self, el: ScriptElement) -> String {
        let Some(idx) = self.check_el(el) else {
            return String::new();
        };
        let mut out = String::new();
        if let Some(node) = self.tb.arena.get(idx) {
            let children: Vec<Child> = node.children.clone();
            // Serialize each child via a scratch walk.
            for child in &children {
                match child {
                    Child::Elem(e) => self.write_element(*e, &mut out),
                    Child::Text(t, _) => {
                        out.push_str(&t.replace('&', "&amp;").replace('<', "&lt;"))
                    }
                    Child::Comment(d, _) => {
                        out.push_str("<!--");
                        out.push_str(d);
                        out.push_str("-->");
                    }
                    _ => {}
                }
            }
        }
        out
    }

    /// Import a fragment `Node` tree under arena element `parent`.
    /// `Node` has a manual [`Drop`](Node) impl, so fields are moved via
    /// `mem::take`/`mem::replace` (the [`Node::take_children`] pattern),
    /// never by value. Iterative: no recursion.
    fn import_fragment_nodes(&mut self, parent: El, nodes: Vec<Node>) {
        // (parent, remaining siblings, consumed count).
        let mut stack: Vec<(El, Vec<Node>, usize)> = vec![(parent, nodes, 0)];
        while !stack.is_empty() {
            let (fparent, mut node) = {
                let top = stack.last_mut().expect("non-empty");
                if top.2 >= top.1.len() {
                    stack.pop();
                    continue;
                }
                let n = std::mem::replace(&mut top.1[top.2], Node::Document(Vec::new()));
                top.2 += 1;
                (top.0, n)
            };
            match &mut node {
                Node::Element(e) => {
                    let tag = std::mem::replace(&mut e.tag, TagName::from("template"));
                    let ns = e.ns.clone();
                    let attrs = std::mem::take(&mut e.attrs);
                    let kids = e
                        .template_contents
                        .take()
                        .unwrap_or_else(|| std::mem::take(&mut e.children));
                    let el = self.tb.create_element(tag, ns, attrs, None);
                    self.tb.arena[el].parent = Some(Parent::Elem(fparent));
                    self.tb.arena[fparent].children.push(Child::Elem(el));
                    stack.push((el, kids, 0));
                }
                Node::Text { text, .. } => {
                    let t = std::mem::take(text);
                    self.tb.arena[fparent].children.push(Child::Text(t, None));
                }
                Node::Comment { data, .. } => {
                    let d = std::mem::take(data);
                    self.tb.arena[fparent]
                        .children
                        .push(Child::Comment(d, None));
                }
                Node::Document(kids) => {
                    let k = std::mem::take(kids);
                    stack.push((fparent, k, 0));
                }
                _ => {}
            }
            // `node` drops here, gutted: the iterative `Drop` finds no
            // children to walk.
        }
    }

    /// `innerHTML = html`: replace children with the parsed fragment
    /// (scripts do not execute, per DOM).
    pub fn set_inner_html(&mut self, el: ScriptElement, html: &str) {
        let Some(idx) = self.check_el(el) else {
            return;
        };
        let (tag, ns, attrs) = {
            let node = &self.tb.arena[idx];
            (node.tag.clone(), node.ns.clone(), node.attrs.clone())
        };
        let ctx = Element {
            tag,
            ns,
            attrs,
            children: Vec::new(),
            template_contents: None,
            shadow_root: None,
            loc: None,
        };
        let (mut frag, _) =
            parse_fragment_with_errors_and_scripting(html.as_bytes(), &ctx, false, &[]);
        let kids = match &mut frag {
            Node::Document(kids) => std::mem::take(kids),
            // A bare non-document root (should not happen): wrap it.
            _ => vec![std::mem::replace(&mut frag, Node::Document(Vec::new()))],
        };
        self.tb.arena[idx].children.clear();
        self.import_fragment_nodes(idx, kids);
        self.tb.bump_mutation();
    }

    /// `(id, element)` for every element with an `id` attribute, in
    /// arena (= document) order. Backs id-named window properties.
    pub fn element_ids(&self) -> Vec<(String, ScriptElement)> {
        let generation = self.tb.generation;
        let mut out = Vec::new();
        for (i, node) in self.tb.arena.iter().enumerate() {
            if let Some((_, v)) = node
                .attrs
                .iter()
                .find(|(k, _)| k.to_string().eq_ignore_ascii_case("id"))
            {
                out.push((v.clone(), ScriptElement::from_el(i, generation)));
            }
        }
        out
    }

    /// Lowercased tag name of an element.
    pub fn tag_name(&self, el: ScriptElement) -> Option<String> {
        let idx = self.check_el(el)?;
        Some(self.tb.arena[idx].tag.to_string().to_ascii_lowercase())
    }

    /// `nextElementSibling` (`next = true`) / `previousElementSibling`.
    /// Returns None for stale handles or when there is no such sibling.
    pub fn sibling_element(&self, el: ScriptElement, next: bool) -> Option<ScriptElement> {
        let idx = self.check_el(el)?;
        let parent_idx = match self.tb.arena[idx].parent? {
            Parent::Elem(p) => p,
            Parent::Doc => return None,
        };
        let siblings = &self.tb.arena[parent_idx].children;
        let pos = siblings
            .iter()
            .position(|c| matches!(c, Child::Elem(e) if *e == idx))?;
        let mut i = pos;
        loop {
            if next {
                i += 1;
                if i >= siblings.len() {
                    return None;
                }
            } else {
                if i == 0 {
                    return None;
                }
                i -= 1;
            }
            if let Child::Elem(e) = siblings[i] {
                return Some(ScriptElement::from_el(e, el.generation));
            }
        }
    }

    /// Shallow arena copy of one element (no children, no parent).
    fn push_element_copy(&mut self, idx: usize) -> usize {
        let src = &self.tb.arena[idx];
        let new_idx = self.tb.arena.len();
        self.tb.arena.push(ArenaNode {
            tag: src.tag.clone(),
            ns: src.ns.clone(),
            attrs: src.attrs.clone(),
            children: Vec::new(),
            template_contents: None,
            dsr: None,
            parent: None,
            loc: None,
            annotation_xml_html: false,
        });
        new_idx
    }

    /// `cloneNode(deep)`: clone into a new detached arena element.
    /// Iterative (the source can be adversarially deep). Returns None
    /// for stale handles.
    pub fn clone_element(&mut self, el: ScriptElement, deep: bool) -> Option<ScriptElement> {
        let idx = self.check_el(el)?;
        let new_idx = self.push_element_copy(idx);
        if deep {
            // (source, copy) pairs whose child lists still need copying.
            // Draining through `drain_clone_pending` copies both
            // `children` and `template_contents`, so a nested
            // `<template>`'s contents survive the deep clone.
            let mut pending = vec![(idx, new_idx)];
            self.tb.drain_clone_pending(&mut pending);
        }
        self.tb.bump_mutation();
        Some(ScriptElement::from_el(new_idx, el.generation))
    }

    /// `template.content`: a detached fragment holding a deep copy of the
    /// template's contents. Returns None for stale handles or
    /// non-template elements.
    pub fn template_content(&mut self, el: ScriptElement) -> Option<ScriptElement> {
        let idx = self.check_el(el)?;
        if self.tb.arena[idx].tag != "template" {
            return None;
        }
        let contents = self.tb.arena[idx].template_contents.clone()?;
        let frag_idx = self.tb.arena.len();
        self.tb.arena.push(ArenaNode {
            tag: TagName::from("template"),
            ns: Namespace::Html,
            attrs: Vec::new(),
            children: Vec::new(),
            template_contents: None,
            dsr: None,
            parent: None,
            loc: None,
            annotation_xml_html: false,
        });
        // Deep-copy the contents into the fragment, carrying nested
        // template contents along (they live in `template_contents`,
        // not `children`).
        let mut pending: Vec<(El, El)> = Vec::new();
        let mut new_contents = Vec::new();
        copy_child_list(
            &mut self.tb.arena,
            &contents,
            Parent::Elem(frag_idx),
            &mut new_contents,
            &mut pending,
        );
        self.tb.arena[frag_idx].children = new_contents;
        self.tb.drain_clone_pending(&mut pending);
        Some(ScriptElement::from_el(frag_idx, el.generation))
    }
}

/// A `childNodes` entry: an element or a text node.
#[derive(Clone)]
pub enum ScriptChild {
    Element(ScriptElement),
    Text(String),
}

/// HTML void elements: no close tag when serializing.
fn is_void_element(tag: &str) -> bool {
    matches!(
        tag,
        "area"
            | "base"
            | "br"
            | "col"
            | "embed"
            | "hr"
            | "img"
            | "input"
            | "link"
            | "meta"
            | "param"
            | "source"
            | "track"
            | "wbr"
    )
}

// ---------------------------------------------------------------------------
// Selector parsing (for `ScriptHost::query_selector`/`query_selector_all`)
// ---------------------------------------------------------------------------

/// Split a selector list on top-level commas (commas inside `[]` or `()`
/// do not separate groups).
fn split_selector_groups(sel: &str) -> Vec<String> {
    let mut groups = Vec::new();
    let mut depth_bracket = 0usize;
    let mut depth_paren = 0usize;
    let mut start = 0;
    for (i, c) in sel.char_indices() {
        match c {
            '[' => depth_bracket += 1,
            ']' => depth_bracket = depth_bracket.saturating_sub(1),
            '(' => depth_paren += 1,
            ')' => depth_paren = depth_paren.saturating_sub(1),
            ',' if depth_bracket == 0 && depth_paren == 0 => {
                groups.push(sel[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    groups.push(sel[start..].trim().to_string());
    groups
}

/// One compound selector: tag/`#id`/`.class`/`[attr]`/`:pseudo` run.
struct CompoundSelector {
    /// Lowercased tag name; `None` is the universal selector.
    tag: Option<String>,
    id: Option<String>,
    classes: Vec<String>,
    attrs: Vec<AttrSelector>,
    pseudos: Vec<PseudoClass>,
}

/// Attribute selector with operator.
struct AttrSelector {
    /// Lowercased attribute name.
    name: String,
    op: AttrOp,
    value: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AttrOp {
    Exists,
    Eq,
    Includes,
    Dash,
    Prefix,
    Suffix,
    Substr,
}

/// A structural pseudo-class.
enum PseudoClass {
    FirstChild,
    LastChild,
    OnlyChild,
    Empty,
    Root,
    FirstOfType,
    LastOfType,
    OnlyOfType,
    NthChild(Nth),
    NthLastChild(Nth),
    NthOfType(Nth),
    NthLastOfType(Nth),
    Not(Box<CompoundSelector>),
}

/// `An+B` microsyntax value for `:nth-*()`.
#[derive(Clone, Copy)]
struct Nth {
    a: i32,
    b: i32,
}

impl Nth {
    /// Whether 1-based `index` matches `An+B`.
    fn matches(self, index: usize) -> bool {
        let n = index as i32;
        if self.a == 0 {
            return n == self.b;
        }
        let d = n - self.b;
        if self.a > 0 {
            d >= 0 && d % self.a == 0
        } else {
            d <= 0 && d % self.a == 0
        }
    }
}

/// Parse `An+B` (`odd`, `even`, `3`, `2n+1`, `-n+3`, `n`, ...).
fn parse_nth(s: &str) -> Option<Nth> {
    let s = s.trim().to_ascii_lowercase();
    if s == "odd" {
        return Some(Nth { a: 2, b: 1 });
    }
    if s == "even" {
        return Some(Nth { a: 2, b: 0 });
    }
    if let Some(npos) = s.find('n') {
        let a_part = s[..npos].trim();
        let b_part = s[npos + 1..].trim();
        let a = match a_part {
            "" | "+" => 1,
            "-" => -1,
            _ => a_part.parse::<i32>().ok()?,
        };
        let b = if b_part.is_empty() {
            0
        } else {
            b_part.parse::<i32>().ok()?
        };
        return Some(Nth { a, b });
    }
    Some(Nth {
        a: 0,
        b: s.parse::<i32>().ok()?,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Combinator {
    Descendant,
    Child,
    NextSibling,
    LaterSibling,
}

/// A parsed complex selector: compounds left-to-right, `combinators[i]`
/// joining `compounds[i]` to `compounds[i + 1]`.
struct ComplexSelector {
    compounds: Vec<CompoundSelector>,
    combinators: Vec<Combinator>,
}

/// Parse one selector group (no top-level commas) into a complex selector.
/// Returns `None` for anything unsupported (other pseudo-classes/elements,
/// namespaces, `:has()`, ...).
fn parse_complex_selector(sel: &str) -> Option<ComplexSelector> {
    let sel = sel.trim();
    if sel.is_empty() {
        return None;
    }
    #[derive(PartialEq)]
    enum Token {
        Compound(String),
        Comb(Combinator),
        Space,
    }
    // Phase 1: raw tokens; whitespace runs collapse to one `Space`.
    let mut toks: Vec<Token> = Vec::new();
    let bytes = sel.as_bytes();
    let mut i = 0usize;
    let mut start = 0usize;
    let mut depth_bracket = 0usize;
    let mut depth_paren = 0usize;
    while i < bytes.len() {
        let c = bytes[i] as char;
        match c {
            '[' => depth_bracket += 1,
            ']' => depth_bracket = depth_bracket.saturating_sub(1),
            '(' => depth_paren += 1,
            ')' => depth_paren = depth_paren.saturating_sub(1),
            _ => {}
        }
        let top = depth_bracket == 0 && depth_paren == 0;
        if top && matches!(c, '>' | '+' | '~') {
            if start < i {
                toks.push(Token::Compound(sel[start..i].trim().to_string()));
            }
            toks.push(Token::Comb(match c {
                '>' => Combinator::Child,
                '+' => Combinator::NextSibling,
                _ => Combinator::LaterSibling,
            }));
            i += 1;
            start = i;
            continue;
        }
        if top && c.is_whitespace() {
            if start < i {
                toks.push(Token::Compound(sel[start..i].trim().to_string()));
            }
            toks.push(Token::Space);
            i += 1;
            while i < bytes.len() && (bytes[i] as char).is_whitespace() {
                i += 1;
            }
            start = i;
            continue;
        }
        i += 1;
    }
    if start < bytes.len() {
        toks.push(Token::Compound(sel[start..].trim().to_string()));
    }
    // Phase 2: whitespace around `>`/`+`/`~` is insignificant.
    let mut clean: Vec<Token> = Vec::new();
    for (idx, t) in toks.iter().enumerate() {
        if *t == Token::Space {
            let next_to_comb = (idx > 0 && matches!(toks[idx - 1], Token::Comb(_)))
                || (idx + 1 < toks.len() && matches!(toks[idx + 1], Token::Comb(_)));
            if next_to_comb {
                continue;
            }
        }
        clean.push(match t {
            Token::Compound(s) => Token::Compound(s.clone()),
            Token::Comb(c) => Token::Comb(*c),
            Token::Space => Token::Space,
        });
    }
    // Phase 3: tokens must alternate `Compound (Comb Compound)*`, where a
    // surviving `Space` is the descendant combinator.
    let mut compounds: Vec<CompoundSelector> = Vec::new();
    let mut combinators: Vec<Combinator> = Vec::new();
    let mut expect_compound = true;
    for t in clean {
        match t {
            Token::Compound(s) if expect_compound && !s.is_empty() => {
                compounds.push(parse_compound(&s)?);
                expect_compound = false;
            }
            Token::Comb(c) if !expect_compound => {
                combinators.push(c);
                expect_compound = true;
            }
            Token::Space if !expect_compound => {
                combinators.push(Combinator::Descendant);
                expect_compound = true;
            }
            _ => return None,
        }
    }
    if expect_compound || compounds.is_empty() {
        return None;
    }
    debug_assert_eq!(combinators.len() + 1, compounds.len());
    Some(ComplexSelector {
        compounds,
        combinators,
    })
}

/// Parse one compound selector (no top-level combinators).
fn parse_compound(sel: &str) -> Option<CompoundSelector> {
    let sel = sel.trim();
    if sel.is_empty() {
        return None;
    }
    let mut compound = CompoundSelector {
        tag: None,
        id: None,
        classes: Vec::new(),
        attrs: Vec::new(),
        pseudos: Vec::new(),
    };
    let mut rest = sel;
    // Tag name or `*` comes first (before any `#`, `.`, `[`, `:`).
    let tag_end = rest.find(['#', '.', '[', ':']).unwrap_or(rest.len());
    if tag_end > 0 {
        let t = rest[..tag_end].trim();
        if t != "*" {
            if t.is_empty() || !is_name_start(t) {
                return None;
            }
            compound.tag = Some(t.to_ascii_lowercase());
        }
    }
    rest = &rest[tag_end..];

    while !rest.is_empty() {
        if rest.starts_with('#') {
            let s = &rest[1..];
            let end = s.find(['.', '[', '#', ':']).unwrap_or(s.len());
            if end == 0 {
                return None;
            }
            compound.id = Some(s[..end].to_string());
            rest = &s[end..];
        } else if rest.starts_with('.') {
            let s = &rest[1..];
            let end = s.find(['.', '[', '#', ':']).unwrap_or(s.len());
            if end == 0 {
                return None;
            }
            compound.classes.push(s[..end].to_string());
            rest = &s[end..];
        } else if rest.starts_with('[') {
            let (attr, next) = parse_attr_selector(rest)?;
            compound.attrs.push(attr);
            rest = next;
        } else if rest.starts_with(':') {
            // Pseudo-class (single colon; `::` pseudo-elements unsupported).
            if rest.starts_with("::") {
                return None;
            }
            let s = &rest[1..];
            let name_end = s.find(['(', '.', '[', '#', ':']).unwrap_or(s.len());
            let name = s[..name_end].to_ascii_lowercase();
            rest = &s[name_end..];
            let arg = if rest.starts_with('(') {
                let close = find_close_paren(rest)?;
                let arg = rest[1..close].to_string();
                rest = &rest[close + 1..];
                Some(arg)
            } else {
                None
            };
            compound.pseudos.push(parse_pseudo(&name, arg.as_deref())?);
        } else {
            return None;
        }
    }
    Some(compound)
}

/// Whether `s` can start a tag name (ASCII letter, `_`, `-`, non-ASCII).
fn is_name_start(s: &str) -> bool {
    s.chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_' || c == '-' || !c.is_ascii())
}

/// Parse `[attr]`, `[attr=val]`, `[attr~=val]`, `[attr|=val]`,
/// `[attr^=val]`, `[attr$=val]`, `[attr*=val]` at the start of `rest`.
/// Returns the selector and the remainder after `]`.
fn parse_attr_selector(rest: &str) -> Option<(AttrSelector, &str)> {
    debug_assert!(rest.starts_with('['));
    let close = rest.find(']')?;
    let inner = rest[1..close].trim();
    let next = &rest[close + 1..];
    // Two-character operators first (`[a*=v]` also contains `=`).
    let two_op = ["~=", "|=", "^=", "$=", "*="]
        .into_iter()
        .filter_map(|op| inner.find(op).map(|pos| (pos, op)))
        .min_by_key(|(pos, _)| *pos);
    let (name, op, value) = if let Some((pos, op_str)) = two_op {
        let op = match op_str {
            "~=" => AttrOp::Includes,
            "|=" => AttrOp::Dash,
            "^=" => AttrOp::Prefix,
            "$=" => AttrOp::Suffix,
            _ => AttrOp::Substr,
        };
        (inner[..pos].trim(), op, inner[pos + 2..].trim())
    } else if let Some(pos) = inner.find('=') {
        (inner[..pos].trim(), AttrOp::Eq, inner[pos + 1..].trim())
    } else {
        (inner, AttrOp::Exists, "")
    };
    if name.is_empty() {
        return None;
    }
    let value = strip_quotes(value).to_string();
    if op != AttrOp::Exists && value.is_empty() {
        return None;
    }
    Some((
        AttrSelector {
            name: name.to_ascii_lowercase(),
            op,
            value,
        },
        next,
    ))
}

/// Strip one pair of matching ASCII quotes.
fn strip_quotes(s: &str) -> &str {
    let b = s.as_bytes();
    if b.len() >= 2 && (b[0] == b'"' || b[0] == b'\'') && b[b.len() - 1] == b[0] {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

/// Index of the `)` closing the `(` at `rest[0]`.
fn find_close_paren(rest: &str) -> Option<usize> {
    debug_assert!(rest.starts_with('('));
    let mut depth = 0usize;
    for (i, c) in rest.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse a pseudo-class name + optional `(arg)`.
fn parse_pseudo(name: &str, arg: Option<&str>) -> Option<PseudoClass> {
    let nth = |a: Option<&str>| a.and_then(parse_nth);
    match name {
        "first-child" if arg.is_none() => Some(PseudoClass::FirstChild),
        "last-child" if arg.is_none() => Some(PseudoClass::LastChild),
        "only-child" if arg.is_none() => Some(PseudoClass::OnlyChild),
        "empty" if arg.is_none() => Some(PseudoClass::Empty),
        "root" if arg.is_none() => Some(PseudoClass::Root),
        "first-of-type" if arg.is_none() => Some(PseudoClass::FirstOfType),
        "last-of-type" if arg.is_none() => Some(PseudoClass::LastOfType),
        "only-of-type" if arg.is_none() => Some(PseudoClass::OnlyOfType),
        "nth-child" => nth(arg).map(PseudoClass::NthChild),
        "nth-last-child" => nth(arg).map(PseudoClass::NthLastChild),
        "nth-of-type" => nth(arg).map(PseudoClass::NthOfType),
        "nth-last-of-type" => nth(arg).map(PseudoClass::NthLastOfType),
        "not" => {
            // `:not()` takes a single compound here (no combinators): a
            // complex argument fails closed as unsupported.
            let a = arg?;
            if !is_single_compound(a) {
                return None;
            }
            let inner = parse_compound(a)?;
            Some(PseudoClass::Not(Box::new(inner)))
        }
        _ => None,
    }
}

/// Whether `s` is a single compound selector (no top-level combinators),
/// for the `:not()` argument check.
fn is_single_compound(s: &str) -> bool {
    let mut depth_bracket = 0usize;
    let mut depth_paren = 0usize;
    for c in s.chars() {
        match c {
            '[' => depth_bracket += 1,
            ']' => depth_bracket = depth_bracket.saturating_sub(1),
            '(' => depth_paren += 1,
            ')' => depth_paren = depth_paren.saturating_sub(1),
            _ => {}
        }
        if depth_bracket == 0
            && depth_paren == 0
            && (c.is_whitespace() || matches!(c, '>' | '+' | '~' | ','))
        {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Static tables
// ---------------------------------------------------------------------------

/// Special elements (HTML namespace), WHATWG 13.2.4.2.
const SPECIAL_HTML: &[NameAtom] = &[
    NameAtom::ADDRESS,
    NameAtom::APPLET,
    NameAtom::AREA,
    NameAtom::ARTICLE,
    NameAtom::ASIDE,
    NameAtom::BASE,
    NameAtom::BASEFONT,
    NameAtom::BGSOUND,
    NameAtom::BLOCKQUOTE,
    NameAtom::BODY,
    NameAtom::BR,
    NameAtom::BUTTON,
    NameAtom::CAPTION,
    NameAtom::CENTER,
    NameAtom::COL,
    NameAtom::COLGROUP,
    NameAtom::DD,
    NameAtom::DETAILS,
    NameAtom::DIR,
    NameAtom::DIV,
    NameAtom::DL,
    NameAtom::DT,
    NameAtom::EMBED,
    NameAtom::FIELDSET,
    NameAtom::FIGCAPTION,
    NameAtom::FIGURE,
    NameAtom::FOOTER,
    NameAtom::FORM,
    NameAtom::FRAME,
    NameAtom::FRAMESET,
    NameAtom::H1,
    NameAtom::H2,
    NameAtom::H3,
    NameAtom::H4,
    NameAtom::H5,
    NameAtom::H6,
    NameAtom::HEAD,
    NameAtom::HEADER,
    NameAtom::HGROUP,
    NameAtom::HR,
    NameAtom::HTML,
    NameAtom::IFRAME,
    NameAtom::IMG,
    NameAtom::INPUT,
    NameAtom::KEYGEN,
    NameAtom::LI,
    NameAtom::LINK,
    NameAtom::LISTING,
    NameAtom::MAIN,
    NameAtom::MARQUEE,
    NameAtom::MENU,
    NameAtom::META,
    NameAtom::NAV,
    NameAtom::NOEMBED,
    NameAtom::NOFRAMES,
    NameAtom::NOSCRIPT,
    NameAtom::OBJECT,
    NameAtom::OL,
    NameAtom::P,
    NameAtom::PARAM,
    NameAtom::PLAINTEXT,
    NameAtom::PRE,
    NameAtom::SCRIPT,
    NameAtom::SEARCH,
    NameAtom::SECTION,
    NameAtom::SELECT,
    NameAtom::SOURCE,
    NameAtom::STYLE,
    NameAtom::SUMMARY,
    NameAtom::TABLE,
    NameAtom::TBODY,
    NameAtom::TD,
    NameAtom::TEMPLATE,
    NameAtom::TEXTAREA,
    NameAtom::TFOOT,
    NameAtom::TH,
    NameAtom::THEAD,
    NameAtom::TITLE,
    NameAtom::TR,
    NameAtom::TRACK,
    NameAtom::UL,
    NameAtom::WBR,
    NameAtom::XMP,
];

/// Elements that trigger implied end tags, WHATWG 13.2.6.2.
const IMPLIED: &[NameAtom] = &[
    NameAtom::DD,
    NameAtom::DT,
    NameAtom::LI,
    NameAtom::OPTGROUP,
    NameAtom::OPTION,
    NameAtom::P,
    NameAtom::RB,
    NameAtom::RP,
    NameAtom::RT,
    NameAtom::RTC,
];

/// Extra elements for "implied end tags thoroughly".
const IMPLIED_THOROUGH_EXTRA: &[NameAtom] = &[
    NameAtom::CAPTION,
    NameAtom::COLGROUP,
    NameAtom::TBODY,
    NameAtom::TD,
    NameAtom::TFOOT,
    NameAtom::TH,
    NameAtom::THEAD,
    NameAtom::TR,
];

/// Default scope list, WHATWG 13.2.4.2 (HTML-namespace members).
const SCOPE_DEFAULT: &[NameAtom] = &[
    NameAtom::APPLET,
    NameAtom::CAPTION,
    NameAtom::HTML,
    NameAtom::TABLE,
    NameAtom::TD,
    NameAtom::TH,
    NameAtom::MARQUEE,
    NameAtom::OBJECT,
    NameAtom::SELECT,
    NameAtom::TEMPLATE,
];

/// Table scope list.
const SCOPE_TABLE: &[NameAtom] = &[NameAtom::HTML, NameAtom::TABLE, NameAtom::TEMPLATE];

/// Tags that break out of foreign content, WHATWG 13.2.6.5.
const FOREIGN_BREAKOUT: &[NameAtom] = &[
    NameAtom::B,
    NameAtom::BIG,
    NameAtom::BLOCKQUOTE,
    NameAtom::BODY,
    NameAtom::BR,
    NameAtom::CENTER,
    NameAtom::CODE,
    NameAtom::DD,
    NameAtom::DIV,
    NameAtom::DL,
    NameAtom::DT,
    NameAtom::EM,
    NameAtom::EMBED,
    NameAtom::H1,
    NameAtom::H2,
    NameAtom::H3,
    NameAtom::H4,
    NameAtom::H5,
    NameAtom::H6,
    NameAtom::HEAD,
    NameAtom::HR,
    NameAtom::I,
    NameAtom::IMG,
    NameAtom::LI,
    NameAtom::LISTING,
    NameAtom::MENU,
    NameAtom::META,
    NameAtom::NOBR,
    NameAtom::OL,
    NameAtom::P,
    NameAtom::PRE,
    NameAtom::RUBY,
    NameAtom::S,
    NameAtom::SMALL,
    NameAtom::SPAN,
    NameAtom::STRONG,
    NameAtom::STRIKE,
    NameAtom::SUB,
    NameAtom::SUP,
    NameAtom::TABLE,
    NameAtom::TT,
    NameAtom::U,
    NameAtom::UL,
    NameAtom::VAR,
];

/// SVG tag name adjustments, WHATWG 13.2.6.5.
const SVG_TAG_ADJUST: &[(&str, &str)] = &[
    ("altglyph", "altGlyph"),
    ("altglyphdef", "altGlyphDef"),
    ("altglyphitem", "altGlyphItem"),
    ("animatecolor", "animateColor"),
    ("animatemotion", "animateMotion"),
    ("animatetransform", "animateTransform"),
    ("clippath", "clipPath"),
    ("feblend", "feBlend"),
    ("fecolormatrix", "feColorMatrix"),
    ("fecomponenttransfer", "feComponentTransfer"),
    ("fecomposite", "feComposite"),
    ("feconvolvematrix", "feConvolveMatrix"),
    ("fediffuselighting", "feDiffuseLighting"),
    ("fedisplacementmap", "feDisplacementMap"),
    ("fedistantlight", "feDistantLight"),
    ("fedropshadow", "feDropShadow"),
    ("feflood", "feFlood"),
    ("fefunca", "feFuncA"),
    ("fefuncb", "feFuncB"),
    ("fefuncg", "feFuncG"),
    ("fefuncr", "feFuncR"),
    ("fegaussianblur", "feGaussianBlur"),
    ("feimage", "feImage"),
    ("femerge", "feMerge"),
    ("femergenode", "feMergeNode"),
    ("femorphology", "feMorphology"),
    ("feoffset", "feOffset"),
    ("fepointlight", "fePointLight"),
    ("fespecularlighting", "feSpecularLighting"),
    ("fespotlight", "feSpotLight"),
    ("fetile", "feTile"),
    ("feturbulence", "feTurbulence"),
    ("foreignobject", "foreignObject"),
    ("glyphref", "glyphRef"),
    ("lineargradient", "linearGradient"),
    ("radialgradient", "radialGradient"),
    // ("solidcolor", "solidColor") is SVG 1.2; the WHATWG adjustment table
    // has it commented out, so it stays lowercase.
    ("textpath", "textPath"),
];

/// SVG attribute adjustments, WHATWG 13.2.6.1.
const SVG_ATTR_ADJUST: &[(&str, &str)] = &[
    ("attributename", "attributeName"),
    ("attributetype", "attributeType"),
    ("basefrequency", "baseFrequency"),
    ("baseprofile", "baseProfile"),
    ("calcmode", "calcMode"),
    ("clippathunits", "clipPathUnits"),
    ("diffuseconstant", "diffuseConstant"),
    ("edgemode", "edgeMode"),
    ("filterunits", "filterUnits"),
    ("glyphref", "glyphRef"),
    ("gradienttransform", "gradientTransform"),
    ("gradientunits", "gradientUnits"),
    ("kernelmatrix", "kernelMatrix"),
    ("kernelunitlength", "kernelUnitLength"),
    ("keypoints", "keyPoints"),
    ("keysplines", "keySplines"),
    ("keytimes", "keyTimes"),
    ("lengthadjust", "lengthAdjust"),
    ("limitingconeangle", "limitingConeAngle"),
    ("markerheight", "markerHeight"),
    ("markerunits", "markerUnits"),
    ("markerwidth", "markerWidth"),
    ("maskcontentunits", "maskContentUnits"),
    ("maskunits", "maskUnits"),
    ("numoctaves", "numOctaves"),
    ("pathlength", "pathLength"),
    ("patterncontentunits", "patternContentUnits"),
    ("patterntransform", "patternTransform"),
    ("patternunits", "patternUnits"),
    ("pointsatx", "pointsAtX"),
    ("pointsaty", "pointsAtY"),
    ("pointsatz", "pointsAtZ"),
    ("preservealpha", "preserveAlpha"),
    ("preserveaspectratio", "preserveAspectRatio"),
    ("primitiveunits", "primitiveUnits"),
    ("refx", "refX"),
    ("refy", "refY"),
    ("repeatcount", "repeatCount"),
    ("repeatdur", "repeatDur"),
    ("requiredextensions", "requiredExtensions"),
    ("requiredfeatures", "requiredFeatures"),
    ("specularconstant", "specularConstant"),
    ("specularexponent", "specularExponent"),
    ("spreadmethod", "spreadMethod"),
    ("startoffset", "startOffset"),
    ("stddeviation", "stdDeviation"),
    ("stitchtiles", "stitchTiles"),
    ("surfacescale", "surfaceScale"),
    ("systemlanguage", "systemLanguage"),
    ("tablevalues", "tableValues"),
    ("targetx", "targetX"),
    ("targety", "targetY"),
    ("textlength", "textLength"),
    ("viewbox", "viewBox"),
    ("viewtarget", "viewTarget"),
    ("xchannelselector", "xChannelSelector"),
    ("ychannelselector", "yChannelSelector"),
    ("zoomandpan", "zoomAndPan"),
];

/// MathML attribute adjustments, WHATWG 13.2.6.1.
const MATHML_ATTR_ADJUST: &[(&str, &str)] = &[("definitionurl", "definitionURL")];

/// Quirks-mode public identifier prefixes, WHATWG 13.2.6.4.1.
const QUIRKS_PUBLIC_PREFIXES: &[&str] = &[
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

/// ASCII case-insensitive prefix check.
fn starts_with_ci(haystack: &str, prefix: &str) -> bool {
    // `haystack[..prefix.len()]` would panic when `prefix.len()` lands
    // inside a multi-byte char of `haystack` (e.g. a doctype public id
    // with `é` at the cut point); `get` returns `None` (no match) instead.
    haystack
        .get(..prefix.len())
        .is_some_and(|s| s.eq_ignore_ascii_case(prefix))
}

/// Rename attributes via an adjustment table (SVG/MathML case fixes).
fn adjust_attrs(attrs: &mut [(AttrName, String)], table: &[(&str, &str)]) {
    for (name, _) in attrs.iter_mut() {
        if let Some(&(_, fixed)) = table.iter().find(|(from, _)| *from == name.as_str()) {
            *name = AttrName::from(fixed);
        }
    }
}

/// Params for a declarative shadow root parsed off a `template` start tag
/// (WHATWG HTML 4.12.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DsrParams {
    mode: ShadowRootMode,
    delegates_focus: bool,
    clonable: bool,
}

/// Read declarative shadow root attributes off a `template` start-tag
/// token. `shadowrootmode` is an enumerated attribute (ASCII
/// case-insensitive): only `open`/`closed` produce a shadow root; a
/// missing or invalid value means a plain template (WHATWG HTML 4.12.3,
/// "the template element"). `shadowrootdelegatesfocus` and
/// `shadowrootclonable` are boolean attributes (presence = true).
fn declarative_shadow_root(attrs: &[(AttrName, String)]) -> Option<DsrParams> {
    let mode = attrs
        .iter()
        .find(|(k, _)| k == "shadowrootmode")
        .and_then(|(_, v)| {
            if v.eq_ignore_ascii_case("open") {
                Some(ShadowRootMode::Open)
            } else if v.eq_ignore_ascii_case("closed") {
                Some(ShadowRootMode::Closed)
            } else {
                None
            }
        })?;
    Some(DsrParams {
        mode,
        delegates_focus: attrs.iter().any(|(k, _)| k == "shadowrootdelegatesfocus"),
        clonable: attrs.iter().any(|(k, _)| k == "shadowrootclonable"),
    })
}

/// Whether an HTML-namespace element named `tag` may host a declarative
/// shadow root: the DOM "valid shadow host name" list (WHATWG DOM 4.9,
/// `attachShadow`; the parser's 13.2.6.4.7 step 7.11.2 runs "attach a
/// shadow root" on the template's parent and keeps the template an
/// ordinary element when it throws).
///
/// `template` is notably absent from the list, so a `<template
/// shadowrootmode>` nested inside a template — including inside another
/// declarative shadow root under construction — stays a plain template
/// instead of emitting a nested shadow-root bracket for the same host,
/// which would corrupt the sink's bracket pairing.
fn is_valid_shadow_host_name(tag: &str) -> bool {
    matches!(
        tag,
        "article"
            | "aside"
            | "blockquote"
            | "body"
            | "div"
            | "footer"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
            | "header"
            | "main"
            | "nav"
            | "p"
            | "section"
            | "span"
    ) || is_valid_custom_element_name(tag)
}

/// Approximation of the HTML "valid custom element name" concept for
/// shadow-host checking (WHATWG DOM 4.9 lists it as a valid shadow host
/// name). The tokenizer lowercases ASCII alphas in tag names, so require
/// a leading ASCII lowercase alpha, at least one hyphen, and only name
/// characters otherwise. Conservative where the full PCENChar grammar
/// gets exotic (non-ASCII); never wrongly accepts a built-in name.
fn is_valid_custom_element_name(tag: &str) -> bool {
    let mut chars = tag.chars();
    if !matches!(chars.next(), Some(c) if c.is_ascii_lowercase()) {
        return false;
    }
    let mut seen_hyphen = false;
    for c in chars {
        match c {
            '-' => seen_hyphen = true,
            c if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '_' => {}
            _ => return false,
        }
    }
    seen_hyphen
}

// ---------------------------------------------------------------------------
// Tree builder
// ---------------------------------------------------------------------------

struct TreeBuilder<'a, 'x> {
    tok: Tokenizer<'a>,
    arena: Vec<ArenaNode>,
    stack: Vec<El>,
    /// Per-level count of HTML `p` elements in button scope.
    /// `p_in_scope_count[i]` belongs to `stack[i]`: the number of `p`
    /// elements from the innermost button-scope boundary (exclusive)
    /// up to and including `stack[i]`. Maintained by the `*_open`
    /// stack helpers so `has_p_in_button_scope` is O(1).
    p_in_scope_count: Vec<u32>,
    afe: Vec<AfeEntry>,
    mode: Mode,
    original_mode: Mode,
    template_modes: Vec<Mode>,
    head_ptr: Option<El>,
    form_ptr: Option<El>,
    foster: bool,
    frameset_ok: bool,
    quirks: Quirks,
    pending_table_chars: String,
    /// Location of the first char in `pending_table_chars`, for the
    /// whitespace flush in "in table text".
    pending_table_chars_loc: Option<SourceLoc>,
    document: Vec<Child>,
    doctype_appended: bool,
    stop: bool,
    /// Ignore a single leading U+000A after pre/listing/textarea start tags.
    skip_lf: bool,
    /// Source location of the token currently being processed; attached to
    /// nodes created for it.
    current_loc: SourceLoc,
    /// Buffered character run not yet inserted into the tree. Consecutive
    /// `Token::Text` runs accumulate here so a whole text run is
    /// inserted with a single allocation at [`flush_text`](Self::flush_text)
    /// time instead of one `String` per character.
    pending_text: String,
    /// Source location of the first character in
    /// [`pending_text`](Self::pending_text): a merged text node keeps its
    /// first character's location.
    pending_text_loc: SourceLoc,
    /// Value of [`foster`](Self::foster) when the current
    /// [`pending_text`](Self::pending_text) run started. The run is flushed
    /// before the flag can change mid-run, so each flushed run is placed
    /// with the foster-parenting decision that was in effect for it.
    /// (Open-element changes flush via [`push_open`](Self::push_open) /
    /// [`pop_open`](Self::pop_open), so no stack snapshot is needed here.)
    pending_foster: bool,
    /// Fragment parsing (WHATWG 13.4): a synthetic copy of the context
    /// element, backing the "adjusted current node" while the stack holds
    /// only the synthetic root. Never inserted into the tree or exposed.
    /// `None` for full documents.
    fragment_ctx: Option<El>,
    /// Fragment parsing: the synthetic `html` root pushed by
    /// [`init_fragment`](Self::init_fragment) (WHATWG 13.4 step 15). `None`
    /// for full documents.
    fragment_root: Option<El>,
    /// Parse errors recorded during tokenization and tree construction, in
    /// source order (WHATWG 13.2.2).
    errors: Vec<ParseError>,
    /// Whether the HTML parser treats scripting as enabled. Parser-local:
    /// snapshotted from the process default ([`SCRIPTING_DISABLED`]) when
    /// the builder is created, so concurrent parses never cross-contaminate.
    /// When enabled, `<noscript>` parses its content as raw text (head:
    /// "in head noscript" RAWTEXT handling; body: raw-text element) per
    /// WHATWG 13.2.6, instead of as an ordinary element.
    scripting: bool,
    /// Buffered run of consecutive foster-parented insertions before the
    /// same table. WHATWG foster parenting inserts each node immediately
    /// before the table; buffering a run and splicing it in with one
    /// `splice` turns O(nodes) scans+memmoves into one splice per run.
    /// `None` when no foster run is open. See [`FosterBuf`].
    foster_buf: Option<FosterBuf>,
    /// Whether [`document`](Self::document) already holds an element node.
    /// Maintained alongside every document push so the "already has an
    /// element" check in [`insert_at_adjusted`](Self::insert_at_adjusted)
    /// stays O(1) instead of scanning the document vec.
    document_has_element: bool,
    /// The attached script engine, if any. When scripting is enabled, an
    /// eligible parser-inserted classic `<script>` element is handed to it
    /// — element, attributes, and text — as soon as its end tag is
    /// processed (see [`ScriptExecutor`]); `None` preserves the
    /// historical parse where scripts never run.
    executor: Option<&'x mut dyn ScriptExecutor>,
    /// This parse's generation: tags the [`ScriptElement`] handles handed
    /// to the executor so a handle retained across parses fails
    /// validation instead of addressing a stale arena slot. Taken from the
    /// process-wide [`PARSE_GENERATION`] counter.
    generation: u64,
    /// Mutation counter: bumped by [`bump_mutation`](Self::bump_mutation)
    /// on every structural, text, or attribute mutation of the working
    /// tree. Mid-parse geometry snapshots are keyed on this count, so a
    /// cached layout is never reused across a mutation. Over-bumping (a
    /// no-op call site) only costs a cache miss; under-bumping would serve
    /// stale geometry.
    mutation_count: u64,
}

/// One buffered run of foster-parented insertions.
///
/// WHATWG 13.2.6.3 "foster parenting" places each foster-parented node
/// immediately before the table (`ref_child`) in the target child vec. The
/// naive implementation re-scans for the table and `Vec::insert`s per
/// node, which is O(n) per node (O(n^2) for n consecutive foster nodes:
/// the html5lib `foster-*.dat` cases hit this). Instead, consecutive
/// insertions before the same table accumulate in `nodes` and are spliced
/// in with a single [`Vec::splice`] at
/// [`flush_foster_buf`](TreeBuilder::flush_foster_buf) time — O(n) per run.
///
/// The table never moves while a run is open: appends to the target vec
/// cannot move it, and every non-append mutation of the target vec
/// ([`detach`](TreeBuilder::detach), the adoption agency algorithm)
/// flushes the buffer first. `table_idx` is the table's position when the
/// run was seeded (or last re-validated); it is re-validated before each
/// use, and any mismatch fails closed through flush+rescan.
struct FosterBuf {
    /// The table the buffered nodes go before.
    table: El,
    /// The target parent, and whether the template-contents vec (vs. the
    /// regular children vec) is used — mirrors
    /// [`child_vec`](TreeBuilder::child_vec)'s selection exactly.
    target: Parent,
    contents: bool,
    /// `table`'s index in the target child vec.
    table_idx: usize,
    /// The buffered nodes, in document order.
    nodes: Vec<Child>,
}

/// Dispatch outcome for one token in one insertion mode.
enum Flow {
    /// Token consumed; fetch the next token.
    Done,
    /// Reprocess the same token (the insertion mode was switched).
    Again,
}

impl<'a, 'x> TreeBuilder<'a, 'x> {
    fn new(input: &'a str, scripting: bool, executor: Option<&'x mut dyn ScriptExecutor>) -> Self {
        TreeBuilder {
            tok: Tokenizer::new(input),
            arena: Vec::new(),
            stack: Vec::new(),
            p_in_scope_count: Vec::new(),
            afe: Vec::new(),
            mode: Mode::Initial,
            original_mode: Mode::Initial,
            template_modes: Vec::new(),
            head_ptr: None,
            form_ptr: None,
            foster: false,
            frameset_ok: true,
            quirks: Quirks::None,
            pending_table_chars: String::new(),
            pending_table_chars_loc: None,
            document: Vec::new(),
            doctype_appended: false,
            stop: false,
            skip_lf: false,
            current_loc: SourceLoc { line: 1, col: 1 },
            pending_text: String::new(),
            pending_text_loc: SourceLoc { line: 1, col: 1 },
            pending_foster: false,
            fragment_ctx: None,
            fragment_root: None,
            errors: Vec::new(),
            scripting,
            foster_buf: None,
            document_has_element: false,
            executor,
            generation: PARSE_GENERATION.fetch_add(1, Ordering::Relaxed),
            mutation_count: 0,
        }
    }

    /// Record a parse error at the current token's source location.
    fn parse_error(&mut self, code: ParseErrorCode) {
        self.errors.push(ParseError {
            code,
            loc: Some(self.current_loc),
        });
    }

    /// True when parsing a fragment rather than a full document.
    fn is_fragment(&self) -> bool {
        self.fragment_root.is_some()
    }

    /// Set up fragment parsing (WHATWG 13.4 "parsing HTML fragments") for
    /// the given context element. Must be called once, before
    /// [`run`](Self::run).
    ///
    /// The context element's tag name drives the tokenizer's initial state
    /// (step 12) and the initial insertion mode via "reset the insertion
    /// mode appropriately" (step 20); its namespace drives
    /// foreign-content dispatch through the "adjusted current node" (step
    /// 14). The synthetic `html` root (steps 13-15) is never exposed:
    /// insertions targeting it are redirected to the fragment (see
    /// [`appropriate_place`](Self::appropriate_place)).
    ///
    /// The tag name is ASCII-lowercased only for HTML-namespace contexts:
    /// foreign elements keep their case (e.g. SVG `foreignObject`), which
    /// the HTML-integration-point checks depend on. The RCDATA / RAWTEXT /
    /// script-data / PLAINTEXT initial tokenizer states likewise apply
    /// only to HTML-namespace contexts (an `svg title` context parses in
    /// the data state); `noscript` selects RAWTEXT only when scripting is
    /// enabled (step 12).
    ///
    /// Form element pointer (step 21): with an HTML `form` context, the
    /// pointer is initialized to the context copy (a bare [`Element`]
    /// carries no ancestors, so the nearest-`form`-ancestor lookup cannot
    /// see past the context element itself); with a non-`form` context,
    /// an HTML `form` in `ancestors` puts the pointer in scope the same
    /// way.
    fn init_fragment(&mut self, context: &Element, ancestors: &[&Element]) {
        let html_ctx = context.ns == Namespace::Html;
        let tag = if html_ctx {
            context.tag.to_ascii_lowercase()
        } else {
            context.tag.to_string()
        };
        self.tok.set_state(if html_ctx {
            match tag.as_str() {
                "title" | "textarea" => State::Rcdata,
                "style" | "xmp" | "iframe" | "noembed" | "noframes" => State::Rawtext,
                "script" => State::ScriptData,
                "plaintext" => State::Plaintext,
                // Step 12: `noscript` selects RAWTEXT only if scripting is
                // enabled; otherwise the data state.
                "noscript" if self.scripting => State::Rawtext,
                _ => State::Data,
            }
        } else {
            State::Data
        });
        let root = self.create_element(TagName::from("html"), Namespace::Html, Vec::new(), None);
        self.push_open(root);
        self.fragment_root = Some(root);
        // A synthetic copy of the context element, backing the WHATWG
        // "adjusted current node" while the stack holds only the root. It
        // is never inserted into the tree or exposed.
        let ctx = self.create_element(
            TagName::from(tag.as_str()),
            context.ns.clone(),
            context.attrs.clone(),
            None,
        );
        self.fragment_ctx = Some(ctx);
        if html_ctx && tag == "form" {
            // Step 21: the form element pointer starts at the context
            // element, so `<form><input>` in a form context ignores the
            // nested form start tag.
            self.form_ptr = Some(ctx);
        } else if html_ctx
            && ancestors
                .iter()
                .any(|a| a.ns == Namespace::Html && a.tag == "form")
        {
            // Step 21, nearest-form-ancestor lookup: the context is not a
            // form, but an HTML `form` ancestor is in scope above the
            // fragment, so a `<form>` start tag is ignored just as if the
            // pointer named that ancestor. The synthetic context copy
            // stands in for it: like a real ancestor above the fragment
            // root it is never on the open-element stack, so end-tag
            // scope checks behave identically.
            self.form_ptr = Some(ctx);
        }
        if html_ctx && tag == "template" {
            // Step 18: a template context parses its template contents.
            self.template_modes.push(Mode::InTemplate);
        }
        self.reset_insertion_mode();
    }

    /// The WHATWG "adjusted current node" (13.2.4.2): the fragment context
    /// element when the stack holds only the synthetic root, otherwise the
    /// current node.
    fn adjusted_current_node(&mut self) -> El {
        if let (Some(root), Some(ctx)) = (self.fragment_root, self.fragment_ctx)
            && self.stack.len() == 1
            && self.stack[0] == root
        {
            return ctx;
        }
        self.current_node()
    }

    /// The adjusted current node's tag name.
    fn adjusted_current_tag(&self) -> Option<NameAtom> {
        if let (Some(root), Some(ctx)) = (self.fragment_root, self.fragment_ctx)
            && self.stack.len() == 1
            && self.stack[0] == root
        {
            return Some(self.arena[ctx].tag.atom());
        }
        self.current_tag()
    }

    // -- arena / stack basics -------------------------------------------

    fn create_element(
        &mut self,
        tag: TagName,
        ns: Namespace,
        attrs: Vec<(AttrName, String)>,
        loc: Option<SourceLoc>,
    ) -> El {
        let annotation_xml_html = ns == Namespace::MathMl
            && tag.atom() == NameAtom::ANNOTATION_XML
            && attrs.iter().any(|(k, v)| {
                k.atom() == NameAtom::ENCODING
                    && (v.eq_ignore_ascii_case("text/html")
                        || v.eq_ignore_ascii_case("application/xhtml+xml"))
            });
        let template_contents = if ns == Namespace::Html && tag == "template" {
            Some(Vec::new())
        } else {
            None
        };
        let el = self.arena.len();
        self.arena.push(ArenaNode {
            tag,
            ns,
            attrs,
            children: Vec::new(),
            template_contents,
            dsr: None,
            parent: None,
            loc,
            annotation_xml_html,
        });
        el
    }

    /// The current node (top of the stack of open elements).
    ///
    /// Fails closed instead of panicking: the stack bottom (`html`, or the
    /// synthetic fragment root) is pinned by [`pop_open`](Self::pop_open) /
    /// [`remove_open`](Self::remove_open), so an empty stack is unreachable
    /// in a correct run. If it ever happens anyway, record a parse error
    /// and re-seed the stack with a synthetic `html` element so parsing
    /// continues on a best-effort tree.
    fn current_node(&mut self) -> El {
        if let Some(&el) = self.stack.last() {
            return el;
        }
        self.parse_error(ParseErrorCode::TreeInvariant);
        let el = self.create_element(TagName::from("html"), Namespace::Html, Vec::new(), None);
        self.push_open(el);
        el
    }

    fn current_tag(&self) -> Option<NameAtom> {
        self.stack.last().map(|&el| self.arena[el].tag.atom())
    }

    /// Whether the current node has the given namespace and tag. Useful in
    /// match guards, where the node cannot be bound first.
    fn current_node_is(&mut self, ns: Namespace, tag: &str) -> bool {
        let cur = self.current_node();
        let n = &self.arena[cur];
        n.ns == ns && n.tag == tag
    }

    fn tag_of(&self, el: El) -> NameAtom {
        self.arena[el].tag.atom()
    }

    fn is_special(&self, el: El) -> bool {
        let n = &self.arena[el];
        match n.ns {
            Namespace::Html => SPECIAL_HTML.contains(&n.tag.atom()),
            Namespace::MathMl => matches!(
                n.tag.atom(),
                NameAtom::MI
                    | NameAtom::MO
                    | NameAtom::MN
                    | NameAtom::MS
                    | NameAtom::MTEXT
                    | NameAtom::ANNOTATION_XML
            ),
            Namespace::Svg => matches!(
                n.tag.atom(),
                NameAtom::FOREIGNOBJECT | NameAtom::DESC | NameAtom::TITLE
            ),
            // Custom namespaces never occur in parsed input (the parser
            // only produces the three well-known ones); not special.
            Namespace::Other(_) => false,
        }
    }

    /// Whether an element terminates a default-scope walk (and therefore
    /// button-scope and list-item-scope walks): the MathML text-integration
    /// points, every MathML `annotation-xml` element (whether or not it is
    /// an HTML integration point), and the SVG elements that terminate
    /// scope, WHATWG 13.2.4.2.
    /// Foreign elements that terminate the default/button/list-item scope
    /// walks (WHATWG 13.2.4.2). Table scope (html/table/template only) does
    /// not include them, so the walk passes straight through foreign
    /// elements there.
    fn is_foreign_scope_boundary(n: &ArenaNode) -> bool {
        match n.ns {
            Namespace::MathMl => {
                matches!(
                    n.tag.atom(),
                    NameAtom::MI
                        | NameAtom::MO
                        | NameAtom::MN
                        | NameAtom::MS
                        | NameAtom::MTEXT
                        | NameAtom::ANNOTATION_XML
                )
            }
            Namespace::Svg => matches!(
                n.tag.atom(),
                NameAtom::FOREIGNOBJECT | NameAtom::DESC | NameAtom::TITLE
            ),
            // Only MathML/SVG are "foreign elements" per WHATWG 13.2.4.2;
            // a custom namespace is not scope-terminating.
            Namespace::Other(_) => false,
            Namespace::Html => false,
        }
    }

    /// "have an element *tag* in *scope*" (WHATWG 13.2.4.2): the walk only
    /// stops at the target tag or a scope-boundary element. Foreign
    /// integration points terminate the default/button/list-item scopes but
    /// not table scope.
    fn has_in_scope(&self, tag: &TagName, scope: &[NameAtom]) -> bool {
        let foreign_terminates = !std::ptr::eq(scope, SCOPE_TABLE);
        for &el in self.stack.iter().rev() {
            let n = &self.arena[el];
            if n.ns == Namespace::Html && n.tag == *tag {
                return true;
            }
            if n.ns == Namespace::Html && scope.contains(&n.tag.atom()) {
                return false;
            }
            if foreign_terminates && Self::is_foreign_scope_boundary(n) {
                return false;
            }
        }
        false
    }

    /// "has *node* in *scope*": walk from the top of the stack until the
    /// node itself or a scope boundary is found. Foreign integration points
    /// terminate the default/button/list-item scopes but not table scope.
    fn node_in_scope(&self, target: El, scope: &[NameAtom]) -> bool {
        let foreign_terminates = !std::ptr::eq(scope, SCOPE_TABLE);
        for &el in self.stack.iter().rev() {
            if el == target {
                return true;
            }
            let n = &self.arena[el];
            if n.ns == Namespace::Html && scope.contains(&n.tag.atom()) {
                return false;
            }
            if foreign_terminates && Self::is_foreign_scope_boundary(n) {
                return false;
            }
        }
        false
    }

    /// "has a `p` element in button scope" (WHATWG 13.2.4.2): O(1) via the
    /// per-level count kept in `p_in_scope_count`. A full stack walk here is
    /// O(depth) per token, which is quadratic on deeply nested input with no
    /// open `p` (the walk only stops at a `p` or a scope boundary).
    fn has_p_in_button_scope(&self) -> bool {
        self.p_in_scope_count.last().is_some_and(|&c| c > 0)
    }

    /// Push an element on the stack of open elements, maintaining the
    /// per-level `p`-in-button-scope count. Flushes any buffered character
    /// run first: the run belongs to the pre-push tree.
    fn push_open(&mut self, el: El) {
        self.flush_text();
        self.p_in_scope_count.push(Self::p_count_for(
            &self.arena[el],
            self.p_in_scope_count.last().copied().unwrap_or(0),
        ));
        self.stack.push(el);
    }

    /// Pop the stack of open elements, keeping the per-level count in
    /// lockstep.
    ///
    /// The stack bottom (the `html` element, or the synthetic fragment
    /// root) is pinned: it is never popped, matching WHATWG where the
    /// stack always bottoms out at `html`. Returns `None` once only the
    /// bottom remains, so `pop_until_*` / `clear_back_to` loops terminate
    /// there instead of emptying the stack (which previously panicked in
    /// `current_node` on adversarial input).
    fn pop_open(&mut self) -> Option<El> {
        // Flush first: a buffered run belongs to the pre-pop tree (e.g. a
        // whitespace run buffered in `head` before a character pops it).
        self.flush_text();
        if self.stack.len() <= 1 {
            return None;
        }
        let popped = self.stack.pop();
        if popped.is_some() {
            self.p_in_scope_count.pop();
        }
        debug_assert_eq!(self.stack.len(), self.p_in_scope_count.len());
        popped
    }

    /// Remove the stack entry at `pos` (adoption agency only), then repair
    /// the per-level counts above it. Position 0 (the pinned stack bottom)
    /// is a defensive no-op instead of a panic.
    fn remove_open(&mut self, pos: usize) {
        self.flush_text();
        if pos == 0 {
            // Fail-closed no-op in all builds (never panic).
            return;
        }
        self.stack.remove(pos);
        self.p_in_scope_count.remove(pos);
        self.recompute_pcount_from(pos);
    }

    /// Insert `el` at stack index `idx` (adoption agency only), then repair
    /// the per-level counts above it.
    fn insert_open(&mut self, idx: usize, el: El) {
        self.flush_text();
        let parent = if idx == 0 {
            0
        } else {
            self.p_in_scope_count[idx - 1]
        };
        let count = Self::p_count_for(&self.arena[el], parent);
        self.stack.insert(idx, el);
        self.p_in_scope_count.insert(idx, count);
        self.recompute_pcount_from(idx + 1);
    }

    /// Per-level `p`-in-button-scope count for an element pushed onto a stack
    /// whose current top has count `parent`. Button scope includes the
    /// foreign integration points (WHATWG 13.2.4.2), so they reset the count.
    fn p_count_for(n: &ArenaNode, parent: u32) -> u32 {
        if n.ns == Namespace::Html && n.tag.atom() == NameAtom::P {
            parent + 1
        } else if (n.ns == Namespace::Html
            && (SCOPE_DEFAULT.contains(&n.tag.atom()) || n.tag.atom() == NameAtom::BUTTON))
            || Self::is_foreign_scope_boundary(n)
        {
            0
        } else {
            parent
        }
    }

    /// Recompute per-level `p` counts from stack index `from` upward. Only
    /// used after mid-stack surgery (`remove_open`/`insert_open`); the common
    /// push/pop paths maintain counts incrementally in O(1).
    fn recompute_pcount_from(&mut self, from: usize) {
        let mut count = if from == 0 {
            0
        } else {
            self.p_in_scope_count[from - 1]
        };
        for i in from..self.stack.len() {
            count = Self::p_count_for(&self.arena[self.stack[i]], count);
            self.p_in_scope_count[i] = count;
        }
        debug_assert_eq!(self.stack.len(), self.p_in_scope_count.len());
    }

    fn has_in_list_item_scope(&self, tag: &TagName) -> bool {
        for &el in self.stack.iter().rev() {
            let n = &self.arena[el];
            if n.ns == Namespace::Html && n.tag == *tag {
                return true;
            }
            if n.ns == Namespace::Html
                && (SCOPE_DEFAULT.contains(&n.tag.atom())
                    || n.tag.atom() == NameAtom::OL
                    || n.tag.atom() == NameAtom::UL)
            {
                return false;
            }
            if Self::is_foreign_scope_boundary(n) {
                return false;
            }
        }
        false
    }

    /// Generate implied end tags, optionally excepting one tag (or the
    /// thorough set).
    fn generate_implied(&mut self, except: Option<NameAtom>, thorough: bool) {
        loop {
            let cur = self.current_node();
            let tag = self.tag_of(cur);
            let in_list =
                IMPLIED.contains(&tag) || (thorough && IMPLIED_THOROUGH_EXTRA.contains(&tag));
            if !in_list || except == Some(tag) {
                break;
            }
            self.pop_open();
        }
    }

    fn pop_until_tag(&mut self, tag: &TagName) {
        while let Some(el) = self.pop_open() {
            if self.arena[el].ns == Namespace::Html && self.arena[el].tag == *tag {
                break;
            }
        }
    }

    fn pop_until_one_of(&mut self, tags: &[NameAtom]) {
        while let Some(el) = self.pop_open() {
            if self.arena[el].ns == Namespace::Html && tags.contains(&self.arena[el].tag.atom()) {
                break;
            }
        }
    }

    /// Clear the stack back to a table-ish context.
    fn clear_back_to(&mut self, tags: &[NameAtom]) {
        loop {
            let cur = self.current_node();
            if self.arena[cur].ns == Namespace::Html && tags.contains(&self.tag_of(cur)) {
                break;
            }
            // The stack bottom is pinned: stop there instead of spinning
            // forever (pop_open returns None once only it remains).
            if self.pop_open().is_none() {
                break;
            }
        }
    }

    fn close_p(&mut self) {
        self.generate_implied(Some(NameAtom::P), false);
        self.pop_until_tag(&TagName::from("p"));
    }

    // -- insertion -------------------------------------------------------

    /// Compute the adjusted insertion location, WHATWG 13.2.6.1.
    fn appropriate_place(&mut self, override_target: Option<Parent>) -> (Parent, bool, Option<El>) {
        let mut target = override_target.unwrap_or_else(|| Parent::Elem(self.current_node()));
        // Fragment case (WHATWG 13.4 steps 15-17: the synthetic root is the
        // fragment's stand-in via the root insertion target): anything that
        // would be inserted into the root lands at the top level of the
        // fragment instead, i.e. in `self.document`.
        if let Some(root) = self.fragment_root
            && target == Parent::Elem(root)
        {
            return (Parent::Doc, false, None);
        }
        let mut ref_child: Option<El> = None;
        if self.foster
            && let Parent::Elem(t) = target
            && matches!(
                self.arena[t].tag.atom(),
                NameAtom::TABLE
                    | NameAtom::TBODY
                    | NameAtom::TFOOT
                    | NameAtom::THEAD
                    | NameAtom::TR
            )
        {
            let ltt = self.stack.iter().rev().find(|&&e| {
                matches!(
                    self.arena[e].tag.atom(),
                    NameAtom::TEMPLATE | NameAtom::TABLE
                )
            });
            match ltt {
                None => return (Parent::Doc, false, None),
                Some(&t2) => {
                    if self.arena[t2].tag.atom() == NameAtom::TEMPLATE {
                        target = Parent::Elem(t2);
                    } else if let Some(p) = self.arena[t2].parent {
                        target = p;
                        ref_child = Some(t2);
                    } else {
                        let Some(pos) = self.stack.iter().position(|&e| e == t2) else {
                            // Defensive: the table left the stack through an
                            // untracked path; insert at the document level
                            // instead of panicking.
                            self.parse_error(ParseErrorCode::TreeInvariant);
                            return (Parent::Doc, false, None);
                        };
                        if pos == 0 {
                            // Defensive: the table is never the root `html`
                            // element; fail closed instead of underflowing.
                            self.parse_error(ParseErrorCode::TreeInvariant);
                            return (Parent::Doc, false, None);
                        }
                        target = Parent::Elem(self.stack[pos - 1]);
                    }
                }
            }
        }
        if let Parent::Elem(t) = target
            && self.arena[t].tag.atom() == NameAtom::TEMPLATE
            && self.arena[t].ns == Namespace::Html
        {
            // Template contents go in the content fragment. The reference
            // child is kept: foster parenting before a table that lives in
            // template contents must insert before that table.
            return (Parent::Elem(t), true, ref_child);
        }
        (target, false, ref_child)
    }

    fn child_vec(&mut self, p: El, contents: bool) -> &mut Vec<Child> {
        let node = &mut self.arena[p];
        // Defensive: `contents` is only ever true for template targets
        // (see `appropriate_place`), which always have a contents
        // fragment. Fail closed by falling back to regular children
        // instead of panicking.
        if contents && let Some(tc) = node.template_contents.as_mut() {
            return tc;
        }
        &mut node.children
    }

    /// The child vec a foster run targets, mirroring
    /// [`child_vec`](Self::child_vec)'s template-contents selection exactly.
    fn foster_vec(&mut self, target: Parent, contents: bool) -> &mut Vec<Child> {
        match target {
            Parent::Doc => &mut self.document,
            Parent::Elem(p) => self.child_vec(p, contents),
        }
    }

    /// `table`'s position in the target child vec, scanning from the end
    /// (foster tables sit near the end). `None` when absent — unreachable
    /// via [`appropriate_place`](Self::appropriate_place), which only
    /// yields tables that have a parent; callers fail closed.
    fn foster_table_pos(&self, target: Parent, contents: bool, table: El) -> Option<usize> {
        let v = match target {
            Parent::Doc => &self.document,
            Parent::Elem(p) => {
                let node = &self.arena[p];
                if contents && let Some(tc) = node.template_contents.as_ref() {
                    tc
                } else {
                    &node.children
                }
            }
        };
        v.iter()
            .rposition(|c| matches!(c, Child::Elem(e) if *e == table))
    }

    /// Make sure a foster run is open for `(table, target, contents)` and
    /// return the table's validated index in the target vec. A run for a
    /// different key is flushed first; a stale cached index fails closed
    /// through flush+rescan. `None` only when the table cannot be found.
    fn foster_ensure(&mut self, table: El, target: Parent, contents: bool) -> Option<usize> {
        let key_matches = matches!(&self.foster_buf, Some(b)
            if b.table == table && b.target == target && b.contents == contents);
        if key_matches {
            let idx = self.foster_buf.as_ref().unwrap().table_idx;
            // Re-validate: appends to the target vec cannot move the table,
            // and every conflicting mutation flushes first, so a mismatch
            // is a bug — fail closed through flush+rescan.
            let valid = match target {
                Parent::Doc => self.document.get(idx) == Some(&Child::Elem(table)),
                Parent::Elem(p) => {
                    let node = &self.arena[p];
                    let v = if contents && let Some(tc) = node.template_contents.as_ref() {
                        tc
                    } else {
                        &node.children
                    };
                    v.get(idx) == Some(&Child::Elem(table))
                }
            };
            if valid {
                return Some(idx);
            }
        }
        self.flush_foster_buf();
        let idx = self.foster_table_pos(target, contents, table)?;
        self.foster_buf = Some(FosterBuf {
            table,
            target,
            contents,
            table_idx: idx,
            nodes: Vec::new(),
        });
        Some(idx)
    }

    /// Splice an open foster run into its target vec immediately before its
    /// table. Runs are consecutive, so one splice per run replaces one O(n)
    /// scan+memmove per node: O(n) per run instead of O(n^2).
    fn flush_foster_buf(&mut self) {
        let Some(buf) = self.foster_buf.take() else {
            return;
        };
        if buf.nodes.is_empty() {
            return;
        }
        let v = self.foster_vec(buf.target, buf.contents);
        let has_element = buf.nodes.iter().any(|c| matches!(c, Child::Elem(_)));
        match v
            .iter()
            .rposition(|c| matches!(c, Child::Elem(e) if *e == buf.table))
        {
            Some(i) => {
                v.splice(i..i, buf.nodes);
            }
            // Fail-closed: the table vanished through an untracked path;
            // keep the nodes in the tree rather than dropping them.
            None => v.extend(buf.nodes),
        }
        if has_element && matches!(buf.target, Parent::Doc) {
            self.document_has_element = true;
        }
        self.bump_mutation();
    }

    /// Buffer one foster-parented element insertion (the
    /// [`insert_at_adjusted`](Self::insert_at_adjusted) `ref_child` path).
    /// The node's parent pointer is set by the caller; its position before
    /// the table materializes at flush time.
    fn foster_insert(&mut self, target: Parent, contents: bool, table: El, el: El) {
        if self.foster_ensure(table, target, contents).is_none() {
            // Fail-closed: table not found; append like the old `None`
            // branch did.
            self.foster_vec(target, contents).push(Child::Elem(el));
        } else {
            self.foster_buf
                .as_mut()
                .unwrap()
                .nodes
                .push(Child::Elem(el));
        }
    }

    /// Buffer one foster-parented character run (the
    /// [`insert_str_at`](Self::insert_str_at) `ref_child` path), merging
    /// into an adjacent text node exactly like the old immediate-insert
    /// path did: the buffered run's tail text node first, then the vec's
    /// text node just before the table (only when the buffer is otherwise
    /// empty, so nothing is skipped), else a fresh buffered text node.
    fn foster_text(
        &mut self,
        target: Parent,
        contents: bool,
        table: El,
        s: &str,
        loc: Option<SourceLoc>,
    ) {
        // Fast path: the buffered run's tail is a text node — merge there.
        if let Some(buf) = self.foster_buf.as_mut()
            && buf.table == table
            && buf.target == target
            && buf.contents == contents
            && let Some(Child::Text(t, _)) = buf.nodes.last_mut()
        {
            t.push_str(s);
            return;
        }
        let Some(idx) = self.foster_ensure(table, target, contents) else {
            // Fail-closed: table not found; append.
            self.foster_vec(target, contents)
                .push(Child::Text(s.to_string(), loc));
            return;
        };
        // With an empty buffer the vec is exactly where the old path left
        // it, so merge into the text node before the table when there is
        // one. A non-empty buffer means the logical predecessor is the
        // buffered tail (handled above or pushed below).
        if self.foster_buf.as_ref().is_some_and(|b| b.nodes.is_empty()) {
            let v = self.foster_vec(target, contents);
            if idx > 0
                && let Child::Text(t, _) = &mut v[idx - 1]
            {
                t.push_str(s);
                return;
            }
        }
        self.foster_buf
            .as_mut()
            .unwrap()
            .nodes
            .push(Child::Text(s.to_string(), loc));
    }

    /// Insert `el` at the adjusted insertion location (with the spec's
    /// no-reparent / no-cycle / document guards).
    fn insert_at_adjusted(&mut self, el: El) {
        if self.arena[el].parent.is_some() {
            return;
        }
        let (target, contents, ref_child) = self.appropriate_place(None);
        // Cycle guard: never insert an ancestor of the target.
        //
        // The walk to the root is O(depth) per insertion, which is
        // quadratic on deeply nested input. It can be skipped entirely
        // when `el` cannot be an ancestor of anything: `el` is detached
        // (checked above, so it is not in the tree and `t != el`), and
        // with no children and no template contents it has no
        // descendants either. The full walk is kept for the (currently
        // unreachable) case of inserting a detached subtree.
        let el_has_descendants = !self.arena[el].children.is_empty()
            || self.arena[el]
                .template_contents
                .as_ref()
                .is_some_and(|v| !v.is_empty());
        if el_has_descendants && let Parent::Elem(t) = target {
            let mut cur = Some(t);
            while let Some(c) = cur {
                if c == el {
                    return;
                }
                cur = match self.arena[c].parent {
                    Some(Parent::Elem(p)) => Some(p),
                    _ => None,
                };
            }
        }
        match target {
            Parent::Doc => {
                // In fragment mode the synthetic root is never exposed: the
                // root insertion target is the fragment itself, and a
                // foster-parented node inserts before the table that
                // triggered the foster parenting (`ref_child`).
                if !self.is_fragment() && self.document_has_element {
                    return;
                }
                match ref_child {
                    // Foster-parented: buffer the node before the table;
                    // one splice per consecutive run instead of one O(n)
                    // scan+memmove per node.
                    Some(t) => self.foster_insert(target, contents, t, el),
                    None => {
                        self.document.push(Child::Elem(el));
                        self.document_has_element = true;
                    }
                }
                self.arena[el].parent = Some(Parent::Doc);
            }
            Parent::Elem(p) => {
                match ref_child {
                    Some(t) => self.foster_insert(target, contents, t, el),
                    None => self.child_vec(p, contents).push(Child::Elem(el)),
                }
                self.arena[el].parent = Some(Parent::Elem(p));
            }
        }
        self.bump_mutation();
    }

    /// Insert an HTML element for `tag`: create, insert at the adjusted
    /// insertion location, push on the stack of open elements.
    fn insert_html_element(&mut self, tag: &mut Tag) -> El {
        // Move the name/attributes out of the token: element creation is
        // the terminal consumer of a start tag, so no clone is needed.
        let name = std::mem::take(&mut tag.name);
        let attrs = std::mem::take(&mut tag.attrs);
        let el = self.create_element(name, Namespace::Html, attrs, Some(self.current_loc));
        self.insert_at_adjusted(el);
        self.push_open(el);
        el
    }

    /// Insert an HTML element and immediately pop it (void elements).
    fn insert_void_element(&mut self, tag: &mut Tag) {
        self.insert_html_element(tag);
        self.pop_open();
    }

    /// Insert a foreign (MathML/SVG) element for `tag`.
    fn insert_foreign_element(&mut self, tag: &mut Tag, ns: Namespace) -> El {
        // Move the name/attributes out of the token (see
        // `insert_html_element`).
        let name = std::mem::take(&mut tag.name);
        let attrs = std::mem::take(&mut tag.attrs);
        let el = self.create_element(name, ns, attrs, Some(self.current_loc));
        self.insert_at_adjusted(el);
        self.push_open(el);
        el
    }

    /// Insert a character, merging with an adjacent text node. A merged
    /// text node keeps the location of its first character.
    fn insert_str(&mut self, s: &str) {
        let loc = self.current_loc;
        self.insert_str_at(s, loc);
    }

    /// Buffer one character of a text run. The run is inserted as a whole
    /// by [`flush_text`](Self::flush_text) once a non-character token
    /// arrives, so text-heavy input costs one allocation per run instead of
    /// one per character. If foster parenting changed since the run started
    /// (table text re-emitted via `foster_in_body`), the run is flushed
    /// first so each run keeps the placement it was emitted with. (Stack
    /// changes flush via [`push_open`](Self::push_open) /
    /// [`pop_open`](Self::pop_open).)
    /// Begin a new buffered run when the buffer is empty or the foster
    /// flag changed since the run started; flush the old run first in the
    /// latter case so each run keeps the placement it was emitted with.
    /// (Stack changes flush via [`push_open`](Self::push_open) /
    /// [`pop_open`](Self::pop_open).)
    fn start_text_run_if_needed(&mut self) {
        if self.pending_text.is_empty() {
            self.start_text_run();
        } else if self.pending_foster != self.foster {
            self.flush_text();
            self.start_text_run();
        }
    }

    /// Buffer a text-run slice, merging with the pending run. Copies;
    /// prefer [`push_text_string`](Self::push_text_string) when an owned
    /// run is available.
    fn push_text_str(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        self.start_text_run_if_needed();
        self.pending_text.push_str(s);
    }

    /// Buffer a text run, moving the string when the pending buffer is
    /// empty (zero-copy for the common single-run case) and appending
    /// otherwise.
    fn push_text_string(&mut self, s: String) {
        if s.is_empty() {
            return;
        }
        if self.pending_text.is_empty() {
            self.start_text_run();
            self.pending_text = s;
        } else {
            self.start_text_run_if_needed();
            self.pending_text.push_str(&s);
        }
    }

    /// Byte index of the first non-whitespace byte in `s`, or `None` when
    /// `s` is all whitespace. (The five whitespace characters are ASCII,
    /// so byte and character scans agree.)
    fn first_non_ws(s: &str) -> Option<usize> {
        s.bytes()
            .position(|b| !matches!(b, b'\t' | b'\n' | b'\x0C' | b'\r' | b' '))
    }

    /// Advance [`current_loc`](Self::current_loc) past `prefix`, which
    /// must be ASCII whitespace: newlines start a new line, other bytes
    /// advance the column. Used after stripping a whitespace prefix from
    /// a text run so the remainder keeps its source location.
    fn advance_loc_past(&mut self, prefix: &str) {
        for &b in prefix.as_bytes() {
            if b == b'\n' {
                self.current_loc.line += 1;
                self.current_loc.col = 1;
            } else {
                self.current_loc.col += 1;
            }
        }
    }

    /// Strip the leading ASCII-whitespace prefix from a text run for
    /// whitespace-filtering insertion modes that *ignore* whitespace,
    /// advancing [`current_loc`](Self::current_loc) past the stripped
    /// characters. Returns `true` when non-whitespace remains in `s` (the
    /// mode's anything-else path applies); `false` for an all-whitespace
    /// run, which is left intact.
    fn strip_ws_prefix(&mut self, s: &mut String) -> bool {
        let Some(i) = Self::first_non_ws(s) else {
            return false;
        };
        if i > 0 {
            self.advance_loc_past(&s[..i]);
            s.drain(..i);
        }
        true
    }

    /// Buffer the leading whitespace prefix of a text run for
    /// whitespace-filtering insertion modes that *insert* whitespace,
    /// leaving the non-whitespace remainder in `s` (with
    /// [`current_loc`](Self::current_loc) advanced past the prefix).
    /// Returns `true` when a non-whitespace remainder needs the mode's
    /// anything-else path; `false` when the whole run was whitespace.
    fn split_ws_prefix_insert(&mut self, s: &mut String) -> bool {
        let Some(i) = Self::first_non_ws(s) else {
            self.push_text_str(s);
            return false;
        };
        if i > 0 {
            self.push_text_str(&s[..i]);
            self.advance_loc_past(&s[..i]);
            s.drain(..i);
        }
        true
    }

    /// Insert the whitespace characters of `s`, ignoring the rest.
    /// "in frameset"/"after frameset" insert whitespace and ignore other
    /// characters (no parse error), so a mixed run contributes each of its
    /// whitespace characters in order.
    fn insert_ws_only(&mut self, s: &mut String) {
        if Self::first_non_ws(s).is_none() {
            // All whitespace (or empty): move the run as-is.
            let run = std::mem::take(s);
            if !run.is_empty() {
                self.push_text_string(run);
            }
        } else {
            let ws: String = s
                .chars()
                .filter(|c| matches!(c, '\u{9}' | '\u{A}' | '\u{C}' | '\u{D}' | '\u{20}'))
                .collect();
            let _ = std::mem::take(s);
            if !ws.is_empty() {
                self.push_text_string(ws);
            }
        }
    }

    /// Split a mixed text run for whitespace-filtering insertion modes
    /// that *delegate* whitespace to another mode (`delegate`, e.g.
    /// [`mode_in_body`](Self::mode_in_body)): process the leading
    /// whitespace prefix with `delegate`, leaving the non-whitespace
    /// remainder in `tok` with [`current_loc`](Self::current_loc)
    /// advanced past the prefix. Returns `true` when non-whitespace
    /// remains for the caller's anything-else path; `false` when the
    /// token was fully delegated (all whitespace).
    fn delegate_ws_prefix(
        &mut self,
        tok: &mut Token,
        delegate: fn(&mut Self, &mut Token) -> Flow,
    ) -> bool {
        let s = match tok {
            Token::Text(s) => s,
            _ => return false,
        };
        let Some(i) = Self::first_non_ws(s) else {
            let flow = delegate(self, tok);
            debug_assert!(matches!(flow, Flow::Done));
            return false;
        };
        if i == 0 {
            return true;
        }
        let mut run = std::mem::take(s);
        let rest = run.split_off(i);
        let prefix_loc = self.current_loc;
        self.advance_loc_past(&run);
        let remainder_loc = self.current_loc;
        self.current_loc = prefix_loc;
        let mut prefix_tok = Token::Text(run);
        let flow = delegate(self, &mut prefix_tok);
        debug_assert!(matches!(flow, Flow::Done));
        self.current_loc = remainder_loc;
        *s = rest;
        true
    }

    /// Snapshot the insertion context for a new buffered text run.
    fn start_text_run(&mut self) {
        self.pending_text_loc = self.current_loc;
        self.pending_foster = self.foster;
    }

    /// True if [`reconstruct`](Self::reconstruct) would insert elements.
    /// Mirrors its early-return conditions exactly, so calling it and then
    /// skipping [`reconstruct`](Self::reconstruct) on `false` is safe.
    /// Used to flush a buffered text run *before* reconstruction changes
    /// the insertion point.
    fn reconstruct_pending(&self) -> bool {
        let Some(last) = self.afe.last() else {
            return false;
        };
        if last.is_marker() {
            return false;
        }
        // A non-marker entry always carries an element; `None` is treated
        // as "would insert" to stay conservative.
        match last.element() {
            Some(el) => !self.stack.contains(&el),
            None => true,
        }
    }

    /// Insert the buffered character run, if any, as a single text update.
    /// The run is inserted with the foster-parenting flag that was in
    /// effect when its first character was buffered (foster-parented table
    /// text is re-emitted with the flag set transiently). The buffer's
    /// capacity is retained for the next run.
    fn flush_text(&mut self) {
        if self.pending_text.is_empty() {
            return;
        }
        let loc = self.pending_text_loc;
        let foster = self.pending_foster;
        let mut run = std::mem::take(&mut self.pending_text);
        let saved_foster = std::mem::replace(&mut self.foster, foster);
        self.insert_str_at(&run, loc);
        self.foster = saved_foster;
        run.clear();
        self.pending_text = run;
    }

    /// Flush the buffered "in table text" character run, if any: a run
    /// containing non-whitespace is foster-parented through the in-table
    /// "anything else" path, an all-whitespace run is inserted as one text
    /// node. The mode switch back to `original_mode` stays with the
    /// caller ([`mode_in_table_text`](Self::mode_in_table_text)); callers
    /// that must not disturb the mode ([`snapshot_into`](Self::snapshot_into))
    /// leave it alone. A no-op when no run is buffered.
    fn flush_pending_table_chars(&mut self) {
        if self.pending_table_chars.is_empty() {
            return;
        }
        let run = std::mem::take(&mut self.pending_table_chars);
        let first_loc = self.pending_table_chars_loc.take();
        if Self::first_non_ws(&run).is_some() {
            // Foster-parent the run via in-table "anything else"
            // (batched: one text token and one parse error for the
            // run instead of one per character; the tree is
            // identical).
            let mut t = Token::Text(run);
            // The foster path re-buffers through `pending_text`, which
            // would otherwise stamp the run with the flushing token's
            // location. Keep the run's first character location instead
            // (like the whitespace branch below). Sound only when the
            // run starts a fresh buffer: the deferred path always does
            // (`handle_token` flushes before dispatch); a mid-parse
            // snapshot with a live foster run keeps the old location.
            let fresh = self.pending_text.is_empty();
            self.foster_in_body(&mut t);
            if fresh && let Some(loc) = first_loc {
                self.pending_text_loc = loc;
            }
        } else {
            // Whitespace is inserted as one run located at its first
            // character (not at the token that ended the run).
            match first_loc {
                Some(loc) => self.insert_str_at(&run, loc),
                None => self.insert_str(&run),
            }
        }
    }

    /// Insert a character run with an explicit source location (used when
    /// the run's characters were consumed before the current token, e.g.
    /// the "in table text" whitespace flush).
    fn insert_str_at(&mut self, s: &str, loc: SourceLoc) {
        if s.is_empty() {
            return;
        }
        let loc = Some(loc);
        let (target, contents, ref_child) = self.appropriate_place(None);
        match target {
            Parent::Doc => {
                // Foster-parented text in fragment mode inserts before the
                // table that triggered it (`ref_child`); in document mode
                // `ref_child` is always `None` here.
                match ref_child {
                    Some(t) => self.foster_text(target, contents, t, s, loc),
                    None => match self.document.last_mut() {
                        Some(Child::Text(t, _)) => t.push_str(s),
                        _ => self.document.push(Child::Text(s.to_string(), loc)),
                    },
                }
            }
            Parent::Elem(p) => match ref_child {
                Some(t) => self.foster_text(target, contents, t, s, loc),
                None => {
                    let v = self.child_vec(p, contents);
                    match v.last_mut() {
                        Some(Child::Text(t, _)) => t.push_str(s),
                        _ => v.push(Child::Text(s.to_string(), loc)),
                    }
                }
            },
        }
        self.bump_mutation();
    }

    /// Insert a comment at the adjusted insertion location for `parent`.
    fn insert_comment(&mut self, data: &str, parent: Parent) {
        let (target, contents, _) = self.appropriate_place(Some(parent));
        let child = Child::Comment(data.to_string(), Some(self.current_loc));
        match target {
            Parent::Doc => self.document.push(child),
            Parent::Elem(p) => self.child_vec(p, contents).push(child),
        }
        self.bump_mutation();
    }

    /// Insert a processing instruction at the adjusted insertion location
    /// for `parent` (WHATWG 13.2.6.1 "insert a processing instruction").
    fn insert_pi(&mut self, target: &str, data: &str, parent: Parent) {
        let (target_parent, contents, _) = self.appropriate_place(Some(parent));
        let child = Child::ProcessingInstruction {
            target: target.to_string(),
            data: data.to_string(),
            loc: Some(self.current_loc),
        };
        match target_parent {
            Parent::Doc => self.document.push(child),
            Parent::Elem(p) => self.child_vec(p, contents).push(child),
        }
        self.bump_mutation();
    }

    /// Detach `el` from its parent (used by frameset/body swap and the
    /// adoption agency algorithm).
    fn detach(&mut self, el: El) {
        match self.arena[el].parent {
            None => {}
            Some(Parent::Doc) => {
                // A foster run buffered for this parent must be visible
                // before the retain scans the vec.
                self.flush_foster_buf();
                self.document.retain(|c| *c != Child::Elem(el));
                self.arena[el].parent = None;
                self.document_has_element =
                    self.document.iter().any(|c| matches!(c, Child::Elem(_)));
                self.bump_mutation();
            }
            Some(Parent::Elem(p)) => {
                // A foster run buffered for this parent must be visible
                // before the retain scans the vec.
                self.flush_foster_buf();
                // The element lives in exactly one of the two vecs; scan
                // regular children first and only scan template contents
                // when nothing was removed — one pass in the common case
                // instead of a `contains` pre-scan plus a `retain`.
                let node = &mut self.arena[p];
                let before = node.children.len();
                node.children.retain(|c| *c != Child::Elem(el));
                if node.children.len() == before
                    && let Some(tc) = node.template_contents.as_mut()
                {
                    tc.retain(|c| *c != Child::Elem(el));
                }
                self.arena[el].parent = None;
                self.bump_mutation();
            }
        }
    }

    /// Remove an element from the stack of open elements without detaching
    /// it from the tree.
    fn remove_from_stack(&mut self, el: El) {
        if let Some(pos) = self.stack.iter().position(|&e| e == el) {
            self.remove_open(pos);
        }
    }

    fn merge_attrs(&mut self, el: El, attrs: &[(AttrName, String)]) {
        for (k, v) in attrs {
            if !self.arena[el].attrs.iter().any(|(ek, _)| ek == k) {
                self.arena[el].attrs.push((k.clone(), v.clone()));
            }
        }
        self.bump_mutation();
    }
}

// ---------------------------------------------------------------------------
// Active formatting elements
// ---------------------------------------------------------------------------

impl<'a, 'x> TreeBuilder<'a, 'x> {
    fn afe_push_marker(&mut self) {
        self.afe.push(AfeEntry::Marker);
    }

    /// Push an element on the list of active formatting elements, applying
    /// the Noah's Ark clause.
    fn afe_push(&mut self, el: El, token_name: TagName, attrs: Vec<(AttrName, String)>) {
        let start = self
            .afe
            .iter()
            .rposition(|e| e.is_marker())
            .map_or(0, |p| p + 1);
        let mut same = 0usize;
        let mut first = None;
        for (i, e) in self.afe.iter().enumerate().skip(start) {
            if let AfeEntry::Element {
                token_name: tn,
                attrs: at,
                ..
            } = e
                && tn == &token_name
                && attrs_eq(at, &attrs)
            {
                same += 1;
                if first.is_none() {
                    first = Some(i);
                }
            }
        }
        if same >= 3 {
            // `first` is set when `same` becomes 1, so it is always `Some`
            // here in a correct run; fail closed (skip the removal) rather
            // than panicking if the invariant ever breaks.
            if let Some(f) = first {
                self.afe.remove(f);
            }
        }
        self.afe.push(AfeEntry::Element {
            element: el,
            token_name,
            attrs,
        });
    }

    fn afe_remove(&mut self, el: El) {
        if let Some(i) = self.afe.iter().position(|e| e.element() == Some(el)) {
            self.afe.remove(i);
        }
    }

    fn afe_contains(&self, el: El) -> bool {
        self.afe.iter().any(|e| e.element() == Some(el))
    }

    /// Clear the list of active formatting elements up to the last marker.
    fn afe_clear_to_marker(&mut self) {
        while let Some(e) = self.afe.pop() {
            if e.is_marker() {
                break;
            }
        }
    }

    /// Reconstruct the active formatting elements, WHATWG 13.2.4.3.
    fn reconstruct(&mut self) {
        let n = self.afe.len();
        if n == 0 {
            return;
        }
        if self.afe[n - 1].is_marker() {
            return;
        }
        if let Some(el) = self.afe[n - 1].element()
            && self.stack.contains(&el)
        {
            return;
        }
        // Rewind to the first entry that needs recreation.
        let mut i = n - 1;
        while i > 0 {
            match &self.afe[i - 1] {
                AfeEntry::Marker => break,
                AfeEntry::Element { element, .. } => {
                    if self.stack.contains(element) {
                        break;
                    }
                    i -= 1;
                }
            }
        }
        // Advance: recreate each entry.
        loop {
            let old_el = self.afe[i].element();
            let (token_name, attrs) = match &self.afe[i] {
                AfeEntry::Marker => break, // cannot happen
                AfeEntry::Element {
                    token_name, attrs, ..
                } => (token_name.clone(), attrs.clone()),
            };
            // The recreated element stands in for the original token's
            // element, so it keeps the original's source location.
            let loc = old_el.and_then(|e| self.arena[e].loc);
            let el = self.create_element(token_name, Namespace::Html, attrs, loc);
            self.insert_at_adjusted(el);
            self.push_open(el);
            self.afe[i] = AfeEntry::Element {
                element: el,
                token_name: self.arena[el].tag.clone(),
                attrs: self.arena[el].attrs.clone(),
            };
            if i + 1 >= self.afe.len() {
                break;
            }
            i += 1;
        }
    }

    /// The last formatting element after the last marker with `tag`, if any.
    fn afe_last(&self, tag: &TagName) -> Option<El> {
        for e in self.afe.iter().rev() {
            match e {
                AfeEntry::Marker => break,
                AfeEntry::Element {
                    element,
                    token_name,
                    ..
                } => {
                    if token_name == tag {
                        return Some(*element);
                    }
                }
            }
        }
        None
    }

    fn afe_token_of(&self, el: El) -> (TagName, Vec<(AttrName, String)>) {
        for e in &self.afe {
            if let AfeEntry::Element {
                element,
                token_name,
                attrs,
            } = e
                && *element == el
            {
                return (token_name.clone(), attrs.clone());
            }
        }
        let n = &self.arena[el];
        (n.tag.clone(), n.attrs.clone())
    }

    // ------------------------------------------------------------------
    // Adoption agency algorithm, WHATWG 13.2.6.4.7
    // ------------------------------------------------------------------

    /// Run the adoption agency algorithm for `subject` (a tag name).
    fn adoption_agency(&mut self, subject: &TagName) {
        // The algorithm below moves nodes between child vecs with direct
        // position arithmetic; an open foster run would be invisible to
        // it, so materialize any buffered run first (no-op when none).
        self.flush_foster_buf();
        // Step 2: current node has the tag but is not a formatting element.
        let cur = self.current_node();
        if self.arena[cur].ns == Namespace::Html
            && self.arena[cur].tag == *subject
            && !self.afe_contains(cur)
        {
            self.pop_open();
            return;
        }
        let mut outer = 0u32;
        loop {
            if outer >= 8 {
                return;
            }
            outer += 1;
            // formattingElement: last matching element after the last marker.
            let formatting_element = match self.afe_last(subject) {
                Some(el) => el,
                None => {
                    self.any_other_end_tag(subject);
                    return;
                }
            };
            if !self.stack.contains(&formatting_element) {
                self.afe_remove(formatting_element);
                return;
            }
            if !self.has_in_scope(subject, SCOPE_DEFAULT) {
                return;
            }
            // (formattingElement != current node is only a parse error.)
            // furthestBlock: topmost special element below formattingElement
            // (stack index 0 is the bottom of the stack, so the first
            // special element after fe_pos is the topmost one below the
            // formatting element).
            let Some(fe_pos) = self.stack.iter().position(|&e| e == formatting_element) else {
                // Defensive: `formatting_element` was just confirmed on the
                // stack above; bail out of this agency round instead of
                // panicking.
                self.parse_error(ParseErrorCode::TreeInvariant);
                return;
            };
            let furthest_block = self.stack[fe_pos + 1..]
                .iter()
                .find(|&&e| self.is_special(e))
                .copied();
            let furthest_block = match furthest_block {
                None => {
                    while let Some(el) = self.pop_open() {
                        if el == formatting_element {
                            break;
                        }
                    }
                    self.afe_remove(formatting_element);
                    return;
                }
                Some(fb) => fb,
            };
            // MAJOR 7 guard: the formatting element is never the stack
            // bottom, so `fe_pos > 0` in a correct run; fail closed instead
            // of underflowing `fe_pos - 1`.
            if fe_pos == 0 {
                self.parse_error(ParseErrorCode::TreeInvariant);
                return;
            }
            let common_ancestor = self.stack[fe_pos - 1];
            // Bookmark: index of formattingElement in the afe list.
            let Some(bookmark0) = self
                .afe
                .iter()
                .position(|e| e.element() == Some(formatting_element))
            else {
                // Defensive: `formatting_element` came from the afe above;
                // bail out instead of panicking.
                self.parse_error(ParseErrorCode::TreeInvariant);
                return;
            };
            let mut bookmark = bookmark0;
            let mut node = furthest_block;
            let mut last_node = furthest_block;
            let mut inner = 0u32;
            // Tracks the element above a node removed from the stack, for
            // step 13.2's "was immediately above before removal".
            let mut removed_above: Option<(El, El)> = None;
            loop {
                inner += 1;
                // Step 13.2: element immediately above node.
                node = match self.stack.iter().position(|&e| e == node) {
                    // MAJOR 7 guard: never index below the stack bottom.
                    Some(pos) if pos > 0 => self.stack[pos - 1],
                    Some(_) => {
                        self.parse_error(ParseErrorCode::TreeInvariant);
                        return;
                    }
                    None => match removed_above {
                        Some((removed, above)) if removed == node => above,
                        // Defensive fallback: `node` only leaves the stack
                        // through the tracked removal below, so this arm is
                        // unreachable in a correct run. If it ever fires
                        // (adversarial input, or a future refactor breaking
                        // the tracking), record a parse error and abandon
                        // the adoption agency rather than crashing the
                        // whole parse.
                        _ => {
                            self.parse_error(ParseErrorCode::AdoptionAgencyLostNode);
                            return;
                        }
                    },
                };
                if node == formatting_element {
                    break;
                }
                if inner > 3 && self.afe_contains(node) {
                    self.afe_remove(node);
                }
                if !self.afe_contains(node) {
                    let Some(pos) = self.stack.iter().position(|&e| e == node) else {
                        self.parse_error(ParseErrorCode::TreeInvariant);
                        return;
                    };
                    if pos == 0 {
                        // MAJOR 7 guard: never index below the stack bottom.
                        self.parse_error(ParseErrorCode::TreeInvariant);
                        return;
                    }
                    removed_above = Some((node, self.stack[pos - 1]));
                    self.remove_open(pos);
                    continue;
                }
                // Create a replacement element for node's original token.
                let (token_name, attrs) = self.afe_token_of(node);
                let new_el = self.create_element(token_name, Namespace::Html, attrs, None);
                if let Some(i) = self.afe.iter().position(|e| e.element() == Some(node)) {
                    // The position matched `element() == Some(node)`, so
                    // entry `i` is an `Element` in a correct run; a `Marker`
                    // here is defensively ignored instead of panicking.
                    if let AfeEntry::Element {
                        token_name, attrs, ..
                    } = &self.afe[i]
                    {
                        let (tn, at) = (token_name.clone(), attrs.clone());
                        self.afe[i] = AfeEntry::Element {
                            element: new_el,
                            token_name: tn,
                            attrs: at,
                        };
                    }
                }
                if let Some(pos) = self.stack.iter().position(|&e| e == node) {
                    self.stack[pos] = new_el;
                }
                node = new_el;
                if last_node == furthest_block {
                    // `new_el` was just stored in the afe above, so this
                    // finds it in a correct run; keep the previous bookmark
                    // (clamped at its use site) instead of panicking.
                    bookmark = self
                        .afe
                        .iter()
                        .position(|e| e.element() == Some(new_el))
                        .map(|p| p + 1)
                        .unwrap_or(bookmark);
                }
                // Append lastNode to node.
                self.detach(last_node);
                self.arena[node].children.push(Child::Elem(last_node));
                self.arena[last_node].parent = Some(Parent::Elem(node));
                self.bump_mutation();
                last_node = node;
            }
            // Steps 14-16: move lastNode to the adjusted insertion location.
            let (target, contents, ref_child) =
                self.appropriate_place(Some(Parent::Elem(common_ancestor)));
            self.detach(last_node);
            let mut ok = self.arena[last_node].parent.is_none();
            if ok {
                // lastNode must not be an ancestor-or-self of target.
                if let Parent::Elem(t) = target {
                    let mut cur = Some(t);
                    while let Some(c) = cur {
                        if c == last_node {
                            ok = false;
                            break;
                        }
                        cur = match self.arena[c].parent {
                            Some(Parent::Elem(p)) => Some(p),
                            _ => None,
                        };
                    }
                }
            }
            if ok {
                match target {
                    // In fragment mode the "document" is the fragment
                    // itself, which legally holds many element children.
                    Parent::Doc if !self.is_fragment() => {
                        ok = !self.document_has_element;
                    }
                    Parent::Doc | Parent::Elem(_) => {}
                }
            }
            if ok && let Some(r) = ref_child {
                ok = match self.arena[r].parent {
                    Some(p) => p == target,
                    None => false,
                };
            }
            if ok {
                match target {
                    Parent::Doc => {
                        self.document.push(Child::Elem(last_node));
                        self.arena[last_node].parent = Some(Parent::Doc);
                        self.document_has_element = true;
                        self.bump_mutation();
                    }
                    Parent::Elem(t) => {
                        let v = self.child_vec(t, contents);
                        match ref_child.and_then(|r| v.iter().position(|c| *c == Child::Elem(r))) {
                            Some(i) => v.insert(i, Child::Elem(last_node)),
                            None => v.push(Child::Elem(last_node)),
                        }
                        self.arena[last_node].parent = Some(Parent::Elem(t));
                        self.bump_mutation();
                    }
                }
            }
            // Steps 17-19: new formatting element wrapping furthestBlock's
            // children. (Cloned: the token is also needed for step 20.)
            let (fe_name, fe_attrs) = self.afe_token_of(formatting_element);
            let new_fe =
                self.create_element(fe_name.clone(), Namespace::Html, fe_attrs.clone(), None);
            let kids = std::mem::take(&mut self.arena[furthest_block].children);
            for k in &kids {
                if let Child::Elem(e) = k {
                    self.arena[*e].parent = Some(Parent::Elem(new_fe));
                }
            }
            self.arena[new_fe].children = kids;
            self.arena[furthest_block]
                .children
                .push(Child::Elem(new_fe));
            self.arena[new_fe].parent = Some(Parent::Elem(furthest_block));
            self.bump_mutation();
            // Step 20: afe bookkeeping.
            self.afe_remove(formatting_element);
            self.afe.insert(
                bookmark.min(self.afe.len()),
                AfeEntry::Element {
                    element: new_fe,
                    token_name: fe_name,
                    attrs: fe_attrs,
                },
            );
            // Step 21: stack bookkeeping.
            if let Some(pos) = self.stack.iter().position(|&e| e == formatting_element) {
                self.remove_open(pos);
            }
            let Some(fb_pos) = self.stack.iter().position(|&e| e == furthest_block) else {
                // Defensive: the inner loop only removes nodes strictly
                // above `furthest_block`; bail out instead of panicking.
                self.parse_error(ParseErrorCode::TreeInvariant);
                return;
            };
            self.insert_open(fb_pos + 1, new_fe);
        }
    }

    /// "Any other end tag" in body, WHATWG 13.2.6.4.7.
    fn any_other_end_tag(&mut self, name: &TagName) {
        let mut node = self.current_node();
        loop {
            let n = &self.arena[node];
            if n.ns == Namespace::Html && n.tag == *name {
                self.generate_implied(Some(name.atom()), false);
                while let Some(el) = self.pop_open() {
                    if el == node {
                        break;
                    }
                }
                return;
            }
            if self.is_special(node) {
                // No matching open element: parse error, ignore the token.
                self.parse_error(ParseErrorCode::StrayEndTag);
                return;
            }
            let Some(pos) = self.stack.iter().position(|&e| e == node) else {
                // Defensive: `node` left the stack through an untracked
                // path; ignore the token instead of panicking.
                return;
            };
            if pos == 0 {
                // MAJOR 7 guard: the `html` element is special, so the walk
                // above always stops there first; never index below it.
                return;
            }
            node = self.stack[pos - 1];
        }
    }
}

/// Attribute multiset equality for the Noah's Ark clause.
///
/// The tokenizer drops duplicate attribute names, so names are unique
/// within each list; with equal lengths, multiset equality reduces to
/// "every (name, value) pair of `a` occurs in `b`". O(n^2) comparisons,
/// zero allocation, no sort — the old version built and sorted two
/// temporary vectors per call on this hot path.
fn attrs_eq(a: &[(AttrName, String)], b: &[(AttrName, String)]) -> bool {
    a.len() == b.len()
        && a.iter()
            .all(|(k, v)| b.iter().any(|(ek, ev)| ek == k && ev == v))
}

/// ASCII case-insensitive tag-name equality for the foreign-content
/// end-tag step (WHATWG 13.2.6.5): the token name is already
/// ASCII-lowercased by the tokenizer, while foreign element names keep
/// their spec case (`foreignObject`). Well-known names compare by atom
/// (one `u32` compare); anything else falls back to a string compare.
fn tag_eq_fold(a: &TagName, b: &TagName) -> bool {
    let (aa, ab) = (a.atom(), b.atom());
    if aa == ab && aa != NameAtom::UNKNOWN {
        // Equal atoms imply equal strings (the table holds unique
        // strings); unequal well-known atoms imply unequal strings.
        return true;
    }
    a.as_str().eq_ignore_ascii_case(b.as_str())
}

// ---------------------------------------------------------------------------
// Token dispatch
// ---------------------------------------------------------------------------

impl<'a, 'x> TreeBuilder<'a, 'x> {
    fn run(&mut self) {
        while !self.stop {
            if self.step_token() {
                break;
            }
        }
        // Safety net: the EOF break above bypasses `handle_token`, so a run
        // buffered by the final token would otherwise never be inserted.
        self.flush_text();
        // Any foster run still open at EOF (the table was never closed)
        // materializes here; `drive_sink` runs after this and must see the
        // finished tree.
        self.flush_foster_buf();
    }

    /// Process one tokenizer token: the body of [`run`](Self::run),
    /// extracted so tests can drive a partial parse (mid-parse snapshots,
    /// mutation counting). Returns true when the EOF break fired and the
    /// caller should stop stepping.
    fn step_token(&mut self) -> bool {
        // CDATA sections are only allowed in foreign content; the
        // adjusted current node (not the synthetic fragment root)
        // decides, so foreign fragment contexts allow them.
        if self.stack.is_empty() {
            self.tok.cdata_allowed = false;
        } else {
            let cur = self.adjusted_current_node();
            self.tok.cdata_allowed = self.arena[cur].ns != Namespace::Html;
        }
        let mut tok = self.tok.next_token();
        self.current_loc = self.tok.token_loc();
        // A single leading LF after pre/listing/textarea is ignored.
        // Text runs batch characters, so a leading LF is stripped from
        // the run (and its location advanced past the newline) instead
        // of dropping a whole token.
        if self.skip_lf {
            self.skip_lf = false;
            let mut stripped = false;
            if let Token::Text(s) = &mut tok
                && s.as_bytes().first() == Some(&b'\n')
            {
                s.drain(..1);
                stripped = true;
            }
            if stripped {
                if matches!(&tok, Token::Text(s) if s.is_empty()) {
                    tok = self.tok.next_token();
                    self.current_loc = self.tok.token_loc();
                } else {
                    self.current_loc.line += 1;
                    self.current_loc.col = 1;
                }
            }
        }
        // Tokenizer-level parse errors, in source order relative to the
        // tree-construction errors for the token they precede.
        let tok_errors = self.tok.take_errors();
        self.errors.extend(tok_errors);
        // The tokenizer keeps returning Eof; a single stop flag ends us.
        if matches!(tok, Token::Eof) && self.stop_eof_pending() {
            return true;
        }
        self.handle_token(&mut tok);
        false
    }

    /// True once an EOF token has been fully processed (stop was requested).
    fn stop_eof_pending(&self) -> bool {
        self.stop
    }

    fn handle_token(&mut self, tok: &mut Token) {
        loop {
            // A buffered character run belongs before anything this
            // dispatch step inserts; flush it whenever the token being
            // (re)processed is not itself character data.
            if !matches!(tok, Token::Text(_)) {
                self.flush_text();
            }
            let flow = if self.use_foreign_content(tok) {
                self.foreign_content(tok)
            } else {
                self.dispatch(tok, self.mode)
            };
            match flow {
                Flow::Done => break,
                Flow::Again => {}
            }
            if self.stop {
                break;
            }
        }
    }

    /// Tree construction dispatcher: HTML vs foreign content, WHATWG 13.2.6.
    fn use_foreign_content(&mut self, tok: &mut Token) -> bool {
        if self.stack.is_empty() {
            return false;
        }
        if matches!(tok, Token::Eof) {
            return false;
        }
        // WHATWG 13.2.6: foreign-content dispatch uses the adjusted
        // current node, so a foreign fragment context (e.g. `svg`) parses
        // its content as foreign.
        let cur = self.adjusted_current_node();
        let n = &self.arena[cur];
        if n.ns == Namespace::Html {
            return false;
        }
        let is_mathml_text = n.ns == Namespace::MathMl
            && matches!(
                n.tag.atom(),
                NameAtom::MI | NameAtom::MO | NameAtom::MN | NameAtom::MS | NameAtom::MTEXT
            );
        let is_html_integration = (n.ns == Namespace::MathMl
            && n.tag.atom() == NameAtom::ANNOTATION_XML
            && n.annotation_xml_html)
            || (n.ns == Namespace::Svg
                && matches!(
                    n.tag.atom(),
                    NameAtom::FOREIGNOBJECT | NameAtom::DESC | NameAtom::TITLE
                ));
        match tok {
            Token::StartTag(t) => {
                if is_mathml_text
                    && t.name.atom() != NameAtom::MGLYPH
                    && t.name.atom() != NameAtom::MALIGNMARK
                {
                    return false;
                }
                if n.ns == Namespace::MathMl
                    && n.tag.atom() == NameAtom::ANNOTATION_XML
                    && t.name.atom() == NameAtom::SVG
                {
                    return false;
                }
                if is_html_integration {
                    return false;
                }
                true
            }
            Token::Text(_) => !(is_mathml_text || is_html_integration),
            // Comments, doctypes and end tags in foreign content use the
            // foreign rules.
            _ => true,
        }
    }

    fn dispatch(&mut self, tok: &mut Token, mode: Mode) -> Flow {
        match mode {
            Mode::Initial => self.mode_initial(tok),
            Mode::BeforeHtml => self.mode_before_html(tok),
            Mode::BeforeHead => self.mode_before_head(tok),
            Mode::InHead => self.mode_in_head(tok),
            Mode::InHeadNoscript => self.mode_in_head_noscript(tok),
            Mode::AfterHead => self.mode_after_head(tok),
            Mode::InBody => self.mode_in_body(tok),
            Mode::Text => self.mode_text(tok),
            Mode::InTable => self.mode_in_table(tok),
            Mode::InTableText => self.mode_in_table_text(tok),
            Mode::InCaption => self.mode_in_caption(tok),
            Mode::InColumnGroup => self.mode_in_column_group(tok),
            Mode::InTableBody => self.mode_in_table_body(tok),
            Mode::InRow => self.mode_in_row(tok),
            Mode::InCell => self.mode_in_cell(tok),
            Mode::InTemplate => self.mode_in_template(tok),
            Mode::AfterBody => self.mode_after_body(tok),
            Mode::InFrameset => self.mode_in_frameset(tok),
            Mode::AfterFrameset => self.mode_after_frameset(tok),
            Mode::AfterAfterBody => self.mode_after_after_body(tok),
            Mode::AfterAfterFrameset => self.mode_after_after_frameset(tok),
        }
    }

    fn parsing_template_contents(&self) -> bool {
        self.stack.iter().any(|&e| {
            self.arena[e].ns == Namespace::Html && self.arena[e].tag.atom() == NameAtom::TEMPLATE
        }) || self.fragment_ctx_is_template()
    }

    /// Whether the fragment context element is an HTML `template`: the
    /// synthetic context never sits on the stack, so the stack-only
    /// "parsing template contents" checks miss it otherwise.
    fn fragment_ctx_is_template(&self) -> bool {
        self.fragment_ctx.is_some_and(|ctx| {
            let n = &self.arena[ctx];
            n.ns == Namespace::Html && n.tag.atom() == NameAtom::TEMPLATE
        })
    }

    /// Whether the fragment context element is an HTML `select`: the
    /// "in body" `input` and `select` start-tag rules ignore those
    /// tokens in a select fragment context (WHATWG 13.2.6.4.7; the
    /// "in select" insertion modes were removed from the spec, but this
    /// fragment carve-out preserves the ignore behavior).
    fn fragment_ctx_is_select(&self) -> bool {
        self.fragment_ctx.is_some_and(|ctx| {
            let n = &self.arena[ctx];
            n.ns == Namespace::Html && n.tag.atom() == NameAtom::SELECT
        })
    }

    /// One step of "reset the insertion mode appropriately" (WHATWG 13.2.4.1)
    /// for an HTML element with the given tag name. Returns the mode to
    /// switch to, or `None` when the tag selects no mode ("otherwise: go to
    /// the next step").
    fn reset_mode_for_tag(&self, tag: NameAtom, last: bool) -> Option<Mode> {
        Some(match tag {
            NameAtom::TD | NameAtom::TH if !last => Mode::InCell,
            NameAtom::TR => Mode::InRow,
            NameAtom::TBODY | NameAtom::THEAD | NameAtom::TFOOT => Mode::InTableBody,
            NameAtom::CAPTION => Mode::InCaption,
            NameAtom::COLGROUP => Mode::InColumnGroup,
            NameAtom::TABLE => Mode::InTable,
            NameAtom::TEMPLATE => self.template_modes.last().copied().unwrap_or(Mode::InBody),
            NameAtom::HEAD if !last => Mode::InHead,
            NameAtom::BODY => Mode::InBody,
            NameAtom::FRAMESET => Mode::InFrameset,
            NameAtom::HTML => {
                if self.head_ptr.is_none() {
                    Mode::BeforeHead
                } else {
                    Mode::AfterHead
                }
            }
            _ => return None,
        })
    }

    fn reset_insertion_mode(&mut self) {
        let fragment = self.fragment_root.is_some();
        for (idx, &el) in self.stack.iter().enumerate().rev() {
            let last = idx == 0;
            // In fragment mode the synthetic root is transparent to the
            // walk: the spec substitutes the fragment context element for
            // the first stack node (WHATWG 13.2.4.1 step 3).
            if !(fragment && last) {
                let n = &self.arena[el];
                if n.ns == Namespace::Html
                    && let Some(m) = self.reset_mode_for_tag(n.tag.atom(), last)
                {
                    self.mode = m;
                    return;
                }
            }
            if last {
                // Fragment case (WHATWG 13.2.4.1 step 3): substitute the
                // context element for the synthetic root. The reset cases
                // refer to HTML-namespace elements only, so a foreign
                // context keeps the "in body" fallback.
                if let Some(ctx) = self.fragment_ctx
                    && self.arena[ctx].ns == Namespace::Html
                {
                    let tag = self.arena[ctx].tag.atom();
                    if let Some(m) = self.reset_mode_for_tag(tag, true) {
                        self.mode = m;
                        return;
                    }
                }
                self.mode = Mode::InBody;
                return;
            }
        }
    }

    // -- "initial" --------------------------------------------------------

    fn mode_initial(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::Text(s) => {
                // Whitespace is ignored; anything else triggers quirks
                // mode and reprocessing (per-character semantics, batched).
                if self.strip_ws_prefix(s) {
                    self.quirks = Quirks::Full;
                    self.mode = Mode::BeforeHtml;
                    Flow::Again
                } else {
                    Flow::Done
                }
            }
            Token::Comment(data) => {
                self.insert_comment(data, Parent::Doc);
                Flow::Done
            }
            Token::ProcessingInstruction { target, data } => {
                self.insert_pi(target, data, Parent::Doc);
                Flow::Done
            }
            Token::Doctype(dt) => {
                self.handle_doctype(dt);
                self.mode = Mode::BeforeHtml;
                Flow::Done
            }
            _ => {
                self.quirks = Quirks::Full;
                self.mode = Mode::BeforeHtml;
                Flow::Again
            }
        }
    }

    fn handle_doctype(&mut self, dt: &DoctypeToken) {
        if !self.doctype_appended {
            let has_element = self.document.iter().any(|c| matches!(c, Child::Elem(_)));
            let has_doctype = self.document.iter().any(|c| matches!(c, Child::Doctype(_)));
            if !has_element && !has_doctype {
                self.document.push(Child::Doctype(dt.clone()));
                self.doctype_appended = true;
                self.bump_mutation();
            }
        }
        if dt.force_quirks {
            self.quirks = Quirks::Full;
            return;
        }
        let name = dt.name.as_deref().unwrap_or("");
        if name != "html" {
            self.quirks = Quirks::Full;
            return;
        }
        let public = dt.public_id.as_deref().unwrap_or("");
        let system = dt.system_id.as_deref().unwrap_or("");
        let system_missing = dt.system_id.is_none();
        let quirks = [
            "-//W3O//DTD W3 HTML Strict 3.0//EN//",
            // NOTE: single `-` before `/W3C` per WHATWG 13.2.6.4.1 (not
            // `-//W3C` like the other identifiers).
            "-/W3C/DTD HTML 4.0 Transitional/EN",
            "HTML",
        ]
        .iter()
        .any(|p| public.eq_ignore_ascii_case(p))
            || system
                .eq_ignore_ascii_case("http://www.ibm.com/data/dtd/v11/ibmxhtml1-transitional.dtd")
            || QUIRKS_PUBLIC_PREFIXES
                .iter()
                .any(|p| starts_with_ci(public, p))
            || ((system_missing || system.is_empty())
                && (starts_with_ci(public, "-//W3C//DTD HTML 4.01 Frameset//")
                    || starts_with_ci(public, "-//W3C//DTD HTML 4.01 Transitional//")));
        if quirks {
            self.quirks = Quirks::Full;
            return;
        }
        let limited = starts_with_ci(public, "-//W3C//DTD XHTML 1.0 Frameset//")
            || starts_with_ci(public, "-//W3C//DTD XHTML 1.0 Transitional//")
            || (!system_missing
                && !system.is_empty()
                && (starts_with_ci(public, "-//W3C//DTD HTML 4.01 Frameset//")
                    || starts_with_ci(public, "-//W3C//DTD HTML 4.01 Transitional//")));
        if limited {
            self.quirks = Quirks::Limited;
        }
    }

    /// The quirks mode computed from the doctype (WHATWG 13.2.6.4.1).
    fn quirks_mode(&self) -> QuirksMode {
        match self.quirks {
            Quirks::None => QuirksMode::NoQuirks,
            Quirks::Limited => QuirksMode::LimitedQuirks,
            Quirks::Full => QuirksMode::Quirks,
        }
    }

    // -- "before html" ----------------------------------------------------

    fn mode_before_html(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::Doctype(_) => Flow::Done,
            Token::Comment(data) => {
                self.insert_comment(data, Parent::Doc);
                Flow::Done
            }
            Token::ProcessingInstruction { target, data } => {
                self.insert_pi(target, data, Parent::Doc);
                Flow::Done
            }
            Token::Text(s) => {
                // Whitespace is ignored; anything else inserts an implied
                // `html` element and reprocesses.
                if self.strip_ws_prefix(s) {
                    self.insert_implied_html();
                    self.mode = Mode::BeforeHead;
                    Flow::Again
                } else {
                    Flow::Done
                }
            }
            Token::StartTag(t) if t.name.atom() == NameAtom::HTML => {
                let name = std::mem::take(&mut t.name);
                let attrs = std::mem::take(&mut t.attrs);
                let el = self.create_element(name, Namespace::Html, attrs, Some(self.current_loc));
                if !self.document_has_element {
                    self.document.push(Child::Elem(el));
                    self.arena[el].parent = Some(Parent::Doc);
                    self.document_has_element = true;
                    self.bump_mutation();
                }
                self.push_open(el);
                self.mode = Mode::BeforeHead;
                Flow::Done
            }
            Token::EndTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::HEAD | NameAtom::BODY | NameAtom::HTML | NameAtom::BR
                ) =>
            {
                self.insert_implied_html();
                self.mode = Mode::BeforeHead;
                Flow::Again
            }
            Token::EndTag(_) => {
                self.parse_error(ParseErrorCode::StrayEndTag);
                Flow::Done
            }
            _ => {
                self.insert_implied_html();
                self.mode = Mode::BeforeHead;
                Flow::Again
            }
        }
    }

    fn insert_implied_html(&mut self) {
        let el = self.create_element(TagName::from("html"), Namespace::Html, Vec::new(), None);
        self.document.push(Child::Elem(el));
        self.arena[el].parent = Some(Parent::Doc);
        self.document_has_element = true;
        self.bump_mutation();
        self.push_open(el);
    }

    // -- "before head" ----------------------------------------------------

    fn mode_before_head(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::Text(s) => {
                // Whitespace is ignored; anything else inserts an implied
                // `head` element and reprocesses.
                if self.strip_ws_prefix(s) {
                    self.insert_implied_head();
                    self.mode = Mode::InHead;
                    Flow::Again
                } else {
                    Flow::Done
                }
            }
            Token::Comment(data) => {
                let p = Parent::Elem(self.current_node());
                self.insert_comment(data, p);
                Flow::Done
            }
            Token::ProcessingInstruction { target, data } => {
                let p = Parent::Elem(self.current_node());
                self.insert_pi(target, data, p);
                Flow::Done
            }
            Token::Doctype(_) => Flow::Done,
            Token::StartTag(t) if t.name.atom() == NameAtom::HTML => self.mode_in_body(tok),
            Token::StartTag(t) if t.name.atom() == NameAtom::HEAD => {
                let el = self.insert_html_element(t);
                self.head_ptr = Some(el);
                self.mode = Mode::InHead;
                Flow::Done
            }
            Token::EndTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::HEAD | NameAtom::BODY | NameAtom::HTML | NameAtom::BR
                ) =>
            {
                self.insert_implied_head();
                self.mode = Mode::InHead;
                Flow::Again
            }
            Token::EndTag(_) => {
                self.parse_error(ParseErrorCode::StrayEndTag);
                Flow::Done
            }
            _ => {
                self.insert_implied_head();
                self.mode = Mode::InHead;
                Flow::Again
            }
        }
    }

    fn insert_implied_head(&mut self) {
        // Implied: no source token, so no location.
        let el = self.create_element(TagName::from("head"), Namespace::Html, Vec::new(), None);
        self.insert_at_adjusted(el);
        self.push_open(el);
        self.head_ptr = Some(el);
    }

    // -- "in head" --------------------------------------------------------

    /// Shared `template` start-tag handling for the "in head" and "in body"
    /// insertion modes (WHATWG 13.2.6.4.4 / 13.2.6.4.7).
    fn insert_template_element(&mut self, t: &mut Tag) {
        // Declarative shadow root attributes are read before
        // `insert_html_element` moves the token's attributes into the arena.
        let dsr = declarative_shadow_root(&t.attrs);
        let el = self.insert_html_element(t);
        self.afe_push_marker();
        self.frameset_ok = false;
        self.mode = Mode::InTemplate;
        self.template_modes.push(Mode::InTemplate);
        // WHATWG HTML 13.2.6.4.7 ("in body", "a start tag whose tag name is
        // template"): with a valid `shadowrootmode`, attach a shadow root
        // to the template's parent immediately at the start tag, so the
        // template's contents parse directly into the shadow root
        // (streaming). The template element itself is dropped from the
        // final tree; the sink driver emits a shadow-root bracket in its
        // place.
        //
        // Fail-closed skips (the template stays ordinary):
        // - fragment parsing (WHATWG 13.4: declarative shadow roots are
        //   opt-in per parse; the fragment path is `innerHTML`-equivalent,
        //   which never creates them);
        // - no element parent (nothing to host the shadow root);
        // - the parent is not an HTML element that may host a shadow root:
        //   13.2.6.4.7 step 7.11.2 runs DOM "attach a shadow root" on the
        //   parent, which throws for a non-HTML namespace or a local name
        //   that is not a valid shadow host name (WHATWG DOM 4.9). In
        //   particular `template` itself cannot host, so a nested
        //   `<template shadowrootmode>` inside a template — including
        //   inside another declarative shadow root under construction —
        //   stays a plain template element of the shadow tree instead of
        //   emitting a nested bracket that would corrupt the sink;
        // - the parent already got a declarative shadow root: only the
        //   first `<template shadowrootmode>` per parent wins (DOM
        //   `attachShadow` throws on a second root; the parser must not
        //   throw, so the later template stays a plain template).
        //
        // Together these make the driver's sink guarantee true: a
        // `start_shadow_root` is never emitted while the innermost open
        // element already has an open shadow bracket.
        if let Some(dsr) = dsr
            && !self.is_fragment()
            && let Some(Parent::Elem(host)) = self.arena[el].parent
            && {
                let h = &self.arena[host];
                h.ns == Namespace::Html
                    && is_valid_shadow_host_name(&h.tag)
                    && !h
                        .children
                        .iter()
                        .any(|c| matches!(c, Child::Elem(e) if self.arena[*e].dsr.is_some()))
            }
        {
            self.arena[el].dsr = Some(dsr);
            // Attaching the shadow root changes how the driver emits this
            // node (shadow-root bracket instead of element brackets).
            self.bump_mutation();
        }
    }

    fn mode_in_head(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::Text(s) => {
                // Whitespace is inserted; anything else pops the head
                // element and reprocesses.
                if self.split_ws_prefix_insert(s) {
                    self.pop_open();
                    self.mode = Mode::AfterHead;
                    Flow::Again
                } else {
                    Flow::Done
                }
            }
            Token::Comment(data) => {
                let p = Parent::Elem(self.current_node());
                self.insert_comment(data, p);
                Flow::Done
            }
            Token::ProcessingInstruction { target, data } => {
                let p = Parent::Elem(self.current_node());
                self.insert_pi(target, data, p);
                Flow::Done
            }
            Token::Doctype(_) => Flow::Done,
            Token::StartTag(t) if t.name.atom() == NameAtom::HTML => self.mode_in_body(tok),
            Token::StartTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::BASE | NameAtom::BASEFONT | NameAtom::BGSOUND | NameAtom::LINK
                ) =>
            {
                self.insert_void_element(t);
                Flow::Done
            }
            Token::StartTag(t) if t.name.atom() == NameAtom::META => {
                self.insert_void_element(t);
                Flow::Done
            }
            Token::StartTag(t) if t.name.atom() == NameAtom::TITLE => {
                self.generic_rcdata(t);
                Flow::Done
            }
            Token::StartTag(t)
                if (t.name.atom() == NameAtom::NOSCRIPT && self.scripting)
                    || matches!(t.name.atom(), NameAtom::NOFRAMES | NameAtom::STYLE) =>
            {
                self.generic_raw_text(t);
                Flow::Done
            }
            Token::StartTag(t) if t.name.atom() == NameAtom::NOSCRIPT && !self.scripting => {
                self.insert_html_element(t);
                self.mode = Mode::InHeadNoscript;
                Flow::Done
            }
            Token::StartTag(t) if t.name.atom() == NameAtom::SCRIPT => {
                self.head_script(t);
                Flow::Done
            }
            Token::EndTag(t) if t.name.atom() == NameAtom::HEAD => {
                self.pop_open();
                self.mode = Mode::AfterHead;
                Flow::Done
            }
            Token::EndTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::BODY | NameAtom::HTML | NameAtom::BR
                ) =>
            {
                self.pop_open();
                self.mode = Mode::AfterHead;
                Flow::Again
            }
            Token::StartTag(t) if t.name.atom() == NameAtom::TEMPLATE => {
                // Declarative shadow roots are handled inside
                // `insert_template_element` (which attaches the shadow root
                // to the template's parent when `shadowrootmode` is valid).
                self.insert_template_element(t);
                Flow::Done
            }
            Token::EndTag(t) if t.name.atom() == NameAtom::TEMPLATE => {
                if self.template_modes.is_empty() {
                    return Flow::Done;
                }
                self.generate_implied(None, true);
                self.pop_until_tag(&TagName::from("template"));
                self.afe_clear_to_marker();
                self.template_modes.pop();
                self.reset_insertion_mode();
                Flow::Done
            }
            Token::StartTag(t) if t.name.atom() == NameAtom::HEAD => Flow::Done,
            Token::EndTag(_) => {
                self.parse_error(ParseErrorCode::StrayEndTag);
                Flow::Done
            }
            _ => {
                self.pop_open();
                self.mode = Mode::AfterHead;
                Flow::Again
            }
        }
    }

    fn generic_raw_text(&mut self, t: &mut Tag) {
        self.insert_html_element(t);
        self.tok.set_state(State::Rawtext);
        self.original_mode = self.mode;
        self.mode = Mode::Text;
    }

    fn generic_rcdata(&mut self, t: &mut Tag) {
        self.insert_html_element(t);
        self.tok.set_state(State::Rcdata);
        self.original_mode = self.mode;
        self.mode = Mode::Text;
    }

    fn head_script(&mut self, t: &mut Tag) {
        // Move the name/attributes out of the token (see
        // `insert_html_element`).
        let name = std::mem::take(&mut t.name);
        let attrs = std::mem::take(&mut t.attrs);
        let el = self.create_element(name, Namespace::Html, attrs, Some(self.current_loc));
        self.insert_at_adjusted(el);
        self.push_open(el);
        self.tok.set_state(State::ScriptData);
        self.original_mode = self.mode;
        self.mode = Mode::Text;
    }

    // -- "in head noscript" -----------------------------------------------

    fn mode_in_head_noscript(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::Doctype(_) => Flow::Done,
            Token::StartTag(t) if t.name.atom() == NameAtom::HTML => self.mode_in_body(tok),
            Token::EndTag(t) if t.name.atom() == NameAtom::NOSCRIPT => {
                self.pop_open();
                self.mode = Mode::InHead;
                Flow::Done
            }
            Token::Text(_) => {
                // Whitespace follows the "in head" rules; anything else
                // pops the noscript element and reprocesses.
                if self.delegate_ws_prefix(tok, Self::mode_in_head) {
                    self.pop_open();
                    self.mode = Mode::InHead;
                    Flow::Again
                } else {
                    Flow::Done
                }
            }
            Token::Comment(_) | Token::ProcessingInstruction { .. } => self.mode_in_head(tok),
            Token::StartTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::BASEFONT
                        | NameAtom::BGSOUND
                        | NameAtom::LINK
                        | NameAtom::META
                        | NameAtom::NOFRAMES
                        | NameAtom::STYLE
                ) =>
            {
                self.mode_in_head(tok)
            }
            Token::EndTag(t) if t.name.atom() == NameAtom::BR => {
                self.pop_open();
                self.mode = Mode::InHead;
                Flow::Again
            }
            Token::StartTag(t) if matches!(t.name.atom(), NameAtom::HEAD | NameAtom::NOSCRIPT) => {
                Flow::Done
            }
            Token::EndTag(_) => {
                self.parse_error(ParseErrorCode::StrayEndTag);
                Flow::Done
            }
            _ => {
                self.pop_open();
                self.mode = Mode::InHead;
                Flow::Again
            }
        }
    }

    // -- "after head" -----------------------------------------------------

    fn mode_after_head(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::Text(s) => {
                // Whitespace is inserted; anything else inserts an implied
                // `body` element and reprocesses.
                if self.split_ws_prefix_insert(s) {
                    self.insert_implied_body();
                    self.mode = Mode::InBody;
                    Flow::Again
                } else {
                    Flow::Done
                }
            }
            Token::Comment(data) => {
                let p = Parent::Elem(self.current_node());
                self.insert_comment(data, p);
                Flow::Done
            }
            Token::ProcessingInstruction { target, data } => {
                let p = Parent::Elem(self.current_node());
                self.insert_pi(target, data, p);
                Flow::Done
            }
            Token::Doctype(_) => Flow::Done,
            Token::StartTag(t) if t.name.atom() == NameAtom::HTML => self.mode_in_body(tok),
            Token::StartTag(t) if t.name.atom() == NameAtom::BODY => {
                self.insert_html_element(t);
                self.frameset_ok = false;
                self.mode = Mode::InBody;
                Flow::Done
            }
            Token::StartTag(t) if t.name.atom() == NameAtom::FRAMESET => {
                self.insert_html_element(t);
                self.mode = Mode::InFrameset;
                Flow::Done
            }
            Token::StartTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::BASE
                        | NameAtom::BASEFONT
                        | NameAtom::BGSOUND
                        | NameAtom::LINK
                        | NameAtom::META
                        | NameAtom::NOFRAMES
                        | NameAtom::SCRIPT
                        | NameAtom::STYLE
                        | NameAtom::TEMPLATE
                        | NameAtom::TITLE
                ) =>
            {
                let Some(head) = self.head_ptr else {
                    // Defensive: the head element was never created (e.g.
                    // a stray `</head>` path); ignore the token instead of
                    // panicking.
                    self.parse_error(ParseErrorCode::TreeInvariant);
                    return Flow::Done;
                };
                self.push_open(head);
                let flow = self.mode_in_head(tok);
                if let Some(pos) = self.stack.iter().position(|&e| e == head) {
                    self.remove_open(pos);
                }
                flow
            }
            Token::EndTag(t) if t.name.atom() == NameAtom::TEMPLATE => self.mode_in_head(tok),
            Token::EndTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::BODY | NameAtom::HTML | NameAtom::BR
                ) =>
            {
                self.insert_implied_body();
                self.mode = Mode::InBody;
                Flow::Again
            }
            Token::StartTag(t) if t.name.atom() == NameAtom::HEAD => Flow::Done,
            Token::EndTag(_) => {
                self.parse_error(ParseErrorCode::StrayEndTag);
                Flow::Done
            }
            _ => {
                self.insert_implied_body();
                self.mode = Mode::InBody;
                Flow::Again
            }
        }
    }

    fn insert_implied_body(&mut self) {
        // Implied: no source token, so no location.
        let el = self.create_element(TagName::from("body"), Namespace::Html, Vec::new(), None);
        self.insert_at_adjusted(el);
        self.push_open(el);
        self.frameset_ok = true;
    }
}

// ---------------------------------------------------------------------------
// "in body"
// ---------------------------------------------------------------------------

impl<'a, 'x> TreeBuilder<'a, 'x> {
    fn mode_in_body(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::Text(s) => {
                // Nulls are parse errors; the tokenizer reports them and
                // the tree builder drops them (batched: stripped from the
                // run, which the tokenizer may have merged with text).
                s.retain(|c| c != '\0');
                if s.is_empty() {
                    return Flow::Done;
                }
                // Reconstruction can insert formatting elements,
                // changing where a buffered run belongs: flush the run
                // first, under the pre-reconstruction tree. Skipped when
                // reconstruction would be a no-op (the common case).
                if self.reconstruct_pending() {
                    self.flush_text();
                    self.reconstruct();
                }
                if Self::first_non_ws(s).is_some() {
                    self.frameset_ok = false;
                }
                let run = std::mem::take(s);
                self.push_text_string(run);
                Flow::Done
            }
            Token::Comment(data) => {
                let p = Parent::Elem(self.current_node());
                self.insert_comment(data, p);
                Flow::Done
            }
            Token::ProcessingInstruction { target, data } => {
                let p = Parent::Elem(self.current_node());
                self.insert_pi(target, data, p);
                Flow::Done
            }
            Token::Doctype(_) => Flow::Done,
            Token::StartTag(t) => self.in_body_start(t),
            Token::EndTag(t) => self.in_body_end(t),
            Token::Eof => self.in_body_eof(),
        }
    }

    fn in_body_start(&mut self, t: &mut Tag) -> Flow {
        match t.name.atom() {
            NameAtom::HTML => {
                // Spec: if there is a template element on the stack of
                // open elements, ignore the token.
                if !self.stack_has_template() {
                    let cur = self.stack[0];
                    self.merge_attrs(cur, &t.attrs);
                }
                Flow::Done
            }
            NameAtom::BASE | NameAtom::BASEFONT | NameAtom::BGSOUND | NameAtom::LINK => {
                self.insert_void_element(t);
                Flow::Done
            }
            NameAtom::META => {
                self.insert_void_element(t);
                Flow::Done
            }
            NameAtom::TITLE => self.in_body_rcdata(t),
            n if (n == NameAtom::NOSCRIPT && self.scripting)
                || n == NameAtom::NOFRAMES
                || n == NameAtom::STYLE =>
            {
                self.generic_raw_text(t);
                Flow::Done
            }
            NameAtom::NOSCRIPT => {
                // scripting disabled: ordinary element (in-body
                // "any other start tag").
                self.reconstruct();
                self.insert_html_element(t);
                Flow::Done
            }
            NameAtom::SCRIPT => {
                let name = std::mem::take(&mut t.name);
                let attrs = std::mem::take(&mut t.attrs);
                let el = self.create_element(name, Namespace::Html, attrs, Some(self.current_loc));
                self.insert_at_adjusted(el);
                self.push_open(el);
                self.tok.set_state(State::ScriptData);
                self.original_mode = self.mode;
                self.mode = Mode::Text;
                Flow::Done
            }
            NameAtom::TEMPLATE => {
                self.insert_template_element(t);
                Flow::Done
            }
            NameAtom::BODY => {
                // Merge attributes into the body element (second on the
                // stack) unless this is a fragment-ish case.
                if self.is_second_on_stack("body") && !self.stack_has_template() {
                    self.frameset_ok = false;
                    let el = self.stack[1];
                    self.merge_attrs(el, &t.attrs);
                }
                Flow::Done
            }
            NameAtom::FRAMESET => {
                if self.stack.len() <= 1 || !self.is_second_on_stack("body") || !self.frameset_ok {
                    Flow::Done
                } else {
                    // Detach the body element, pop down to the root html
                    // element, then insert the frameset.
                    let body = self.stack[1];
                    self.detach(body);
                    while self.stack.len() > 1 {
                        self.pop_open();
                    }
                    self.insert_html_element(t);
                    self.mode = Mode::InFrameset;
                    Flow::Done
                }
            }
            NameAtom::ADDRESS
            | NameAtom::ARTICLE
            | NameAtom::ASIDE
            | NameAtom::BLOCKQUOTE
            | NameAtom::CENTER
            | NameAtom::DETAILS
            | NameAtom::DIALOG
            | NameAtom::DIR
            | NameAtom::DIV
            | NameAtom::DL
            | NameAtom::FIELDSET
            | NameAtom::FIGCAPTION
            | NameAtom::FIGURE
            | NameAtom::FOOTER
            | NameAtom::HEADER
            | NameAtom::HGROUP
            | NameAtom::MAIN
            | NameAtom::MENU
            | NameAtom::NAV
            | NameAtom::OL
            | NameAtom::P
            | NameAtom::SEARCH
            | NameAtom::SECTION
            | NameAtom::SUMMARY
            | NameAtom::UL => {
                if self.has_p_in_button_scope() {
                    self.close_p();
                }
                self.insert_html_element(t);
                Flow::Done
            }
            NameAtom::H1
            | NameAtom::H2
            | NameAtom::H3
            | NameAtom::H4
            | NameAtom::H5
            | NameAtom::H6 => {
                if self.has_p_in_button_scope() {
                    self.close_p();
                }
                if matches!(
                    self.current_tag(),
                    Some(NameAtom::H1)
                        | Some(NameAtom::H2)
                        | Some(NameAtom::H3)
                        | Some(NameAtom::H4)
                        | Some(NameAtom::H5)
                        | Some(NameAtom::H6)
                ) {
                    self.pop_open();
                }
                self.insert_html_element(t);
                Flow::Done
            }
            NameAtom::PRE | NameAtom::LISTING => {
                if self.has_p_in_button_scope() {
                    self.close_p();
                }
                self.insert_html_element(t);
                self.frameset_ok = false;
                self.skip_lf = true;
                Flow::Done
            }
            NameAtom::FORM => {
                if self.form_ptr.is_some() && !self.stack_has_template() {
                    Flow::Done
                } else {
                    if self.has_p_in_button_scope() {
                        self.close_p();
                    }
                    let el = self.insert_html_element(t);
                    if !self.stack_has_template() {
                        self.form_ptr = Some(el);
                    }
                    Flow::Done
                }
            }
            NameAtom::LI => {
                self.frameset_ok = false;
                // Close a stacked li, stopping at special elements other
                // than address/div/p.
                let mut close_li = false;
                for &e in self.stack.iter().rev() {
                    let is_li = self.arena[e].tag.atom() == NameAtom::LI;
                    let tag = self.arena[e].tag.clone();
                    if is_li {
                        close_li = true;
                        break;
                    }
                    if self.is_special(e)
                        && !matches!(tag.atom(), NameAtom::ADDRESS | NameAtom::DIV | NameAtom::P)
                    {
                        break;
                    }
                }
                if close_li {
                    self.generate_implied(Some(NameAtom::LI), false);
                    self.pop_until_tag(&TagName::from("li"));
                }
                if self.has_p_in_button_scope() {
                    self.close_p();
                }
                self.insert_html_element(t);
                Flow::Done
            }
            NameAtom::DD | NameAtom::DT => {
                self.frameset_ok = false;
                let mut close: Option<TagName> = None;
                for &e in self.stack.iter().rev() {
                    let tag = self.arena[e].tag.clone();
                    if matches!(tag.atom(), NameAtom::DD | NameAtom::DT) {
                        close = Some(tag);
                        break;
                    }
                    if self.is_special(e)
                        && !matches!(tag.atom(), NameAtom::ADDRESS | NameAtom::DIV | NameAtom::P)
                    {
                        break;
                    }
                }
                if let Some(tag) = close {
                    self.generate_implied(Some(tag.atom()), false);
                    self.pop_until_tag(&tag);
                }
                if self.has_p_in_button_scope() {
                    self.close_p();
                }
                self.insert_html_element(t);
                Flow::Done
            }
            NameAtom::PLAINTEXT => {
                if self.has_p_in_button_scope() {
                    self.close_p();
                }
                self.insert_html_element(t);
                self.tok.set_state(State::Plaintext);
                Flow::Done
            }
            NameAtom::BUTTON => {
                if self.has_in_scope(&TagName::from("button"), SCOPE_DEFAULT) {
                    self.generate_implied(None, false);
                    self.pop_until_tag(&TagName::from("button"));
                }
                self.reconstruct();
                self.insert_html_element(t);
                self.frameset_ok = false;
                Flow::Done
            }
            NameAtom::A => {
                // If an `a` element is on the list of active formatting
                // elements after the last marker, run adoption and drop it
                // from the list and the stack if still present.
                if let Some(el) = self.afe_last(&TagName::from("a")) {
                    self.adoption_agency(&TagName::from("a"));
                    if self.afe_contains(el) {
                        self.afe_remove(el);
                    }
                    if self.stack.contains(&el) {
                        self.remove_from_stack(el);
                    }
                }
                self.reconstruct();
                let afe_attrs = t.attrs.clone();
                let el = self.insert_html_element(t);
                self.afe_push(el, TagName::from("a"), afe_attrs);
                Flow::Done
            }
            NameAtom::B
            | NameAtom::BIG
            | NameAtom::CODE
            | NameAtom::EM
            | NameAtom::FONT
            | NameAtom::I
            | NameAtom::S
            | NameAtom::SMALL
            | NameAtom::STRIKE
            | NameAtom::STRONG
            | NameAtom::TT
            | NameAtom::U => {
                self.reconstruct();
                let afe_name = t.name.clone();
                let afe_attrs = t.attrs.clone();
                let el = self.insert_html_element(t);
                self.afe_push(el, afe_name, afe_attrs);
                Flow::Done
            }
            NameAtom::NOBR => {
                self.reconstruct();
                if self.has_in_scope(&TagName::from("nobr"), SCOPE_DEFAULT) {
                    self.adoption_agency(&TagName::from("nobr"));
                    self.reconstruct();
                }
                let afe_name = t.name.clone();
                let afe_attrs = t.attrs.clone();
                let el = self.insert_html_element(t);
                self.afe_push(el, afe_name, afe_attrs);
                Flow::Done
            }
            NameAtom::APPLET | NameAtom::MARQUEE | NameAtom::OBJECT => {
                self.reconstruct();
                self.insert_html_element(t);
                self.afe_push_marker();
                self.frameset_ok = false;
                Flow::Done
            }
            NameAtom::TABLE => {
                if !self.quirks_no_table() && self.has_p_in_button_scope() {
                    self.close_p();
                }
                self.insert_html_element(t);
                self.frameset_ok = false;
                self.mode = Mode::InTable;
                Flow::Done
            }
            NameAtom::AREA
            | NameAtom::BR
            | NameAtom::EMBED
            | NameAtom::IMG
            | NameAtom::KEYGEN
            | NameAtom::WBR => {
                self.reconstruct();
                self.insert_void_element(t);
                self.frameset_ok = false;
                Flow::Done
            }
            NameAtom::INPUT => {
                if self.fragment_ctx_is_select() {
                    // WHATWG "in body": a select fragment context
                    // ignores `<input>` (parse error).
                    self.parse_error(ParseErrorCode::UnexpectedToken);
                    return Flow::Done;
                }
                if self.has_in_scope(&TagName::from("select"), SCOPE_DEFAULT) {
                    // Parse error: pop until the select is closed, then
                    // process the token normally.
                    self.parse_error(ParseErrorCode::UnexpectedToken);
                    self.pop_until_tag(&TagName::from("select"));
                }
                self.reconstruct();
                // Read the attributes before `insert_void_element` moves
                // them out of the token.
                let has_type_hidden = t.attrs.iter().any(|(k, v)| {
                    k.eq_ignore_ascii_case("type") && v.eq_ignore_ascii_case("hidden")
                });
                self.insert_void_element(t);
                if !has_type_hidden {
                    self.frameset_ok = false;
                }
                Flow::Done
            }
            NameAtom::PARAM | NameAtom::SOURCE | NameAtom::TRACK => {
                self.insert_void_element(t);
                Flow::Done
            }
            NameAtom::HR => {
                if self.has_p_in_button_scope() {
                    self.close_p();
                }
                if self.has_in_scope(&TagName::from("select"), SCOPE_DEFAULT) {
                    // WHATWG "in body" `hr`: generate implied end tags,
                    // then a parse error only when an `option` or
                    // `optgroup` is in scope.
                    self.generate_implied(None, false);
                    if self.has_in_scope(&TagName::from("option"), SCOPE_DEFAULT)
                        || self.has_in_scope(&TagName::from("optgroup"), SCOPE_DEFAULT)
                    {
                        self.parse_error(ParseErrorCode::UnexpectedToken);
                    }
                }
                self.insert_void_element(t);
                self.frameset_ok = false;
                Flow::Done
            }
            NameAtom::IMAGE => {
                // The token is owned here (`in_body_start` never
                // reprocesses), so rename in place instead of cloning.
                t.name = TagName::from("img");
                self.in_body_start(t)
            }
            NameAtom::TEXTAREA => {
                self.insert_html_element(t);
                self.tok.set_state(State::Rcdata);
                self.original_mode = self.mode;
                self.mode = Mode::Text;
                self.frameset_ok = false;
                self.skip_lf = true;
                Flow::Done
            }
            NameAtom::XMP => {
                if self.has_p_in_button_scope() {
                    self.close_p();
                }
                self.reconstruct();
                self.frameset_ok = false;
                self.insert_html_element(t);
                self.tok.set_state(State::Rawtext);
                self.original_mode = self.mode;
                self.mode = Mode::Text;
                Flow::Done
            }
            NameAtom::IFRAME => {
                self.frameset_ok = false;
                self.generic_raw_text(t);
                Flow::Done
            }
            NameAtom::NOEMBED => {
                self.generic_raw_text(t);
                Flow::Done
            }
            NameAtom::SELECT => {
                if self.fragment_ctx_is_select() {
                    // WHATWG "in body": a select fragment context
                    // ignores a nested `<select>` start tag (parse error).
                    self.parse_error(ParseErrorCode::UnexpectedToken);
                } else if self.has_in_scope(&TagName::from("select"), SCOPE_DEFAULT) {
                    // Parse error: drop the token and pop the open select.
                    self.parse_error(ParseErrorCode::UnexpectedToken);
                    self.pop_until_tag(&TagName::from("select"));
                } else {
                    self.reconstruct();
                    self.insert_html_element(t);
                    self.frameset_ok = false;
                }
                Flow::Done
            }
            NameAtom::OPTION => {
                if self.has_in_scope(&TagName::from("select"), SCOPE_DEFAULT) {
                    self.generate_implied(Some(NameAtom::OPTGROUP), false);
                    // WHATWG "in body" `option`: parse error when an
                    // `option` element is in scope.
                    if self.has_in_scope(&TagName::from("option"), SCOPE_DEFAULT) {
                        self.parse_error(ParseErrorCode::UnexpectedToken);
                    }
                } else if matches!(self.adjusted_current_tag(), Some(NameAtom::OPTION)) {
                    // The spec pops until an option has been popped; the
                    // synthetic fragment root is never popped (the html
                    // node is only popped when the parser finishes).
                    if self.stack.len() > 1 {
                        self.pop_open();
                    } else {
                        self.parse_error(ParseErrorCode::UnexpectedToken);
                    }
                }
                self.reconstruct();
                self.insert_html_element(t);
                Flow::Done
            }
            NameAtom::OPTGROUP => {
                if self.has_in_scope(&TagName::from("select"), SCOPE_DEFAULT) {
                    self.generate_implied(None, false);
                    // WHATWG "in body" `optgroup`: parse error when an
                    // `option` or `optgroup` element is in scope.
                    if self.has_in_scope(&TagName::from("option"), SCOPE_DEFAULT)
                        || self.has_in_scope(&TagName::from("optgroup"), SCOPE_DEFAULT)
                    {
                        self.parse_error(ParseErrorCode::UnexpectedToken);
                    }
                } else if matches!(self.adjusted_current_tag(), Some(NameAtom::OPTION)) {
                    // As above: never pop the synthetic fragment root.
                    if self.stack.len() > 1 {
                        self.pop_open();
                    } else {
                        self.parse_error(ParseErrorCode::UnexpectedToken);
                    }
                }
                self.reconstruct();
                self.insert_html_element(t);
                Flow::Done
            }
            NameAtom::RB | NameAtom::RTC => {
                if self.has_in_scope(&TagName::from("ruby"), SCOPE_DEFAULT) {
                    self.generate_implied(None, false);
                }
                self.insert_html_element(t);
                Flow::Done
            }
            NameAtom::RT | NameAtom::RP => {
                if self.has_in_scope(&TagName::from("ruby"), SCOPE_DEFAULT) {
                    self.generate_implied(Some(NameAtom::RTC), false);
                }
                self.insert_html_element(t);
                Flow::Done
            }
            NameAtom::MATH => {
                self.reconstruct();
                adjust_attrs(&mut t.attrs, MATHML_ATTR_ADJUST);
                let self_closing = t.self_closing;
                self.insert_foreign_element(t, Namespace::MathMl);
                if self_closing {
                    self.pop_open();
                }
                Flow::Done
            }
            NameAtom::SVG => {
                self.reconstruct();
                if let Some(&(_, fixed)) = SVG_TAG_ADJUST.iter().find(|(from, _)| *from == t.name) {
                    t.name = TagName::from(fixed);
                }
                adjust_attrs(&mut t.attrs, SVG_ATTR_ADJUST);
                let self_closing = t.self_closing;
                self.insert_foreign_element(t, Namespace::Svg);
                if self_closing {
                    self.pop_open();
                }
                Flow::Done
            }
            NameAtom::CAPTION
            | NameAtom::COL
            | NameAtom::COLGROUP
            | NameAtom::FRAME
            | NameAtom::HEAD
            | NameAtom::TBODY
            | NameAtom::TD
            | NameAtom::TFOOT
            | NameAtom::TH
            | NameAtom::THEAD
            | NameAtom::TR => {
                // Parse error: ignore the token.
                self.parse_error(ParseErrorCode::UnexpectedToken);
                Flow::Done
            }
            _ => {
                self.reconstruct();
                self.insert_html_element(t);
                Flow::Done
            }
        }
    }

    /// Insert an rcdata element in body context (title).
    fn in_body_rcdata(&mut self, t: &mut Tag) -> Flow {
        self.generic_rcdata(t);
        Flow::Done
    }

    fn quirks_no_table(&self) -> bool {
        self.quirks == Quirks::Full
    }

    /// Body-fragment parsing helpers (not used for full documents, but the
    /// spec gates body merging on them).
    fn is_second_on_stack(&self, tag: &str) -> bool {
        self.stack.len() > 1
            && self.arena[self.stack[1]].ns == Namespace::Html
            && self.arena[self.stack[1]].tag == tag
    }

    fn stack_has_template(&self) -> bool {
        self.stack.iter().any(|&e| {
            self.arena[e].ns == Namespace::Html && self.arena[e].tag.atom() == NameAtom::TEMPLATE
        }) || self.fragment_ctx_is_template()
    }

    fn in_body_eof(&mut self) -> Flow {
        if !self.template_modes.is_empty() {
            return self.mode_in_template(&mut Token::Eof);
        }
        // WHATWG 13.2.7 "the end": EOF with anything but the implied-end-tag
        // set (plus body/html) still open is a parse error. The synthetic
        // fragment root is `html`, so it never triggers this.
        if self.stack.iter().any(|&el| {
            let n = &self.arena[el];
            !(n.ns == Namespace::Html
                && matches!(
                    n.tag.atom(),
                    NameAtom::DD
                        | NameAtom::DT
                        | NameAtom::LI
                        | NameAtom::OPTGROUP
                        | NameAtom::OPTION
                        | NameAtom::P
                        | NameAtom::RB
                        | NameAtom::RP
                        | NameAtom::RT
                        | NameAtom::RTC
                        | NameAtom::TBODY
                        | NameAtom::TD
                        | NameAtom::TFOOT
                        | NameAtom::TH
                        | NameAtom::THEAD
                        | NameAtom::TR
                        | NameAtom::BODY
                        | NameAtom::HTML
                ))
        }) {
            self.parse_error(ParseErrorCode::UnclosedElements);
        }
        self.stop = true;
        Flow::Done
    }

    fn in_body_end(&mut self, t: &Tag) -> Flow {
        match t.name.atom() {
            NameAtom::BODY => {
                if self.has_in_scope(&TagName::from("body"), SCOPE_DEFAULT) {
                    self.mode = Mode::AfterBody;
                } else {
                    self.parse_error(ParseErrorCode::StrayEndTag);
                }
                Flow::Done
            }
            NameAtom::HTML => {
                if self.has_in_scope(&TagName::from("body"), SCOPE_DEFAULT) {
                    self.mode = Mode::AfterBody;
                    Flow::Again
                } else {
                    self.parse_error(ParseErrorCode::StrayEndTag);
                    Flow::Done
                }
            }
            NameAtom::ADDRESS
            | NameAtom::ARTICLE
            | NameAtom::ASIDE
            | NameAtom::BLOCKQUOTE
            | NameAtom::BUTTON
            | NameAtom::CENTER
            | NameAtom::DETAILS
            | NameAtom::DIALOG
            | NameAtom::DIR
            | NameAtom::DIV
            | NameAtom::DL
            | NameAtom::FIELDSET
            | NameAtom::FIGCAPTION
            | NameAtom::FIGURE
            | NameAtom::FOOTER
            | NameAtom::HEADER
            | NameAtom::HGROUP
            | NameAtom::LISTING
            | NameAtom::MAIN
            | NameAtom::MENU
            | NameAtom::NAV
            | NameAtom::OL
            | NameAtom::PRE
            | NameAtom::SEARCH
            | NameAtom::SECTION
            | NameAtom::SUMMARY
            | NameAtom::UL => {
                if !self.has_in_scope(&t.name, SCOPE_DEFAULT) {
                    self.parse_error(ParseErrorCode::StrayEndTag);
                    Flow::Done
                } else {
                    self.generate_implied(None, false);
                    self.pop_until_tag(&t.name);
                    Flow::Done
                }
            }
            NameAtom::FORM => {
                if self.parsing_template_contents() {
                    if !self.has_in_scope(&TagName::from("form"), SCOPE_DEFAULT) {
                        self.parse_error(ParseErrorCode::StrayEndTag);
                        return Flow::Done;
                    }
                    self.generate_implied(None, false);
                    self.pop_until_tag(&TagName::from("form"));
                    Flow::Done
                } else {
                    let node = self.form_ptr.take();
                    let el = match node {
                        Some(e) => e,
                        None => {
                            self.parse_error(ParseErrorCode::StrayEndTag);
                            return Flow::Done;
                        }
                    };
                    if !self.node_in_scope(el, SCOPE_DEFAULT) {
                        self.parse_error(ParseErrorCode::StrayEndTag);
                        return Flow::Done;
                    }
                    self.generate_implied(None, false);
                    self.remove_from_stack(el);
                    Flow::Done
                }
            }
            NameAtom::P => {
                if !self.has_p_in_button_scope() {
                    // Insert a p element, then close it.
                    let el = self.insert_html_element(&mut Tag {
                        name: TagName::from("p"),
                        attrs: Vec::new(),
                        self_closing: false,
                    });
                    let _ = el;
                    self.close_p();
                } else {
                    self.close_p();
                }
                Flow::Done
            }
            NameAtom::LI => {
                if !self.has_in_list_item_scope(&TagName::from("li")) {
                    self.parse_error(ParseErrorCode::StrayEndTag);
                    Flow::Done
                } else {
                    self.generate_implied(Some(NameAtom::LI), false);
                    self.pop_until_tag(&TagName::from("li"));
                    Flow::Done
                }
            }
            NameAtom::DD | NameAtom::DT => {
                if !self.has_in_scope(&t.name, SCOPE_DEFAULT) {
                    self.parse_error(ParseErrorCode::StrayEndTag);
                    Flow::Done
                } else {
                    self.generate_implied(Some(t.name.atom()), false);
                    self.pop_until_tag(&t.name);
                    Flow::Done
                }
            }
            NameAtom::H1
            | NameAtom::H2
            | NameAtom::H3
            | NameAtom::H4
            | NameAtom::H5
            | NameAtom::H6 => {
                if !self.has_in_scope(&TagName::from("h1"), SCOPE_DEFAULT)
                    && !self.has_in_scope(&TagName::from("h2"), SCOPE_DEFAULT)
                    && !self.has_in_scope(&TagName::from("h3"), SCOPE_DEFAULT)
                    && !self.has_in_scope(&TagName::from("h4"), SCOPE_DEFAULT)
                    && !self.has_in_scope(&TagName::from("h5"), SCOPE_DEFAULT)
                    && !self.has_in_scope(&TagName::from("h6"), SCOPE_DEFAULT)
                {
                    self.parse_error(ParseErrorCode::StrayEndTag);
                    Flow::Done
                } else {
                    self.generate_implied(None, false);
                    self.pop_until_heading();
                    Flow::Done
                }
            }
            NameAtom::A
            | NameAtom::B
            | NameAtom::BIG
            | NameAtom::CODE
            | NameAtom::EM
            | NameAtom::FONT
            | NameAtom::I
            | NameAtom::NOBR
            | NameAtom::S
            | NameAtom::SMALL
            | NameAtom::STRIKE
            | NameAtom::STRONG
            | NameAtom::TT
            | NameAtom::U => {
                self.adoption_agency(&t.name);
                Flow::Done
            }
            NameAtom::APPLET | NameAtom::MARQUEE | NameAtom::OBJECT => {
                if !self.has_in_scope(&t.name, SCOPE_DEFAULT) {
                    self.parse_error(ParseErrorCode::StrayEndTag);
                    Flow::Done
                } else {
                    self.generate_implied(None, false);
                    self.pop_until_tag(&t.name);
                    self.afe_clear_to_marker();
                    Flow::Done
                }
            }
            NameAtom::BR => {
                // Parse error: treat as start tag, then ignore.
                self.parse_error(ParseErrorCode::StrayEndTag);
                self.in_body_start(&mut Tag {
                    name: TagName::from("br"),
                    attrs: Vec::new(),
                    self_closing: false,
                });
                Flow::Done
            }
            NameAtom::TEMPLATE => self.mode_in_head(&mut Token::EndTag(t.clone())),
            NameAtom::SELECT => {
                if !self.has_in_scope(&TagName::from("select"), SCOPE_DEFAULT) {
                    self.parse_error(ParseErrorCode::StrayEndTag);
                    Flow::Done
                } else {
                    self.generate_implied(None, false);
                    self.pop_until_tag(&TagName::from("select"));
                    Flow::Done
                }
            }
            _ => {
                self.any_other_end_tag(&t.name);
                Flow::Done
            }
        }
    }

    fn pop_until_heading(&mut self) {
        while let Some(el) = self.pop_open() {
            if matches!(
                self.arena[el].tag.atom(),
                NameAtom::H1
                    | NameAtom::H2
                    | NameAtom::H3
                    | NameAtom::H4
                    | NameAtom::H5
                    | NameAtom::H6
            ) {
                break;
            }
        }
    }

    // -- "text" -----------------------------------------------------------

    /// Whether `type`/`language` mark a script classic per WHATWG "prepare
    /// the script": a missing or empty `type`, or a JavaScript MIME type
    /// essence match (ASCII case-insensitive, surrounding whitespace
    /// ignored, `;`-separated parameters stripped). Anything else (`module`, `importmap`, `application/json`,
    /// ...) never executes as a classic script. Without `type`, the legacy
    /// `language` attribute maps to `text/<language>`.
    fn is_classic_script_type(script_type: Option<&str>, language: Option<&str>) -> bool {
        const JS_MIME: &[&str] = &[
            "application/ecmascript",
            "application/javascript",
            "application/x-ecmascript",
            "application/x-javascript",
            "text/ecmascript",
            "text/javascript",
            "text/javascript1.0",
            "text/javascript1.1",
            "text/javascript1.2",
            "text/javascript1.3",
            "text/javascript1.4",
            "text/javascript1.5",
            "text/jscript",
            "text/livescript",
            "text/x-ecmascript",
            "text/x-javascript",
        ];
        match script_type {
            None => match language {
                None => true,
                Some(l) => JS_MIME.iter().any(|m| {
                    m.strip_prefix("text/")
                        .is_some_and(|suffix| suffix.eq_ignore_ascii_case(l.trim()))
                }),
            },
            Some(t) => {
                let t = t.trim();
                if t.is_empty() {
                    return true;
                }
                // WHATWG "prepare the script" matches the `type` on its
                // MIME essence: strip `;`-separated parameters before the
                // case-insensitive comparison, so e.g.
                // `text/javascript;version=1.8` is still classic.
                let essence = t.split(';').next().map_or("", str::trim);
                JS_MIME.iter().any(|m| m.eq_ignore_ascii_case(essence))
            }
        }
    }

    /// Whether the element `el` sitting in "text" mode may execute when it
    /// closes: scripting is on, it is an HTML `script`, its `type` (or
    /// legacy `language`) marks it classic, and it is not inside a
    /// `template` (template contents are inert).
    fn script_execution_eligible(&self, el: El) -> bool {
        if !self.scripting {
            return false;
        }
        if self.arena[el].ns != Namespace::Html || self.arena[el].tag != "script" {
            return false;
        }
        // A template ancestor on the stack of open elements means this
        // script lives in template contents, which never execute.
        // (`mode_in_template` routes template `<script>` start tags
        // through "in head", so they reach "text" mode like any script.)
        let in_template = self.stack.iter().any(|&open| {
            let n = &self.arena[open];
            n.ns == Namespace::Html && n.tag.atom() == NameAtom::TEMPLATE
        });
        if in_template {
            return false;
        }
        let mut script_type = None;
        let mut language = None;
        for (name, value) in &self.arena[el].attrs {
            if name.atom() == NameAtom::TYPE {
                script_type = Some(value.as_str());
            } else if name.atom() == NameAtom::LANGUAGE {
                language = Some(value.as_str());
            }
        }
        Self::is_classic_script_type(script_type, language)
    }

    /// Close the current "text"-mode element (`script`, `style`, `title`,
    /// `textarea`) and, when it is an eligible parser-inserted classic
    /// script, execute it before the next token is tokenized (WHATWG HTML
    /// 13.2.6: "if there is a script that will execute as soon as the
    /// parser resumes ... execute the script"). Shared by the `</script>`
    /// arm and the EOF arm so an unclosed `<script>` at EOF executes
    /// exactly like a closed one.
    ///
    /// The text is collected after the pop (which flushes the buffered run
    /// into the element); evaluation itself belongs to the engine, so
    /// without scripting or an attached executor this stays a plain close,
    /// exactly as before.
    fn close_text_element(&mut self) {
        let el = self.current_node();
        let eligible = self.script_execution_eligible(el);
        self.pop_open();
        self.mode = self.original_mode;
        if !eligible {
            return;
        }
        // A buffered foster run would be invisible to the engine's DOM
        // walk; materialize it first (no-op when none). Text was already
        // flushed by `pop_open`.
        self.flush_foster_buf();
        if let Some(executor) = self.executor.take() {
            let text = self.script_text(el);
            let script_el = ScriptElement::from_el(el, self.generation);
            // Snapshot the attributes: the engine implements
            // `src`/`async`/`defer` from them, and may mutate the element
            // mid-execution through the host.
            let attrs: Vec<(String, String)> = self.arena[el]
                .attrs
                .iter()
                .map(|(n, v)| (n.as_str().to_string(), v.clone()))
                .collect();
            {
                let mut host = ScriptHost { tb: self };
                executor.execute_script(&mut host, script_el, &attrs, &text);
            }
            self.executor = Some(executor);
        }
    }

    fn mode_text(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::Text(s) => {
                let run = std::mem::take(s);
                self.push_text_string(run);
                Flow::Done
            }
            Token::Eof => {
                self.close_text_element();
                Flow::Again
            }
            Token::EndTag(t) if t.name.atom() == NameAtom::SCRIPT => {
                self.close_text_element();
                Flow::Done
            }
            Token::EndTag(t) => {
                self.pop_open();
                self.mode = self.original_mode;
                self.dispatch_end_tag(&t.name);
                Flow::Done
            }
            _ => Flow::Done,
        }
    }

    /// Handle an end tag emitted while in "text" mode, per its element type.
    fn dispatch_end_tag(&mut self, name: &TagName) {
        // Called from "text" mode after the element was already popped and
        // the original insertion mode restored; per WHATWG 13.2.6.4.3
        // (`</title>`) there is no further processing, and routing it
        // through "in head" would misreport a stray end tag.
        if name.atom() == NameAtom::TITLE {
            self.mode = self.original_mode;
        }
    }

    /// The concatenated text children of `el`: a script element's source.
    /// Called after the end tag popped (and flushed) the element.
    fn script_text(&self, el: El) -> String {
        let mut s = String::new();
        for c in &self.arena[el].children {
            if let Child::Text(t, _) = c {
                s.push_str(t);
            }
        }
        s
    }

    /// Walk every element in the in-progress tree in document order,
    /// calling `f` until it returns false. Template contents are skipped,
    /// per DOM (they are not in the document tree).
    fn each_element(&self, mut f: impl FnMut(El) -> bool) {
        let mut stack = Vec::new();
        for c in self.document.iter().rev() {
            if let Child::Elem(e) = c {
                stack.push(*e);
            }
        }
        while let Some(el) = stack.pop() {
            if !f(el) {
                return;
            }
            for c in self.arena[el].children.iter().rev() {
                if let Child::Elem(e) = c {
                    stack.push(*e);
                }
            }
        }
    }

    /// `document.getElementById`: first element in document order whose
    /// `id` attribute equals `id`.
    fn find_element_by_id(&self, id: &str) -> Option<El> {
        let mut found = None;
        self.each_element(|el| {
            let hit = self.arena[el]
                .attrs
                .iter()
                .any(|(k, v)| k == "id" && v == id);
            if hit {
                found = Some(el);
            }
            !hit
        });
        found
    }

    /// `document.getElementsByTagName`: HTML-namespace elements with tag
    /// `tag` (ASCII case-insensitive), in document order. `"*"` matches
    /// every HTML-namespace element.
    fn elements_by_tag_name(&self, tag: &str) -> Vec<El> {
        let mut out = Vec::new();
        self.each_element(|el| {
            let n = &self.arena[el];
            if n.ns == Namespace::Html && (tag == "*" || n.tag.eq_ignore_ascii_case(tag)) {
                out.push(el);
            }
            true
        });
        out
    }

    /// `setAttribute`: replace the attribute's value when present, append
    /// otherwise. The name is ASCII-lowercased for HTML-namespace
    /// elements, per DOM.
    fn set_attr(&mut self, el: El, name: &str, value: &str) {
        let name = if self.arena[el].ns == Namespace::Html {
            AttrName::intern_lowered(name)
        } else {
            AttrName::from(name)
        };
        match self.arena[el].attrs.iter_mut().find(|(k, _)| *k == name) {
            Some((_, v)) => *v = value.to_string(),
            None => self.arena[el].attrs.push((name, value.to_string())),
        }
        self.bump_mutation();
    }
}

// ---------------------------------------------------------------------------
// Table modes
// ---------------------------------------------------------------------------

impl<'a, 'x> TreeBuilder<'a, 'x> {
    /// Process a token with foster parenting enabled, using in-body rules.
    fn foster_in_body(&mut self, tok: &mut Token) -> Flow {
        // WHATWG parse error: misnested table content is reprocessed with
        // foster parenting enabled (html5lib:
        // foster-parenting-character-in-table).
        self.parse_error(ParseErrorCode::FosterParented);
        self.foster = true;
        let flow = self.mode_in_body(tok);
        self.foster = false;
        flow
    }

    fn has_in_scope_any(&self, tags: &[NameAtom], scope: &[NameAtom]) -> bool {
        let foreign_terminates = !std::ptr::eq(scope, SCOPE_TABLE);
        for &el in self.stack.iter().rev() {
            let n = &self.arena[el];
            if n.ns == Namespace::Html && tags.contains(&n.tag.atom()) {
                return true;
            }
            if n.ns == Namespace::Html && scope.contains(&n.tag.atom()) {
                return false;
            }
            if foreign_terminates && Self::is_foreign_scope_boundary(n) {
                return false;
            }
        }
        false
    }

    fn mode_in_table(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::Text(_) => {
                let cur = self.current_node();
                let n = &self.arena[cur];
                if n.ns == Namespace::Html
                    && matches!(
                        n.tag.atom(),
                        NameAtom::TABLE
                            | NameAtom::TBODY
                            | NameAtom::TEMPLATE
                            | NameAtom::TFOOT
                            | NameAtom::THEAD
                            | NameAtom::TR
                    )
                {
                    self.pending_table_chars.clear();
                    self.pending_table_chars_loc = None;
                    self.original_mode = self.mode;
                    self.mode = Mode::InTableText;
                    Flow::Again
                } else {
                    self.foster_in_body(tok)
                }
            }
            Token::Comment(data) => {
                let p = Parent::Elem(self.current_node());
                self.insert_comment(data, p);
                Flow::Done
            }
            Token::ProcessingInstruction { target, data } => {
                let p = Parent::Elem(self.current_node());
                self.insert_pi(target, data, p);
                Flow::Done
            }
            Token::Doctype(_) => Flow::Done,
            Token::StartTag(t) => match t.name.atom() {
                NameAtom::CAPTION => {
                    self.clear_back_to(&[NameAtom::TABLE, NameAtom::TEMPLATE, NameAtom::HTML]);
                    self.afe_push_marker();
                    self.insert_html_element(t);
                    self.mode = Mode::InCaption;
                    Flow::Done
                }
                NameAtom::COLGROUP => {
                    self.clear_back_to(&[NameAtom::TABLE, NameAtom::TEMPLATE, NameAtom::HTML]);
                    self.insert_html_element(t);
                    self.mode = Mode::InColumnGroup;
                    Flow::Done
                }
                NameAtom::COL => {
                    self.clear_back_to(&[NameAtom::TABLE, NameAtom::TEMPLATE, NameAtom::HTML]);
                    self.insert_html_element(&mut Tag {
                        name: TagName::from("colgroup"),
                        attrs: Vec::new(),
                        self_closing: false,
                    });
                    self.mode = Mode::InColumnGroup;
                    Flow::Again
                }
                NameAtom::TBODY | NameAtom::TFOOT | NameAtom::THEAD => {
                    self.clear_back_to(&[NameAtom::TABLE, NameAtom::TEMPLATE, NameAtom::HTML]);
                    self.insert_html_element(t);
                    self.mode = Mode::InTableBody;
                    Flow::Done
                }
                NameAtom::TD | NameAtom::TH | NameAtom::TR => {
                    self.clear_back_to(&[NameAtom::TABLE, NameAtom::TEMPLATE, NameAtom::HTML]);
                    self.insert_html_element(&mut Tag {
                        name: TagName::from("tbody"),
                        attrs: Vec::new(),
                        self_closing: false,
                    });
                    self.mode = Mode::InTableBody;
                    Flow::Again
                }
                NameAtom::TABLE => {
                    // Parse error in both branches (WHATWG "in table").
                    self.parse_error(ParseErrorCode::UnexpectedToken);
                    if !self.has_in_scope(&TagName::from("table"), SCOPE_TABLE) {
                        Flow::Done
                    } else {
                        self.pop_until_tag(&TagName::from("table"));
                        self.reset_insertion_mode();
                        Flow::Again
                    }
                }
                NameAtom::STYLE | NameAtom::SCRIPT | NameAtom::TEMPLATE => self.mode_in_head(tok),
                NameAtom::INPUT => {
                    let hidden = t
                        .attrs
                        .iter()
                        .any(|(k, v)| k == "type" && v.eq_ignore_ascii_case("hidden"));
                    if !hidden {
                        self.foster_in_body(tok)
                    } else {
                        self.insert_html_element(t);
                        self.pop_open();
                        Flow::Done
                    }
                }
                NameAtom::FORM => {
                    if self.form_ptr.is_some() && !self.parsing_template_contents() {
                        Flow::Done
                    } else {
                        let el = self.insert_html_element(t);
                        if !self.parsing_template_contents() {
                            self.form_ptr = Some(el);
                        }
                        self.pop_open();
                        Flow::Done
                    }
                }
                _ => self.foster_in_body(tok),
            },
            Token::EndTag(t) => match t.name.atom() {
                NameAtom::TABLE => {
                    if !self.has_in_scope(&TagName::from("table"), SCOPE_TABLE) {
                        self.parse_error(ParseErrorCode::UnexpectedToken);
                        Flow::Done
                    } else {
                        self.pop_until_tag(&TagName::from("table"));
                        self.reset_insertion_mode();
                        Flow::Done
                    }
                }
                NameAtom::TEMPLATE => self.mode_in_head(tok),
                NameAtom::BODY
                | NameAtom::CAPTION
                | NameAtom::COL
                | NameAtom::COLGROUP
                | NameAtom::HTML
                | NameAtom::TBODY
                | NameAtom::TD
                | NameAtom::TFOOT
                | NameAtom::TH
                | NameAtom::THEAD
                | NameAtom::TR => Flow::Done,
                _ => self.foster_in_body(tok),
            },
            Token::Eof => self.mode_in_body(tok),
        }
    }

    fn mode_in_table_text(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::Text(s) => {
                // Nulls are parse errors; drop them without starting a
                // pending run for an all-null run (matches the old
                // per-character null arm).
                if s.as_bytes().contains(&0) {
                    s.retain(|c| c != '\0');
                }
                if !s.is_empty() {
                    if self.pending_table_chars.is_empty() {
                        self.pending_table_chars_loc = Some(self.current_loc);
                    }
                    self.pending_table_chars.push_str(s);
                }
                Flow::Done
            }
            _ => {
                self.flush_pending_table_chars();
                self.mode = self.original_mode;
                Flow::Again
            }
        }
    }

    fn mode_in_caption(&mut self, tok: &mut Token) -> Flow {
        // Fake </caption>, shared by several token kinds.
        fn fake_close(tb: &mut TreeBuilder<'_, '_>) -> Flow {
            tb.generate_implied(None, false);
            tb.pop_until_tag(&TagName::from("caption"));
            tb.afe_clear_to_marker();
            tb.mode = Mode::InTable;
            Flow::Done
        }
        match tok {
            Token::EndTag(t) if t.name.atom() == NameAtom::CAPTION => {
                if !self.has_in_scope(&TagName::from("caption"), SCOPE_TABLE) {
                    Flow::Done
                } else {
                    fake_close(self)
                }
            }
            Token::StartTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::CAPTION
                        | NameAtom::COL
                        | NameAtom::COLGROUP
                        | NameAtom::TBODY
                        | NameAtom::TD
                        | NameAtom::TFOOT
                        | NameAtom::TH
                        | NameAtom::THEAD
                        | NameAtom::TR
                ) =>
            {
                if !self.has_in_scope(&TagName::from("caption"), SCOPE_TABLE) {
                    Flow::Done
                } else {
                    fake_close(self);
                    Flow::Again
                }
            }
            Token::EndTag(t) if t.name.atom() == NameAtom::TABLE => {
                if !self.has_in_scope(&TagName::from("caption"), SCOPE_TABLE) {
                    Flow::Done
                } else {
                    fake_close(self);
                    Flow::Again
                }
            }
            Token::EndTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::BODY
                        | NameAtom::COL
                        | NameAtom::COLGROUP
                        | NameAtom::HTML
                        | NameAtom::TBODY
                        | NameAtom::TD
                        | NameAtom::TFOOT
                        | NameAtom::TH
                        | NameAtom::THEAD
                        | NameAtom::TR
                ) =>
            {
                Flow::Done
            }
            _ => self.mode_in_body(tok),
        }
    }

    fn mode_in_column_group(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::Text(s) => {
                // Whitespace is inserted; anything else fakes a
                // `</colgroup>` and reprocesses.
                if self.split_ws_prefix_insert(s) {
                    if self.current_tag() != Some(NameAtom::COLGROUP) {
                        Flow::Done
                    } else {
                        self.pop_open();
                        self.mode = Mode::InTable;
                        Flow::Again
                    }
                } else {
                    Flow::Done
                }
            }
            Token::Comment(data) => {
                let p = Parent::Elem(self.current_node());
                self.insert_comment(data, p);
                Flow::Done
            }
            Token::ProcessingInstruction { target, data } => {
                let p = Parent::Elem(self.current_node());
                self.insert_pi(target, data, p);
                Flow::Done
            }
            Token::Doctype(_) => Flow::Done,
            Token::StartTag(t) if t.name.atom() == NameAtom::HTML => self.mode_in_body(tok),
            Token::StartTag(t) if t.name.atom() == NameAtom::COL => {
                self.insert_html_element(t);
                self.pop_open();
                Flow::Done
            }
            Token::EndTag(t) if t.name.atom() == NameAtom::COLGROUP => {
                if self.current_tag() != Some(NameAtom::COLGROUP) {
                    Flow::Done
                } else {
                    self.pop_open();
                    self.mode = Mode::InTable;
                    Flow::Done
                }
            }
            Token::EndTag(t) if t.name.atom() == NameAtom::COL => Flow::Done,
            Token::StartTag(t) if t.name.atom() == NameAtom::TEMPLATE => self.mode_in_head(tok),
            Token::EndTag(t) if t.name.atom() == NameAtom::TEMPLATE => self.mode_in_head(tok),
            Token::Eof => self.mode_in_body(tok),
            _ => {
                // Fake </colgroup>.
                if self.current_tag() != Some(NameAtom::COLGROUP) {
                    Flow::Done
                } else {
                    self.pop_open();
                    self.mode = Mode::InTable;
                    Flow::Again
                }
            }
        }
    }

    fn mode_in_table_body(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::StartTag(t) if t.name.atom() == NameAtom::TR => {
                self.clear_back_to(&[
                    NameAtom::TBODY,
                    NameAtom::TFOOT,
                    NameAtom::THEAD,
                    NameAtom::TEMPLATE,
                    NameAtom::HTML,
                ]);
                self.insert_html_element(t);
                self.mode = Mode::InRow;
                Flow::Done
            }
            Token::StartTag(t)
                if t.name.atom() == NameAtom::TH || t.name.atom() == NameAtom::TD =>
            {
                self.clear_back_to(&[
                    NameAtom::TBODY,
                    NameAtom::TFOOT,
                    NameAtom::THEAD,
                    NameAtom::TEMPLATE,
                    NameAtom::HTML,
                ]);
                self.insert_html_element(&mut Tag {
                    name: TagName::from("tr"),
                    attrs: Vec::new(),
                    self_closing: false,
                });
                self.mode = Mode::InRow;
                Flow::Again
            }
            Token::EndTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::TBODY | NameAtom::TFOOT | NameAtom::THEAD
                ) =>
            {
                if !self.has_in_scope(&t.name, SCOPE_TABLE) {
                    Flow::Done
                } else {
                    self.clear_back_to(&[
                        NameAtom::TBODY,
                        NameAtom::TFOOT,
                        NameAtom::THEAD,
                        NameAtom::TEMPLATE,
                        NameAtom::HTML,
                    ]);
                    self.pop_open();
                    self.mode = Mode::InTable;
                    Flow::Done
                }
            }
            Token::StartTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::CAPTION
                        | NameAtom::COL
                        | NameAtom::COLGROUP
                        | NameAtom::TBODY
                        | NameAtom::TFOOT
                        | NameAtom::THEAD
                ) =>
            {
                self.fake_table_body_end(tok)
            }
            Token::EndTag(t) if t.name.atom() == NameAtom::TABLE => self.fake_table_body_end(tok),
            Token::EndTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::BODY
                        | NameAtom::CAPTION
                        | NameAtom::COL
                        | NameAtom::COLGROUP
                        | NameAtom::HTML
                        | NameAtom::TD
                        | NameAtom::TH
                        | NameAtom::TR
                ) =>
            {
                Flow::Done
            }
            _ => self.mode_in_table(tok),
        }
    }

    /// Fake </tbody>/</tfoot>/</thead> then reprocess.
    fn fake_table_body_end(&mut self, tok: &mut Token) -> Flow {
        if !self.has_in_scope_any(
            &[NameAtom::TBODY, NameAtom::THEAD, NameAtom::TFOOT],
            SCOPE_TABLE,
        ) {
            Flow::Done
        } else {
            self.clear_back_to(&[
                NameAtom::TBODY,
                NameAtom::TFOOT,
                NameAtom::THEAD,
                NameAtom::TEMPLATE,
                NameAtom::HTML,
            ]);
            self.pop_open();
            self.mode = Mode::InTable;
            let _ = tok;
            Flow::Again
        }
    }

    fn mode_in_row(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::StartTag(t)
                if t.name.atom() == NameAtom::TH || t.name.atom() == NameAtom::TD =>
            {
                self.clear_back_to(&[NameAtom::TR, NameAtom::TEMPLATE, NameAtom::HTML]);
                self.insert_html_element(t);
                self.mode = Mode::InCell;
                self.afe_push_marker();
                Flow::Done
            }
            Token::EndTag(t) if t.name.atom() == NameAtom::TR => {
                if !self.has_in_scope(&TagName::from("tr"), SCOPE_TABLE) {
                    Flow::Done
                } else {
                    self.clear_back_to(&[NameAtom::TR, NameAtom::TEMPLATE, NameAtom::HTML]);
                    self.pop_open();
                    self.mode = Mode::InTableBody;
                    Flow::Done
                }
            }
            Token::StartTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::CAPTION
                        | NameAtom::COL
                        | NameAtom::COLGROUP
                        | NameAtom::TBODY
                        | NameAtom::TFOOT
                        | NameAtom::THEAD
                        | NameAtom::TR
                ) =>
            {
                self.fake_row_end()
            }
            Token::EndTag(t) if t.name.atom() == NameAtom::TABLE => self.fake_row_end_again(),
            Token::EndTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::TBODY | NameAtom::TFOOT | NameAtom::THEAD
                ) =>
            {
                if !self.has_in_scope(&t.name, SCOPE_TABLE)
                    || !self.has_in_scope(&TagName::from("tr"), SCOPE_TABLE)
                {
                    Flow::Done
                } else {
                    self.clear_back_to(&[NameAtom::TR, NameAtom::TEMPLATE, NameAtom::HTML]);
                    self.pop_open();
                    self.mode = Mode::InTableBody;
                    Flow::Again
                }
            }
            Token::EndTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::BODY
                        | NameAtom::CAPTION
                        | NameAtom::COL
                        | NameAtom::COLGROUP
                        | NameAtom::HTML
                        | NameAtom::TD
                        | NameAtom::TH
                ) =>
            {
                Flow::Done
            }
            _ => self.mode_in_table(tok),
        }
    }

    /// Fake </tr> for the in-row start-tag case.
    fn fake_row_end(&mut self) -> Flow {
        if !self.has_in_scope(&TagName::from("tr"), SCOPE_TABLE) {
            Flow::Done
        } else {
            self.clear_back_to(&[NameAtom::TR, NameAtom::TEMPLATE, NameAtom::HTML]);
            self.pop_open();
            self.mode = Mode::InTableBody;
            Flow::Again
        }
    }

    /// Fake </tr> for the in-row `</table>` case.
    fn fake_row_end_again(&mut self) -> Flow {
        self.fake_row_end()
    }

    fn mode_in_cell(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::EndTag(t) if t.name.atom() == NameAtom::TD || t.name.atom() == NameAtom::TH => {
                if !self.has_in_scope(&t.name, SCOPE_TABLE) {
                    Flow::Done
                } else {
                    self.close_cell();
                    Flow::Done
                }
            }
            Token::StartTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::CAPTION
                        | NameAtom::COL
                        | NameAtom::COLGROUP
                        | NameAtom::TBODY
                        | NameAtom::TD
                        | NameAtom::TFOOT
                        | NameAtom::TH
                        | NameAtom::THEAD
                        | NameAtom::TR
                ) =>
            {
                self.close_cell();
                Flow::Again
            }
            Token::EndTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::BODY
                        | NameAtom::CAPTION
                        | NameAtom::COL
                        | NameAtom::COLGROUP
                        | NameAtom::HTML
                ) =>
            {
                Flow::Done
            }
            Token::EndTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::TABLE
                        | NameAtom::TBODY
                        | NameAtom::TFOOT
                        | NameAtom::THEAD
                        | NameAtom::TR
                ) =>
            {
                if !self.has_in_scope(&t.name, SCOPE_TABLE) {
                    Flow::Done
                } else {
                    self.close_cell();
                    Flow::Again
                }
            }
            _ => self.mode_in_body(tok),
        }
    }

    /// "Close the cell": implied end tags, pop to td/th, clear AFE to
    /// marker, switch to "in row".
    fn close_cell(&mut self) {
        self.generate_implied(None, false);
        self.pop_until_one_of(&[NameAtom::TD, NameAtom::TH]);
        self.afe_clear_to_marker();
        self.mode = Mode::InRow;
    }
}

// ---------------------------------------------------------------------------
// "in template"
// ---------------------------------------------------------------------------

impl<'a, 'x> TreeBuilder<'a, 'x> {
    fn mode_in_template(&mut self, tok: &mut Token) -> Flow {
        // Switch the template insertion mode, then reprocess.
        fn switch_template(tb: &mut TreeBuilder, mode: Mode) -> Flow {
            tb.template_modes.pop();
            tb.template_modes.push(mode);
            tb.mode = mode;
            Flow::Again
        }
        match tok {
            Token::Text(_)
            | Token::Comment(_)
            | Token::ProcessingInstruction { .. }
            | Token::Doctype(_) => self.mode_in_body(tok),
            Token::StartTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::BASE
                        | NameAtom::BASEFONT
                        | NameAtom::BGSOUND
                        | NameAtom::LINK
                        | NameAtom::META
                        | NameAtom::NOFRAMES
                        | NameAtom::SCRIPT
                        | NameAtom::STYLE
                        | NameAtom::TEMPLATE
                        | NameAtom::TITLE
                ) =>
            {
                self.mode_in_head(tok)
            }
            Token::EndTag(t) if t.name.atom() == NameAtom::TEMPLATE => self.mode_in_head(tok),
            Token::StartTag(t)
                if matches!(
                    t.name.atom(),
                    NameAtom::CAPTION
                        | NameAtom::COLGROUP
                        | NameAtom::TBODY
                        | NameAtom::TFOOT
                        | NameAtom::THEAD
                ) =>
            {
                switch_template(self, Mode::InTable)
            }
            Token::StartTag(t) if t.name.atom() == NameAtom::COL => {
                switch_template(self, Mode::InColumnGroup)
            }
            Token::StartTag(t) if t.name.atom() == NameAtom::TR => {
                switch_template(self, Mode::InTableBody)
            }
            Token::StartTag(t)
                if t.name.atom() == NameAtom::TD || t.name.atom() == NameAtom::TH =>
            {
                switch_template(self, Mode::InRow)
            }
            Token::StartTag(_) => switch_template(self, Mode::InBody),
            Token::EndTag(_) => {
                self.parse_error(ParseErrorCode::StrayEndTag);
                Flow::Done
            }
            Token::Eof => {
                if !self.stack.iter().any(|&e| {
                    self.arena[e].ns == Namespace::Html
                        && self.arena[e].tag.atom() == NameAtom::TEMPLATE
                }) {
                    self.stop = true;
                    Flow::Done
                } else {
                    // Fake </template>.
                    self.pop_until_tag(&TagName::from("template"));
                    self.afe_clear_to_marker();
                    self.template_modes.pop();
                    self.reset_insertion_mode();
                    Flow::Again
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// After body / frameset modes
// ---------------------------------------------------------------------------

impl<'a, 'x> TreeBuilder<'a, 'x> {
    fn mode_after_body(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::Text(_) => {
                // Whitespace follows the "in body" rules; anything else
                // switches to "in body" and reprocesses.
                if self.delegate_ws_prefix(tok, Self::mode_in_body) {
                    self.mode = Mode::InBody;
                    Flow::Again
                } else {
                    Flow::Done
                }
            }
            Token::Comment(data) => {
                let html = self.stack[0];
                self.insert_comment(data, Parent::Elem(html));
                Flow::Done
            }
            Token::ProcessingInstruction { target, data } => {
                let html = self.stack[0];
                self.insert_pi(target, data, Parent::Elem(html));
                Flow::Done
            }
            Token::Doctype(_) => Flow::Done,
            Token::StartTag(t) if t.name.atom() == NameAtom::HTML => self.mode_in_body(tok),
            Token::EndTag(t) if t.name.atom() == NameAtom::HTML => {
                self.mode = Mode::AfterAfterBody;
                Flow::Done
            }
            Token::Eof => {
                self.stop = true;
                Flow::Done
            }
            _ => {
                self.mode = Mode::InBody;
                Flow::Again
            }
        }
    }

    fn mode_in_frameset(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::Text(s) => {
                // Whitespace is inserted; anything else is ignored.
                self.insert_ws_only(s);
                Flow::Done
            }
            Token::Comment(data) => {
                let p = Parent::Elem(self.current_node());
                self.insert_comment(data, p);
                Flow::Done
            }
            Token::ProcessingInstruction { target, data } => {
                let p = Parent::Elem(self.current_node());
                self.insert_pi(target, data, p);
                Flow::Done
            }
            Token::Doctype(_) => Flow::Done,
            Token::StartTag(t) if t.name.atom() == NameAtom::HTML => self.mode_in_body(tok),
            Token::StartTag(t) if t.name.atom() == NameAtom::FRAMESET => {
                self.insert_html_element(t);
                Flow::Done
            }
            Token::EndTag(t) if t.name.atom() == NameAtom::FRAMESET => {
                if self.stack.len() <= 1 {
                    Flow::Done
                } else {
                    self.pop_open();
                    if !matches!(self.current_tag(), Some(NameAtom::FRAMESET)) {
                        self.mode = Mode::AfterFrameset;
                    }
                    Flow::Done
                }
            }
            Token::StartTag(t) if t.name.atom() == NameAtom::FRAME => {
                self.insert_html_element(t);
                self.pop_open();
                Flow::Done
            }
            Token::StartTag(t) if t.name.atom() == NameAtom::NOFRAMES => self.mode_in_head(tok),
            Token::Eof => {
                self.stop = true;
                Flow::Done
            }
            _ => Flow::Done,
        }
    }

    fn mode_after_frameset(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::Text(s) => {
                // Whitespace is inserted; anything else is ignored.
                self.insert_ws_only(s);
                Flow::Done
            }
            Token::Comment(data) => {
                let p = Parent::Elem(self.current_node());
                self.insert_comment(data, p);
                Flow::Done
            }
            Token::ProcessingInstruction { target, data } => {
                let p = Parent::Elem(self.current_node());
                self.insert_pi(target, data, p);
                Flow::Done
            }
            Token::Doctype(_) => Flow::Done,
            Token::StartTag(t) if t.name.atom() == NameAtom::HTML => self.mode_in_body(tok),
            Token::EndTag(t) if t.name.atom() == NameAtom::HTML => {
                self.mode = Mode::AfterAfterFrameset;
                Flow::Done
            }
            Token::StartTag(t) if t.name.atom() == NameAtom::NOFRAMES => self.mode_in_head(tok),
            Token::Eof => {
                self.stop = true;
                Flow::Done
            }
            _ => Flow::Done,
        }
    }

    fn mode_after_after_body(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::Comment(data) => {
                self.insert_comment(data, Parent::Doc);
                Flow::Done
            }
            Token::ProcessingInstruction { target, data } => {
                self.insert_pi(target, data, Parent::Doc);
                Flow::Done
            }
            Token::Doctype(_) => Flow::Done,
            Token::Text(_) => {
                // Whitespace follows the "in body" rules; anything else
                // switches to "in body" and reprocesses.
                if self.delegate_ws_prefix(tok, Self::mode_in_body) {
                    self.mode = Mode::InBody;
                    Flow::Again
                } else {
                    Flow::Done
                }
            }
            Token::StartTag(t) if t.name.atom() == NameAtom::HTML => self.mode_in_body(tok),
            Token::Eof => {
                self.stop = true;
                Flow::Done
            }
            _ => {
                self.mode = Mode::InBody;
                Flow::Again
            }
        }
    }

    fn mode_after_after_frameset(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::Comment(data) => {
                self.insert_comment(data, Parent::Doc);
                Flow::Done
            }
            Token::ProcessingInstruction { target, data } => {
                self.insert_pi(target, data, Parent::Doc);
                Flow::Done
            }
            Token::Doctype(_) => Flow::Done,
            Token::Text(_) => {
                // Whitespace follows the "in body" rules; anything else
                // is ignored.
                self.delegate_ws_prefix(tok, Self::mode_in_body);
                Flow::Done
            }
            Token::StartTag(t) if t.name.atom() == NameAtom::HTML => self.mode_in_body(tok),
            Token::StartTag(t) if t.name.atom() == NameAtom::NOFRAMES => self.mode_in_head(tok),
            Token::Eof => {
                self.stop = true;
                Flow::Done
            }
            _ => Flow::Done,
        }
    }
}

// ---------------------------------------------------------------------------
// Foreign content (WHATWG 13.2.6.5)
// ---------------------------------------------------------------------------

impl<'a, 'x> TreeBuilder<'a, 'x> {
    fn is_html_integration(&self, el: El) -> bool {
        let n = &self.arena[el];
        (n.ns == Namespace::MathMl
            && n.tag.atom() == NameAtom::ANNOTATION_XML
            && n.annotation_xml_html)
            || (n.ns == Namespace::Svg
                && matches!(
                    n.tag.atom(),
                    NameAtom::FOREIGNOBJECT | NameAtom::DESC | NameAtom::TITLE
                ))
    }

    fn is_mathml_text_integration(&self, el: El) -> bool {
        let n = &self.arena[el];
        n.ns == Namespace::MathMl
            && matches!(
                n.tag.atom(),
                NameAtom::MI | NameAtom::MO | NameAtom::MN | NameAtom::MS | NameAtom::MTEXT
            )
    }

    fn foreign_content(&mut self, tok: &mut Token) -> Flow {
        match tok {
            Token::Text(s) => {
                // A non-whitespace, non-NUL character clears frameset-ok.
                // NULs become U+FFFD below without affecting it, matching
                // the old per-character path (the tokenizer reports the
                // parse error, so the tree builder stays silent).
                if s.chars().any(|c| {
                    c != '\0' && !matches!(c, '\u{9}' | '\u{A}' | '\u{C}' | '\u{D}' | '\u{20}')
                }) {
                    self.frameset_ok = false;
                }
                if s.as_bytes().contains(&0) {
                    *s = s.replace('\0', "�");
                }
                let run = std::mem::take(s);
                self.push_text_string(run);
                Flow::Done
            }
            Token::Comment(data) => {
                let p = Parent::Elem(self.current_node());
                self.insert_comment(data, p);
                Flow::Done
            }
            Token::ProcessingInstruction { target, data } => {
                let p = Parent::Elem(self.current_node());
                self.insert_pi(target, data, p);
                Flow::Done
            }
            Token::Doctype(_) => Flow::Done,
            // Breakout tags: pop to an integration point or HTML element,
            // then reprocess in HTML content.
            Token::StartTag(t)
                if FOREIGN_BREAKOUT.contains(&t.name.atom())
                    || (t.name.atom() == NameAtom::FONT
                        && t.attrs.iter().any(|(k, _)| {
                            matches!(k.atom(), NameAtom::COLOR | NameAtom::FACE | NameAtom::SIZE)
                        })) =>
            {
                while !self.stack.is_empty() {
                    let cur = self.current_node();
                    if self.is_mathml_text_integration(cur)
                        || self.is_html_integration(cur)
                        || self.arena[cur].ns == Namespace::Html
                    {
                        break;
                    }
                    self.pop_open();
                }
                // Reprocess the token directly in the current insertion
                // mode: going back through `handle_token` would re-select
                // foreign content via the adjusted current node (still
                // foreign for a foreign fragment context) and loop forever.
                self.dispatch(tok, self.mode)
            }
            Token::EndTag(t) if t.name.atom() == NameAtom::BR || t.name.atom() == NameAtom::P => {
                while !self.stack.is_empty() {
                    let cur = self.current_node();
                    if self.is_mathml_text_integration(cur)
                        || self.is_html_integration(cur)
                        || self.arena[cur].ns == Namespace::Html
                    {
                        break;
                    }
                    self.pop_open();
                }
                self.dispatch(tok, self.mode)
            }
            Token::StartTag(t) => {
                // The element inherits the adjusted current node's namespace
                // (WHATWG 13.2.6.5), so a foreign fragment context parses
                // its content as foreign.
                let cur = self.adjusted_current_node();
                let ns = &self.arena[cur].ns;
                if *ns == Namespace::MathMl {
                    adjust_attrs(&mut t.attrs, MATHML_ATTR_ADJUST);
                }
                if *ns == Namespace::Svg {
                    if let Some(&(_, fixed)) =
                        SVG_TAG_ADJUST.iter().find(|(from, _)| *from == t.name)
                    {
                        t.name = TagName::from(fixed);
                    }
                    adjust_attrs(&mut t.attrs, SVG_ATTR_ADJUST);
                }
                // Foreign attribute namespace adjustment (WHATWG 13.2.6.1,
                // "adjust foreign attributes"): the flat `Element` model
                // keeps qualified names as-is (`xlink:href`); bowser_dom
                // applies the namespaced adjustment when building its
                // `Attribute` model at sink time.
                let self_closing = t.self_closing;
                self.insert_foreign_element(t, ns.clone());
                if self_closing {
                    // Scripting is disabled, so the SVG script sub-case also
                    // just pops.
                    self.pop_open();
                }
                Flow::Done
            }
            Token::EndTag(t)
                if t.name.atom() == NameAtom::SCRIPT
                    && self.current_node_is(Namespace::Svg, "script") =>
            {
                self.pop_open();
                Flow::Done
            }
            Token::EndTag(t) => {
                // "Any other end tag" in foreign content. Step 2: parse
                // error when the current node's tag does not match.
                let cur = self.current_node();
                if !tag_eq_fold(&self.arena[cur].tag, &t.name) {
                    self.parse_error(ParseErrorCode::MismatchedEndTag);
                }
                let mut i = self.stack.len() - 1;
                loop {
                    if i == 0 {
                        return Flow::Done;
                    }
                    let el = self.stack[i];
                    if tag_eq_fold(&self.arena[el].tag, &t.name) {
                        while let Some(popped) = self.pop_open() {
                            if popped == el {
                                break;
                            }
                        }
                        return Flow::Done;
                    }
                    i -= 1;
                    if self.arena[self.stack[i]].ns == Namespace::Html {
                        return self.dispatch(tok, self.mode);
                    }
                }
            }
            Token::Eof => self.dispatch(tok, self.mode),
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests: fragment parsing and parse-error reporting
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an HTML-namespace context element for [`parse_fragment`].
    fn ctx(tag: &str) -> Element {
        Element {
            tag: TagName::from(tag),
            ns: Namespace::Html,
            attrs: Vec::new(),
            children: Vec::new(),
            template_contents: None,
            shadow_root: None,
            loc: None,
        }
    }

    /// Compact serialization for assertions (attributes and template
    /// contents included; text emitted unescaped).
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
                if let Some(tc) = &e.template_contents {
                    for k in tc {
                        render(k, out);
                    }
                }
                if let Some(shadow) = &e.shadow_root {
                    out.push_str("<#shadow-root(");
                    out.push_str(match shadow.mode {
                        ShadowRootMode::Open => "open",
                        ShadowRootMode::Closed => "closed",
                    });
                    out.push_str(")>");
                    for k in &shadow.children {
                        render(k, out);
                    }
                    out.push_str("</#shadow-root>");
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

    fn frag(html: &str, context: &Element) -> String {
        let mut s = String::new();
        render(&parse_fragment(html.as_bytes(), context), &mut s);
        s
    }

    fn frag_errors(html: &str, context: &Element) -> (String, Vec<ParseError>) {
        let (node, errors) = parse_fragment_with_errors(html.as_bytes(), context);
        let mut s = String::new();
        render(&node, &mut s);
        (s, errors)
    }

    fn codes(errors: &[ParseError]) -> Vec<ParseErrorCode> {
        errors.iter().map(|e| e.code).collect()
    }

    // -- fragment parsing (WHATWG 13.4) ------------------------------------

    #[test]
    fn fragment_returns_document_node() {
        assert!(matches!(
            parse_fragment(b"<p>hi", &ctx("body")),
            Node::Document(_)
        ));
    }

    #[test]
    fn fragment_body_context() {
        assert_eq!(frag("<p>hi", &ctx("body")), "<p>hi</p>");
    }

    #[test]
    fn fragment_table_context() {
        // WHATWG 13.4 with a `table` context: "in table" mode synthesizes
        // the tbody. The context element itself is not part of the
        // fragment.
        assert_eq!(
            frag("<tr><td>foo</td></tr>", &ctx("table")),
            "<tbody><tr><td>foo</td></tr></tbody>"
        );
    }

    #[test]
    fn fragment_table_context_drops_nested_table() {
        // "in table" `<table>` with no table in scope: parse error, ignore.
        let (s, errors) = frag_errors("<table><tr>", &ctx("table"));
        assert_eq!(s, "<tbody><tr></tr></tbody>");
        assert_eq!(codes(&errors), vec![ParseErrorCode::UnexpectedToken]);
    }

    #[test]
    fn fragment_td_context() {
        // `td` as the inspected (last) node selects no mode: "in body",
        // where `<td>` is a parse error and ignored.
        let (s, errors) = frag_errors("<td>foo</td>", &ctx("td"));
        assert_eq!(s, "foo");
        assert_eq!(
            codes(&errors),
            vec![ParseErrorCode::UnexpectedToken, ParseErrorCode::StrayEndTag]
        );
    }

    #[test]
    fn fragment_select_context() {
        assert_eq!(
            frag("<option>a</option><option>b</option>", &ctx("select")),
            "<option>a</option><option>b</option>"
        );
    }

    #[test]
    fn fragment_select_stray_end_tag() {
        let (s, errors) = frag_errors("</select><option>", &ctx("select"));
        assert_eq!(s, "<option></option>");
        assert_eq!(codes(&errors), vec![ParseErrorCode::StrayEndTag]);
    }

    #[test]
    fn fragment_select_context_ignores_input() {
        // WHATWG "in body": the fragment context element is `select`,
        // so `<input>` is a parse error and ignored.
        let (s, errors) = frag_errors("<input><option>", &ctx("select"));
        assert_eq!(s, "<option></option>");
        assert_eq!(codes(&errors), vec![ParseErrorCode::UnexpectedToken]);
    }

    #[test]
    fn fragment_select_context_ignores_nested_select() {
        // WHATWG "in body": the fragment context element is `select`,
        // so a nested `<select>` start tag is a parse error and ignored.
        let (s, errors) = frag_errors("<select><option>", &ctx("select"));
        assert_eq!(s, "<option></option>");
        assert_eq!(codes(&errors), vec![ParseErrorCode::UnexpectedToken]);
    }

    #[test]
    fn fragment_template_context() {
        // Template context pushes "in template"; other content pops back
        // out to "in body" with no extra template wrapper in the output.
        let (s, errors) = frag_errors("<div>x</div>", &ctx("template"));
        assert_eq!(s, "<div>x</div>");
        assert!(errors.is_empty());
    }

    #[test]
    fn fragment_template_table_content() {
        let (s, errors) = frag_errors("<tr><td>x</td></tr>", &ctx("template"));
        assert_eq!(s, "<tr><td>x</td></tr>");
        assert!(errors.is_empty());
    }

    #[test]
    fn fragment_textarea_rcdata() {
        // RCDATA initial state: markup lexes as text.
        let node = parse_fragment(b"<b>x</b>", &ctx("textarea"));
        assert!(
            matches!(
                &node,
                Node::Document(kids)
                    if kids.len() == 1
                        && matches!(&kids[0], Node::Text { text, .. } if text == "<b>x</b>")
            ),
            "unexpected fragment: {node:?}"
        );
    }

    #[test]
    fn fragment_title_rcdata() {
        let node = parse_fragment(b"a<b", &ctx("title"));
        assert!(
            matches!(
                &node,
                Node::Document(kids)
                    if kids.len() == 1
                        && matches!(&kids[0], Node::Text { text, .. } if text == "a<b")
            ),
            "unexpected fragment: {node:?}"
        );
    }

    #[test]
    fn fragment_foreign_svg_context() {
        // Adjusted current node: an `svg` context parses content as SVG.
        let mut c = ctx("svg");
        c.ns = Namespace::Svg;
        let node = parse_fragment(b"<circle>", &c);
        assert!(
            matches!(
                &node,
                Node::Document(kids)
                    if kids.len() == 1
                        && matches!(&kids[0], Node::Element(e) if e.tag == "circle" && e.ns == Namespace::Svg)
            ),
            "unexpected fragment: {node:?}"
        );
    }

    #[test]
    fn fragment_root_never_exposed() {
        // Comments, PIs and text targeting the synthetic root land at the
        // fragment top level instead.
        assert_eq!(
            frag("<!--c--><?p d?>text", &ctx("div")),
            "<!--c--><?p d?>text"
        );
    }

    #[test]
    fn fragment_foster_parenting() {
        // Misnested table text foster-parents before the table, at the
        // fragment top level.
        let (s, errors) = frag_errors("<table>x</table>", &ctx("div"));
        assert_eq!(s, "x<table></table>");
        assert_eq!(codes(&errors), vec![ParseErrorCode::FosterParented]);
    }

    #[test]
    fn fragment_option_context_never_pops_root() {
        // Adjusted current node is `option`, but the synthetic root is
        // never popped (the html node only pops when the parser finishes).
        let (s, errors) = frag_errors("<option>a", &ctx("option"));
        assert_eq!(s, "<option>a</option>");
        assert_eq!(codes(&errors), vec![ParseErrorCode::UnexpectedToken]);
    }

    #[test]
    fn fragment_context_tag_case_insensitive() {
        assert_eq!(frag("<tr></tr>", &ctx("TABLE")), "<tbody><tr></tr></tbody>");
    }

    // -- parse-error reporting ----------------------------------------------

    #[test]
    fn errors_clean_document() {
        let (_, errors) = parse_with_errors(
            b"<!DOCTYPE html><html><head><title>t</title></head><body><p>hi</p></body></html>",
        );
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    #[test]
    fn errors_null_character() {
        // U+0000 is a tokenizer-level error; the char itself is dropped.
        let (node, errors) = parse_with_errors("<div>\0</div>".as_bytes());
        let mut s = String::new();
        render(&node, &mut s);
        assert!(s.contains("<div></div>"));
        assert_eq!(
            codes(&errors),
            vec![ParseErrorCode::UnexpectedNullCharacter]
        );
        assert!(errors.iter().all(|e| e.loc.is_some()));
    }

    #[test]
    fn errors_truncated_doctype() {
        let (_, errors) = parse_with_errors(b"<!doctype");
        assert_eq!(codes(&errors), vec![ParseErrorCode::InvalidDoctype]);
        assert!(errors.iter().all(|e| e.loc.is_some()));
    }

    #[test]
    fn errors_unclosed_elements() {
        let (_, errors) = parse_with_errors(b"<div><span>");
        assert_eq!(codes(&errors), vec![ParseErrorCode::UnclosedElements]);
    }

    #[test]
    fn errors_no_unclosed_for_implied_end_tags() {
        let (_, errors) = parse_with_errors(b"<!DOCTYPE html><p>hi<li>x");
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    #[test]
    fn errors_stray_end_tag() {
        let (_, errors) = parse_with_errors(b"</div>");
        assert_eq!(codes(&errors), vec![ParseErrorCode::StrayEndTag]);
        assert!(errors.iter().all(|e| e.loc.is_some()));
    }

    #[test]
    fn errors_foster_parenting() {
        let (node, errors) = parse_with_errors(b"<table>x</table>");
        let mut s = String::new();
        render(&node, &mut s);
        assert!(s.contains("x<table></table>"));
        assert_eq!(codes(&errors), vec![ParseErrorCode::FosterParented]);
    }

    #[test]
    fn foster_parented_run_batches() {
        // A multi-character foster-parented run inserts as one text run
        // before the table, with one parse error for the batched run.
        let (node, errors) = parse_with_errors(b"<table>ab</table>");
        let mut s = String::new();
        render(&node, &mut s);
        assert!(s.contains("ab<table></table>"), "got {s}");
        assert_eq!(codes(&errors), vec![ParseErrorCode::FosterParented]);
    }

    #[test]
    fn batched_text_with_entities_and_markup() {
        // Batched runs decode entities and split at tags exactly like
        // per-character tokens did.
        let (node, _) = parse_with_errors(b"<div>a&amp;b&lt;c<i>d</i>e</div>");
        let mut s = String::new();
        render(&node, &mut s);
        assert!(s.contains("<div>a&b<c<i>d</i>e</div>"), "got {s}");
    }

    #[test]
    fn mixed_ws_text_run_splits() {
        // Whitespace-filtering modes split mixed runs: leading whitespace
        // is ignored, the non-whitespace remainder reprocesses.
        let (node, _) = parse_with_errors(b"  x<p>y</p>");
        let mut s = String::new();
        render(&node, &mut s);
        assert!(s.contains("x<p>y</p>"), "got {s}");
    }

    #[test]
    fn errors_all_carry_locations() {
        let (_, errors) = parse_with_errors("<div>\0</span><table>x".as_bytes());
        assert!(!errors.is_empty());
        assert!(errors.iter().all(|e| e.loc.is_some()));
    }

    #[test]
    fn parse_still_ignores_errors() {
        // The plain `parse` entry point is unchanged by error collection.
        let mut s = String::new();
        render(&parse(b"<div>\0</div>"), &mut s);
        assert!(s.contains("<div></div>"));
    }

    // -- foreign-content breakout (item 1): previously infinite loops ----

    fn svg_ctx(tag: &str) -> Element {
        let mut e = ctx(tag);
        e.ns = Namespace::Svg;
        e
    }

    /// Run the parse on a worker thread so a breakout regression fails the
    /// test instead of hanging the suite.
    fn frag_timeout(input: &str, c: &Element) -> String {
        let c = c.clone();
        let input = input.to_string();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(frag(&input, &c));
        });
        rx.recv_timeout(std::time::Duration::from_secs(10))
            .expect("foreign-content breakout hung")
    }

    #[test]
    fn fragment_foreign_breakout_nobr_terminates() {
        // `svg path` ctx + `<nobr>X`: the breakout re-dispatched through
        // `handle_token`, which re-selected foreign content through the
        // adjusted current node -> unbounded loop (foreign-fragment.dat).
        assert_eq!(frag_timeout("<nobr>X", &svg_ctx("path")), "<nobr>X</nobr>");
    }

    #[test]
    fn fragment_foreign_breakout_font_terminates() {
        // `svg path` ctx + `<font color></font>X`: the `font` start tag
        // breaks out of foreign content (13.2.6.5), so the dispatcher
        // routes `</font>` to "in body", where the adoption agency finds
        // no furthest block and simply pops `font`. `X` is a sibling.
        assert_eq!(
            frag_timeout("<font color></font>X", &svg_ctx("path")),
            "<font color=\"\"></font>X"
        );
    }

    #[test]
    fn fragment_foreign_breakout_end_tag_p_terminates() {
        // `</p><foo>` with `svg` ctx: `</p>` is reprocessed in "in body",
        // then `<foo>` goes back to foreign content (foreign-fragment.dat).
        assert_eq!(
            frag_timeout("</p><foo>", &svg_ctx("svg")),
            "<p></p><foo></foo>"
        );
    }

    // -- fragment adoption agency (item 2): the `aside` subtree survives --

    #[test]
    fn fragment_adoption_agency_keeps_aside_subtree() {
        // adoption01.dat fragment shape (ctx div).
        assert_eq!(
            frag("<b><em><foo><foob><fooc><aside></b></em>", &ctx("div")),
            "<b><em><foo><foob><fooc></fooc></foob></foo></em></b><aside><b></b></aside>"
        );
    }

    #[test]
    fn fragment_adoption_agency_webkit_shapes() {
        // webkit02.dat fragment shapes #0/#1/#2 (ctx div).
        assert_eq!(
            frag(
                "<b><em><dcell><postfield><postfield><postfield><postfield><missing_glyph><missing_glyph><missing_glyph><missing_glyph><hkern><aside></b></em>",
                &ctx("div")
            ),
            "<b><em><dcell><postfield><postfield><postfield><postfield><missing_glyph><missing_glyph><missing_glyph><missing_glyph><hkern></hkern></missing_glyph></missing_glyph></missing_glyph></missing_glyph></postfield></postfield></postfield></postfield></dcell></em></b><aside><b></b></aside>"
        );
        assert_eq!(
            frag(
                "<b><em><foo><foo><foo><foo><foo><foo><foo><foo><foo><foo><aside></b></em>",
                &ctx("div")
            ),
            "<b><em><foo><foo><foo><foo><foo><foo><foo><foo><foo><foo></foo></foo></foo></foo></foo></foo></foo></foo></foo></foo></em></b><aside><b></b></aside>"
        );
    }

    #[test]
    fn adoption_agency_removed_node_resolved_via_tracking() {
        // `<span>` is neither special nor a formatting element, so the
        // adoption agency's inner loop removes it from the stack of open
        // elements and the next iteration resolves it through the
        // removed-above tracking (the `None` arm next to the old
        // "lost track of node" panic). Must not panic; the tree matches
        // the WHATWG algorithm step by step.
        assert_eq!(
            frag("<b><span><div></b>", &ctx("div")),
            "<b><span></span></b><div><b></b></div>"
        );
    }

    // -- foreign fragment contexts (item 3) ------------------------------

    fn first_child_element(node: &Node) -> &Element {
        match node {
            Node::Document(kids) => match &kids[0] {
                Node::Element(e) => e,
                other => panic!("expected element, got {other:?}"),
            },
            _ => panic!("expected document"),
        }
    }

    #[test]
    fn fragment_foreign_object_keeps_camel_case() {
        // `foreignObject` is an HTML integration point, so `<div>` inside
        // parses as an HTML element. Lowercasing the context tag (13.4
        // step 4 only lowercases HTML elements) breaks this.
        let node = parse_fragment(b"<div>x</div>", &svg_ctx("foreignObject"));
        let e = first_child_element(&node);
        assert_eq!(e.tag, "div");
        assert_eq!(e.ns, Namespace::Html);
    }

    #[test]
    fn fragment_svg_title_is_not_rcdata() {
        // `svg title` is an HTML integration point, so `<b>` parses as an
        // HTML element. The tokenizer must still not treat `svg title`
        // as the RCDATA element (which would yield text instead of `<b>`).
        let node = parse_fragment(b"<b>x</b>", &svg_ctx("title"));
        let e = first_child_element(&node);
        assert_eq!(e.tag, "b");
        assert_eq!(e.ns, Namespace::Html);
    }

    // -- fragment CDATA (item 4) ------------------------------------------

    #[test]
    fn fragment_foreign_cdata_allowed() {
        // CDATA permission follows the adjusted current node, not the
        // synthetic root.
        let (s, errors) = frag_errors("<![CDATA[x]]>", &svg_ctx("svg"));
        assert_eq!(s, "x");
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    #[test]
    fn fragment_html_cdata_rejected() {
        // CDATA is not allowed in HTML fragment contexts: bogus comment.
        let (s, errors) = frag_errors("<![CDATA[x]]>", &ctx("div"));
        assert_eq!(s, "<!--[CDATA[x]]-->");
        assert_eq!(codes(&errors), vec![ParseErrorCode::CdataInHtmlContent]);
    }

    // -- form pointer from the fragment context (item 5) ------------------

    #[test]
    fn fragment_form_context_ignores_nested_form() {
        // 13.4 step 21: the form element pointer starts at the context
        // element, so the inner `<form>` start tag is ignored. The
        // fragment serializer renders the void `input` with a close tag.
        assert_eq!(frag("<form><input>", &ctx("form")), "<input></input>");
    }

    #[test]
    fn fragment_form_ancestor_ignores_nested_form() {
        // 13.4 step 21, nearest-form-ancestor lookup: the context is a
        // `div` inside a `form`, so the form element pointer is in scope
        // and the `<form>` start tag in the fragment is ignored.
        let form = ctx("form");
        let div = ctx("div");
        let mut s = String::new();
        render(
            &parse_fragment_with_ancestors(b"<form><input>", &div, &[&form]),
            &mut s,
        );
        assert_eq!(s, "<input></input>");
    }

    #[test]
    fn fragment_no_form_ancestor_keeps_nested_form() {
        // No form at or above the context: the `<form>` start tag is
        // inserted normally and becomes the form element pointer.
        let div = ctx("div");
        let section = ctx("section");
        let mut s = String::new();
        render(
            &parse_fragment_with_ancestors(b"<form><input>", &div, &[&section]),
            &mut s,
        );
        assert_eq!(s, "<form><input></input></form>");
    }

    #[test]
    fn fragment_form_ancestor_must_be_html() {
        // A foreign-namespace `form` ancestor is not a form element per
        // the spec, so it does not put the pointer in scope.
        let svg_form = Element {
            tag: TagName::from("form"),
            ns: Namespace::Svg,
            attrs: Vec::new(),
            children: Vec::new(),
            template_contents: None,
            shadow_root: None,
            loc: None,
        };
        let div = ctx("div");
        let mut s = String::new();
        render(
            &parse_fragment_with_ancestors(b"<form><input>", &div, &[&svg_form]),
            &mut s,
        );
        assert_eq!(s, "<form><input></input></form>");
    }

    // -- template fragment context (item 6) -------------------------------

    #[test]
    fn fragment_template_context_nested_forms() {
        // The form pointer is unused inside template contents even when
        // the template-ness comes from the fragment context.
        assert_eq!(
            frag("<form><form>", &ctx("template")),
            "<form><form></form></form>"
        );
    }

    // -- parser-local scripting flag (item 7) ------------------------------

    #[test]
    fn scripting_enabled_noscript_is_raw_text() {
        // A document-level `<noscript>` first token lands in "in head",
        // so with scripting enabled its content is RAWTEXT inside `head`.
        let node = parse_with_scripting(b"<noscript><p>hi</p></noscript>", true);
        let mut s = String::new();
        render(&node, &mut s);
        assert_eq!(
            s,
            "<html><head><noscript><p>hi</p></noscript></head><body></body></html>"
        );
        // The default (scripting disabled) parses the inner markup.
        let node = parse_with_scripting(b"<noscript><p>hi</p></noscript>", false);
        let mut s = String::new();
        render(&node, &mut s);
        let mut d = String::new();
        render(&parse(b"<noscript><p>hi</p></noscript>"), &mut d);
        assert_eq!(s, d);
    }

    #[test]
    fn fragment_noscript_context_scripting_selects_rawtext() {
        // 13.4 step 12: RAWTEXT only when the parser-local scripting flag
        // is on. The context element itself is not part of the fragment,
        // so the raw text is the fragment's direct child.
        let node = parse_fragment_with_scripting(b"<b>x", &ctx("noscript"), true);
        match &node {
            Node::Document(kids) => {
                assert!(
                    matches!(&kids[..], [Node::Text { text, .. }] if text == "<b>x"),
                    "unexpected fragment: {kids:?}"
                );
            }
            _ => panic!("expected document"),
        }
    }

    // -- CDATA open is case-sensitive (item 8) -----------------------------

    #[test]
    fn cdata_open_is_case_sensitive() {
        let (node, errors) = parse_with_errors(b"<![cdata[x]]>");
        let mut s = String::new();
        render(&node, &mut s);
        // The bogus comment is emitted in "initial", before `<html>`.
        assert_eq!(
            s,
            "<!--[cdata[x]]--><html><head></head><body></body></html>"
        );
        assert!(codes(&errors).contains(&ParseErrorCode::IncorrectlyOpenedComment));
        assert!(!codes(&errors).contains(&ParseErrorCode::CdataInHtmlContent));
    }

    #[test]
    fn cdata_in_html_content_is_bogus_comment() {
        let (node, errors) = parse_with_errors(b"<![CDATA[x]]>");
        let mut s = String::new();
        render(&node, &mut s);
        // The bogus comment is emitted in "initial", before `<html>`.
        assert_eq!(
            s,
            "<!--[CDATA[x]]--><html><head></head><body></body></html>"
        );
        assert!(codes(&errors).contains(&ParseErrorCode::CdataInHtmlContent));
    }

    // -- tokenizer error codes and locations (item 9) ----------------------

    #[test]
    fn errors_tokenizer_missing_end_tag_name() {
        let (_, errors) = parse_with_errors(b"a</>");
        assert_eq!(codes(&errors), vec![ParseErrorCode::MissingEndTagName]);
        assert!(errors.iter().all(|e| e.loc.is_some()));
    }

    #[test]
    fn errors_tokenizer_invalid_tag_open() {
        let (_, errors) = parse_with_errors(b"<1>");
        assert_eq!(
            codes(&errors),
            vec![ParseErrorCode::InvalidFirstCharacterOfTagName]
        );
        assert!(errors.iter().all(|e| e.loc.is_some()));
    }

    #[test]
    fn errors_tokenizer_eof_before_tag_name() {
        let (_, errors) = parse_with_errors(b"<");
        assert_eq!(codes(&errors), vec![ParseErrorCode::EofBeforeTagName]);
        assert!(errors.iter().all(|e| e.loc.is_some()));
    }

    #[test]
    fn errors_tokenizer_eof_in_tag() {
        let (_, errors) = parse_with_errors(b"<div");
        assert_eq!(codes(&errors), vec![ParseErrorCode::EofInTag]);
        assert!(errors.iter().all(|e| e.loc.is_some()));
    }

    #[test]
    fn errors_tokenizer_eof_in_comment() {
        let (_, errors) = parse_with_errors(b"<!--x");
        assert_eq!(codes(&errors), vec![ParseErrorCode::EofInComment]);
        assert!(errors.iter().all(|e| e.loc.is_some()));
    }

    #[test]
    fn errors_tokenizer_attribute_codes() {
        // `=` before the attribute name.
        let (_, errors) = parse_with_errors(b"<div =x>");
        assert!(codes(&errors).contains(&ParseErrorCode::UnexpectedEqualsSignBeforeAttributeName));
        // Missing whitespace between attributes (quoted adjacency).
        let (_, errors) = parse_with_errors(b"<div a=\"1\"b=\"2\">");
        assert!(codes(&errors).contains(&ParseErrorCode::MissingWhitespaceBetweenAttributes));
        // Missing attribute value.
        let (_, errors) = parse_with_errors(b"<div a=>");
        assert!(codes(&errors).contains(&ParseErrorCode::MissingAttributeValue));
        // Unexpected solidus in tag.
        let (_, errors) = parse_with_errors(b"<div/ >");
        assert!(codes(&errors).contains(&ParseErrorCode::UnexpectedSolidusInTag));
        assert!(errors.iter().all(|e| e.loc.is_some()));
    }

    #[test]
    fn errors_tokenizer_abrupt_comment_close() {
        // `<!-->` is an abrupt closing of an empty comment: the empty
        // comment is still emitted (before `<html>`, from "initial").
        let (node, errors) = parse_with_errors(b"<!-->text");
        let mut s = String::new();
        render(&node, &mut s);
        assert_eq!(s, "<!----><html><head></head><body>text</body></html>");
        assert!(codes(&errors).contains(&ParseErrorCode::AbruptClosingOfEmptyComment));
    }

    #[test]
    fn errors_foreign_end_tag_mismatch() {
        // 13.2.6.5 "any other end tag" step 2 (foreign-fragment.dat shape).
        let (s, errors) = frag_errors("<g></path>X", &svg_ctx("path"));
        assert_eq!(s, "<g>X</g>");
        assert!(codes(&errors).contains(&ParseErrorCode::MismatchedEndTag));
        assert!(errors.iter().all(|e| e.loc.is_some()));
    }

    #[test]
    fn decode_utf16le_bom() {
        // A UTF-16LE BOM overrides everything (including any meta) and is
        // stripped before tokenizing.
        let mut bytes = vec![0xFF, 0xFE];
        for c in "<p>hi</p>".encode_utf16() {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
        let node = parse(&bytes);
        let mut s = String::new();
        render(&node, &mut s);
        assert_eq!(s, "<html><head></head><body><p>hi</p></body></html>");
    }

    #[test]
    fn decode_meta_charset_windows1252() {
        // No BOM: the meta prescan selects windows-1252, so 0xE9 decodes
        // to U+00E9 rather than U+FFFD.
        let html =
            b"<html><head><meta charset=\"windows-1252\"></head><body><p>\xe9</p></body></html>";
        let node = parse(html);
        let mut s = String::new();
        render(&node, &mut s);
        assert_eq!(
            s,
            "<html><head><meta charset=\"windows-1252\"></meta></head><body><p>\u{e9}</p></body></html>"
        );
    }

    // -- declarative shadow roots (WHATWG HTML 4.12.3 / 13.2.6.4.7) --------

    /// The `<div>` under `<body>` in a single-host parse.
    fn host_div(node: &Node) -> &Element {
        let html = first_child_element(node);
        let body = html
            .children
            .iter()
            .find_map(|k| match k {
                Node::Element(e) if e.tag == "body" => Some(e),
                _ => None,
            })
            .expect("body");
        match &body.children[0] {
            Node::Element(e) => e,
            other => panic!("expected div, got {other:?}"),
        }
    }

    #[test]
    fn dsr_open_attaches_to_parent() {
        // `<template shadowrootmode=open>` becomes an open shadow root on
        // its parent; the template element itself leaves the tree and its
        // contents parse into the shadow root.
        let node = parse(b"<div><template shadowrootmode=\"open\"><p>hi</p></template></div>");
        let div = host_div(&node);
        assert_eq!(div.tag, "div");
        assert!(div.children.is_empty(), "template element removed");
        let shadow = div.shadow_root.as_ref().expect("shadow root");
        assert_eq!(shadow.mode, ShadowRootMode::Open);
        assert!(!shadow.delegates_focus);
        assert!(!shadow.clonable);
        let mut s = String::new();
        for k in &shadow.children {
            render(k, &mut s);
        }
        assert_eq!(s, "<p>hi</p>");
    }

    #[test]
    fn dsr_closed_with_flags() {
        let node = parse(
            b"<div><template shadowrootmode=\"closed\" shadowrootdelegatesfocus shadowrootclonable><span>x</span></template></div>",
        );
        let shadow = host_div(&node).shadow_root.as_ref().expect("shadow root");
        assert_eq!(shadow.mode, ShadowRootMode::Closed);
        assert!(shadow.delegates_focus);
        assert!(shadow.clonable);
    }

    #[test]
    fn dsr_mode_is_case_insensitive() {
        // Enumerated attribute: ASCII case-insensitive match.
        let node = parse(b"<div><template shadowrootmode=\"OPEN\"><p>x</p></template></div>");
        assert_eq!(
            host_div(&node)
                .shadow_root
                .as_ref()
                .expect("shadow root")
                .mode,
            ShadowRootMode::Open
        );
    }

    #[test]
    fn dsr_invalid_or_missing_mode_is_plain_template() {
        for html in [
            "<div><template shadowrootmode=\"bogus\"><p>x</p></template></div>",
            "<div><template><p>x</p></template></div>",
        ] {
            let node = parse(html.as_bytes());
            let div = host_div(&node);
            assert!(div.shadow_root.is_none());
            // Ordinary template: element present, contents in the fragment.
            let Node::Element(tmpl) = &div.children[0] else {
                panic!("expected template element");
            };
            assert_eq!(tmpl.tag, "template");
            let mut s = String::new();
            for k in tmpl.template_contents.as_ref().expect("contents") {
                render(k, &mut s);
            }
            assert_eq!(s, "<p>x</p>");
        }
    }

    #[test]
    fn dsr_only_first_template_per_parent() {
        // A second `<template shadowrootmode>` on the same parent stays a
        // plain template (DOM `attachShadow` would throw; the parser keeps
        // parsing instead).
        let node = parse(
            b"<div><template shadowrootmode=\"open\"><p>a</p></template><template shadowrootmode=\"open\"><p>b</p></template></div>",
        );
        let div = host_div(&node);
        let shadow = div.shadow_root.as_ref().expect("shadow root");
        let mut s = String::new();
        for k in &shadow.children {
            render(k, &mut s);
        }
        assert_eq!(s, "<p>a</p>");
        assert_eq!(div.children.len(), 1);
        let Node::Element(tmpl) = &div.children[0] else {
            panic!("expected template element");
        };
        assert!(tmpl.shadow_root.is_none());
        let mut s = String::new();
        for k in tmpl.template_contents.as_ref().expect("contents") {
            render(k, &mut s);
        }
        assert_eq!(s, "<p>b</p>");
    }

    #[test]
    fn dsr_nested_template_stays_plain() {
        // A `<template shadowrootmode>` nested inside a declarative shadow
        // root under construction: `template` is not a valid shadow host
        // name (WHATWG DOM 4.9), so the inner template stays an ordinary
        // template element in the shadow tree instead of emitting a nested
        // shadow-root bracket for the same host (which used to close the
        // outer root early and leak `C` into the light DOM).
        let node = parse(
            b"<div><template shadowrootmode=\"open\">A<template shadowrootmode=\"open\">B</template>C</template></div>",
        );
        let div = host_div(&node);
        assert!(div.children.is_empty(), "nothing leaks into the light DOM");
        let shadow = div.shadow_root.as_ref().expect("shadow root");
        assert_eq!(shadow.children.len(), 3);
        assert!(matches!(&shadow.children[0], Node::Text { text, .. } if text == "A"));
        let Node::Element(tmpl) = &shadow.children[1] else {
            panic!("expected inner template element");
        };
        assert_eq!(tmpl.tag, "template");
        assert!(tmpl.shadow_root.is_none(), "inner template is plain");
        let mut s = String::new();
        for k in tmpl.template_contents.as_ref().expect("contents") {
            render(k, &mut s);
        }
        assert_eq!(s, "B");
        assert!(matches!(&shadow.children[2], Node::Text { text, .. } if text == "C"));
    }

    #[test]
    fn dsr_nested_hosts_still_nest() {
        // A shadow root on a *descendant* host inside a shadow tree is
        // legitimate: the inner bracket pairs with the inner host's own
        // element brackets, not with the outer bracket.
        let node = parse(
            b"<div><template shadowrootmode=\"open\"><div><template shadowrootmode=\"open\">x</template></div></template></div>",
        );
        let div = host_div(&node);
        let outer = div.shadow_root.as_ref().expect("outer shadow root");
        assert_eq!(outer.children.len(), 1);
        let Node::Element(inner) = &outer.children[0] else {
            panic!("expected inner div");
        };
        assert_eq!(inner.tag, "div");
        let inner_shadow = inner.shadow_root.as_ref().expect("inner shadow root");
        let mut s = String::new();
        for k in &inner_shadow.children {
            render(k, &mut s);
        }
        assert_eq!(s, "x");
    }

    #[test]
    fn dsr_non_host_parent_stays_plain() {
        // `table` is not a valid shadow host name (WHATWG DOM 4.9): the
        // template stays a plain template element (spec 13.2.6.4.7 step
        // 7.11.2: "attach a shadow root" throws, and the parser catches
        // it instead of propagating).
        let node = parse(b"<table><template shadowrootmode=\"open\"><p>x</p></template></table>");
        fn has_shadow(n: &Node) -> bool {
            match n {
                Node::Element(e) => e.shadow_root.is_some() || e.children.iter().any(has_shadow),
                Node::Document(kids) => kids.iter().any(has_shadow),
                _ => false,
            }
        }
        assert!(!has_shadow(&node), "no shadow root attached anywhere");
    }

    #[test]
    fn dsr_fragment_parsing_opts_out() {
        // WHATWG 13.4: fragment parsing (`innerHTML`-equivalent) never
        // creates declarative shadow roots.
        let node = parse_fragment(
            b"<template shadowrootmode=\"open\"><p>x</p></template>",
            &ctx("div"),
        );
        let tmpl = first_child_element(&node);
        assert_eq!(tmpl.tag, "template");
        assert!(tmpl.shadow_root.is_none());
        assert!(tmpl.template_contents.is_some());
    }

    #[test]
    fn dsr_light_children_stay_light() {
        // Shadow content brackets at the template's position; light-DOM
        // siblings are unaffected.
        let node = parse(
            b"<div><b>light</b><template shadowrootmode=\"open\"><i>shadow</i></template></div>",
        );
        let mut s = String::new();
        render(&node, &mut s);
        assert_eq!(
            s,
            "<html><head></head><body><div><b>light</b><#shadow-root(open)><i>shadow</i></#shadow-root></div></body></html>"
        );
    }

    // -- is_ancestor_or_self ------------------------------------------------

    /// Build a bare arena node (no parent linkage yet).
    fn bare_node() -> ArenaNode {
        ArenaNode {
            tag: TagName::from("div"),
            ns: Namespace::Html,
            attrs: Vec::new(),
            children: Vec::new(),
            template_contents: None,
            dsr: None,
            parent: None,
            loc: None,
            annotation_xml_html: false,
        }
    }

    #[test]
    fn is_ancestor_or_self_terminates_on_deep_chain() {
        // A deep ancestor chain must terminate with the correct answers.
        let mut tb = TreeBuilder::new("", false, None);
        let n = 10_000usize;
        for i in 0..n {
            let mut node = bare_node();
            node.parent = if i == 0 {
                Some(Parent::Doc)
            } else {
                Some(Parent::Elem(i - 1))
            };
            tb.arena.push(node);
        }
        let host = ScriptHost { tb: &mut tb };
        assert!(host.is_ancestor_or_self(0, n - 1));
        assert!(host.is_ancestor_or_self(n - 1, n - 1));
        assert!(!host.is_ancestor_or_self(n - 1, 0));
    }

    #[test]
    fn is_ancestor_or_self_terminates_on_parent_cycle() {
        // Defense in depth: parent links form a tree through the guarded
        // mutation APIs, so a cycle is unreachable in a correct run. If
        // one ever appears, the walk must fail closed instead of hanging.
        let mut tb = TreeBuilder::new("", false, None);
        for _ in 0..3 {
            tb.arena.push(bare_node());
        }
        // 0 -> 1 -> 2 -> 1 (cycle; never reaches Doc).
        tb.arena[0].parent = Some(Parent::Elem(1));
        tb.arena[1].parent = Some(Parent::Elem(2));
        tb.arena[2].parent = Some(Parent::Elem(1));
        let host = ScriptHost { tb: &mut tb };
        assert!(!host.is_ancestor_or_self(0, 2));
    }

    // ------------------------------------------------------------------
    // Mid-parse geometry primitives: mutation_count / snapshot_into
    // ------------------------------------------------------------------

    /// Sink event, for comparing `snapshot_into` against `drive_sink`.
    #[derive(Debug, PartialEq, Eq)]
    #[allow(dead_code)]
    enum SnapEv {
        Doctype(String),
        Text(String),
        Comment(String),
        Pi(String, String),
        Start(String, Vec<(String, String)>),
        End,
        StartTemplate,
        EndTemplate,
        StartShadow(String),
        EndShadow,
    }

    /// [`TreeSink`] collecting events into a [`Vec`].
    #[derive(Debug, Default)]
    struct VecSink {
        events: Vec<SnapEv>,
    }

    impl TreeSink for VecSink {
        fn doctype(&mut self, d: Doctype) {
            self.events.push(SnapEv::Doctype(d.name));
        }
        fn text(&mut self, t: String, _loc: Option<SourceLoc>) {
            self.events.push(SnapEv::Text(t));
        }
        fn comment(&mut self, d: String, _loc: Option<SourceLoc>) {
            self.events.push(SnapEv::Comment(d));
        }
        fn processing_instruction(
            &mut self,
            target: String,
            data: String,
            _loc: Option<SourceLoc>,
        ) {
            self.events.push(SnapEv::Pi(target, data));
        }
        fn start_element(
            &mut self,
            tag: String,
            _ns: Namespace,
            attrs: Vec<(String, String)>,
            _loc: Option<SourceLoc>,
        ) {
            self.events.push(SnapEv::Start(tag, attrs));
        }
        fn end_element(&mut self) {
            self.events.push(SnapEv::End);
        }
        fn start_template_contents(&mut self) {
            self.events.push(SnapEv::StartTemplate);
        }
        fn end_template_contents(&mut self) {
            self.events.push(SnapEv::EndTemplate);
        }
        fn start_shadow_root(&mut self, mode: ShadowRootMode, _d: bool, _c: bool) {
            self.events.push(SnapEv::StartShadow(format!("{mode:?}")));
        }
        fn end_shadow_root(&mut self) {
            self.events.push(SnapEv::EndShadow);
        }
    }

    /// Step `tb` to completion (the tail of [`TreeBuilder::run`] without
    /// the loop).
    fn finish(tb: &mut TreeBuilder<'_, '_>) {
        while !tb.step_token() {}
        tb.flush_text();
        tb.flush_foster_buf();
    }

    #[test]
    fn snapshot_into_matches_drive_sink_on_completed_tree() {
        let html = "<!DOCTYPE html><html><head><title>t</title></head><body>\
            <div id=a>hello <b>w</b><!--c--><table><tr><td>x</td></tr></table>\
            <template><p>tc</p></template>\
            <div id=d><template shadowrootmode=open><span>s</span></template></div>\
            </div></body></html>";
        // Completed via stepping, snapshotted by borrow.
        let mut tb = TreeBuilder::new(html, false, None);
        finish(&mut tb);
        let mut s1 = VecSink::default();
        let order = tb.snapshot_into(&mut s1);
        // Completed separately, driven by value.
        let mut tb2 = TreeBuilder::new(html, false, None);
        tb2.run();
        let mut s2 = VecSink::default();
        tb2.drive_sink(&mut s2);
        assert_eq!(s1.events, s2.events);

        // The declarative-shadow-root branch was exercised: the
        // `<template shadowrootmode>` is replaced by a shadow-root
        // bracket, not an element.
        assert!(
            s1.events
                .iter()
                .any(|e| matches!(e, SnapEv::StartShadow(_))),
            "expected a shadow-root bracket in the snapshot: {s1:?}"
        );

        // The order vector zips emission order back to arena nodes: one
        // entry per emitted element, each a valid arena index whose tag
        // matches the corresponding `start_element` event. (Template
        // elements replaced by a shadow-root bracket emit no element and
        // contribute no entry.)
        let starts: Vec<&String> = s1
            .events
            .iter()
            .filter_map(|e| match e {
                SnapEv::Start(tag, _) => Some(tag),
                _ => None,
            })
            .collect();
        assert_eq!(order.len(), starts.len());
        for (el, tag) in order.iter().zip(starts.iter()) {
            assert!(*el < tb.arena.len(), "order entry out of arena range");
            assert_eq!(tb.arena[*el].tag.to_string(), **tag);
        }
    }

    #[test]
    fn snapshot_into_mid_parse_is_repeatable_and_undisturbing() {
        let html = "<html><head><title>t</title></head><body><div>hello <b>world</b>\
            <table>xy<tr><td>z</td></tr></table><p>tail</p></div>";
        let mut tb = TreeBuilder::new(html, false, None);
        // Step until the table-text run "xy" is buffered but not yet
        // flushed: the snapshot must flush pending_table_chars (through
        // the foster path), pending_text, and foster_buf to be complete.
        for _ in 0..1000 {
            if !tb.pending_table_chars.is_empty() {
                break;
            }
            assert!(!tb.step_token(), "EOF before table text buffered");
        }
        assert!(
            !tb.pending_table_chars.is_empty(),
            "table text was never buffered"
        );

        // Two back-to-back snapshots: identical output and order.
        let mut s1 = VecSink::default();
        let order1 = tb.snapshot_into(&mut s1);
        let mut s2 = VecSink::default();
        let order2 = tb.snapshot_into(&mut s2);
        assert_eq!(s1.events, s2.events);
        assert_eq!(order1, order2);
        // The buffered table text made it into the snapshot (fostered
        // before the table), so measured widths would be complete.
        assert!(
            s1.events
                .iter()
                .any(|e| matches!(e, SnapEv::Text(t) if t.contains("xy"))),
            "buffered table text missing from snapshot: {s1:?}"
        );

        // Parsing continues correctly after the snapshot: the finished
        // tree equals a clean parse with no snapshot taken.
        finish(&mut tb);
        let mut ns = crate::sink::NodeSink::new();
        tb.drive_sink(&mut ns);
        assert_eq!(ns.into_node(), parse(html.as_bytes()));
    }

    #[test]
    fn mutation_count_bumps_on_foster_parented_inserts() {
        let mut tb = TreeBuilder::new("<table>xyz<tr><td>q</td></tr></table>", false, None);
        assert_eq!(tb.mutation_count(), 0);
        // Step until the table-text run is buffered (not yet in the tree).
        for _ in 0..1000 {
            if !tb.pending_table_chars.is_empty() {
                break;
            }
            assert!(!tb.step_token(), "EOF before table text buffered");
        }
        assert!(!tb.pending_table_chars.is_empty());
        let before = tb.mutation_count();
        // Flush the run through the foster path manually: the text lands in
        // the foster buffer (invisible), and only the buffer splice
        // mutates the tree — so any bump here is the foster path's.
        tb.flush_pending_table_chars();
        tb.flush_text();
        assert!(tb.foster_buf.is_some(), "foster run was not buffered");
        let buffered = tb.mutation_count();
        tb.flush_foster_buf();
        assert!(
            tb.mutation_count() > buffered,
            "foster-buffer splice did not bump mutation_count"
        );
        assert!(tb.mutation_count() > before);
    }

    /// Assert `op` bumps the host's mutation count, recording `name` on
    /// failure.
    fn assert_bump(
        host: &mut ScriptHost,
        name: &'static str,
        failures: &mut Vec<&'static str>,
        op: impl FnOnce(&mut ScriptHost),
    ) {
        let before = host.mutation_count();
        op(host);
        if host.mutation_count() <= before {
            failures.push(name);
        }
    }

    /// [`ScriptExecutor`] asserting every [`ScriptHost`] mutator bumps
    /// [`ScriptHost::mutation_count`].
    struct MutationProbe {
        failures: Vec<&'static str>,
    }

    impl ScriptExecutor for MutationProbe {
        fn execute_script(
            &mut self,
            host: &mut ScriptHost,
            _el: ScriptElement,
            _attrs: &[(String, String)],
            _script: &str,
        ) {
            let t = host.get_element_by_id("t").expect("target element");
            let c = host.get_element_by_id("c").expect("child element");
            assert_bump(host, "set_attribute", &mut self.failures, |h| {
                h.set_attribute(t, "data-x", "1")
            });
            assert_bump(host, "append_child", &mut self.failures, |h| {
                assert!(h.append_child(t, c))
            });
            assert_bump(host, "remove_child", &mut self.failures, |h| {
                assert!(h.remove_child(t, c))
            });
            assert_bump(host, "insert_before", &mut self.failures, |h| {
                assert!(h.insert_before(t, c, None))
            });
            assert_bump(host, "set_text_content", &mut self.failures, |h| {
                h.set_text_content(t, "hi")
            });
            assert_bump(host, "set_inner_html", &mut self.failures, |h| {
                h.set_inner_html(t, "<b>x</b>")
            });
            assert_bump(host, "clone_element", &mut self.failures, |h| {
                assert!(h.clone_element(t, true).is_some())
            });
        }
    }

    #[test]
    fn mutation_count_bumps_on_every_script_host_mutator() {
        let html = "<div id=\"t\"><span id=\"c\"></span></div><script>1</script>";
        let mut probe = MutationProbe {
            failures: Vec::new(),
        };
        parse_with_scripting_and_executor(html.as_bytes(), true, &mut probe);
        assert!(
            probe.failures.is_empty(),
            "mutators that did not bump mutation_count: {:?}",
            probe.failures
        );
    }

    // -- shim querySelector / querySelectorAll document order --------

    /// `id` attribute of a shim element, for QSA probe assertions.
    fn shim_id(host: &ScriptHost, el: ScriptElement) -> String {
        host.get_attributes(el)
            .into_iter()
            .find(|(k, _)| k == "id")
            .map(|(_, v)| v)
            .unwrap_or_default()
    }

    /// [`ScriptExecutor`] asserting multi-group shim QSA is deduped and
    /// in document order, and shim QS returns the earliest element in
    /// document order across groups (not the first group's first match).
    struct QsaOrderProbe {
        ran: bool,
        all: Vec<String>,
        one: String,
        single: Vec<String>,
    }

    impl ScriptExecutor for QsaOrderProbe {
        fn execute_script(
            &mut self,
            host: &mut ScriptHost<'_, '_, '_>,
            _el: ScriptElement,
            _attrs: &[(String, String)],
            _script: &str,
        ) {
            // `.y` matches b, c, d; `.x` matches a, c. Group order with
            // no dedup would be [b, c, d, a, c]; document order deduped
            // is [a, b, c, d].
            self.all = host
                .query_selector_all(".y, .x")
                .into_iter()
                .map(|el| shim_id(host, el))
                .collect();
            self.one = host
                .query_selector(".y, .x")
                .map(|el| shim_id(host, el))
                .unwrap_or_default();
            self.single = host
                .query_selector_all(".x")
                .into_iter()
                .map(|el| shim_id(host, el))
                .collect();
            self.ran = true;
        }
    }

    #[test]
    fn shim_qsa_multi_group_deduped_in_document_order() {
        let html = "<div id=\"a\" class=\"x\"></div><span id=\"b\" class=\"y\"></span>\
                    <div id=\"c\" class=\"x y\"></div><p id=\"d\" class=\"y\"></p><script>1</script>";
        let mut probe = QsaOrderProbe {
            ran: false,
            all: Vec::new(),
            one: String::new(),
            single: Vec::new(),
        };
        parse_with_scripting_and_executor(html.as_bytes(), true, &mut probe);
        assert!(probe.ran, "script executor never ran");
        assert_eq!(probe.all, vec!["a", "b", "c", "d"]);
        assert_eq!(probe.one, "a");
        assert_eq!(probe.single, vec!["a", "c"]);
    }

    /// [`ScriptExecutor`] asserting a moved element is reported in its
    /// new document position, not its (unchanged) arena slot: arena
    /// order is creation order, so it diverges from document order
    /// after `appendChild` moves a node.
    struct QsaMoveProbe {
        ran: bool,
        all: Vec<String>,
        one: String,
    }

    impl ScriptExecutor for QsaMoveProbe {
        fn execute_script(
            &mut self,
            host: &mut ScriptHost<'_, '_, '_>,
            _el: ScriptElement,
            _attrs: &[(String, String)],
            _script: &str,
        ) {
            let a = host.get_element_by_id("a").expect("a");
            let parent = host.parent_element(a).expect("a has a parent");
            assert!(host.append_child(parent, a), "move a to the end");
            // A detached element must not match, even though it sits in
            // the arena with a matching class.
            let detached = host.create_element("div");
            host.set_attribute(detached, "class", "x");
            self.all = host
                .query_selector_all("#c, .x")
                .into_iter()
                .map(|el| shim_id(host, el))
                .collect();
            self.one = host
                .query_selector(".x")
                .map(|el| shim_id(host, el))
                .unwrap_or_default();
            self.ran = true;
        }
    }

    #[test]
    fn shim_qsa_follows_moves_not_arena_order() {
        let html = "<div id=\"a\" class=\"x\"></div><div id=\"b\" class=\"x\"></div>\
                    <div id=\"c\" class=\"x\"></div><script>1</script>";
        let mut probe = QsaMoveProbe {
            ran: false,
            all: Vec::new(),
            one: String::new(),
        };
        parse_with_scripting_and_executor(html.as_bytes(), true, &mut probe);
        assert!(probe.ran, "script executor never ran");
        assert_eq!(probe.all, vec!["b", "c", "a"]);
        assert_eq!(probe.one, "b");
    }

    /// [`ScriptExecutor`] asserting a foster-parented element is
    /// reported before the table it was foster-parented in front of,
    /// even though the table was created (and sits in the arena)
    /// first.
    struct QsaFosterProbe {
        ran: bool,
        tags: Vec<String>,
    }

    impl ScriptExecutor for QsaFosterProbe {
        fn execute_script(
            &mut self,
            host: &mut ScriptHost<'_, '_, '_>,
            _el: ScriptElement,
            _attrs: &[(String, String)],
            _script: &str,
        ) {
            self.tags = host
                .query_selector_all("table, div")
                .into_iter()
                .map(|el| host.tag_name(el).unwrap_or_default())
                .collect();
            self.ran = true;
        }
    }

    #[test]
    fn shim_qsa_foster_parented_element_precedes_table() {
        let html = "<table><div id=\"f\"></div><tr><td>cell</td></tr></table><script>1</script>";
        let mut probe = QsaFosterProbe {
            ran: false,
            tags: Vec::new(),
        };
        parse_with_scripting_and_executor(html.as_bytes(), true, &mut probe);
        assert!(probe.ran, "script executor never ran");
        assert_eq!(probe.tags, vec!["div", "table"]);
    }

    /// [`ScriptExecutor`] asserting the element-scoped shim QSA/QS have
    /// the same document-order, deduped behaviour as the document ones.
    struct QsaScopedProbe {
        ran: bool,
        all: Vec<String>,
        one: String,
    }

    impl ScriptExecutor for QsaScopedProbe {
        fn execute_script(
            &mut self,
            host: &mut ScriptHost<'_, '_, '_>,
            _el: ScriptElement,
            _attrs: &[(String, String)],
            _script: &str,
        ) {
            let scope = host.get_element_by_id("scope").expect("scope");
            self.all = host
                .query_selector_all_in(scope, ".y, .x")
                .into_iter()
                .map(|el| shim_id(host, el))
                .collect();
            self.one = host
                .query_selector_in(scope, ".y, .x")
                .map(|el| shim_id(host, el))
                .unwrap_or_default();
            self.ran = true;
        }
    }

    #[test]
    fn shim_qsa_scoped_multi_group_deduped_in_document_order() {
        let html = "<div id=\"scope\"><span id=\"s1\" class=\"x\"></span>\
                    <b id=\"s2\" class=\"y\"></b><i id=\"s3\" class=\"x y\"></i></div><script>1</script>";
        let mut probe = QsaScopedProbe {
            ran: false,
            all: Vec::new(),
            one: String::new(),
        };
        parse_with_scripting_and_executor(html.as_bytes(), true, &mut probe);
        assert!(probe.ran, "script executor never ran");
        assert_eq!(probe.all, vec!["s1", "s2", "s3"]);
        assert_eq!(probe.one, "s1");
    }

    #[test]
    fn script_element_arena_index_exposes_arena_slot() {
        let se = ScriptElement::from_el(42, 7);
        assert_eq!(se.arena_index(), 42);
    }

    /// [`ScriptExecutor`] asserting a deep `cloneNode` of a `<template>`
    /// carries the template contents — including a nested `<template>`'s
    /// own contents, which live in `template_contents` rather than
    /// `children`.
    struct CloneTemplateProbe {
        ran: bool,
    }

    impl ScriptExecutor for CloneTemplateProbe {
        fn execute_script(
            &mut self,
            host: &mut ScriptHost<'_, '_, '_>,
            _el: ScriptElement,
            _attrs: &[(String, String)],
            _script: &str,
        ) {
            let t = host.get_element_by_id("t").expect("outer template");
            let clone = host.clone_element(t, true).expect("deep clone");
            assert_eq!(host.tag_name(clone).as_deref(), Some("template"));
            // The clone must keep the outer template's contents.
            let content = host
                .template_content(clone)
                .expect("clone has template contents");
            let kids = host.child_elements(content);
            assert_eq!(kids.len(), 1, "outer template content children");
            // The nested template survives with its own contents intact.
            assert_eq!(host.tag_name(kids[0]).as_deref(), Some("template"));
            let inner = host
                .template_content(kids[0])
                .expect("nested template content survives deep clone");
            let inner_kids = host.child_elements(inner);
            assert_eq!(inner_kids.len(), 1, "nested template content children");
            assert_eq!(host.tag_name(inner_kids[0]).as_deref(), Some("b"));
            assert_eq!(host.text_content(inner_kids[0]), "hi");
            self.ran = true;
        }
    }

    #[test]
    fn clone_element_deep_copies_nested_template_contents() {
        let html = "<template id=\"t\"><template id=\"inner\"><b>hi</b></template></template><script>1</script>";
        let mut probe = CloneTemplateProbe { ran: false };
        parse_with_scripting_and_executor(html.as_bytes(), true, &mut probe);
        assert!(probe.ran, "script executor never ran");
    }
}
